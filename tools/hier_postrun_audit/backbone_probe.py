"""Explicit, bounded, read-only shared-encoder gradient probes.

These probes reuse real HyCoRe crop/CE/intra and HIER kernels on a fixed clean
batch with eval BN. They do not replay augmented train BN, update parameters,
simulate optimizer steps, or forward validation/test objects.
"""
from __future__ import annotations

from contextlib import contextmanager
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

import numpy as np

from .extract import REPO, checkpoint_epoch, idle_gpu, load_training_shards, sha256, state_digest
from .mechanisms import _balanced_positions, gradient_summary


def _save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def _array_sha(values):
    values = np.asarray(values)
    return hashlib.sha256(str(values.dtype).encode() + str(values.shape).encode() + values.tobytes()).hexdigest()


@contextmanager
def preserved_model(model):
    import torch
    modes = [(module, module.training) for module in model.modules()]
    state_before = state_digest(model.state_dict())
    grad_before = [(parameter, None if parameter.grad is None else parameter.grad.detach().clone())
                   for parameter in model.parameters()]
    python_rng, numpy_rng, cpu_rng = random.getstate(), np.random.get_state(), torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    try:
        model.eval()
        yield
        if state_digest(model.state_dict()) != state_before:
            raise RuntimeError("Frozen probe changed model parameters or BN buffers")
        for parameter, original in grad_before:
            if original is None:
                if parameter.grad is not None:
                    raise RuntimeError("Probe accumulated model .grad buffers")
            elif parameter.grad is None or not torch.equal(parameter.grad, original):
                raise RuntimeError("Probe changed pre-existing model .grad buffers")
    finally:
        for module, mode in modes:
            module.training = mode
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)


def gradient_comparison(first, second):
    import torch
    first, second = first.reshape(-1), second.reshape(-1)
    left, right = float(first.norm()), float(second.norm())
    return {"first_norm": left, "second_norm": right,
            "norm_ratio": left / right if right else None,
            "cosine": float(torch.dot(first, second) / (left * right)) if left and right else None,
            "semantics": "Euclidean partial gradients on identical saved shared parameters, before clipping/optimizer"}


def clean_alias_forward(model, clouds, labels, sample_ids, *, whole_count=1024,
                        child_count=256, microbatch_size=32, seed=22, activation_checkpoint=True):
    """Use actual crop aliasing but explicit same-ID crop centers/eval BN."""
    import torch
    from inter_hierarchy_MN40.hier_proxy_scratch_v5.base_protocol import get_children_alias
    clouds = np.asarray(clouds, dtype=np.float32)
    labels, sample_ids = np.asarray(labels).reshape(-1), np.asarray(sample_ids).reshape(-1)
    if clouds.ndim != 3 or clouds.shape[2] != 3 or len(labels) != len(clouds) or len(sample_ids) != len(clouds):
        raise ValueError("Probe needs aligned clouds[N,P,3], labels and fixed sample IDs")
    if not 1 <= child_count <= whole_count <= clouds.shape[1] or microbatch_size < 1:
        raise ValueError("Invalid fixed crop/microbatch counts")
    if np.any(labels == labels[::-1]):
        raise ValueError("Fixed global flip negative shares an anchor class; provide class-grouped balanced64 order")
    before = _array_sha(clouds)
    rng = np.random.default_rng(seed)
    whole_centers = rng.integers(0, clouds.shape[1], size=len(clouds))
    child_centers = rng.integers(0, whole_count, size=len(clouds))
    device = next(model.parameters()).device
    mu, nu, stats = [], [], []
    for start in range(0, len(clouds), microbatch_size):
        end = start + microbatch_size
        source = torch.as_tensor(clouds[start:end].copy(), device=device).transpose(1, 2).contiguous()
        _, whole, _ = get_children_alias(source, whole_count, centers=whole_centers[start:end].tolist())
        before_child = whole.detach().clone()
        _, child, _ = get_children_alias(whole, child_count, centers=child_centers[start:end].tolist())
        if activation_checkpoint:
            from torch.utils.checkpoint import checkpoint, set_checkpoint_early_stop
            # Geoopt's scripted project can wrap checkpoint's Python
            # _StopRecomputationError as an opaque TorchScript RuntimeError.
            # The frame captures this setting at forward construction.
            with set_checkpoint_early_stop(False):
                child_mu, _ = checkpoint(model, child, emb=True, use_reentrant=False,
                                         preserve_rng_state=True)
                whole_mu, _ = checkpoint(model, whole, emb=True, use_reentrant=False,
                                         preserve_rng_state=True)
        else:
            child_mu, _ = model(child, emb=True)
            whole_mu, _ = model(whole, emb=True)
        nu.append(child_mu)
        mu.append(whole_mu)
        stats.append({"microbatch_size": len(whole), "child_aliases_whole":
                      child.untyped_storage().data_ptr() == whole.untyped_storage().data_ptr(),
                      "whole_changed_by_child": not bool(torch.equal(whole, before_child))})
    mu, nu = torch.cat(mu), torch.cat(nu)
    # Hype_pointMLP's actual classifier has no batch-dependent BN. Form it
    # from this same concatenated mu so CE's ball-coordinate partial is valid.
    logits = model.classifier(mu)
    if _array_sha(clouds) != before:
        raise RuntimeError("Original point-cloud inputs changed")
    return {"mu": mu, "nu": nu, "logits": logits,
            "gold": torch.as_tensor(labels, dtype=torch.long, device=device),
            "sample_ids": sample_ids, "child_count": child_count,
            "input_identity": {"input_sha256": before, "sample_ids": sample_ids.tolist(),
                               "labels": labels.tolist(), "whole_count": whole_count,
                               "child_count": child_count, "microbatch_size": microbatch_size,
                               "whole_centers": whole_centers.tolist(),
                               "child_centers": child_centers.tolist(), "crop_seed": seed,
                               "mode": "clean fixed crop / eval BN / original alias overwrite",
                               "classifier_on_concatenated_mu": True,
                               "activation_checkpoint": bool(activation_checkpoint),
                               "checkpoint_use_reentrant": False,
                               "checkpoint_early_stop": False,
                               "checkpoint_preserve_rng_state": True},
            "microbatches": stats}


def _loss_components(values, proxy_tangent, config, seed):
    import torch
    from inter_hierarchy_MN40.hier_proxy_scratch_v5.base_protocol import smoothed_ce, intra_losses
    from inter_hierarchy_MN40.hier_proxy_scratch_v5.relations import (
        mine_reciprocal_triplets, mine_sample_triplets, poincare_distance)
    from inter_hierarchy_MN40.hier_proxy_scratch_v6.hier_loss import expmap0_c1, ghhc_loss
    ce = smoothed_ce(values["logits"], values["gold"], epsilon=float(config.get("label_smoothing", .2)))
    intra = intra_losses(values["mu"], values["nu"], values["child_count"])
    weighted_intra = (float(config.get("alpha_contrastive", .01)) * intra["intra_contrastive"] +
                      float(config.get("alpha_radial", .01)) * intra["intra_radial"])
    losses = {"ce": ce, "weighted_intra": weighted_intra, "base": ce + weighted_intra}
    stats = {"hier_available": proxy_tangent is not None,
             "intra_contrastive": float(intra["intra_contrastive"].detach()),
             "intra_radial": float(intra["intra_radial"].detach())}
    if proxy_tangent is None:
        return losses, stats
    topk = int(config.get("sample_K", config.get("topk", 20)))
    proxy_topk = int(config.get("proxy_K", config.get("proxy_topk", topk)))
    exclude = bool(config.get("exclude_self_negative", True))
    count = int(config.get("t_per_anchor", 50))
    margin, tau = float(config.get("margin", .1)), float(config.get("tau", .1))
    mu, gold = values["mu"], values["gold"]
    sample_mined = mine_sample_triplets(mu, gold, topk=min(topk, len(mu)), t_per_anchor=count,
                                       seed=seed, exclude_self_negative=exclude,
                                       data_ids=torch.as_tensor(values["sample_ids"], device=mu.device))
    proxies = expmap0_c1(proxy_tangent)
    proxy_distances = poincare_distance(proxies)
    proxy_mined = mine_reciprocal_triplets(torch.exp(-proxy_distances.detach()),
                                         topk=min(proxy_topk, len(proxies)), t_per_anchor=count,
                                         seed=seed + 1, exclude_self_negative=exclude)
    generator = torch.Generator(device=mu.device).manual_seed(seed)
    sample_loss, sample_stats = ghhc_loss(poincare_distance(mu, proxies), sample_mined["triplets"],
                                         tau=tau, margin=margin, generator=generator)
    proxy_loss, proxy_stats = ghhc_loss(proxy_distances, proxy_mined["triplets"],
                                       tau=tau, margin=margin, generator=generator)
    losses.update(sample=sample_loss, proxy=proxy_loss)
    stats.update(sample={**sample_mined["stats"], **sample_stats},
                 proxy={**proxy_mined["stats"], **proxy_stats},
                 production_operator="v5_v6_v7_c1_full_unsquared_asinh",
                 margin=margin, tau=tau)
    return losses, stats


def _flat_grad(loss, parameters):
    import torch
    if not loss.requires_grad:
        return torch.cat([torch.zeros(parameter.numel(), dtype=torch.float32) for parameter in parameters])
    gradients = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
    return torch.cat([(torch.zeros_like(parameter) if gradient is None else gradient)
                      .detach().float().cpu().reshape(-1)
                      for parameter, gradient in zip(parameters, gradients)])


def probe_model_batch(model, clouds, labels, sample_ids, *, proxy_tangent=None,
                      config=None, seed=22, whole_count=1024, child_count=256,
                      microbatch_size=32, activation_checkpoint=True):
    """Testable low-level probe on an already loaded model; CPU or GPU."""
    import torch
    config = {} if config is None else dict(config)
    weight = float(config.get("lambda_hier", 0.))
    if not np.isfinite(weight) or weight < 0:
        raise ValueError("Invalid actual HIER objective weight")
    named = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    if not named:
        raise ValueError("Probe requires trainable model parameter identities")
    parameters = [parameter for _, parameter in named]
    indices = {"all_model": [], "shared_encoder": [], "classifier": []}
    offset = 0
    for name, parameter in named:
        positions = torch.arange(offset, offset + parameter.numel(), dtype=torch.long)
        indices["all_model"].append(positions)
        indices["classifier" if name.startswith("classifier.") else "shared_encoder"].append(positions)
        offset += parameter.numel()
    indices = {name: torch.cat(parts) if parts else torch.empty(0, dtype=torch.long)
               for name, parts in indices.items()}
    with preserved_model(model):
        torch.manual_seed(seed)
        if next(model.parameters()).is_cuda:
            torch.cuda.manual_seed_all(seed)
        values = clean_alias_forward(model, clouds, labels, sample_ids, whole_count=whole_count,
                                     child_count=child_count, microbatch_size=microbatch_size, seed=seed,
                                     activation_checkpoint=activation_checkpoint)
        tangent = None if proxy_tangent is None else torch.as_tensor(
            np.asarray(proxy_tangent).copy(), dtype=values["mu"].dtype,
            device=values["mu"].device).requires_grad_(True)
        losses, stats = _loss_components(values, tangent, config, seed)
        vectors = {name: _flat_grad(loss, parameters)
                   for name, loss in losses.items() if name != "base"}
        vectors["base"] = vectors["ce"] + vectors["weighted_intra"]
        if tangent is not None:
            vectors["hier_unweighted"] = vectors["sample"] + vectors["proxy"]
            vectors["hier_weighted"] = weight * vectors["hier_unweighted"]
            vectors["total"] = vectors["base"] + vectors["hier_weighted"]
        else:
            vectors["total"] = vectors["base"]
        groups = {}
        for name, positions in indices.items():
            group = {"parameter_elements": len(positions),
                     "norms": {key: float(vector[positions].norm()) for key, vector in vectors.items()}}
            if tangent is not None:
                group.update(hier_vs_base=gradient_comparison(vectors["hier_weighted"][positions],
                                                             vectors["base"][positions]),
                             sample_vs_base=gradient_comparison(vectors["sample"][positions],
                                                                vectors["base"][positions]),
                             total_vs_base=gradient_comparison(vectors["total"][positions],
                                                               vectors["base"][positions]))
            groups[name] = group
        if tangent is not None:
            losses["hier_unweighted"] = losses["sample"] + losses["proxy"]
            losses["hier_weighted"] = weight * losses["hier_unweighted"]
            losses["total"] = losses["base"] + losses["hier_weighted"]
        else:
            losses["total"] = losses["base"]
        whole_partials, proxy_partials = {}, {}
        for name, loss in losses.items():
            gradient, = torch.autograd.grad(loss, (values["mu"],), retain_graph=True, allow_unused=True)
            whole_partials[name] = gradient_summary(values["mu"], gradient, sample_ids)
            if tangent is not None:
                gradient, = torch.autograd.grad(loss, (tangent,), retain_graph=True, allow_unused=True)
                proxy_partials[name] = gradient_summary(tangent, gradient, np.arange(len(tangent)))
        report = {"format": "hier-frozen-backbone-gradient-v1",
                  "scope": "Actual shared-parameter partials under fixed clean crops/eval BN; no optimizer step or train-BN causal claim",
                  "lambda_hier": weight, "objective_values": {key: float(value.detach()) for key, value in losses.items()},
                  "hier_stats": stats, "parameter_groups": groups,
                  "whole_ball_partials": whole_partials, "proxy_tangent_partials": proxy_partials,
                  "input_identity": values["input_identity"], "microbatches": values["microbatches"],
                  "gradient_identity": {"model_parameter_names": [name for name, _ in named],
                                        "model_parameter_elements": offset,
                                        "before_clipping": True, "no_optimizer_state_consulted": True},
                  "read_only": {"optimizer_updates": 0, "BN_parameters_grad_buffers_unchanged": True,
                                "RNG_restored": True, "source_clouds_unchanged": True,
                                "new_validation_forwards": 0, "new_test_forwards": 0}}
    return report


def run_backbone_probe(spec, output_dir, data_dir, gpu, fixed_ids=None, seed=22,
                       whole_count=1024, child_count=256, microbatch_size=32,
                       max_seconds=600, activation_checkpoint=True):
    """Run one own trusted checkpoint in a fresh process on an explicit idle GPU."""
    started = time.monotonic()
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError("Backbone probe output must be a fresh directory")
    if gpu is None:
        raise ValueError("An explicit physical GPU is required")
    if not spec.get("trusted_checkpoint"):
        raise ValueError("Backbone checkpoint must be explicitly identified as the user's own trusted artifact")
    checkpoint_path = Path(spec.get("optimizer_checkpoint") or spec.get("checkpoint", "")).resolve()
    if not checkpoint_path.is_file() or checkpoint_path == output or output in checkpoint_path.parents:
        raise ValueError("Explicit existing checkpoint and separate fresh output are required")
    idle_gpu(int(gpu))
    os.environ["CUDA_VISIBLE_DEVICES"], os.environ["CUDA_DEVICE_ORDER"] = str(gpu), "PCI_BUS_ID"
    import torch
    if torch.cuda.is_initialized():
        raise RuntimeError("Use a fresh process; CUDA was initialized before explicit idle-GPU selection")
    sys.path.insert(0, str(REPO / "pointnet2_ops_lib"))
    sys.path.insert(0, str(REPO / "inter_hierarchy_MN40"))
    from models.pointmlp import Hype_pointMLP
    from .mechanisms import _checkpoint
    source_sha = sha256(checkpoint_path)
    saved = _checkpoint(checkpoint_path, True)
    epoch = checkpoint_epoch(saved, "best" if checkpoint_path.name == "best.pth" else None)
    if spec.get("epoch") is not None and int(spec["epoch"]) != epoch:
        raise ValueError("Backbone checkpoint is not the declared actual epoch")
    config = {**saved.get("training_config", {}), **spec}
    if float(config.get("c", 0)) != 1.:
        raise ValueError("Frozen HyCoRe/HIER objective adapter requires actual c=1")
    points, all_labels, shards = load_training_shards(data_dir)
    labels_sha = hashlib.sha256(all_labels.astype(np.int64).tobytes()).hexdigest()
    expected_labels_sha = spec.get("train_labels_sha256", saved.get("train_labels_sha256"))
    if expected_labels_sha is not None and labels_sha != expected_labels_sha:
        raise ValueError("Loaded training labels differ from recorded input identity")
    train_ids = np.asarray(saved.get("train_ids", []), dtype=np.int64)
    if not len(train_ids) or len(np.unique(train_ids)) != len(train_ids) or train_ids.min() < 0 or train_ids.max() >= len(points):
        raise ValueError("Saved train IDs are required; do not guess a split")
    if fixed_ids is None:
        positions = _balanced_positions(all_labels[train_ids], 64, seed)
        ids = train_ids[positions]
        ids = ids[np.lexsort((ids, all_labels[ids]))]
    else:
        ids = np.asarray(fixed_ids, dtype=np.int64)
        if len(ids) != 64 or len(np.unique(ids)) != 64 or not set(ids.tolist()).issubset(set(train_ids.tolist())):
            raise ValueError("Fixed probe IDs must be64 unique objects in the saved training split")
    clouds, labels = points[ids, :1024].copy(), all_labels[ids].copy()
    if len(ids) != 64 or np.any(labels == labels[::-1]):
        raise ValueError("Probe requires64 ordered objects with different-class global-flip negatives")
    model = Hype_pointMLP()
    model.load_state_dict({key.removeprefix("module."): value for key, value in saved["net"].items()}, strict=True)
    model.manifold.requires_grad_(False)
    model.manifold2.requires_grad_(False)
    proxy_values = [value for key, value in (saved.get("proxy") or {}).items()
                    if str(key).split(".")[-1] == "tangent_proxies"]
    if len(proxy_values) > 1:
        raise ValueError("Ambiguous proxy parameter identity")
    proxy_tangent = proxy_values[0].detach().cpu().numpy() if proxy_values else None
    if proxy_tangent is not None:
        from .extract import resolve_proxy_mapping
        if resolve_proxy_mapping(saved, saved.get("training_config", {})) != {"numeric_radius_fraction": .999}:
            raise ValueError("Backbone probe supports only the inspected live proxy map")
    output.mkdir(parents=True, exist_ok=False)
    manifest = {"format": "hier-frozen-backbone-gradient-v1", "status": "running",
                "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "physical_gpu": int(gpu),
                "source_checkpoint_file": checkpoint_path.name, "source_checkpoint_sha256": source_sha,
                "epoch": epoch, "run_key": spec.get("run_key"), "seed": seed,
                "optimizer_updates": 0, "new_validation_forwards": 0, "new_test_forwards": 0,
                "max_seconds": max_seconds}
    _save(output / "backbone_probe_manifest.json", manifest)
    try:
        if time.monotonic() - started > max_seconds:
            raise TimeoutError("Probe preparation exceeded the bounded budget")
        idle_gpu(int(gpu))  # Fresh check immediately before the first CUDA allocation.
        model = model.to("cuda:0")
        report = probe_model_batch(model, clouds, labels, ids, proxy_tangent=proxy_tangent,
                                   config=config, seed=seed, whole_count=whole_count,
                                   child_count=child_count, microbatch_size=microbatch_size,
                                   activation_checkpoint=activation_checkpoint)
        torch.cuda.synchronize()
        if sha256(checkpoint_path) != source_sha:
            raise RuntimeError("Source checkpoint changed during the probe")
        report.update(run_key=spec.get("run_key"), display_name=spec.get("display_name"),
                      epoch=epoch, checkpoint_sha256=source_sha, source_checkpoint_file=checkpoint_path.name,
                      source_shards=shards, source_commit=spec.get("source_commit"),
                      train_labels_sha256=labels_sha,
                      recorded_train_labels_identity_verified=expected_labels_sha is not None,
                      model_source_sha256=sha256(REPO / "inter_hierarchy_MN40/models/pointmlp.py"),
                      input_condition={"mode": "fixed clean crop evalBN original alias",
                                       "microbatch_size": microbatch_size, "seed": seed,
                                       "whole_count": whole_count, "child_count": child_count,
                                       "TF32_matmul": torch.backends.cuda.matmul.allow_tf32,
                                       "activation_checkpoint": bool(activation_checkpoint),
                                       "checkpoint_use_reentrant": False,
                                       "checkpoint_early_stop": False,
                                       "checkpoint_preserve_rng_state": True})
        _save(output / "backbone_probe_summary.json", report)
        manifest.update(status="completed", elapsed_seconds=time.monotonic() - started,
                        source_unchanged=True, peak_allocated_MiB=torch.cuda.max_memory_allocated() / 1024**2)
        _save(output / "backbone_probe_manifest.json", manifest)
        return report
    except Exception as exc:
        manifest.update(status="failed", error=repr(exc), elapsed_seconds=time.monotonic() - started)
        _save(output / "backbone_probe_manifest.json", manifest)
        raise
