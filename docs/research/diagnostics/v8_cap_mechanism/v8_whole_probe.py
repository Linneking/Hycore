"""Bounded V8 whole/part pre-cap and coordinate-gradient diagnostic.

Fixed training objects, eval BN, source crop aliasing, no encoder backward,
no optimizer/test. A capped coordinate leaf and the actual raw-tangent path
are measured separately; their gradients have different meanings.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time

import h5py
import numpy as np
import torch

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[4]
for p in (ROOT, ROOT / "classification_ModelNet40", ROOT / "inter_hierarchy_MN40", ROOT / "pointnet2_ops_lib"):
    sys.path.insert(0, str(p))
from models.pointmlp import Hype_pointMLP  # noqa: E402
from inter_hierarchy_MN40.hier_proxy_scratch_v8.geometry import configure_backbone, to_ball, depth  # noqa: E402
from inter_hierarchy_MN40.hier_proxy_scratch_v8.base_protocol import smoothed_ce, intra_losses  # noqa: E402
from inter_hierarchy_MN40.hier_proxy_scratch_v8.hier_loss import HIERLoss  # noqa: E402
from tools.hier_postrun_audit.backbone_probe import clean_alias_forward  # noqa: E402


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def state_hash(model):
    h = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def write_json(path, data):
    temporary = Path(str(path) + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def stats(value):
    a = np.asarray(value, dtype=np.float64).reshape(-1)
    if not len(a) or not np.isfinite(a).all():
        raise ValueError("Empty/nonfinite probe values")
    return {"min": float(a.min()), "p05": float(np.quantile(a, .05)),
            "median": float(np.median(a)), "p95": float(np.quantile(a, .95)),
            "max": float(a.max()), "mean": float(a.mean()), "std": float(a.std())}


def radial(gradient, position):
    if gradient is None:
        gradient = torch.zeros_like(position)
    direction = position.detach() / position.detach().norm(dim=-1, keepdim=True).clamp_min(1e-12)
    signed = (gradient.detach() * direction).sum(-1)
    norm = gradient.detach().norm(dim=-1)
    angular = (gradient.detach() - signed[:, None] * direction).norm(dim=-1)
    active = norm > 1e-12
    signed_np = signed.cpu().numpy()
    return {"gradient_norm": stats(norm.cpu().numpy()), "radial_signed": stats(signed_np),
            "angular_norm": stats(angular.cpu().numpy()),
            "negative_gradient_outward_fraction_all": float((signed < -1e-12).float().mean()),
            "negative_gradient_inward_fraction_all": float((signed > 1e-12).float().mean()),
            "nonzero_gradient_count": int(active.sum()),
            "signed_radial_values": signed_np.tolist(),
            "semantics": "Positive radial derivative means inward under plain negative-gradient descent; before encoder sharing/optimizer."}


def choose_ids(labels, allowed, seed=22):
    rng = np.random.default_rng(seed)
    classes = np.sort(rng.choice(np.unique(labels[allowed]), 32, replace=False))
    result = np.concatenate([rng.choice(allowed[labels[allowed] == c], 2, replace=False) for c in classes])
    if np.any(labels[result] == labels[result][::-1]):
        raise AssertionError("Fixed flip negatives must be different classes")
    return result


def leaf_probe(model, proxy, raw_mu, raw_nu, actual_mu, actual_nu, labels, ids, child_count, seed):
    results = {}
    values_reference = None
    for space in ("capped_ball_leaf", "raw_precap_tangent_leaf"):
        if space == "capped_ball_leaf":
            a = actual_mu.detach().clone().requires_grad_(True)
            b = actual_nu.detach().clone().requires_grad_(True)
            mu, nu = a, b
        else:
            a = raw_mu.detach().clone().requires_grad_(True)
            b = raw_nu.detach().clone().requires_grad_(True)
            mu, nu = to_ball(a), to_ball(b)
            error = max(float((mu.detach() - actual_mu).abs().max()),
                        float((nu.detach() - actual_nu).abs().max()))
            if error > 2e-5:
                raise AssertionError(f"Precap reconstruction mismatch {error}")
        logits = model.classifier(mu)
        ce = smoothed_ce(logits, labels)
        intra = intra_losses(mu, nu, child_count)
        # Frozen proxy parameters; full production HIER operator supplies the
        # identical sample component, with self-k repair and ST retained.
        torch.manual_seed(seed)
        generator = torch.Generator(device=mu.device).manual_seed(seed)
        _, info = proxy(mu, labels, topk=20, seed=seed, data_ids=torch.as_tensor(ids, device=mu.device),
                        generator=generator, return_components=True, monitor_structure=False)
        losses = {"CE": ce, "0.01_Rcontr": .01 * intra["intra_contrastive"],
                  "0.01_Rhier": .01 * intra["intra_radial"],
                  "0.5_sample_HIER": .5 * info["_components"]["sample_loss"]}
        scalar_values = {name: float(value.detach()) for name, value in losses.items()}
        if values_reference is None:
            values_reference = scalar_values
        elif any(abs(scalar_values[name] - values_reference[name]) > 2e-5 for name in scalar_values):
            raise AssertionError("Leaf spaces changed forward losses or random selections")
        terms = {}
        for name, loss in losses.items():
            grads = torch.autograd.grad(loss, (a, b), retain_graph=True, allow_unused=True)
            terms[name] = {"loss": scalar_values[name], "whole": radial(grads[0], a),
                           "part": radial(grads[1], b)}
        total = sum(losses.values())
        grads = torch.autograd.grad(total, (a, b), allow_unused=True)
        terms["sum"] = {"loss": float(total.detach()), "whole": radial(grads[0], a),
                        "part": radial(grads[1], b)}
        results[space] = terms
        del total, losses, info, ce, intra, mu, nu, a, b
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inputs", required=True)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--microbatch", type=int, default=8)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--seed", type=int, default=22)
    p.add_argument("--last-epoch", type=int, default=300)
    args = p.parse_args()
    if not 1 <= args.microbatch <= 8:
        p.error("Use microbatch1--8")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    started = time.time()
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    last = f"e{args.last_epoch:03d}"
    sources = {"e005": Path(args.inputs) / "model_e005.pt", last: Path(args.inputs) / f"model_{last}.pt"}
    hashes = {name: sha(path) for name, path in sources.items()}
    snapshots = {name: torch.load(path, map_location="cpu", weights_only=False) for name, path in sources.items()}
    files = sorted(Path(args.data_dir).glob("ply_data_train*.h5"))
    if not files:
        raise FileNotFoundError("No official training HDF5 shards")
    clouds, labels, shard_hashes = [], [], {}
    for file in files:
        with h5py.File(file, "r") as f:
            clouds.append(np.asarray(f["data"], dtype=np.float32)[:, :1024])
            labels.append(np.asarray(f["label"]).reshape(-1).astype(np.int64))
        shard_hashes[file.name] = sha(file)
    clouds, labels = np.concatenate(clouds), np.concatenate(labels)
    if len(clouds) != 9840:
        raise AssertionError("Expected full official9840 training objects")
    allowed = np.asarray(snapshots[last].get("train_ids", np.arange(9840)), dtype=np.int64)
    if not np.array_equal(allowed, np.asarray(snapshots["e005"].get("train_ids", allowed))):
        raise AssertionError("Snapshot training memberships differ")
    ids = choose_ids(labels, allowed, args.seed)
    gold = torch.as_tensor(labels[ids], device=device, dtype=torch.long)
    rows = []
    write_json(out / "progress.json", {"phase": "inputs_validated", "source_sha256": hashes,
                                       "selected_training_objects": len(ids), "no_test": True})
    for name, counts in (("e000", (400,)), ("e005", (400,)), (last, (200, 400, 600))):
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        model = configure_backbone(Hype_pointMLP())
        initial_hash = state_hash(model)
        cp = snapshots["e005"] if name == "e000" else snapshots[name]
        expected_init = cp.get("initialized_model_sha256")
        if expected_init and initial_hash != expected_init:
            raise AssertionError(f"Initialization mismatch {initial_hash} != {expected_init}")
        if name != "e000":
            model.load_state_dict(cp["net"], strict=True)
        model.to(device).eval().requires_grad_(False)
        proxy = HIERLoss(seed=args.seed).to(device)
        if name != "e000":
            proxy.load_state_dict(cp["proxy"], strict=True)
        proxy.requires_grad_(False)
        state_before = state_hash(model)
        proxy_before = state_hash(proxy)
        for count in counts:
            collected = []

            def capture(module, _inputs, raw_ball):
                collected.append(module.manifold.logmap0(raw_ball).detach())

            # Existing FinalGeometry was installed first: prepend observes
            # Mobius output before cap/hook, returns None so output is unaltered.
            handle = model.emb.register_forward_hook(capture, prepend=True)
            try:
                with torch.no_grad():
                    value = clean_alias_forward(model, clouds[ids], labels[ids], ids,
                                                whole_count=1024, child_count=count,
                                                microbatch_size=args.microbatch, seed=args.seed)
            finally:
                handle.remove()
            if len(collected) % 2 or len(collected) == 0:
                raise AssertionError("Expected alternating part/whole raw-hook values")
            raw_nu, raw_mu = torch.cat(collected[0::2]), torch.cat(collected[1::2])
            mu, nu = value["mu"].detach(), value["nu"].detach()
            derivatives = leaf_probe(model, proxy, raw_mu, raw_nu, mu, nu, gold, ids,
                                     count, args.seed + count)
            row = {"stage": name, "child_count": count, "objects": len(ids),
                   "raw_whole_tangent_norm": stats(raw_mu.norm(dim=-1).cpu().numpy()),
                   "raw_part_tangent_norm": stats(raw_nu.norm(dim=-1).cpu().numpy()),
                   "whole_precap_over_2_3_count": int((raw_mu.norm(dim=-1) > 2.3).sum()),
                   "part_precap_over_2_3_count": int((raw_nu.norm(dim=-1) > 2.3).sum()),
                   "whole_depth": stats(depth(mu).cpu().numpy()),
                   "part_depth": stats(depth(nu).cpu().numpy()),
                   "gradient_terms": derivatives,
                   "model_BN_unchanged": state_before == state_hash(model),
                   "proxy_unchanged": proxy_before == state_hash(proxy),
                   "initial_model_sha256": initial_hash, "source_sha256": None if name == "e000" else hashes[name],
                   "condition": "fixed clean1024 original crop alias, evalBN; coordinate derivatives only",
                   "encoder_backward_calls": 0, "optimizer_updates": 0, "test_forward_calls": 0}
            if not row["model_BN_unchanged"] or not row["proxy_unchanged"]:
                raise AssertionError("A source state changed")
            write_json(out / f"{name}_N{count}.json", row)
            np.savez_compressed(out / f"private_{name}_N{count}.npz", sample_ids=ids, labels=labels[ids],
                                raw_mu=raw_mu.cpu().numpy(), raw_nu=raw_nu.cpu().numpy(),
                                mu=mu.cpu().numpy(), nu=nu.cpu().numpy())
            rows.append(row)
            write_json(out / "progress.json", {"phase": "stage_complete", "stage": name,
                                               "child_count": count, "completed_stages": len(rows)})
            print(json.dumps({"stage": name, "child_count": count, "completed": True,
                              "whole_depth_median": row["whole_depth"]["median"],
                              "whole_precap_over": row["whole_precap_over_2_3_count"]}), flush=True)
        del model, proxy
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if any(sha(sources[name]) != hashes[name] for name in hashes):
        raise AssertionError("A source checkpoint file changed")
    summary = {"complete": True, "source_sha256": hashes, "training_shard_sha256": shard_hashes,
               "stages": rows, "train_pool_objects": 9840, "eligible_training_objects": len(allowed),
               "sample_selection": "seed22 fixed32 classes x2, common checkpoint training membership",
               "device": str(device), "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
               "seconds": time.time() - started, "finished_utc": datetime.now(timezone.utc).isoformat(),
               "runner_sha256": sha(__file__), "optimizer_updates": 0, "test_forward_calls": 0,
               "limits": "One fixed64-object clean/evalBN panel; coordinate derivatives are not actual shared-encoder optimizer moves."}
    write_json(out / "summary.json", summary)
    write_json(out / "progress.json", {"phase": "complete", "completed_stages": len(rows)})


if __name__ == "__main__":
    main()
