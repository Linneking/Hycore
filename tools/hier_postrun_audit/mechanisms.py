"""CPU HIER query/noise/gradient audits of explicitly identified frozen caches.

The production HIER kernel and production proxy map are reused. No model,
optimizer, GPU or test-set forward is run. A cache partial derivative is not
a shared-encoder gradient and a signed gradient is not an AdamW displacement.
"""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import pickle
import time

import numpy as np

from .geometry import ball_geometry, pool_hash, quantiles, unique_ids

REPO = Path(__file__).resolve().parents[2]
PRODUCTION_OPERATOR = "v5_v6_v7_c1_full_unsquared_asinh"
REFERENCE_OPERATOR = "reviewed_pmath_mobius_eps1e-5_artanh_clip1e-5_counterfactual"


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False), encoding="utf-8")


def _tensor_digest(tensor):
    array = tensor.detach().cpu().contiguous().numpy()
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode())
    digest.update(str(array.dtype).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _checkpoint(path, trusted):
    import torch
    try:
        saved = torch.load(path, map_location="cpu", weights_only=not trusted)
    except pickle.UnpicklingError as exc:
        raise ValueError("Legacy RNG/Python checkpoint needs explicit trusted_checkpoint=true for an own trusted artifact") from exc
    if not isinstance(saved, dict):
        raise ValueError("Checkpoint must be a mapping")
    return saved


def _load(spec):
    path = Path(spec["cache"])
    before = _sha(path)
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    metadata = {}
    for key in ("metadata_json", "metadata"):
        if key in arrays and arrays[key].shape == () and arrays[key].dtype.kind in "US":
            decoded = json.loads(str(arrays[key].item()))
            if not isinstance(decoded, dict):
                raise ValueError("Cache metadata must be an object")
            metadata.update(decoded)
    for key in ("input_sha256", "input_mode", "inputmode", "c", "curvature"):
        if key in arrays and arrays[key].shape == ():
            metadata[key] = arrays[key].item()
    if _sha(path) != before:
        raise ValueError("Source cache changed during reading")
    ids = unique_ids(arrays["sample_ids"])
    labels = np.asarray(arrays["labels"]).reshape(-1)
    whole = arrays.get("mu", arrays.get("whole_mu"))
    if whole is None or whole.ndim != 2 or len(whole) != len(ids) or len(labels) != len(ids):
        raise ValueError("Cache needs aligned mu/sample_ids/labels")
    if labels.dtype.kind not in "iuUS":
        raise ValueError("Labels must contain integer or string identities")
    c = float(spec.get("c", metadata.get("c", metadata.get("curvature", 0))))
    if c != 1.:
        raise ValueError("Mechanism production adapter currently supports only the actual c=1 V5/V6/V7 kernel")
    ball_geometry(whole, c)
    version = str(spec.get("version", "")).lower()
    if version not in ("v5", "v6", "v7"):
        raise ValueError("Mechanism audit requires explicit version=v5/v6/v7")
    mapping = spec.get("proxy_mapping")
    if not isinstance(mapping, dict) or mapping.get("numeric_radius_fraction") != .999 or mapping.get("source_tangent_cap") is not None:
        raise ValueError("Only declared production .999 projection without extra forward tangent cap is supported")
    checkpoint = None
    checkpoint_hash = None
    checkpoint_path = spec.get("optimizer_checkpoint") or spec.get("checkpoint")
    if checkpoint_path:
        checkpoint_hash = _sha(checkpoint_path)
        checkpoint = _checkpoint(checkpoint_path, bool(spec.get("trusted_checkpoint")))
        if _sha(checkpoint_path) != checkpoint_hash:
            raise ValueError("Source checkpoint changed during reading")
    tangent = arrays.get("proxy_tangent")
    if tangent is None and checkpoint is not None:
        values = [value for key, value in checkpoint.get("proxy", {}).items()
                  if str(key).split(".")[-1] == "tangent_proxies"]
        if len(values) == 1:
            tangent = values[0].detach().cpu().numpy().copy()
        elif len(values) > 1:
            raise ValueError("Ambiguous saved proxy parameters")
    if tangent is None:
        raise ValueError("Proxy tangent parameters are required; mapped ball positions cannot reconstruct saturated stored parameters")
    tangent = np.asarray(tangent)
    if tangent.ndim != 2 or len(tangent) < 3 or len(whole) < 3 or tangent.shape[1] != whole.shape[1] or not np.isfinite(tangent).all():
        raise ValueError("Invalid saved proxy tangent parameter matrix")
    proxy_ids = unique_ids(arrays.get("proxy_ids", spec.get("proxy_ids", [])), "proxy_ids")
    if len(proxy_ids) != len(tangent):
        raise ValueError("Explicit stable proxy IDs must identify every tangent parameter row")
    order, proxy_order = np.argsort(ids, kind="stable"), np.argsort(proxy_ids, kind="stable")
    optimizer = checkpoint.get("proxy_optimizer") if checkpoint is not None else None
    # Adam state row order is checkpoint parameter order: reorder it with IDs too.
    if optimizer and not np.array_equal(proxy_order, np.arange(len(proxy_order))):
        optimizer = dict(optimizer)
        optimizer["state"] = {key: dict(value) for key, value in optimizer.get("state", {}).items()}
        for value in optimizer["state"].values():
            for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
                if key in value and tuple(value[key].shape) == tuple(tangent.shape):
                    value[key] = value[key][proxy_order].clone()
    input_sha = spec.get("input_sha256", metadata.get("input_sha256"))
    input_mode = spec.get("input_mode", spec.get("inputmode", metadata.get("input_mode", metadata.get("inputmode"))))
    inference = spec.get("inference_condition", metadata.get("inference_condition"))
    if spec.get("inference_condition") is not None and metadata.get("inference_condition") is not None and spec["inference_condition"] != metadata["inference_condition"]:
        raise ValueError("Declared inference condition contradicts cache identity")
    inference_sha = hashlib.sha256(json.dumps(inference, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest() if inference is not None else None
    exclude = bool(spec.get("exclude_self_negative", version != "v5"))
    if version != "v5" and not exclude:
        raise ValueError("V6/V7 production mining excludes self-negative queries")
    if not spec.get("run_key"):
        raise ValueError("Explicit run_key is required; proxy identities are only compared within a run")
    epoch = spec.get("epoch")
    if epoch is None:
        raise ValueError("Explicit actual checkpoint epoch is required")
    result = {"spec": dict(spec), "whole": whole[order].copy(),
              "sample_ids": ids[order], "labels": labels[order],
              "tangent": tangent[proxy_order].copy(), "proxy_ids": proxy_ids[proxy_order],
              "optimizer": optimizer, "version": version, "exclude_self_negative": exclude,
              "run_key": str(spec["run_key"]), "epoch": int(epoch),
              "identity": {"cache_file": path.name, "cache_sha256": before,
                           "checkpoint_file": Path(spec.get("checkpoint") or checkpoint_path).name if checkpoint_path else None,
                           "optimizer_checkpoint_file": Path(checkpoint_path).name if checkpoint_path else None,
                           "optimizer_checkpoint_sha256": checkpoint_hash,
                           "inference_condition": inference,
                           "inference_condition_sha256": inference_sha,
                           "checkpoint_sha256": spec.get("checkpoint_sha256") or checkpoint_hash,
                           "sample_pool_sha256": pool_hash(ids), "input_sha256": input_sha,
                           "input_mode": input_mode, "proxy_id_policy": spec.get("proxy_id_policy")}}
    if checkpoint is not None:
        saved_epoch = checkpoint.get("completed_epochs", checkpoint.get("epoch"))
        if saved_epoch is not None and int(saved_epoch) != int(epoch):
            raise ValueError("Declared epoch disagrees with actual saved checkpoint epoch")
    return result


def _kernel():
    from inter_hierarchy_MN40.hier_proxy_scratch_v5.relations import (
        mine_reciprocal_triplets, mine_sample_triplets, poincare_distance)
    from inter_hierarchy_MN40.hier_proxy_scratch_v6.hier_loss import expmap0_c1, ghhc_loss
    return mine_reciprocal_triplets, mine_sample_triplets, poincare_distance, expmap0_c1, ghhc_loss


def reference_distance(path=None):
    """Load only reviewed, local mathematical definitions, never arbitrary imports."""
    import torch
    path = Path(path) if path else REPO / "inter_hierarchy_MN40/models/ops/pmath.py"
    source = path.read_text(encoding="utf-8")
    names = {"Artanh", "artanh", "_tensor_dot", "_mobius_addition_batch", "_dist_matrix", "dist_matrix"}
    nodes = [node for node in ast.parse(source).body
             if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
    if {node.name for node in nodes} != names:
        raise ValueError("Reviewed reference mathematical definitions are missing")
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    function = namespace["dist_matrix"]

    def distance(x, y):
        # The original N*M*D intermediate is bounded by the chunk width.
        if not len(x):
            return x.new_empty((0, len(y)))
        return torch.cat([function(x[start:start + 16], y, c=1.)
                          for start in range(0, len(x), 16)], dim=0)
    return distance, {"operator": REFERENCE_OPERATOR, "file": path.name,
                      "source_sha256": _sha(path), "role": "same-coordinate numerical counterfactual; not the production objective"}


def _select_queries(mined, count, seed):
    import torch
    triples = mined["triplets"].detach().cpu()
    if len(triples) <= count:
        return triples.clone()
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(len(triples), count, replace=False))
    return triples[torch.as_tensor(indices, dtype=torch.long)].clone()


def _balanced_positions(labels, size=64, seed=22):
    """One fixed clean batch; up to32 classes x2, no repeated object identities."""
    labels = np.asarray(labels)
    size = min(int(size), len(labels))
    rng = np.random.default_rng(seed)
    available = [label for label in np.unique(labels) if np.sum(labels == label) >= 2]
    count = min(len(available), size // 2)
    chosen_labels = rng.choice(np.asarray(available), size=count, replace=False)
    chosen = [int(position) for label in chosen_labels
              for position in rng.choice(np.flatnonzero(labels == label), size=2, replace=False)]
    if len(chosen) < size:
        remaining = np.asarray([position for position in range(len(labels)) if position not in set(chosen)])
        chosen.extend(rng.choice(remaining, size=size-len(chosen), replace=False).tolist())
    return np.asarray(sorted(chosen), dtype=np.int64)


def make_query_plan(snapshot, query_count=64, seed=22, sample_batch_size=64):
    """Freeze query object/proxy IDs mined once at this run's first snapshot."""
    import torch
    reciprocal, sample_mining, distance, expmap, _ = _kernel()
    spec = snapshot["spec"]
    topk = int(spec.get("sample_K", spec.get("topk", 20)))
    proxy_topk = int(spec.get("proxy_K", spec.get("proxy_topk", topk)))
    t_per_anchor = int(spec.get("t_per_anchor", 50))
    batch_positions = _balanced_positions(snapshot["labels"], sample_batch_size, seed)
    whole = torch.as_tensor(snapshot["whole"][batch_positions], dtype=torch.float32, device="cpu")
    label_values, encoded_labels = np.unique(snapshot["labels"][batch_positions], return_inverse=True)
    labels = torch.as_tensor(encoded_labels, dtype=torch.long)
    ids = torch.arange(len(whole), dtype=torch.long)
    tangent = torch.as_tensor(snapshot["tangent"], dtype=torch.float32, device="cpu")
    with torch.no_grad():
        sample = sample_mining(whole, labels, topk=min(topk, len(whole)),
                               t_per_anchor=t_per_anchor, seed=seed,
                               exclude_self_negative=snapshot["exclude_self_negative"], data_ids=ids)
        proxy_ball = expmap(tangent)
        proxy = reciprocal(torch.exp(-distance(proxy_ball)), topk=min(proxy_topk, len(proxy_ball)),
                           t_per_anchor=t_per_anchor, seed=seed + 1,
                           exclude_self_negative=snapshot["exclude_self_negative"])
    sample_rows = _select_queries(sample, query_count, seed + 2).numpy()
    proxy_rows = _select_queries(proxy, query_count, seed + 3).numpy()
    result = {"run_key": snapshot["run_key"], "reference_epoch": snapshot["epoch"],
              "scope": "fixed clean object batch from candidate pool, production mining function, sampled query subset; not a replay of augmented training batches",
              "query_count_requested_per_component": query_count, "seed": seed,
              "sample_batch_ids": snapshot["sample_ids"][batch_positions].tolist(),
              "sample_batch_size": len(batch_positions),
              "sample_query_ids": snapshot["sample_ids"][batch_positions[sample_rows]].tolist(),
              "proxy_query_ids": snapshot["proxy_ids"][proxy_rows].tolist(),
              "mining_parameters": {"topk_including_self": min(topk, len(whole)),
                                    "proxy_topk_including_self": min(proxy_topk, len(proxy_ball)),
                                    "t_per_anchor": t_per_anchor,
                                    "exclude_self_negative": snapshot["exclude_self_negative"]},
              "sample_mining_stats": sample["stats"], "proxy_mining_stats": proxy["stats"],
              "reference_identity": snapshot["identity"],
              "proxy_ids": snapshot["proxy_ids"].tolist(),
              "labels": snapshot["labels"].tolist(),
              "sample_ids": snapshot["sample_ids"].tolist()}
    result["query_plan_sha256"] = hashlib.sha256(json.dumps(
        {key: result[key] for key in ("sample_query_ids", "proxy_query_ids",
                                    "mining_parameters")}, sort_keys=True).encode()).hexdigest()
    return result


def _remap(query_ids, identities):
    lookup = {value: index for index, value in enumerate(identities.tolist())}
    try:
        return np.asarray([[lookup[value] for value in row] for row in query_ids],
                          dtype=np.int64).reshape(-1, 3)
    except KeyError as exc:
        raise ValueError("Fixed query identity is absent from this snapshot") from exc


def _compatibility(plan, snapshot):
    if plan["run_key"] != snapshot["run_key"]:
        return "Different runs do not share proxy identity"
    first, current = plan["reference_identity"], snapshot["identity"]
    for key in ("sample_pool_sha256", "input_sha256", "input_mode", "inference_condition_sha256"):
        if first.get(key) is None or current.get(key) is None:
            return "Temporal pairing lacks explicit " + key
        if first[key] != current[key]:
            return "Temporal pairing identity differs: " + key
    if plan["proxy_ids"] != snapshot["proxy_ids"].tolist():
        return "Stable proxy identity set differs"
    if plan["sample_ids"] != snapshot["sample_ids"].tolist() or plan["labels"] != snapshot["labels"].tolist():
        return "Object identities or labels differ"
    if plan["mining_parameters"]["exclude_self_negative"] != snapshot["exclude_self_negative"]:
        return "Self-negative policy differs within the run"
    return None


def gradient_summary(parameter, gradient, ids, absolute_tolerance=1e-12):
    """Euclidean partials: positive signed radial derivative means GD inward."""
    import torch
    if gradient is None:
        gradient = torch.zeros_like(parameter)
    if not bool(torch.isfinite(gradient).all()):
        raise ValueError("Nonfinite parameter partial derivative")
    parameter, gradient = parameter.detach(), gradient.detach()
    radius = parameter.norm(dim=-1, keepdim=True)
    direction = parameter / radius.clamp_min(1e-15)
    signed = (gradient * direction).sum(-1)
    angular = gradient - signed[:, None] * direction
    norms = gradient.norm(dim=-1)
    above = norms > absolute_tolerance
    defined = radius[:, 0] > 0
    result = {"gradient_norm": float(gradient.norm()), "radial_norm": float(signed.norm()),
              "angular_norm": float(angular.norm()),
              "exact_nonzero_row_count": int((norms > 0).sum()),
              "above_tolerance_row_count": int(above.sum()),
              "absolute_tolerance": float(absolute_tolerance),
              "signed_radial_mean": float(signed.mean()) if len(signed) else None,
              "positive_signed_radial_count_above_tolerance": int((signed > absolute_tolerance).sum()),
              "negative_signed_radial_count_above_tolerance": int((signed < -absolute_tolerance).sum()),
              "radial_direction_undefined_count": int((~defined).sum()),
              "semantics": "Partial gradient before clipping/optimizer. Positive radial derivative gives inward plain-GD displacement; AdamW displacement is not measured.",
              "row_gradient_norm_quantiles": quantiles(norms.cpu().numpy()),
              "signed_radial_quantiles": quantiles(signed.cpu().numpy()),
              "per_id": [{"id": identity, "norm": float(norm), "signed_radial": float(radial),
                          "angular_norm": float(angle), "radial_direction_defined": bool(valid)}
                         for identity, norm, radial, angle, valid in zip(
                             np.asarray(ids).tolist(), norms, signed, angular.norm(dim=-1), defined)]}
    return result


def adam_state_summary(snapshot):
    """Read existing moments/preconditioners without estimating an actual step."""
    import torch
    optimizer = snapshot.get("optimizer")
    if optimizer is None:
        return {"available": False, "reason": "No explicitly supplied saved proxy optimizer state"}
    states = optimizer.get("state", {})
    groups = optimizer.get("param_groups", [])
    if len(states) != 1 or len(groups) != 1 or len(groups[0].get("params", [])) != 1:
        return {"available": False, "reason": "Proxy optimizer identity is ambiguous"}
    state, group = next(iter(states.values())), groups[0]
    if "exp_avg" not in state or "exp_avg_sq" not in state:
        return {"available": False, "reason": "Saved proxy optimizer has no Adam moments"}
    parameter = torch.as_tensor(snapshot["tangent"], dtype=torch.float32)
    first, second = state["exp_avg"].detach().cpu(), state["exp_avg_sq"].detach().cpu()
    if first.shape != parameter.shape or second.shape != parameter.shape or not bool(torch.isfinite(first).all() and torch.isfinite(second).all()) or bool((second < 0).any()):
        raise ValueError("Saved Adam moment axes/values are invalid")
    step = float(state.get("step", 0))
    beta2 = float(group.get("betas", (.9, .999))[1])
    denominator = second / (1. - beta2 ** step) if step > 0 else second
    effective = float(group["lr"]) / (denominator.sqrt() + float(group.get("eps", 1e-8)))
    return {"available": True, "saved_step": step, "saved_lr": float(group["lr"]),
            "weight_decay": float(group.get("weight_decay", 0)),
            "first_moment": gradient_summary(parameter, first, snapshot["proxy_ids"]),
            "effective_coordinate_multiplier_quantiles": quantiles(effective.numpy().reshape(-1)),
            "semantics": "Existing moment geometry and bias-corrected denominator only. No optimizer step or counterfactual displacement.",
            "optimizer_updates": 0}


def _draw_summary(loss, stats, details, proxy_ids):
    pair = details["pair_proxy_idx"].detach().cpu().numpy()
    triple = details["triple_proxy_idx"].detach().cpu().numpy()
    valid = pair != triple
    active = (details["per_draw_loss"].detach().cpu().numpy() > 0) & valid
    domains = {}
    for name, mask in (("all_draws", np.ones(len(pair), dtype=bool)),
                       ("noncollision", valid), ("active_noncollision", active)):
        counts = np.bincount(np.concatenate((pair[mask], triple[mask])),
                             minlength=len(proxy_ids))
        probabilities = counts / max(int(counts.sum()), 1)
        positive = probabilities[probabilities > 0]
        domains[name] = {"selection_slots": int(counts.sum()),
                         "used_proxy_count": int(np.count_nonzero(counts)),
                         "effective_proxy_count": float(np.exp(-np.sum(positive * np.log(positive)))) if len(positive) else 0.,
                         "counts": counts.tolist()}
    return {"loss": float(loss.detach()), "stats": stats,
            "pair_proxy_ids": proxy_ids[pair].tolist(),
            "triple_proxy_ids": proxy_ids[triple].tolist(),
            "collision_mask": (~valid).tolist(), "active_hinge_mask": active.tolist(),
            "masked_hinges": details["masked_hinges"].detach().cpu().tolist(),
            "usage": domains}


def draw_agreement(first, second):
    """Query-aligned agreement; both-empty query components remain unavailable."""
    if len(first["pair_proxy_ids"]) != len(second["pair_proxy_ids"]):
        raise ValueError("Selection comparisons need the same query length")
    n = len(first["pair_proxy_ids"])
    if not n:
        return {"available": False, "reason": "No eligible fixed queries"}
    result = {"available": True, "query_count": n}
    for field in ("pair_proxy_ids", "triple_proxy_ids", "collision_mask", "active_hinge_mask"):
        result[field + "_agreement"] = float(np.mean(np.asarray(first[field]) == np.asarray(second[field])))
    first_set = set(first["pair_proxy_ids"] + first["triple_proxy_ids"])
    second_set = set(second["pair_proxy_ids"] + second["triple_proxy_ids"])
    result["selected_proxy_set_jaccard"] = len(first_set & second_set) / len(first_set | second_set) if first_set | second_set else None
    result["absolute_loss_change"] = abs(first["loss"] - second["loss"])
    return result


def _aggregate_noise(draws):
    keys = ("loss", "collision_rate", "active_fraction_all_draws", "active_fraction_noncollision")
    aggregate = {}
    for key in keys:
        values = [draw["loss"] if key == "loss" else draw["stats"][key] for draw in draws]
        aggregate[key] = {"mean": float(np.mean(values)), "std": float(np.std(values)),
                          "quantiles": quantiles(values)}
    pairs = [draw_agreement(draws[i], draws[j]) for i in range(len(draws))
             for j in range(i + 1, len(draws))]
    agreement = {}
    for key in ("pair_proxy_ids_agreement", "triple_proxy_ids_agreement",
                "collision_mask_agreement", "active_hinge_mask_agreement",
                "selected_proxy_set_jaccard"):
        values = [item[key] for item in pairs if item.get("available") and item.get(key) is not None]
        agreement[key] = {"mean": float(np.mean(values)) if values else None,
                          "pairs": len(values), "quantiles": quantiles(values)}
    return {"draws": len(draws), "scalar_variation": aggregate,
            "independent_noise_pair_agreement": agreement}


def audit_snapshot(snapshot, plan, noise_repeats=8, seed=22, gradient_repeats=1,
                   compare_source_operator=True):
    import torch
    _, _, distance, expmap, ghhc = _kernel()
    sample_rows = _remap(plan["sample_query_ids"], snapshot["sample_ids"])
    proxy_rows = _remap(plan["proxy_query_ids"], snapshot["proxy_ids"])
    # Only query endpoints need differentiable whole-to-proxy distances.
    whole_positions = np.unique(sample_rows.reshape(-1))
    compact_index = {int(position): index for index, position in enumerate(whole_positions)}
    compact_rows = np.asarray([[compact_index[int(position)] for position in row]
                               for row in sample_rows], dtype=np.int64).reshape(-1, 3)
    spec = snapshot["spec"]
    margin, tau = float(spec.get("margin", .1)), float(spec.get("tau", .1))
    weight = float(spec.get("lambda_hier", .5))
    if not np.isfinite(weight) or weight < 0:
        raise ValueError("Invalid explicit HIER gradient weight")
    initial_whole = torch.as_tensor(snapshot["whole"][whole_positions], dtype=torch.float32)
    initial_tangent = torch.as_tensor(snapshot["tangent"], dtype=torch.float32)
    before_whole, before_tangent = _tensor_digest(initial_whole), _tensor_digest(initial_tangent)
    source, source_identity = (reference_distance(spec.get("official_pmath"))
                               if compare_source_operator else (None, None))
    components = {}
    source_components = {}
    for name, triples in (("sample", compact_rows), ("proxy", proxy_rows)):
        triplets = torch.as_tensor(triples, dtype=torch.long)
        draws, gradients, source_draws = [], [], []
        with torch.no_grad():
            cached_proxies = expmap(initial_tangent)
            cached_distances = (distance(initial_whole, cached_proxies) if name == "sample"
                                else distance(cached_proxies))
            cached_source_distances = ((source(initial_whole, cached_proxies) if name == "sample"
                                        else source(cached_proxies, cached_proxies)) if source else None)
        for repeat in range(noise_repeats):
            whole = initial_whole.clone().requires_grad_(True)
            tangent = initial_tangent.clone().requires_grad_(True)
            proxies = expmap(tangent)
            distances = ((distance(whole, proxies) if name == "sample" else distance(proxies))
                         if repeat < gradient_repeats else cached_distances)
            generator = torch.Generator(device="cpu").manual_seed(seed + repeat * 1009 + (0 if name == "sample" else 1000003))
            loss, stats, details = ghhc(distances, triplets, tau=tau, margin=margin,
                                        generator=generator, return_details=True)
            if not bool(torch.isfinite(loss)):
                raise ValueError("Nonfinite production fixed-query loss")
            current = _draw_summary(loss, stats, details, snapshot["proxy_ids"])
            current.update(seed=generator.initial_seed(), repeat=repeat)
            draws.append(current)
            if repeat < gradient_repeats:
                proxy_gradient, whole_gradient = torch.autograd.grad(
                    loss, (tangent, whole), allow_unused=True)
                proxy_unweighted, whole_unweighted = proxy_gradient, whole_gradient
                proxy_gradient = None if proxy_gradient is None else weight * proxy_gradient
                whole_gradient = None if whole_gradient is None else weight * whole_gradient
                gradients.append({"repeat": repeat, "seed": generator.initial_seed(),
                                  "proxy_tangent": gradient_summary(tangent, proxy_gradient, snapshot["proxy_ids"]),
                                  "whole_ball": gradient_summary(whole, whole_gradient,
                                      snapshot["sample_ids"][whole_positions]),
                                  "unweighted_proxy_tangent": gradient_summary(tangent, proxy_unweighted, snapshot["proxy_ids"]),
                                  "unweighted_whole_ball": gradient_summary(whole, whole_unweighted,
                                      snapshot["sample_ids"][whole_positions]),
                                  "whole_dependency": name == "sample",
                                  "lambda_hier": weight,
                                  "semantics": "Frozen cached features/production ST autograd partials; not a backbone/base-gradient replay"})
            if source:
                with torch.no_grad():
                    source_distances = cached_source_distances
                    source_generator = torch.Generator(device="cpu").manual_seed(generator.initial_seed())
                    source_loss, source_stats, source_details = ghhc(
                        source_distances, triplets, tau=tau, margin=margin,
                        generator=source_generator, return_details=True)
                    source_current = _draw_summary(source_loss, source_stats, source_details, snapshot["proxy_ids"])
                    source_draws.append(source_current)
                    difference = (source_distances - distances.detach()).abs().cpu().numpy()
                if repeat == 0:
                    source_components[name] = {
                        "distance_absolute_difference_quantiles": quantiles(difference),
                        "fraction_distance_difference_gt_margin": float(np.mean(difference > margin)) if difference.size else None,
                        "same_query_same_noise_operator_agreements": []}
                source_components[name]["same_query_same_noise_operator_agreements"].append(
                    draw_agreement(current, source_current))
            if tangent.grad is not None or whole.grad is not None:
                raise RuntimeError("Read-only autograd.grad unexpectedly accumulated .grad")
        # Exact same seed/coordinates reproducibility is checked independently.
        with torch.no_grad():
            proxies = expmap(initial_tangent)
            distances = distance(initial_whole, proxies) if name == "sample" else distance(proxies)
            generator = torch.Generator().manual_seed(draws[0]["seed"])
            loss, stats, detail = ghhc(distances, triplets, tau=tau, margin=margin,
                                       generator=generator, return_details=True)
            repeated = _draw_summary(loss, stats, detail, snapshot["proxy_ids"])
            fixed = draw_agreement(draws[0], repeated)
            if fixed.get("available") and any(fixed[key] != 1 for key in fixed
                    if key.endswith("_agreement")):
                raise RuntimeError("Fixed-Gumbel reproducibility failed")
        components[name] = {"query_count": len(triples), "queries_sha256": plan["query_plan_sha256"],
                            "fixed_seed_reproducibility": fixed,
                            "noise_summary": _aggregate_noise(draws),
                            "noise_draws": draws, "gradient_repeats": gradients}
        if source:
            source_components[name]["reference_noise_summary"] = _aggregate_noise(source_draws)
    if _tensor_digest(initial_whole) != before_whole or _tensor_digest(initial_tangent) != before_tangent:
        raise RuntimeError("Frozen parameter/cache copy changed")
    return {"epoch": snapshot["epoch"], "run_key": snapshot["run_key"],
            "version": snapshot["version"], "identity": snapshot["identity"],
            "production_operator": PRODUCTION_OPERATOR,
            "margin_distance_units": margin, "gumbel_distance_scale_tau": tau,
            "lambda_hier": weight, "noise_plan": {name: [draw["seed"] for draw in component["noise_draws"]]
                for name, component in components.items()}, "components": components,
            "reference_operator": source_identity, "reference_counterfactual": source_components,
            "adam_state": adam_state_summary(snapshot),
            "read_only": {"optimizer_updates": 0, "cache_copy_unchanged": True,
                          "grad_buffers_unchanged": True, "GPU_forwards": 0,
                          "new_test_forwards": 0}}


def analyze_mechanisms(specs, output_dir, query_count=64, noise_repeats=8,
                       seed=22, gradient_repeats=1, compare_source_operator=True,
                       max_seconds=1800, sample_batch_size=64):
    """Create a fresh, CPU-only report; identities gate every temporal pairing."""
    import torch
    if query_count < 1 or noise_repeats < 2 or not 0 <= gradient_repeats <= noise_repeats or max_seconds <= 0 or sample_batch_size < 3:
        raise ValueError("Invalid bounded mechanism audit budget")
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError("Mechanism audits require a fresh output directory")
    for spec in specs:
        source = Path(spec["cache"]).resolve()
        if source == output or output in source.parents:
            raise ValueError("Source caches must not be nested within audit output")
        if spec.get("optimizer_checkpoint") or spec.get("checkpoint"):
            checkpoint = Path(spec.get("optimizer_checkpoint") or spec["checkpoint"]).resolve()
            if checkpoint == output or output in checkpoint.parents:
                raise ValueError("Source checkpoints must not be nested within audit output")
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    original_threads = torch.get_num_threads()
    torch.set_num_threads(min(original_threads, 2))
    manifest = {"format": "hier-fixed-query-mechanism-v1", "status": "running",
                "scope": "CPU fixed clean-cache production-kernel diagnostic, not a training intervention",
                "budget": {"query_count": query_count, "noise_repeats": noise_repeats,
                           "gradient_repeats": gradient_repeats, "max_seconds": max_seconds,
                           "sample_batch_size": sample_batch_size},
                "optimizer_updates": 0, "GPU_forwards": 0, "new_test_forwards": 0}
    _write(output / "mechanism_manifest.json", manifest)
    try:
        snapshots = sorted([_load(spec) for spec in specs], key=lambda value: (value["run_key"], value["epoch"]))
        plans, results, warnings, transitions = {}, [], [], []
        for snapshot in snapshots:
            if time.monotonic() - started > max_seconds:
                warnings.append("Budget reached before the next snapshot; later checkpoints were not evaluated")
                manifest["status"] = "partial_budget"
                break
            run_key = snapshot["run_key"]
            if run_key not in plans:
                plans[run_key] = make_query_plan(snapshot, query_count=query_count, seed=seed, sample_batch_size=sample_batch_size)
            plan = plans[run_key]
            incompatible = _compatibility(plan, snapshot)
            if incompatible:
                warnings.append(run_key + " e" + str(snapshot["epoch"]) + ": " + incompatible)
                # A local snapshot still receives a self-contained plan; it is
                # never paired with an incompatible earlier checkpoint.
                plan = make_query_plan(snapshot, query_count=query_count, seed=seed, sample_batch_size=sample_batch_size)
            result = audit_snapshot(snapshot, plan, noise_repeats=noise_repeats,
                                    seed=seed, gradient_repeats=gradient_repeats,
                                    compare_source_operator=compare_source_operator)
            result["query_plan_sha256"] = plan["query_plan_sha256"]
            result["temporal_identity_available"] = incompatible is None
            if incompatible:
                result["local_query_plan"] = plan
            name = "snapshot_" + str(len(results)).zfill(3) + "_e" + str(snapshot["epoch"]) + ".json"
            _write(output / name, result)
            result["detail_file"] = name
            previous = next((value for value in reversed(results) if value["run_key"] == run_key), None)
            if previous:
                transition = {"run_key": run_key, "from_epoch": previous["epoch"],
                              "to_epoch": result["epoch"], "epoch_gap": result["epoch"] - previous["epoch"],
                              "available": incompatible is None and previous["temporal_identity_available"]}
                if transition["available"]:
                    transition["same_query_same_noise"] = {component: [
                        draw_agreement(first, second)
                        for first, second in zip(previous["components"][component]["noise_draws"],
                                                 result["components"][component]["noise_draws"])]
                        for component in ("sample", "proxy")}
                    transition["semantics"] = "Same query IDs/noise, coarse actual saved-checkpoint interval; no continuous retention claim"
                else:
                    transition["reason"] = incompatible or "Earlier snapshot lacked a temporal identity"
                transitions.append(transition)
            results.append(result)
        compact = []
        for result in results:
            compact.append({key: value for key, value in result.items()
                            if key not in ("components", "adam_state", "reference_counterfactual", "local_query_plan")})
            compact[-1]["components"] = {name: {key: value for key, value in component.items()
                if key not in ("noise_draws", "gradient_repeats")}
                for name, component in result["components"].items()}
            compact[-1]["gradient_summaries"] = {name: [{
                "repeat": item["repeat"], "lambda_hier": item["lambda_hier"],
                "proxy_tangent": {key: value for key, value in item["proxy_tangent"].items() if key != "per_id"},
                "whole_ball": {key: value for key, value in item["whole_ball"].items() if key != "per_id"},
                "unweighted_proxy_tangent": {key: value for key, value in item["unweighted_proxy_tangent"].items() if key != "per_id"},
                "unweighted_whole_ball": {key: value for key, value in item["unweighted_whole_ball"].items() if key != "per_id"},
                "whole_dependency": item["whole_dependency"]}
                for item in component["gradient_repeats"]]
                for name, component in result["components"].items()}
            compact[-1]["reference_counterfactual"] = result["reference_counterfactual"]
            compact[-1]["adam_state"] = {key: value for key, value in result["adam_state"].items()
                if key != "first_moment"}
            if result["adam_state"].get("available"):
                compact[-1]["adam_state"]["first_moment"] = {key: value
                    for key, value in result["adam_state"]["first_moment"].items() if key != "per_id"}
        summary = {"format": manifest["format"], "scope": manifest["scope"],
                   "snapshots": compact, "transitions": transitions, "warnings": warnings,
                   "limits": ["Frozen clean panel differs from actual augmented training query population.",
                              "Whole partials are ball-coordinate gradients, not model/shared-encoder gradients.",
                              "Zero selected/active/gradient are distinct concepts; ST can give unselected candidates a gradient.",
                              "Gradient signs do not measure AdamW displacement or long-training causality.",
                              "Reference operator comparisons are same-coordinate numerical counterfactuals.",
                              "Proxy identities and queries are never paired across different runs/seeds."]}
        _write(output / "query_plans.json", list(plans.values()))
        _write(output / "mechanism_summary.json", summary)
        manifest["status"] = manifest["status"] if manifest["status"] == "partial_budget" else "completed"
        manifest["elapsed_seconds"] = time.monotonic() - started
        manifest["completed_snapshots"] = len(results)
        manifest["production_source_sha256"] = {
            filename: _sha(REPO / filename) for filename in (
                "inter_hierarchy_MN40/hier_proxy_scratch_v5/relations.py",
                "inter_hierarchy_MN40/hier_proxy_scratch_v6/hier_loss.py")}
        _write(output / "mechanism_manifest.json", manifest)
        return summary
    except Exception as exc:
        manifest.update(status="failed", error=repr(exc),
                        elapsed_seconds=time.monotonic() - started)
        _write(output / "mechanism_manifest.json", manifest)
        raise
    finally:
        torch.set_num_threads(original_threads)
