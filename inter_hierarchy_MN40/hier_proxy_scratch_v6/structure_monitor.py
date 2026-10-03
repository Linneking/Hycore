"""Detached V6 ancestor observations and RNG/BN-preserving view probes.

Training reports reuse the actual hard-Gumbel choices. The independent
augmentation probe uses deterministic max-cost argmin, explicitly a diagnostic
selection rather than a replay of training choices. No loss is introduced.
"""
from __future__ import annotations

from contextlib import contextmanager
import math
import random

import numpy as np
import torch

from ..hier_proxy_scratch_v5.relations import mine_sample_triplets, poincare_distance


@torch.no_grad()
def depth(points):
    radius = points.detach().norm(dim=-1).clamp(max=1 - torch.finfo(points.dtype).eps)
    return 2 * torch.atanh(radius)


def _moments(values):
    values = values.detach().double().reshape(-1)
    count = values.numel()
    if not count:
        return {"count": 0, "sum": 0., "sum_squares": 0., "min": None,
                "max": None, "positive_count": 0, "negative_count": 0}
    return {"count": count, "sum": float(values.sum()),
            "sum_squares": float(values.square().sum()), "min": float(values.min()),
            "max": float(values.max()), "positive_count": int((values > 0).sum()),
            "negative_count": int((values < 0).sum())}


def _distribution(moment):
    count = moment["count"]
    mean = moment["sum"] / count if count else None
    variance = max(0., moment["sum_squares"] / count - mean ** 2) if count else None
    return {**moment, "mean": mean, "std": math.sqrt(variance) if count else None,
            "positive_fraction": moment["positive_count"] / count if count else None,
            "negative_fraction": moment["negative_count"] / count if count else None}


@torch.no_grad()
def ancestor_summary(points, proxies, triplets, details):
    """Mergeable moments/counts; positive endpoint gap means ancestor is shallower."""
    # Transfer detached arrays in bulk. Per-metric Python scalar extraction
    # would introduce dozens of GPU synchronizations in every training step.
    triplets = triplets.detach().cpu().reshape(-1, 3)
    i, j, k = triplets.unbind(-1)
    pair, triple = details["pair_proxy_idx"].detach().cpu(), details["triple_proxy_idx"].detach().cpu()
    valid = pair != triple
    active = valid & (details["per_draw_loss"].detach().cpu() > 0)
    proxy_depth, point_depth = depth(proxies.detach().cpu()), depth(points.detach().cpu())
    pair_depth, triple_depth = proxy_depth[pair], proxy_depth[triple]
    gap = pair_depth - triple_depth
    quantities = {"pair_depth": pair_depth, "triple_depth": triple_depth,
                  "pair_minus_triple_depth": gap,
                  "pair_i_endpoint_gap": point_depth[i] - pair_depth,
                  "pair_j_endpoint_gap": point_depth[j] - pair_depth,
                  "triple_i_endpoint_gap": point_depth[i] - triple_depth,
                  "triple_j_endpoint_gap": point_depth[j] - triple_depth,
                  "triple_k_endpoint_gap": point_depth[k] - triple_depth,
                  "pair_min_endpoint_gap": torch.minimum(point_depth[i], point_depth[j]) - pair_depth,
                  "triple_min_endpoint_gap": torch.minimum(torch.minimum(point_depth[i], point_depth[j]), point_depth[k]) - triple_depth}
    moments, counts = {}, {}
    for domain, mask in (("all_draws", torch.ones_like(valid)),
                         ("noncollision", valid), ("active_noncollision", active)):
        for name, values in quantities.items():
            moments[f"{domain}/{name}"] = _moments(values[mask])
        for role, selected in (("pair", pair), ("triple", triple)):
            counts[f"{domain}/{role}"] = torch.bincount(selected[mask], minlength=len(proxies)).cpu().tolist()
    return {"selection": "actual independent hard-Gumbel pair/triple selections",
            "draw_count": len(triplets), "noncollision_count": int(valid.sum()),
            "active_noncollision_count": int(active.sum()), "self_k_count": int((i == k).sum()),
            "proxy_count": len(proxies), "moments": moments, "selected_proxy_counts": counts}


class StructureAccumulator:
    """Pool actual draw counts, rather than averaging per-step fractions."""
    def __init__(self):
        self.reports = {}

    def update(self, structure):
        for component, current in structure.items():
            if component not in self.reports:
                self.reports[component] = {
                    "selection": current["selection"], "steps": 0,
                    "proxy_count": current["proxy_count"], "draw_count": 0,
                    "noncollision_count": 0, "active_noncollision_count": 0,
                    "self_k_count": 0, "moments": {}, "selected_proxy_counts": {}}
            accumulated = self.reports[component]
            if accumulated["proxy_count"] != current["proxy_count"]:
                raise ValueError("proxy count changed within an epoch")
            accumulated["steps"] += 1
            for key in ("draw_count", "noncollision_count", "active_noncollision_count", "self_k_count"):
                accumulated[key] += current[key]
            for key, moment in current["moments"].items():
                target = accumulated["moments"].setdefault(key, _moments(torch.empty(0)))
                for field in ("count", "sum", "sum_squares", "positive_count", "negative_count"):
                    target[field] += moment[field]
                if moment["count"]:
                    target["min"] = moment["min"] if target["min"] is None else min(target["min"], moment["min"])
                    target["max"] = moment["max"] if target["max"] is None else max(target["max"], moment["max"])
            for key, counts in current["selected_proxy_counts"].items():
                target = accumulated["selected_proxy_counts"].setdefault(key, [0] * len(counts))
                for position, count in enumerate(counts):
                    target[position] += count

    def summary(self):
        result = {}
        for component, report in self.reports.items():
            usages = {}
            for key, counts in report["selected_proxy_counts"].items():
                total = sum(counts)
                entropy = -sum((n / total) * math.log(n / total) for n in counts if n) if total else None
                usages[key] = {"counts": list(counts), "selection_count": total,
                               "used_proxy_count": sum(n > 0 for n in counts),
                               "used_proxy_fraction": sum(n > 0 for n in counts) / len(counts),
                               "entropy": entropy, "effective_proxy_count": math.exp(entropy) if total else None}
            result[component] = {**{k: v for k, v in report.items() if k not in ("moments", "selected_proxy_counts")},
                                 "depth_distributions": {key: _distribution(value) for key, value in report["moments"].items()},
                                 "selected_ancestor_usage": usages}
        return result


@torch.no_grad()
def radial_angular_gradient(tangent, gradient):
    """Euclidean tangent-parameter gradient, not an optimizer update."""
    tangent, gradient = tangent.detach(), gradient.detach()
    direction = tangent / tangent.norm(dim=-1, keepdim=True).clamp_min(1e-15)
    signed = (gradient * direction).sum(-1)
    radial = signed[:, None] * direction
    angular = gradient - radial
    return {"gradient_norm": float(gradient.norm()), "radial_norm": float(radial.norm()),
            "angular_norm": float(angular.norm()), "signed_radial_mean": float(signed.mean()),
            "positive_radial_gradient_fraction": float((signed > 0).float().mean()),
            "negative_radial_gradient_fraction": float((signed < 0).float().mean()),
            "nonzero_proxy_gradient_count": int((gradient.norm(dim=-1) > 0).sum()),
            "semantics": "Positive radial gradient points outward; gradient descent subtracts it. AdamW updates are reported separately."}


@torch.no_grad()
def proxy_update_stats(before_tangent, after_tangent):
    before, after = before_tangent.detach(), after_tangent.detach()
    change = after - before
    direction = before / before.norm(dim=-1, keepdim=True).clamp_min(1e-15)
    signed = (change * direction).sum(-1)
    radial = signed[:, None] * direction
    angular = change - radial
    before_norm, after_norm = before.norm(dim=-1), after.norm(dim=-1)
    return {"update_norm": float(change.norm()), "radial_update_norm": float(radial.norm()),
            "angular_update_norm": float(angular.norm()), "signed_radial_update_mean": float(signed.mean()),
            "tangent_norm_change_mean": float((after_norm - before_norm).mean()),
            "outward_update_fraction": float((signed > 0).float().mean()),
            "inward_update_fraction": float((signed < 0).float().mean()),
            "raw_ball_radius_change_mean": float((after_norm.tanh() - before_norm.tanh()).mean()),
            "semantics": "Actual optimizer displacement decomposed relative to the pre-step tangent direction; raw radius precedes .999 safety projection."}


@contextmanager
def preserved_eval(module):
    """Even failed probes restore every mode, buffer, and global RNG stream."""
    modes = [(item, item.training) for item in module.modules()]
    buffers = [(buffer, buffer.detach().clone()) for buffer in module.buffers()]
    python_rng, numpy_rng, cpu_rng = random.getstate(), np.random.get_state(), torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    try:
        module.eval()
        yield
    finally:
        with torch.no_grad():
            for buffer, saved in buffers:
                buffer.copy_(saved)
        for item, training in modes:
            item.training = training
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)


@torch.no_grad()
def feature_snapshot(mu, proxies, labels, ids, *, topk=20, seed=22, triplets=None):
    """Fixed checkpoint/position order; deterministic selections consume no RNG."""
    mining = mine_sample_triplets(mu, labels, topk=min(topk, len(mu)), seed=seed,
                                  data_ids=ids, exclude_self_negative=True)
    triplets = mining["triplets"] if triplets is None else triplets.to(mu.device)
    distances = poincare_distance(mu, proxies)
    nearest = distances.topk(min(4, len(proxies)), largest=False).indices
    i, j, k = triplets.unbind(-1)
    pair_cost = torch.maximum(distances[i], distances[j])
    triple_cost = torch.maximum(pair_cost, distances[k])
    return {"ids": torch.as_tensor(ids).detach().cpu(), "nearest_proxy": nearest.cpu(),
            "mutual": mining["mutual"].cpu(), "triplets": triplets.cpu(),
            "pair_proxy": pair_cost.argmin(-1).cpu(), "triple_proxy": triple_cost.argmin(-1).cpu(),
            "mining": mining["stats"]}


@torch.no_grad()
def compare_feature_snapshots(first, second):
    if not torch.equal(first["ids"], second["ids"]):
        raise ValueError("view probes must retain identical ID/position order")
    a, b = first["nearest_proxy"], second["nearest_proxy"]
    intersections = (a[:, :, None] == b[:, None, :]).any(-1).sum(-1).double()
    edge_a, edge_b = first["mutual"], second["mutual"]
    union = int((edge_a | edge_b).sum())
    report = {"positions": len(a), "unique_ids": len(torch.unique(first["ids"])),
              "nearest_proxy_retention": float((a[:, 0] == b[:, 0]).float().mean()) if len(a) else None,
              "nearest_four_proxy_jaccard_mean": float((intersections / (a.shape[1] + b.shape[1] - intersections)).mean()) if len(a) else None,
              "sample_mutual_graph_jaccard": int((edge_a & edge_b).sum()) / union if union else None,
              "shared_triplet_draws": len(first["triplets"]),
              "selection": "deterministic pair/triple minimax-cost argmin, not training hard Gumbel",
              "triplet_domain": "Both views use the first view's fixed draws; second view eligibility need not match."}
    if not torch.equal(first["triplets"], second["triplets"]):
        raise ValueError("ancestor comparison requires identical ordered triplet draws")
    for role in ("pair", "triple"):
        x, y = first[f"{role}_proxy"], second[f"{role}_proxy"]
        report[f"{role}_ancestor_retention"] = float((x == y).float().mean()) if len(x) else None
    for view, snapshot in (("first", first), ("second", second)):
        report[f"{view}_deterministic_collision_fraction"] = float((snapshot["pair_proxy"] == snapshot["triple_proxy"]).float().mean()) if len(snapshot["triplets"]) else None
        report[f"{view}_sample_anchor_fraction"] = snapshot["mining"]["anchor_coverage"]
    return report


@torch.no_grad()
def run_augmentation_probe(backbone, proxy, points, labels, ids, device, seed, batch_size=32):
    """Rank0-only direct backbone eval on two independent source-style views.

    Points/labels are full arrays and ids are their global indices. Whole inputs
    have1024 points; no part overwrite or BN update is performed in this probe.
    Local NumPy generators implement the original scale/shift/shuffle formulas.
    """
    ids_np = torch.as_tensor(ids).detach().cpu().numpy().astype(np.int64)
    if len(ids_np) < 3 or batch_size < 1:
        raise ValueError("probe requires at least three positions and positive batch size")
    features = []
    with preserved_eval(backbone):
        for view in range(2):
            rng = np.random.default_rng(seed + 1000003 * view)
            augmented = []
            for global_id in ids_np:
                cloud = np.asarray(points[int(global_id), :1024]).copy()
                cloud = (cloud * rng.uniform(2 / 3, 3 / 2, size=3) + rng.uniform(-.2, .2, size=3)).astype(np.float32)
                rng.shuffle(cloud)
                augmented.append(cloud)
            batches = []
            for start in range(0, len(augmented), batch_size):
                cloud = torch.from_numpy(np.stack(augmented[start:start + batch_size])).to(device)
                mu, _ = backbone(cloud.transpose(1, 2).contiguous())
                batches.append(mu.detach())
            features.append(torch.cat(batches))
        proxies = proxy.proxies().detach()
        gold = torch.as_tensor(np.asarray(labels)[ids_np], device=device).reshape(-1)
        first = feature_snapshot(features[0], proxies, gold, ids_np, seed=seed)
        second = feature_snapshot(features[1], proxies, gold, ids_np, seed=seed,
                                  triplets=first["triplets"])
        report = compare_feature_snapshots(first, second)
    return {**report, "seed": seed, "topk_including_self": min(20, len(ids_np)),
            "input": "two independent scale/shift/shuffle whole1024 views; fixed checkpoint eval",
            "BN_RNG_training_modes_restored": True, "optimizer_steps": 0}
