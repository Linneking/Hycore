"""Frozen embedding readout for the user-approved, simplified PartHier audit.

This module never constructs an encoder, optimizer, new queue, or training
experiment. Production distance/mining/selection operators are reused. Saved
queue snapshots remain static; all class centers use the full clean cache.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from inter_hierarchy_MN40.parthier_v1.loss import origin_depth, _selection_losses
from inter_hierarchy_MN40.hier_proxy_scratch_v5.relations import mine_sample_triplets
from inter_hierarchy_MN40.hier_proxy_scratch_v5.sampler import SourceClassBatchSampler
from inter_hierarchy_MN40.hier_proxy_scratch_v8.geometry import official_distance


LABELS = ("TS18_best", "TS18_e300", "TS02_best", "TS02_e300")
SCALES = (200, 400, 600)
TAU = 1e-3
ANGLE_EPS = 1e-6
QUANTILES = (("p1", .01), ("p5", .05), ("p25", .25), ("median", .5),
             ("p75", .75), ("p90", .9), ("p95", .95), ("p99", .99))


def array(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def describe(value):
    x = array(value).astype(np.float64, copy=False).reshape(-1)
    x = x[np.isfinite(x)]
    out = {"count": int(len(x))}
    if not len(x):
        return {**out, **{k: None for k in ("mean", "min", "max", *[q[0] for q in QUANTILES])}}
    out.update(mean=float(x.mean()), min=float(x.min()), max=float(x.max()))
    out.update({k: float(np.quantile(x, q)) for k, q in QUANTILES})
    return out


def gap_summary(value, margin):
    raw = array(value)
    effective_margin = float(np.asarray(margin, dtype=raw.dtype)) if np.issubdtype(raw.dtype, np.floating) else float(margin)
    x = raw.astype(np.float64, copy=False).reshape(-1)
    if not np.isfinite(x).all():
        raise ValueError("Nonfinite gap must not disappear from the denominator")
    out = describe(x)
    out.update(margin=float(margin), comparison_margin=effective_margin,
               basic_fraction=float((x > 0).mean()) if len(x) else None,
               margin_fraction=float((x >= effective_margin).mean()) if len(x) else None,
               mean_hinge=float(np.maximum(effective_margin - x, 0).mean()) if len(x) else None,
               near_zero_fraction=float((np.abs(x) < TAU).mean()) if len(x) else None,
               near_margin_fraction=float((np.abs(x - effective_margin) < TAU).mean()) if len(x) else None)
    return out


def joint_fraction(gaps, margin):
    x = array(gaps)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError("Joint conditions require finite [rows,conditions]")
    if not len(x):
        return {"count": 0, "joint_basic_fraction": None, "joint_margin_fraction": None}
    return {"count": len(x), "joint_basic_fraction": float((x > 0).all(1).mean()),
            "joint_margin_fraction": float((x >= margin).all(1).mean())}


def class_centers(clean, labels):
    """Full-cache spherical direction means; not hyperbolic points."""
    z = torch.as_tensor(clean, dtype=torch.float64)
    y = torch.as_tensor(labels, dtype=torch.long, device=z.device)
    norms = z.norm(dim=-1)
    if bool((norms <= ANGLE_EPS).any()):
        raise ValueError("A clean whole has undefined direction; inspect before building centers")
    unit = z / norms[:, None]
    classes = torch.unique(y, sorted=True)
    means = torch.stack([unit[y == c].mean(0) for c in classes])
    lengths = means.norm(dim=-1)
    centers = means / lengths.clamp_min(ANGLE_EPS)[:, None]
    return centers, classes, lengths


def angle_measurements(vectors, centers, center_labels, descendants):
    """Angles in degrees; repeated descendant classes are counted once pooled.

    Per-position i/j/k angles are also returned separately. Invalid directions
    are NaN with an explicit valid-row mask, not manufactured zero angles.
    """
    z = torch.as_tensor(vectors).to(dtype=torch.float64)
    centers = torch.as_tensor(centers, dtype=torch.float64, device=z.device)
    center_labels = torch.as_tensor(center_labels, dtype=torch.long, device=z.device)
    ds = torch.as_tensor(descendants, dtype=torch.long, device=z.device)
    if ds.ndim != 2 or len(ds) != len(z):
        raise ValueError("Descendant labels must be [rows,roles]")
    cn = centers.norm(dim=-1)
    valid_centers = cn > ANGLE_EPS
    norm = z.norm(dim=-1)
    valid = norm > ANGLE_EPS
    dots = (z / norm.clamp_min(ANGLE_EPS)[:, None]) @ centers.T
    angles = torch.rad2deg(torch.acos(dots.clamp(-1, 1)))
    angles[:, ~valid_centers] = float("inf")
    if int(valid_centers.sum()) < 2:
        raise ValueError("At least two defined class centers are required")
    nearest, positions = angles.topk(2, dim=1, largest=False)
    nearest_labels = center_labels[positions[:, 0]]
    hit = (nearest_labels[:, None] == ds).any(1)
    desc_pos = torch.searchsorted(center_labels, ds)
    if bool((desc_pos >= len(center_labels)).any()) or not torch.equal(center_labels[desc_pos], ds):
        raise ValueError("Unknown descendant class")
    desc_angles = angles.gather(1, desc_pos)
    pooled = desc_angles.clone()
    for col in range(1, ds.shape[1]):
        duplicates = (ds[:, :col] == ds[:, col:col + 1]).any(1)
        pooled[duplicates, col] = float("nan")
    nearest[~valid] = float("nan")
    desc_angles[~valid] = float("nan")
    pooled[~valid] = float("nan")
    result = {"valid": array(valid), "nearest": array(nearest[:, 0]),
              "second_gap": array(nearest[:, 1] - nearest[:, 0]),
              "nearest_in_descendants": array(hit & valid),
              "descendant_angles": array(pooled)}
    for col, name in enumerate(("descendant_i", "descendant_j", "descendant_k")[:ds.shape[1]]):
        result[name] = array(desc_angles[:, col])
    return result


def validate_cache(data, expected_count):
    keys = ("ids", "labels", "clean", "whole", "parts")
    if any(k not in data for k in keys):
        raise ValueError("Cache missing required arrays")
    ids = np.asarray(data["ids"]).reshape(-1)
    labels = np.asarray(data["labels"]).reshape(-1)
    order = np.argsort(ids)
    if len(ids) != expected_count or not np.array_equal(ids[order], np.arange(expected_count)):
        raise ValueError("Canonical cache IDs must be exactly 0..N-1")
    d = np.asarray(data["whole"]).shape[-1]
    shapes = {"clean": (expected_count, d), "whole": (expected_count, d),
              "parts": (expected_count, 3, d)}
    result = {"ids": ids[order].astype(np.int64), "labels": labels[order].astype(np.int64)}
    for key, shape in shapes.items():
        x = np.asarray(data[key])
        if x.shape != shape or x.dtype != np.float32:
            raise ValueError(f"Invalid {key} shape/dtype: {x.shape}/{x.dtype}")
        norms = np.linalg.norm(x, axis=-1)
        if not np.isfinite(x).all() or not (norms < 1).all():
            raise ValueError(f"Nonfinite or ball-exterior embeddings in {key}")
        result[key] = np.ascontiguousarray(x[order])
    return result


def write_json(path, data):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def write_csv(path, rows, fields=None):
    rows = list(rows)
    if fields is None:
        fields = list(rows[0]) if rows else []
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def histogram_rows(model, values, bins, **extra):
    x = array(values).astype(np.float64).reshape(-1)
    x = x[np.isfinite(x)]
    counts, _ = np.histogram(x, bins=bins)
    if int(counts.sum()) != len(x):
        raise ValueError("Histogram range silently omitted observations")
    return [{"model": model, **extra, "bin_left": float(a), "bin_right": float(b),
             "count": int(n), "percentage": 100 * int(n) / len(x) if len(x) else 0.}
            for a, b, n in zip(bins[:-1], bins[1:], counts)]


def _concat(pieces, shape=None):
    return np.concatenate(pieces, axis=0) if pieces else np.empty(shape or (0,), dtype=np.float32)


class RelationAccumulator:
    def __init__(self):
        self.draws = self.collisions = self.same_ij_draws = self.eligible_anchors = 0
        self.native_hier_sum = self.native_ord_sum = 0.
        self.ord = []
        self.hier = []
        self.same = []
        self.roles = {r: {"scales": [], "current": [], "self": [], "depth": [],
                          "queue_indices": set(), "live_occurrences": set(), "source_object_scales": set()}
                      for r in ("A", "B")}
        self.angles = {g: {"sampled_count": 0, "valid": [], "nearest": [], "second_gap": [],
                           "nearest_in_descendants": [], "descendant_angles": [],
                           "descendant_i": [], "descendant_j": [], "descendant_k": []}
                       for g in ("A_same", "A_cross", "B2", "B3")}

    def finish(self):
        og = _concat(self.ord, (0, 5))
        hg = _concat(self.hier, (0, 3))
        count = len(og)
        summary = {"draws": self.draws, "valid_draws": count, "collisions": self.collisions,
                   "collision_fraction": self.collisions / self.draws if self.draws else None,
                   "same_ij_fraction": self.same_ij_draws / self.draws if self.draws else None,
                   "eligible_anchor_presentations": self.eligible_anchors,
                   "native_hier_all_draw_mean": self.native_hier_sum / self.draws if self.draws else None,
                   "native_ord_all_draw_mean": self.native_ord_sum / self.draws if self.draws else None,
                   "ord": {**joint_fraction(og[:, [2, 3]], .2), "edges": {
                       name: gap_summary(og[:, col], .2) for col, name in enumerate(("A_i", "A_j", "A_minij", "B_A", "B_k"))}},
                   "hier": {**joint_fraction(hg, .1), "edges": {
                       name: gap_summary(hg[:, col], .1) for col, name in enumerate(("i", "j", "k"))}},
                   "radial_and_distance_margin_fraction": float(((og[:, [2, 3]] >= .2).all(1) & (hg >= .1).all(1)).mean()) if count else None,
                   "roles": {}, "angles": {}}
        for role, items in self.roles.items():
            scale = _concat(items["scales"]).astype(np.int64)
            n = len(scale)
            counts = np.bincount(scale, minlength=3).tolist()
            summary["roles"][role] = {
                "count": n, "scale_counts": counts,
                "scale_fractions": [v / n if n else None for v in counts],
                "current_fraction": float(_concat(items["current"]).mean()) if n else None,
                "queue_fraction": float(1 - _concat(items["current"]).mean()) if n else None,
                "self_source_fraction": float(_concat(items["self"]).mean()) if n else None,
                "unique_queue_entries": len(items["queue_indices"]),
                "unique_live_occurrences": len(items["live_occurrences"]),
                "unique_source_object_scales": len(items["source_object_scales"]),
                "depth": describe(_concat(items["depth"]))}
        for group, items in self.angles.items():
            valid = _concat(items["valid"]).astype(bool)
            n = int(valid.sum())
            summary["angles"][group] = {"sampled_count": items["sampled_count"], "count": n,
                "undefined_count": items["sampled_count"] - n,
                "nearest_in_descendant_fraction": float(_concat(items["nearest_in_descendants"])[valid].mean()) if n else None,
                **{key: describe(_concat(items[key])) for key in ("nearest", "second_gap", "descendant_angles", "descendant_i", "descendant_j", "descendant_k")}}
        return summary


@torch.no_grad()
def accumulate_relations(acc, panel, mu, parts, labels, ids, queue, centers, center_labels, seed):
    current = parts.reshape(-1, parts.shape[-1])
    candidate = torch.cat((current, queue["features"]))
    candidate_ids = torch.cat((ids.repeat_interleave(3), queue["object_ids"]))
    scales = torch.cat((torch.arange(3, device=mu.device).repeat(len(mu)), queue["scales"]))
    mined = mine_sample_triplets(mu, labels, topk=20, t_per_anchor=50, seed=seed,
                                 exclude_self_negative=True, data_ids=ids)
    trips = mined["triplets"]
    acc.eligible_anchors += mined["stats"]["eligible_anchors"]
    acc.draws += len(trips)
    if not len(trips):
        return
    hd, cd = origin_depth(mu), origin_depth(candidate)
    distance = official_distance(mu, candidate, c=1.)
    gen = torch.Generator(device=mu.device).manual_seed(seed)
    hloss, oloss, detail = _selection_losses(distance, trips, hd, cd, generator=gen)
    a, b, valid = detail["pair_idx"], detail["triple_idx"], detail["valid"]
    i, j, k = trips.unbind(-1)
    same = labels[i] == labels[j]
    acc.same_ij_draws += int(same.sum())
    acc.collisions += int((~valid).sum())
    acc.native_hier_sum += float(hloss) * len(trips)
    acc.native_ord_sum += float(oloss) * len(trips)
    og = torch.stack((hd[i] - cd[a], hd[j] - cd[a], torch.minimum(hd[i], hd[j]) - cd[a],
                      cd[a] - cd[b], hd[k] - cd[b]), dim=1)
    hg = torch.stack((distance[i, b] - distance[i, a], distance[j, b] - distance[j, a],
                      distance[k, a] - distance[k, b]), dim=1)
    # Reconstructed gaps must reproduce production hinges and collision denominator.
    torch.testing.assert_close((torch.relu(.1 - hg).sum(1) * valid).mean(), hloss, rtol=1e-5, atol=3e-6)
    torch.testing.assert_close((torch.relu(.2 - og[:, [2, 3]]).sum(1) * valid).mean(), oloss, rtol=1e-5, atol=3e-6)
    acc.ord.append(array(og[valid]))
    acc.hier.append(array(hg[valid]))
    acc.same.append(array(same[valid]))
    for role, chosen, descendants, own in (
        ("A", a, labels[trips[:, :2]], (candidate_ids[a] == ids[i]) | (candidate_ids[a] == ids[j])),
        ("B", b, labels[trips], (candidate_ids[b] == ids[i]) | (candidate_ids[b] == ids[j]) | (candidate_ids[b] == ids[k]))):
        selected = chosen[valid]
        items = acc.roles[role]
        selected_cpu = array(selected).astype(np.int64)
        selected_scales = array(scales[selected]).astype(np.int64)
        selected_ids = array(candidate_ids[selected]).astype(np.int64)
        items["scales"].append(selected_scales)
        items["current"].append(array(selected < len(current)))
        items["self"].append(array(own[valid]))
        items["depth"].append(array(cd[selected]))
        items["queue_indices"].update(int(t - len(current)) for t in selected_cpu if t >= len(current))
        items["live_occurrences"].update((int(panel), int(t)) for t in selected_cpu if t < len(current))
        items["source_object_scales"].update(zip(selected_ids.tolist(), selected_scales.tolist()))
        ds = descendants[valid]
        measures = angle_measurements(candidate[selected], centers, center_labels, ds)
        if role == "A":
            groups = {"A_same": array(same[valid]), "A_cross": array(~same[valid])}
        else:
            sorted_labels = ds.sort(1).values
            category_count = 1 + (sorted_labels[:, 1:] != sorted_labels[:, :-1]).sum(1)
            if bool(((category_count < 2) | (category_count > 3)).any()):
                raise ValueError("Unexpected triple class count for two-instance sampler")
            groups = {"B2": array(category_count == 2), "B3": array(category_count == 3)}
        for group, mask in groups.items():
            item = acc.angles[group]
            item["sampled_count"] += int(mask.sum())
            for name, data in measures.items():
                if name not in item:
                    continue
                data = data[mask]
                if name == "descendant_angles":
                    data = data.reshape(-1)
                item[name].append(data)


def load_queue(path, device, expected_count):
    saved = torch.load(path, map_location="cpu", weights_only=False)
    queue = saved.get("queue", saved)
    for key in ("features", "object_ids", "scales", "steps"):
        if key not in queue:
            raise ValueError(f"Queue is missing {key}")
    if queue["features"].shape[0] != 2048 or queue["features"].dtype != torch.float32:
        raise ValueError("Expected the model's own full FP32 Q2048")
    ids, scales = queue["object_ids"], queue["scales"]
    if ids.shape != (2048,) or scales.shape != (2048,):
        raise ValueError("Queue metadata shape mismatch")
    if bool(((ids < 0) | (ids >= expected_count)).any()) or bool(((scales < 0) | (scales > 2)).any()):
        raise ValueError("Queue object IDs are not mapped to canonical IDs or scales invalid")
    z = queue["features"]
    if not bool(torch.isfinite(z).all()) or not bool((z.norm(dim=-1) < 1).all()):
        raise ValueError("Invalid saved queue features")
    step = saved.get("global_step")
    if step is not None and bool((queue["steps"] > step).any()):
        raise ValueError("Queue dates after its own checkpoint")
    return {k: v.to(device) for k, v in queue.items() if isinstance(v, torch.Tensor)}, {
        k: saved[k] for k in ("global_step", "completed_epochs", "source_sha256") if k in saved}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batches", type=int, default=200)
    parser.add_argument("--seed", type=int, default=4780)
    parser.add_argument("--expected-count", type=int, default=9840)
    args = parser.parse_args()
    if args.batches < 1:
        parser.error("At least one audit batch is required")
    # Native intra depends on Geoopt. Do not silently substitute another distance.
    from geoopt import PoincareBall
    from inter_hierarchy_MN40.hier_proxy_scratch_v9.base_protocol import intra_losses
    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    device = torch.device(args.device)
    args.out_dir.mkdir(parents=True, exist_ok=False)
    private = args.out_dir / "private"
    private.mkdir()
    caches, depths = {}, {}
    for label in LABELS:
        with np.load(args.cache_dir / f"{label}.npz", allow_pickle=False) as source:
            caches[label] = validate_cache(source, args.expected_count)
        cache = caches[label]
        if not np.array_equal(cache["labels"], caches[LABELS[0]]["labels"]):
            raise ValueError("Labels differ across model caches")
        views = {"clean": cache["clean"], "whole": cache["whole"],
                 **{f"p{g}": cache["parts"][:, k] for k, g in enumerate(SCALES)}}
        depths[label] = {view: array(origin_depth(torch.from_numpy(z))) for view, z in views.items()}
    max_depth = max(float(v.max()) for d in depths.values() for v in d.values())
    bins = np.arange(0, (math.ceil(max_depth / .1) + 1) * .1 + 1e-8, .1)
    depth_hist, depth_ecdf, depth_class, chain_rows, chain_class = [], [], [], [], []
    intra_rows, relation_rows, role_rows, angle_rows, angle_hist = [], [], [], [], []
    summary = {"version": "audit94-simplified-v1", "config": {
        "batches": args.batches, "seed": args.seed, "count": args.expected_count,
        "K": 20, "T": 50, "tau": .1, "chain_margin": .2, "ord_margin": .2, "hier_margin": .1,
        "queue": "checkpoint-own static historical train-mode snapshot; no enqueue",
        "class_centers": "full clean9840, unit directions averaged per class then normalized",
        "relation_success_denominator": "noncollision selected relations",
        "native_loss_denominator": "all sampled relations including zeroed collisions",
        "near_threshold_tolerance": TAU, "angle_undefined_norm": ANGLE_EPS,
        "device": str(device)}, "models": {}}
    for name in ("input_identity.json", "checkpoints.json"):
        p = args.cache_dir / name
        if p.exists():
            summary[name[:-5]] = json.loads(p.read_text(encoding="utf-8"))
    for label in LABELS:
        started = time.perf_counter()
        print(f"MODEL {label} start", flush=True)
        cache, dh = caches[label], depths[label]
        labels_np = cache["labels"]
        labels = torch.from_numpy(labels_np).to(device)
        whole = torch.from_numpy(cache["whole"]).to(device)
        parts = torch.from_numpy(cache["parts"]).to(device)
        centers, center_ids, center_lengths = class_centers(cache["clean"], labels_np)
        centers, center_ids = centers.to(device), center_ids.to(device)
        model = {"depth": {}, "class_center_mean_lengths": describe(center_lengths)}
        for view, data in dh.items():
            model["depth"][view] = describe(data)
            vectors = cache[view] if view in ("clean", "whole") else cache["parts"][:, SCALES.index(int(view[1:]))]
            model["depth"][view]["rho_ge_0_995_fraction"] = float((np.linalg.norm(vectors, axis=-1) >= .995).mean())
            depth_hist.extend(histogram_rows(label, data, bins, view=view))
            unique, counts = np.unique(data, return_counts=True)
            depth_ecdf.extend({"model": label, "view": view, "depth": float(x), "percentage": float(p)}
                              for x, p in zip(unique, 100 * counts.cumsum() / len(data)))
            for c in np.unique(labels_np):
                depth_class.append({"model": label, "view": view, "class": int(c), **describe(data[labels_np == c])})
        cg = np.stack((dh["p400"] - dh["p200"], dh["p600"] - dh["p400"], dh["whole"] - dh["p600"]), axis=1)
        model["chain"] = {**joint_fraction(cg, .2), "edges": {}}
        for col, edge in enumerate(("200_400", "400_600", "600_whole")):
            info = gap_summary(cg[:, col], .2)
            model["chain"]["edges"][edge] = info
            chain_rows.append({"model": label, "edge": edge, **info})
        chain_rows.append({"model": label, "edge": "joint", **{
            **describe(cg.min(1)), "margin": .2,
            "basic_fraction": model["chain"]["joint_basic_fraction"],
            "margin_fraction": model["chain"]["joint_margin_fraction"],
            "mean_hinge": float(np.maximum(.2 - cg, 0).sum(1).mean()),
            "near_zero_fraction": float((np.abs(cg) < TAU).any(1).mean()),
            "near_margin_fraction": float((np.abs(cg - .2) < TAU).any(1).mean())}})
        for c in np.unique(labels_np):
            mask = labels_np == c
            chain_class.append({"model": label, "class": int(c), **joint_fraction(cg[mask], .2)})
        ball = PoincareBall(c=1.).to(device=device, dtype=whole.dtype)
        radial = {g: array(ball.dist0(whole) - ball.dist0(parts[:, index])) for index, g in enumerate(SCALES)}
        contrasts = {g: [] for g in SCALES}
        sampler = SourceClassBatchSampler(labels_np, steps=args.batches, seed=args.seed)
        acc = RelationAccumulator() if label.startswith("TS18") else None
        queue = None
        if acc is not None:
            queue, qinfo = load_queue(args.cache_dir / f"{label}_queue.pt", device, args.expected_count)
            model["queue_snapshot"] = {**qinfo, "count": len(queue["features"]),
                "depth": describe(origin_depth(queue["features"]))}
        for panel in range(args.batches):
            ix = torch.tensor(sampler.global_plan(panel).indices, device=device, dtype=torch.long)
            mu, pp, yy = whole[ix], parts[ix], labels[ix]
            if bool((yy == yy.flip(0)).any()):
                raise ValueError("Audit flip-negative labels collide")
            for col, g in enumerate(SCALES):
                positive, negative = ball.dist(mu, pp[:, col]), ball.dist(mu, pp[:, col].flip(0))
                contrasts[g].append(array(negative - positive))
                if panel == 0:
                    native = intra_losses(mu, pp[:, col], g)
                    torch.testing.assert_close(torch.maximum(positive - negative + 4, torch.zeros_like(positive)).mean(),
                                               native["intra_contrastive"], rtol=1e-6, atol=1e-6)
            if acc is not None:
                seed = args.seed + (panel + 1) * 1009 + 70000027
                accumulate_relations(acc, panel, mu, pp, yy, ix, queue, centers, center_ids, seed)
            if (panel + 1) % 10 == 0:
                print(f"MODEL {label} batch {panel + 1}/{args.batches}", flush=True)
        model["intra"] = {}
        for g in SCALES:
            rs, cs = gap_summary(radial[g], 1000 / g), gap_summary(_concat(contrasts[g]), 4.)
            model["intra"][str(g)] = {"radial": rs, "contrastive": cs}
            intra_rows.extend(({"model": label, "scale": g, "term": "radial", **rs},
                               {"model": label, "scale": g, "term": "contrastive", **cs}))
        np.savez_compressed(private / f"{label}_objects.npz", ids=cache["ids"], labels=labels_np,
                            chain_gaps=cg, **{f"depth_{k}": v for k, v in dh.items()})
        if acc is not None:
            model["relations"] = acc.finish()
            rr = model["relations"]
            for family in ("ord", "hier"):
                for edge, info in rr[family]["edges"].items():
                    relation_rows.append({"model": label, "family": family, "edge": edge, **info})
            for role, item in rr["roles"].items():
                role_rows.append({"model": label, "role": role, **{
                    k: v for k, v in item.items() if k not in ("scale_counts", "scale_fractions", "depth")},
                    **{f"scale_{g}_count": item["scale_counts"][col] for col, g in enumerate(SCALES)},
                    **{f"scale_{g}_fraction": item["scale_fractions"][col] for col, g in enumerate(SCALES)},
                    "depth_median": item["depth"]["median"], "depth_p90": item["depth"]["p90"]})
            for group, items in acc.angles.items():
                for metric in ("nearest", "second_gap", "descendant_angles", "descendant_i", "descendant_j", "descendant_k"):
                    xx = _concat(items[metric])
                    angle_hist.extend(histogram_rows(label, xx, np.arange(0, 181, 1), group=group, metric=metric))
                    angle_rows.append({"model": label, "group": group, "metric": metric, **describe(xx)})
        else:
            model["relations"] = None
        model["wall_seconds"] = time.perf_counter() - started
        summary["models"][label] = model
        write_json(args.out_dir / "summary.partial.json", summary)
        print(f"MODEL {label} finished {model['wall_seconds']:.1f}s", flush=True)
        del whole, parts, labels
        if device.type == "cuda":
            torch.cuda.empty_cache()
    for name, rows in (("depth_hist", depth_hist), ("depth_ecdf", depth_ecdf), ("depth_by_class", depth_class),
                       ("chain_summary", chain_rows), ("chain_by_class", chain_class), ("intra_summary", intra_rows),
                       ("relation_summary", relation_rows), ("role_summary", role_rows), ("angle_summary", angle_rows),
                       ("angle_hist", angle_hist)):
        write_csv(args.out_dir / f"{name}.csv", rows)
    write_json(args.out_dir / "summary.json", summary)
    print("METRICS_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
