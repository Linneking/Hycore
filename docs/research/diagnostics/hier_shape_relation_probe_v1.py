"""Bounded CPU independent geometry audit of within-class HIER relations.

Each selected class contributes a fixed cohort of at most64 train objects.
Only that cohort receives a pairwise shape matrix (2016 unordered pairs at
N64); ranks/nearest neighbours are cohort-conditional, never global claims.
Original and equal-radius sample mining are compared on identical objects;
ancestor losses always retain original whole/proxy coordinates.

Geometry signal: original first1024 points, centroid removal, one max-radius
scale, deterministic farthest-point sampling256, symmetric mean squared
Chamfer. No rotation alignment is performed. This is an independent shape
similarity signal, not semantic truth or evidence of directed specificity.
Class-only K20 mining is a diagnostic intervention and differs from training's
global32classes x2 batch. New FP32 CPU hard-Gumbel replays are distinguished
from historical training selections. No model/GPU/optimizer is used.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from inter_hierarchy_MN40.hier_proxy_scratch_v5.relations import mine_sample_triplets, poincare_distance
from inter_hierarchy_MN40.hier_proxy_scratch_v6.hier_loss import ghhc_loss
from inter_hierarchy_MN40.hier_proxy_scratch_v6.structure_monitor import ancestor_summary, StructureAccumulator

NAMES = "airplane bathtub bed bench bookshelf bottle bowl car chair cone cup curtain desk door dresser flower_pot glass_box guitar keyboard lamp laptop mantel monitor night_stand person piano plant radio range_hood sink sofa stairs stool table tent toilet tv_stand vase wardrobe xbox".split()
FORMAT = "hycore-hier-v6-h20-selfk300-1"
MODES = ("original", "sample_mining_equal_radius")


def sha_file(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def sha_array(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def json_write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def distribution(values):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if not len(values):
        return {"n": 0, "mean": None, "quantiles": None, "positive_fraction": None}
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite shape/relationship quantity")
    return {"n": len(values), "mean": float(values.mean()),
            "quantiles": np.quantile(values, [0., .1, .5, .9, 1.]).tolist(),
            "positive_fraction": float(np.mean(values > 0))}


def average_ranks(values):
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    ends = np.cumsum(counts)
    return ((ends - counts + ends - 1) / 2.)[inverse]


def spearman(left, right):
    left, right = average_ranks(np.asarray(left)), average_ranks(np.asarray(right))
    if not len(left) or np.std(left) == 0 or np.std(right) == 0:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def cohort_rng(seed, label, purpose):
    key = f"hier-shape-cohort-v1:{seed}:{label}:{purpose}".encode("utf-8")
    return np.random.default_rng(int.from_bytes(hashlib.sha256(key).digest()[:8], "little"))


def load_cache(cache_path, metadata_path, checkpoint_path):
    hashes = {"cache": sha_file(cache_path), "cache_metadata": sha_file(metadata_path)}
    metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    if metadata.get("status") != "complete" or metadata["checkpoint"].get("format") != FORMAT:
        raise ValueError("Require verified complete original V6 cache metadata")
    if metadata["geometry"].get("c") != 1 or metadata["geometry"].get("D") != 256 or metadata["geometry"].get("P") != 512:
        raise ValueError("Require original c1/D256/P512 cache")
    data_identity = metadata["dataset"]
    if data_identity.get("split") != "train_ids" or data_identity.get("test_read") is not False:
        raise ValueError("Require clean train cache; no official test reading")
    shards = data_identity.get("training_shards_sorted")
    if not shards or shards != sorted(shards) or len(set(shards)) != len(shards):
        raise ValueError("Require unique sorted-shard cache identity")
    with np.load(cache_path, allow_pickle=False) as z:
        ids, labels, whole, proxies = (z[key].copy() for key in ("sample_ids", "labels", "mu", "proxy_ball"))
    ids, labels = np.asarray(ids, dtype=np.int64).reshape(-1), np.asarray(labels, dtype=np.int64).reshape(-1)
    if len(ids) != 8856 or len(np.unique(ids)) != len(ids) or len(labels) != len(ids):
        raise ValueError("Expected8856 distinct cached train objects")
    if sha_array(ids) != data_identity["ids_sha256"] or sha_array(labels) != data_identity["labels_sha256"]:
        raise ValueError("Cached ID/label hashes disagree with metadata")
    if whole.shape != (8856, 256) or proxies.shape != (512, 256) or whole.dtype != np.float32 or proxies.dtype != np.float32:
        raise ValueError("Expected original FP32 whole/proxy geometry")
    for name, points in (("whole", whole), ("proxy", proxies)):
        if not np.isfinite(points).all() or not np.all(np.sum(points * points, axis=1) < 1):
            raise ValueError("Expected finite strict c1 ball vectors")
        if name == "whole" and np.any(np.linalg.norm(points, axis=1) == 0):
            raise ValueError("Whole angle/equal-radius mining undefined for zero vectors")
    if checkpoint_path is not None:
        hashes["checkpoint"] = sha_file(checkpoint_path)
        if hashes["checkpoint"] != metadata["checkpoint"]["sha256"]:
            raise ValueError("Supplied checkpoint SHA256 mismatches cache")
    return ids, labels, torch.from_numpy(whole), torch.from_numpy(proxies), {
        "hashes": hashes, "epoch": int(metadata["checkpoint"]["epoch"]),
        "checkpoint_sha256_from_metadata": metadata["checkpoint"]["sha256"],
        "checkpoint_bytes_verified": checkpoint_path is not None,
        "checkpoint_split_sha256": data_identity["checkpoint_split_sha256"],
        "ids_sha256": sha_array(ids), "labels_sha256": sha_array(labels),
        "training_shards_sorted": shards, "feature_input": data_identity.get("feature_input")}


def inspect_training_shards(data_dir, expected_shards, cached_ids, cached_labels):
    import h5py
    paths = sorted(Path(data_dir).resolve(strict=True).glob("ply_data_train*.h5"))
    if [path.name for path in paths] != expected_shards:
        raise ValueError("HDF5 sorted shard names disagree with cache canonical identity")
    labels, identity, offset = [], [], 0
    for path in paths:
        before = sha_file(path)
        with h5py.File(path, "r") as z:
            shape = z["data"].shape
            shard_labels = np.asarray(z["label"], dtype=np.int64).reshape(-1)
            if len(shape) != 3 or shape[1] < 1024 or shape[2] != 3 or len(shard_labels) != shape[0]:
                raise ValueError("Unexpected ModelNet train shard layout")
        identity.append({"name": path.name, "count": shape[0], "points_per_object": shape[1],
                         "canonical_offset": offset, "sha256": before})
        labels.append(shard_labels)
        offset += len(shard_labels)
    all_labels = np.concatenate(labels)
    if len(all_labels) != 9840 or cached_ids.min() < 0 or cached_ids.max() >= len(all_labels):
        raise ValueError("Cache does not address original9840 sorted train objects")
    np.testing.assert_array_equal(all_labels[cached_ids], cached_labels)
    return paths, identity


def canonical_location(data_id, shard_identity):
    for info in shard_identity:
        row = int(data_id) - info["canonical_offset"]
        if 0 <= row < info["count"]:
            return {"shard": info["name"], "row": row}
    raise ValueError("ID outside sorted-shard canonical range")


def read_clouds(paths, shard_identity, selected_ids):
    """Canonical IDs are sorted shard offsets plus row, never glob order."""
    import h5py
    result = {}
    for path, info in zip(paths, shard_identity):
        offset, count = info["canonical_offset"], info["count"]
        in_shard = np.sort(np.asarray([i for i in selected_ids if offset <= i < offset + count], dtype=np.int64))
        if not len(in_shard):
            continue
        rows = in_shard - offset
        with h5py.File(path, "r") as z:
            points = np.asarray(z["data"][rows, :1024, :3], dtype=np.float32)
        for data_id, cloud in zip(in_shard.tolist(), points):
            if not np.isfinite(cloud).all():
                raise ValueError("Nonfinite source point cloud")
            result[data_id] = cloud
    if set(result) != set(selected_ids):
        raise ValueError("Missing canonical cloud IDs")
    return result


def normalize_fps(cloud, point_count=256):
    """One isotropic scale preserves aspect ratios; deterministic FPS ties by row."""
    cloud = np.asarray(cloud, dtype=np.float64)
    centered = cloud - cloud.mean(axis=0, keepdims=True)
    scale = np.linalg.norm(centered, axis=1).max()
    if not np.isfinite(scale) or scale <= 1e-12 or len(cloud) < point_count:
        raise ValueError("Degenerate or undersized source cloud")
    normalized = centered / scale
    closest = np.full(len(cloud), np.inf)
    at = int(np.argmax(np.sum(normalized * normalized, axis=1)))
    selected = []
    for _ in range(point_count):
        selected.append(at)
        squared = np.sum((normalized - normalized[at]) ** 2, axis=1)
        closest = np.minimum(closest, squared)
        # Do not choose an already sampled row even for duplicated vertices.
        closest[np.asarray(selected, dtype=np.int64)] = -np.inf
        at = int(np.argmax(closest))
    selected = np.asarray(selected, dtype=np.int64)
    return normalized[selected].astype(np.float32), {
        "source_points": len(cloud), "selected_points": point_count,
        "centroid": cloud.mean(axis=0).tolist(), "uniform_scale": float(scale),
        "FPS_source_row_sha256": sha_array(selected), "normalized_cloud_sha256": sha_array(normalized[selected].astype(np.float32))}


@torch.no_grad()
def symmetric_squared_chamfer(left, right):
    left, right = torch.as_tensor(left, dtype=torch.float32), torch.as_tensor(right, dtype=torch.float32)
    if left.ndim == 2:
        left, right = left[None], right[None]
    distances = (left.square().sum(-1)[:, :, None] + right.square().sum(-1)[:, None, :]
                 - 2 * torch.bmm(left, right.transpose(1, 2))).clamp_min(0)
    return .5 * (distances.min(-1).values.mean(-1) + distances.min(-2).values.mean(-1))


def shape_matrix(clouds, batch_size):
    count = len(clouds)
    i, j = np.triu_indices(count, 1)
    result = np.zeros((count, count), dtype=np.float64)
    tensor = torch.from_numpy(np.stack(clouds))
    for start in range(0, len(i), batch_size):
        stop = start + batch_size
        values = symmetric_squared_chamfer(tensor[i[start:stop]], tensor[j[start:stop]]).numpy()
        result[i[start:stop], j[start:stop]] = values
        result[j[start:stop], i[start:stop]] = values
    if not np.isfinite(result).all():
        raise RuntimeError("Nonfinite shape matrix")
    return result


def angular_matrix(whole):
    array = whole.numpy().astype(np.float64)
    unit = array / np.linalg.norm(array, axis=1, keepdims=True)
    result = np.degrees(np.arccos(np.clip(unit @ unit.T, -1., 1.)))
    np.fill_diagonal(result, 0.)
    return result


def anchor_percentiles(shape):
    n = len(shape)
    result = np.zeros_like(shape)
    for at in range(n):
        candidates = np.flatnonzero(np.arange(n) != at)
        result[at, candidates] = average_ranks(shape[at, candidates]) / max(1, n - 2)
    return result


def edge_report(mask, shape, angle, hyperbolic, shape_rank):
    mask = np.asarray(mask, dtype=bool).copy()
    np.fill_diagonal(mask, False)
    upper = np.triu(mask, 1)
    degrees = mask.sum(-1)
    eligible = degrees > 0
    anchor_shape = (shape * mask).sum(-1)[eligible] / degrees[eligible]
    anchor_rank = (shape_rank * mask).sum(-1)[eligible] / degrees[eligible]
    return {"unordered_pairs": int(upper.sum()), "directed_edges": int(mask.sum()),
        "anchors_with_selected_edges": int(mask.any(-1).sum()),
        "shape_chamfer": distribution(shape[upper]), "angle_degrees": distribution(angle[upper]),
        "original_hyperbolic_distance": distribution(hyperbolic[upper]),
        "cohort_conditional_shape_rank_percentile": distribution(shape_rank[mask]),
        "anchor_equal_mean_chamfer": distribution(anchor_shape),
        "anchor_equal_mean_shape_rank_percentile": distribution(anchor_rank)}


def grouping_report(triplets, details, role, shape, angle):
    pair, triple = details["pair_proxy_idx"].numpy(), details["triple_proxy_idx"].numpy()
    valid = pair != triple
    active = valid & (details["per_draw_loss"].numpy() > 0)
    groups = {}
    selected = pair if role == "pair" else triple
    endpoints = triplets.numpy()[:, :2] if role == "pair" else triplets.numpy()
    class_baseline = float(shape[np.triu_indices(len(shape), 1)].mean())
    for domain, mask in (("noncollision", valid), ("active_noncollision", active)):
        by_proxy = defaultdict(set)
        for proxy_id, members in zip(selected[mask].tolist(), endpoints[mask].tolist()):
            by_proxy[proxy_id].update(members)
        rows, pooled_shapes = [], []
        for proxy_id, members in sorted(by_proxy.items()):
            members = np.array(sorted(members), dtype=np.int64)
            if len(members) < 2:
                continue
            i, j = np.triu_indices(len(members), 1)
            values = shape[members[i], members[j]]
            pooled_shapes.append(values)
            rows.append({"proxy_id": proxy_id, "unique_cohort_endpoints": len(members),
                "cohort_endpoint_rows": members.tolist(), "unordered_endpoint_pairs": len(values),
                "shape_mean": float(values.mean()), "shape_mean_over_full_cohort_mean": float(values.mean() / class_baseline) if class_baseline else None,
                "angle_mean_degrees": float(angle[members[i], members[j]].mean())})
        groups[domain] = {"groups": rows,
            "group_endpoint_pair_occurrence_weighted_shape": distribution(np.concatenate(pooled_shapes) if pooled_shapes else []),
            "full_cohort_mean_chamfer_baseline": class_baseline,
            "interpretation": "Unions of actually selected draw endpoints; overlaps permitted. These are not verified descendants or a partition."}
    return groups


@torch.no_grad()
def class_probe(label, ids, whole, proxies, normalized, args):
    n = len(ids)
    shape = shape_matrix(normalized, args.shape_batch_size)
    angle = angular_matrix(whole)
    original_dist = poincare_distance(whole).numpy().astype(np.float64)
    ranks = anchor_percentiles(shape)
    targets = torch.full((n,), label, dtype=torch.long)
    data_ids = torch.from_numpy(ids)
    relation_seed = args.seed + label * 10007
    equal = args.mining_radius * whole / whole.norm(dim=-1, keepdim=True)
    mined = {mode: mine_sample_triplets(whole if mode == "original" else equal, targets,
        topk=20, t_per_anchor=50, seed=relation_seed, exclude_self_negative=True, data_ids=data_ids) for mode in MODES}
    raw, controlled = (mined[mode]["mutual"].numpy() for mode in MODES)
    baseline = np.zeros((n, n), dtype=bool)
    rng = cohort_rng(args.seed, label, "random_baseline")
    for at in range(n):
        available = np.flatnonzero(np.arange(n) != at)
        selected = rng.choice(available, min(args.random_per_anchor, n - 1), replace=False)
        baseline[at, selected] = True
    # Symmetric baseline union is reported as unique undirected pairs.
    baseline = baseline | baseline.T
    masks = {"original_mutual": raw, "equal_radius_mutual": controlled,
        "intersection": raw & controlled, "original_only": raw & ~controlled,
        "equal_only": controlled & ~raw, "fixed_random_same_class": baseline,
        "all_cohort_pairs": ~np.eye(n, dtype=bool)}
    upper = np.triu_indices(n, 1)
    union = int((raw | controlled).sum())
    common_eligible = (mined["original"]["eligible"] & mined["sample_mining_equal_radius"]["eligible"]).numpy()
    anchor_changes = []
    for at in np.flatnonzero(common_eligible):
        anchor_changes.append({"sample_id": int(ids[at]),
            "equal_minus_original_mean_chamfer": float(shape[at, controlled[at]].mean() - shape[at, raw[at]].mean()),
            "equal_minus_original_mean_shape_rank_percentile": float(ranks[at, controlled[at]].mean() - ranks[at, raw[at]].mean())})
    report = {"class": NAMES[label], "label": label, "cohort_size": n,
        "pair_count": len(upper[0]), "sample_ids": ids.tolist(),
        "mining_stats": {mode: mined[mode]["stats"] for mode in MODES},
        "mutual_jaccard": int((raw & controlled).sum()) / union if union else None,
        "paired_common_eligible_anchor_changes": {"anchors": anchor_changes,
            "common_anchor_count": len(anchor_changes),
            "equal_minus_original_mean_chamfer": distribution([v["equal_minus_original_mean_chamfer"] for v in anchor_changes]),
            "equal_minus_original_mean_shape_rank_percentile": distribution([v["equal_minus_original_mean_shape_rank_percentile"] for v in anchor_changes]),
            "interpretation": "Negative changes favor equal-radius mining on this independent geometry signal; no semantic/hierarchy guarantee"},
        "edge_geometry": {name: edge_report(mask, shape, angle, original_dist, ranks) for name, mask in masks.items()},
        "all_fixed_pair_spearman": {"shape_chamfer_vs_original_whole_distance": spearman(shape[upper], original_dist[upper]),
                                   "shape_chamfer_vs_whole_angle": spearman(shape[upper], angle[upper])},
        "ancestor_replays": {}}
    # The loss-distance matrix is identical across mining modes.
    loss_distances = poincare_distance(whole, proxies)
    for mode in MODES:
        structure = StructureAccumulator()
        repetitions = []
        triplets = mined[mode]["triplets"]
        draw_i, draw_j, draw_k = triplets.numpy().T
        for repeat in range(args.gumbel_repeats):
            seed = args.seed + 900001 + label * 1009 + repeat * 1000003
            loss, stats, details = ghhc_loss(loss_distances, triplets, .1, .1,
                torch.Generator(device="cpu").manual_seed(seed), return_details=True)
            structure.update({"sample": ancestor_summary(whole, proxies, triplets, details)})
            i, j, k = triplets.numpy().T
            valid = details["pair_proxy_idx"].numpy() != details["triple_proxy_idx"].numpy()
            positive, negative_i, negative_j = shape[i, j], shape[i, k], shape[j, k]
            gap = .5 * (negative_i + negative_j) - positive
            repetitions.append({"repeat": repeat, "seed": seed, "unweighted_loss": float(loss), "source_loss_stats": stats,
                "independent_shape_noncollision_positive": distribution(positive[valid]),
                "independent_shape_positive_all_mined_draws": distribution(shape[draw_i, draw_j]),
                "independent_shape_noncollision_average_negative_minus_positive": distribution(gap[valid]),
                "positive_strictly_closer_than_both_negative_endpoints_fraction": float(np.mean((positive < np.minimum(negative_i, negative_j))[valid])) if valid.any() else None,
                "pair_endpoint_groups": grouping_report(triplets, details, "pair", shape, angle),
                "triple_endpoint_groups": grouping_report(triplets, details, "triple", shape, angle)})
        report["ancestor_replays"][mode] = {"repeats": repetitions, "pooled_actual_cpu_selection_structure": structure.summary()}
    # Full bounded pair table lets a reviewer inspect the same pair in either
    # mining condition without independently chosen model neighbours.
    pair_rows = [{"class": NAMES[label], "id_i": int(ids[i]), "id_j": int(ids[j]),
        "shape_chamfer_squared": float(shape[i, j]), "angle_degrees": float(angle[i, j]),
        "original_hyperbolic_distance": float(original_dist[i, j]),
        "original_mutual": bool(raw[i, j]), "equal_radius_mutual": bool(controlled[i, j]),
        "fixed_random_baseline": bool(baseline[i, j])} for i, j in zip(*upper)]
    return report, pair_rows


def self_test():
    shard_identity = [{"name": "ply_data_train0.h5", "count": 3, "canonical_offset": 0},
                      {"name": "ply_data_train1.h5", "count": 2, "canonical_offset": 3}]
    assert canonical_location(4, shard_identity) == {"shard": "ply_data_train1.h5", "row": 1}
    assert canonical_location(0, shard_identity) == {"shard": "ply_data_train0.h5", "row": 0}
    try:
        canonical_location(5, shard_identity)
    except ValueError:
        pass
    else:
        raise AssertionError("Out-of-range canonical ID accepted")
    rng = np.random.default_rng(22)
    cloud = rng.normal(size=(1024, 3)) * np.array([2., 1., .4])
    sampled, first = normalize_fps(cloud)
    transformed, second = normalize_fps(3.7 * cloud + np.array([5., -4., 2.]))
    np.testing.assert_allclose(sampled, transformed, atol=1e-6, rtol=1e-6)
    assert first["FPS_source_row_sha256"] == second["FPS_source_row_sha256"]
    a, b = torch.from_numpy(sampled), torch.from_numpy(sampled[::-1].copy())
    assert float(symmetric_squared_chamfer(a, a)[0]) < 1e-6
    assert float(symmetric_squared_chamfer(a, b)[0]) < 1e-6
    other = a * torch.tensor([1., .8, 1.])
    torch.testing.assert_close(symmetric_squared_chamfer(a, other), symmetric_squared_chamfer(other, a), atol=1e-7, rtol=1e-6)
    rotation = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    assert float(symmetric_squared_chamfer(a, a @ rotation)[0]) > 1e-4
    shape = shape_matrix([sampled, transformed, other.numpy()], 2)
    np.testing.assert_allclose(shape, shape.T, atol=0, rtol=0)
    assert np.max(np.abs(np.diag(shape))) == 0
    synthetic_shape = np.array([[0., .1, .8], [.1, 0., .9], [.8, .9, 0.]])
    percentile = anchor_percentiles(synthetic_shape)
    assert percentile[0, 1] == 0 and percentile[0, 2] == 1
    generator = torch.Generator().manual_seed(123)
    tangent = torch.randn((32, 8), generator=generator) * .2
    whole = torch.tanh(tangent.norm(dim=-1, keepdim=True)) * tangent / tangent.norm(dim=-1, keepdim=True)
    proxies_tangent = torch.randn((32, 8), generator=generator) * .3
    proxies = torch.tanh(proxies_tangent.norm(dim=-1, keepdim=True)) * proxies_tangent / proxies_tangent.norm(dim=-1, keepdim=True)
    clouds = [normalize_fps(cloud + rng.normal(size=cloud.shape) * .05, 32)[0] for _ in range(32)]
    args = argparse.Namespace(seed=22, mining_radius=.98, random_per_anchor=3, gumbel_repeats=1, shape_batch_size=4)
    report, pairs = class_probe(14, np.arange(32), whole, proxies, clouds, args)
    assert len(pairs) == 496 and report["cohort_size"] == 32
    assert report["mining_stats"]["original"]["self_k_triplets"] == 0
    # Ensure JSON-safe counters and empty active-group handling.
    json.dumps(report, allow_nan=False)
    print(json.dumps({"self_test": "passed", "checks": ["sorted-shard offset/row canonical ID mapping", "translation/isotropic scale invariance", "deterministic FPS",
        "Chamfer symmetry/identity/permutation", "relative rotation changes shape signal", "bounded pair matrix/ranks",
        "source mining and hard-Gumbel grouping with self-k excluded", "JSON-safe output"]}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--cache-metadata", type=Path)
    parser.add_argument("--checkpoint", type=Path, help="Optional source byte-hash verification; no model loading")
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--classes", nargs="+", default=["dresser", "bathtub", "chair", "flower_pot"])
    parser.add_argument("--per-class", type=int, choices=(32, 64), default=64)
    parser.add_argument("--points", type=int, choices=(256,), default=256)
    parser.add_argument("--seed", type=int, default=22)
    parser.add_argument("--mining-radius", type=float, default=.98)
    parser.add_argument("--random-per-anchor", type=int, default=5)
    parser.add_argument("--gumbel-repeats", type=int, choices=(1, 3, 5), default=3)
    parser.add_argument("--shape-batch-size", type=int, default=16)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.threads < 1 or args.shape_batch_size < 1 or args.random_per_anchor < 1 or not 0 < args.mining_radius < 1:
        parser.error("Require positive CPU threads/batch/random count and mining radius in(0,1)")
    torch.set_num_threads(args.threads)
    if args.self_test:
        self_test()
        return
    if any(value is None for value in (args.cache, args.data_dir, args.output_dir)):
        parser.error("--cache, --data-dir and --output-dir are required")
    if args.classes == ["all"]:
        selected_classes = list(range(40))
    else:
        try:
            selected_classes = [NAMES.index(value) if value in NAMES else int(value) for value in args.classes]
        except ValueError:
            parser.error("Use class names, integer labels0..39 or all")
        if len(set(selected_classes)) != len(selected_classes) or any(not 0 <= value < 40 for value in selected_classes):
            parser.error("Class labels must be distinct and in0..39")
    args.cache_metadata = args.cache_metadata or args.cache.with_name("summary.json")
    if args.output_dir.exists():
        raise FileExistsError("Output must be new; existing results are immutable")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    report = {"status": "running", "format": "hier-shape-relation-probe-v1", "started_utc": datetime.now(timezone.utc).isoformat()}
    json_write(args.output_dir / "summary.json", report)
    try:
        ids, labels, whole, proxies, identity = load_cache(args.cache, args.cache_metadata, args.checkpoint)
        paths, shard_identity = inspect_training_shards(args.data_dir, identity["training_shards_sorted"], ids, labels)
        selected_rows, selection = {}, {}
        for label in selected_classes:
            available = np.flatnonzero(labels == label)
            # Start from canonical ID order before seeded sampling, independent
            # of the cache row numbering or a caller's HDF5 glob order.
            available = available[np.argsort(ids[available])]
            count = min(args.per_class, len(available))
            if count <= 20:
                raise ValueError(f"Class {NAMES[label]} needs more than20 cohort objects for K20")
            chosen = cohort_rng(args.seed, label, "objects").choice(available, count, replace=False)
            chosen = chosen[np.argsort(ids[chosen])]
            selected_rows[label] = chosen
            selection[NAMES[label]] = {"available_cached_class_objects": len(available), "cohort_objects": count,
                "class_population_coverage": count / len(available), "sample_ids": ids[chosen].tolist()}
        selected_ids = sorted({int(ids[row]) for rows in selected_rows.values() for row in rows})
        clouds = read_clouds(paths, shard_identity, selected_ids)
        normalized, normalization_identity = {}, {}
        for data_id in selected_ids:
            normalized[data_id], normalization_identity[str(data_id)] = normalize_fps(clouds[data_id], args.points)
        json_write(args.output_dir / "cohort_identity.json", {"selection": selection,
            "canonical_locations": {str(data_id): canonical_location(data_id, shard_identity) for data_id in selected_ids},
            "normalization": normalization_identity, "shards": shard_identity})
        cohort_sha = sha_file(args.output_dir / "cohort_identity.json")
        class_reports, pair_rows = [], []
        for label in selected_classes:
            rows = selected_rows[label]
            current, pairs = class_probe(label, ids[rows], whole[rows], proxies,
                [normalized[int(ids[row])] for row in rows], args)
            current["class_population_coverage"] = selection[NAMES[label]]["class_population_coverage"]
            class_reports.append(current)
            pair_rows.extend(pairs)
            print(f"Shape relation CPU probe complete: {NAMES[label]}, objects={len(rows)}, pairs={len(pairs)}", flush=True)
        import csv
        with (args.output_dir / "fixed_pair_geometry.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(pair_rows[0]))
            writer.writeheader()
            writer.writerows(pair_rows)
        comparison = {}
        for group in ("original_mutual", "equal_radius_mutual", "original_only", "equal_only", "fixed_random_same_class"):
            comparison[group] = {
                "class_equal_mean_chamfer": distribution([c["edge_geometry"][group]["shape_chamfer"]["mean"] for c in class_reports if c["edge_geometry"][group]["shape_chamfer"]["n"]]),
                "class_equal_mean_cohort_shape_rank_percentile": distribution([c["edge_geometry"][group]["cohort_conditional_shape_rank_percentile"]["mean"] for c in class_reports if c["edge_geometry"][group]["cohort_conditional_shape_rank_percentile"]["n"]]),
                "total_unordered_selected_pairs": sum(c["edge_geometry"][group]["unordered_pairs"] for c in class_reports)}
        comparison["paired_equal_minus_original_common_anchors"] = {
            "class_equal_mean_chamfer_change": distribution([c["paired_common_eligible_anchor_changes"]["equal_minus_original_mean_chamfer"]["mean"] for c in class_reports if c["paired_common_eligible_anchor_changes"]["common_anchor_count"]]),
            "class_equal_mean_shape_rank_change": distribution([c["paired_common_eligible_anchor_changes"]["equal_minus_original_mean_shape_rank_percentile"]["mean"] for c in class_reports if c["paired_common_eligible_anchor_changes"]["common_anchor_count"]]),
            "interpretation": "Each class contributes one mean paired-anchor change; negative favors equal-radius geometry concordance"}
        for key, path in (("cache", args.cache), ("cache_metadata", args.cache_metadata)):
            if sha_file(path) != identity["hashes"][key]:
                raise RuntimeError(f"Source {key} changed during shape diagnosis")
        if args.checkpoint is not None and sha_file(args.checkpoint) != identity["hashes"]["checkpoint"]:
            raise RuntimeError("Source checkpoint changed during shape diagnosis")
        for path, info in zip(paths, shard_identity):
            if sha_file(path) != info["sha256"]:
                raise RuntimeError("Source training shard changed during shape diagnosis")
        try:
            code_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            code_commit = None
        report.update(status="complete", identity=identity, training_shard_identity=shard_identity,
            cohort_identity_sha256=cohort_sha, configuration={"classes": [NAMES[c] for c in selected_classes],
                "per_class_limit": args.per_class, "geometry_points": args.points, "source_points": 1024,
                "seed": args.seed, "K": 20, "T": 50, "margin": .1, "tau": .1, "mining_radius": args.mining_radius,
                "gumbel_repeats": args.gumbel_repeats, "shape_batch_size": args.shape_batch_size, "threads": args.threads},
            selection=selection, class_equal_comparison=comparison, classes=class_reports,
            independent_geometry={"centering": "mean of first1024 points", "scale": "one maximum centered Euclidean radius",
                "sampling": "deterministic FPS256, starts at furthest-from-centroid point, row-index ties",
                "distance": "0.5*(mean_x min_y squared_distance + mean_y min_x squared_distance)",
                "orientation": "original dataset orientation; no rotation alignment",
                "point_pair_arithmetic": "FP32 batched squared distance with nonnegative numerical clamp"},
            limits=["All ranks/nearest neighbours/relations are conditional on the fixed class cohort, not full-class/global nearest neighbours",
                "Class-only K20/T50 mining differs from the training balanced32classes x2 protocol",
                "Whole angle/equal-radius ranking equivalence is mathematical, not independent morphology evidence",
                "Chamfer after isotropic normalization measures aligned surface proximity, not semantic similarity or hierarchy truth",
                "Point sampling, outliers and relative rotation can influence shape distance; no reflection/rotation alignment",
                "Cache metadata supplies sorted shard names and labels, not historical HDF byte hashes; current authoritative HDF provenance is explicitly fingerprinted here",
                "Proxy groups are overlapping unions of selected endpoints, not verified descendants or a clustering partition",
                "Source CPU hard-Gumbel is a new replay; cross-condition noise draws cannot identify identical triplets when mining changes",
                "Gumbel repeat variation is conditional algorithmic randomness, not multi-seed training uncertainty",
                "No model forward, CUDA, gradients, optimizer update or official test reading"],
            checks={"full8856_cache_HDF_label_match": True, "sorted_shard_canonical_identity": True,
                "source_file_hashes_unchanged": True, "selected_clouds_normalized_finite": True},
            runtime={"python": sys.version, "torch": torch.__version__, "numpy": np.__version__, "device": "cpu"},
            optimizer_updates=0, code_commit=code_commit, diagnostic_script_sha256=sha_file(__file__),
            finished_utc=datetime.now(timezone.utc).isoformat(), wall_seconds=time.perf_counter() - started)
        json_write(args.output_dir / "summary.json", report)
        print(json.dumps({"status": "complete", "epoch": identity["epoch"], "classes": len(selected_classes),
            "cohort_objects": len(selected_ids), "unordered_shape_pairs": len(pair_rows), "wall_seconds": report["wall_seconds"]}), flush=True)
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}",
            finished_utc=datetime.now(timezone.utc).isoformat(), wall_seconds=time.perf_counter() - started)
        json_write(args.output_dir / "summary.json", report)
        raise


if __name__ == "__main__":
    main()
