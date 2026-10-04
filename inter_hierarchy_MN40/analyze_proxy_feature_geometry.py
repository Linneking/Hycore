"""CPU geometry audit of a clean V5/V6 feature_cache.npz (NumPy only).

No checkpoint loading, model execution, graph mining, training, or GPU use.
The cached eligibility mask is preserved. Proxy->sample nearest neighbours
and sample->proxy assignments are reported separately; neither is the
historical Gumbel pair/triple ancestor selection used during training.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

# Bound CPU BLAS parallelism when the caller has not set its own preference.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
import numpy as np

CLASS_NAMES = (
    "airplane", "bathtub", "bed", "bench", "bookshelf", "bottle", "bowl", "car",
    "chair", "cone", "cup", "curtain", "desk", "door", "dresser", "flower_pot",
    "glass_box", "guitar", "keyboard", "lamp", "laptop", "mantel", "monitor",
    "night_stand", "person", "piano", "plant", "radio", "range_hood", "sink",
    "sofa", "stairs", "stool", "table", "tent", "toilet", "tv_stand", "vase",
    "wardrobe", "xbox",
)


def class_name(label):
    return CLASS_NAMES[label] if 0 <= label < len(CLASS_NAMES) else f"label_{label}"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def distribution(values):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if not len(values):
        return {"count": 0, "mean": None, "std": None, "min": None,
                "p10": None, "p25": None, "median": None, "p75": None,
                "p90": None, "max": None}
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite geometry statistic")
    quantiles = np.quantile(values, [.1, .25, .5, .75, .9])
    return {"count": len(values), "mean": float(values.mean()),
            "std": float(values.std()), "min": float(values.min()),
            **dict(zip(("p10", "p25", "median", "p75", "p90"), map(float, quantiles))),
            "max": float(values.max())}


def norms_squared(x):
    return np.einsum("ij,ij->i", x, x)


def pair_matrices(x, y):
    """Return exact c1 ball distance and origin-direction angle, in FP64."""
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    xn, yn = norms_squared(x), norms_squared(y)
    dot = x @ y.T
    squared = np.maximum(xn[:, None] + yn[None, :] - 2 * dot, 0)
    distance = 2 * np.arcsinh(np.sqrt(squared / ((1 - xn[:, None]) * (1 - yn[None, :]))))
    angle = np.degrees(np.arccos(np.clip(dot / np.sqrt(xn[:, None] * yn[None, :]), -1, 1)))
    return distance, angle


def row_pair_values(x, y):
    xn, yn = norms_squared(x), norms_squared(y)
    dot = np.einsum("ij,ij->i", x, y)
    angle = np.degrees(np.arccos(np.clip(dot / np.sqrt(xn * yn), -1, 1)))
    squared = np.maximum(xn + yn - 2 * dot, 0)
    return angle, 2 * np.arcsinh(np.sqrt(squared / ((1 - xn) * (1 - yn))))


def validated_cache(cache):
    required = ("mu", "proxy_ball", "sample_ids", "labels", "proxy_eligible")
    if any(key not in cache for key in required):
        raise ValueError("Cache needs mu/proxy_ball/sample_ids/labels/proxy_eligible")
    mu = np.asarray(cache["mu"], dtype=np.float64)
    proxies = np.asarray(cache["proxy_ball"], dtype=np.float64)
    ids = np.asarray(cache["sample_ids"], dtype=np.int64).reshape(-1)
    labels = np.asarray(cache["labels"], dtype=np.int64).reshape(-1)
    eligible = np.asarray(cache["proxy_eligible"], dtype=bool).reshape(-1)
    if mu.ndim != 2 or proxies.ndim != 2 or mu.shape[1] != proxies.shape[1]:
        raise ValueError("Features/proxies must be aligned 2-D arrays")
    if len(mu) != len(ids) or len(ids) != len(labels) or len(proxies) != len(eligible):
        raise ValueError("Cache arrays are not aligned")
    if len(ids) < 4 or len(proxies) < 2 or len(set(ids.tolist())) != len(ids):
        raise ValueError("Need at least4 distinct sample IDs and2 proxies")
    for points in (mu, proxies):
        n2 = norms_squared(points)
        if not np.isfinite(points).all() or (n2 <= 0).any() or (n2 >= 1).any():
            raise ValueError("Need finite nonzero c1 ball points; no remapping is performed")
    classes, counts = np.unique(labels, return_counts=True)
    if len(classes) < 2 or (counts < 2).any():
        raise ValueError("Need at least two classes with two samples each")
    # Canonical ID order makes equal-seed V5/V6 cross draws use the same IDs
    # whenever their split/labels match, even if cache row order differs.
    order = np.argsort(ids, kind="stable")
    return {"mu": mu[order], "proxy_ball": proxies, "sample_ids": ids[order],
            "labels": labels[order], "proxy_eligible": eligible,
            "cached_top4_ids": np.asarray(cache["all_proxy_nearest_ids"], dtype=np.int64)
            if "all_proxy_nearest_ids" in cache else None}


def proxy_sample_neighbours(mu, proxies, ids, block_size=128):
    """Compute raw and direction-only top4, plus actual sample->proxy nearest.

    Query blocks avoid materialising a sample x sample matrix. Stable top4
    ties use canonical sample IDs; nearest-proxy ties use proxy index.
    """
    raw = np.empty((len(proxies), 4), dtype=np.int64)
    directional = np.empty_like(raw)
    raw_distance, raw_angle = np.empty_like(raw, dtype=float), np.empty_like(raw, dtype=float)
    directional_angle = np.empty_like(raw, dtype=float)
    assignment = np.zeros(len(mu), dtype=np.int64)
    assignment_distance = np.full(len(mu), np.inf)
    for start in range(0, len(proxies), block_size):
        stop = min(start + block_size, len(proxies))
        distance, angle = pair_matrices(proxies[start:stop], mu)
        selected_raw = np.argsort(distance, axis=1, kind="stable")[:, :4]
        selected_direction = np.argsort(angle, axis=1, kind="stable")[:, :4]
        raw[start:stop], directional[start:stop] = selected_raw, selected_direction
        raw_distance[start:stop] = np.take_along_axis(distance, selected_raw, axis=1)
        raw_angle[start:stop] = np.take_along_axis(angle, selected_raw, axis=1)
        directional_angle[start:stop] = np.take_along_axis(angle, selected_direction, axis=1)
        local = distance.argmin(axis=0)
        best = distance[local, np.arange(len(mu))]
        improves = best < assignment_distance
        assignment[improves] = start + local[improves]
        assignment_distance[improves] = best[improves]
    return {"raw_positions": raw, "direction_positions": directional,
            "raw_ids": ids[raw], "direction_ids": ids[directional],
            "raw_distances": raw_distance, "raw_angles": raw_angle,
            "direction_angles": directional_angle,
            "sample_to_proxy_index": assignment, "sample_to_proxy_distance": assignment_distance}


def neighbour_summary(neighbours, labels, ids, radius, low_mask, selected):
    positions = neighbours[selected]
    if not len(positions):
        return {"proxy_count": 0, "slots": 0, "unique_sample_ids": 0, "unique_unordered_top4_sets": 0}
    nids = ids[positions]
    top1, top4 = Counter(map(int, nids[:, 0])), Counter(map(int, nids.reshape(-1)))
    sets = Counter(tuple(sorted(map(int, row))) for row in nids)
    by_id = {int(sample_id): index for index, sample_id in enumerate(ids)}

    def hubs(counter):
        return [{"sample_id": sample_id, "class_name": class_name(int(labels[by_id[sample_id]])),
                 "proxy_count": count, "radius": float(radius[by_id[sample_id]]),
                 "in_lowest_radius_1percent": bool(low_mask[by_id[sample_id]])}
                for sample_id, count in counter.most_common(10)]
    class_counts = Counter(map(int, labels[positions].reshape(-1)))
    top1_classes = Counter(map(int, labels[positions[:, 0]]))
    probabilities = np.asarray(list(top4.values()), dtype=float) / nids.size
    return {"proxy_count": len(positions), "slots": nids.size,
            "unique_sample_ids": len(top4), "unique_sample_fraction": len(top4) / len(ids),
            "unique_top1_ids": len(top1), "unique_unordered_top4_sets": len(sets),
            "unique_ordered_top4_lists": len({tuple(row) for row in nids.tolist()}),
            "rows_in_repeated_sets": sum(count for count in sets.values() if count > 1),
            "max_set_repeat_count": max(sets.values()),
            "most_repeated_sets": [{"sample_ids": list(key), "proxy_count": count}
                                   for key, count in sets.most_common(10)],
            "top1_hubs": hubs(top1), "top4_hubs": hubs(top4),
            "top4_effective_sample_count_from_slot_entropy": float(np.exp(-np.sum(probabilities * np.log(probabilities)))),
            "classes_in_top4": len(class_counts), "classes_in_top1": len(top1_classes),
            "top4_class_slot_counts": {class_name(k): v for k, v in sorted(class_counts.items())},
            "top1_class_proxy_counts": {class_name(k): v for k, v in sorted(top1_classes.items())},
            "single_class_rows": sum(len(set(labels[row].tolist())) == 1 for row in positions),
            "lowest_radius_1percent_top1_fraction": float(low_mask[positions[:, 0]].mean()),
            "lowest_radius_1percent_top4_slot_fraction": float(low_mask[positions].mean()),
            "lowest_radius_1percent_unique_neighbor_fraction": float(np.mean([low_mask[by_id[i]] for i in top4]))}


def proxy_geometry(proxies, neighbour_ids, mask):
    points, neighbours = proxies[mask], neighbour_ids[mask]
    radius = np.sqrt(norms_squared(points))
    result = {"proxy_count": len(points), "radius": distribution(radius),
              "depth_d_origin": distribution(2 * np.arctanh(radius))}
    if len(points) < 2:
        return result
    distance, angle = pair_matrices(points, points)
    upper = np.triu_indices(len(points), 1)
    signatures = [tuple(sorted(map(int, row))) for row in neighbours]
    same = np.array([signatures[i] == signatures[j] for i, j in zip(*upper)])
    for name, condition in (("all_distinct_pairs", np.ones(len(same), dtype=bool)),
                            ("same_unordered_top4", same), ("different_unordered_top4", ~same)):
        values = distance[upper][condition]
        result[name] = {"hyperbolic_distance": distribution(values),
                        "angle_degrees": distribution(angle[upper][condition]),
                        "distance_le_1e_minus6_pair_count": int(np.sum(values <= 1e-6))}
    np.fill_diagonal(distance, np.inf)
    near = distance.argmin(axis=1)
    ix = np.arange(len(points))
    result["nearest_proxy_within_domain"] = {"self_excluded": True,
        "hyperbolic_distance": distribution(distance[ix, near]),
        "angle_degrees": distribution(angle[ix, near]),
        "same_unordered_top4_fraction": float(np.mean([signatures[i] == signatures[j] for i, j in enumerate(near)]))}
    result["interpretation"] = "Equal top4 sets do not establish coincident proxies or common parent relationships; inspect their actual distances/angles."
    return result


def assignment_summary(assigned, distances, proxy_count, eligible):
    counts = np.bincount(assigned, minlength=proxy_count)
    fractions = counts[counts > 0].astype(float) / len(assigned)
    entropy = float(-np.sum(fractions * np.log(fractions)))
    return {"direction": "each sample -> its nearest proxy among ALL cached proxies",
            "sample_count": len(assigned), "per_proxy_assignment_counts": counts.tolist(),
            "occupied_proxies": int(np.sum(counts > 0)),
            "zero_assignment_fraction": float(np.mean(counts == 0)),
            "largest_assignment_fraction": float(counts.max() / len(assigned)),
            "assignment_entropy_nats": entropy, "effective_proxies_exp_entropy": float(np.exp(entropy)),
            "effective_proxies_inverse_simpson": float(1 / np.sum(fractions ** 2)),
            "samples_assigned_to_eligible_fraction": float(eligible[assigned].mean()),
            "nearest_hyperbolic_distance": distribution(distances),
            "interpretation": "Nearest-proxy occupancy, not actual pair/triple ancestor usage; zero occupancy does not by itself imply a dead ancestor."}


def all_sample_nearest(mu, block_size=128):
    """Exact all-sample neighbours, excludes self; ties use cache's ID order."""
    n2 = norms_squared(mu)
    radius = np.sqrt(n2)
    nearest_hyp, nearest_angle = np.empty(len(mu), dtype=np.int64), np.empty(len(mu), dtype=np.int64)
    nearest_distance = np.empty(len(mu))
    for start in range(0, len(mu), block_size):
        stop = min(start + block_size, len(mu))
        dot = mu[start:stop] @ mu.T
        squared = np.maximum(n2[start:stop, None] + n2[None, :] - 2 * dot, 0)
        # Query-side (1-r_i²) is a positive row constant: omit for argmin.
        cost = squared / (1 - n2[None, :])
        local, glob = np.arange(stop - start), np.arange(start, stop)
        cost[local, glob] = np.inf
        near = cost.argmin(axis=1)
        nearest_hyp[start:stop] = near
        nearest_distance[start:stop] = 2 * np.arcsinh(np.sqrt(cost[local, near] / (1 - n2[start:stop])))
        cosine = dot / (radius[start:stop, None] * radius[None, :])
        cosine[local, glob] = -np.inf
        nearest_angle[start:stop] = cosine.argmax(axis=1)
    return nearest_hyp, nearest_angle, nearest_distance


def separation_probability(inner, outer):
    """P(exact empirical within distance < sampled cross distance), half ties."""
    ordered = np.sort(inner)
    left, right = np.searchsorted(ordered, outer, "left"), np.searchsorted(ordered, outer, "right")
    return float(np.mean((left + right) * .5 / len(ordered)))


def class_geometry(mu, labels, seed=22, cross_pairs=10000, block_size=128):
    radius = np.sqrt(norms_squared(mu))
    direction = mu / radius[:, None]
    rng, rows = np.random.default_rng(seed), []
    within_angles, within_distances, cross_angles, cross_distances = [], [], [], []
    nearest_hyp, nearest_angle, nearest_distance = all_sample_nearest(mu, block_size)
    hyp_correct, angle_correct = labels[nearest_hyp] == labels, labels[nearest_angle] == labels
    for label in np.unique(labels):
        ix, other = np.flatnonzero(labels == label), np.flatnonzero(labels != label)
        distance, angle = pair_matrices(mu[ix], mu[ix])
        upper = np.triu_indices(len(ix), 1)
        wi_angle, wi_distance = angle[upper], distance[upper]
        # Same seed, canonical IDs and labels match cross draws across caches.
        anchors = rng.choice(ix, cross_pairs, replace=True)
        partners = rng.choice(other, cross_pairs, replace=True)
        co_angle, co_distance = row_pair_values(mu[anchors], mu[partners])
        rows.append({"label": int(label), "class_name": class_name(int(label)), "samples": len(ix),
            "direction_resultant_R": float(np.linalg.norm(direction[ix].mean(axis=0))),
            "radius": distribution(radius[ix]), "depth_d_origin": distribution(2 * np.arctanh(radius[ix])),
            "near_whole_native_boundary_fraction": float(np.mean(radius[ix] >= .99599)),
            "within_angle_degrees": distribution(wi_angle), "within_hyperbolic": distribution(wi_distance),
            "cross_angle_degrees": distribution(co_angle), "cross_hyperbolic": distribution(co_distance),
            "probability_within_less_than_cross_angle": separation_probability(wi_angle, co_angle),
            "probability_within_less_than_cross_hyperbolic": separation_probability(wi_distance, co_distance),
            "hyperbolic_nearest_same_class_fraction": float(hyp_correct[ix].mean()),
            "angular_nearest_same_class_fraction": float(angle_correct[ix].mean()),
            "nearest_hyperbolic_distance": distribution(nearest_distance[ix])})
        within_angles.append(wi_angle)
        within_distances.append(wi_distance)
        cross_angles.append(co_angle)
        cross_distances.append(co_distance)
    def macro_distribution_mean(key):
        return float(np.mean([row[key]["mean"] for row in rows]))
    result = {"classes": len(rows), "sample_count": len(mu),
        "exact_within_distinct_pair_count": sum(len(values) for values in within_angles),
        "sampled_cross_pair_count": len(rows) * cross_pairs,
        "direction_resultant_R": distribution([row["direction_resultant_R"] for row in rows]),
        "pooled": {"within_angle_degrees": distribution(np.concatenate(within_angles)),
                   "within_hyperbolic": distribution(np.concatenate(within_distances)),
                   "cross_angle_degrees": distribution(np.concatenate(cross_angles)),
                   "cross_hyperbolic": distribution(np.concatenate(cross_distances))},
        "macro_equal_class": {key + "_mean": macro_distribution_mean(key) for key in
            ("within_angle_degrees", "within_hyperbolic", "cross_angle_degrees", "cross_hyperbolic")},
        "all_sample_nearest": {"self_excluded": True, "hyperbolic_same_class_count": int(hyp_correct.sum()),
            "hyperbolic_same_class_fraction": float(hyp_correct.mean()),
            "angular_same_class_count": int(angle_correct.sum()), "angular_same_class_fraction": float(angle_correct.mean()),
            "nearest_hyperbolic_distance": distribution(nearest_distance)},
        "whole_radius": distribution(radius), "whole_depth": distribution(2 * np.arctanh(radius)),
        "near_whole_native_boundary_fraction": float(np.mean(radius >= .99599))}
    for key in ("probability_within_less_than_cross_angle", "probability_within_less_than_cross_hyperbolic",
                "hyperbolic_nearest_same_class_fraction", "angular_nearest_same_class_fraction"):
        result["macro_equal_class"][key] = float(np.mean([row[key] for row in rows]))
    return result, rows


def analyze_geometry(cache, seed=22, cross_pairs=10000, block_size=128):
    if cross_pairs < 1 or block_size < 1:
        raise ValueError("cross_pairs/block_size must be positive")
    data = validated_cache(cache)
    mu, proxies = data["mu"], data["proxy_ball"]
    ids, labels, eligible = data["sample_ids"], data["labels"], data["proxy_eligible"]
    radius = np.sqrt(norms_squared(mu))
    lowest_count = max(1, math.ceil(len(mu) * .01))
    low = np.zeros(len(mu), dtype=bool)
    low[np.lexsort((ids, radius))[:lowest_count]] = True
    neighbours = proxy_sample_neighbours(mu, proxies, ids, block_size)
    retrieval, geometry = {}, {}
    for name, selected in (("eligible", eligible), ("all", np.ones(len(proxies), dtype=bool))):
        raw = neighbour_summary(neighbours["raw_positions"], labels, ids, radius, low, selected)
        control = neighbour_summary(neighbours["direction_positions"], labels, ids, radius, low, selected)
        intersections = [len(set(a) & set(b)) for a, b in zip(neighbours["raw_ids"][selected], neighbours["direction_ids"][selected])]
        intersection_sum, union_sum = sum(intersections), sum(8 - n for n in intersections)
        raw["neighbor_hyperbolic_distances"] = distribution(neighbours["raw_distances"][selected])
        raw["neighbor_angle_degrees"] = distribution(neighbours["raw_angles"][selected])
        control["neighbor_angle_degrees"] = distribution(neighbours["direction_angles"][selected])
        retrieval[name] = {"hyperbolic": raw, "direction_only_control": control,
                          "hyperbolic_vs_direction_top4_jaccard_pooled": intersection_sum / union_sum if union_sum else None,
                          "hyperbolic_vs_direction_top1_match_fraction": float(np.mean(neighbours["raw_ids"][selected, 0] == neighbours["direction_ids"][selected, 0])) if np.any(selected) else None}
        geometry[name] = proxy_geometry(proxies, neighbours["raw_ids"], selected)
    classes, rows = class_geometry(mu, labels, seed, cross_pairs, block_size)
    source_ids = data["cached_top4_ids"]
    report = {"format": "proxy-feature-geometry-c1-1", "sample_count": len(mu), "proxy_count": len(proxies),
        "eligible_proxy_count": int(eligible.sum()), "dimension": mu.shape[1], "curvature": 1,
        "sampling_and_denominators": {"seed": seed, "class_cross_pairs": cross_pairs, "block_size": block_size,
            "within": "All unordered distinct-ID pairs per class, exact, diagonal excluded.",
            "cross": "Equal draw count per anchor class, with replacement. Anchor uniform within class; partner uniform among all other rows (foreign class sizes are not equalized). Canonical ascending sample-ID order, ascending labels, NumPy default_rng(seed).",
            "pooled": "Within-class distributions weight classes by N_c*(N_c-1)/2. Cross distributions weight anchor classes equally, because each gets the same draws.",
            "macro": "Mean of per-class means/probabilities/purities, equal class weight. R distribution gives each class one value.",
            "proxy_pairs": "All unordered distinct proxy-index pairs in the stated domain, classified by equality of sorted raw top4 sample IDs. Not an identity or ancestor-equivalence claim.",
            "self_exclusion": "Whole NN and proxy NN exclude their own row; identical embeddings at different IDs/indices remain legitimate candidates.",
            "ties": "Distances/angles computed FP64. Sample ties use ascending ID; proxy-assignment ties use ascending proxy index.",
            "direction_control": "Unit directions only, ranking by angle; radius removed without remapping the cached ball features.",
            "lowest_radius_1percent": "Exactly ceil(.01*N) lowest-radius samples, ties by sample ID; slot fractions use all stated top1/top4 positions, not unique-ID denominators."},
        "lowest_radius_set": {"count": lowest_count, "dataset_fraction": lowest_count / len(mu),
                              "radius_max": float(radius[low].max())},
        "cached_top4_ids_match_recomputed": bool(np.array_equal(source_ids, neighbours["raw_ids"])) if source_ids is not None else None,
        "proxy_to_sample": retrieval, "proxy_geometry": geometry,
        "sample_to_proxy": assignment_summary(neighbours["sample_to_proxy_index"], neighbours["sample_to_proxy_distance"], len(proxies), eligible),
        "class_geometry": classes, "per_class": rows,
        "limitations": ["Descriptive clean-cache geometry; does not recover actual Gumbel pair/triple ancestor use.",
                        "Training features do not establish validation/test structure, morphology semantics, or the causal contribution of HIER.",
                        "Eligibility is the cached relation-mining mask, not a count of effective ancestors; top4 is a visualization/retrieval choice.",
                        "Repeated proxy neighbourhoods can coexist with noncoincident proxy directions/radii; inspect pair geometry.",
                        "Cross pairs are sampled; no confidence interval or independent morphology signal is supplied."]}
    return report


def write_outputs(output, report):
    (output / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    columns = ("class_name", "label", "samples", "R", "radius_median", "within_angle_median", "within_angle_p90",
               "cross_angle_median", "within_hyperbolic_median", "within_hyperbolic_p90", "cross_hyperbolic_median",
               "hyperbolic_nn_same_class", "angular_nn_same_class", "within_less_than_cross_hyperbolic_probability")
    with (output / "per_class.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, columns)
        writer.writeheader()
        for row in report["per_class"]:
            writer.writerow({"class_name": row["class_name"], "label": row["label"], "samples": row["samples"],
                "R": row["direction_resultant_R"], "radius_median": row["radius"]["median"],
                "within_angle_median": row["within_angle_degrees"]["median"], "within_angle_p90": row["within_angle_degrees"]["p90"],
                "cross_angle_median": row["cross_angle_degrees"]["median"],
                "within_hyperbolic_median": row["within_hyperbolic"]["median"], "within_hyperbolic_p90": row["within_hyperbolic"]["p90"],
                "cross_hyperbolic_median": row["cross_hyperbolic"]["median"],
                "hyperbolic_nn_same_class": row["hyperbolic_nearest_same_class_fraction"],
                "angular_nn_same_class": row["angular_nearest_same_class_fraction"],
                "within_less_than_cross_hyperbolic_probability": row["probability_within_less_than_cross_hyperbolic"]})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True, help="NEW private/ignored directory")
    parser.add_argument("--seed", type=int, default=22)
    parser.add_argument("--cross-pairs", type=int, default=10000)
    parser.add_argument("--block-size", type=int, default=128)
    args = parser.parse_args(argv)
    if args.cross_pairs < 1 or args.block_size < 1:
        parser.error("cross-pairs and block-size must be positive")
    started = time.perf_counter()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    with np.load(args.cache, allow_pickle=False) as cache:
        report = analyze_geometry(cache, args.seed, args.cross_pairs, args.block_size)
    source_summary = args.cache.parent / "summary.json"
    source_identity = None
    if source_summary.exists():
        source = json.loads(source_summary.read_text(encoding="utf-8"))
        source_identity = {key: source.get(key) for key in ("checkpoint", "dataset", "configuration")}
    report["identity"] = {"cache_path": str(args.cache.resolve()), "cache_sha256": sha256(args.cache),
        "code_sha256": sha256(Path(__file__)), "source_visualization_identity": source_identity,
        "command_argv": [sys.executable, *sys.argv], "numpy_version": np.__version__,
        "device": "CPU", "completed_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "wall_seconds": time.perf_counter() - started}
    write_outputs(args.output_dir, report)
    print(json.dumps({"status": "complete", "output_dir": str(args.output_dir.resolve()),
        "samples": report["sample_count"], "proxies": report["proxy_count"], "eligible": report["eligible_proxy_count"],
        "eligible_top4_unique_ids": report["proxy_to_sample"]["eligible"]["hyperbolic"]["unique_sample_ids"],
        "all_top4_unique_ids": report["proxy_to_sample"]["all"]["hyperbolic"]["unique_sample_ids"],
        "hyperbolic_nn_same_class": report["class_geometry"]["all_sample_nearest"]["hyperbolic_same_class_fraction"],
        "wall_seconds": report["identity"]["wall_seconds"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
