"""Frozen whole/proxy radial and directional evolution (CPU NumPy only).

Usage: python hier_v7_directional_evolution_v1.py --spec INPUT.json --output NEW_DIR
Spec: {"snapshots": [{"epoch": 0, "cache": "absolute/path.npz", ...}],
       "normalized": "optional/normalized_runs.json", "run_id": "optional",
       "class_names": ["optional label-ordered names"]}.

Full IDs, per-object arrays and CSVs are private delivery artifacts, never Git
summaries. No encoder forward, optimizer update, training import or GPU use.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import datetime as dt
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

EPS = 1e-12
QLEVELS = [0., .01, .05, .25, .5, .75, .95, .99, 1.]
QKEYS = ["min", "p01", "p05", "p25", "median", "p75", "p95", "p99", "max"]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def stats(values):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"n": 0}
    return {"n": int(len(values)), **dict(zip(QKEYS, map(float, np.quantile(values, QLEVELS)))),
            "mean": float(values.mean()), "std_population": float(values.std()),
            "variance_population": float(values.var())}


def unit(values):
    values = np.asarray(values, dtype=np.float64)
    norms = np.linalg.norm(values, axis=1)
    result = np.zeros_like(values)
    np.divide(values, norms[:, None], out=result, where=norms[:, None] > EPS)
    return result, norms > EPS


def angle(cosine):
    return np.degrees(np.arccos(np.clip(cosine, -1., 1.)))


def scalar(value):
    return value.item() if isinstance(value, np.generic) else value


def centers(directions, labels, classes):
    sums = np.array([directions[labels == label].sum(axis=0) for label in classes])
    counts = np.array([(labels == label).sum() for label in classes], dtype=np.int64)
    if np.any(counts == 0):
        raise ValueError("Every declared class needs at least one candidate")
    means = sums / counts[:, None]
    normalized, defined = unit(means)
    return sums, counts, means, normalized, defined


def class_geometry(directions, labels, classes, depths=None, logits=None):
    """Exact class macro pair cosines, with no sampled pairs or self pairs.

    Within_c = (||sum u||^2-n)/(n(n-1)); between_cd = mean_c dot mean_d.
    Leave-one-out changes only the query's true class center, including its norm.
    """
    sums, counts, means, normal, defined = centers(directions, labels, classes)
    lookup = {scalar(label): j for j, label in enumerate(classes)}
    true = np.array([lookup[scalar(label)] for label in labels])
    resultant = np.linalg.norm(means, axis=1)
    within = np.full(len(classes), np.nan)
    eligible = counts > 1
    within[eligible] = (np.sum(sums[eligible] ** 2, axis=1) - counts[eligible]) / (counts[eligible] * (counts[eligible] - 1))
    pair_indices = np.triu_indices(len(classes), 1)
    between = (means @ means.T)[pair_indices]
    centroid_angles = angle(normal @ normal.T)
    valid_pairs = (defined[:, None] & defined[None, :])[pair_indices]
    centroid_pair_angles = centroid_angles[pair_indices][valid_pairs]

    similarity = directions @ normal.T
    similarity[:, ~defined] = -np.inf
    own_sums = sums[true] - directions
    own_norm = np.linalg.norm(own_sums, axis=1)
    own_valid = (own_norm > EPS) & (counts[true] > 1)
    own_cos = np.full(len(directions), -np.inf)
    own_cos[own_valid] = (np.sum(directions[own_valid] * sums[true[own_valid]], axis=1) - 1.) / own_norm[own_valid]
    similarity[np.arange(len(directions)), true] = own_cos
    predicted = np.argmax(similarity, axis=1)
    correct = (predicted == true) & own_valid
    other = similarity.copy()
    other[np.arange(len(directions)), true] = -np.inf
    best_other = np.max(other, axis=1)
    margin_cos = own_cos - best_other
    margin_deg = angle(best_other) - angle(own_cos)
    margin_cos[~own_valid] = np.nan
    margin_deg[~own_valid] = np.nan
    sample_angle = angle(np.sum(directions * normal[true], axis=1))
    sample_angle[~defined[true]] = np.nan
    clean_correct = None
    if logits is not None:
        if logits.ndim != 2 or logits.shape[0] != len(labels) or not np.isfinite(logits).all():
            raise ValueError("Invalid cached logits")
        if labels.dtype.kind not in "iu" or min(labels) < 0 or max(labels) >= logits.shape[1]:
            raise ValueError("Logit columns require explicit integer labels")
        clean_correct = np.argmax(logits, axis=1) == labels

    rows, confusion = [], np.zeros((len(classes), len(classes)), dtype=np.int64)
    for j, label in enumerate(classes):
        mask = true == j
        np.add.at(confusion[j], predicted[mask], 1)
        competitor = centroid_angles[j].copy()
        competitor[~defined] = np.inf
        competitor[j] = np.inf
        nearest = int(np.argmin(competitor))
        row = {"class_label": scalar(label), "count": int(counts[j]), "resultant_length": float(resultant[j]),
               "center_defined": bool(defined[j]), "within_cosine_excluding_self": scalar(within[j]) if eligible[j] else None,
               "loo_accuracy": float(correct[mask].mean()), "loo_valid_fraction": float(own_valid[mask].mean()),
               "loo_cosine_margin_mean": stats(margin_cos[mask]).get("mean"),
               "loo_angle_margin_mean_deg": stats(margin_deg[mask]).get("mean"),
               "angle_to_own_center_deg": stats(sample_angle[mask]),
               "nearest_other_class": scalar(classes[nearest]) if np.isfinite(competitor[nearest]) and defined[j] else None,
               "nearest_other_center_angle_deg": float(competitor[nearest]) if np.isfinite(competitor[nearest]) and defined[j] else None,
               "most_confused_other_class": None}
        mistaken = confusion[j].copy()
        mistaken[j] = 0
        if mistaken.max() > 0:
            row["most_confused_other_class"] = scalar(classes[np.argmax(mistaken)])
        if depths is not None:
            row["origin_depth"] = stats(depths[mask])
        if clean_correct is not None:
            row["cached_clean_prediction_accuracy"] = float(clean_correct[mask].mean())
        rows.append(row)
    common_mean = means.mean(axis=0)
    common_norm = np.linalg.norm(common_mean)
    common_angles = angle(directions @ (common_mean / common_norm)) if common_norm > EPS else np.array([])
    centroid_cos = (normal @ normal.T)[pair_indices][valid_pairs]
    macro_acc = np.mean([row["loo_accuracy"] for row in rows])
    summary = {"within_class_cosine_macro": stats(within), "between_class_cosine_macro_pairs": stats(between),
               "within_minus_between_macro_mean": stats(within).get("mean", 0.) - stats(between).get("mean", 0.),
               "class_resultant_lengths": stats(resultant), "class_center_angles_deg": stats(centroid_pair_angles),
               "class_center_cosines": stats(centroid_cos), "class_balanced_common_resultant_length": float(common_norm),
               "common_cone_sample_angles_deg": stats(common_angles),
               "common_cone_class_balanced_mean_angle_deg": float(np.mean([np.mean(common_angles[true == j]) for j in range(len(classes))])) if common_angles.size else None,
               "loo_direction_accuracy_micro": float(correct.mean()), "loo_direction_accuracy_macro": float(macro_acc),
               "loo_valid_fraction": float(own_valid.mean()), "loo_cosine_margin": stats(margin_cos),
               "loo_angle_margin_deg": stats(margin_deg),
               "cached_clean_prediction_accuracy_micro": float(clean_correct.mean()) if clean_correct is not None else None,
               "cached_clean_prediction_accuracy_macro": float(np.mean([r["cached_clean_prediction_accuracy"] for r in rows])) if clean_correct is not None else None}
    arrays = {"class_counts": counts, "class_means": means, "class_centers": normal, "class_centers_defined": defined,
              "class_resultant_lengths": resultant, "class_center_angles_deg": centroid_angles,
              "class_confusion": confusion, "whole_true_class_index": true,
              "whole_loo_predicted_class_index": predicted, "whole_loo_correct": correct,
              "whole_loo_angle_margin_deg": margin_deg, "whole_angle_to_own_class_center_deg": sample_angle,
              "common_mean": common_mean}
    return summary, rows, arrays


def top_rows(scores, candidate_ids, k):
    """Descending score; exact ties use ascending stable sample IDs."""
    if scores.shape[1] < k:
        raise ValueError("Candidate pool smaller than topk")
    order = np.argsort(candidate_ids, kind="stable")
    local = np.argsort(-scores[:, order], axis=1, kind="stable")[:, :k]
    return order[local]


def raw_score(proxy, whole, c):
    numerator = np.sum(proxy ** 2, axis=1)[:, None] + np.sum(whole ** 2, axis=1)[None, :] - 2. * (proxy @ whole.T)
    if (numerator < -1e-12).any():
        raise ValueError("Invalid squared distance")
    return -np.maximum(numerator, 0.) / (1. - c * np.sum(whole ** 2, axis=1)[None, :])


def retrieval_summary(rows, ids, labels, classes):
    counts = Counter(ids[rows].reshape(-1).tolist())
    sets = Counter(tuple(sorted(ids[row].tolist())) for row in rows)
    class_slots = {str(scalar(label)): int((labels[rows] == label).sum()) for label in classes}
    leading = [{"sample_id": scalar(i), "slots": n} for i, n in counts.most_common(20)]
    return {"covered_samples": len(counts), "unique_neighbor_sets": len(sets),
            "maximum_repeated_set": max(sets.values(), default=0),
            "top4_sample_slot_share": sum(sorted(counts.values(), reverse=True)[:4]) / rows.size if rows.size else None,
            "class_slot_counts": class_slots, "hottest_samples": leading}


def proxy_affinity(proxy_direction, whole_direction, labels, ids, classes, class_arrays, mask=None, ks=(16, 32, 64)):
    mask = np.ones(len(ids), dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    selected = np.flatnonzero(mask)
    if not len(selected):
        return {"available": False, "reason": "No retained candidates"}, {}, None
    retained_classes = np.array([label for label in classes if np.any(labels[mask] == label)])
    _, counts, means, normal, defined = centers(whole_direction[mask], labels[mask], retained_classes)
    norm_scores = proxy_direction @ normal.T
    norm_scores[:, ~defined] = -np.inf
    mean_scores = proxy_direction @ means.T
    norm_order = np.argsort(-norm_scores, axis=1, kind="stable")
    mean_order = np.argsort(-mean_scores, axis=1, kind="stable")
    first, second = norm_order[:, 0], norm_order[:, min(1, len(retained_classes) - 1)]
    near = retained_classes[first]
    nearest_defined = defined[first]
    similarity = norm_scores[np.arange(len(first)), first]
    margin = similarity - norm_scores[np.arange(len(first)), second]
    if len(retained_classes) < 2:
        margin[:] = np.nan
    similarity[~nearest_defined] = np.nan
    margin[~nearest_defined] = np.nan
    data = {"nearest_class": near, "nearest_class_defined": nearest_defined,
            "nearest_center_cosine": similarity, "nearest_center_angle_deg": angle(similarity),
            "center_top2_cosine_margin": margin,
            "mean_vector_nearest_class": retained_classes[mean_order[:, 0]],
            "mean_vector_top2_cosine_margin": mean_scores[np.arange(len(first)), mean_order[:, 0]] - mean_scores[np.arange(len(first)), mean_order[:, min(1, len(retained_classes)-1)]]}
    scores = proxy_direction @ whole_direction[mask].T
    maxk = min(max(ks), len(selected))
    nearest_rows = selected[top_rows(scores, ids[mask], maxk)]
    for k in ks:
        if k > maxk:
            continue
        neighbours = labels[nearest_rows[:, :k]]
        purity, dominant = [], []
        for row in neighbours:
            frequency = Counter(row.tolist())
            label = sorted(frequency, key=lambda value: (-frequency[value], value))[0]
            dominant.append(label)
            purity.append(frequency[label] / k)
        data[f"top{k}_dominant_class"] = np.asarray(dominant)
        data[f"top{k}_dominant_purity"] = np.asarray(purity)
        data[f"top{k}_nearest_center_class_purity"] = np.mean(neighbours == near[:, None], axis=1)
    # Cone comparison uses complete-pool class angles when no exclusion; controls
    # reestimate the 90% cone from their retained candidates and class centers.
    cone = np.array([np.quantile(angle(whole_direction[mask][labels[mask] == label] @ normal[j]), .9)
                     if defined[j] else np.nan for j, label in enumerate(retained_classes)])
    data["nearest_class_whole_cone90_deg"] = cone[first]
    data["inside_nearest_class_whole_cone90"] = (data["nearest_center_angle_deg"] <= cone[first]) & nearest_defined
    summary = {"available": True, "candidate_count": int(mask.sum()), "retained_class_count": int(len(retained_classes)),
               "class_center_defined_count": int(defined.sum()),
               "nearest_center_class_counts": {str(scalar(label)): int((near == label).sum()) for label in retained_classes},
               "mean_vector_nearest_class_counts": {str(scalar(label)): int((data['mean_vector_nearest_class'] == label).sum()) for label in retained_classes},
               "center_vs_mean_vector_class_agreement": float(np.mean(near == data["mean_vector_nearest_class"])),
               "nearest_center_angle_deg": stats(data["nearest_center_angle_deg"]),
               "center_top2_cosine_margin": stats(margin),
               "inside_nearest_class_cone90_fraction": float(data["inside_nearest_class_whole_cone90"].mean())}
    for key, value in data.items():
        if "purity" in key:
            summary[key] = stats(value)
    return summary, data, nearest_rows


def usage_for_epoch(normalized, epoch, pids):
    if normalized is None:
        return {}, None
    matching = [row for row in normalized.get("proxy_usage", []) if row["epoch"] == epoch]
    result = {}
    for row in matching:
        key = "/".join([row["component"], row["domain"], row["role"]])
        if key in result:
            raise ValueError("Ambiguous duplicate usage record")
        lookup = dict(zip(row["proxy_ids"], row["counts"]))
        if set(lookup) != set(pids.tolist()) or any(not np.isfinite(x) or x < 0 for x in lookup.values()):
            raise ValueError("Usage identities/counts disagree with cache")
        result[key] = np.array([lookup[scalar(i)] for i in pids], dtype=np.float64)
    selected = result.get("sample/noncollision/combined")
    return result, None if selected is None else selected > 0


def proxy_direction_groups(proxy_direction, assigned_classes, classes, used=None):
    """Proxy dispersion grouped by full-whole nearest directional center.

    Group labels are obtained from geometry, not training supervision. Reported
    separation must not be treated as independently validated semantic classes.
    """
    mean = proxy_direction.mean(axis=0)
    n = len(proxy_direction)
    all_pair = (np.linalg.norm(proxy_direction.sum(axis=0)) ** 2 - n) / (n * (n - 1)) if n > 1 else None
    present = np.array([label for label in classes if np.any(assigned_classes == label)])
    sums, counts, means, normal, defined = centers(proxy_direction, assigned_classes, present)
    eligible = counts > 1
    within = (np.sum(sums[eligible] ** 2, axis=1) - counts[eligible]) / (counts[eligible] * (counts[eligible] - 1))
    pair = np.triu_indices(len(present), 1)
    between = (means @ means.T)[pair]
    center_angles = angle(normal @ normal.T)[pair]
    row_angles = np.full(n, np.nan)
    records = []
    for j, label in enumerate(present):
        mask = assigned_classes == label
        row_angles[mask] = angle(proxy_direction[mask] @ normal[j]) if defined[j] else np.nan
        record = {"class_label": scalar(label), "proxy_count": int(counts[j]),
                  "proxy_direction_resultant": float(np.linalg.norm(means[j])),
                  "proxy_group_center_defined": bool(defined[j]),
                  "proxy_to_group_center_angle_deg": stats(row_angles[mask])}
        if used is not None:
            record["sample_used_proxy_count"] = int((mask & used).sum())
            for name, active in [("sample_used", used), ("sample_inactive", ~used)]:
                vectors = proxy_direction[mask & active]
                record[name + "_direction_resultant"] = float(np.linalg.norm(vectors.mean(axis=0))) if len(vectors) else None
        records.append(record)
    result = {"grouping": "nearest full-whole class directional center; endogenous descriptive groups",
              "all_proxy_common_resultant_length": float(np.linalg.norm(mean)),
              "all_proxy_pair_cosine_excluding_self": float(all_pair) if all_pair is not None else None,
              "assigned_class_group_count": len(present), "assigned_group_within_cosine_macro": stats(within),
              "assigned_group_between_cosine_macro_pairs": stats(between),
              "assigned_group_center_angles_deg": stats(center_angles), "groups": records}
    return result, row_angles


def weighted_alignment(previous_direction, next_direction, labels):
    """One orthogonal matrix fits same-ID whole, equal total weight per class.

    Next row vectors @ rotation are in the previous coordinate system. The SAME
    matrix must transform next whole and proxy. Reflection is allowed; rotation
    rank/singular values expose underdetermined directions.
    """
    classes, inverse, counts = np.unique(labels, return_inverse=True, return_counts=True)
    weights = 1. / counts[inverse]
    u, singular, vt = np.linalg.svd(next_direction.T @ (weights[:, None] * previous_direction), full_matrices=False)
    rotation = u @ vt
    residual = np.linalg.norm(next_direction @ rotation - previous_direction, axis=1)
    return rotation, {"weighting": "each class total weight=1; each object=1/n_class", "reflection_allowed": True,
                      "singular_values": singular.tolist(), "fit_rank_tolerance_1e-10": int((singular > 1e-10 * singular.max()).sum()),
                      "class_balanced_mean_squared_residual": float(np.sum(weights * residual ** 2) / weights.sum()),
                      "object_residual": stats(residual), "determinant": float(np.linalg.det(rotation))}


def transition(previous, current):
    ids, labels = previous["sample_ids"], previous["labels"]
    if not np.array_equal(ids, current["sample_ids"]) or not np.array_equal(labels, current["labels"]) or not np.array_equal(previous["proxy_ids"], current["proxy_ids"]):
        raise ValueError("Transition identities or labels differ")
    rotation, fit = weighted_alignment(previous["whole_direction"], current["whole_direction"], labels)
    aligned_w, aligned_p = current["whole_direction"] @ rotation, current["proxy_direction"] @ rotation
    cross = {}
    for condition, new_w, new_p in [("unaligned", current["whole_direction"], current["proxy_direction"]), ("whole_classbalanced_procrustes", aligned_w, aligned_p)]:
        panel = {}
        for name, pd, wd in [("Pprev_Wprev", previous["proxy_direction"], previous["whole_direction"]),
                             ("Pprev_Wnext", previous["proxy_direction"], new_w),
                             ("Pnext_Wprev", new_p, previous["whole_direction"]),
                             ("Pnext_Wnext", new_p, new_w)]:
            rows = top_rows(pd @ wd.T, ids, 4)
            panel[name] = retrieval_summary(rows, ids, labels, previous["classes"])
        cross[condition] = panel
    common_keys = [key for key in previous["affinity"] if key in current["affinity"]]
    affinity_changes = {}
    for key in common_keys:
        old, new = previous["affinity"][key], current["affinity"][key]
        if not old or not new:
            continue
        retained = old["nearest_class"] == new["nearest_class"]
        same_defined = old["nearest_class_defined"] & new["nearest_class_defined"]
        affinity_changes[key] = {"nearest_center_class_retention": float(retained[same_defined].mean()) if same_defined.any() else None,
                                 "both_defined_count": int(same_defined.sum()),
                                 "same_epoch_nearest_center_angle_change_deg": stats(new["nearest_center_angle_deg"] - old["nearest_center_angle_deg"]),
                                 "same_epoch_center_margin_change": stats(new["center_top2_cosine_margin"] - old["center_top2_cosine_margin"])}
        used = current["sample_used"]
        if used is not None:
            affinity_changes[key]["current_sample_used_retention"] = float(retained[used & same_defined].mean()) if (used & same_defined).any() else None
            affinity_changes[key]["current_sample_inactive_retention"] = float(retained[(~used) & same_defined].mean()) if ((~used) & same_defined).any() else None
    old_near = previous["affinity"]["all"]["nearest_class"]
    old_class_index = np.array([np.where(previous["classes"] == label)[0][0] for label in old_near])
    old_center = previous["class_centers"][old_class_index]
    new_old_center = current["class_centers"][old_class_index] @ rotation
    center_motion = angle(np.sum(old_center * new_old_center, axis=1))
    proxy_motion = angle(np.sum(previous["proxy_direction"] * aligned_p, axis=1))
    alignment_before = angle(np.sum(previous["proxy_direction"] * old_center, axis=1))
    alignment_after = angle(np.sum(current["proxy_direction"] * current["class_centers"][old_class_index], axis=1))
    summary = {"from_epoch": previous["epoch"], "to_epoch": current["epoch"], "epoch_gap": current["epoch"] - previous["epoch"],
               "whole_depth_change_same_ids": stats(current["whole_depth"] - previous["whole_depth"]),
               "proxy_depth_change_same_ids": stats(current["proxy_depth"] - previous["proxy_depth"]),
               "whole_angular_motion_unaligned_deg": stats(angle(np.sum(previous["whole_direction"] * current["whole_direction"], axis=1))),
               "whole_angular_motion_aligned_deg": stats(angle(np.sum(previous["whole_direction"] * aligned_w, axis=1))),
               "proxy_angular_motion_unaligned_deg": stats(angle(np.sum(previous["proxy_direction"] * current["proxy_direction"], axis=1))),
               "proxy_angular_motion_aligned_deg": stats(proxy_motion), "procrustes": fit,
               "previous_assigned_class_center_motion_aligned_deg": stats(center_motion),
               "proxy_angle_to_previous_assigned_class_before_deg": stats(alignment_before),
               "proxy_angle_to_previous_assigned_class_after_deg": stats(alignment_after),
               "proxy_previous_class_angle_change_deg": stats(alignment_after - alignment_before),
               "affinity_changes": affinity_changes, "direction_top4_cross_counterfactual": cross,
               "interpretation": "Both unaligned and one shared whole-ID fitted alignment are reported. Cross-epoch coordinate rotation and incomplete-rank alignment can confound cross panels; they are descriptive frozen controls, not optimizer causality."}
    arrays = {"rotation_next_to_previous": rotation, "proxy_aligned_angular_motion_deg": proxy_motion,
              "proxy_previous_class_center_aligned_motion_deg": center_motion,
              "proxy_previous_class_angle_change_deg": alignment_after - alignment_before,
              "whole_depth_change": current["whole_depth"] - previous["whole_depth"],
              "proxy_depth_change": current["proxy_depth"] - previous["proxy_depth"]}
    return summary, arrays


def write_csv(path, rows):
    keys = sorted({key for row in rows for key in row})
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False, allow_nan=False) if isinstance(value, (dict, list)) else value for key, value in row.items()})


def cache_load(record):
    path = Path(record["cache"])
    before = sha(path)
    if record.get("cache_sha256") and record["cache_sha256"] != before:
        raise ValueError("Declared cache hash differs")
    with np.load(path, allow_pickle=False) as cache:
        values = {key: cache[key].copy() for key in cache.files}
    for required in ["mu", "sample_ids", "labels", "proxy_tangent", "proxy_ids", "input_sha256"]:
        if required not in values:
            raise ValueError("Missing cache field " + required)
    if "c" not in values and "curvature" not in values:
        raise ValueError("Explicit curvature absent")
    c = float(values.get("c", values.get("curvature")))
    if "c" in values and "curvature" in values and float(values["c"]) != float(values["curvature"]):
        raise ValueError("Cache curvature fields disagree")
    ids, pids = values["sample_ids"].reshape(-1), values["proxy_ids"].reshape(-1)
    if ids.dtype.kind not in "iuUS" or pids.dtype.kind not in "iuUS" or len(np.unique(ids)) != len(ids) or len(np.unique(pids)) != len(pids):
        raise ValueError("Invalid or repeated stable identities")
    si, pi = np.argsort(ids, kind="stable"), np.argsort(pids, kind="stable")
    whole = values["mu"][si].astype(np.float64)
    tangent = values["proxy_tangent"][pi].astype(np.float64)
    if whole.ndim != 2 or tangent.ndim != 2 or whole.shape[1] != tangent.shape[1] or len(whole) != len(ids) or len(tangent) != len(pids):
        raise ValueError("Invalid embedding shape")
    return values, {"epoch": int(record["epoch"]), "source_path": path, "source_sha": before,
                    "sample_ids": ids[si], "proxy_ids": pids[pi], "labels": values["labels"].reshape(-1)[si],
                    "whole": whole, "proxy_tangent": tangent, "c": c,
                    "logits": values["logits"][si] if "logits" in values else None,
                    "input_sha256": str(values["input_sha256"]), "input_mode": str(values.get("input_mode", "unknown"))}


def analyze(spec_path, output):
    repo = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(repo))
    from tools.hier_postrun_audit.geometry import ball_geometry, expmap0, pool_hash
    out = Path(output)
    if out.exists():
        raise FileExistsError("Fresh output directory required")
    spec_path = Path(spec_path)
    spec = json.loads(spec_path.read_text(encoding="utf-8-sig"))
    records = sorted(spec["snapshots"], key=lambda row: row["epoch"])
    if not records or len({r["epoch"] for r in records}) != len(records) or any(not isinstance(r["epoch"], int) or r["epoch"] < 0 for r in records):
        raise ValueError("Unique nonnegative integer snapshot epochs required")
    normalized, normalized_sha = None, None
    if spec.get("normalized"):
        normalized_sha = sha(spec["normalized"])
        runs = json.loads(Path(spec["normalized"]).read_text(encoding="utf-8-sig"))
        if isinstance(runs, dict):
            runs = [runs]
        selected = [run for run in runs if not spec.get("run_id") or run.get("run_id") == spec["run_id"]]
        if len(selected) != 1:
            raise ValueError("Exactly one matching normalized run required")
        normalized = selected[0]
    out.mkdir(parents=True)
    output_json = {"schema": "hier_v7_directional_evolution_v1", "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                   "identity": {"spec_sha256": sha(spec_path), "script_sha256": sha(__file__), "normalized_sha256": normalized_sha,
                                "run_id": normalized.get("run_id") if normalized else spec.get("run_id")},
                   "protocol": {"CPU_NumPy_only": True, "optimizer_updates": 0, "encoder_forwards": 0,
                                "units": "full origin hyperbolic depth; direction cosine/angle degrees; variance population",
                                "pair_weighting": "class-macro within; unordered class-pair macro between", "proxy_numeric_radius_fraction": .999,
                                "center_epsilon": EPS, "snapshot_epochs": [r["epoch"] for r in records],
                                "missing_epoch_interpretation": "No continuous onset time inferred across unsaved intervals"},
                   "snapshots": [], "transitions": [],
                   "limitations": ["Class centers and purity are descriptive affinity, not supervised proxy labels or proof of hierarchy.",
                                   "Clean eval embeddings differ from train augmentations and per-step ancestor selections.",
                                   "Common-mean-subtracted directions are an analytical control, not the original trained geometry.",
                                   "Top4 retrieval is distinct from hard-Gumbel ancestor activation; missing usage is unknown.",
                                   "Cross-time shared Procrustes is a descriptive alignment and can be underdetermined."]}
    previous, reference, source_hashes = None, None, {}
    full_class_rows, full_proxy_rows = [], []
    for record in records:
        _, data = cache_load(record)
        source_hashes[data["source_path"]] = data["source_sha"]
        if reference is None:
            reference = data
        else:
            for key in ["sample_ids", "proxy_ids", "labels"]:
                if not np.array_equal(reference[key], data[key]):
                    raise ValueError("Snapshot identity/label pools disagree: " + key)
            for key in ["c", "input_sha256", "input_mode"]:
                if reference[key] != data[key]:
                    raise ValueError("Snapshot source protocol disagrees: " + key)
        epoch = data["epoch"]
        print(json.dumps({"phase": "snapshot", "epoch": epoch}), flush=True)
        whole, labels, ids, pids = data["whole"], data["labels"], data["sample_ids"], data["proxy_ids"]
        if len(labels) != len(ids):
            raise ValueError("Label shape differs")
        proxy = expmap0(data["proxy_tangent"], data["c"], numeric_radius_fraction=.999)
        wg, pg = ball_geometry(whole, data["c"]), ball_geometry(proxy, data["c"])
        if not wg["direction_defined"].all() or not pg["direction_defined"].all():
            raise ValueError("Directional audit requires all whole/proxy directions defined")
        data.update(whole_direction=wg["direction"], proxy_direction=pg["direction"], whole_depth=wg["depth"], proxy_depth=pg["depth"])
        classes = np.unique(labels)
        data["classes"] = classes
        base, class_rows, class_arrays = class_geometry(wg["direction"], labels, classes, wg["depth"], data["logits"])
        data["class_centers"] = class_arrays["class_centers"]
        centered, centered_defined = unit(wg["direction"] - class_arrays["common_mean"])
        if centered_defined.all():
            centered_summary, _, _ = class_geometry(centered, labels, classes)
        else:
            centered_summary = {"available": False, "undefined_directions": int((~centered_defined).sum())}
        usage, used = usage_for_epoch(normalized, epoch, pids)
        data["sample_used"] = used
        raw_rows = top_rows(raw_score(proxy, whole, data["c"]), ids, 4)
        direction_rows = top_rows(pg["direction"] @ wg["direction"].T, ids, 4)
        raw_union = np.unique(raw_rows)
        bottom_count = max(1, int(np.ceil(.01 * len(ids))))
        # Depth ties receive stable ascending ID order from cache_load.
        bottom_rows = np.argsort(wg["depth"], kind="stable")[:bottom_count]
        bottom_mask, union_mask = np.ones(len(ids), dtype=bool), np.ones(len(ids), dtype=bool)
        bottom_mask[bottom_rows] = False
        union_mask[raw_union] = False
        controls = {"all": np.ones(len(ids), dtype=bool), "exclude_bottom_radius_1pct": bottom_mask,
                    "exclude_raw_top4_union": union_mask, "exclude_both": bottom_mask & union_mask}
        affinity_summaries, affinity_arrays = {}, {}
        for name, mask in controls.items():
            summary, arrays, _ = proxy_affinity(pg["direction"], wg["direction"], labels, ids, classes, class_arrays, mask)
            affinity_summaries[name], affinity_arrays[name] = summary, arrays
            if name != "all" and arrays:
                affinity_summaries[name]["nearest_class_retention_vs_all"] = float(np.mean(arrays["nearest_class"] == affinity_arrays["all"]["nearest_class"]))
        data["affinity"] = affinity_arrays
        proxy_groups, proxy_group_angles = proxy_direction_groups(pg['direction'], affinity_arrays['all']['nearest_class'], classes, used)
        tangent_norm = np.linalg.norm(data["proxy_tangent"], axis=1)
        usage_summary = {"available": used is not None, "sample_used_count": int(used.sum()) if used is not None else None,
                         "sample_inactive_count": int((~used).sum()) if used is not None else None, "groups": {}}
        if used is not None:
            for name, mask in [("sample_used", used), ("sample_inactive", ~used)]:
                group = {"proxy_count": int(mask.sum()), "origin_depth": stats(pg["depth"][mask]),
                         "nearest_center_angle_deg": stats(affinity_arrays['all']['nearest_center_angle_deg'][mask]),
                         "center_margin": stats(affinity_arrays['all']['center_top2_cosine_margin'][mask]),
                         "nearest_class_counts": {str(scalar(label)): int((affinity_arrays['all']['nearest_class'][mask] == label).sum()) for label in classes}}
                for k in [16, 32, 64]:
                    key = f"top{k}_dominant_purity"
                    if key in affinity_arrays['all']:
                        group[key] = stats(affinity_arrays['all'][key][mask])
                usage_summary["groups"][name] = group
        # Add class nearest-center occupancy/role count, independent of raw NN4.
        for row in class_rows:
            class_label = row["class_label"]
            row["epoch"] = epoch
            if spec.get("class_names") and isinstance(class_label, int) and class_label < len(spec["class_names"]):
                row["class_name"] = spec["class_names"][class_label]
            bound = affinity_arrays['all']['nearest_class'] == class_label
            row["nearest_center_proxy_count"] = int(bound.sum())
            row["nearest_center_sample_used_proxy_count"] = int((bound & used).sum()) if used is not None else None
            row["raw_top4_slots"] = int((labels[raw_rows] == class_label).sum())
            row["direction_top4_slots"] = int((labels[direction_rows] == class_label).sum())
            row["bottom_radius_1pct_count"] = int((labels[bottom_rows] == class_label).sum())
            proxy_record = next((item for item in proxy_groups['groups'] if item['class_label'] == class_label), None)
            row['nearest_center_proxy_group_direction_resultant'] = proxy_record['proxy_direction_resultant'] if proxy_record else None
            row['nearest_center_proxy_group_angle_median_deg'] = proxy_record['proxy_to_group_center_angle_deg'].get('median') if proxy_record else None
        raw_counts = Counter(raw_rows.reshape(-1).tolist())
        ranks = np.empty(len(ids), dtype=np.int64)
        ranks[np.argsort(wg['depth'], kind='stable')] = np.arange(1, len(ids)+1)
        raw_hottest = [{"sample_id": scalar(ids[index]), "class_label": scalar(labels[index]), "slots": count,
                        "origin_depth": float(wg['depth'][index]), "global_depth_rank": int(ranks[index])}
                       for index, count in raw_counts.most_common(20)]
        summary = {"epoch": epoch, "identity": {key: value for key, value in record.items() if key != "cache"},
                   "cache_sha256": data['source_sha'], "sample_pool_sha256": pool_hash(ids), "input_sha256": data['input_sha256'],
                   "whole_count": len(ids), "proxy_count": len(pids), "class_count": len(classes),
                   "whole_origin_depth": stats(wg['depth']), "whole_coordinate_radius": stats(wg['radius']),
                   "proxy_origin_depth": stats(pg['depth']), "proxy_coordinate_radius": stats(pg['radius']),
                   "proxy_saved_tangent_norm": stats(tangent_norm), "whole_directions": base,
                   "proxy_direction_dispersion": proxy_groups,
                   "common_mean_subtracted_direction_control": centered_summary,
                   "proxy_class_affinity": affinity_summaries, "usage": usage_summary,
                   "raw_top4": retrieval_summary(raw_rows, ids, labels, classes),
                   "direction_top4": retrieval_summary(direction_rows, ids, labels, classes),
                   "raw_hotspot_depth_ranks": raw_hottest,
                   "raw_top4_slot_fraction_bottom_radius_1pct": float(np.isin(raw_rows, bottom_rows).mean()),
                   "raw_top4_union_count": len(raw_union), "excluded_bottom_1pct_count": bottom_count}
        proxy_rows = []
        for j, pid in enumerate(pids):
            row = {"epoch": epoch, "proxy_id": scalar(pid), "origin_depth": float(pg['depth'][j]),
                   "saved_tangent_norm": float(tangent_norm[j]), "sample_used": bool(used[j]) if used is not None else None,
                   "angle_to_assigned_proxy_group_center_deg": float(proxy_group_angles[j]) if np.isfinite(proxy_group_angles[j]) else None,
                   "raw_top4_ids": ids[raw_rows[j]].tolist(), "direction_top4_ids": ids[direction_rows[j]].tolist()}
            for control, arrays in affinity_arrays.items():
                for key, values in arrays.items():
                    value = scalar(values[j])
                    row[control + "__" + key] = None if isinstance(value, float) and not np.isfinite(value) else value
            for key, values in usage.items():
                row["usage__" + key] = float(values[j])
            proxy_rows.append(row)
        epoch_dir = out / f"e{epoch:03d}"
        epoch_dir.mkdir()
        (epoch_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        write_csv(epoch_dir / "classes.csv", class_rows)
        write_csv(epoch_dir / "proxies.csv", proxy_rows)
        arrays = {"sample_ids": ids, "labels": labels, "proxy_ids": pids, "classes": classes,
                  "whole_depth": wg['depth'], "whole_radius": wg['radius'], "proxy_depth": pg['depth'],
                  "proxy_direction": pg['direction'], "proxy_group_angle_deg": proxy_group_angles,
                  "proxy_to_whole_class_center_cosine": pg['direction'] @ class_arrays['class_centers'].T,
                  "proxy_to_whole_class_mean_cosine": pg['direction'] @ class_arrays['class_means'].T,
                  "proxy_saved_tangent_norm": tangent_norm, "raw_top4_ids": ids[raw_rows], "direction_top4_ids": ids[direction_rows],
                  "raw_slot_count_per_whole": np.bincount(raw_rows.reshape(-1), minlength=len(ids)),
                  "global_depth_rank": ranks, "bottom_radius_1pct_mask": ~bottom_mask, "raw_top4_union_mask": ~union_mask,
                  "sample_usage_available": np.array(used is not None), "sample_used": used if used is not None else np.full(len(pids), -1, dtype=np.int8),
                  **class_arrays}
        arrays.update({name + '__' + key: value for name, values in affinity_arrays.items() for key, value in values.items()})
        np.savez_compressed(epoch_dir / "arrays.npz", **arrays)
        output_json["snapshots"].append(summary)
        full_class_rows.extend(class_rows)
        full_proxy_rows.extend(proxy_rows)
        if previous is not None:
            print(json.dumps({"phase": "transition", "from_epoch": previous['epoch'], "to_epoch": epoch}), flush=True)
            delta, delta_arrays = transition(previous, data)
            output_json["transitions"].append(delta)
            np.savez_compressed(out / f"transition_{previous['epoch']:03d}_{epoch:03d}.npz", proxy_ids=pids, sample_ids=ids, **delta_arrays)
        previous = data
    if any(sha(path) != expected for path, expected in source_hashes.items()):
        raise RuntimeError("Source cache changed during audit")
    if spec.get('normalized') and sha(spec['normalized']) != normalized_sha:
        raise RuntimeError("Normalized source changed")
    output_json['source_files_unchanged'] = True
    (out / "evolution.json").write_text(json.dumps(output_json, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    write_csv(out / "all_classes.csv", full_class_rows)
    write_csv(out / "all_proxies.csv", full_proxy_rows)
    print(json.dumps({"completed": True, "snapshot_count": len(records), "source_files_unchanged": True}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    analyze(args.spec, args.output)
