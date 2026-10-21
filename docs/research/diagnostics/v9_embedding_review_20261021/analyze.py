"""Anonymous, NumPy-only analysis of already exported V9 audit embeddings.

Usage: python analyze.py --export-root PRIVATE_EXPORT --out PUBLIC_AGGREGATES
Reads export_manifest.json; never opens checkpoints or calls a model.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


QUANTILES = [("min", 0), ("p01", .01), ("p05", .05), ("p25", .25),
             ("p50", .5), ("p75", .75), ("p95", .95), ("p99", .99), ("max", 1)]
CAP_DEPTH_TOLERANCE = 1e-3
DEPTH_ORDER_TOLERANCE = 1e-4


def describe(values):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if not len(values):
        return None
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite scalar input")
    result = {"n": int(len(values)), "mean": float(values.mean()),
              "std": float(values.std()), "variance": float(values.var())}
    result.update({name: float(np.quantile(values, q)) for name, q in QUANTILES})
    return result


def directions(points):
    points = np.asarray(points, dtype=np.float64)
    radii = np.linalg.norm(points, axis=1)
    if np.any(radii <= 1e-12):
        raise ValueError("A zero vector has no defined direction")
    return points / radii[:, None]


def radial(points, c):
    if not np.isfinite(c) or c <= 0:
        raise ValueError("Curvature magnitude must be positive")
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or not np.isfinite(points).all():
        raise ValueError("Expected finite [N,D] embeddings")
    q = np.sqrt(c) * np.linalg.norm(points, axis=1)
    if np.any(q >= 1) or np.any(q < 0):
        raise ValueError("An embedding lies outside its declared curvature ball")
    return 2 / np.sqrt(c) * np.arctanh(q), q


def usage_metrics(counts):
    if counts is None:
        return None
    counts = np.asarray(counts, dtype=np.float64)
    total = counts.sum()
    p = counts[counts > 0] / total if total else np.empty(0)
    return {"used": int(np.count_nonzero(counts)), "total_selections": float(total),
            "effective_count": float(np.exp(-(p * np.log(p)).sum())) if total else 0.,
            "maximum_proxy_selection_share": float(counts.max() / total) if total else None}


def nearest_centers(whole_dirs, labels, proxy_dirs):
    classes = np.unique(labels)
    centers = np.stack([whole_dirs[labels == k].mean(axis=0) for k in classes])
    norm = np.linalg.norm(centers, axis=1)
    if np.any(norm <= 1e-12):
        raise ValueError("A class direction center is undefined")
    centers /= norm[:, None]
    cosine = np.clip(proxy_dirs @ centers.T, -1, 1)
    nearest = cosine.argmax(axis=1)
    angles = np.degrees(np.arccos(cosine[np.arange(len(proxy_dirs)), nearest]))
    return classes[nearest], angles, centers


def direction_metrics(unit_vectors):
    return {"R": float(np.linalg.norm(unit_vectors.mean(axis=0))),
            "mean_pairwise_cosine_off_diagonal": float(
                ((unit_vectors.sum(axis=0) ** 2).sum() - len(unit_vectors)) /
                (len(unit_vectors) * (len(unit_vectors) - 1))) if len(unit_vectors) > 1 else None}


def retrieval(indices, depth, labels):
    indices = np.asarray(indices)
    if indices.ndim != 2 or indices.shape[1] != 4:
        raise ValueError("NN4 must be [P,4]")
    if not np.issubdtype(indices.dtype, np.integer):
        raise ValueError("NN4 must contain integer row indices")
    if np.any(indices < 0) or np.any(indices >= len(depth)):
        raise ValueError("NN4 row outside the exported whole pool")
    if any(len(np.unique(row)) != 4 for row in indices):
        raise ValueError("NN4 must refer to four distinct object rows per proxy")
    counts = np.bincount(indices.ravel(), minlength=len(depth))
    _, quartet_counts = np.unique(np.sort(indices, axis=1), axis=0, return_counts=True)
    bottom_n = max(1, int(math.ceil(.01 * len(depth))))
    # Stable row order resolves ties, but the tie count is reported explicitly.
    bottom = np.argsort(depth, kind="stable")[:bottom_n]
    boundary = float(np.max(depth[bottom]))
    top = np.argsort(-counts, kind="stable")[:4]
    same_class = np.all(labels[indices] == labels[indices[:, :1]], axis=1)
    report = {"covered_samples": int(np.count_nonzero(counts)),
              "coverage_fraction_of_pool": float(np.count_nonzero(counts) / len(depth)),
              "slots": int(indices.size), "maximum_repeated_quartet": int(quartet_counts.max()),
              "distinct_quartets": int(len(quartet_counts)),
              "top4_sample_slot_share": float(counts[top].sum() / indices.size),
              "bottom_depth_1pct_slot_share": float(counts[bottom].sum() / indices.size),
              "bottom_depth_1pct_n": bottom_n,
              "bottom_depth_boundary_tie_n": int(np.count_nonzero(np.abs(depth - boundary) < 1e-6)),
              "whole_depth_span": float(np.ptp(depth)),
              "bottom_1pct_meaningful_radial_order": bool(np.ptp(depth) > 1e-3),
              "same_class_quartet_proxy_count": int(same_class.sum()),
              "max_object_slot_count": int(counts.max())}
    return report, counts


def csv_write(path, rows):
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    root, out = args.export_root.resolve(), args.out
    manifest = json.loads((root / "export_manifest.json").read_text(encoding="utf-8-sig"))
    out.mkdir(parents=True, exist_ok=True)
    rows, class_rows, proxy_rows, hotspot_rows, reports, plot = [], [], [], [], [], {}
    seen = set()
    identity_reference = None
    for spec in manifest["observations"]:
        key = str(spec["key"])
        if key in seen or not key.replace("_", "").replace("-", "").isalnum():
            raise ValueError("Keys must be unique and use letters/digits/underscore/hyphen")
        seen.add(key)
        cache = (root / spec["cache"]).resolve()
        if not cache.is_relative_to(root):
            raise ValueError("Cache must lie inside export root")
        with np.load(cache, allow_pickle=False) as saved:
            native = np.asarray(saved["native_mu"], dtype=np.float64)
            labels = np.asarray(saved["labels"]).reshape(-1)
            ids = np.asarray(saved["canonical_ids"]).reshape(-1)
            if len(native) != len(labels) or len(ids) != len(native) or len(np.unique(ids)) != len(ids):
                raise ValueError("Whole/label/canonical identity count mismatch or duplicate identities")
            # Identity rows may have different order between original and versioned loaders.
            ordered = np.argsort(ids)
            identity = (ids[ordered], labels[ordered])
            if identity_reference is None:
                identity_reference = identity
            elif not all(np.array_equal(x, y) for x, y in zip(identity_reference, identity)):
                raise ValueError("Observations must contain the same canonical objects and labels")
            c_native = float(spec["c_native"])
            base = {k: spec.get(k) for k in ("key", "label", "stage", "epoch", "kind")}
            result = {**base, "c_native": c_native, "c_hier": spec.get("c_hier"),
                      "checkpoint_sha256": spec.get("checkpoint_sha256"),
                      "N": len(native), "D": native.shape[1], "populations": {},
                      "proxy": None, "retrieval": None}

            def population(domain, name, points, c, cap_tangent=None, group="all"):
                depth, q = radial(points, c)
                cap_depth = None if cap_tangent is None else 2 * float(cap_tangent)
                desc = {"depth": describe(depth), "q": describe(q), "cap_depth": cap_depth,
                        "at_cap_fraction": float(np.mean(np.abs(depth - cap_depth) <= CAP_DEPTH_TOLERANCE))
                        if cap_depth is not None and len(depth) else None}
                result["populations"][f"{domain}__{name}__{group}"] = desc
                for metric, array in (("depth", depth), ("q", q)):
                    value = describe(array)
                    if value:
                        rows.append({**base, "domain": domain, "c": c, "population": name,
                                     "group": group, "metric": metric, **value,
                                     "cap_depth": cap_depth,
                                     "at_cap_fraction": desc["at_cap_fraction"]})
                    plot[f"{key}__{domain}__{name}__{group}__{metric}"] = array.astype(np.float64)
                return depth, q

            native_depth, native_q = population("native", "whole", native, c_native,
                                               spec.get("native_cap_tangent"))
            native_dirs = directions(native)
            result["native_whole_directions"] = direction_metrics(native_dirs)
            for cls in np.unique(labels):
                mask = labels == cls
                class_rows.append({**base, "domain": "native", "c": c_native, "class": int(cls),
                                   **describe(native_depth[mask]),
                                   "q_mean": float(native_q[mask].mean()),
                                   "direction_R": float(np.linalg.norm(native_dirs[mask].mean(axis=0)))})
            if "proxy_ball" not in saved.files:
                if "hier_mu" in saved.files or "proxy_tangent" in saved.files:
                    raise ValueError("Partial HIER export: proxy_ball is missing")
                reports.append(result)
                continue
            for required in ("hier_mu", "proxy_tangent"):
                if required not in saved.files:
                    raise ValueError(f"Partial HIER export: {required} missing")
            c_hier = float(spec["c_hier"])
            whole = np.asarray(saved["hier_mu"], dtype=np.float64)
            proxies = np.asarray(saved["proxy_ball"], dtype=np.float64)
            tangent = np.asarray(saved["proxy_tangent"], dtype=np.float64)
            if whole.shape != native.shape or tangent.shape != proxies.shape or proxies.shape[1] != whole.shape[1]:
                raise ValueError("HIER export tensor shape mismatch")
            hier_cap = spec.get("hier_cap_tangent", 2.3)
            whole_depth, _ = population("hier", "whole", whole, c_hier,
                                       spec.get("hier_whole_cap_tangent", hier_cap))
            proxy_depth, proxy_q = population("hier", "proxy", proxies, c_hier, hier_cap)
            tangent_norm = np.linalg.norm(tangent, axis=1)
            plot[f"{key}__hier__proxy__all__tangentnorm"] = tangent_norm
            proxy_dirs = directions(proxies)
            whole_dirs = directions(whole)
            nearest_class, angles, centers = nearest_centers(whole_dirs, labels, proxy_dirs)
            tangent_stat = describe(tangent_norm)
            rows.append({**base, "domain": "hier", "c": c_hier, "population": "proxy",
                         "group": "all", "metric": "tangent_norm", **tangent_stat})
            counts = {}
            epoch = spec.get("epoch")
            usage_epoch = spec.get("usage_source_epoch", epoch)
            if usage_epoch != epoch:
                raise ValueError("Usage statistics must come from the checkpoint's own epoch")
            for role in ("sample", "proxy", "sample_effective", "proxy_effective"):
                field = role + "_usage"
                value = np.asarray(saved[field], dtype=np.float64).reshape(-1) if field in saved.files else None
                if value is not None and (len(value) != len(proxies) or not np.isfinite(value).all() or (value < 0).any()):
                    raise ValueError("Invalid per-proxy usage")
                counts[role] = value
                if value is not None:
                    plot[f"{key}__hier__proxy__all__{role}_usage"] = value
            union = None if any(counts[role] is None for role in ("sample", "proxy")) else counts["sample"] + counts["proxy"]
            counts["combined"] = union
            depth_delta = proxy_depth - np.median(whole_depth)
            groups = {"all": np.ones(len(proxies), dtype=bool),
                      "shallower_than_whole_median": depth_delta < -DEPTH_ORDER_TOLERANCE,
                      "same_depth_as_whole_median": np.abs(depth_delta) <= DEPTH_ORDER_TOLERANCE,
                      "deeper_than_whole_median": depth_delta > DEPTH_ORDER_TOLERANCE}
            for role, value in counts.items():
                if value is not None:
                    groups[role + "_used"] = value > 0
                    groups[role + "_unused"] = value == 0
            group_reports = {}
            for group, mask in groups.items():
                group_reports[group] = {"n": int(mask.sum()), "depth": describe(proxy_depth[mask]),
                                        "tangent_norm": describe(tangent_norm[mask]),
                                        "nearest_class_angle_deg": describe(angles[mask]),
                                        "direction_R": float(np.linalg.norm(proxy_dirs[mask].mean(axis=0))) if mask.any() else None,
                                        "sample_usage": usage_metrics(counts["sample"][mask]) if counts["sample"] is not None and mask.any() else None,
                                        "proxy_usage": usage_metrics(counts["proxy"][mask]) if counts["proxy"] is not None and mask.any() else None,
                                        "sample_effective_usage": usage_metrics(counts["sample_effective"][mask]) if counts["sample_effective"] is not None and mask.any() else None,
                                        "proxy_effective_usage": usage_metrics(counts["proxy_effective"][mask]) if counts["proxy_effective"] is not None and mask.any() else None}
                if group != "all":
                    population("hier", "proxy", proxies[mask], c_hier, hier_cap, group)
            proxy_report = {"P": len(proxies), "saved_tangent_norm": tangent_stat,
                            "tangent_over_cap_count": int(np.count_nonzero(tangent_norm > float(hier_cap) + 1e-6)) if hier_cap is not None else None,
                            "direction": direction_metrics(proxy_dirs),
                            "whole_class_balanced_direction_R": float(np.linalg.norm(centers.mean(axis=0))),
                            "usage_available": {k: v is not None for k, v in counts.items()},
                            "usage": {k: usage_metrics(v) for k, v in counts.items()},
                            "groups": group_reports,
                            "proxy_depth_minus_hier_whole_median": describe(depth_delta),
                            "hier_whole_depth_span": float(np.ptp(whole_depth))}
            for p in range(len(proxies)):
                proxy_rows.append({**base, "proxy_index": p, "c": c_hier,
                                   "depth": float(proxy_depth[p]), "q": float(proxy_q[p]),
                                   "saved_tangent_norm": float(tangent_norm[p]),
                                   "depth_minus_hier_whole_median": float(depth_delta[p]),
                                   "nearest_direction_class": int(nearest_class[p]),
                                   "nearest_class_angle_deg": float(angles[p]),
                                   "sample_usage": float(counts["sample"][p]) if counts["sample"] is not None else None,
                                   "proxy_usage": float(counts["proxy"][p]) if counts["proxy"] is not None else None,
                                   "sample_effective_usage": float(counts["sample_effective"][p]) if counts["sample_effective"] is not None else None,
                                   "proxy_effective_usage": float(counts["proxy_effective"][p]) if counts["proxy_effective"] is not None else None})
            retrieval_report = {}
            for field, label in (("nn4", "raw"), ("nn4_equal", "equal_whole_depth"), ("nn4_angle", "angular")):
                if field not in saved.files:
                    retrieval_report[label] = None
                    continue
                report, slots = retrieval(saved[field], whole_depth, labels)
                retrieval_report[label] = report
                # Full object identity is intentionally omitted, including source row index.
                selected = np.flatnonzero(slots)
                order = selected[np.argsort(-slots[selected], kind="stable")]
                for rank, row in enumerate(order, 1):
                    hotspot_rows.append({**base, "retrieval": label, "hotspot_rank": rank,
                                         "class": int(labels[row]), "slot_count": int(slots[row]),
                                         "slot_fraction": float(slots[row] / np.asarray(saved[field]).size),
                                         "hier_whole_depth": float(whole_depth[row]),
                                         "native_whole_depth": float(native_depth[row])})
            result["proxy"], result["retrieval"] = proxy_report, retrieval_report
            reports.append(result)
    summary = {"format": "v9-embedding-anonymous-numpy-review-v1", "observations": reports,
               "checks": {"same_canonical_object_pool_and_labels": True,
                          "embedding_or_sample_identity_exported": False,
                          "model_forward_calls": 0, "optimizer_updates": 0},
               "definitions": {"std": "population standard deviation (ddof=0)",
                               "cap_depth_tolerance": CAP_DEPTH_TOLERANCE,
                               "depth_order_tolerance": DEPTH_ORDER_TOLERANCE,
                               "proxy_usage": "same-epoch all_draws pair+triple selection counts; missing remains null; counts precede and span updates, not frozen-cache reselection",
                               "effective_usage": "same-epoch active_noncollision pair+triple selection counts; positive hinge and no collision, not proof of nonzero radial gradient",
                               "deep_shallow_reference": "same-domain HIER whole median, never native c1 whole",
                               "nearest_class": "maximum cosine to normalized mean unit whole direction per class",
                               "bottom1pct_ties": "stable input row tie break; radial ordering meaningless if depth span <= 1e-3"}}
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    for filename, table in (("stats.csv", rows), ("class_stats.csv", class_rows),
                            ("proxy_stats.csv", proxy_rows), ("hotspots.csv", hotspot_rows)):
        csv_write(out / filename, table)
    assert all(value.ndim == 1 for value in plot.values())
    np.savez_compressed(out / "plot_data.npz", **plot)
    print(json.dumps({"complete": True, "observations": len(reports),
                      "stats_rows": len(rows), "class_rows": len(class_rows),
                      "proxy_rows": len(proxy_rows), "hotspot_rows": len(hotspot_rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
