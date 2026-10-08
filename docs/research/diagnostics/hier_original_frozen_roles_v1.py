"""Audit archived original HIER frozen-evaluation ancestor roles on CPU.

The input archive stores sample-term deterministic argmin picks, not actual
training activation and not the per-draw Gumbel picks. This script never maps
already mapped ball coordinates again. Full identity artifacts stay private.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


VERSION = "original-frozen-roles-v1"
QUANTILES = (0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def statistics(values):
    x = np.asarray(values, dtype=np.float64)
    if not len(x):
        return {"n": 0}
    q = np.quantile(x, QUANTILES)
    result = {"n": int(len(x)), "mean": float(x.mean()),
              "std_population": float(x.std()),
              "variance_population": float(x.var())}
    result.update({f"q{int(p * 100):02d}": float(v)
                   for p, v in zip(QUANTILES, q)})
    return result


def depth(ball, c):
    r = np.linalg.norm(np.asarray(ball, dtype=np.float64), axis=1)
    rho = np.sqrt(c) * r
    if np.any(rho >= 1) or np.any(rho < 0):
        raise ValueError("Mapped coordinates must be strictly inside the ball")
    return 2 * np.arctanh(rho) / np.sqrt(c)


def make_batches(labels, n_batches, seed):
    """Reproduce the shipped frozen-batch generator, with hash verification."""
    rng = np.random.default_rng(seed)
    classes = np.unique(labels)
    by_class = {cl: np.flatnonzero(labels == cl) for cl in classes}
    batches = []
    for _ in range(n_batches):
        chosen = rng.choice(classes, size=45, replace=False)
        batches.append(np.concatenate(
            [rng.choice(by_class[cl], size=2, replace=True) for cl in chosen]))
    return np.stack(batches)


def role_statistics(pair, triple, proxy_depth):
    pair = np.asarray(pair, dtype=np.int64)
    triple = np.asarray(triple, dtype=np.int64)
    p = len(proxy_depth)
    pair_counts = np.bincount(pair, minlength=p)
    triple_counts = np.bincount(triple, minlength=p)
    pair_set, triple_set = pair_counts > 0, triple_counts > 0
    union, overlap = pair_set | triple_set, pair_set & triple_set
    pd, td = proxy_depth[pair], proxy_depth[triple]
    result = {
        "draws": int(len(pair)),
        "pair_occurrence_weighted_depth": statistics(pd),
        "triple_occurrence_weighted_depth": statistics(td),
        "pair_unique_proxy_depth": statistics(proxy_depth[pair_set]),
        "triple_unique_proxy_depth": statistics(proxy_depth[triple_set]),
        "pair_minus_triple_depth_occurrence_weighted": statistics(pd - td),
        "pair_deeper_fraction": float(np.mean(pd > td)) if len(pd) else None,
        "pair_shallower_fraction": float(np.mean(pd < td)) if len(pd) else None,
        "equal_depth_fraction": float(np.mean(pd == td)) if len(pd) else None,
        "pair_deeper_fraction_gt1e_4": float(np.mean(pd - td > 1e-4)) if len(pd) else None,
        "pair_shallower_fraction_gt1e_4": float(np.mean(td - pd > 1e-4)) if len(pd) else None,
        "depth_difference_within1e_4_fraction": float(np.mean(np.abs(pd - td) <= 1e-4)) if len(pd) else None,
        "pair_proxies": int(pair_set.sum()),
        "triple_proxies": int(triple_set.sum()),
        "overlap_proxies": int(overlap.sum()),
        "union_proxies": int(union.sum()),
        "pair_only_proxies": int((pair_set & ~triple_set).sum()),
        "triple_only_proxies": int((triple_set & ~pair_set).sum()),
        "unused_proxies": int((~union).sum()),
        "role_set_jaccard": float(overlap.sum() / union.sum()) if union.any() else None,
    }
    if len(pair):
        f, g = pair_counts / len(pair), triple_counts / len(pair)
        result["role_distribution_total_variation"] = float(np.abs(f - g).sum() / 2)
        for name, counts in (("pair", pair_counts), ("triple", triple_counts)):
            fr = counts[counts > 0] / counts.sum()
            result[f"{name}_effective_proxy_count"] = float(np.exp(-(fr * np.log(fr)).sum()))
            result[f"{name}_largest_proxy_share"] = float(counts.max() / counts.sum())
    return result, pair_counts, triple_counts


def collision_statistics(pair, triple, include=None):
    if include is None:
        include = np.ones(len(pair), dtype=bool)
    n = int(include.sum())
    collision = int(((pair == triple) & include).sum())
    return {"all_draws": n, "collision_draws": collision,
            "noncollision_draws": n - collision,
            "collision_rate": collision / n if n else None}


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def audit_dataset(root, dataset, output):
    import torch

    cache_path = root / "proxy_eval" / "cache" / f"{dataset}_train.pt"
    source = root / "proxy_eval" / "output" / f"{dataset}_train"
    ck = torch.load(cache_path, map_location="cpu", weights_only=True)
    proxies = ck["hierarchy_proxies"].detach().cpu().numpy()
    whole = ck["embeddings"].detach().cpu().numpy()
    labels = ck["label"].detach().cpu().numpy().astype(np.int64)
    sample_ids = ck["sample_id"].detach().cpu().numpy().astype(np.int64)
    c = float(ck["c"])
    hp, hx = depth(proxies, c), depth(whole, c)
    # The original report computes norm/atanh in FP32. At the clip surface,
    # signs of sub-micro depth differences are reduction/precision dependent.
    hp_fp32 = (2 * torch.atanh((ck["hierarchy_proxies"].float().norm(dim=1)
                               * np.sqrt(c)).clamp(max=1 - 1e-7)) / np.sqrt(c)).numpy()
    manifest = read_json(source / "ancestors_batches.json")
    previous = read_json(source / "ancestors.json")
    graph = read_json(source / "ancestors_graph.json")
    with np.load(source / "ancestors_draws.npz", allow_pickle=False) as z:
        required = {"batch", "a", "j", "k", "pij_argmin", "pijk_argmin"}
        if not required.issubset(z.files):
            raise ValueError("Missing archived deterministic sample-role fields")
        arrays = {k: z[k].astype(np.int64) for k in required}
    if len({len(v) for v in arrays.values()}) != 1:
        raise ValueError("Draw archive columns have different lengths")
    batches = make_batches(labels, int(manifest["n_batches"]), int(manifest["seed"]))
    hashes = [hashlib.md5(b.tobytes()).hexdigest()[:12] for b in batches]
    if hashes != manifest["batch_hashes"]:
        raise ValueError("Reconstructed frozen-batch identities differ from archive")
    bid, pair, triple = arrays["batch"], arrays["pij_argmin"], arrays["pijk_argmin"]
    if (bid.min() < 0 or bid.max() >= len(batches) or pair.min() < 0
            or triple.min() < 0 or pair.max() >= len(proxies)
            or triple.max() >= len(proxies)):
        raise ValueError("Archive index outside available cache/batches")
    endpoint_rows = {key: batches[bid, arrays[key]] for key in ("a", "j", "k")}
    valid = pair != triple
    all_stats, all_pair, all_triple = role_statistics(pair, triple, hp)
    valid_stats, vp, vt = role_statistics(pair[valid], triple[valid], hp)
    fp32_valid_stats, _, _ = role_statistics(pair[valid], triple[valid], hp_fp32)
    coll_stats = collision_statistics(pair, triple)

    combo_labels = {key: labels[rows] for key, rows in endpoint_rows.items()}
    all_same = ((combo_labels["a"] == combo_labels["j"])
                & (combo_labels["a"] == combo_labels["k"]))
    ij_same = combo_labels["a"] == combo_labels["j"]
    combos = {"all_same": all_same, "ij_same_k_diff": ij_same & ~all_same,
              "other": ~ij_same}
    combo_summary = {}
    for name, mask in combos.items():
        collision = collision_statistics(pair, triple, mask)
        role, _, _ = role_statistics(pair[mask & valid], triple[mask & valid], hp)
        combo_summary[name] = {"collision": collision, "noncollision_roles": role,
                               "old_reported_collision_rate": previous.get(
                                   "by_class_combo_argmin", {}).get(name, {}).get("collision_rate")}
    identity_self_k = sample_ids[endpoint_rows["a"]] == sample_ids[endpoint_rows["k"]]
    position_self_k = arrays["a"] == arrays["k"]
    identities_distinct = ((sample_ids[endpoint_rows["a"]] != sample_ids[endpoint_rows["j"]])
                           & (sample_ids[endpoint_rows["a"]] != sample_ids[endpoint_rows["k"]])
                           & (sample_ids[endpoint_rows["j"]] != sample_ids[endpoint_rows["k"]]))

    # Reconcile archived argmin edges with the original graph's weighted counts.
    edges, edge_counts = np.unique(np.stack([pair[valid], triple[valid]], axis=1),
                                   axis=0, return_counts=True)
    source_edges = {(int(a), int(b)): int(w) for a, b, w in graph["edges"]}
    actual_edges = {(int(a), int(b)): int(w) for (a, b), w in zip(edges, edge_counts)}
    if actual_edges != source_edges:
        raise ValueError("Archived deterministic role picks do not reproduce graph edges")
    delta_edges = hp[edges[:, 0]] - hp[edges[:, 1]]
    # The shipped graph depth mean is unweighted over unique edges, unlike draws.
    edge_report = {"unique_edges": int(len(edges)), "draw_weight_total": int(edge_counts.sum()),
                   "unique_edge_equal_weight_depth_difference": statistics(delta_edges),
                   "unique_edge_pair_deeper_fraction": float(np.mean(delta_edges > 0)),
                   "draw_weighted_depth_difference_mean": float(np.average(delta_edges, weights=edge_counts)),
                   "draw_weighted_pair_deeper_fraction": float(np.average(delta_edges > 0, weights=edge_counts)),
                   "shipped_graph_report": graph["report"]}

    per_batch = []
    for batch in range(len(batches)):
        selected = bid == batch
        roles, _, _ = role_statistics(pair[selected & valid], triple[selected & valid], hp)
        row = {"dataset": dataset, "batch": batch,
               **collision_statistics(pair, triple, selected),
               "pair_proxies_noncollision": roles["pair_proxies"],
               "triple_proxies_noncollision": roles["triple_proxies"],
               "pair_deeper_fraction_noncollision": roles["pair_deeper_fraction"],
               "pair_minus_triple_mean_noncollision": roles[
                   "pair_minus_triple_depth_occurrence_weighted"].get("mean")}
        per_batch.append(row)
    role_rows = [{"dataset": dataset, "proxy_id": int(i), "depth": float(hp[i]),
                  "pair_all_draws": int(all_pair[i]), "triple_all_draws": int(all_triple[i]),
                  "pair_noncollision_draws": int(vp[i]), "triple_noncollision_draws": int(vt[i]),
                  "pair_only_noncollision": bool(vp[i] > 0 and vt[i] == 0),
                  "triple_only_noncollision": bool(vt[i] > 0 and vp[i] == 0),
                  "both_roles_noncollision": bool(vp[i] > 0 and vt[i] > 0)}
                 for i in range(len(hp))]
    role_summary_rows = []
    for denominator, roles in (("all_draws_including_collision", all_stats),
                               ("noncollision_draws", valid_stats)):
        for role in ("pair", "triple"):
            for weighting in ("occurrence_weighted", "unique_proxy"):
                stat = roles[f"{role}_{weighting}_depth"]
                role_summary_rows.append({"dataset": dataset, "denominator": denominator,
                                          "role": role, "weighting": weighting, **stat})
    proxy_unique_same_count = np.array([r["pair_proxies_noncollision"] for r in per_batch]).mean()
    triple_unique_same_count = np.array([r["triple_proxies_noncollision"] for r in per_batch]).mean()
    pair_depth, triple_depth = hp[pair[valid]], hp[triple[valid]]
    a_h, j_h, k_h = (hx[endpoint_rows[key][valid]] for key in ("a", "j", "k"))
    endpoint_summary = {
        "pair_minus_shallower_ij_endpoint_depth": statistics(pair_depth - np.minimum(a_h, j_h)),
        "triple_minus_shallowest_ijk_endpoint_depth": statistics(triple_depth - np.minimum(np.minimum(a_h, j_h), k_h)),
        "pair_shallower_than_both_ij_fraction": float(np.mean(pair_depth < np.minimum(a_h, j_h))),
        "triple_shallower_than_all_ijk_fraction": float(np.mean(triple_depth < np.minimum(np.minimum(a_h, j_h), k_h))),
    }
    summary = {
        "dataset": dataset, "checkpoint_epoch": int(ck["checkpoint_epoch"]),
        "checkpoint_sha256": ck["checkpoint_hash"], "c": c,
        "whole_count": int(len(whole)), "proxy_count": int(len(hp)),
        "whole_depth": statistics(hx), "proxy_depth": statistics(hp),
        "draw_archive_schema": {key: {"shape": list(v.shape), "dtype": str(v.dtype)}
                                for key, v in sorted(arrays.items())},
        "frozen_batch_identity_hashes_verified": True,
        "graph_edges_and_occurrence_counts_verified": True,
        "sample_argmin_collision": coll_stats,
        "sample_argmin_all_draws_roles": all_stats,
        "sample_argmin_noncollision_roles": valid_stats,
        "sample_argmin_local_cpu_fp32_depth_sign_sensitivity": {
            "pair_deeper_fraction_strict_fp32": fp32_valid_stats["pair_deeper_fraction"],
            "pair_deeper_fraction_strict_float64_from_stored_coordinates": valid_stats["pair_deeper_fraction"],
            "pair_deeper_fraction_gt1e_4_float64": valid_stats["pair_deeper_fraction_gt1e_4"],
            "pair_shallower_fraction_gt1e_4_float64": valid_stats["pair_shallower_fraction_gt1e_4"],
            "depth_difference_within1e_4_fraction_float64": valid_stats["depth_difference_within1e_4_fraction"],
            "note": "Strict depth-difference signs near a common clip surface can change with FP32 norm reduction. A tiny positive sign is not evidence of a substantive radial layer."},
        "sample_argmin_endpoint_depth_comparison_noncollision": endpoint_summary,
        "sample_argmin_by_class_combo": combo_summary,
        "sample_argmin_self_k": {
            "position_self_k_draws": int(position_self_k.sum()),
            "same_instance_self_k_draws": int(identity_self_k.sum()),
            "three_distinct_instance_draws": int(identities_distinct.sum()),
            "all_draws": int(len(pair))},
        "sample_argmin_mean_batch_usage": {"pair": float(proxy_unique_same_count),
                                           "triple": float(triple_unique_same_count)},
        "sample_argmin_graph": edge_report,
        "shipped_sample_term_report": previous["sample_term"],
        "shipped_proxy_term_report_unverified_per_draw": previous["proxy_term_batch0"],
        "role_limitations": [
            "All picks are clean frozen-evaluation diagnostics; no actual training epoch activation is supplied.",
            "Only sample-term deterministic argmin per-draw picks are archived.",
            "Three-seed Gumbel role statistics are only aggregate JSON; per-proxy counts, role overlap and depth medians cannot be recovered from those aggregates.",
            "Proxy-term per-draw triplets/picks are absent; original JSON gives depth-difference summaries and usage counts but no role-wise distributions.",
            "Sample all-batch union is not the mean batch usage, and both are distinct from actual training activation.",
            "Mapped cache radii are used directly; no second expmap or tangent cap is applied.",
        ],
        "corrected_denominators": [
            "Class-combo collision_rate = collisions within that combo / all draws within that combo. The shipped analyzer line 188 uses the complement of combo AND noncollision mask instead.",
            "Overall frozen occurrence statistics pool all draws; shipped sample JSON generally averages per-batch means.",
            "Graph shipped depth statistics use equal weight per unique directed edge; the new draw-weighted statistic retains edge occurrence weights.",
        ],
        "input_sha256": {p.name: sha256(p) for p in (
            cache_path, source / "ancestors_draws.npz", source / "ancestors_batches.json",
            source / "ancestors.json", source / "ancestors_graph.json")},
    }
    destination = output / dataset
    destination.mkdir()
    write_csv(destination / "proxy_roles.csv", role_rows)
    write_csv(destination / "batch_metrics.csv", per_batch)
    write_csv(destination / "role_depth_metrics.csv", role_summary_rows)
    (destination / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    np.savez_compressed(destination / "plot_arrays.npz", proxy_depth=hp,
                        pair_noncollision_count=vp, triple_noncollision_count=vt,
                        pair_noncollision_depth=pair_depth, triple_noncollision_depth=triple_depth)
    return summary, {"proxy_depth": hp, "pair_counts": vp, "triple_counts": vt,
                     "pair_depth": pair_depth, "triple_depth": triple_depth}


def render(output, summaries, arrays):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, len(arrays), figsize=(15, 8), constrained_layout=True)
    for col, (ds, data) in enumerate(arrays.items()):
        hp = data["proxy_depth"]
        bins = np.linspace(float(hp.min()), float(hp.max()) + 1e-9, 35)
        axes[0, col].hist(data["pair_depth"], bins=bins, density=True,
                          alpha=.55, color="#2563eb", label="pair: occurrence weighted")
        axes[0, col].hist(data["triple_depth"], bins=bins, density=True,
                          alpha=.55, color="#ea580c", label="triple: occurrence weighted")
        roles = summaries[ds]["sample_argmin_noncollision_roles"]
        axes[0, col].set_title(f"{ds.upper()}: frozen sample argmin\n"
                               f"pair deeper by >1e-4: {roles['pair_deeper_fraction_gt1e_4']:.1%}")
        axes[0, col].set_xlabel("Proxy depth (hyperbolic distance to origin)")
        axes[0, col].set_ylabel("Density of noncollision role occurrences")
        axes[0, col].legend(fontsize=8)
        vp, vt = data["pair_counts"], data["triple_counts"]
        axes[1, col].hist(hp[vp > 0], bins=bins, density=True,
                          alpha=.55, color="#2563eb", label="pair: each proxy once")
        axes[1, col].hist(hp[vt > 0], bins=bins, density=True,
                          alpha=.55, color="#ea580c", label="triple: each proxy once")
        axes[1, col].set_title(f"Used proxy ID overlap {roles['overlap_proxies']}/"
                               f"{roles['union_proxies']} union")
        axes[1, col].set_xlabel("Proxy depth (hyperbolic distance to origin)")
        axes[1, col].set_ylabel("Density across unique proxy IDs")
        axes[1, col].legend(fontsize=8)
    fig.suptitle("Original HIER sample ancestor roles: occurrence and unique-ID distributions\n"
                 "40 clean frozen-eval batches; deterministic argmin; not actual training activation", fontsize=14)
    fig.savefig(output / "sample_argmin_role_depth.png", dpi=170)
    fig.savefig(output / "sample_argmin_role_depth.svg")
    plt.close(fig)

    fig, axes = plt.subplots(1, len(arrays), figsize=(15, 4.5), constrained_layout=True)
    for ax, (ds, data) in zip(axes, arrays.items()):
        vp, vt = data["pair_counts"], data["triple_counts"]
        total = vp + vt
        used = total > 0
        ax.scatter(data["proxy_depth"][used], vp[used] / total[used],
                   s=10 + 100 * total[used] / total.max(), alpha=.6, color="#6366f1")
        ax.axhline(.5, linestyle="--", color="gray", linewidth=1)
        ax.set_ylim(-.05, 1.05)
        ax.set_xlabel("Proxy depth")
        ax.set_ylabel("Pair share of pair+triple occurrences")
        ax.set_title(ds.upper())
    fig.suptitle("Same proxy can serve both roles; marker area reflects usage\n"
                 "Frozen sample-term argmin, noncollision draws only")
    fig.savefig(output / "sample_argmin_proxy_role_mixture.png", dpi=170)
    fig.savefig(output / "sample_argmin_proxy_role_mixture.svg")
    plt.close(fig)


def self_test():
    hp = np.array([1., 3., 6.])
    pair, triple = np.array([0, 1, 1, 2]), np.array([1, 0, 0, 2])
    valid = pair != triple
    role, pc, tc = role_statistics(pair[valid], triple[valid], hp)
    assert role["pair_occurrence_weighted_depth"]["mean"] == 7 / 3
    assert role["pair_unique_proxy_depth"]["mean"] == 2
    assert role["pair_deeper_fraction"] == 2 / 3
    assert role["overlap_proxies"] == 2 and role["union_proxies"] == 2
    assert pc.sum() == tc.sum() == valid.sum()
    # One combo contains only the collided row, another all valid rows.
    mask = np.array([False, False, False, True])
    assert collision_statistics(pair, triple, mask)["collision_rate"] == 1
    assert collision_statistics(pair, triple)["collision_rate"] == .25
    radii = np.tanh(hp / 2)
    assert np.allclose(depth(radii[:, None], 1), hp)
    print("Four mathematical/accounting checks passed")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path)
    ap.add_argument("--output", type=Path)
    ap.add_argument("--datasets", nargs="+", default=["cub", "cars", "sop"])
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        self_test()
        return
    if args.root is None or args.output is None:
        ap.error("--root and --output are required except with --self-test")
    import torch
    torch.set_num_threads(2)
    args.output.mkdir(parents=True, exist_ok=False)
    summaries, arrays = {}, {}
    for ds in args.datasets:
        summaries[ds], arrays[ds] = audit_dataset(args.root, ds, args.output)
        r = summaries[ds]["sample_argmin_noncollision_roles"]
        print(json.dumps({"dataset": ds, "collision": summaries[ds]["sample_argmin_collision"],
                          "pair_deeper": r["pair_deeper_fraction"],
                          "pair_median": r["pair_occurrence_weighted_depth"]["q50"],
                          "triple_median": r["triple_occurrence_weighted_depth"]["q50"],
                          "pair_ids": r["pair_proxies"], "triple_ids": r["triple_proxies"],
                          "overlap": r["overlap_proxies"]}), flush=True)
    render(args.output, summaries, arrays)
    report = {"version": VERSION, "execution": "CPU, existing mapped caches and archived picks only",
              "datasets": summaries}
    (args.output / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved frozen ancestor review: {args.output}")


if __name__ == "__main__":
    main()
