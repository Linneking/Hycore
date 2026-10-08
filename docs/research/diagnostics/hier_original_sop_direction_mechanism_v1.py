"""Frozen CPU directional mechanism audit for original HIER best caches.

This is an intervention on stored coordinates, not model continuation or a
training-benefit estimate. Full per-object identities remain in private output.
The only expected input is the validated original_hier_full_geometry export.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
from collections import Counter
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hier_original_vs_v7_geometry_v1 import (
    unit_rows, radial_coordinates, stats, finite_json, stable_topk,
)

DATASETS = ("cub", "cars", "sop")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def angle_to_center(u, center):
    norm = np.linalg.norm(center)
    if norm < 1e-12:
        return np.full(len(u), np.nan)
    return np.degrees(np.arccos(np.clip(u @ (center / norm), -1, 1)))


def eigen_summary(eigen):
    eigen = np.maximum(np.asarray(eigen, float), 0)
    eigen = np.sort(eigen)[::-1]
    total = float(eigen.sum())
    if total < 1e-12:
        return {"entropy_effective_rank": None, "participation_effective_rank": None,
                "numerical_rank_rel_1e8": 0, "top1_share": None, "top5_share": None,
                "top20_share": None, "shares": np.zeros_like(eigen), "status": "zero directional variance"}
    shares = eigen / total
    positive = shares > 1e-14
    return {"entropy_effective_rank": float(np.exp(-(shares[positive] * np.log(shares[positive])).sum())),
            "participation_effective_rank": float(1 / np.square(shares).sum()),
            "numerical_rank_rel_1e8": int((shares > 1e-8).sum()),
            "top1_share": float(shares[0]), "top5_share": float(shares[:5].sum()),
            "top20_share": float(shares[:20].sum()), "shares": shares}


def direction_summary(u, pair_distribution=False):
    mean = u.mean(axis=0)
    R = float(np.linalg.norm(mean))
    gram = u.T @ u / len(u)
    centered = gram - np.outer(mean, mean)
    out = {"n": len(u), "R": R,
           "mean_pair_cos_excluding_self": (len(u) * R * R - 1) / (len(u) - 1) if len(u) > 1 else None,
           "uncentered_spectrum": eigen_summary(np.linalg.eigvalsh(gram)),
           "centered_spectrum": eigen_summary(np.linalg.eigvalsh(centered))}
    if pair_distribution and len(u) > 1:
        g = u @ u.T
        out["pair_cosine_distribution"] = stats(g[np.triu_indices(len(u), 1)])
    return out


def coverage(top, n, labels=None):
    slots = np.bincount(top[:, :4].ravel(), minlength=n)
    quartets = Counter(tuple(sorted(x)) for x in top[:, :4].tolist())
    hot = np.argsort(-slots, kind="stable")
    result = {"unique_top4_objects": int((slots > 0).sum()), "distinct_top4_quartets": len(quartets),
            "max_repeated_quartet": max(quartets.values()),
            "hottest4_slot_fraction": float(slots[hot[:4]].sum() / slots.sum()),
            "hottest16_slot_fraction": float(slots[hot[:16]].sum() / slots.sum()),
            "max_object_slots": int(slots.max())}
    if labels is not None:
        nnlabels = np.asarray(labels)[top[:, :4]]
        result["mean_fine_class_top4_purity"] = float(np.mean([max(Counter(row).values()) / 4 for row in nnlabels]))
        result["proxy_count_all4_same_fine_class"] = int(np.all(nnlabels == nnlabels[:, :1], axis=1).sum())
    return result, slots


def directional_top(v, u, ids, k=16):
    # Float64 normalized coordinates, top-k tie resolution by stable sample ID.
    top = np.empty((len(v), k), int)
    for lo in range(0, len(v), 32):
        hi = min(lo + 32, len(v))
        top[lo:hi] = stable_topk(-(v[lo:hi] @ u.T), ids, k)
    return top


def same_radius_raw_top(v, radii, x, c, ids, k=16):
    """Exact-distance rank uses increasing acosh argument; no acosh is needed."""
    p = v * radii[:, None]
    p2 = np.square(radii)
    x2 = np.einsum("ij,ij->i", x, x)
    top = np.empty((len(p), k), int)
    for lo in range(0, len(p), 32):
        hi = min(lo + 32, len(p))
        delta2 = np.maximum(p2[lo:hi, None] + x2[None, :] - 2 * p[lo:hi] @ x.T, 0)
        score = delta2 / ((1 - c * p2[lo:hi, None]) * (1 - c * x2[None, :]))
        top[lo:hi] = stable_topk(score, ids, k)
    return top


def official_top(path, ids):
    rows = list(csv.DictReader(Path(path).open(encoding="utf-8-sig", newline="")))
    index = {int(v): i for i, v in enumerate(ids)}
    return np.array([[index[int(row[f"top{k}_id"])] for k in range(16)] for row in rows], int)


def path_categories(path, ids, labels):
    rows = list(csv.DictReader(Path(path).open(encoding="utf-8-sig", newline="")))
    by_id = {int(row["sample_id"]): row for row in rows}
    categories, paths = [], []
    for sid, label in zip(ids, labels):
        row = by_id[int(sid)]
        if int(row["label"]) != int(label):
            raise ValueError("Handoff label mismatch")
        image_path = row["image_path"].replace("\\", "/")
        categories.append(image_path.split("/")[-2])
        paths.append(image_path)
    return np.array(categories), np.array(paths)


def own_class_loo(u, labels):
    classes, inv, counts = np.unique(labels, return_inverse=True, return_counts=True)
    sums = np.zeros((len(classes), u.shape[1]))
    np.add.at(sums, inv, u)
    angle = np.empty(len(u))
    for lo in range(0, len(u), 2048):
        hi = min(lo + 2048, len(u))
        center = sums[inv[lo:hi]] - u[lo:hi]
        center, norms = unit_rows(center)
        angle[lo:hi] = np.degrees(np.arccos(np.clip(np.einsum("ij,ij->i", u[lo:hi], center), -1, 1)))
        angle[lo:hi][(counts[inv[lo:hi]] < 2) | (norms < 1e-12)] = np.nan
    return angle


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(args):
    source, handoff, output = map(Path, (args.source, args.handoff, args.output))
    if output.exists():
        raise FileExistsError(f"Fresh output required: {output}")
    output.mkdir(parents=True)
    (output / "figures").mkdir()
    started = time.time()
    results, private_arrays = {}, {}
    for ds in DATASETS:
        stamp = time.time()
        cache_path = source / "converted" / f"{ds}_train.npz"
        previous_path = source / "analysis" / f"HIER_{ds}_train" / "plot_arrays.npz"
        input_cache_sha, input_plot_sha = digest(cache_path), digest(previous_path)
        raw = np.load(cache_path, allow_pickle=False)
        previous = np.load(previous_path, allow_pickle=False)
        x, p = raw["mu"].astype(float), raw["proxy_ball"].astype(float)
        tangent = raw["proxy_tangent"].astype(float)
        labels, ids, c = raw["labels"], raw["sample_ids"], float(raw["c"])
        u, _ = unit_rows(x)
        v, radii = unit_rows(p)
        rd, pd = radial_coordinates(x, c), radial_coordinates(p, c)
        umean, vmean = u.mean(axis=0), v.mean(axis=0)
        mean_alignment = float(angle_to_center(unit_rows(vmean[None])[0], umean)[0])
        official = official_top(handoff / "proxy_eval" / "output" / f"{ds}_train" / "proxy_stats.csv", ids)
        summary, slots = coverage(official, len(x), labels)
        learned_direction, direction_slots = coverage(previous["direction_top_index"], len(x), labels)
        proxy_to_mean = angle_to_center(v, vmean)
        proxy_to_whole_mean = angle_to_center(v, umean)
        whole_to_proxy_mean = angle_to_center(u, vmean)
        whole_to_whole_mean = angle_to_center(u, umean)
        own_loo = own_class_loo(u, labels)
        tangent_norm = np.linalg.norm(tangent, axis=1)
        cap_mask = tangent_norm > 2.3
        groups = {"all": np.ones(len(p), bool), "forward_cap": cap_mask, "not_forward_cap": ~cap_mask,
                  "depth_below_4p4": pd["d0"] < 4.4, "depth_4p4_or_above": pd["d0"] >= 4.4}
        proxy_groups = {}
        for name, mask in groups.items():
            if not mask.any():
                proxy_groups[name] = {"n": 0}
                continue
            dsum = direction_summary(v[mask], pair_distribution=True)
            dsum.update({"depth": stats(pd["d0"][mask]),
                         "to_all_proxy_mean_angle": stats(proxy_to_mean[mask]),
                         "to_whole_mean_angle": stats(proxy_to_whole_mean[mask]),
                         "nearest_class_angle": stats(previous["proxy_nearest_center_angle_deg"][mask]),
                         "raw_top4_coverage": coverage(official[mask], len(x))[0]})
            proxy_groups[name] = dsum
        hottop = np.argsort(-slots, kind="stable")
        sets = {"all": np.arange(len(x)), "covered": np.flatnonzero(slots),
                "hottest4": hottop[:4], "hottest16": hottop[:16]}
        hotspots = {}
        for name, ix in sets.items():
            hotspots[name] = {"n": len(ix), "to_proxy_mean_angle": stats(whole_to_proxy_mean[ix]),
                              "to_whole_mean_angle": stats(whole_to_whole_mean[ix]),
                              "own_class_loo_angle": stats(own_loo[ix]), "depth": stats(rd["d0"][ix])}
        # One deterministic diagnostic changes the direction distribution itself.
        residual, residual_norm = unit_rows(v - vmean)
        if np.any(residual_norm < 1e-12):
            raise ValueError("Undefined centered proxy direction")
        residual_top = same_radius_raw_top(residual, radii, x, c, ids)
        residual_cov, residual_slots = coverage(residual_top, len(x), labels)
        # A common orthogonal rotation preserves the COMPLETE proxy Gram matrix.
        # Five independent Haar rotations assess global alignment with whole.
        rotation_results, rotation_coverage = [], []
        for seed in args.rotation_seeds:
            rng = np.random.default_rng(seed)
            q, rr = np.linalg.qr(rng.normal(size=(v.shape[1], v.shape[1])))
            q *= np.sign(np.diag(rr))[None, :]
            rotated = v @ q
            error = float(np.max(np.abs(rotated @ rotated.T - v @ v.T)))
            if error > 1e-12:
                raise AssertionError("Rotation failed Gram conservation")
            top = same_radius_raw_top(rotated, radii, x, c, ids)
            cov, _ = coverage(top, len(x), labels)
            cov.update({"seed": seed, "max_proxy_gram_error": error})
            rotation_results.append(cov)
            rotation_coverage.append(cov["unique_top4_objects"])
        object_rows = []
        for i in np.flatnonzero(slots):
            object_rows.append({"sample_id": int(ids[i]), "class_label": int(labels[i]),
                                "raw_top4_slots": int(slots[i]), "direction_top4_slots": int(direction_slots[i]),
                                "depth": rd["d0"][i], "to_proxy_mean_angle": whole_to_proxy_mean[i],
                                "to_whole_mean_angle": whole_to_whole_mean[i], "own_class_loo_angle": own_loo[i]})
        semantic = None
        if ds == "sop":
            categories, paths = path_categories(handoff / "proxy_eval" / "output" / "handoff" /
                                                "sample_id_map_sop_train.csv", ids, labels)
            semantic = []
            for cat in np.unique(categories):
                mask = categories == cat
                slots_n = int(slots[mask].sum())
                semantic.append({"category_from_image_path": cat, "objects": int(mask.sum()),
                                 "pool_fraction": float(mask.mean()), "raw_top4_slots": slots_n,
                                 "raw_top4_slot_fraction": slots_n / int(slots.sum()),
                                 "slot_pool_enrichment": (slots_n / int(slots.sum())) / mask.mean(),
                                 "hottest4_objects": int(np.isin(np.flatnonzero(mask), hottop[:4]).sum()),
                                 "own_class_loo_angle_median": float(np.nanmedian(own_loo[mask]))})
            semantic.sort(key=lambda z: -z["raw_top4_slots"])
            write_csv(output / "sop_semantic_categories.csv", semantic)
            lookup = {int(ids[i]): i for i in range(len(ids))}
            for row in object_rows:
                i = lookup[row["sample_id"]]
                row["semantic_category"] = categories[i]
                row["image_path"] = paths[i]
        write_csv(output / f"{ds}_private_hotspot_objects.csv", object_rows)
        write_csv(output / f"{ds}_private_proxy_directions.csv", [
            {"proxy_id": i, "depth": pd["d0"][i], "raw_tangent_norm": tangent_norm[i],
             "forward_cap": bool(cap_mask[i]), "angle_to_proxy_mean": proxy_to_mean[i],
             "angle_to_whole_mean": proxy_to_whole_mean[i],
             "nearest_class_angle": previous["proxy_nearest_center_angle_deg"][i],
             "nearest_class_label": int(previous["proxy_nearest_class"][i]),
             "official_top4_ids": " ".join(str(int(ids[j])) for j in official[i, :4])}
            for i in range(len(p))])
        result = {"cache_sha256": input_cache_sha, "plot_arrays_sha256": input_plot_sha,
                  "n_whole": len(x), "n_proxy": len(p), "dimension": x.shape[1], "c": c,
                  "whole_depth": stats(rd["d0"]), "proxy_depth": stats(pd["d0"]),
                  "whole_direction": direction_summary(u), "proxy_direction": direction_summary(v, True),
                  "angle_proxy_mean_vs_whole_mean": mean_alignment, "proxy_groups": proxy_groups,
                  "retrieval_official": summary, "retrieval_direction": learned_direction,
                  "hotspot_angles": hotspots, "sop_semantic_categories": semantic,
                  "controls": {"proxy_center_residual_same_radius": residual_cov,
                               "whole_rotated_proxy_cloud_same_radius": rotation_results,
                               "rotation_coverage": stats(rotation_coverage)},
                  "elapsed_seconds": time.time() - stamp}
        if digest(cache_path) != result["cache_sha256"] or digest(previous_path) != result["plot_arrays_sha256"]:
            raise AssertionError("Frozen input changed during analysis")
        results[ds] = result
        private_arrays[ds] = {"proxy_to_mean": proxy_to_mean, "proxy_to_whole_mean": proxy_to_whole_mean,
                              "whole_to_proxy_mean": whole_to_proxy_mean, "whole_to_whole_mean": whole_to_whole_mean,
                              "own_loo": own_loo, "whole_depth": rd["d0"], "proxy_depth": pd["d0"],
                              "proxy_nearest_class_angle": previous["proxy_nearest_center_angle_deg"],
                              "slots": slots, "cap_mask": cap_mask}
        np.savez_compressed(output / f"{ds}_plot_arrays.npz", **private_arrays[ds])
        print(f"{ds}: Rproxy={result['proxy_direction']['R']:.4f}, official={summary['unique_top4_objects']}, "
              f"residual={residual_cov['unique_top4_objects']}, rotations={rotation_coverage}, seconds={time.time()-stamp:.1f}", flush=True)
    manifest = {"models": results, "execution_seconds": time.time() - started,
                "script_sha256": digest(__file__), "training_updates": 0, "gpu_use": False,
                "definitions": {"direction_angle": "arccos of normalized vectors in full 512-dimensional space",
                                "R": "norm of mean unit directions; exact mean distinct-pair cosine=(nR^2-1)/(n-1)",
                                "effective_rank": "entropy exp(-sum p log p) of eigenvalue shares; uncentered and centered reported",
                                "hotspot": "official released-code full-pool reverse NN4 counts, not ancestor activation",
                                "own_class_loo": "angle to same-class mean excluding the inspected object",
                                "cap_group": "raw checkpoint tangent norm > 2.3, clipped only in forward",
                                "rotation": "one shared Haar orthogonal matrix applied to all proxy unit directions; all proxy-pair angles and radii preserved",
                                "residual": "normalize(proxy_unit_direction - mean_proxy_unit_direction), preserving individual ball radius; changes internal angles and positions"},
                "limits": ["Best checkpoint is a static cross-section; no temporal cause or training benefit inferred.",
                           "Three data domains/class counts/optimization histories differ; observational contrasts are confounded.",
                           "SOP raw radius dispersion is floating-point shell variation and cannot be read as radial hierarchy.",
                           "No model forward, optimizer update, image download or actual training activation was performed."]}
    (output / "summary.json").write_text(json.dumps(finite_json(manifest), ensure_ascii=False, indent=2), encoding="utf-8")
    plot(results, private_arrays, output)
    return manifest


def save(fig, output, name):
    fig.savefig(output / "figures" / f"{name}.png", dpi=180, bbox_inches="tight")
    fig.savefig(output / "figures" / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)


def plot(results, arrays, output):
    colors = {"cub": "#0072B2", "cars": "#D55E00", "sop": "#009E73"}
    fig, axs = plt.subplots(2, 3, figsize=(15, 8))
    for col, ds in enumerate(DATASETS):
        a, r = arrays[ds], results[ds]
        axs[0, col].scatter(a["proxy_depth"], a["proxy_to_mean"], s=14,
                            c=np.where(a["cap_mask"], "#CC79A7", colors[ds]), alpha=.65)
        axs[0, col].set(title=f"{ds.upper()}: proxy direction vs own common pole", xlabel="Proxy d(0,p)", ylabel="Angle to proxy mean (degree)")
        axs[0, col].axvline(4.6, c="gray", ls="--", lw=1)
        axs[1, col].scatter(a["proxy_depth"], a["proxy_nearest_class_angle"], s=14,
                            c=np.where(a["cap_mask"], "#CC79A7", colors[ds]), alpha=.65)
        axs[1, col].set(xlabel="Proxy d(0,p)", ylabel="Nearest fine-class center angle (degree)")
        axs[1, col].axvline(4.6, c="gray", ls="--", lw=1)
        axs[0, col].text(.02, .97, f"Forward-cap proxies: {a['cap_mask'].sum()}/512\nR(proxy)={r['proxy_direction']['R']:.3f}",
                         transform=axs[0, col].transAxes, va="top", fontsize=10)
    fig.suptitle("Full 512D angles: magenta points are clipped tangent parameters", y=1.01)
    fig.tight_layout()
    save(fig, output, "01_proxy_radial_angular_groups")

    fig, axs = plt.subplots(1, 3, figsize=(15, 4.7))
    for ds in DATASETS:
        r = results[ds]
        shares = r["proxy_direction"]["uncentered_spectrum"]["shares"]
        axs[0].plot(np.arange(1, len(shares)+1), np.cumsum(shares), c=colors[ds], label=ds.upper())
        axs[1].plot(np.arange(1, len(shares)+1), np.cumsum(r["whole_direction"]["uncentered_spectrum"]["shares"]), c=colors[ds], label=ds.upper())
    axs[0].set(xscale="log", xlabel="Number of eigenvectors", ylabel="Cumulative unit-direction energy", title="Proxy uncentered spectrum")
    axs[1].set(xscale="log", xlabel="Number of eigenvectors", ylabel="Cumulative unit-direction energy", title="Whole uncentered spectrum")
    loc = np.arange(3)
    axs[2].bar(loc-.18, [results[d]["proxy_direction"]["R"] for d in DATASETS], width=.36, label="Proxy R", color="#CC79A7")
    axs[2].bar(loc+.18, [results[d]["whole_direction"]["R"] for d in DATASETS], width=.36, label="Whole R", color="#56B4E9")
    axs[2].set(xticks=loc, xticklabels=[d.upper() for d in DATASETS], ylim=(0, 1), title="Common direction concentration")
    for ax in axs: ax.legend(fontsize=9)
    fig.tight_layout(); save(fig, output, "02_direction_spectrum_common_pole")

    fig, axs = plt.subplots(2, 3, figsize=(15, 8))
    for col, ds in enumerate(DATASETS):
        a = arrays[ds]
        hot = np.argsort(-a["slots"], kind="stable")[:4]
        for row, metric in enumerate(("whole_to_proxy_mean", "own_loo")):
            allv = np.sort(a[metric][np.isfinite(a[metric])])
            coveredv = np.sort(a[metric][(a["slots"] > 0) & np.isfinite(a[metric])])
            axs[row,col].plot(allv, np.arange(1,len(allv)+1)/len(allv), c="#888888", label="All whole")
            axs[row,col].plot(coveredv,np.arange(1,len(coveredv)+1)/len(coveredv),c=colors[ds], label="NN4-covered whole")
            axs[row,col].scatter(a[metric][hot], np.full(4,.04), marker="*", s=100,c="#D55E00",label="Hottest 4")
            axs[row,col].set(title=ds.upper(),xlabel="Angle to proxy mean (degree)" if row==0 else "Angle to own fine-class LOO center (degree)", ylabel="Cumulative fraction")
            axs[row,col].legend(fontsize=8)
    fig.tight_layout(); save(fig, output, "03_hotspot_direction_location")

    fig, axs = plt.subplots(1, 2, figsize=(13, 5))
    loc = np.arange(3)
    for j,(key,label,col) in enumerate((("official","Learned official raw","#0072B2"),("center","Centered residual; same radii","#D55E00"),("rotation","Shared orthogonal rotation; same radii","#009E73"))):
        vals=[]
        for ds in DATASETS:
            r=results[ds]
            vals.append(r["retrieval_official"]["unique_top4_objects"] if key=="official" else r["controls"]["proxy_center_residual_same_radius"]["unique_top4_objects"] if key=="center" else r["controls"]["rotation_coverage"]["mean"])
        axs[0].bar(loc+(j-1)*.25, vals, width=.24,color=col,label=label)
        if key=="rotation":
            lower=[vals[i]-results[ds]["controls"]["rotation_coverage"]["p00"] for i,ds in enumerate(DATASETS)]
            upper=[results[ds]["controls"]["rotation_coverage"]["p100"]-vals[i] for i,ds in enumerate(DATASETS)]
            axs[0].errorbar(loc+(j-1)*.25,vals,yerr=[lower,upper],fmt="none",ecolor="black",capsize=3)
    axs[0].set(xticks=loc,xticklabels=[d.upper() for d in DATASETS],ylabel="Unique objects among 512 x NN4 slots",title="Frozen direction controls (no updates)")
    axs[0].legend(fontsize=8, loc="upper left", bbox_to_anchor=(0, -.12), frameon=False)
    sem=results["sop"]["sop_semantic_categories"]
    cats=[r["category_from_image_path"].replace("_final","") for r in sem]
    yy=np.arange(len(cats))
    axs[1].barh(yy-.18,[r["pool_fraction"]*100 for r in sem],height=.36,color="#888888",label="Whole pool %")
    axs[1].barh(yy+.18,[r["raw_top4_slot_fraction"]*100 for r in sem],height=.36,color="#CC79A7",label="Official NN4 slots %")
    axs[1].set(yticks=yy,yticklabels=cats,xlabel="Percent",title="SOP broad categories from image paths")
    axs[1].invert_yaxis(); axs[1].legend(fontsize=9)
    fig.tight_layout(); save(fig, output, "04_direction_controls_semantic_hotspots")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--handoff", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--rotation-seeds", nargs="+", type=int, default=[301,302,303,304,305])
    run(parser.parse_args())


if __name__ == "__main__":
    main()
