"""Static scientific figures from sanitized HyCoRe/HIER aggregates only.

No checkpoints, embeddings, point clouds, host paths, training or GPU calls.
Missing optional whole/one-step aggregates are skipped. PNG/PDF figures do
not contain source hashes; a separate manifest fingerprints aggregate inputs.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS = Path(__file__).resolve().parent / "results"
COLORS = {"blue": "#0072B2", "orange": "#D55E00", "green": "#009E73", "purple": "#CC79A7"}
CONDITIONS = ("eval", "train_bn", "train_bn_regrouped")
CONDITION_LABELS = {"eval": "Eval BN", "train_bn": "Train BN", "train_bn_regrouped": "Regroup BN"}


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def json_read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def style():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
        "axes.titlesize": 11, "axes.labelsize": 10, "legend.fontsize": 8.5,
        "xtick.labelsize": 9, "ytick.labelsize": 9, "axes.spines.top": False,
        "axes.spines.right": False, "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.facecolor": "white", "figure.facecolor": "white"})


def finish_figure(fig, output_dir, stem, dpi, caption, footer_height=.18):
    fig.text(.015, .016, caption, fontsize=8.5, ha="left", va="bottom", linespacing=1.45)
    fig.tight_layout(rect=(0, footer_height, 1, .93), pad=1.7)
    paths = []
    for extension in ("png", "pdf"):
        path = output_dir / f"{stem}.{extension}"
        fig.savefig(path, dpi=dpi, bbox_inches="tight", metadata={"Title": stem})
        paths.append(path.name)
    plt.close(fig)
    return paths


def point_labels(ax, x, values, formatter, color, offset=6):
    for xx, value in zip(x, values):
        ax.annotate(formatter(value), (xx, value), xytext=(0, offset), textcoords="offset points",
                    ha="center", fontsize=8.5, color=color)


def proxy_figure(document, output_dir, dpi):
    rows = []
    for name, modes in document["data"].items():
        match = re.search(r"(?:e|epoch[_-]?)(\d+)", name)
        if match is None:
            raise ValueError("Cannot infer checkpoint epoch from sanitized CPU run label")
        rows.append((int(match.group(1)), modes["original"]))
    rows.sort(key=lambda row: row[0])
    x = np.arange(len(rows))
    epochs = [row[0] for row in rows]
    all_order = [100 * row[1]["sample"]["noncollision"]["pair_minus_triple_depth"]["positive_fraction"] for row in rows]
    active_order = [100 * row[1]["sample"]["active_noncollision"]["pair_minus_triple_depth"]["positive_fraction"] for row in rows]
    endpoint = [100 * row[1]["endpoint"]["pair_endpoint_hit_fraction"] for row in rows]
    partial_key = "sample/whole_ball/class/dresser/signed_radial_gradient"
    dresser = [row[1]["whole_partial"][partial_key]["mean"] for row in rows]
    dresser_n = [row[1]["whole_partial"][partial_key]["n"] for row in rows]
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.5))
    fig.suptitle("V6 frozen-cache mechanism replay", fontsize=14, y=.985)
    axes[0].plot(x, all_order, "o-", color=COLORS["blue"], label="All noncollision draws", lw=1.8)
    axes[0].plot(x, active_order, "s-", color=COLORS["orange"], label="Active noncollision draws", lw=1.8)
    point_labels(axes[0], x, all_order, lambda v: f"{v:.1f}", COLORS["blue"])
    point_labels(axes[0], x, active_order, lambda v: f"{v:.1f}", COLORS["orange"], offset=-15)
    axes[0].set(title="A  Sample ancestor depth order", ylabel="Pair deeper than triple (%)", ylim=(0, 105))
    axes[0].legend(loc="lower left", frameon=False)
    axes[1].plot(x, endpoint, "o-", color=COLORS["purple"], lw=1.8)
    point_labels(axes[1], x, endpoint, lambda v: f"{v:.1f}%", "#5F385F")
    axes[1].set(title="B  Proxy pair equals an endpoint", ylabel="Noncollision proxy draws (%)", ylim=(0, 100))
    bars = axes[2].bar(x, dresser, color=[COLORS["orange"] if v >= 0 else COLORS["blue"] for v in dresser], width=.56)
    axes[2].axhline(0, color="#555555", lw=.8)
    for bar, value, n in zip(bars, dresser, dresser_n):
        axes[2].annotate(f"{value:+.3f}\nn={n}", (bar.get_x() + bar.get_width() / 2, value),
            xytext=(0, 5 if value >= 0 else -6), textcoords="offset points", ha="center",
            va="bottom" if value >= 0 else "top", fontsize=8.5)
    axes[2].set(title=r"C  Dresser radial $\mu$ partial", ylabel=r"Mean $g_\mu\!\cdot\!\hat\mu$ for $0.5R_{sample}$")
    magnitude = max(max(abs(v) for v in dresser), .02)
    axes[2].set_ylim(-1.25 * magnitude, 1.45 * magnitude)
    for ax in axes:
        ax.set_xticks(x, [str(epoch) for epoch in epochs])
        ax.set_xlabel("Checkpoint epoch")
        ax.grid(axis="y", alpha=.18)
        ax.set_axisbelow(True)
    caption = ("Original mining; 16 matched balanced64 batches per checkpoint; one-seed FP32 CPU replay of clean eval embeddings.\n"
        "Counts pool source draws; active draws exclude collisions and have nonzero hinge loss. Dresser n counts sampled positions, with repeats possible.\n"
        r"Negative radial partial implies an outward $-\nabla_\mu$ step; this is not a backbone or optimizer update. No morphology truth is used.")
    files = finish_figure(fig, output_dir, "proxy_cpu_mechanisms", dpi, caption, .23)
    return files, {"epochs": epochs, "sample_order_noncollision_percent": all_order,
                   "sample_order_active_percent": active_order, "proxy_pair_endpoint_percent": endpoint,
                   "dresser_mu_radial_partial_mean": dresser, "dresser_positions": dresser_n}


def ratio(numerator, denominator):
    return numerator / denominator if denominator and numerator is not None else np.nan


def paired_panel(ax, values, labels, title, ylabel, logarithmic=False, percent=False):
    values = np.asarray(values, dtype=float)
    x = np.arange(values.shape[1])
    jitter = np.linspace(-.07, .07, len(values)) if len(values) > 1 else np.array([0.])
    plotted = values * (100 if percent else 1)
    if logarithmic:
        plotted = np.where(plotted > 0, plotted, np.nan)
    for row, offset in zip(plotted, jitter):
        ax.plot(x + offset, row, "o-", color="#738395", alpha=.28, lw=.8, ms=3.5)
    medians = np.nanmedian(plotted, axis=0)
    ax.plot(x, medians, "D", color="#111111", ms=5, zorder=5)
    ax.set(title=title, ylabel=ylabel)
    ax.set_xticks(x, labels)
    if logarithmic:
        ax.set_yscale("log")
    if percent:
        ax.set_ylim(-2, 103)
    ax.grid(axis="y", alpha=.18)
    ax.set_axisbelow(True)
    return {"paired_batches": len(values), "condition_medians": [float(v) if np.isfinite(v) else None for v in medians],
            "nonplotted_nonfinite_or_nonpositive_values": int(np.sum(~np.isfinite(plotted)))}


def whole_figure(document, output_dir, dpi):
    batches = document["batches"]
    conditions = [condition for condition in CONDITIONS if all(condition in batch["conditions"] for batch in batches)]
    if not batches or len(conditions) < 2:
        raise ValueError("Whole aggregate needs paired batches with at least two common BN conditions")
    labels = [CONDITION_LABELS[condition] for condition in conditions]
    metrics = {key: [] for key in ("st_base", "st_fixed", "gap", "violation")}
    for batch in batches:
        rows = [batch["conditions"][condition] for condition in conditions]
        metrics["st_base"].append([ratio(row["parameter_gradient"]["st_norm"], row["parameter_gradient"]["base_norm"]) for row in rows])
        metrics["st_fixed"].append([ratio(row["parameter_gradient"]["st_norm"], row["parameter_gradient"]["fixed_norm"]) for row in rows])
        metrics["gap"].append([row["geometry"]["whole_minus_part_mean"] for row in rows])
        metrics["violation"].append([row["geometry"]["radial_margin_violation_fraction"] for row in rows])
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 7.6))
    epoch = document.get("identity", {}).get("checkpoint_epoch", "?")
    fig.suptitle(f"V6 e{epoch}: paired BN and gradient diagnosis", fontsize=14, y=.985)
    plotted = {}
    plotted["st_base"] = paired_panel(axes[0, 0], metrics["st_base"], labels, "A  ST / base gradient norm", "All model parameters: norm ratio", logarithmic=True)
    plotted["st_fixed"] = paired_panel(axes[0, 1], metrics["st_fixed"], labels, "B  ST / fixed-ancestor norm", "All model parameters: norm ratio", logarithmic=True)
    plotted["gap"] = paired_panel(axes[0, 2], metrics["gap"], labels, "C  Whole-part depth gap", "Mean whole depth - part depth")
    plotted["violation"] = paired_panel(axes[1, 0], metrics["violation"], labels, "D  Intra radial-margin violation", "Positions violating margin (%)", percent=True)
    compared = [condition for condition in conditions if condition != "eval" and all(
        "relation_comparison" in batch["conditions"][condition] for batch in batches)]
    if compared:
        compare_labels = [CONDITION_LABELS[condition] for condition in compared]
        for column, role in ((1, "pair"), (2, "triple")):
            values = [[batch["conditions"][condition]["relation_comparison"][f"fixed_draw_{role}_ancestor_agreement"] for condition in compared] for batch in batches]
            plotted[role + "_agreement"] = paired_panel(axes[1, column], values, compare_labels,
                f"{'E' if role == 'pair' else 'F'}  Fixed-triplet {role} agreement", "Same-noise agreement vs Eval BN (%)", percent=True)
    else:
        for ax in axes[1, 1:]:
            ax.axis("off")
            ax.text(.5, .5, "Fixed-triplet comparison unavailable", transform=ax.transAxes, ha="center")
    caption = (f"Same {len(batches)} augmented diagnostic batches; global64 assembled from two ordinary local32 forwards. Gray lines pair the same batch; black diamonds are medians.\n"
        "Weighted FP32 preclip VJPs include all model parameters (classifier included). Fixed-ancestor and ST losses have identical forward hinges.\n"
        "Ancestor agreement holds eval triplets and Gumbel noise fixed; a resampling-noise baseline is not shown. This is neither a rolling BN trajectory nor multi-seed evidence.")
    files = finish_figure(fig, output_dir, "whole_bn_parameter_mechanisms", dpi, caption, .16)
    return files, {"checkpoint_epoch": epoch, "conditions": conditions, "plotted_metrics": plotted}


def response_group(arm, group):
    return arm.get("response", arm.get("actual_clean_oldBN_response"))[group]


def one_step_figure(document, output_dir, dpi):
    batches = document["batches"]
    arms = ("base", "base_sample_st", "base_sample_fixed")
    names = {"base": "Base", "base_sample_st": "Base + sample ST", "base_sample_fixed": "Base + sample fixed"}
    colors = [COLORS["blue"], COLORS["orange"], COLORS["green"]]
    x = np.arange(len(batches))
    fig, axes = plt.subplots(2, 2, figsize=(10.3, 7.4))
    epoch = document.get("identity", {}).get("checkpoint_epoch", "?")
    fig.suptitle(f"V6 e{epoch}: actual one-step responses", fontsize=14, y=.985)
    extracted = {}
    for column, group in enumerate(("all64", "hotspot")):
        total = {arm: [response_group(batch["arms"][arm], group)["depth_delta"]["mean"] for batch in batches] for arm in arms}
        increment = {arm: [batch["increments_vs_base"][arm][group]["depth_delta"]["mean"] for batch in batches] for arm in arms[1:]}
        extracted[group] = {"total_mean_depth_delta": total, "incremental_mean_depth_delta_vs_base": increment}
        for at, arm in enumerate(arms):
            axes[0, column].bar(x + (at - 1) * .23, total[arm], .22, color=colors[at], label=names[arm])
        for at, arm in enumerate(arms[1:]):
            axes[1, column].bar(x + (at - .5) * .28, increment[arm], .27, color=colors[at + 1], label=names[arm] + " minus base")
        axes[0, column].set_title(f"{'A' if column == 0 else 'B'}  Total response: {'all64' if column == 0 else 'hotspots'}")
        axes[1, column].set_title(f"{'C' if column == 0 else 'D'}  Increment vs base: {'all64' if column == 0 else 'hotspots'}")
    for ax in axes.flat:
        ax.axhline(0, color="#555555", lw=.8)
        ax.set_xticks(x, [f"Batch {batch['batch'] + 1}" for batch in batches])
        ax.set_ylabel("Mean depth change")
        ax.grid(axis="y", alpha=.18)
        ax.set_axisbelow(True)
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
    axes[0, 0].legend(frameon=False, fontsize=8.5)
    axes[1, 0].legend(frameon=False, fontsize=8)
    caption = (f"{len(batches)} independent forced-hotspot panels; each arm restores the same model, RiemannianSGD momentum, LR and OLD BN readout buffers.\n"
        "Positive total depth change is outward from the shared start; positive increment is deeper than the base arm. These are different comparisons.\n"
        "Proxies are fixed; historical H20 momentum is retained. Batch means are not uncertainty intervals or evidence of a long-run training cause.")
    files = finish_figure(fig, output_dir, "one_step_total_and_increment", dpi, caption, .17)
    return files, {"checkpoint_epoch": epoch, "batches": len(batches), "responses": extracted,
        "clip_applied_by_arm": {arm: [batch["arms"][arm]["update"]["clip_applied"] for batch in batches] for arm in arms},
        "max_no_update_readout_error": max(float(value) for batch in batches for arm in arms for value in batch["arms"][arm]["no_update_readout_max_abs_error"].values())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proxy", type=Path, default=RESULTS / "20261006_proxy_cpu16_aggregate.json")
    parser.add_argument("--whole", type=Path, default=RESULTS / "20261006_whole_natural16_aggregate.json")
    parser.add_argument("--one-step", type=Path, default=RESULTS / "20261006_one_step_e300_hotspot2_aggregate.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dpi", type=int, default=240)
    args = parser.parse_args()
    if args.dpi < 100:
        parser.error("Use dpi>=100")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    manifest = {"status": "running", "format": "hier-mechanism-figures-v1", "inputs": {}, "figures": {},
                "skipped_optional": [], "started_utc": datetime.now(timezone.utc).isoformat()}
    write_json(args.output_dir / "plot_manifest.json", manifest)
    try:
        style()
        for name, path, plot in (("proxy", args.proxy, proxy_figure), ("whole", args.whole, whole_figure),
                                 ("one_step", args.one_step, one_step_figure)):
            if not path.exists():
                if name == "proxy":
                    raise FileNotFoundError("Required sanitized CPU aggregate is missing")
                manifest["skipped_optional"].append(name)
                print(f"Optional {name} aggregate missing; skipped", flush=True)
                continue
            before = digest(path)
            data = json_read(path)
            files, plotted = plot(data, args.output_dir, args.dpi)
            if digest(path) != before:
                raise RuntimeError("Sanitized aggregate changed during plotting")
            manifest["inputs"][name] = {"basename": path.name, "sha256": before}
            manifest["figures"][name] = {"files": files, "plotted_values": plotted}
        manifest.update(status="complete", finished_utc=datetime.now(timezone.utc).isoformat(),
            wall_seconds=time.perf_counter() - started, matplotlib_version=matplotlib.__version__,
            source_aggregates_unchanged=True, source_hashes_in_figures=False)
        write_json(args.output_dir / "plot_manifest.json", manifest)
        print(json.dumps({"status": "complete", "figures": list(manifest["figures"]),
                          "skipped_optional": manifest["skipped_optional"]}), flush=True)
    except Exception as exc:
        manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        write_json(args.output_dir / "plot_manifest.json", manifest)
        raise


if __name__ == "__main__":
    main()
