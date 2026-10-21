"""Scientific figures for the V9 embedding audit; no model loading or updates.

Usage: python plot.py --export EXPORT_DIRECTORY --out FIGURE_DIRECTORY
The input is export_manifest.json and its NPZ caches.  Raw vectors and IDs stay
in the private export directory; only aggregate figures are written here.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


COLORS = ["#2475a6", "#d97824", "#2b936b", "#aa5295", "#bc4950", "#728436", "#7159a1", "#687b8d"]
STAGES = ("best", "e300")
WARNINGS: list[str] = []


def stage_name(value: Any) -> str:
    value = str(value).lower()
    return "e300" if value in {"e300", "last", "300", "final"} else value


def vector_depth(x: np.ndarray, c: float) -> np.ndarray:
    q = np.sqrt(c) * np.linalg.norm(np.asarray(x, dtype=np.float64), axis=-1)
    if np.any(~np.isfinite(q)) or np.any(q >= 1):
        raise ValueError(f"Non-finite or out-of-ball vector found (curvature {c}).")
    return 2 / np.sqrt(c) * np.arctanh(q)


def q_radius(x: np.ndarray, c: float) -> np.ndarray:
    return np.sqrt(c) * np.linalg.norm(np.asarray(x, dtype=np.float64), axis=-1)


def english_label(row: dict[str, Any]) -> str:
    value = str(row.get("plot_label", row.get("label", row["key"])))
    return value if value.isascii() else str(row["key"])


def usage(row: dict[str, Any], name: str) -> np.ndarray | None:
    data = row["data"]
    if name not in data:
        return None
    result = np.asarray(data[name], dtype=np.float64)
    if result.ndim > 1:
        result = result.sum(axis=tuple(range(result.ndim - 1)))
    result = result.reshape(-1)
    if len(result) != len(data["proxy_ball"]):
        WARNINGS.append(f"{row['key']}/{row['stage']}: ignored incompatible {name} shape.")
        return None
    return result


def save(fig: plt.Figure, out: Path, name: str, caption: str) -> dict[str, str]:
    fig.text(.012, .012, caption, fontsize=8, color="#4d5964", va="bottom")
    fig.tight_layout(rect=(0, .065, 1, .96))
    for ext in ("png", "svg"):
        fig.savefig(out / f"{name}.{ext}", dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return {"name": name, "png": name + ".png", "svg": name + ".svg", "caption": caption}


def ecdf(ax: plt.Axes, x: np.ndarray, **kwargs: Any) -> None:
    x = np.sort(np.asarray(x, dtype=np.float64).reshape(-1))
    x = x[np.isfinite(x)]
    if len(x):
        ax.step(x, np.arange(1, len(x) + 1) / len(x), where="post", **kwargs)


def load_rows(root: Path) -> list[dict[str, Any]]:
    manifest = json.loads((root / "export_manifest.json").read_text(encoding="utf-8-sig"))
    if isinstance(manifest, list):
        observations = manifest
    else:
        observations = next((manifest[k] for k in ("observations", "exports", "models", "items") if isinstance(manifest.get(k), list)), [])
    if not observations:
        raise ValueError("The manifest must contain an observations list.")
    rows = []
    for item in observations:
        row = dict(item)
        row["stage"] = stage_name(row.get("stage", "best"))
        path = Path(row["cache"])
        if not path.is_absolute():
            path = root / path
        if not path.exists():
            WARNINGS.append(f"Missing cache: {row['key']}/{row['stage']}.")
            continue
        with np.load(path, allow_pickle=False) as cache:
            row["data"] = {k: cache[k] for k in cache.files}
        if "native_mu" not in row["data"]:
            WARNINGS.append(f"Missing native_mu: {row['key']}/{row['stage']}.")
            continue
        row["c_native"] = float(row["c_native"])
        row["label_plot"] = english_label(row)
        row["native_depth"] = vector_depth(row["data"]["native_mu"], row["c_native"])
        row["native_q"] = q_radius(row["data"]["native_mu"], row["c_native"])
        if "proxy_ball" in row["data"]:
            row["c_hier"] = float(row.get("c_hier", .1))
            row["proxy_depth"] = vector_depth(row["data"]["proxy_ball"], row["c_hier"])
            row["tangent_norm"] = np.linalg.norm(row["data"]["proxy_tangent"].astype(np.float64), axis=-1)
            if "hier_mu" in row["data"]:
                row["hier_depth"] = vector_depth(row["data"]["hier_mu"], row["c_hier"])
        rows.append(row)
    keys = list(dict.fromkeys(row["key"] for row in rows))
    for row in rows:
        row["color"] = COLORS[keys.index(row["key"]) % len(COLORS)]
    return rows


def native_figures(rows: list[dict[str, Any]], out: Path) -> list[dict[str, str]]:
    figures = []
    curvatures = sorted({r["c_native"] for r in rows}, reverse=True)
    fig, axes = plt.subplots(2, len(curvatures), figsize=(7 * len(curvatures), 8), squeeze=False)
    for si, stage in enumerate(STAGES):
        for ci, curvature in enumerate(curvatures):
            ax = axes[si, ci]
            selected = [r for r in rows if r["stage"] == stage and np.isclose(r["c_native"], curvature)]
            values = [r["native_depth"] for r in selected]
            xmax = max([float(np.max(x)) for x in values] + [4.6])
            bins = np.linspace(0, xmax * 1.015, 80)
            for row in selected:
                ax.hist(row["native_depth"], bins=bins, density=True, histtype="step", linewidth=1.7,
                        color=row["color"], label=row["label_plot"])
            if np.isclose(curvature, .1):
                ax.axvline(4.6, color="#555555", ls="--", lw=1, label="HIER cap depth 4.6")
            ax.set(title=f"{stage}: native whole, c={curvature:g}", xlabel="Origin hyperbolic depth", ylabel="Probability density")
            ax.set_xlim(0, xmax * 1.02)
            ax.grid(alpha=.18)
            if selected:
                ax.legend(fontsize=8)
            else:
                ax.text(.5, .5, "No checkpoint available", transform=ax.transAxes, ha="center")
    fig.suptitle("Native whole embeddings | clean inputs, eval BN", fontsize=15)
    figures.append(save(fig, out, "01_native_whole_depth", "Curvatures are separated. Different best epochs are descriptive; e300 is the common training endpoint. Histogram spikes include cap point masses."))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), squeeze=False)
    for ax, stage in zip(axes[0], STAGES):
        for row in rows:
            if row["stage"] == stage:
                ecdf(ax, row["native_q"], color=row["color"], lw=1.6,
                     label=f"{row['label_plot']} (c={row['c_native']:g})")
        ax.set(title=stage, xlabel=r"Normalized coordinate radius $q=\sqrt{c}\|x\|$", ylabel="Fraction of objects", xlim=(0, 1), ylim=(0, 1.01))
        ax.grid(alpha=.18)
        ax.legend(fontsize=8, loc="upper left")
    fig.suptitle("Native whole radius ECDF | clean inputs, eval BN", fontsize=15)
    figures.append(save(fig, out, "02_normalized_radius_ecdf", "q compares position relative to each ball boundary. Equal q across different curvatures does not imply equal hyperbolic distance or equivalent representation capacity."))
    return figures


def hier_figures(rows: list[dict[str, Any]], out: Path) -> list[dict[str, str]]:
    selected = [r for r in rows if "proxy_depth" in r]
    keys = list(dict.fromkeys(r["key"] for r in selected))
    if not keys:
        return []
    figures = []
    fig, axes = plt.subplots(2, len(keys), figsize=(5.5 * len(keys), 8), squeeze=False)
    for si, stage in enumerate(STAGES):
        for ki, key in enumerate(keys):
            ax = axes[si, ki]
            matches = [r for r in selected if r["key"] == key and r["stage"] == stage]
            if not matches:
                ax.text(.5, .5, "No checkpoint available", transform=ax.transAxes, ha="center")
                continue
            row = matches[0]
            if not np.isclose(row["c_hier"], .1):
                raise ValueError("HIER comparison figure requires c_hier=0.1; separate other curvatures.")
            if "hier_depth" in row:
                ecdf(ax, row["hier_depth"], color="#465b72", label="Whole seen by HIER", lw=1.8)
            ecdf(ax, row["proxy_depth"], color="#cc6e23", label="All hierarchy proxies", lw=1.8)
            su = usage(row, "sample_usage")
            if su is not None and np.any(su > 0):
                ecdf(ax, row["proxy_depth"][su > 0], color="#21885c", label="Sample-selected proxies", ls="--", lw=1.5)
            ax.axvline(4.6, color="#555555", ls=":", lw=1)
            ax.set(title=f"{row['label_plot']} | {stage}", xlabel="Origin depth in HIER domain (c=0.1)", ylabel="Fraction", xlim=(0, 4.72), ylim=(0, 1.01))
            ax.grid(alpha=.18)
            ax.legend(fontsize=8, loc="upper left")
    fig.suptitle("HIER whole view and hierarchy proxies | clean/eval", fontsize=15)
    figures.append(save(fig, out, "03_hier_whole_proxy_depth", "V9 HIER whole depth is fixed by construction, not learned. Compare proxies to the HIER view in the same c=0.1 domain; native c=1 whole is a different representation."))

    fig, axes = plt.subplots(2, len(keys), figsize=(5.5 * len(keys), 8), squeeze=False)
    role_colors = {0: "#9ba4ad", 1: "#287fb3", 2: "#dd8b2e", 3: "#279268"}
    for si, stage in enumerate(STAGES):
        for ki, key in enumerate(keys):
            ax = axes[si, ki]
            matches = [r for r in selected if r["key"] == key and r["stage"] == stage]
            if not matches:
                ax.text(.5, .5, "No checkpoint available", transform=ax.transAxes, ha="center")
                continue
            row = matches[0]
            su, pu = usage(row, "sample_usage"), usage(row, "proxy_usage")
            known = su is not None and pu is not None
            roles = ((su > 0).astype(int) + 2 * (pu > 0).astype(int)) if known else np.zeros(len(row["proxy_depth"]), dtype=int)
            for role in range(4):
                mask = roles == role
                if np.any(mask):
                    ax.scatter(row["tangent_norm"][mask], row["proxy_depth"][mask], s=15, alpha=.65,
                               color=role_colors[role], edgecolors="none", label=f"{('Neither selected', 'Sample only', 'Proxy only', 'Both selected')[role]} ({mask.sum()})" if known else "Usage unavailable")
            ax.axvline(2.3, color="#555555", ls="--", lw=1)
            ax.axhline(4.6, color="#555555", ls=":", lw=1)
            if np.max(row["tangent_norm"]) > 50 * max(np.min(row["tangent_norm"]), 1e-8):
                ax.set_xscale("log")
            ax.set(title=f"{row['label_plot']} | {stage}", xlabel="Persistent tangent-parameter norm (before cap)", ylabel="Proxy output depth (after cap)")
            ax.grid(alpha=.18)
            ax.legend(fontsize=7, loc="best")
    fig.suptitle("Proxy parameters versus actual mapped positions", fontsize=15)
    figures.append(save(fig, out, "04_proxy_parameter_output", "The vertical threshold is tangent cap 2.3; output cap depth is 4.6. Selected means hard ancestor usage in the supplied measurement window, not every source of gradient."))

    last = [r for r in selected if r["stage"] == "e300"]
    if last:
        fig, axes = plt.subplots(len(last), 2, figsize=(16, 3.1 * len(last)), squeeze=False)
        for ri, row in enumerate(last):
            ids = np.arange(len(row["proxy_depth"]))
            axes[ri, 0].scatter(ids, row["proxy_depth"], s=9, color=row["color"])
            axes[ri, 0].axhline(4.6, color="#555555", ls="--", lw=1)
            axes[ri, 0].set(title=row["label_plot"] + " | e300", ylabel="Proxy origin depth", xlabel="Persistent proxy ID")
            for name, label, color in (("sample_usage", "Sample ancestor", "#287fb3"), ("proxy_usage", "Proxy ancestor", "#dd8b2e")):
                counts = usage(row, name)
                if counts is not None:
                    axes[ri, 1].plot(ids, np.log1p(counts), lw=.8, alpha=.8, label=label, color=color)
            axes[ri, 1].set(title="Ancestor usage in measurement window", ylabel="log(1 + count)", xlabel="Same persistent proxy ID")
            handles, _ = axes[ri, 1].get_legend_handles_labels()
            if handles:
                axes[ri, 1].legend(fontsize=8)
            for ax in axes[ri]:
                ax.grid(alpha=.18)
        fig.suptitle("E300 proxy positions and ancestor usage", fontsize=15)
        figures.append(save(fig, out, "05_proxy_id_and_usage", "Proxy IDs are aligned within each run, not between independently trained runs. Zero hard selections do not imply zero straight-through or query-endpoint gradients."))
    return figures


def hotspot_figures(rows: list[dict[str, Any]], out: Path) -> list[dict[str, str]]:
    selected = [r for r in rows if "nn4" in r["data"] and r["stage"] in STAGES]
    if not selected:
        return []
    fig, axes = plt.subplots(1, 2, figsize=(15, max(5, len(selected) * .5)))
    variants = (("nn4", "Raw", "#287fb3"), ("nn4_equal", "Equal-depth", "#dd8b2e"), ("nn4_angle", "Angular", "#279268"))
    ys = np.arange(len(selected))
    for vi, (name, label, color) in enumerate(variants):
        coverages, shares = [], []
        for row in selected:
            if name not in row["data"]:
                coverages.append(np.nan)
                shares.append(np.nan)
                continue
            nn = np.asarray(row["data"][name]).reshape(-1)
            _, counts = np.unique(nn, return_counts=True)
            coverages.append(len(counts))
            shares.append(100 * np.sort(counts)[-4:].sum() / len(nn))
        offset = (vi - 1) * .23
        axes[0].barh(ys + offset, coverages, height=.22, color=color, label=label)
        axes[1].barh(ys + offset, shares, height=.22, color=color, label=label)
    labels = [f"{r['label_plot']} | {r['stage']}" for r in selected]
    for ax in axes:
        ax.set_yticks(ys, labels, fontsize=8)
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=.18)
        ax.legend(fontsize=8)
    axes[0].set(xlabel="Distinct training objects in proxy NN4 slots", title="Reverse NN4 coverage")
    axes[1].set(xlabel="Share of all NN4 slots (%)", title="Four hottest objects")
    fig.suptitle("Frozen full-pool retrieval | clean/eval", fontsize=15)
    return [save(fig, out, "06_reverse_nn4_coverage", "NN4 coverage is a reverse full-pool retrieval statistic, not a measure of training activation. Equal-depth and angular variants are frozen controls, not retrained models.")]


def longitudinal_figures(root: Path, out: Path) -> list[dict[str, str]]:
    path = root / "longitudinal.json"
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    rows = payload if isinstance(payload, list) else payload.get("observations", payload.get("snapshots", payload.get("rows", [])))
    if not rows:
        return []
    keys = list(dict.fromkeys(str(r["key"]) for r in rows))
    fig, axes = plt.subplots(2, 2, figsize=(15, 9), squeeze=False)
    fields = [("native_whole_depth", axes[0, 0], "Native whole origin depth"), ("hier_whole_depth", axes[0, 1], "HIER whole-view origin depth"),
              ("proxy_depth", axes[1, 0], "Hierarchy proxy origin depth"), ("proxy_tangent_norm", axes[1, 1], "Proxy tangent-parameter norm")]
    for ki, key in enumerate(keys):
        selected = sorted([r for r in rows if str(r["key"]) == key], key=lambda r: float(r["epoch"]))
        for field, ax, title in fields:
            points = [(float(r["epoch"]), np.asarray(r.get(field, []), dtype=np.float64).reshape(-1)) for r in selected]
            points = [(epoch, values[np.isfinite(values)]) for epoch, values in points if len(values)]
            points = [(epoch, values) for epoch, values in points if len(values)]
            if not points:
                continue
            epochs = np.array([p[0] for p in points])
            stats = np.array([np.percentile(p[1], [25, 50, 75]) for p in points])
            color = COLORS[ki % len(COLORS)]
            ax.plot(epochs, stats[:, 1], marker="o", ms=3, color=color, label=key)
            ax.fill_between(epochs, stats[:, 0], stats[:, 2], color=color, alpha=.13)
            ax.set(title=title, xlabel="Epoch of stored first batch", ylabel="Median and interquartile range")
    for ax in axes.flat:
        ax.grid(alpha=.18)
        handles, _ = ax.get_legend_handles_labels()
        if handles:
            ax.legend(fontsize=7)
    axes[0, 1].axhline(4.6, color="#555555", ls="--", lw=1)
    axes[1, 0].axhline(4.6, color="#555555", ls="--", lw=1)
    axes[1, 1].axhline(2.3, color="#555555", ls="--", lw=1)
    fig.suptitle("Actual train first batch | pre-optimizer, current-batch BN", fontsize=15)
    result = [save(fig, out, "07_actual_training_trajectories", "Each point uses the stored first training batch (64 positions), before that batch's optimizer step. It is not an epoch-end checkpoint or a clean/eval full-pool distribution. Native curvatures can differ.")]

    proxy_keys = [key for key in keys if any(len(r.get("proxy_depth", [])) for r in rows if str(r["key"]) == key)]
    if proxy_keys:
        fig, axes = plt.subplots(len(proxy_keys), 2, figsize=(16, max(3.5, 3.2 * len(proxy_keys))), squeeze=False)
        for ki, key in enumerate(proxy_keys):
            selected = sorted([r for r in rows if str(r["key"]) == key and len(r.get("proxy_depth", []))], key=lambda r: float(r["epoch"]))
            depths = np.asarray([r["proxy_depth"] for r in selected], dtype=float)
            im = axes[ki, 0].imshow(depths, aspect="auto", interpolation="nearest", cmap="viridis", vmin=0, vmax=4.6)
            fig.colorbar(im, ax=axes[ki, 0], label="Origin depth")
            counts = []
            has_usage = False
            for row in selected:
                values = np.asarray(row.get("sample_usage", []), dtype=float).reshape(-1)
                if len(values) == depths.shape[1]:
                    counts.append(np.log1p(values))
                    has_usage = True
                else:
                    counts.append(np.full(depths.shape[1], np.nan))
            if has_usage:
                im2 = axes[ki, 1].imshow(np.asarray(counts), aspect="auto", interpolation="nearest", cmap="magma")
                fig.colorbar(im2, ax=axes[ki, 1], label="log(1 + sample ancestor count)")
            else:
                axes[ki, 1].text(.5, .5, "Usage not supplied", transform=axes[ki, 1].transAxes, ha="center")
            for ai, ax in enumerate(axes[ki]):
                ax.set(title=key + (" | proxy depths" if ai == 0 else " | sample ancestor usage"), xlabel="Persistent proxy ID", ylabel="Snapshot epoch")
                ax.set_yticks(np.arange(len(selected)), [str(r["epoch"]) for r in selected])
        fig.suptitle("Actual train first batch | pre-optimizer proxy depth and activity", fontsize=15)
        result.append(save(fig, out, "08_training_proxy_heatmaps", "These are first-batch pre-optimizer proxy states, not epoch-end archive weights. Columns preserve proxy identity within a run. Counts describe only the supplied first-batch selection window."))
    return result


def cycle_interim_figures(rows: list[dict[str, Any]], out: Path) -> list[dict[str, str]]:
    selected = [r for r in rows if str(r["key"]).startswith("cycle_") and int(r.get("epoch", -1)) == 120]
    keys = {str(r["key"]) for r in selected}
    if not {"cycle_off", "cycle_hier"}.issubset(keys):
        return []
    selected = [next(r for r in selected if r["key"] == key) for key in ("cycle_off", "cycle_hier")]
    if any(not np.isclose(r["c_native"], .1) for r in selected):
        raise ValueError("Cycle interim comparison requires native c=0.1 in both runs.")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    for row in selected:
        ecdf(axes[0], row["native_depth"], lw=1.7, color=row["color"], label=row["label_plot"])
        if "hier_depth" in row:
            ecdf(axes[1], row["hier_depth"], lw=1.7, color="#405b74", label="HIER whole view")
        if "proxy_depth" in row:
            ecdf(axes[1], row["proxy_depth"], lw=1.7, color="#cf782a", label="Hierarchy proxies")
            su = usage(row, "sample_usage")
            if su is not None and np.any(su > 0):
                ecdf(axes[1], row["proxy_depth"][su > 0], lw=1.5, ls="--", color="#2b936b", label="Sample-selected proxies")
            ecdf(axes[2], row["tangent_norm"], lw=1.7, color="#cf782a", label="Before-cap parameter norm")
    for ax in axes:
        ax.grid(alpha=.18)
        ax.legend(fontsize=8)
        ax.set_ylabel("Fraction")
        ax.set_ylim(0, 1.01)
    for ax in axes[:2]:
        ax.axvline(4.6, color="#555555", ls="--", lw=1)
        ax.set_xlim(0, 4.72)
        ax.set_xlabel("Origin hyperbolic depth (c=0.1)")
    axes[0].set_title("Native whole | same e120")
    axes[1].set_title("HIER whole view and proxies | e120")
    axes[2].axvline(2.3, color="#555555", ls="--", lw=1)
    axes[2].set(title="Proxy parameters | e120", xlabel="Tangent-parameter norm")
    fig.suptitle("INTERIM: cycle off versus HIER | matched e120, clean/eval", fontsize=16, color="#a04420")
    return [save(fig, out, "09_cycle_interim", "INTERIM comparison at the same e120 only. The HIER run was still training when collected; this is not its best or final result. Shared c=0.1 does not remove differences from original V9 sampling/seed.")]


def checkpoint_history_figures(root: Path, out: Path) -> list[dict[str, str]]:
    path = root / "proxy_checkpoint_history.json"
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    rows = payload if isinstance(payload, list) else payload.get("observations", payload.get("snapshots", payload.get("rows", [])))
    rows = [r for r in rows if len(r.get("proxy_depth", [])) and len(r.get("proxy_tangent_norm", []))]
    if not rows:
        return []
    keys = list(dict.fromkeys(str(r["key"]) for r in rows))
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    for ki, key in enumerate(keys):
        selected = sorted([r for r in rows if str(r["key"]) == key], key=lambda r: float(r["epoch"]))
        epochs = [float(r["epoch"]) for r in selected]
        for ax, field in zip(axes, ("proxy_depth", "proxy_tangent_norm")):
            stats = np.asarray([np.percentile(np.asarray(r[field], dtype=float), [25, 50, 75]) for r in selected])
            color = COLORS[ki % len(COLORS)]
            ax.plot(epochs, stats[:, 1], lw=1.5, marker="o", ms=3, color=color, label=key)
            ax.fill_between(epochs, stats[:, 0], stats[:, 2], color=color, alpha=.12)
    axes[0].axhline(4.6, color="#555555", ls="--", lw=1)
    axes[1].axhline(2.3, color="#555555", ls="--", lw=1)
    for ax, title in zip(axes, ("Mapped proxy origin depth", "Persistent tangent-parameter norm")):
        ax.set(title=title, xlabel="Completed epoch of archived checkpoint", ylabel="Median and interquartile range")
        ax.grid(alpha=.18)
        ax.legend(fontsize=8)
    fig.suptitle("Epoch-end checkpoint proxies | after all updates in the labelled epoch", fontsize=15)
    result = [save(fig, out, "10_checkpoint_proxy_trajectories", "These proxy states come from archived epoch-end weights, not first-batch pre-optimizer snapshots in Figures 07-08. Epoch 0 denotes initialization if supplied. Cycle observations are incomplete runs.")]

    fig, axes = plt.subplots(len(keys), 2, figsize=(16, max(3.5, 3.1 * len(keys))), squeeze=False)
    for ki, key in enumerate(keys):
        selected = sorted([r for r in rows if str(r["key"]) == key], key=lambda r: float(r["epoch"]))
        for ci, (field, label) in enumerate((("proxy_depth", "Origin depth"), ("proxy_tangent_norm", "log(1 + tangent norm)"))):
            values = np.asarray([r[field] for r in selected], dtype=float)
            if ci:
                values = np.log1p(values)
            image = axes[ki, ci].imshow(values, aspect="auto", interpolation="nearest", cmap="viridis" if ci == 0 else "magma", vmin=0, vmax=4.6 if ci == 0 else None)
            fig.colorbar(image, ax=axes[ki, ci], label=label)
            axes[ki, ci].set(title=key + (" | output depth" if ci == 0 else " | parameter norm"), xlabel="Persistent proxy ID", ylabel="Completed checkpoint epoch")
            axes[ki, ci].set_yticks(np.arange(len(selected)), [str(r["epoch"]) for r in selected], fontsize=8)
    fig.suptitle("Epoch-end checkpoint proxy histories | post-epoch optimizer state", fontsize=15)
    result.append(save(fig, out, "11_checkpoint_proxy_heatmaps", "Column identity is preserved within each run. Output depth and pre-cap parameter norm are different measurements. Archive time points must not be interpreted as first-batch states of the same numbered epoch."))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--export", dest="export_root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--include-keys", nargs="*", help="Explicit keys for core Figures 01-06; by default exclude cycle_* interim runs.")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "svg.fonttype": "none"})
    rows = load_rows(args.export_root)
    core = [r for r in rows if r["key"] in args.include_keys] if args.include_keys is not None else [r for r in rows if not str(r["key"]).startswith("cycle_")]
    core = [r for r in core if r["stage"] in STAGES]
    if not core:
        raise ValueError("No best/e300 observations selected for core figures.")
    figures = native_figures(core, args.out)
    figures += hier_figures(core, args.out)
    figures += hotspot_figures(core, args.out)
    figures += longitudinal_figures(args.export_root, args.out)
    figures += cycle_interim_figures(rows, args.out)
    figures += checkpoint_history_figures(args.export_root, args.out)
    (args.out / "figure_manifest.json").write_text(json.dumps({"figures": figures, "warnings": WARNINGS, "observations": len(rows), "core_observations": len(core)}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"figures": len(figures), "observations": len(rows), "warnings": WARNINGS}))


if __name__ == "__main__":
    main()
