"""Render anonymous V9 cap-diagnostic aggregates; never loads model arrays.

Run with the report directory as an optional positional argument.  The three
figures have PNG and editable SVG versions, and a small JSON validation record.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


BLUE = "#2874A6"
TEAL = "#237D7D"
ORANGE = "#D17C28"
RED = "#AD4751"
PURPLE = "#8170A6"
GREEN = "#739747"
INK = "#26333F"
GREY = "#6F7B85"


def configure() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#ACB4BA",
            "axes.labelcolor": INK,
            "text.color": INK,
            "xtick.color": GREY,
            "ytick.color": GREY,
            "grid.color": "#E3E7EA",
            "grid.linewidth": 0.7,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "svg.fonttype": "none",
            "axes.axisbelow": True,
            "lines.linewidth": 2.0,
        }
    )


def save(fig: plt.Figure, output: Path, stem: str) -> list[str]:
    paths = []
    for extension in ("png", "svg"):
        path = output / f"{stem}.{extension}"
        fig.savefig(path, dpi=170, bbox_inches="tight", pad_inches=0.14)
        paths.append(path.name)
    plt.close(fig)
    return paths


def plot_trajectory(summary: dict, output: Path) -> list[str]:
    trajectory = summary["trajectory"]
    epochs = trajectory["epochs"]
    x = np.array([0] + [e["epoch"] for e in epochs])
    raw = [epochs[0]["start_norm"]] + [e["end_norm"] for e in epochs]
    mapped = [epochs[0]["start_mapped_depth"]] + [e["end_mapped_depth"] for e in epochs]
    counts = np.array([trajectory["initial_over_cap"]] + [e["end_over_cap"] for e in epochs])
    n = trajectory["proxy_count"]

    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.5), gridspec_kw={"wspace": 0.30})
    fig.suptitle("V9: raw proxy norms cross the cap while mapped depth saturates", x=0.055, y=1.01, ha="left", fontsize=15, weight="bold")
    for ax, data, color, title, ylabel, cap in (
        (axes[0], raw, BLUE, "A  Raw parameter norm", r"$\|p\|$ (before the forward cap)", 2.3),
        (axes[1], mapped, TEAL, "B  Forward-mapped depth", r"$d_0$ in the HIER branch", 4.6),
    ):
        ax.fill_between(x, [r["min"] for r in data], [r["max"] for r in data], color=color, alpha=0.13, label="Proxy min–max")
        ax.plot(x, [r["median"] for r in data], color=color, marker="o", markersize=4, label="Proxy median")
        ax.axhline(cap, color=RED, linestyle=(0, (4, 3)), linewidth=1.4, label=f"Cap = {cap:g}")
        ax.set(title=title, xlabel="Completed epochs", ylabel=ylabel, xlim=(-0.25, 10.25))
        ax.set_xticks(range(0, 11, 2))
        ax.grid(axis="y")
        ax.legend(loc="upper left" if cap == 2.3 else "lower right", frameon=False, fontsize=8.4)
    axes[0].set_ylim(1.45, 8.85)
    axes[0].text(0.06, 0.51, "Median 2.069 → 3.627", transform=axes[0].transAxes, fontsize=9.5, color=BLUE)
    axes[1].set_ylim(3.52, 4.74)
    axes[1].set_yticks([3.6, 3.8, 4.0, 4.2, 4.4, 4.6])
    axes[1].annotate("Median at the cap from epoch 2", xy=(5, 4.6), xytext=(0.4, 4.08), fontsize=9, color=TEAL)
    ax = axes[2]
    ax.plot(x, counts, color=PURPLE, marker="o", markersize=4)
    ax.axhline(n, color=GREY, linestyle=(0, (2, 3)), linewidth=1)
    ax.fill_between(x, 0, counts, color=PURPLE, alpha=0.09)
    for xi, count in zip(x[1:], counts[1:]):
        if xi in (1, 2, 5, 10):
            ax.annotate(str(count), (xi, count), xytext=(-8 if xi == 10 else 0, 7), textcoords="offset points", ha="center", fontsize=9)
    ax.text(0.40, 0.28, "502 / 512 by epoch 10\n0 observed returns inside", transform=ax.transAxes, fontsize=10, color=PURPLE, va="top")
    ax.set(title="C  Proxies outside the raw cap", xlabel="Completed epochs", ylabel="Count / 512 proxies", xlim=(-0.25, 10.25), ylim=(-15, 565))
    ax.set_xticks(range(0, 11, 2))
    ax.set_yticks([0, 128, 256, 384, 512])
    ax.grid(axis="y")
    fig.subplots_adjust(bottom=0.20, top=0.86, left=0.055, right=0.99)
    fig.text(0.055, 0.035, "Epoch 0: saved pre-first-update state. Other points: epoch end. Bands show proxy min–max, not uncertainty.\nOutside raw-cap and near-mapped-cap counts coincide at these epoch ends (mapped threshold 4.5999).", fontsize=9, color=GREY)
    return save(fig, output, "01_proxy_trajectory")


def plot_components(rows: list[dict], output: Path) -> tuple[list[str], float]:
    outside = [row for row in rows if row["cohort"] == "outside"]
    x = np.array([int(r["epoch"]) for r in outside])
    keys = (
        "history_radial_m_step",
        "history_tangent_m_step",
        "current_radial_g_step",
        "current_tangent_g_step",
        "weight_decay_step",
        "finite_step_excess",
    )
    labels = (
        "Historical m: radial component",
        "Historical m: tangential component",
        "Current g: radial component",
        "Current g: tangential component",
        "Weight decay",
        "Finite-step norm excess",
    )
    colors = (ORANGE, TEAL, BLUE, PURPLE, RED, GREEN)
    values = np.array([[float(r[k]) for r in outside] for k in keys])
    exact = np.array([float(r["exact_norm_change"]) for r in outside])
    residual = float(np.max(np.abs(values.sum(axis=0) - exact)))
    # Individual updates were applied in FP32; analytical displacement was
    # reconstructed offline in FP64.  Their round-off residual is recorded.
    if residual > 2e-6:
        raise ValueError(f"Component accounting does not close: {residual}")

    fig, ax = plt.subplots(figsize=(11.8, 5.4))
    positive = np.zeros(len(x))
    negative = np.zeros(len(x))
    for value, label, color in zip(values * 1e3, labels, colors):
        bottom = np.where(value >= 0, positive, negative)
        ax.bar(x, value, bottom=bottom, width=0.62, label=label, color=color, edgecolor="white", linewidth=0.35)
        positive += np.maximum(value, 0)
        negative += np.minimum(value, 0)
    ax.plot(x, exact * 1e3, "o-", color=INK, markersize=5, linewidth=1.7, label="Exact observed norm change", zorder=4)
    ax.axhline(0, color=INK, linewidth=0.8)
    ax.set(title="Already outside cap: signed accounting of each actual AdamW step", xlabel="Epoch (200 updates per epoch)", ylabel=r"Mean contribution to $\Delta\|p\|$ per update  ($\times 10^{-3}$)", xlim=(0.35, 10.65))
    ax.set_xticks(x)
    ax.grid(axis="y")
    ax.text(0.98, 0.92, "Historical radial m is the largest outward term.\nCurrent radial g contribution is approximately zero\nafter the forward cap (epoch 10: −2.34 × 10⁻⁹).", transform=ax.transAxes, ha="right", va="top", fontsize=10, bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.93, "pad": 6})
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3, frameon=False, fontsize=8.6)
    fig.subplots_adjust(left=0.085, right=0.985, top=0.89, bottom=0.29)
    fig.text(0.085, 0.025, "Dynamic cohort: raw norm > 2.3 before each update; means weighted by proxy-update observations.\nShared actual second moment; signed additive accounting, not separate interventions. Maximum aggregate FP32 residual: 1.03 × 10⁻⁷.", fontsize=9, color=GREY)
    return save(fig, output, "02_optimizer_accounting"), residual


def plot_counterfactuals(counterfactuals: dict, output: Path) -> list[str]:
    wanted = {(1, 5), (1, 20), (1, 100), (1, 200), (5, 200), (10, 200)}
    states = [q for q in counterfactuals["snapshots"] if (q["epoch"], q["step_in_epoch"]) in wanted]
    variants = ("native", "eps_1e4_only", "history_m_radial_removed")
    colors = (BLUE, TEAL, ORANGE)
    if len(states) != 6:
        raise ValueError("Expected six saved states")
    fig, axes = plt.subplots(2, 6, figsize=(15.5, 6.6), gridspec_kw={"hspace": 0.38, "wspace": 0.36})
    fig.suptitle("Single-step interventions can reverse outward drift at the same saved state", x=0.055, y=0.995, ha="left", fontsize=15, weight="bold")
    fig.text(0.055, 0.935, "Cohort: proxies already outside raw cap before each step. Top-row y scales differ; all values are labelled.", fontsize=10, color=GREY)

    for column, state in enumerate(states):
        stats = [state["variants"][variant]["before_over_cap"] for variant in variants]
        n = stats[0]["count"]
        if n <= 0 or any(s["count"] != n for s in stats):
            raise ValueError("Counterfactual cohorts must stay identical within a snapshot")
        means = np.array([s["mean_exact_norm_change"] for s in stats]) * 1e3
        fractions = np.array([s["norm_outward_count"] / n for s in stats]) * 100
        top, bottom = axes[:, column]
        top.bar(range(3), means, color=colors, width=0.68)
        top.axhline(0, color=INK, linewidth=0.8)
        top.grid(axis="y")
        top.set_title(f"E{state['epoch']} · step {state['step_in_epoch']}\nn = {n}", pad=10, fontsize=10)
        maximum, minimum = max(means.max(), 0), min(means.min(), 0)
        span = max(maximum - minimum, 0.01)
        top.set_ylim(minimum - 0.24 * span, maximum + 0.25 * span)
        top.yaxis.set_major_locator(mticker.MaxNLocator(4))
        top.set_xticks([])
        for xi, value in enumerate(means):
            top.annotate(f"{value:.2f}", (xi, value), xytext=(0, 4 if value >= 0 else -5), textcoords="offset points", ha="center", va="bottom" if value >= 0 else "top", fontsize=8.8, color=INK)
        bottom.bar(range(3), fractions, color=colors, width=0.68)
        bottom.set_ylim(0, 116)
        bottom.set_yticks([0, 50, 100])
        bottom.grid(axis="y")
        bottom.set_xticks(range(3), ["N", "E", "R"])
        bottom.tick_params(axis="x", length=0)
        for xi, value in enumerate(fractions):
            bottom.annotate(f"{value:.1f}", (xi, value), xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=8.4)
        if column != 0:
            bottom.set_yticklabels([])
    axes[0, 0].set_ylabel(r"Mean $\Delta\|p\|$ ($\times 10^{-3}$)")
    axes[1, 0].set_ylabel("Proxies with outward norm change (%)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=color) for color in colors]
    fig.legend(handles, ["N  Native AdamW", "E  Change eps to 10⁻⁴ only", "R  Remove historical m radial component"], loc="lower center", bbox_to_anchor=(0.51, 0.095), ncol=3, frameon=False, fontsize=10)
    fig.subplots_adjust(left=0.055, right=0.99, top=0.82, bottom=0.20)
    fig.text(0.055, 0.02, "Each intervention starts from the same saved parameter/current gradient/second moment/step. Changing eps preserves native historical m and v.\nOnly a private CPU optimizer clone takes one step; these results do not predict a retrained trajectory or classification accuracy.", fontsize=9, color=GREY)
    return save(fig, output, "03_fixed_state_counterfactuals")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report_dir", nargs="?", type=Path, default=Path(__file__).resolve().parents[2] / "71_V9_PROXY_CAP_CAUSE_2026-10-22")
    args = parser.parse_args()
    report_dir = args.report_dir.resolve()
    inputs = [report_dir / name for name in ("summary.json", "counterfactuals.json", "optimizer_components.csv")]
    initial_hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in inputs}
    summary = json.loads(inputs[0].read_text(encoding="utf-8"))
    counterfactuals = json.loads(inputs[1].read_text(encoding="utf-8"))
    with inputs[2].open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    configure()
    output = report_dir / "figures"
    output.mkdir(exist_ok=True)
    files = plot_trajectory(summary, output)
    component_files, residual = plot_components(rows, output)
    files.extend(component_files)
    files.extend(plot_counterfactuals(counterfactuals, output))
    final_hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in inputs}
    assert initial_hashes == final_hashes, "Input aggregate changed during rendering"
    record = {
        "matplotlib": matplotlib.__version__,
        "numpy": np.__version__,
        "source_sha256": initial_hashes,
        "source_unchanged": True,
        "gpu_used": False,
        "model_arrays_loaded": False,
        "component_sum_max_absolute_error": residual,
        "files": files,
    }
    (output / "plot_validation.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
