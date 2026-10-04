"""Plot reviewed epoch_curves.csv from a V5/V6 read-only monitoring audit."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    with args.csv.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    args.output_dir.mkdir(parents=True, exist_ok=False)
    colors = {"v5_H20": "#2668a5", "v6_H20": "#df7928"}
    panels = [
        ("Validation OA", [("val_oa", "", 1.)], "%"),
        ("Clean training OA (every 10 epochs)", [("clean_train_oa", "", 1.)], "%"),
        ("Learning rate: model", [("model_lr", "", 1.)], "LR"),
        ("Sample anchor eligibility", [("sample_anchor", "", 100.)], "%"),
        ("Proxy anchor eligibility", [("proxy_anchor", "", 100.)], "%"),
        ("Noncollision active triples (i != k)", [("sample_distinct_active_fraction", "sample", 100.), ("proxy_distinct_active_fraction", "proxy", 100.)], "%"),
        ("Whole / part near native ball boundary", [("whole_boundary", "whole", 100.), ("part_boundary", "part", 100.)], "%"),
        ("Proxy raw radius > .999 before projection", [("proxy_project", "", 100.)], "%"),
        ("Model gradient norm clipping frequency", [("model_clip_fraction", "", 100.)], "% of steps"),
    ]
    fig, axes = plt.subplots(3, 3, figsize=(15, 11), sharex=True)
    for ax, (title, fields, unit) in zip(axes.flat, panels):
        for run, color in colors.items():
            subset = [row for row in rows if row["run"] == run]
            for number, (field, label, scale) in enumerate(fields):
                valid = [row for row in subset if row[field] != ""]
                x = [float(row["epoch"]) for row in valid]
                y = [float(row[field]) * scale for row in valid]
                ax.plot(x, y, color=color, lw=1.35, alpha=.88,
                        ls="-" if number == 0 else "--",
                        marker="o" if field == "clean_train_oa" else None,
                        markersize=3, label=run.replace("_H20", "") + (" " + label if label else ""))
        ax.axvspan(1, 20, color="#eeeeee", zorder=0)
        ax.axvline(200, color="#999999", ls=":", lw=.9)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel(unit, fontsize=9)
        ax.grid(alpha=.2)
        ax.legend(fontsize=8, loc="best")
        ax.set_xlim(1, 300)
        if title in ("Validation OA", "Clean training OA (every 10 epochs)"):
            ax.set_ylim(80, 100)
    for ax in axes[-1]:
        ax.set_xlabel("Epoch")
    fig.suptitle("V5 / V6 H20 monitoring | same initialization and split; different cosine periods", fontsize=14)
    fig.text(.5, .013, "Shaded: 20 base-only epochs. V5 ends at 200; V6 ends at 300. Boundary figures are training-mode observations.\n"
             "Active fractions use noncollision draws with i != k in both runs. Curves are one seed, not uncertainty across seeds.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .055, 1, .95))
    fig.savefig(args.output_dir / "training_monitoring.png", dpi=160)
    fig.savefig(args.output_dir / "training_monitoring.pdf")
    plt.close(fig)
    print(args.output_dir / "training_monitoring.png", flush=True)


if __name__ == "__main__":
    main()
