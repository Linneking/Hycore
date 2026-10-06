"""Redraw reviewed shuffle-baseline CSV with canonical experiment names.

CPU plotting only; no checkpoint, model, point cloud or server-result access.
Input columns retain their historical names. Output files must be new.
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


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def points(rows, field, minimum_epoch=1):
    valid = [row for row in rows if row[field] != "" and int(row["epoch"]) >= minimum_epoch]
    return [int(row["epoch"]) for row in valid], [float(row[field]) for row in valid]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    before = digest(args.csv)
    with args.csv.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if [int(row["epoch"]) for row in rows] != list(range(1, 301)):
        raise ValueError("Expected reviewed e1–300 CSV, in unchanged order")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = [args.output_dir / f"monitoring_canonical.{ext}" for ext in ("png", "pdf")]
    manifest_path = args.output_dir / "monitoring_canonical_manifest.json"
    if any(path.exists() for path in [*output_paths, manifest_path]):
        raise FileExistsError("Canonical display outputs must use new paths")

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "pdf.fonttype": 42, "savefig.facecolor": "white"})
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 9.2))
    series = {}
    for field, label, color in (("b64_test_oa", "V6-B0-64: test", "#1f77b4"),
                                ("b32_test_oa", "V6-B0-32: test", "#ff7f0e")):
        x, y = points(rows, field, 21)
        axes[0, 0].plot(x, y, label=label, color=color, lw=1.1)
        series[field] = {"epochs": x, "values": y}
        index = max(range(len(y)), key=lambda i: y[i])
        axes[0, 0].plot(x[index], y[index], "o", color=color, ms=5)
        annotation_options = ({"xytext": (-100, 16), "textcoords": "offset points"}
                              if field == "b64_test_oa" else
                              {"xytext": (135, 95), "textcoords": "data",
                               "bbox": {"facecolor": "white", "edgecolor": "none", "alpha": .85, "pad": 2}})
        axes[0, 0].annotate(f"{y[index]:.3f}% @ {x[index]}", (x[index], y[index]),
                             color=color, arrowprops={"arrowstyle": "-", "color": color},
                             fontsize=9, **annotation_options)
    axes[0, 0].set(title="Test OA, e21–300; both select on test", ylabel="OA (%)", ylim=(86, 96))
    axes[0, 0].legend(loc="lower right", fontsize=9)

    for field, label, color, marker in (("b64_test_oa", "V6-B0-64 test (every epoch)", "#1f77b4", None),
                                       ("b64_clean_train_oa", "V6-B0-64 clean train (every 10)", "#258b73", "o")):
        x, y = points(rows, field)
        axes[0, 1].plot(x, y, label=label, color=color, lw=1.3, marker=marker, ms=3)
        series["clean_panel_" + field] = {"epochs": x, "values": y}
    axes[0, 1].set(title="V6-B0-64 clean training and test", ylabel="OA (%)", ylim=(78, 101))
    axes[0, 1].legend(loc="lower right", fontsize=8.5)

    for field, label, color in (("whole_boundary_percent", "Whole", "#8864b5"),
                                ("part_boundary_percent", "Part", "#258b73")):
        x, y = points(rows, field)
        axes[1, 0].plot(x, y, label=label, color=color, lw=1.1)
        series[field] = {"epochs": x, "values": y}
    axes[1, 0].set(title="V6-B0-64 training-mode native-boundary proximity",
                   ylabel="Positions with radius ≥ 0.99599 (%)", ylim=(0, 100))
    axes[1, 0].legend(loc="upper right", fontsize=9)

    for field, label, color in (("b64_clip_step_percent", "Norm-1 clipping: steps", "#cf596c"),
                                ("b64_sameclass_flip_percent", "Same-class flip: pairs", "#538393")):
        x, y = points(rows, field)
        axes[1, 1].plot(x, y, label=label, color=color, lw=1.1)
        series[field] = {"epochs": x, "values": y}
    axes[1, 1].set(title="V6-B0-64 gradient and negative-pair monitoring",
                   ylabel="Fraction (%)", ylim=(0, 20))
    axes[1, 1].legend(loc="upper left", fontsize=9)

    for ax in axes.flat:
        ax.set_xlabel("Epoch")
        ax.grid(alpha=.2)
        ax.set_xlim(1, 300)
    fig.suptitle("V6-B0-64 versus V6-B0-32: HyCoRe shuffle baselines\n"
                 "Single seed; protocols differ from V6-HIER64-K20-W20", fontsize=14)
    fig.text(.5, .012, "Display labels only. Original CSV fields and values are retained. "
             "These shuffle baselines do not isolate the HIER objective.", ha="center", fontsize=8.5)
    fig.tight_layout(rect=(0, .038, 1, .94))
    for path in output_paths:
        fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    if digest(args.csv) != before:
        raise RuntimeError("Reviewed CSV changed during plotting")
    manifest = {"format": "hycore-canonical-display-plot-v1", "status": "complete",
                "input_csv_basename": args.csv.name, "input_csv_sha256": before,
                "display_names": {"b64": "V6-B0-64", "b32": "V6-B0-32"},
                "outputs": [path.name for path in output_paths],
                "input_unchanged": True, "numerical_operation": "float parsing and epoch-window selection only",
                "plotted_series": series}
    manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"status": "complete", "outputs": manifest["outputs"], "input_unchanged": True}), flush=True)


if __name__ == "__main__":
    main()
