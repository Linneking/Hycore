"""Plot existing original-HIER training-cache direction aggregates.

This reads anonymous, published aggregate measurements only. It performs no
model forward, training, or test evaluation. Cosines compare unit directions;
they are not angles or classification accuracies.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


MODEL_LABELS = (
    ("HIER_cub_train", "CUB"),
    ("HIER_cars_train", "Cars"),
    ("HIER_sop_train", "SOP细类"),
)
WITHIN_FIELD = "within_class_cosine_macro"
BETWEEN_FIELD = "between_class_cosine_macro_pairs"


def load_rows(aggregate_path: Path) -> list[dict]:
    payload = aggregate_path.read_bytes()
    aggregate = json.loads(payload)
    models = aggregate["models"]
    lookup = {model["name"]: model for model in models}
    if len(lookup) != len(models):
        raise ValueError("Aggregate repeats a model name")
    source_sha = hashlib.sha256(payload).hexdigest()
    rows = []
    for name, label in MODEL_LABELS:
        model = lookup[name]
        direction = model["whole_direction"]
        within = float(direction[WITHIN_FIELD])
        between = float(direction[BETWEEN_FIELD])
        if not all(math.isfinite(value) and -1 <= value <= 1
                   for value in (within, between)):
            raise ValueError(f"Invalid direction cosine for {name}")
        whole = model["radial"]["whole"]
        rows.append({
            "model": name,
            "dataset": label,
            "split": "train",
            "sample_count": int(model["N"]),
            "class_count": int(direction["class_count"]),
            "within_class_cosine_macro": within,
            "between_class_cosine_macro_pairs": between,
            "within_minus_between": within - between,
            "whole_depth_median": float(whole["d0"]["p50"]),
            "whole_fraction_within_1e_4_d0_of_cap": float(
                whole["fraction_within_1e_4_d0_of_cap"]),
            "measurement": "unit-direction cosine; not angle or test accuracy",
            "source_aggregate_sha256": source_sha,
        })
    return rows


def plot_rows(rows: list[dict], output_dir: Path) -> None:
    plt.rcParams.update({
        "font.family": "Microsoft YaHei",
        "axes.unicode_minus": False,
        "svg.fonttype": "none",
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    x = np.arange(len(rows))
    width = 0.31
    within = np.array([row[WITHIN_FIELD] for row in rows])
    between = np.array([row[BETWEEN_FIELD] for row in rows])
    fig, ax = plt.subplots(figsize=(10.8, 6.3))
    fig.subplots_adjust(left=0.095, right=0.975, bottom=0.245, top=0.80)
    bars_within = ax.bar(x - width / 2, within, width, color="#2568A3",
                         label="类内样本对（各类别等权）", zorder=3)
    bars_between = ax.bar(x + width / 2, between, width, color="#D18E25",
                          label="跨类样本对（类别对等权）", zorder=3)
    ax.bar_label(bars_within, labels=[f"{v:.4f}" for v in within],
                 padding=5, fontsize=11)
    ax.bar_label(bars_between, labels=[f"{v:.4f}" for v in between],
                 padding=5, fontsize=11)
    ax.set_xticks(x, [f"{row['dataset']}\n{row['class_count']:,}类 / "
                      f"{row['sample_count']:,}对象" for row in rows])
    ax.set_ylim(0, 1)
    ax.set_ylabel("单位方向余弦相似度（越大越同向）")
    ax.grid(axis="y", color="#DEE3E9", linewidth=0.8, zorder=0)
    ax.legend(loc="upper center", bbox_to_anchor=(.5, 1.15), ncol=2,
              frameon=False, fontsize=11)
    fig.suptitle("原版HIER：样本同壳，类别方向仍有分离", fontsize=19,
                 fontweight="bold", x=.52, y=.97)
    fig.text(.52, .905,
             "三份best的完整训练缓存；whole均位于原点双曲深度约4.6的薄壳",
             ha="center", color="#53657D", fontsize=11)
    fig.text(.095, .13,
             "类内：各类不同对象对的余弦均值，再对类别取平均。",
             va="center", color="#53657D", fontsize=10.5)
    fig.text(.095, .085,
             "跨类：每个类别对内所有跨类对象对的余弦均值，再对类别对取平均。",
             va="center", color="#53657D", fontsize=10.5)
    fig.text(.095, .035,
             "描述训练集方向几何，不是分类test成绩；纵轴为cos，不是角度。不同数据集不可直接排名。",
             va="center", color="#53657D", fontsize=10.5)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / "06_original_hier_class_direction_cosine"
    fig.savefig(stem.with_suffix(".png"), dpi=180)
    fig.savefig(stem.with_suffix(".svg"))
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aggregate", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True,
                        help="Report folder receiving figures/ and tables/")
    args = parser.parse_args()
    rows = load_rows(args.aggregate)
    table_dir = args.out_dir / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    with (table_dir / "original_hier_class_direction.csv").open(
            "w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    plot_rows(rows, args.out_dir / "figures")
    print(json.dumps({"model_count": len(rows), "rows": rows},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
