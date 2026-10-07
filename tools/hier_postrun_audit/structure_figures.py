"""Scientific views of independently measured fixed-object shape relations."""
from pathlib import Path
import numpy as np


def save_structure_figures(structure, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .plots import _slug
    figures, groups = [], {}
    out = Path(output_dir) / "figures"
    out.mkdir(parents=True, exist_ok=True)
    for row in structure.get("snapshots", []):
        groups.setdefault(row.get("run_key", "snapshot"), []).append(row)
    for run, rows in groups.items():
        rows.sort(key=lambda row: row["epoch"])
        name = rows[0].get("display_name") or run
        modes = sorted({mode for row in rows for mode in row.get("shape", {})})
        for mode in modes:
            fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
            fields = [("within_class_spearman", "Class-balanced query Spearman"),
                      ("chance_adjusted_overlap", "Chance-adjusted shape top-k overlap"),
                      ("full_pool_chamfer_enrichment", "Chamfer enrichment vs same-class random")]
            for axis, (field, title) in zip(axes, fields):
                for method, color in (("hyperbolic", "#2563eb"), ("direction", "#d97706")):
                    x, y, lows, highs = [], [], [], []
                    for row in rows:
                        value = row.get("shape", {}).get(mode, {}).get("whole_relations", {}).get(method, {}).get(field, {})
                        mean, interval = value.get("class_balanced_mean"), value.get("class_bootstrap_95ci")
                        if mean is None:
                            continue
                        x.append(row["epoch"])
                        y.append(mean)
                        lows.append(interval[0] if interval else mean)
                        highs.append(interval[1] if interval else mean)
                    if x:
                        axis.plot(x, y, marker="o", label=method, color=color)
                        axis.fill_between(x, lows, highs, color=color, alpha=.14)
                axis.set(title=title, xlabel="Saved checkpoint epoch")
                axis.grid(alpha=.2)
                if field == "chance_adjusted_overlap":
                    axis.set_ylim(top=1.05)
                    axis.axhline(0, color="#777777", linewidth=.7)
                else:
                    axis.axhline(0, color="#777777", linewidth=.7)
            axes[0].legend(loc="best", fontsize=8)
            fig.suptitle(f"{name}: independent shape ({mode})\nFixed query objects; class bootstrap bands; saved rotations retained", fontsize=11)
            fig.tight_layout(rect=(0, 0, 1, .89))
            stem = _slug(run) + "_" + mode + "_independent_shape"
            png, svg = out / (stem + ".png"), out / (stem + ".svg")
            fig.savefig(png, dpi=150, bbox_inches="tight")
            fig.savefig(svg, bbox_inches="tight")
            plt.close(fig)
            figures.append({"run_id": str(run), "kind": "independent_shape", "title": f"{name}: independent shape ({mode})",
                            "note": "Class-balanced fixed-query means with class resampling intervals. Full declared cache remains retrieval pool. Chamfer is geometric evidence, not semantic hierarchy proof.",
                            "png": "figures/" + png.name, "svg": "figures/" + svg.name})
        last = rows[-1]
        classes = last.get("classification", {}).get("classes", [])
        if classes:
            fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
            labels = [str(item["label"]) for item in classes]
            axes[0].bar(np.arange(len(classes)), [1. - item["clean_accuracy"] for item in classes], color="#b91c1c")
            axes[1].plot(np.arange(len(classes)), [item["depth_quantiles"]["p50"] for item in classes], "o", color="#2563eb")
            for axis in axes:
                axis.set_xticks(np.arange(len(classes)))
                axis.set_xticklabels(labels, rotation=90, fontsize=7)
                axis.set_xlabel("Dataset class label")
                axis.grid(axis="y", alpha=.2)
            axes[0].set_ylabel("Fixed clean training error fraction")
            axes[1].set_ylabel("Median whole origin depth d0")
            fig.suptitle(f"{name} e{last['epoch']}: class and failure slices\nClean training objects; no validation/test forward", fontsize=11)
            fig.tight_layout(rect=(0, 0, 1, .89))
            stem = _slug(run) + f"_e{last['epoch']}_class_failure"
            png, svg = out / (stem + ".png"), out / (stem + ".svg")
            fig.savefig(png, dpi=150, bbox_inches="tight")
            fig.savefig(svg, bbox_inches="tight")
            plt.close(fig)
            figures.append({"run_id": str(run), "kind": "class_failure", "title": f"{name}: class radial/error slices",
                            "note": "Cached clean training logits and true class labels. Class purity is never substituted for morphology evidence.",
                            "png": "figures/" + png.name, "svg": "figures/" + svg.name})
    if len(groups) > 1:
        for mode in sorted({mode for row in structure.get("snapshots", []) for mode in row.get("shape", {})}):
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            ticks, names, all_values = [], [], {"hyperbolic": [], "direction": []}
            for index, (run, rows) in enumerate(groups.items()):
                last = max(rows, key=lambda row: row["epoch"])
                name = last.get("display_name") or run
                ticks.append(index)
                names.append(f"{name}\ne{last['epoch']}")
                for method in all_values:
                    relation = last.get("shape", {}).get(mode, {}).get("whole_relations", {}).get(method, {})
                    all_values[method].append(relation)
            for method, offset, color in (("hyperbolic", -.18, "#2563eb"), ("direction", .18, "#d97706")):
                for axis, field in zip(axes, ("within_class_spearman", "full_pool_chamfer_enrichment")):
                    values = [row.get(field, {}).get("class_balanced_mean") for row in all_values[method]]
                    for index, value in enumerate(values):
                        if value is not None:
                            axis.bar(index + offset, value, width=.34, color=color, label=method if index == 0 else None)
            for axis in axes:
                axis.set_xticks(ticks)
                axis.set_xticklabels(names, rotation=20, ha="right", fontsize=8)
                axis.axhline(0, color="#777777", linewidth=.7)
                axis.grid(axis="y", alpha=.2)
            axes[0].set_ylabel("Class-balanced query Spearman")
            axes[1].set_ylabel("Same-class random Chamfer enrichment")
            axes[0].legend()
            fig.suptitle(f"Historical final snapshots: independent shape ({mode})\nShared fixed queries; changed protocol/parameters do not support causal attribution", fontsize=11)
            fig.tight_layout(rect=(0, 0, 1, .89))
            png, svg = out / (f"cross_version_{mode}_shape.png"), out / (f"cross_version_{mode}_shape.svg")
            fig.savefig(png, dpi=150, bbox_inches="tight")
            fig.savefig(svg, bbox_inches="tight")
            plt.close(fig)
            figures.append({"run_id": "all_runs", "kind": "independent_shape_comparison",
                            "title": "Historical final independent shape comparison",
                            "note": "Shared fixed query objects and normalization. Historical versions change more than HIER; this is descriptive, not a single-variable causal ablation.",
                            "png": "figures/" + png.name, "svg": "figures/" + svg.name})
    return {"figures": figures, "warnings": structure.get("warnings", [])}
