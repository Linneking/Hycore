"""Five figure groups and a concise report for the simplified PartHier audit.

This module only consumes aggregate metrics. It never loads model weights,
embeddings, object IDs, or GPU libraries. Plot imports are lazy so formatting
and publication-boundary checks can also run without Matplotlib installed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


ORDER = ("TS02_best", "TS02_e300", "TS18_best", "TS18_e300")
PH_ORDER = ("TS18_best", "TS18_e300")
DISPLAY = {
    "TS02_best": "HyCoRe best e189",
    "TS02_e300": "HyCoRe e300",
    "TS18_best": "PartHier best e201",
    "TS18_e300": "PartHier e300",
}
ZH = {
    "TS02_best": "原版 best e189",
    "TS02_e300": "原版 e300",
    "TS18_best": "PartHier best e201",
    "TS18_e300": "PartHier e300",
}
VIEWS = ("clean", "whole", "p200", "p400", "p600")
VIEW_TEXT = {"clean": "Clean whole", "whole": "Overwritten whole", "p200": "P200", "p400": "P400", "p600": "P600"}
VIEW_COLOR = {"clean": "#111827", "whole": "#8b5cf6", "p200": "#0072b2", "p400": "#009e73", "p600": "#d55e00"}
MODEL_COLOR = {"TS02_best": "#7c98b3", "TS02_e300": "#314e71", "TS18_best": "#e69f00", "TS18_e300": "#b44c29"}
ANGLE_GROUPS = ("A_same", "A_cross", "B2", "B3")
ANGLE_TEXT = {"A_same": "A: same-class i/j", "A_cross": "A: cross-class i/j", "B2": "B: two classes", "B3": "B: three classes"}
GROUP_COLOR = dict(zip(ANGLE_GROUPS, ("#0072b2", "#d55e00", "#009e73", "#8b5cf6")))


def number(value):
    """Finite scalar or None; missing values must not turn into zero."""
    try:
        result = float(value)
    except (ValueError, TypeError):
        return None
    return result if math.isfinite(result) else None


def fmt(value, digits=3):
    value = number(value)
    return "—" if value is None else f"{value:.{digits}f}"


def pct(value):
    value = number(value)
    return "—" if value is None else f"{100 * value:.2f}%"


def intfmt(value):
    value = number(value)
    return "—" if value is None else f"{int(value):,}"


def qpair(stats):
    stats = stats or {}
    return f"{fmt(stats.get('median'), 2)}° / {fmt(stats.get('p90'), 2)}°"


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, columns, rows):
    with Path(path).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def validate_summary(summary, expected_count=9840):
    models = summary.get("models", {})
    missing = set(ORDER) - set(models)
    if missing:
        raise ValueError(f"Missing required models: {sorted(missing)}")
    for label in ORDER:
        model = models[label]
        for view in VIEWS:
            count = model.get("depth", {}).get(view, {}).get("count")
            if count != expected_count:
                raise ValueError(f"{label}/{view}: expected {expected_count} objects, got {count}")
        if model.get("chain", {}).get("count") != expected_count:
            raise ValueError(f"{label}: incomplete object-level chain audit")
    for label in PH_ORDER:
        relation = models[label].get("relations", {})
        draws, valid, collisions = (relation.get(key) for key in ("draws", "valid_draws", "collisions"))
        if not all(isinstance(value, (int, float)) for value in (draws, valid, collisions)):
            raise ValueError(f"{label}: missing relation denominators")
        if draws <= 0 or valid < 0 or collisions < 0 or valid + collisions != draws:
            raise ValueError(f"{label}: inconsistent relation denominators")
    return models


def public_models(models):
    """Drop incidental paths/identity fields before publishing aggregates."""
    allowed = ("depth", "chain", "intra", "relations")
    return {label: {key: models[label][key] for key in allowed if key in models[label]}
            for label in ORDER}


def _fraction_check(value, name):
    numeric = number(value)
    if numeric is not None and not -1e-8 <= numeric <= 1 + 1e-8:
        raise ValueError(f"Fraction out of range at {name}: {numeric}")


def publication_tables(analysis_dir, out_dir, models):
    """Whitelist aggregate columns so private per-object fields cannot leak."""
    table_dir = out_dir / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    schemas = {
        "depth_hist.csv": ("model", "view", "bin_left", "bin_right", "count", "percentage"),
        "depth_ecdf.csv": ("model", "view", "depth", "percentage"),
        "angle_hist.csv": ("model", "group", "metric", "bin_left", "bin_right", "count", "percentage"),
    }
    tables = {}
    for filename, columns in schemas.items():
        path = analysis_dir / filename
        rows = read_csv(path)
        if not rows:
            raise ValueError(f"Empty aggregate table: {filename}")
        tables[filename] = rows
        write_csv(table_dir / filename, columns, rows)

    intra = []
    for label in ORDER:
        for scale in ("200", "400", "600"):
            by_scale = models[label]["intra"][scale]
            for term in ("radial", "contrastive"):
                stat = by_scale[term]
                intra.append({"model": label, "scale": scale, "term": term, **{key: stat.get(key) for key in ("count", "margin", "basic_fraction", "margin_fraction", "mean_hinge", "median", "p90")}})
    write_csv(table_dir / "intra.csv", ("model", "scale", "term", "count", "margin", "basic_fraction", "margin_fraction", "mean_hinge", "median", "p90"), intra)

    chain = []
    for label in ORDER:
        source = models[label]["chain"]
        for edge, stat in source["edges"].items():
            chain.append({"model": label, "edge": edge, **{key: stat.get(key) for key in ("count", "margin", "basic_fraction", "margin_fraction", "mean_hinge", "median", "p25", "p75")}})
        chain.append({"model": label, "edge": "joint", "count": source["count"], "basic_fraction": source["joint_basic_fraction"], "margin_fraction": source["joint_margin_fraction"]})
    write_csv(table_dir / "chain.csv", ("model", "edge", "count", "margin", "basic_fraction", "margin_fraction", "mean_hinge", "median", "p25", "p75"), chain)

    relation_rows, role_rows, angle_rows = [], [], []
    for label in PH_ORDER:
        source = models[label]["relations"]
        for family in ("ord", "hier"):
            for edge, stat in source[family]["edges"].items():
                relation_rows.append({"model": label, "family": family, "edge": edge, **{key: stat.get(key) for key in ("count", "margin", "basic_fraction", "margin_fraction", "mean_hinge")}})
            relation_rows.append({"model": label, "family": family, "edge": "joint", "count": source["valid_draws"], "basic_fraction": source[family]["joint_basic_fraction"], "margin_fraction": source[family]["joint_margin_fraction"]})
        for role, stat in source["roles"].items():
            scale_fraction = stat.get("scale_fractions", [])
            if isinstance(scale_fraction, dict):
                scale_fraction = [scale_fraction.get(str(value), scale_fraction.get(value)) for value in (200, 400, 600)]
            role_rows.append({"model": label, "role": role, "count": stat["count"], "p200_fraction": scale_fraction[0], "p400_fraction": scale_fraction[1], "p600_fraction": scale_fraction[2], **{key: stat.get(key) for key in ("current_fraction", "queue_fraction", "self_source_fraction", "unique_queue_entries", "unique_live_occurrences")}})
        for group, stat in source["angles"].items():
            angle_rows.append({"model": label, "group": group, "count": stat.get("count"), "nearest_median": stat.get("nearest", {}).get("median"), "nearest_p90": stat.get("nearest", {}).get("p90"), "descendant_median": stat.get("descendant_angles", {}).get("median"), "descendant_p90": stat.get("descendant_angles", {}).get("p90"), "second_gap_median": stat.get("second_gap", {}).get("median"), "nearest_in_descendant_fraction": stat.get("nearest_in_descendant_fraction")})
    write_csv(table_dir / "relation.csv", ("model", "family", "edge", "count", "margin", "basic_fraction", "margin_fraction", "mean_hinge"), relation_rows)
    write_csv(table_dir / "roles.csv", ("model", "role", "count", "p200_fraction", "p400_fraction", "p600_fraction", "current_fraction", "queue_fraction", "self_source_fraction", "unique_queue_entries", "unique_live_occurrences"), role_rows)
    write_csv(table_dir / "angles.csv", ("model", "group", "count", "nearest_median", "nearest_p90", "descendant_median", "descendant_p90", "second_gap_median", "nearest_in_descendant_fraction"), angle_rows)
    return tables


def _plotting():
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "svg.fonttype": "none", "savefig.facecolor": "white"})
    return np, plt


def _save(fig, directory, basename, plt):
    fig.savefig(directory / (basename + ".png"), dpi=180, bbox_inches="tight")
    fig.savefig(directory / (basename + ".svg"), bbox_inches="tight")
    plt.close(fig)


def plot_depth(tables, models, directory, np, plt):
    hist, ecdf = tables["depth_hist.csv"], tables["depth_ecdf.csv"]
    fig, axes = plt.subplots(3, 4, figsize=(16, 9), sharex=True, sharey="row")
    for column, label in enumerate(ORDER):
        axes[0, column].set_title(DISPLAY[label])
        for view in VIEWS:
            rows = sorted((row for row in hist if row["model"] == label and row["view"] == view), key=lambda row: float(row["bin_left"]))
            if not rows:
                raise ValueError(f"Missing depth histogram {label}/{view}")
            x = [(float(row["bin_left"]) + float(row["bin_right"])) / 2 for row in rows]
            for row_number, key in ((0, "count"), (1, "percentage")):
                axes[row_number, column].step(x, [float(row[key]) for row in rows], where="mid", color=VIEW_COLOR[view], label=f"{VIEW_TEXT[view]} (n={int(models[label]['depth'][view]['count']):,})", linestyle="--" if view == "clean" else "-")
            rows = sorted((row for row in ecdf if row["model"] == label and row["view"] == view), key=lambda row: float(row["depth"]))
            if not rows:
                raise ValueError(f"Missing ECDF {label}/{view}")
            axes[2, column].plot([float(row["depth"]) for row in rows], [float(row["percentage"]) for row in rows], color=VIEW_COLOR[view], linestyle="--" if view == "clean" else "-")
        axes[2, column].set_xlabel("Depth from origin, c=1")
        axes[2, column].set_ylim(0, 100)
        axes[0, column].legend(fontsize=6.5, loc="upper left")
        for row_number in range(3):
            axes[row_number, column].grid(alpha=.15)
    axes[0, 0].set_ylabel("Objects per bin")
    axes[1, 0].set_ylabel("Within-view percentage (%)")
    axes[2, 0].set_ylabel("Cumulative percentage (%)")
    fig.suptitle("01  Whole and part depth distributions | each curve has its own denominator", fontsize=12)
    fig.tight_layout()
    _save(fig, directory, "01_depth", plt)


def _grouped_bars(ax, labels, series, np, *, ylabel="Relations (%)", ylim=(0, 103)):
    x = np.arange(len(labels), dtype=float)
    width = .78 / max(len(series), 1)
    for index, (name, values, color) in enumerate(series):
        y = np.array([float("nan") if number(value) is None else 100 * float(value) for value in values])
        bars = ax.bar(x + (index - (len(series) - 1) / 2) * width, y, width=width, color=color, label=name)
        if len(labels) * len(series) <= 10:
            for bar, value in zip(bars, y):
                if math.isfinite(value):
                    ax.text(bar.get_x() + bar.get_width() / 2, value + 1, f"{value:.1f}", ha="center", fontsize=7)
    ax.set_xticks(x, labels)
    ax.set_ylabel(ylabel)
    ax.set_ylim(*ylim)
    ax.grid(axis="y", alpha=.15)


def plot_chain(models, directory, np, plt):
    fig = plt.figure(figsize=(12, 8))
    grid = fig.add_gridspec(2, 2)
    edge_keys = ("200_400", "400_600", "600_whole")
    edge_names = ("P200 < P400", "P400 < P600", "P600 < Whole")
    for col, (field, title) in enumerate((("basic", "Basic order: gap > 0"), ("margin", "Designed depth gap: >= 0.2"))):
        ax = fig.add_subplot(grid[0, col])
        series = []
        for label in ORDER:
            source = models[label]["chain"]
            values = [source["edges"][edge][field + "_fraction"] for edge in edge_keys] + [source["joint_" + field + "_fraction"]]
            series.append((DISPLAY[label], values, MODEL_COLOR[label]))
        _grouped_bars(ax, (*edge_names, "All three"), series, np, ylabel="Objects (%)")
        ax.set_title(title)
        ax.tick_params(axis="x", labelsize=8)
        if col == 0:
            ax.legend(fontsize=7, loc="lower left")
    ax = fig.add_subplot(grid[1, :])
    x = np.arange(3)
    for offset, label in enumerate(ORDER):
        stats = [models[label]["chain"]["edges"][edge] for edge in edge_keys]
        median = np.array([stat["median"] for stat in stats], dtype=float)
        lower = np.maximum(0, median - np.array([stat["p25"] for stat in stats], dtype=float))
        upper = np.maximum(0, np.array([stat["p75"] for stat in stats], dtype=float) - median)
        ax.errorbar(x + (offset - 1.5) * .12, median, yerr=np.vstack([lower, upper]), fmt="o", capsize=4, color=MODEL_COLOR[label], label=DISPLAY[label])
    ax.axhline(0, color="#555555", linestyle=":", label="Basic-order threshold 0")
    ax.axhline(.2, color="#aa2222", linestyle="--", label="Designed margin 0.2")
    ax.set_xticks(x, edge_names)
    ax.set_ylabel("Depth gap: median and IQR")
    ax.grid(axis="y", alpha=.15)
    ax.legend(fontsize=7, ncol=3)
    fig.suptitle("02  Object-level chains | depth gaps, not Euclidean-radius margins", fontsize=12)
    fig.tight_layout()
    _save(fig, directory, "02_chain", plt)


def plot_relations(models, directory, np, plt):
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5))
    definitions = (("ord", ("A_minij", "B_A"), ("A < min(i,j)", "B < A"), .2), ("hier", ("i", "j", "k"), ("i closer to A", "j closer to A", "k closer to B"), .1))
    for row, (family, keys, names, margin) in enumerate(definitions):
        for column, field in enumerate(("basic", "margin")):
            series = []
            for label in PH_ORDER:
                source = models[label]["relations"][family]
                values = [source["edges"][edge][field + "_fraction"] for edge in keys] + [source["joint_" + field + "_fraction"]]
                series.append((DISPLAY[label], values, MODEL_COLOR[label]))
            ax = axes[row, column]
            _grouped_bars(ax, (*names, "Joint"), series, np)
            ax.set_title(f"{family.upper()}: " + ("basic order" if field == "basic" else f"margin >= {margin}"))
            ax.tick_params(axis="x", labelsize=8)
            if row == column == 0:
                ax.legend(fontsize=8, loc="lower left")
    fig.suptitle("03  Ancestor constraints | rates conditioned on A != B", fontsize=12)
    fig.tight_layout()
    _save(fig, directory, "03_relations", plt)


def plot_roles(models, directory, np, plt):
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5))
    items = [(label, role, models[label]["relations"]["roles"][role]) for label in PH_ORDER for role in ("A", "B")]
    labels = [f"{'best' if label.endswith('best') else 'e300'} {role}" for label, role, _ in items]
    x = np.arange(len(items))
    baseline = np.zeros(len(items))
    for index, (scale, color) in enumerate(((200, VIEW_COLOR["p200"]), (400, VIEW_COLOR["p400"]), (600, VIEW_COLOR["p600"]))):
        values = []
        for _, _, stat in items:
            fractions = stat["scale_fractions"]
            values.append(100 * (fractions.get(str(scale), fractions.get(scale, 0)) if isinstance(fractions, dict) else fractions[index]))
        axes[0, 0].bar(x, values, bottom=baseline, color=color, label=f"P{scale}")
        baseline += np.array(values)
    axes[0, 0].set_title("Part scale among valid selections")
    axes[0, 0].set_ylim(0, 103)
    axes[0, 0].legend(fontsize=8)
    current = np.array([100 * stat["current_fraction"] for _, _, stat in items])
    queue = np.array([100 * stat["queue_fraction"] for _, _, stat in items])
    axes[0, 1].bar(x, current, color="#56b4e9", label="Current batch")
    axes[0, 1].bar(x, queue, bottom=current, color="#666666", label="Saved queue")
    axes[0, 1].set_title("Candidate source")
    axes[0, 1].set_ylim(0, 103)
    axes[0, 1].legend(fontsize=8)
    axes[1, 0].bar(x, [100 * stat["self_source_fraction"] for _, _, stat in items], color=[MODEL_COLOR[label] for label, _, _ in items])
    axes[1, 0].set_title("Source object belongs to its descendants")
    axes[1, 0].set_ylim(0, 103)
    for index, (_, _, stat) in enumerate(items):
        depth = stat["depth"]
        mid, lo, hi = (number(depth.get(key)) for key in ("median", "p25", "p75"))
        if None not in (mid, lo, hi):
            axes[1, 1].errorbar(index, mid, yerr=[[max(0, mid - lo)], [max(0, hi - mid)]], fmt="o", color=MODEL_COLOR[items[index][0]], capsize=5)
    axes[1, 1].set_title("Selected ancestor depth: median and IQR")
    for ax in axes.flat:
        ax.set_xticks(x, labels)
        ax.grid(axis="y", alpha=.15)
    for ax in (axes[0, 0], axes[0, 1], axes[1, 0]):
        ax.set_ylabel("Valid selections (%)")
    axes[1, 1].set_ylabel("Depth from origin")
    fig.suptitle("04  Roles are per-relation selections, not permanent part identities", fontsize=12)
    fig.tight_layout()
    _save(fig, directory, "04_roles", plt)


def plot_angles(tables, directory, np, plt):
    rows = tables["angle_hist.csv"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5), sharex=True, sharey="row")
    available_metrics = {row["metric"] for row in rows}
    descendant_metric = next((key for key in ("descendant_angles", "descendant", "descendants") if key in available_metrics), None)
    if "nearest" not in available_metrics or descendant_metric is None:
        raise ValueError(f"Required angle histograms not present: {available_metrics}")
    for column, label in enumerate(PH_ORDER):
        for row_number, metric in enumerate(("nearest", descendant_metric)):
            ax = axes[row_number, column]
            for group in ANGLE_GROUPS:
                subset = sorted((entry for entry in rows if entry["model"] == label and entry["group"] == group and entry["metric"] == metric), key=lambda entry: float(entry["bin_left"]))
                if not subset:
                    continue
                x = [(float(entry["bin_left"]) + float(entry["bin_right"])) / 2 for entry in subset]
                ax.step(x, [float(entry["percentage"]) for entry in subset], where="mid", color=GROUP_COLOR[group], label=ANGLE_TEXT[group])
            ax.set_title(DISPLAY[label] + (": nearest class center" if row_number == 0 else ": descendant class centers (pooled)"))
            ax.grid(alpha=.15)
            ax.legend(fontsize=7)
            ax.set_xlim(0, 180)
        axes[1, column].set_xlabel("Direction angle (degrees)")
    axes[0, 0].set_ylabel("Within-group percentage (%)")
    axes[1, 0].set_ylabel("Within-group percentage (%)")
    fig.suptitle("05  Ancestor angles | all 9,840 clean wholes define the class centers", fontsize=12)
    fig.tight_layout()
    _save(fig, directory, "05_angles", plt)


def report_markdown(models):
    """All statements below are populated from measured aggregate values."""
    lines = [
        "# PartHier v1嵌入审查结果", "",
        "四份冻结权重使用相同9840个训练对象、一个固定同中心嵌套视图；全量clean whole建立类别中心，未另分参考集。没有参数更新或BN校准。", "",
        "A/B按正常ijk挖掘后选择，使用各检查点保存队列与当前part；历史队列和新eval特征共同构成保存状态读出，**不是精确训练重放**。以下祖先满足率只以A≠B的有效关系为分母。", "",
        "## 1. whole与part的分布", "",
        "深度为到原点的双曲距离，各条曲线分别归一化；clean与覆写whole分开读。下表为深度中位数，不能替代逐对象链检查。", "",
        "| 权重 | clean whole | 覆写whole | p200 | p400 | p600 |", "|---|---:|---:|---:|---:|---:|",
    ]
    for label in ORDER:
        depth = models[label]["depth"]
        lines.append("| " + ZH[label] + " | " + " | ".join(fmt(depth[view]["median"]) for view in VIEWS) + " |")
    lines += ["", "![数量、百分比与累积分布](figures/01_depth.png)", "", "## 2. Lchain", "", "每个对象检查三条相邻边。基本次序为深度差>0，设计间隔为各≥0.2；下表是三条边联合满足率。", "", "| 权重 | 基本次序联合 | 0.2间隔联合 |", "|---|---:|---:|"]
    for label in ORDER:
        chain = models[label]["chain"]
        lines.append(f"| {ZH[label]} | {pct(chain['joint_basic_fraction'])} | {pct(chain['joint_margin_fraction'])} |")
    lines += ["", "![局部链达标与深度差](figures/02_chain.png)", "", "## 3. Lord与HIER距离关系", "", "Lord要求A浅于i/j且B浅于A，深度间隔各0.2；HIER要求i/j更近A、k更近B，距离间隔各0.1。B浅于k仅作诊断，不算第三项ord。", "", "| 权重 | 有效/全部关系 | A=B | ord间隔联合 | HIER间隔联合 | 两者同时 |", "|---|---:|---:|---:|---:|---:|"]
    for label in PH_ORDER:
        relation = models[label]["relations"]
        lines.append(f"| {ZH[label]} | {intfmt(relation['valid_draws'])}/{intfmt(relation['draws'])} | {pct(relation['collision_fraction'])} | {pct(relation['ord']['joint_margin_fraction'])} | {pct(relation['hier']['joint_margin_fraction'])} | {pct(relation['radial_and_distance_margin_fraction'])} |")
    lines += ["", "![祖先径向和距离目标](figures/03_relations.png)", "", "## 4. A/B由谁承担", "", "下图按有效选择事件计数。同一条part可被多次选择或兼任不同角色；队列来源占比也受候选条目数量影响。独立条目使用量及来源比例见[角色表](tables/roles.csv)。", "", "![尺度、来源和深度](figures/04_roles.png)", "", "## 5. 类别中心角度", "", "分别看同类pair、异类pair，以及含两类/三类的triple。下表为到最近类中心的中位数/P90；到后代类中心的角度另见图与[角度表](tables/angles.csv)。", "", "| 关系组 | best中位/P90 | e300中位/P90 |", "|---|---:|---:|"]
    angle_zh = {"A_same": "A：i/j同类", "A_cross": "A：i/j异类", "B2": "B：两类triple", "B3": "B：三类triple"}
    for group in ANGLE_GROUPS:
        values = [models[label]["relations"]["angles"].get(group, {}).get("nearest", {}) for label in PH_ORDER]
        lines.append(f"| {angle_zh[group]} | {qpair(values[0])} | {qpair(values[1])} |")
    lines += ["", "![A/B类别中心角度](figures/05_angles.png)", "", "原intra附表：[三尺度径向及对比间隔](tables/intra.csv)。图表细项另见[局部链](tables/chain.csv)、[关系目标](tables/relation.csv)和[汇总JSON](summary.json)。", "", "原版与PartHier不是匹配H0。以上只描述这些权重的几何与目标兑现程度；排序、角度或类别绑定不能单独证明形态树，也不能单独解释test下降。", ""]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    source = args.analysis_dir / "summary.json"
    summary = json.loads(source.read_text(encoding="utf-8"))
    models = validate_summary(summary, expected_count=9840)
    for label in ORDER:
        for field in ("joint_basic_fraction", "joint_margin_fraction"):
            _fraction_check(models[label]["chain"][field], label + "/chain/" + field)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    figures = args.out_dir / "figures"
    figures.mkdir(exist_ok=True)
    tables = publication_tables(args.analysis_dir, args.out_dir, models)
    np, plt = _plotting()
    plot_depth(tables, models, figures, np, plt)
    plot_chain(models, figures, np, plt)
    plot_relations(models, figures, np, plt)
    plot_roles(models, figures, np, plt)
    plot_angles(tables, figures, np, plt)
    (args.out_dir / "report.md").write_text(report_markdown(models), encoding="utf-8")
    (args.out_dir / "summary.json").write_text(json.dumps({"version": summary.get("version"), "models": public_models(models)}, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    acceptance = {"input_summary_sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "expected_objects_per_view": 9840, "figure_groups": 5, "figure_files": sorted(path.name for path in figures.iterdir()), "public_tables": sorted(path.name for path in (args.out_dir / "tables").iterdir()), "queue_semantics": "saved historical queue plus current eval part features; not training replay", "model_weights_loaded": False, "object_ids_published": False}
    (args.out_dir / "report_acceptance.json").write_text(json.dumps(acceptance, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": "completed", "report": str(args.out_dir / "report.md"), "figure_groups": 5}, ensure_ascii=False))


if __name__ == "__main__":
    main()
