"""CPU-only clarification of N600 whole/part radial order and cap ties.

Read the existing private 18-cache analysis config; publish aggregates only.
Usage: --config PRIVATE_ANALYSIS.json --out-dir NEW_PUBLIC_DIRECTORY
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analyze_pairs import configure_fonts, file_sha256, write_csv, write_json


def identity_sha(values):
    return hashlib.sha256(np.asarray(values, dtype="<i8").tobytes()).hexdigest()


def analyze(model, tolerance):
    source = Path(model["cache"])
    before = file_sha256(source)
    with np.load(source, allow_pickle=False) as data:
        ids = np.asarray(data["sample_ids"], dtype=np.int64).copy()
        labels = np.asarray(data["labels"], dtype=np.int64).copy()
        result = {k: model[k] for k in ("label", "stage", "condition", "kind", "epoch", "gamma", "delta", "checkpoint_sha256")}
        result.update(sample_count=len(ids), cache_sha256=before,
                      sample_ids_sha256=identity_sha(ids), labels_sha256=identity_sha(labels))
        for n in (200, 400, 600):
            whole = np.asarray(data[f"whole_depth_{n}"] if f"whole_depth_{n}" in data else data["whole_depth"], dtype=np.float64)
            part = np.asarray(data[f"part_depth_{n}"], dtype=np.float64)
            if whole.shape != ids.shape or part.shape != ids.shape or not np.isfinite(whole).all() or not np.isfinite(part).all():
                raise ValueError("Invalid paired depth arrays")
            gap = whole - part
            result[f"n{n}_mean_hinge"] = float(np.maximum(model["gamma"] / n - gap, 0).mean())
            if n != 600:
                continue
            shallow, tie, deeper = gap > tolerance, np.abs(gap) <= tolerance, gap < -tolerance
            assert int(shallow.sum() + tie.sum() + deeper.sum()) == len(ids)
            result.update(whole_mean=float(whole.mean()), part_mean=float(part.mean()),
                          gap_mean=float(gap.mean()), gap_median=float(np.median(gap)),
                          raw_gap_gt_zero_count=int((gap > 0).sum()),
                          shallow_count=int(shallow.sum()), tie_count=int(tie.sum()), deeper_count=int(deeper.sum()))
            for key in ("raw_gap_gt_zero", "shallow", "tie", "deeper"):
                result[f"{key}_percent"] = 100 * result[f"{key}_count"] / len(ids)
            if model["kind"] == "v8":
                both_cap = (whole >= 4.6 - tolerance) & (part >= 4.6 - tolerance)
                rest = ~both_cap
                result.update(both_cap_count=int(both_cap.sum()),
                              both_cap_equals_tie=bool(np.array_equal(both_cap, tie)),
                              without_both_cap_count=int(rest.sum()),
                              without_both_cap_shallow_count=int((rest & shallow).sum()),
                              without_both_cap_shallow_percent=float(100 * (rest & shallow).sum() / rest.sum()) if rest.any() else None)
            else:
                result.update(both_cap_count=None, both_cap_equals_tie=None,
                              without_both_cap_count=None, without_both_cap_shallow_count=None,
                              without_both_cap_shallow_percent=None)
    if before != file_sha256(source):
        raise RuntimeError("Source cache changed while reading")
    return result, ids, labels


def create_figure(rows, output, tolerance):
    configure_fonts()
    fig, ax = plt.subplots(figsize=(12.5, 7))
    labels = []
    positions = np.arange(len(rows))
    left = np.zeros(len(rows))
    for field, color, label in (("shallow", "#309863", "part更浅"), ("tie", "#b5b8bd", "基本同深"), ("deeper", "#ce5a54", "part更深")):
        values = np.array([r[f"{field}_percent"] for r in rows])
        ax.barh(positions, values, left=left, color=color, edgecolor="white", height=.68, label=label)
        for i, value in enumerate(values):
            if value >= 3:
                ax.text(left[i] + value / 2, i, f"{value:.2f}%", ha="center", va="center", fontsize=10, color="#141414")
        left += values
    for row in rows:
        mode = "历史BN / eval" if row["condition"] == "native_eval_panel" else "当前批BN"
        labels.append(f"{row['label']} · {mode}")
    ax.set_yticks(positions, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("同1280对象的whole–part配对比例（%）")
    ax.set_title("N600：V8当前批BN主要把径向顺序压成同壳，不能解读为大量反转", fontsize=14, pad=18)
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(.5, -.12), frameon=False)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.text(.02, .035, f"g = whole深度 − part深度；同深定义 |g| ≤ {tolerance:g}。同深不代表方向或位置相同。", fontsize=10)
    fig.text(.02, .005, "V8当前批BN的raw g>0约51%受同壳浮点微差影响，不能表示真实浅层率；真实反转仅约8%。", fontsize=10)
    fig.tight_layout(rect=(0, .1, 1, 1))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".png"), dpi=180)
    fig.savefig(output.with_suffix(".svg"))
    plt.close(fig)


def make_report(rows, output, tolerance):
    def percent(row, field):
        return f"{row[field + '_count']}（{row[field + '_percent']:.2f}%）"
    table = ["| 终点模型 | BN读出 | part更浅 | 基本同深 | part更深 |", "| --- | --- | ---: | ---: | ---: |"]
    for row in rows:
        mode = "历史/eval" if row["condition"] == "native_eval_panel" else "当前批"
        table.append(f"| {row['label']} | {mode} | {percent(row, 'shallow')} | {percent(row, 'tie')} | {percent(row, 'deeper')} |")
    original_eval, original_bn, old_eval, old_bn, full_eval, full_bn = rows
    text = fr"""# N600当前批BN顺序统计的补充说明

补充日期：2026-10-20；基础冻结诊断数据采集于2026-10-19。

这份补充纠正原报告严格使用g>0作为“顺序正确率”在大量贴cap时的解释局限。原报告正文与用户修改保持原样；本页为CPU重算已有缓存，没有模型前向或训练更新。

## 1. 比较的是同一个对象的whole和part

每行都是同一批1280个训练对象，同一对象的whole与其N600 part逐对相减。两种BN条件的对象ID及标签逐元素一致。定义：

$$
g=d_0(whole)-d_0(part).
$$

g>{tolerance:g}为part更浅，|g|≤{tolerance:g}为基本同深，g<−{tolerance:g}为part更深。这个容差仅区分数值上相等的深度，不代表满足500/N或1000/N的完整margin。**同深也不表示双曲位置相同：方向仍可不同。**

{chr(10).join(table)}

![N600三类径向顺序](figures/01_n600_radial_order_ties.png)

## 2. 为什么原版切BN后顺序改善

原版同1280对象从历史/eval改用当前批统计，whole平均深度由{original_eval['whole_mean']:.5f}降至{original_bn['whole_mean']:.5f}，part由{original_eval['part_mean']:.5f}降至{original_bn['part_mean']:.5f}。part平均收缩更多，平均whole−part间隔由{original_eval['gap_mean']:.5f}增至{original_bn['gap_mean']:.5f}，顺序正确比例64.45%→80.08%。这些是同一权重的两种读出条件，不是模型重新训练后的提升。BN使用的各层均值与方差改变了whole和part经过网络后的特征，影响不必同比例。

## 3. V8下降约三成，主要不是更多真实反转

旧V8当前批BN有{old_bn['both_cap_count']}对、FULL有{full_bn['both_cap_count']}对whole和part同时贴cap；它们恰好就是全部“基本同深”配对。原报告raw g>0得到51.25%和51.95%，其中大量g仅约10⁻⁷：微小浮点差把同壳配对分成正负，无法用来判断谁实际更浅。

按容差重算，V8真实反转仅7.81%和8.52%，低于原版当前批BN的19.92%。代价是72%–73%的配对失去可辨的径向间隔。去掉两者同时贴cap后，旧V8{old_bn['without_both_cap_shallow_count']}/{old_bn['without_both_cap_count']}（{old_bn['without_both_cap_shallow_percent']:.2f}%）、FULL {full_bn['without_both_cap_shallow_count']}/{full_bn['without_both_cap_count']}（{full_bn['without_both_cap_shallow_percent']:.2f}%）的part更浅。原版没有V8的4.6深度cap，不对其使用“同时贴cap”指标。

## 4. N600的margin较小，但不能等同于惩罚力度更弱

径向hinge为：

$$
R_N=\max(0,\gamma/N-g),\qquad
\frac{{\partial R_N}}{{\partial g}}=-1\quad\text{{（约束未满足时）}}.
$$

N600的目标间隔比N200/400小，表示允许part更接近whole；但约束未满足时，单个hinge对g的导数并未因为margin较小而减弱。这里各N均使用同一0.01系数。链式传回网络的实际梯度还取决于映射、BN和cap，不能由margin大小单独断言。

当前批BN下，旧V8的N600平均残余hinge为{old_bn['n600_mean_hinge']:.4f}，FULL为{full_bn['n600_mean_hinge']:.4f}；N200反而分别仅{old_bn['n200_mean_hinge']:.4f}和{full_bn['n200_mean_hinge']:.4f}。因此本次可确认N600更容易同壳、完整径向约束仍未满足；不能证明历史训练中“margin较小、压力较弱”就是原因。part点数改变带来的输入差异、同一个cap与其他共享目标仍需单独干预区分。

## 5. 可复核范围

脚本读取原先18份缓存，对6份终点同1280面板生成上表，并保存全部18份的匿名统计与源SHA。源缓存SHA前后不变；对象ID与标签仅保存哈希，不公开原始值。best结果也保存在JSON供复核，不增加新的GPU实验。CSV见[tables/n600_last_same1280.csv](tables/n600_last_same1280.csv)，完整统计与校验见[tables/n600_order_clarification.json](tables/n600_order_clarification.json)。
"""
    (output / "report.md").write_text(text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8-sig"))
    tolerance = float(config.get("tolerance", 1e-4))
    observations = []
    groups = {}
    for model in config["models"]:
        row, ids, labels = analyze(model, tolerance)
        observations.append(row)
        if model["condition"] in ("native_eval_panel", "BN_current_batch_only"):
            key = (model["label"], model["stage"])
            if key in groups:
                if not np.array_equal(ids, groups[key][0]) or not np.array_equal(labels, groups[key][1]):
                    raise ValueError("BN and native panels must contain identical ordered objects")
            else:
                groups[key] = (ids, labels)
    if len(observations) != 18:
        raise ValueError("Expected the 18 source observations from report59")
    order = list(dict.fromkeys(m["label"] for m in config["models"]))
    rows = [next(r for r in observations if r["label"] == label and r["stage"] == "last" and r["condition"] == condition)
            for label in order for condition in ("native_eval_panel", "BN_current_batch_only")]
    if len(rows) != 6 or {r["sample_count"] for r in rows} != {1280}:
        raise ValueError("Expected six same1280 terminal observations")
    if len({r["sample_ids_sha256"] for r in rows}) != 1 or len({r["labels_sha256"] for r in rows}) != 1:
        raise ValueError("Six panels are not aligned")
    output = args.out_dir
    output.mkdir(parents=True, exist_ok=True)
    (output / "tables").mkdir(exist_ok=True)
    write_csv(output / "tables/n600_last_same1280.csv", rows)
    write_json(output / "tables/n600_order_clarification.json", {
        "tolerance": tolerance, "same_1280_ids_and_labels_verified": True,
        "source_cache_count": len(observations), "all_source_sha_preserved": True,
        "count_definition": "g=d0(whole)-d0(part); shallow>tol, tie abs<=tol, deeper<-tol",
        "cap_definition_v8_only": "whole and part >= 4.6-tolerance",
        "no_gpu_no_forward_no_training": True, "last_same1280": rows, "all_observations": observations})
    create_figure(rows, output / "figures/01_n600_radial_order_ties", tolerance)
    make_report(rows, output, tolerance)
    print(json.dumps({"source_cache_count": len(observations), "public_rows": len(rows), "verified": True}))


if __name__ == "__main__":
    main()
