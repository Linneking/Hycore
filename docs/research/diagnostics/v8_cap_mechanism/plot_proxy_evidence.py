"""Plot anonymous aggregate evidence for V8 proxy cap mechanisms.

Reads existing private arrays/JSON only. No torch, model forward, GPU, or
optimizer update. The output contains no object/proxy identities or paths.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


COLORS = {"both_eps8": "#286d9b", "both_eps4": "#d55e00",
          "both_lr001": "#009e73", "both_nohook": "#8c6bb1",
          "both_whole_shell": "#bf8b00", "sample_eps8": "#56b4e9",
          "proxy_eps8": "#a35362"}
NAMES = {"both_eps8": "合用 · eps 1e−8", "both_eps4": "合用 · eps 1e−4",
         "both_lr001": "合用 · LR .001", "both_nohook": "合用 · 去 hook",
         "both_whole_shell": "合用 · whole 等深",
         "sample_eps8": "仅 sample · eps 1e−8", "proxy_eps8": "仅 proxy · eps 1e−8"}
MOMENTS = ["saved_native", "zero_m", "tangent_m", "isotropic_v", "tangent_m_isotropic_v"]
MOMENT_NAMES = ["保留原状态", "m 清零", "m 切向化", "v 行内均一", "m 切向 + v 均一"]


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_csv(path, rows):
    if not rows:
        raise ValueError("Refusing empty aggregate table")
    fields = list(rows[0])
    if any(set(row) != set(fields) for row in rows):
        raise ValueError("CSV row schemas differ")
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def save(fig, root, name):
    fig.savefig(root / (name + ".png"), dpi=180, bbox_inches="tight",
                metadata={"Software": "HyCoRe aggregate scientific audit"})
    fig.savefig(root / (name + ".svg"), bbox_inches="tight",
                metadata={"Date": None, "Creator": "HyCoRe aggregate scientific audit"})
    svg = root / (name + ".svg")
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text(encoding="utf-8").splitlines()) + "\n", encoding="utf-8")
    plt.close(fig)


def style_axis(ax):
    ax.grid(axis="y", color="#dce3e8", linewidth=.65)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)


def norm_summary(values):
    values = np.asarray(values, dtype=np.float64)
    q = np.quantile(values, [0, .05, .25, .5, .75, .95, 1])
    return dict(count=len(values), minimum=q[0], p05=q[1], p25=q[2], median=q[3],
                p75=q[4], p95=q[5], maximum=q[6], mean=float(values.mean()),
                std=float(values.std()), cap_count=int(np.sum(values > 2.3)))


def figure_one(work, figures, tables):
    epochs = [0, 1, 5, 20, 100, 300]
    rows, arrays = [], {}
    for epoch in epochs:
        with np.load(work / "complete_inputs" / (f"e{epoch:03d}_whole_cache.npz"),
                     allow_pickle=False) as cache:
            norms = np.linalg.norm(cache["proxy_tangent"].astype(np.float64), axis=1)
        arrays[epoch] = norms
        rows.append({"model": "V8", "epoch": epoch, **norm_summary(norms)})
    original = {}
    for dataset in ["cub", "cars", "sop"]:
        with np.load(work / "original_optimizer" / dataset / "proxy_adam.npz",
                     allow_pickle=False) as cache:
            values = np.linalg.norm(cache["proxy_tangent"].astype(np.float64), axis=1)
        original[dataset] = values
        epoch = read(work / "original_optimizer" / dataset / "metadata.json")["epoch"]
        rows.append({"model": "HIER " + dataset.upper(), "epoch": epoch,
                     **norm_summary(values)})
    write_csv(tables / "proxy_raw_norm_distributions.csv", rows)
    fig, axes = plt.subplots(1, 2, figsize=(14.7, 5.3), layout="constrained")
    fig.suptitle("代理 raw 切空间模长：forward cap 不会限制保存参数", fontsize=18,
                 fontweight="bold")
    ax = axes[0]
    qs = np.array([[np.min(arrays[e]), *np.quantile(arrays[e], [.05, .5, .95]),
                    np.max(arrays[e])] for e in epochs])
    ax.fill_between(epochs, qs[:, 0], qs[:, 4], color="#286d9b", alpha=.09, label="min–max")
    ax.fill_between(epochs, qs[:, 1], qs[:, 3], color="#286d9b", alpha=.24, label="5–95%")
    ax.plot(epochs, qs[:, 2], "o-", color="#286d9b", label="中位数")
    ax.axhline(2.3, color="#b8333a", ls="--", lw=1.5, label="forward cap = 2.3")
    ax.set_xscale("symlog", linthresh=1)
    ax.set(xlabel="原 V8 已完成 epoch（稀疏快照）", ylabel="raw 切空间模长",
           title="A  原 V8 真实训练轨迹", ylim=(0, 12), xlim=(-.15, 350))
    ax.set_xticks(epochs, labels=[str(e) for e in epochs])
    for i, e in enumerate(epochs):
        ax.annotate(str(rows[i]["cap_count"]) + "/512", (e, qs[i, 4]),
                    xytext=(0, 7), textcoords="offset points", ha="center", fontsize=9)
    ax.legend(loc="upper left", fontsize=9)
    ax.text(.98, .02, "数字：raw 模长 > 2.3 的代理数\n连线不代表每轮均有几何快照",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=9, color="#53616b")
    style_axis(ax)
    ax = axes[1]
    values = [original[k] for k in ["cub", "cars", "sop"]] + [arrays[300]]
    names = ["原版 CUB best", "原版 Cars best", "原版 SOP best", "V8 e300"]
    colors = ["#009e73", "#56b4e9", "#8c6bb1", "#d55e00"]
    bins = np.linspace(1.5, 7.4, 74)
    for name, vector, color in zip(names, values, colors):
        ax.hist(vector, bins=bins, density=True, histtype="step", lw=2, color=color,
                label=f"{name} · 中位 {np.median(vector):.3f}")
    ax.axvline(2.3, color="#b8333a", ls="--", lw=1.5)
    ax.set(title="B  best / e300 的 raw 分布", xlabel="raw 切空间模长", ylabel="概率密度")
    ax.legend(fontsize=9, loc="upper right")
    ax.text(.57, .61, "越 cap：CUB 9 · Cars 1\nSOP 497 · V8 510（均 /512）\n不同任务、优化器及训练预算",
            transform=ax.transAxes, ha="center", va="top", fontsize=9, color="#53616b")
    style_axis(ax)
    save(fig, figures, "01_proxy_raw_norm")


def figure_two(work, figures, tables):
    raw = read(work / "moment_controls" / "summary.json")["rows"]
    metrics = ["count", "grad_gradient_norm_median", "grad_signed_radial_gradient_mean",
               "step_norm_median", "signed_radial_step_mean", "exact_norm_change_mean",
               "second_order_excess_mean", "outward_actual_update_fraction"]
    rows = []
    for item in raw:
        rows.append({key: item[key] for key in
                     ["case", "batch_seed", "gradient_mode", "variant", "group"] + metrics})
    write_csv(tables / "proxy_moment_interventions.csv", rows)
    fig, axes = plt.subplots(2, 2, figsize=(14.7, 9.0), layout="constrained")
    fig.suptitle("越 cap 后为什么仍增模：历史一阶矩与有限角向步分开检验", fontsize=18,
                 fontweight="bold")
    for row, (mode, label) in enumerate([
            ("current_both", "当前 sample + proxy 梯度"),
            ("zero_current", "当前梯度置零；仍继承历史 Adam 状态")]):
        sets = [[item for item in rows if item["case"] == "v8_e5"
                 and item["gradient_mode"] == mode and item["variant"] == variant
                 and item["group"] == "clipped"] for variant in MOMENTS]
        if any(len(group) != 2 for group in sets):
            raise ValueError("Expected two matched batch seeds per moment condition")
        avg = lambda field: np.array([np.mean([x[field] for x in group]) for group in sets])
        x = np.arange(5)
        ax = axes[row, 0]
        first, second, actual = [avg(field) * 1000 for field in
                                 ["signed_radial_step_mean", "second_order_excess_mean",
                                  "exact_norm_change_mean"]]
        ax.bar(x - .19, first, width=.36, color="#286d9b", label="一阶径向位移")
        ax.bar(x + .19, second, width=.36, color="#d55e00", label="有限步二阶增模")
        ax.scatter(x, actual, color="#172b3a", marker="D", s=33, zorder=3, label="实际 Δ模长")
        ax.axhline(0, color="#687983", lw=.8)
        ax.set_xticks(x, MOMENT_NAMES, rotation=12)
        ax.set(ylabel="每代理平均变化 × 10⁻³", title=("A  " if row == 0 else "C  ") + label,
               ylim=(-.52, 1.16))
        for pos, val in zip(x, actual):
            ax.annotate(f"{val:+.2f}", (pos, val), xytext=(0, 9 if val >= 0 else -15),
                        textcoords="offset points", ha="center", fontsize=9)
        if row == 0:
            ax.legend(fontsize=9, loc="upper right")
        style_axis(ax)
        ax = axes[row, 1]
        step = avg("step_norm_median")
        bars = ax.bar(x, step, color=["#a3b5c1", "#c3cbd0", "#009e73", "#8c6bb1", "#4d8b73"])
        for bar, val in zip(bars, step):
            ax.text(bar.get_x() + bar.get_width()/2, val + .00035, f"{val:.4f}",
                    ha="center", fontsize=9)
        ax.set_xticks(x, MOMENT_NAMES, rotation=12)
        ax.set(ylabel="单代理更新范数的中位数", ylim=(0, .024),
               title=("B  " if row == 0 else "D  ") + "m 切向化仍保留主要步幅")
        style_axis(ax)
    fig.supxlabel("原 V8 e5，固定 clean64，507 个越 cap 代理；两组固定 batch seed 的均值。\n"
                  "负 Δ模长表示向内。仅一步状态干预；不是完整训练因果消融。", fontsize=10)
    save(fig, figures, "02_proxy_moment_causality")


def read_trajectories(work, group):
    trajectories = {}
    for path in sorted((work / group).glob("trajectory_*.json")):
        data = read(path)
        if not data or len({(x["case"], x["variant"]) for x in data}) != 1:
            raise ValueError("Inconsistent trajectory identity")
        trajectories[data[0]["case"], data[0]["variant"]] = data
    return trajectories


def trajectory_table(trajectories, destination):
    fields = ["step", "cap_count", "norm_median", "norm_max", "case", "variant", "eps", "lr",
              "hook", "whole_shell", "loss", "gradient_norm_median", "descent_outward_fraction",
              "actual_norm_change_mean", "first_order_radial_mean", "second_order_excess_mean"]
    rows = [{key: item.get(key, "") for key in fields}
            for trajectory in trajectories.values() for item in trajectory]
    write_csv(destination, rows)


def figure_three(work, figures, tables):
    trajectories = read_trajectories(work, "gpu_short")
    if len(trajectories) != 10 or any(data[-1]["step"] != 128 for data in trajectories.values()):
        raise ValueError("Ten complete 128-step trajectories required")
    trajectory_table(trajectories, tables / "proxy_short128_trajectories.csv")
    fig = plt.figure(figsize=(14.7, 10.4), layout="constrained")
    grid = fig.add_gridspec(3, 2, height_ratios=[1, .19, 1.3])
    axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1]), fig.add_subplot(grid[2, :])]
    legend_ax = fig.add_subplot(grid[1, :])
    legend_ax.axis("off")
    fig.suptitle("固定 whole 特征，重新初始化代理：128 步拆分影响来源", fontsize=18,
                 fontweight="bold")
    for case, ax in zip(["v8_e5", "v8_e40"], axes[:2]):
        for variant in ["both_eps8", "both_eps4", "both_lr001", "both_nohook",
                        "both_whole_shell", "sample_eps8", "proxy_eps8"]:
            if (case, variant) not in trajectories:
                continue
            data = trajectories[case, variant]
            ax.plot([x["step"] for x in data], [x["cap_count"] for x in data],
                    lw=2, color=COLORS[variant], label=NAMES[variant])
        ax.set(title=("A  FULL9840 e5 特征" if case.endswith("e5") else "B  FULL9840 e40 特征"),
               xlabel="仅代理更新步数", ylabel="raw 模长 > 2.3 的代理数 /512", ylim=(0, 255))
        style_axis(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    legend_ax.legend(handles, labels, ncol=4, loc="center", fontsize=9, frameon=False)
    labels, counts, colors, rows = [], [], [], []
    for case in ["v8_e5", "v8_e40"]:
        for variant in ["both_eps8", "both_eps4", "both_lr001", "both_nohook",
                        "both_whole_shell", "sample_eps8", "proxy_eps8"]:
            if (case, variant) not in trajectories:
                continue
            data = trajectories[case, variant]
            last = data[-1]
            labels.append(case.replace("v8_", "") + " · " + NAMES[variant])
            counts.append(last["cap_count"])
            colors.append(COLORS[variant])
            rows.append({key: last[key] for key in ["case", "variant", "step", "eps", "lr", "hook",
                                                    "whole_shell", "cap_count", "norm_median", "norm_max"]})
    write_csv(tables / "proxy_short128_endpoints.csv", rows)
    ax = axes[2]
    y = np.arange(len(labels))
    ax.barh(y, counts, color=colors, height=.73)
    for pos, val in zip(y, counts):
        ax.text(val + 3, pos, str(val), va="center", fontsize=10)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.axhline(6.5, color="#73848e", lw=.8)
    ax.set(xlabel="第 128 步越 cap 数 /512", title="C  同预算终点：eps、LR、hook、whole 半径与两个损失分项", xlim=(0, 260))
    ax.grid(axis="x", color="#dce3e8", linewidth=.65)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    fig.supxlabel("每组固定 320 个训练对象的缓存特征；全部从同一随机代理与空 Adam 状态开始。\n"
                  "仅代理短诊断；没有继续更新编码器，不代表 128 步主训练。", fontsize=10)
    save(fig, figures, "03_proxy_short_controls")


def figure_four(work, figures, tables):
    trajectories = read_trajectories(work, "gpu_long")
    if len(trajectories) != 2 or any(data[-1]["step"] != 1024 for data in trajectories.values()):
        raise ValueError("Two complete 1024-step trajectories required")
    trajectory_table(trajectories, tables / "proxy_long1024_trajectories.csv")
    scores = {variant: read(work / "gpu_long" / ("score_v8_e5_" + variant + ".json"))
              for variant in ["both_eps8", "both_eps4"]}
    score_rows = []
    for variant, score in scores.items():
        for phase in ["initial", "final"]:
            score_rows.append({"variant": variant, "phase": phase, "step": 0 if phase == "initial" else 1024,
                               **score[phase]})
    write_csv(tables / "proxy_long1024_fixed_scores.csv", score_rows)
    fig, axes = plt.subplots(1, 3, figsize=(16.4, 5.5), layout="constrained")
    fig.suptitle("1024 步固定特征诊断：减少撞 cap 与收敛速度同时评估", fontsize=18,
                 fontweight="bold")
    for variant in ["both_eps8", "both_eps4"]:
        data = trajectories["v8_e5", variant]
        color = COLORS[variant]
        steps = [x["step"] for x in data]
        axes[0].plot(steps, [x["cap_count"] for x in data], color=color, lw=2, label=NAMES[variant])
        axes[0].annotate(str(data[-1]["cap_count"]), (1024, data[-1]["cap_count"]),
                         xytext=(-4, 9), textcoords="offset points", ha="right", fontsize=11, color=color)
        axes[1].plot(steps, [x["norm_median"] for x in data], color=color, lw=2, label=NAMES[variant] + " 中位数")
        axes[1].plot(steps, [x["norm_max"] for x in data], color=color, lw=1.4, ls=":", label="最大值")
    axes[0].set(title="A  越 cap 数", xlabel="仅代理更新步数", ylabel="代理数 /512", ylim=(0, 512))
    axes[0].legend(fontsize=9, loc="upper left")
    axes[1].axhline(2.3, color="#687983", ls="--", lw=1, label="cap 2.3")
    axes[1].set(title="B  raw 参数模长", xlabel="仅代理更新步数", ylabel="raw 切空间模长", ylim=(1.4, 7.5))
    axes[1].legend(fontsize=8.5, loc="upper left", ncol=1)
    ax = axes[2]
    pos = np.arange(2)
    initials = [scores[v]["initial"]["total"] for v in ["both_eps8", "both_eps4"]]
    finals = [scores[v]["final"]["total"] for v in ["both_eps8", "both_eps4"]]
    ax.bar(pos - .18, initials, width=.35, color="#b4c3ce", label="更新前")
    bars = ax.bar(pos + .18, finals, width=.35, color=[COLORS[v] for v in ["both_eps8", "both_eps4"]], label="1024 步后")
    for bar, val in zip(bars, finals):
        ax.text(bar.get_x() + bar.get_width()/2, val + .008, f"{val:.5f}", ha="center", fontsize=10)
    for i, val in enumerate(initials):
        ax.text(i - .18, val + .008, f"{val:.5f}", ha="center", fontsize=9)
    ax.set_xticks(pos, ["eps 1e−8", "eps 1e−4"])
    ax.set(title="C  相同冻结评分面板", ylabel="加权 HIER 损失（分项各 ×0.5）", ylim=(0, .46))
    ax.legend(fontsize=9, loc="upper right")
    for ax in axes:
        style_axis(ax)
    fig.supxlabel("同一 FULL9840 e5 特征、代理初始化与随机流；LR 均 .01，hook 与 cap 保留。\n"
                  "两组损失均下降，eps 1e−4 较慢；这里只检验代理优化，不检验完整训练分类收益。", fontsize=10)
    save(fig, figures, "04_proxy_long_eps")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    matplotlib.rcParams.update({"font.family": ["Microsoft YaHei", "DejaVu Sans"],
                                "font.size": 11, "axes.unicode_minus": False,
                                "svg.fonttype": "none", "svg.hashsalt": "hycore-cap-causes-v1"})
    figures, tables = args.output / "figures", args.output / "tables"
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    figure_one(args.work, figures, tables)
    figure_two(args.work, figures, tables)
    figure_three(args.work, figures, tables)
    figure_four(args.work, figures, tables)
    print(json.dumps({"status": "completed", "figure_groups": 4,
                      "csv_count": 6, "private_identity_exported": False}))


if __name__ == "__main__":
    main()
