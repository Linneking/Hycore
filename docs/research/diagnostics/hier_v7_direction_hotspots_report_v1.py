"""Render private scientific figures from the frozen V7 direction/hotspot audits.

No model forward or training. Full object identities stay in the private output;
public_summary.json deliberately contains aggregates and anonymous H1-H4 only.
Requires NumPy and Matplotlib. Run with --evolution, --hotspots, --spec, --output.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import os
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def setup_style():
    font = Path("C:/Windows/Fonts/msyh.ttc")
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams.update({"axes.unicode_minus": False, "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "figure.dpi": 110, "savefig.dpi": 170})


def render(evolution_path, hotspot_path, spec_path, output, narrative=None):
    evolution_path, hotspot_path, spec_path = map(Path, [evolution_path, hotspot_path, spec_path])
    evo, hot, spec = map(read_json, [evolution_path, hotspot_path, spec_path])
    out = Path(output)
    if out.exists():
        raise FileExistsError("Fresh output directory required")
    out.mkdir(parents=True)
    (out / "figures").mkdir()
    setup_style()
    snapshots = evo["snapshots"]
    epochs = np.array([s["epoch"] for s in snapshots])
    names = spec["class_names"]
    by_epoch = {s["epoch"]: s for s in snapshots}
    final = snapshots[-1]
    if final["epoch"] != hot["final_hotspot_epoch"]:
        raise ValueError("Final epoch differs between audits")
    saved_figures = []

    def save(fig, name):
        fig.savefig(out / "figures" / (name + ".png"), bbox_inches="tight")
        fig.savefig(out / "figures" / (name + ".svg"), bbox_inches="tight")
        plt.close(fig)
        saved_figures.append(name)

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8), constrained_layout=True)
    ax = axes[0, 0]
    for key, label, color in [("whole_origin_depth", "whole", "#2563eb"),
                              ("proxy_origin_depth", "代理（全部）", "#d97706")]:
        ax.fill_between(epochs, [s[key]["p05"] for s in snapshots],
                        [s[key]["p95"] for s in snapshots], color=color, alpha=.12)
        ax.plot(epochs, [s[key]["median"] for s in snapshots], "o-", label=label, color=color)
    ax.set(xlabel="已保存训练轮次", ylabel="原点双曲深度 d0", title="径向变化：中位数与5–95%区间")
    ax.legend()
    ax = axes[0, 1]
    ax.plot(epochs, [s["whole_directions"]["loo_direction_accuracy_macro"] * 100 for s in snapshots], "o-", color="#2563eb")
    ax.set(xlabel="已保存训练轮次", ylabel="训练对象方向LOO宏准确率 (%)", ylim=(0, 102), title="whole类别方向很早开始分离")
    ax = axes[1, 0]
    ax.plot(epochs, [s["whole_directions"]["class_balanced_common_resultant_length"] for s in snapshots], "o-", label="whole（40类等权）", color="#2563eb")
    ax.plot(epochs, [s["proxy_direction_dispersion"]["all_proxy_common_resultant_length"] for s in snapshots], "o-", label="代理（512个等权）", color="#d97706")
    ax.set(xlabel="已保存训练轮次", ylabel="公共方向长度 R：0=分散，1=同向", ylim=(0, 1.02), title="whole散开时，代理反而增加公共方向成分")
    ax.legend()
    ax = axes[1, 1]
    ax.plot(epochs, [s["proxy_class_affinity"]["all"]["top32_dominant_purity"]["mean"] * 100 for s in snapshots], "o-", label="代理方向top32同类比例", color="#059669")
    ax.plot(epochs, [s["proxy_class_affinity"]["all"]["nearest_center_angle_deg"]["median"] for s in snapshots], "o-", label="到最近类中心角度中位数 (°)", color="#9333ea")
    ax.set(xlabel="已保存训练轮次", ylabel="百分数或角度（见图例）", title="检索同类比例高，仍可远离类簇方向")
    ax.legend()
    fig.suptitle("V7：11个冻结训练快照，8856 whole / 512代理；线只连接观测点", fontsize=14)
    save(fig, "01_evolution_overview")

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.7), constrained_layout=True)
    for ax, epoch in zip(axes, [0, 5, 20, 300]):
        a = np.load(evolution_path.parent / f"e{epoch:03d}" / "arrays.npz")
        centers = a["class_centers"]
        angles = np.degrees(np.arccos(np.clip(centers @ centers.T, -1, 1)))
        im = ax.imshow(angles, vmin=0, vmax=120, cmap="viridis")
        ax.set(title=f"e{epoch}", xlabel="类别", ylabel="类别")
    fig.colorbar(im, ax=axes, label="原256维空间中的类中心夹角 (°)", shrink=.85)
    fig.suptitle("whole类中心夹角：从公共窄锥到方向分离（相同类别顺序）")
    save(fig, "02_whole_class_center_angles")

    cm = read_csv(hotspot_path.parent / "class_metrics.csv")
    rc = read_csv(hotspot_path.parent / "retrieval_class_metrics.csv")
    final_rows = [r for r in cm if r["model"] == f"V7_e{final['epoch']}"]
    final_rows.sort(key=lambda r: float(r["depth_median"]))
    order = [int(r["class_label"]) for r in final_rows]
    rc_lookup = {(r["retrieval"], int(r["class_label"])): r for r in rc if r["model"] == f"V7_e{final['epoch']}"}
    fig, axes = plt.subplots(1, 3, figsize=(15, 11), sharey=True, constrained_layout=True)
    yy = np.arange(len(order))
    colors = ["#dc2626" if c == 15 else "#2563eb" for c in order]
    for j, row in enumerate(final_rows):
        axes[0].plot([float(row["depth_p05"]), float(row["depth_p95"])], [j, j], color=colors[j], alpha=.7)
    flower_row = next(j for j, row in enumerate(final_rows) if int(row["class_label"]) == 15)
    flower_min = float(final_rows[flower_row]["depth_min"])
    axes[0].scatter([flower_min], [flower_row], marker="x", color="#dc2626", s=45)
    axes[0].annotate(f"min {flower_min:.3f}", (flower_min, flower_row), xytext=(0, -14), textcoords="offset points", fontsize=8, color="#dc2626")
    axes[0].scatter([float(r["depth_median"]) for r in final_rows], yy, c=colors, s=22)
    axes[0].set(xlabel="whole原点双曲深度 d0", title="中位数与5–95%区间", yticks=yy, yticklabels=[names[c] for c in order])
    axes[0].invert_yaxis()
    raw = [100 * float(rc_lookup["raw_top4", c]["slot_share"]) for c in order]
    direction = [100 * float(rc_lookup["direction_top4", c]["slot_share"]) for c in order]
    axes[1].barh(yy - .18, raw, height=.35, color="#dc2626", label="原始双曲距离")
    axes[1].barh(yy + .18, direction, height=.35, color="#059669", label="仅方向/whole统一深度")
    axes[1].set(xlabel="2048检索槽位的类别份额 (%)", title="原始热点类别与方向偏好不同")
    axes[1].legend(loc="lower right")
    axes[2].scatter([float(r["center_angle_deg_mean"]) for r in final_rows], yy, c=colors)
    axes[2].set(xlabel="样本到自身类中心夹角均值 (°)", title="花盆类方向也更宽")
    fig.suptitle("e300 类别比较：花盆有极浅尾部，但 glass_box / vase 等类中位数更浅", fontsize=14)
    save(fig, "03_class_depth_and_hotspots")

    trajectory = read_csv(hotspot_path.parent / "hotspot_trajectory.csv")
    top4 = final["raw_top4"]["hottest_samples"][:4]
    hot_ids = [h["sample_id"] for h in top4]
    hot_rows = [r for r in trajectory if r["final_hotspot_definition"] == "raw_top4_slots" and int(r["canonical_id"]) in hot_ids]
    row_lookup = {(r["model"], int(r["canonical_id"])): r for r in hot_rows}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.9), constrained_layout=True)
    for j, hid in enumerate(hot_ids):
        rr = [row_lookup[f"V7_e{epoch}", hid] for epoch in epochs]
        axes[0].plot(epochs, [float(r["depth"]) for r in rr], "o-", label=f"H{j+1} / ID {hid}")
        axes[1].plot(epochs, [float(r["common_depth_percentile"]) * 100 for r in rr], "o-")
    axes[0].set(xlabel="已保存轮次", ylabel="d0", title="固定末期四热点：半径并非一直浅")
    axes[0].legend(fontsize=8)
    axes[1].set(xlabel="已保存轮次", ylabel="深度百分位：0=浅，100=深", ylim=(0, 102), title="e200仍深；e300变成最浅四个")
    model_names = ["original_best", "original_last", "source_b32_best", "source_b32_last", "b64_best", "b64_last", "V7_e300"]
    plot_model_names = ["ORIG-B0-32 best", "ORIG-B0-32 last", "V6-B0-32 best", "V6-B0-32 last", "V6-B0-64 best", "V6-B0-64 last", "V7 e300"]
    for j, hid in enumerate(hot_ids):
        axes[2].plot(np.arange(len(model_names)), [float(row_lookup[m, hid]["common_depth_percentile"]) * 100 for m in model_names], "o", label=f"H{j+1}")
    axes[2].set(xticks=np.arange(len(model_names)), xticklabels=plot_model_names, ylabel="共同8856对象池的深度百分位", ylim=(0, 103), title="同ID基线比较：原HyCoRe中不总是浅")
    axes[2].tick_params(axis="x", rotation=55)
    fig.suptitle("没有 e200–e300 中间嵌入快照，无法确定四对象内移的具体轮次")
    save(fig, "04_fixed_hotspot_trajectories_and_baselines")

    final_a = np.load(evolution_path.parent / f"e{final['epoch']:03d}" / "arrays.npz")
    affinity = final_a["proxy_to_whole_class_center_cosine"]
    nearest = np.argmax(affinity, axis=1)
    used = final_a["sample_used"].astype(bool)
    proxy_sort = np.lexsort((final_a["proxy_ids"], final_a["proxy_depth"], ~used, nearest))
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [3, 1]}, constrained_layout=True)
    im = axes[0].imshow(affinity[proxy_sort], aspect="auto", vmin=-.3, vmax=.8, cmap="coolwarm")
    axes[0].set(xticks=np.arange(40), xticklabels=names, ylabel="代理（按最近类别、使用状态、深度排序）", title="e300 全类中心方向亲和：不依赖top4标签")
    axes[0].tick_params(axis="x", rotation=90, labelsize=7)
    fig.colorbar(im, ax=axes[0], label="代理方向·单位类别中心")
    counts_all = np.bincount(nearest, minlength=40)
    selected = np.argsort(counts_all)[-10:][::-1]
    counts_used = np.bincount(nearest[used], minlength=40)
    axes[1].barh(np.arange(10), counts_used[selected], label="当轮 sample-used", color="#2563eb")
    axes[1].barh(np.arange(10), counts_all[selected] - counts_used[selected], left=counts_used[selected], label="当轮 sample-inactive", color="#d97706")
    axes[1].set(yticks=np.arange(10), yticklabels=[names[c] for c in selected], xlabel="代理数", title="类别亲和集中与inactive冗余")
    axes[1].invert_yaxis()
    axes[1].legend(fontsize=8)
    save(fig, "05_proxy_class_affinity")

    # Exact c=1 cosh law gives an independent geometric explanation for raw NN.
    # It compares each proxy's closest one of H1-H4 to its closest-direction whole.
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from tools.hier_postrun_audit.geometry import ball_geometry, expmap0
    final_record = next(r for r in spec["snapshots"] if r["epoch"] == final["epoch"])
    z = np.load(final_record["cache"])
    c = float(z["c"] if "c" in z else z["curvature"])
    if c != 1.:
        raise ValueError("This geometric panel explicitly requires V7 c=1")
    so, po = np.argsort(z["sample_ids"]), np.argsort(z["proxy_ids"])
    wg = ball_geometry(z["mu"][so].astype(float), c)
    pg = ball_geometry(expmap0(z["proxy_tangent"][po].astype(float), c, numeric_radius_fraction=.999), c)
    ids = z["sample_ids"][so]
    hi = np.array([np.flatnonzero(ids == hid)[0] for hid in hot_ids])
    cos = pg["direction"] @ wg["direction"].T
    di = np.argmax(cos, axis=1)
    pd, wd = pg["depth"], wg["depth"]
    d_hot = np.arccosh(np.maximum(1, np.cosh(pd[:, None]) * np.cosh(wd[hi])[None, :] - np.sinh(pd[:, None]) * np.sinh(wd[hi])[None, :] * cos[:, hi]))
    hk = hi[np.argmin(d_hot, axis=1)]
    d_hot = d_hot.min(axis=1)
    d_direction = np.arccosh(np.maximum(1, np.cosh(pd) * np.cosh(wd[di]) - np.sinh(pd) * np.sinh(wd[di]) * cos[np.arange(len(pd)), di]))
    theta_hot = np.degrees(np.arccos(np.clip(cos[np.arange(len(pd)), hk], -1, 1)))
    theta_direction = np.degrees(np.arccos(np.clip(cos[np.arange(len(pd)), di], -1, 1)))
    geometric_explanation = {"proxy_count": len(pd), "closest_of_fixed_four_hotspots_beats_closest_direction_whole_count": int((d_hot < d_direction).sum()),
                             "median_distance_advantage": float(np.median(d_direction - d_hot)),
                             "median_hotspot_angle_deg": float(np.median(theta_hot)),
                             "median_closest_direction_angle_deg": float(np.median(theta_direction)),
                             "median_closest_direction_whole_depth": float(np.median(wd[di])),
                             "formula": "cosh d=cosh d0p cosh d0w-sinh d0p sinh d0w cos(theta), c=1"}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), constrained_layout=True)
    colors = np.where(used, "#2563eb", "#d97706")
    axes[0].scatter(theta_direction, theta_hot, s=13, c=colors, alpha=.5)
    axes[0].plot([0, 120], [0, 120], "k--", lw=1)
    axes[0].set(xlabel="方向最近对象的夹角 (°)", ylabel="四热点中双曲最近对象的夹角 (°)", title="热点在方向上通常更远")
    axes[1].scatter(d_direction, d_hot, s=13, c=colors, alpha=.5)
    limits = [0, max(d_direction.max(), d_hot.max()) * 1.05]
    axes[1].plot(limits, limits, "k--", lw=1)
    axes[1].set(xlabel="代理到方向最近对象的双曲距离", ylabel="代理到四热点中最近对象的双曲距离", title=f"{(d_hot < d_direction).sum()}/{len(pd)}代理的热点双曲距离更小")
    fig.suptitle("角度较远仍可双曲距离更近：浅半径优势压过方向差异（蓝 used / 橙 inactive）")
    save(fig, "06_radial_advantage_geometry")

    public = {"schema": "v7_direction_hotspots_anonymous_summary_v1",
              "training_commit": "165981fb396eb2366dc1c0cf81ca3938219aec3d",
              "sample_count": 8856, "proxy_count": 512, "baseline_common_count": 8856,
              "baseline_full_count": 9840, "snapshot_epochs": epochs.tolist(),
              "baseline_display_names": {"original": "ORIG-B0-32-S4780", "source_b32": "V6-B0-32", "b64": "V6-B0-64"},
              "snapshots": [], "class_comparison_e300": [],
              "anonymous_final_hotspots": [], "geometric_explanation": geometric_explanation,
              "source_assertions": hot["read_only"],
              "limitations": ["Saved snapshots only; no onset inferred in missing intervals.",
                              "Train geometry descriptions are not validation/test accuracy.",
                              "Nearest-object retrieval differs from actual sampled ancestor activation.",
                              "Historical baseline protocols differ; comparisons do not isolate HIER causality.",
                              "Physical shape causes require independent object geometry evidence.",
                              "Late shared Procrustes has underdetermined subspace; no causal motion claim."]}
    for s in snapshots:
        aff = s["proxy_class_affinity"]
        public["snapshots"].append({"epoch": s["epoch"], "whole_depth": s["whole_origin_depth"],
                                   "proxy_depth": s["proxy_origin_depth"], "sample_used_count": s["usage"]["sample_used_count"],
                                   "whole_direction_geometry": s["whole_directions"],
                                   "proxy_common_resultant": s["proxy_direction_dispersion"]["all_proxy_common_resultant_length"],
                                   "proxy_top32_direction_purity": aff["all"]["top32_dominant_purity"]["mean"],
                                   "proxy_nearest_center_angle_median": aff["all"]["nearest_center_angle_deg"]["median"],
                                   "proxy_center_class_counts": aff["all"]["nearest_center_class_counts"],
                                   "proxy_robust_class_retention_exclude_both": aff["exclude_both"]["nearest_class_retention_vs_all"],
                                   "raw_coverage": s["raw_top4"]["covered_samples"],
                                   "direction_coverage": s["direction_top4"]["covered_samples"],
                                   "raw_max_quartet_repeat": s["raw_top4"]["maximum_repeated_set"],
                                   "raw_slots_in_bottom1pct": s["raw_top4_slot_fraction_bottom_radius_1pct"]})
    for row in final_rows:
        row = dict(row)
        c = int(row["class_label"])
        row["raw_top4_class"] = rc_lookup["raw_top4", c]
        row["direction_top4_class"] = rc_lookup["direction_top4", c]
        row["proxy_nearest_center_count"] = int(counts_all[c])
        public["class_comparison_e300"].append(row)
    for j, hid in enumerate(hot_ids):
        records = []
        for r in hot_rows:
            if int(r["canonical_id"]) == hid:
                records.append({k: v for k, v in r.items() if k != "canonical_id"})
        public["anonymous_final_hotspots"].append({"label": f"H{j+1}", "trajectories": records})
    first = np.load(evolution_path.parent / "e005" / "arrays.npz")
    public["fixed_proxy_e5_to_e300_center_class_retained_count"] = int((first["all__nearest_class"] == final_a["all__nearest_class"]).sum())
    public["adjacent_snapshot_class_retention"] = [{"from": t["from_epoch"], "to": t["to_epoch"], **t["affinity_changes"]["all"]} for t in evo["transitions"]]
    (out / "public_summary.json").write_text(json.dumps(public, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    (out / "manifest.json").write_text(json.dumps({"sources": [{"file": p.name, "sha256": digest(p)} for p in [evolution_path, hotspot_path, spec_path]],
                                                 "figures": saved_figures, "script_sha256": digest(__file__), "private_object_ids": True}, indent=2), encoding="utf-8")
    title = "V7 whole / 代理方向演化与样本热点审查"
    captions = ["训练期间径向与方向的总体变化", "whole类别中心方向分离", "40类径向分布、热点及方向宽度", "同一末期热点的全程与基线排名", "代理全类别方向亲和与使用分组", "浅半径如何压过方向接近程度"]
    body = [f"<h1>{title}</h1>", "<p>11个冻结快照，8856个相同训练对象，512代理；另对齐6个无HIER基线。无新前向、训练或GPU使用。图线仅连接已保存观测。</p>",
            "<p><b>结论：</b>whole类别方向早期分离且末期紧致；代理存在方向冗余和频繁类别偏好切换。原始花盆热点主要受极浅径向尾部影响；统一whole深度后仍有laptop方向热点。</p>"]
    if narrative:
        text = Path(narrative).read_text(encoding="utf-8-sig")
        text += "\n\n## 私有交付：对象身份与完整数据\n\n"
        text += "末期热点H1–H4的canonical ID依次为：" + ", ".join(map(str, hot_ids)) + "。\n\n"
        text += "原始完整审查输出见相邻 evolution_r2 与 hotspot_baseline_v1_r2 目录；数据未经删减。\n"
        (out / "report.md").write_text(text, encoding="utf-8")
        body.append("<p><a href='report.md'>完整文字报告与热点对象身份</a></p>")
    data_links = [(hotspot_path.parent / "hotspot_trajectory.csv", "固定热点的逐期/基线轨迹CSV"),
                  (hotspot_path.parent / "class_metrics.csv", "完整逐类径向/方向统计CSV"),
                  (hotspot_path.parent / "retrieval_class_metrics.csv", "逐类热点槽位与富集CSV"),
                  (evolution_path.parent / "all_proxies.csv", "完整代理方向亲和与使用CSV")]
    body.append("<p>" + " ｜ ".join(f"<a href='{html.escape(os.path.relpath(p, out).replace(os.sep, '/'))}'>{label}</a>" for p, label in data_links) + "</p>")
    for name, caption in zip(saved_figures, captions):
        body.append(f'<section><h2>{html.escape(caption)}</h2><a href="figures/{name}.png"><img src="figures/{name}.png" alt="{html.escape(caption)}"></a></section>')
    body.append("<p>完整逐对象ID、全代理邻居和逐类数据在相邻审查输出目录。数值摘要：<a href='public_summary.json'>public_summary.json</a>。训练集方向准确率不代表泛化性能；历史基线协议不同，不能据此单独归因HIER。</p>")
    page = "<!doctype html><html lang='zh-CN'><meta charset='utf-8'><title>" + title + "</title><style>body{max-width:1450px;margin:32px auto;padding:0 24px;color:#18202c;font:16px/1.6 system-ui}img{width:100%;height:auto}section{margin:42px 0}h1,h2{line-height:1.3}a{color:#2563eb}</style>" + "\n".join(body) + "</html>"
    (out / "report.html").write_text(page, encoding="utf-8")
    print(json.dumps({"completed": True, "figures": len(saved_figures), "summary": str(out / "public_summary.json"), "geometric_explanation": geometric_explanation}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evolution", required=True)
    parser.add_argument("--hotspots", required=True)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--narrative", help="Optional saved Markdown explanation; copied only into private output")
    args = parser.parse_args()
    render(args.evolution, args.hotspots, args.spec, args.output, args.narrative)
