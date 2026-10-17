"""Render anonymous whole-distribution evidence; no model/GPU operations."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BLUE, ORANGE, GREEN, PURPLE = "#286d9b", "#d55e00", "#009e73", "#8c6bb1"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_csv(path, rows):
    if not rows:
        raise ValueError("Empty aggregate table")
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save(fig, figures, name):
    fig.savefig(figures / (name + ".png"), dpi=180, bbox_inches="tight")
    fig.savefig(figures / (name + ".svg"), bbox_inches="tight", metadata={"Date": None})
    svg = figures / (name + ".svg")
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text(encoding="utf-8").splitlines()) + "\n", encoding="utf-8")
    plt.close(fig)


def axis_style(ax):
    ax.grid(axis="y", alpha=.22)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)


def depth_numpy(mu, c):
    return 2 / np.sqrt(c) * np.arctanh(np.minimum(np.sqrt(c) * np.linalg.norm(mu.astype(np.float64), axis=1), 1 - 1e-7))


def figure_five(work, figures, tables):
    with (work / "complete_inputs/trajectory.csv").open(encoding="utf-8-sig", newline="") as stream:
        trajectory = list(csv.DictReader(stream))
    epochs = sorted({int(x["epoch"]) for x in trajectory})
    train = []
    rows = []
    for epoch in epochs:
        batches = [x for x in trajectory if int(x["epoch"]) == epoch]
        row = {"epoch": epoch, "whole": np.mean([float(x["whole_depth"]) for x in batches]),
               "part": np.mean([float(x["part_depth"]) for x in batches]), "batches": len(batches)}
        train.append(row)
        for role in ("whole", "part"):
            rows.append({"protocol": "old_V8_8856_train_984_validation", "condition": "actual_training_aug_crop_alias_trainBN",
                         "epoch": epoch, "role": role, "mean_depth": row[role], "median_depth": None,
                         "p05_depth": None, "p95_depth": None, "cap_fraction": None,
                         "count": len(batches), "count_unit": "global64_step_means"})
    clean = []
    for item in read(work / "complete_inputs/snapshot_summary.json"):
        epoch = int(item["epoch"])
        with np.load(work / "complete_inputs" / f"e{epoch:03d}_whole_cache.npz", allow_pickle=False) as cache:
            values = depth_numpy(cache["mu"], float(cache["c"]))
        row = {"epoch": epoch, "mean": float(values.mean()), "median": float(np.median(values)),
               "p05": float(np.quantile(values, .05)), "p95": float(np.quantile(values, .95)),
               "cap": float(np.mean(values >= 4.5999)), "count": len(values)}
        clean.append(row)
        rows.append({"protocol": "old_V8_8856_train_984_validation", "condition": "clean_first1024_evalBN_full_training_pool",
                     "epoch": epoch, "role": "whole", "mean_depth": row["mean"], "median_depth": row["median"],
                     "p05_depth": row["p05"], "p95_depth": row["p95"], "cap_fraction": row["cap"],
                     "count": len(values), "count_unit": "unique_training_objects"})
    write_csv(tables / "whole_training_clean_timeline.csv", rows)
    fig, axes = plt.subplots(1, 2, figsize=(15.8, 5.7), layout="constrained")
    fig.suptitle("旧 V8：训练时近球壳，clean / eval 读出明显变浅", fontsize=18, fontweight="bold")
    x = np.asarray([r["epoch"] for r in clean])
    ax = axes[0]
    ax.plot(epochs, [r["whole"] for r in train], color=BLUE, lw=1.8, label="实际训练 whole · 每轮均值")
    ax.plot(epochs, [r["part"] for r in train], color=GREEN, lw=1.8, label="实际训练 part · 每轮均值")
    ax.fill_between(x, [r["p05"] for r in clean], [r["p95"] for r in clean], color=ORANGE, alpha=.17, label="clean whole · 5–95%")
    ax.plot(x, [r["mean"] for r in clean], "o-", color=ORANGE, lw=1.8, label="clean whole · 全池均值")
    ax.plot(x, [r["median"] for r in clean], "x:", color="#a43b08", lw=1, label="clean whole · 全池中位")
    ax.axhline(4.6, color="#697883", ls="--", lw=1, label="cap 对应深度4.6")
    ax.set(xlabel="已完成 epoch", ylabel="原点双曲深度", ylim=(2.1, 4.85), title="A  同一旧 V8，不同读出条件")
    ax.legend(fontsize=8.5, loc="lower left")
    ax = axes[1]
    ax.plot(x, [100 * r["cap"] for r in clean], "o-", color=ORANGE, lw=2)
    for r in (clean[0], clean[-1]):
        ax.annotate(f"{100*r['cap']:.2f}%", (r["epoch"], 100 * r["cap"]),
                    xytext=(5, 8), textcoords="offset points", fontsize=10)
    ax.set(xlabel="已完成 epoch · 稀疏 clean 快照", ylabel="whole 输出 cap 比例（%）",
           ylim=(-4, 109), title="B  clean / eval 全8856对象的球壳占比")
    ax.text(.04, .2, "训练日志保存批内均深度，\n不能据此倒算训练 cap 比例。", transform=ax.transAxes, fontsize=10, color="#52616a")
    for ax in axes:
        ax.set_xscale("symlog", linthresh=1)
        ax.set_xticks([0, 1, 5, 20, 100, 300], labels=["0", "1", "5", "20", "100", "300"])
        axis_style(ax)
    fig.supxlabel("仅旧 V8：8856训练 / 984验证选模。训练含增强、裁剪覆写和 train BN；clean 是 first1024 / eval BN。\n"
                  "两条曲线的差异尚未分解为单一原因；稀疏快照间连线仅辅助阅读。", fontsize=10)
    save(fig, figures, "05_whole_training_vs_clean")
    return {"training_e300": train[-1], "clean_e300": clean[-1]}


def figure_six(work, figures, tables):
    rows = []
    for dataset in ("cub", "cars", "sop"):
        data = read(work / ("head_" + dataset) / "summary.json")
        rows.append({"name": "原版 " + dataset.upper() + " best", "source": "original_HIER_best",
                     "epoch": data["source_epoch"], "objects": data["anchors"],
                     "condition": "clean_training_anchors_evalBN", "norm": data["pre_to_poincare_norm"],
                     "depth": data["postcap_depth"], "cap_fraction": data["precap_over_2_3_fraction"], "color": GREEN})
    for data in read(work / "gpu_whole/summary.json")["stages"]:
        if data["child_count"] != 400:
            continue
        rows.append({"name": "新 FULL9840 HIER " + data["stage"], "source": "new_FULL9840_test_selected_HIER",
                     "epoch": int(data["stage"][1:]), "objects": data["objects"],
                     "condition": "clean_alias400_evalBN_fixed64", "norm": data["raw_whole_tangent_norm"],
                     "depth": data["whole_depth"], "cap_fraction": data["whole_precap_over_2_3_count"] / data["objects"], "color": BLUE})
    aggregate = []
    for row in rows:
        aggregate.append({"model": row["name"], "source": row["source"], "epoch": row["epoch"],
                          "objects": row["objects"], "condition": row["condition"],
                          "raw_min": row["norm"]["min"], "raw_p05": row["norm"]["p05"],
                          "raw_median": row["norm"]["median"], "raw_p95": row["norm"]["p95"],
                          "raw_max": row["norm"]["max"], "raw_median_over_cap": row["norm"]["median"] / 2.3,
                          "post_depth_median": row["depth"]["median"], "precap_over_fraction": row["cap_fraction"]})
    write_csv(tables / "whole_head_precap_scale.csv", aggregate)
    fig, axes = plt.subplots(1, 2, figsize=(15.4, 5.8), layout="constrained", gridspec_kw={"width_ratios": [1.65, 1]})
    fig.suptitle("相同 cap，原版图像头与 HyCoRe 的 raw 尺度不同", fontsize=18, fontweight="bold")
    y = np.arange(len(rows))[::-1]
    for pos, row in zip(y, rows):
        n, d = row["norm"], row["depth"]
        axes[0].plot([n["p05"], n["p95"]], [pos, pos], color=row["color"], lw=5, alpha=.5)
        axes[0].plot(n["median"], pos, "o", color=row["color"], ms=7)
        axes[0].text(n["p95"] * 1.055, pos, f"{n['median']:.2f}", va="center", color=row["color"], fontsize=10)
        axes[1].plot([d["p05"], d["p95"]], [pos, pos], color=row["color"], lw=5, alpha=.5)
        axes[1].plot(d["median"], pos, "o", color=row["color"], ms=7)
        axes[1].text(4.66, pos, f"{row['cap_fraction']:.1%}", va="center", fontsize=10, color=row["color"])
    axes[0].axvline(2.3, color="#b8333a", ls="--", label="切空间 cap2.3")
    axes[0].set_xscale("log")
    axes[0].set(xlabel="whole 映射前 raw 切空间模长（对数轴）", xlim=(1.7, 72), title="A  中位数与5–95%范围")
    axes[0].set_yticks(y, [r["name"] for r in rows])
    axes[0].legend(loc="lower right", fontsize=9)
    axes[1].axvline(4.6, color="#b8333a", ls="--")
    axes[1].set(xlabel="输出原点深度", xlim=(0, 5.25), title="B  输出深度 / 越 cap 比例")
    axes[1].set_yticks(y, [""] * len(rows))
    for ax in axes:
        ax.grid(axis="x", alpha=.22)
        ax.spines[["top", "right"]].set_visible(False)
    fig.supxlabel("原版每数据集仅12张已核验训练图像；LN→Linear→ToPoincare。新FULL9840曲线仅64对象、alias400 / eval BN。\n"
                  "不同数据、网络、轮次与条件：用于解释尺度和截断，不能当作同协议训练优劣对照。", fontsize=10)
    save(fig, figures, "06_whole_head_scale")
    return aggregate


def figure_seven(work, figures, tables):
    sources = [("whole_B64", "旧 B64 e300 · 无 HIER · 8856训练"),
               ("whole_HIER", "新 HIER e40 · FULL9840训练")]
    models = [(name, title, read(work / "gpu_conditions" / name / "summary.json")) for name, title in sources]
    for name, _, model in models:
        if len(model["conditions"]) < 12:
            raise ValueError("Incomplete factorial model " + name)
    matched_manifest = work / "gpu_conditions/matched_ids_validation.json"
    if matched_manifest.exists():
        validation = read(matched_manifest)
        checks = {row["label"]: row for row in validation["rows"]}
        if (not validation["matched"] or checks["whole_B64"]["count"] != 128
                or checks["whole_HIER"]["count"] != 128
                or checks["whole_B64"]["sample_ids_sha256"] != checks["whole_HIER"]["sample_ids_sha256"]):
            raise ValueError("Server-side matched panel verification failed")
    else:
        ids = []
        for name, _, _ in models:
            candidates = sorted((work / "gpu_conditions" / name).glob("private_*.npz"))
            if not candidates:
                raise ValueError("Require a matched-ID manifest or private NPZ")
            with np.load(candidates[0], allow_pickle=False) as cache:
                ids.append(cache["sample_ids"].copy())
        if not np.array_equal(ids[0], ids[1]):
            raise ValueError("Factorial models use different object panels")
    combinations = [("evalBN", "clean"), ("evalBN", "original_aug"),
                    ("BN_train_only", "clean"), ("BN_train_only", "original_aug")]
    constructions = ["direct_clean1024", "reorder_only1024", "part_alias400"]
    fig, axes = plt.subplots(2, 3, figsize=(16.3, 9.4), layout="constrained")
    fig.suptitle("相同128训练对象：BN、增强与 whole 构造的受控分解", fontsize=18, fontweight="bold")
    aggregate, effects = [], []
    summaries = {}
    for row_index, (name, title, model) in enumerate(models):
        lookup = {(x["BN_mode"], x["augmentation"], x["construction"]): x for x in model["conditions"]}
        for col, (role, quantity, metric_title) in enumerate([
                ("whole", "depth", "whole 平均深度"), ("part", "depth", "part 平均深度"),
                ("whole", "cap", "whole cap比例（%）")]):
            matrix = np.asarray([[lookup[bn, aug, con]["measurements"][role]["depth"]["mean"] if quantity == "depth"
                                  else 100 * lookup[bn, aug, con]["measurements"][role]["output_cap_depth_fraction"]
                                  for con in constructions] for bn, aug in combinations])
            ax = axes[row_index, col]
            image = ax.imshow(matrix, vmin=0, vmax=4.6 if quantity == "depth" else 100,
                              cmap="viridis" if quantity == "depth" else "YlOrRd", aspect="auto")
            for (r, c), value in np.ndenumerate(matrix):
                ax.text(c, r, f"{value:.3f}" if quantity == "depth" else f"{value:.1f}", ha="center", va="center",
                        color="white" if (quantity == "depth" and value < 2.8) or (quantity == "cap" and value > 58) else "#16252e", fontsize=10)
            ax.set_xticks(range(3), ["直接1024", "仅重排1024", "part400覆写"])
            ax.set_yticks(range(4), ["eval BN / clean", "eval BN / 增强", "train BN / clean", "train BN / 增强"])
            ax.set_title(title + "\n" + metric_title, fontsize=11)
            fig.colorbar(image, ax=ax, shrink=.76, pad=.02)
        for key, condition in lookup.items():
            if key[0] not in ("evalBN", "BN_train_only"):
                continue
            for role in ("whole", "part"):
                m = condition["measurements"][role]
                aggregate.append({"source": name, "label": model["label"], "epoch": model["source_epoch"],
                                  "eligible_train_objects": model["eligible_checkpoint_training_count"],
                                  "panel_objects": model["objects"], "BN_mode": key[0], "augmentation": key[1],
                                  "construction": key[2], "role": role, "mean_depth": m["depth"]["mean"],
                                  "median_depth": m["depth"]["median"], "std_depth": m["depth"]["std"],
                                  "p05_depth": m["depth"]["p05"], "p95_depth": m["depth"]["p95"],
                                  "cap_fraction": m["output_cap_depth_fraction"],
                                  "raw_tangent_median": m["raw_precap_tangent_norm"]["median"]})
        pairs = []
        for aug in ("clean", "original_aug"):
            for con in constructions:
                pairs.append(("BN_train_minus_eval", ("evalBN", aug, con), ("BN_train_only", aug, con)))
        for bn in ("evalBN", "BN_train_only"):
            for con in constructions:
                pairs.append(("aug_minus_clean", (bn, "clean", con), (bn, "original_aug", con)))
            for aug in ("clean", "original_aug"):
                pairs.append(("alias_minus_reorder", (bn, aug, "reorder_only1024"), (bn, aug, "part_alias400")))
                pairs.append(("reorder_minus_direct", (bn, aug, "direct_clean1024"), (bn, aug, "reorder_only1024")))
        for effect, before, after in pairs:
            a, b = lookup[before]["measurements"], lookup[after]["measurements"]
            effects.append({"source": name, "epoch": model["source_epoch"], "effect": effect,
                            "before_condition": "__".join(before), "after_condition": "__".join(after),
                            "whole_mean_depth_delta": b["whole"]["depth"]["mean"] - a["whole"]["depth"]["mean"],
                            "part_mean_depth_delta": b["part"]["depth"]["mean"] - a["part"]["depth"]["mean"],
                            "whole_cap_percentage_point_delta": 100 * (b["whole"]["output_cap_depth_fraction"] - a["whole"]["output_cap_depth_fraction"])})
        summaries[name] = {"direct_clean_eval": lookup["evalBN", "clean", "direct_clean1024"]["measurements"],
                           "direct_clean_BNtrain": lookup["BN_train_only", "clean", "direct_clean1024"]["measurements"]}
    write_csv(tables / "whole_condition_aggregates.csv", aggregate)
    write_csv(tables / "whole_condition_simple_effects.csv", effects)
    fig.supxlabel("每条件恢复源 BN / RNG，part 先于whole，local32；train BN仅启用BN，dropout保持eval。\n"
                  "两行使用相同128对象，但旧B64 e300与新HIER e40并非同轮或HIER开关因果对照；只在各行内部解释条件效应。", fontsize=10)
    save(fig, figures, "07_whole_condition_factorial")
    return {"same_128_object_panel": True, "models": summaries}


def figure_eight(work, figures, tables):
    path = work / "gpu_bn/summary.json"
    if not path.exists():
        return None
    data = read(path)
    if (not data["complete"] or data["calibration_holdout_intersection"] != 0
            or not data["source_model_BN_RNG_restored"] or not data["source_checkpoint_unchanged"]
            or data["optimizer_updates"] != 0 or data["backward_calls"] != 0):
        raise ValueError("BN-bank intervention state/identity checks failed")
    banks = {row["mode"]: row for row in data["banks"]}
    matched_path = work / "gpu_bn_matched/summary.json"
    matched = None
    if matched_path.exists():
        matched = read(matched_path)
        if (not matched["complete"] or matched["source_hashes"] != data["source_hashes"]
                or matched["source_bank_baseline"] != data["source_bank_baseline"]
                or matched["calibration_objects"] != data["calibration_objects"]
                or matched["holdout_training_objects"] != data["holdout_training_objects"]
                or matched["training_shard_sha256"] != data["training_shard_sha256"]
                or matched["calibration_holdout_intersection"] != 0
                or not matched["source_model_BN_RNG_restored"] or not matched["source_checkpoint_unchanged"]):
            raise ValueError("Matched BN intervention source/baseline/identity verification failed")
        matched_banks = {row["mode"]: row for row in matched["banks"]}
        control = matched_banks["whole_twice_alias_clean"]
        joint_repeat = matched_banks["joint_alias_clean"]
        if (control["BN_bank"]["local32_forward_calls"] != 40
                or joint_repeat["BN_bank"]["local32_forward_calls"] != 40
                or joint_repeat["measurement"] != banks["joint_alias_clean"]["measurement"]):
            raise ValueError("Matched40-forward control or joint replication failed")
        banks["whole_twice_alias_clean"] = control
    definitions = [("source_bank", "源训练 BN 库", "#7d8a94"),
                   ("whole_only_clean", "新库：仅 whole clean", GREEN),
                   ("joint_alias_clean", "新库：part→whole clean", BLUE),
                   ("joint_alias_original_aug", "新库：part→whole 增强", PURPLE)]
    if matched is not None:
        definitions.append(("whole_twice_alias_clean", "匹配40次：whole→whole alias", ORANGE))
    rows = []
    for mode, name, color in definitions:
        bank = banks.get(mode)
        measurement = bank["measurement"] if bank else data["source_bank_baseline"]
        if bank and not bank["parameter_fingerprint_unchanged"]:
            raise ValueError("A model parameter changed in the BN calibration")
        raw, d = measurement["raw_precap_tangent_norm"], measurement["depth"]
        rows.append({"mode": mode, "name": name, "color": color, "source_epoch": data["source_epoch"],
                     "eligible_source_training_objects": data["eligible_source_training_objects"],
                     "holdout_training_objects": data["holdout_training_objects"],
                     "calibration_objects": data["calibration_objects"] if bank else 0,
                     "calibration_local32_forward_calls": bank["BN_bank"]["local32_forward_calls"] if bank else None,
                     "raw_p05": raw["p05"], "raw_median": raw["median"], "raw_p95": raw["p95"],
                     "depth_p05": d["p05"], "depth_mean": d["mean"], "depth_median": d["median"],
                     "depth_p95": d["p95"], "depth_std": d["std"],
                     "whole_cap_fraction": measurement["cap_depth_fraction"],
                     "calibration_holdout_intersection": 0,
                     "intervention_implementation": "1b934ed" if mode == "whole_twice_alias_clean" else "initial_BN_probe",
                     "model_parameters_unchanged": True, "reset_cumulative_BN": bool(bank)})
    write_csv(tables / "whole_BN_bank_intervention.csv", [{k: v for k, v in row.items() if k not in ("color", "name")} for row in rows])
    fig, axes = plt.subplots(1, 3, figsize=(17, 6.8 if matched is not None else 5.8), layout="constrained",
                             gridspec_kw={"width_ratios": [1.2, 1.2, 1]})
    fig.suptitle("冻结参数，只改变 BN 统计库：whole 的内部 / 球壳分布显著改变", fontsize=17, fontweight="bold")
    positions = np.arange(len(rows))[::-1]
    for y, row in zip(positions, rows):
        color = row["color"]
        axes[0].plot([row["raw_p05"], row["raw_p95"]], [y, y], color=color, lw=5, alpha=.42)
        axes[0].plot(row["raw_median"], y, "o", color=color, ms=8)
        axes[0].annotate(f"{row['raw_median']:.3f}", (row["raw_median"], y),
                         xytext=(0, 12), textcoords="offset points", ha="center", color=color, fontsize=10)
        axes[1].plot([row["depth_p05"], row["depth_p95"]], [y, y], color=color, lw=5, alpha=.42)
        axes[1].plot(row["depth_mean"], y, "o", color=color, ms=8)
        axes[1].annotate(f"{row['depth_mean']:.3f}", (row["depth_mean"], y),
                         xytext=(0, 12), textcoords="offset points", ha="center", color=color, fontsize=10)
        axes[2].barh(y, row["whole_cap_fraction"] * 100, height=.42, color=color, alpha=.85)
        count = int(round(row["whole_cap_fraction"] * data["holdout_training_objects"]))
        axes[2].text(row["whole_cap_fraction"] * 100 + 1.7, y,
                     f"{100*row['whole_cap_fraction']:.2f}%\n{count}/{data['holdout_training_objects']}",
                     va="center", fontsize=10, color=color)
    axes[0].axvline(2.3, color="#b8333a", ls="--", lw=1.3, label="切空间 cap2.3")
    axes[0].set(title="A  raw 切空间模长", xlabel="点：中位数；粗线：5–95%", xlim=(1.35, 3.35))
    axes[0].legend(loc="lower right", fontsize=9)
    axes[1].axvline(4.6, color="#b8333a", ls="--", lw=1.3, label="深度 cap4.6")
    axes[1].set(title="B  同128对象的 eval whole 深度", xlabel="点：均值；粗线：5–95%", xlim=(2.65, 4.85))
    axes[1].legend(loc="lower left", fontsize=9)
    axes[2].set(title="C  whole 输出 cap 比例", xlabel="比例（%）", xlim=(0, 124))
    axes[2].set_xticks([0, 25, 50, 75, 100])
    for i, ax in enumerate(axes):
        ax.set_yticks(positions, [row["name"] for row in rows] if i == 0 else [""] * len(rows))
        ax.set_ylim(-.55, len(rows) - .35)
        ax.grid(axis="x", alpha=.2)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    control_caption = ("whole-only 为20次；匹配 whole→whole 与 joint clean均40次，使用相同alias whole构造，只替换第一次前向输入。\n"
                       if matched is not None else "whole-only 为20次、joint为40次 local32前向。\n")
    fig.supxlabel("旧 B64 e300，8856训练协议；640个校准对象与128个测量对象互不重叠。所有新库均 reset + cumulative。\n"
                  + control_caption +
                  "测量均为whole-only clean / eval，part效果未测。"
                  "支持混合输入统计参与分布差异；不能宣称历史唯一原因、分类改善或已完成训练修复。", fontsize=10)
    save(fig, figures, "08_whole_bn_bank")
    return {"baseline": rows[0], "banks": rows[1:], "frozen_parameter_BN_bank_intervention": True,
            "matched40_control": matched is not None,
            "matched_baseline_source_hashes_and_joint_replication_verified": matched is not None}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--work", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    matplotlib.rcParams.update({"font.family": ["Microsoft YaHei", "DejaVu Sans"], "font.size": 10.5,
                                "axes.unicode_minus": False, "svg.fonttype": "none", "svg.hashsalt": "hycore-whole-cap-v1"})
    figures, tables = args.output / "figures", args.output / "tables"
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    result = {"timeline": figure_five(args.work, figures, tables),
              "head_scale": figure_six(args.work, figures, tables),
              "factorial": figure_seven(args.work, figures, tables)}
    bank = figure_eight(args.work, figures, tables)
    if bank is not None:
        result["BN_calibration"] = bank
    (args.output / "whole_evidence_summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"complete": True, "figure_groups": 3 + int(bank is not None),
                      "CSV_tables": 4 + int(bank is not None), "private_IDs_exported": False}))


if __name__ == "__main__":
    main()
