"""Render anonymous frozen-BN evidence, without model or GPU execution.

python bn_reference_figures.py --summary PRIVATE_RUN/summary.json --out-dir REPORT58

Only approved aggregate measurements and cryptographic identities are exported.
Training-panel accuracy is deliberately excluded; changed-prediction rate is a
distribution-drift diagnostic, never a held-out classification result.
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
from matplotlib import font_manager
from matplotlib.patches import Patch
import numpy as np


MODELS = ("V8_B0_32_e300", "V8_B64_e300", "V8_HIER64_e300")
MODEL_LABELS = ("V8-B0-32\nshuffle / 无HIER", "V8-B64\n类别采样 / 无HIER", "V8-HIER64\n类别采样 / 有HIER")
CONDITIONS = ("native_saved_bank", "reference_original_last_running_buffers",
              "self_whole_only_bank", "BN_batch_statistics")
CONDITION_LABELS = ("自身保存的BN运行库", "移植原HyCoRe末轮运行库",
                    "自身仅whole校准库", "当前批BN（仅诊断）")
COLORS = ("#68778B", "#BD625D", "#328D7D", "#376FA2")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
                          encoding="utf-8")


def set_style():
    available = {font.name for font in font_manager.fontManager.ttflist}
    font = next((name for name in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "SimSun")
                 if name in available), None)
    if font is None:
        raise RuntimeError("A Chinese font is required for readable evidence figures")
    plt.rcParams.update({"font.family": font, "font.size": 10.5, "axes.titlesize": 12,
                         "axes.labelsize": 10.5, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.unicode_minus": False,
                         "figure.facecolor": "#FFFFFF", "axes.facecolor": "#FFFFFF",
                         "savefig.facecolor": "#FFFFFF", "svg.fonttype": "none"})


def grouped_bars(ax, rows, getter, title, ylabel, ylim, fmt=".2f", reference=None):
    centers = np.arange(3)
    width = .185
    for condition_index, condition in enumerate(CONDITIONS):
        positions = centers + (condition_index - 1.5) * width
        values = np.asarray([getter(rows[(model, condition)]) for model in MODELS])
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite graph value")
        bars = ax.bar(positions, values, width=width * .91, color=COLORS[condition_index],
                      edgecolor="white", linewidth=.6, zorder=3)
        for bar, value in zip(bars, values):
            offset = 4 if value >= 0 else -11
            ax.annotate(format(value, fmt), (bar.get_x() + bar.get_width() / 2, value),
                        xytext=(0, offset), textcoords="offset points", ha="center",
                        va="bottom", fontsize=8.2, color="#263544")
    ax.set_xticks(centers, MODEL_LABELS)
    ax.set_xlim(-.58, 2.58)
    ax.set_ylim(*ylim)
    ax.set_title(title, loc="left", fontweight="bold", pad=12)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", color="#E1E7ED", linewidth=.7, zorder=0)
    if reference is not None:
        ax.axhline(reference, color="#9CA9B7", linestyle="--", linewidth=1.0, zorder=2)
    ax.tick_params(axis="x", length=0, pad=8)


def footer_and_save(fig, out, stem, title, subtitle, footer, legend_y=.10):
    fig.suptitle(title, x=.055, y=.975, ha="left", fontsize=19, fontweight="bold", color="#17354D")
    fig.text(.055, .924, subtitle, ha="left", va="top", fontsize=10.6, color="#4C6172")
    handles = [Patch(facecolor=color, label=label) for color, label in zip(COLORS, CONDITION_LABELS)]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, legend_y),
               frameon=False, ncol=2, columnspacing=3.2, handlelength=1.5, fontsize=10.3)
    fig.text(.055, .025, footer, fontsize=9.4, color="#516575", va="bottom", linespacing=1.6)
    fig.savefig(out / (stem + ".png"), dpi=180)
    fig.savefig(out / (stem + ".svg"))
    plt.close(fig)


def public_row(row):
    """An explicit whitelist prevents private metadata/IDs/accuracy from leaking."""
    result = {"model": row["model"], "condition": row["condition"],
              "measurement_objects": row["measurement_objects"],
              "calibration_objects": row["calibration_objects"],
              "calibration_local32_forward_calls": row["calibration_local32_forward_calls"]}
    for role in ("whole", "part"):
        for key in ("mean", "median", "std", "variance", "min", "p01", "p05", "p25", "p75", "p95", "p99", "max"):
            result[role + "_depth_" + key] = float(row[role]["depth"][key])
        result[role + "_cap_percent"] = float(100 * row[role]["cap_fraction"])
        result[role + "_precap_over_2_3_percent"] = float(100 * row[role]["precap_over_2_3_fraction"])
        result[role + "_raw_norm_mean"] = float(row[role]["raw_precap_tangent_norm"]["mean"])
        result[role + "_raw_norm_median"] = float(row[role]["raw_precap_tangent_norm"]["median"])
        result[role + "_direction_cosine_to_native_mean"] = float(row[role]["direction_cosine_to_native"]["mean"])
        result[role + "_direction_cosine_to_native_median"] = float(row[role]["direction_cosine_to_native"]["median"])
    for key in ("mean", "median", "std", "p05", "p95"):
        result["whole_minus_part_depth_" + key] = float(row["whole_minus_part_depth"][key])
    result["whole_deeper_than_part_percent"] = float(100 * row["whole_deeper_than_part_fraction"])
    result["prediction_changed_from_native_percent"] = float(100 * row["prediction_changed_from_native_fraction"])
    result["BN_affine_and_other_parameters_unchanged"] = bool(row["BN_affine_and_other_parameters_unchanged"])
    result["optimizer_updates"] = int(row["optimizer_updates"])
    result["backward_calls"] = int(row["backward_calls"])
    result["test_forward_calls"] = int(row["test_forward_calls"])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    summary_path = Path(args.summary)
    source = json.loads(summary_path.read_text(encoding="utf-8-sig"))
    if not source.get("complete") or not source.get("source_files_unchanged") or not source.get("all_state_BN_RNG_restored"):
        raise ValueError("The frozen probe did not pass completion/source/restoration checks")
    if any(source[key] != 0 for key in ("optimizer_updates", "backward_calls", "test_forward_calls")):
        raise ValueError("This figure requires a frozen, train-input-only probe")
    if source["measurement_objects"] != 320 or source["calibration_objects"] != 640:
        raise ValueError("Unexpected measurement/calibration panel size")
    rows = {(row["model"], row["condition"]): row for row in source["conditions"]}
    expected = {(model, condition) for model in MODELS for condition in CONDITIONS}
    if len(source["conditions"]) != 12 or set(rows) != expected:
        raise ValueError("Require exactly three models x four BN conditions")
    for (model, condition), row in rows.items():
        if not row["BN_affine_and_other_parameters_unchanged"]:
            raise ValueError("Learned weights changed in a BN condition")
        if any(row[key] != 0 for key in ("optimizer_updates", "backward_calls", "test_forward_calls")):
            raise ValueError("A condition includes forbidden updates/test inputs")
        if row["measurement_objects"] != 320 or row["whole"]["depth"]["count"] != 320 or row["part"]["depth"]["count"] != 320:
            raise ValueError("A condition used a different measurement panel size")
        expected_calibration = 640 if condition == "self_whole_only_bank" else 0
        if row["calibration_objects"] != expected_calibration:
            raise ValueError("Unexpected calibration size")
        if condition == "native_saved_bank":
            if row["prediction_changed_from_native_fraction"] != 0 or row["whole"]["direction_cosine_to_native"]["mean"] < .99999:
                raise ValueError("Native reference does not match itself")
    for model in source["models"]:
        if not model["state_restored"] or model["source_epoch"] != 300 or model["BN_layers"] != 38:
            raise ValueError("Unexpected source model or incomplete restoration")
    out = Path(args.out_dir)
    figures, tables = out / "figures", out / "tables"
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    set_style()
    fig, axes = plt.subplots(1, 2, figsize=(14.6, 7.6))
    fig.subplots_adjust(left=.07, right=.975, bottom=.27, top=.83, wspace=.27)
    grouped_bars(axes[0], rows, lambda row: row["whole"]["depth"]["mean"],
                 "A  whole到原点的平均双曲距离", "平均深度 $d_0$", (0, 5.12), reference=4.6)
    grouped_bars(axes[1], rows, lambda row: 100 * row["whole"]["cap_fraction"],
                 "B  whole贴cap的比例", "贴cap样本（%）", (0, 112), fmt=".1f")
    footer_and_save(fig, figures, "07_bn_whole_depth_comparison",
                    "只更换BN统计，whole的半径分布就会显著变化",
                    "旧同协议V8 e300 · 相同320训练对象（40类×8）测量 · 640个独立训练对象（40类×16）仅whole校准 · 参数全冻结",
                    "虚线：V8 cap对应深度4.6；贴cap定义 $d_0$ ≥ 4.5999。原HyCoRe移植项只换运行均值/方差/计数，保留V8自身γ、β。\n"
                    "当前批BN为诊断条件；更贴壳不等于分类或层次结构更好。测量对象与校准对象无交集，无test前向。")
    fig, axes = plt.subplots(2, 2, figsize=(14.6, 10.6))
    fig.subplots_adjust(left=.07, right=.975, bottom=.205, top=.84, wspace=.26, hspace=.62)
    grouped_bars(axes[0, 0], rows, lambda row: 100 * row["part"]["cap_fraction"],
                 "A  part也会受whole-only统计库影响", "part贴cap（%）", (0, 61), fmt=".1f")
    grouped_bars(axes[0, 1], rows, lambda row: row["whole_minus_part_depth"]["mean"],
                 "B  whole与part的径向深度差", "平均 $d_0$(whole) − $d_0$(part)", (-.17, 2.35), reference=0)
    grouped_bars(axes[1, 0], rows, lambda row: row["whole"]["direction_cosine_to_native"]["mean"],
                 "C  与自身保存BN时的whole方向一致性", "方向余弦均值", (-.13, 1.16), fmt=".3f", reference=0)
    grouped_bars(axes[1, 1], rows, lambda row: 100 * row["prediction_changed_from_native_fraction"],
                 "D  与自身保存BN时的预测改变比例", "预测改变的对象（%）", (0, 112), fmt=".2f")
    footer_and_save(fig, figures, "08_bn_whole_part_and_direction",
                    "恢复whole球壳，也可能压缩whole–part的径向间隔",
                    "同320测量对象、同400点固定part · HIER模型仅whole校准后：part贴cap达到50.94%，平均深度差由0.508降至0.331",
                    "方向余弦：同一对象在干预后与native输出的单位方向点积；1表示方向相同。预测改变比例不是错误率。\n"
                    "图中只有训练对象诊断，无test/验证成绩，不按这些结果挑选完整训练配置。BN库与各自网络特征共同适配，不能作为通用MN40常量。",
                    legend_y=.085)
    public_rows = [public_row(rows[(model, condition)]) for model in MODELS for condition in CONDITIONS]
    with (tables / "bn_reference_comparison.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(public_rows[0]))
        writer.writeheader()
        writer.writerows(public_rows)
    checks = {"completed_conditions": 12, "source_files_unchanged": True,
              "all_state_BN_RNG_restored": True, "BN_affine_and_other_parameters_unchanged": True,
              "optimizer_updates": 0, "backward_calls": 0, "test_forward_calls": 0,
              "measurement_objects": 320, "calibration_objects": 640,
              "private_source_summary_sha256": sha(summary_path),
              "probe_runner_sha256": source["runner_sha256"],
              "figures_runner_sha256": sha(__file__),
              "model_state_fingerprints": [{key: row[key] for key in
                                            ("model", "source_epoch", "BN_layers", "state_restored",
                                             "learned_parameter_fingerprint", "native_state_fingerprint")}
                                           for row in source["models"]]}
    manifest_path = summary_path.with_name("manifest.json")
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
        if not manifest["disjoint_panels"] or manifest["measurement_per_class"] != 8 or manifest["calibration_per_class"] != 16:
            raise ValueError("Manifest does not verify disjoint balanced panels")
        checks.update({key: manifest[key] for key in
                       ("disjoint_panels", "classes", "measurement_per_class", "calibration_per_class",
                        "source_train_ids_sha256", "measurement_ids_sha256", "calibration_ids_sha256",
                        "measurement_whole_sha256", "measurement_part_sha256", "measurement_part_indices_sha256")})
        # Hash keys are deliberately whitelisted; never copy arbitrary spec keys.
        checks["input_file_sha256"] = {key: value for key, value in manifest["input_file_sha256"].items()
                                       if key in ("spec", "reference_bn", "source_train_ids") or key.startswith("checkpoint_V8_")}
        checks["training_shard_sha256"] = {key: value for key, value in manifest["training_shard_sha256"].items()
                                          if key in {"ply_data_train%d.h5" % i for i in range(5)}}
    public_summary = {"complete": True, "checks": checks, "conditions": public_rows,
                      "sampling": "Old8856/984 protocol;40classes x8 measurement and40classes x16 independent calibration training objects; same IDs and order across models",
                      "input": source["input_condition"], "reference_transfer": source["reference_transfer"],
                      "self_calibration": source["calibration"],
                      "statistics_scope": "Frozen320-training-object diagnostic; no test/validation scoring, parameter updates, model selection, or publication of full IDs/private paths",
                      "train_panel_accuracy_excluded": True,
                      "notes": ["Original HyCoRe running buffers are network-dependent; cross-network transfer is an intervention, not a universal normalization bank.",
                                "Self whole-only calibration can recover the whole shell while changing part depth and narrowing their radial gap.",
                                "Prediction-change fraction measures agreement with each model's native prediction, not accuracy.",
                                "Current-batch BN uses role-specific batch statistics and is not a standard inference result."]}
    write_json(out / "bn_probe_summary.json", public_summary)
    print(json.dumps({"complete": True, "figures": 4, "aggregate_rows": len(public_rows),
                      "whole_HIER_self_calibration_cap_percent": public_rows[10]["whole_cap_percent"],
                      "part_HIER_self_calibration_cap_percent": public_rows[10]["part_cap_percent"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
