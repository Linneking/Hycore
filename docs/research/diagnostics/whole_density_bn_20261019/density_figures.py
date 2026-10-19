"""Compare frozen whole embeddings with matched canonical ModelNet40 objects.

The private CLI config supplies all cache paths. Public outputs contain only
anonymous aggregates, hashes and condition checks, never object IDs or paths.
No model forward, test evaluation or optimizer update is performed here.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


COLORS = ("#53657D", "#D18E25", "#388C83", "#2568A3")
CAP_DEPTH = 4.6
CAP_TOLERANCE = 1e-4


def file_hash(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            value.update(chunk)
    return value.hexdigest()


def array_hash(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def legacy_canonical_ids(ids: np.ndarray, shards: list[dict]) -> np.ndarray:
    counts = {row["name"]: int(row["count"]) for row in shards}
    if len(counts) != len(shards):
        raise ValueError("Legacy source repeats a shard name")
    offsets = {}
    cursor = 0
    for name in sorted(counts):
        offsets[name] = cursor
        cursor += counts[name]
    mapping = np.concatenate(
        [offsets[row["name"]] + np.arange(int(row["count"])) for row in shards]
    )
    if ids.min() < 0 or ids.max() >= len(mapping):
        raise ValueError("Legacy object ID is outside the documented shard order")
    return mapping[ids]


@dataclass
class Observation:
    key: str
    label: str
    stage: str
    epoch_label: str
    curvature: float
    depth: np.ndarray
    has_cap: bool
    stats: dict
    checks: dict


def load_observation(model: dict, stage: str, ids: np.ndarray,
                     reference_labels: np.ndarray, reference_input_hash: str,
                     legacy_input_proof: dict | None = None) -> Observation:
    entry = model["stages"][stage]
    cache_path = Path(entry["cache"])
    source_hash_before = file_hash(cache_path)
    legacy = model["source_kind"] == "legacy"
    summary = json.loads(Path(entry["summary"]).read_text(encoding="utf-8")) if legacy else {}
    with np.load(cache_path, allow_pickle=False) as cache:
        source_ids = np.asarray(cache["ids" if legacy else "sample_ids"], dtype=np.int64)
        labels = np.asarray(cache["labels"], dtype=np.int64)
        vectors = np.asarray(cache["whole_mu" if legacy else "mu"], dtype=np.float64)
        if legacy:
            source_ids = legacy_canonical_ids(
                source_ids, summary["data_identity"]["source_shards_in_glob_order"]
            )
            curvature = float(model["curvature"])
            if curvature != 1.0 or not summary["patch_identity"]["clean_whole_input_unchanged"]:
                raise ValueError("Expected a clean original HyCoRe c=1 cache")
            condition = "clean first1024 / eval BN; verified original export implementation"
            input_hash = None
            if legacy_input_proof is not None:
                if not (legacy_input_proof["matches"]
                        and legacy_input_proof["legacy_order_input_sha256"]
                        == summary["data_identity"]["points1024_sha256"]
                        == legacy_input_proof["recomputed_legacy_order_input_sha256"]
                        and legacy_input_proof["matched_common_input_sha256"] == reference_input_hash):
                    raise ValueError("Legacy whole input byte-alignment proof does not match the cache")
                input_hash = legacy_input_proof["matched_common_input_sha256"]
            checkpoint_epoch = summary["checkpoint_fields_unmodified"]["epoch"]
            checkpoint_hash = summary["checkpoint_sha256"]
            epoch_label = f"原始 epoch={checkpoint_epoch}（0起）"
        else:
            curvature = float(cache["c"])
            condition = str(cache["input_mode"])
            input_hash = str(cache["input_sha256"])
            if "clean first1024/eval BN" not in condition:
                raise ValueError("V8 cache is not clean first1024 / eval BN")
            if input_hash != reference_input_hash:
                raise ValueError("V8 cache input bytes do not match the reference")
            if not np.isclose(curvature, 0.1, rtol=1e-5):
                raise ValueError("V8 cache curvature differs from 0.1")
            checkpoint_epoch = entry["epoch"]
            checkpoint_hash = entry.get("checkpoint_sha256")
            epoch_label = f"e{checkpoint_epoch}"
        if vectors.shape != (len(labels), 256) or len(source_ids) != len(labels):
            raise ValueError("Unexpected whole embedding / identity shape")
        if len(np.unique(source_ids)) != len(source_ids):
            raise ValueError("Duplicate canonical object IDs")
        lookup = {int(value): index for index, value in enumerate(source_ids)}
        rows = np.array([lookup[int(value)] for value in ids], dtype=np.int64)
        np.testing.assert_array_equal(labels[rows], reference_labels)
        vectors = vectors[rows]
        radius = np.linalg.norm(vectors, axis=1)
        argument = np.sqrt(curvature) * radius
        if not np.isfinite(vectors).all() or np.any(argument < 0) or np.any(argument >= 1):
            raise ValueError("Embedding is not finite and inside its curvature-specific ball")
        depth = 2.0 / np.sqrt(curvature) * np.arctanh(argument)
        if legacy:
            np.testing.assert_allclose(depth, cache["whole_d0"][rows], rtol=1e-6, atol=1e-6)
    has_cap = not legacy
    if file_hash(cache_path) != source_hash_before:
        raise ValueError("A source cache changed during its read-only audit")
    if has_cap and depth.max() > CAP_DEPTH + CAP_TOLERANCE:
        raise ValueError("V8 whole depth exceeds the documented cap tolerance")
    quantiles = np.quantile(depth, [0, .01, .05, .25, .5, .75, .95, .99, 1])
    stats = dict(
        model=model["key"], label=model["label"], stage=stage,
        checkpoint_epoch=checkpoint_epoch, checkpoint_epoch_label=epoch_label,
        checkpoint_sha256=checkpoint_hash, curvature=curvature, count=len(depth),
        mean=float(depth.mean()), median=float(np.median(depth)), std=float(depth.std(ddof=0)),
        variance=float(depth.var(ddof=0)),
        **{name: float(value) for name, value in zip(
            ["min", "p01", "p05", "p25", "p50", "p75", "p95", "p99", "max"], quantiles
        )},
        v8_cap_applies=has_cap,
        cap_fraction=float(np.mean(depth >= CAP_DEPTH - CAP_TOLERANCE)) if has_cap else None,
        cap_tolerance_depth=CAP_TOLERANCE if has_cap else None,
    )
    checks = dict(
        model=model["key"], stage=stage, source_cache_sha256=source_hash_before,
        source_object_count=len(source_ids), common_object_count=len(ids),
        canonical_pool_ids_sha256=array_hash(ids), labels_sha256=array_hash(reference_labels),
        canonical_labels_match=True, inference_condition=condition,
        v8_input_sha256=input_hash,
        v8_input_bytes_match_reference=True if not legacy else None,
        whole_input_bytes_match_reference=True if not legacy or legacy_input_proof is not None else None,
        legacy_input_byte_proof=legacy_input_proof if legacy else None,
        legacy_input_verification=(
            "full original glob-order input bytes match the historical export digest; "
            "canonical HDF5 name+row alignment and common 8856 input bytes match V8 exactly"
            if legacy_input_proof is not None else
            "source HDF5 name+row alignment and labels; clean first1024/eval export verified; "
            "legacy full-pool input digest uses a different ordering and cannot be directly compared"
        ) if legacy else None,
        depth_recomputed_from_ball_vectors=True,
    )
    return Observation(model["key"], model["label"], stage, epoch_label,
                       curvature, depth, has_cap, stats, checks)


def setup_style() -> None:
    plt.rcParams.update({
        "font.family": "Microsoft YaHei", "axes.unicode_minus": False,
        "svg.fonttype": "none", "font.size": 11, "axes.spines.top": False,
        "axes.spines.right": False, "axes.edgecolor": "#AEB8C5",
        "text.color": "#18334C", "axes.labelcolor": "#18334C",
        "xtick.color": "#53657D", "ytick.color": "#53657D",
        "figure.facecolor": "white", "axes.facecolor": "white",
    })


def draw_density(ax, observation: Observation, color: str, bins: np.ndarray,
                 x_max: float, y_max: float) -> None:
    values, _ = np.histogram(observation.depth, bins=bins, density=True)
    if not np.isclose(np.sum(values * np.diff(bins)), 1.0):
        raise ValueError("Histogram normalization failed")
    ax.stairs(values, bins, color=color, fill=True, alpha=.30, linewidth=1.3)
    ax.stairs(values, bins, color=color, linewidth=1.5)
    ax.axvline(observation.stats["mean"], color=color, linestyle="--", lw=1.2, label="均值")
    ax.axvline(observation.stats["median"], color=color, linestyle="-", lw=1.2, label="中位数")
    if observation.has_cap:
        ax.axvline(CAP_DEPTH, color="#B64A48", linestyle=":", lw=1.5, label="V8 cap = 4.6")
    stats = observation.stats
    cap_text = f"贴 cap：{100*stats['cap_fraction']:.2f}%" if observation.has_cap else "V8 cap：不适用"
    text = (f"c = {observation.curvature:.1f}    n = {len(observation.depth):,}\n"
            f"均值 {stats['mean']:.3f}    中位数 {stats['median']:.3f}\n"
            f"标准差 {stats['std']:.3f}    {cap_text}")
    ax.text(.025, .96, text, transform=ax.transAxes, va="top", fontsize=10.5,
            bbox={"facecolor": "white", "alpha": .90, "edgecolor": "none", "pad": 4})
    ax.set(xlim=(0, x_max), ylim=(0, y_max),
           xlabel="到原点的双曲距离  $d_0$", ylabel="样本概率密度",
           title=f"{observation.label} · {observation.epoch_label}")
    ax.grid(axis="y", alpha=.18)
    ax.legend(loc="upper right", fontsize=9, frameon=False)


def save_figure(fig, folder: Path, name: str) -> None:
    for suffix in ["png", "svg"]:
        fig.savefig(folder / f"{name}.{suffix}", dpi=180, facecolor="white")
    plt.close(fig)


def figures(observations: list[Observation], folder: Path) -> list[str]:
    setup_style()
    x_max = max(5.0, np.ceil(max(row.depth.max() for row in observations) * 2) / 2)
    # The cap lies inside a bin, rather than on a numerically ambiguous bin edge.
    bins = np.arange(-.05, x_max + .15001, .10)
    result = []
    for stage in ["last", "best"]:
        group = [row for row in observations if row.stage == stage]
        if not group:
            continue
        if len(group) != 4:
            raise ValueError("Final density figures require exactly four models per stage")
        y_max = max(np.histogram(row.depth, bins=bins, density=True)[0].max()
                    for row in group) * 1.25
        fig, axes = plt.subplots(2, 2, figsize=(13.6, 9.1))
        for ax, row, color in zip(axes.flat, group, COLORS):
            draw_density(ax, row, color, bins, x_max, y_max)
        heading = "训练终点" if stage == "last" else "各自已选定 best 权重"
        fig.suptitle(f"{heading}：四模型 whole 径向分布", fontsize=19, fontweight="bold", y=.975)
        fig.text(.5, .022,
                 "同 8,856 个训练对象 · clean first1024 · eval BN · 各模型保留自身保存的 BN\n"
                 "统一物理距离坐标与纵轴；直方图箱宽 0.10、面积为 1；无 KDE 平滑。原版 c=1，V8 c≈0.1。",
                 ha="center", fontsize=10, color="#53657D")
        fig.subplots_adjust(left=.075, right=.98, top=.90, bottom=.13, hspace=.38, wspace=.20)
        name = f"01_whole_depth_density_{stage}" if stage == "last" else "02_whole_depth_density_best"
        save_figure(fig, folder, name)
        result.append(name)
        fig, ax = plt.subplots(figsize=(11.3, 6.3))
        for row, color in zip(group, COLORS):
            depth = np.sort(row.depth)
            ax.step(np.r_[0.0, depth], np.r_[0.0, np.arange(1, len(depth)+1)/len(depth)],
                    where="post", lw=1.8, color=color,
                    label=f"{row.label} · {row.epoch_label}")
        ax.axvline(CAP_DEPTH, color="#B64A48", ls=":", lw=1.5,
                   label="4.6（仅 V8 的深度上限）")
        ax.set(xlim=(0, x_max), ylim=(0, 1.025), xlabel="到原点的双曲距离  $d_0$",
               ylabel="深度不超过横轴值的样本比例", title=f"{heading}：whole 深度累积分布")
        ax.grid(alpha=.18)
        ax.legend(loc="upper left", fontsize=10, frameon=False)
        fig.text(.5, .025, "同 8,856 对象、相同 clean/eval 条件；曲线直接由全部样本计算，无平滑。",
                 ha="center", fontsize=10, color="#53657D")
        fig.subplots_adjust(left=.09, right=.98, top=.90, bottom=.14)
        name = f"03_whole_depth_ecdf_{stage}" if stage == "last" else "04_whole_depth_ecdf_best"
        save_figure(fig, folder, name)
        result.append(name)
        if stage == "last":
            for index, (row, color) in enumerate(zip(group, COLORS), start=1):
                fig, ax = plt.subplots(figsize=(9.3, 5.8))
                draw_density(ax, row, color, bins, x_max, y_max)
                fig.text(.5, .025, "同 8,856 训练对象 · clean first1024 / eval BN · 箱宽 0.10",
                         ha="center", fontsize=10, color="#53657D")
                fig.subplots_adjust(left=.10, right=.97, top=.88, bottom=.15)
                name = f"05_{index}_{row.key}_whole_depth_last"
                save_figure(fig, folder, name)
                result.append(name)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8-sig"))
    proof_path = config.get("legacy_input_byte_proof")
    legacy_input_proof = json.loads(Path(proof_path).read_text(encoding="utf-8-sig")) if proof_path else None
    with np.load(config["reference_cache"], allow_pickle=False) as cache:
        ids = np.asarray(cache["sample_ids"], dtype=np.int64)
        labels = np.asarray(cache["labels"], dtype=np.int64)
        input_hash = str(cache["input_sha256"])
    if len(ids) != 8856 or len(np.unique(ids)) != 8856 or not np.all(np.diff(ids) > 0):
        raise ValueError("Expected 8,856 sorted unique canonical training object IDs")
    observations = [load_observation(model, stage, ids, labels, input_hash, legacy_input_proof)
                    for stage in ["last", "best"] for model in config["models"]]
    summary = {
        "scope": "frozen cached training objects; no new model forward or test evaluation",
        "common_object_count": len(ids), "models": [row.stats for row in observations],
        "histogram": {"normalization": "probability density, area=1 per model",
                      "bin_width": .1, "KDE": False,
                      "common_axes": "same x for all; same y within each four-model stage",
                      "cap_definition": "d0 >= 4.5999 for V8 only; matches report57"},
    }
    validation = {
        "matched": True, "object_count": len(ids),
        "canonical_pool_ids_sha256": array_hash(ids), "labels_sha256": array_hash(labels),
        "reference_clean_input_sha256": input_hash,
        "source_inputs_unmodified": True,
        "checks": [row.checks for row in observations],
    }
    if args.validate_only:
        print(json.dumps({"summary": summary, "validation": validation}, ensure_ascii=False, indent=2))
        return
    if args.output is None:
        parser.error("--output is required unless --validate-only is supplied")
    figure_folder = args.output / "figures"
    table_folder = args.output / "tables"
    figure_folder.mkdir(parents=True, exist_ok=True)
    table_folder.mkdir(parents=True, exist_ok=True)
    summary["figures"] = figures(observations, figure_folder)
    for name, value in [("whole_density_summary.json", summary), ("whole_density_validation.json", validation)]:
        (table_folder / name).write_text(json.dumps(value, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    with (table_folder / "whole_depth_distribution.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(observations[0].stats))
        writer.writeheader()
        writer.writerows(row.stats for row in observations)
    print(json.dumps({"matched": True, "objects": len(ids),
                      "model_stage_count": len(observations), "figure_groups": len(summary["figures"])},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
