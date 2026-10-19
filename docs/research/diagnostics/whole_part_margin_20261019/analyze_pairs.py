"""Frozen whole/part margin audit; all public outputs are anonymous aggregates.

Usage: --config PRIVATE.json --out-dir PUBLIC_DIR [--private-out-dir NEW_DIR].
Config models contain label, stage, cache, kind, c, gamma, delta, epoch and an
optional condition. Each NPZ contains canonical sample_ids/labels, whole_mu,
whole_depth and part_mu_N/part_depth_N for N=200/400/600. Per-N whole_mu_N and
whole_depth_N, if present, take precedence. No encoder, GPU, optimizer or test
evaluation is used. The 20 cross-class negatives are a reproducible diagnostic
panel, not a reconstruction of historical training batches or child.flip(0).
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np


PART_COUNTS = (200, 400, 600)
DEFAULT_TOLERANCE = 1e-4
COLORS = ("#53657D", "#2568A3", "#D18E25", "#388C83", "#9B4E82")
MODELNET40_NAMES = (
    "airplane", "bathtub", "bed", "bench", "bookshelf", "bottle", "bowl",
    "car", "chair", "cone", "cup", "curtain", "desk", "door", "dresser",
    "flower_pot", "glass_box", "guitar", "keyboard", "lamp", "laptop",
    "mantel", "monitor", "night_stand", "person", "piano", "plant",
    "radio", "range_hood", "sink", "sofa", "stairs", "stool", "table",
    "tent", "toilet", "tv_stand", "vase", "wardrobe", "xbox",
)


def configure_fonts():
    available = {font.name for font in font_manager.fontManager.ttflist}
    candidates = ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Arial Unicode MS")
    chinese = next((name for name in candidates if name in available), None)
    if chinese is None:
        raise RuntimeError("A Chinese font is required for the report figures")
    plt.rcParams.update({"font.family": [chinese, "DejaVu Sans"],
                         "axes.unicode_minus": False, "svg.fonttype": "path",
                         "font.size": 10, "axes.titlesize": 11,
                         "savefig.facecolor": "white"})


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2,
                                    allow_nan=False), encoding="utf-8")


def write_csv(path, rows):
    if not rows:
        raise ValueError("Cannot write an empty result table")
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def describe(values, prefix):
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not len(array) or not np.isfinite(array).all():
        raise ValueError("Expected a finite nonempty one-dimensional sample")
    quantiles = np.quantile(array, [0, .01, .05, .1, .25, .5, .75, .9, .95, .99, 1])
    names = ("min", "p01", "p05", "p10", "p25", "median", "p75", "p90", "p95", "p99", "max")
    result = {f"{prefix}_{key}": float(value) for key, value in zip(names, quantiles)}
    result[f"{prefix}_mean"] = float(array.mean())
    result[f"{prefix}_std"] = float(array.std())
    result[f"{prefix}_variance"] = float(array.var())
    return result


def ball_depth(mu, c):
    radius = np.linalg.norm(np.asarray(mu, dtype=np.float64), axis=-1)
    scaled = math.sqrt(c) * radius
    if np.any(scaled >= 1) or np.any(scaled < 0):
        raise ValueError("An embedding is outside the open Poincare ball")
    return 2 / math.sqrt(c) * np.arctanh(scaled)


def ball_distance(x, y, c):
    """Exact Poincare distance in float64, with ordinary NumPy broadcasting."""
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    x_square, y_square = np.sum(x * x, axis=-1), np.sum(y * y, axis=-1)
    left, right = 1 - c * x_square, 1 - c * y_square
    if np.any(left <= 0) or np.any(right <= 0):
        raise ValueError("Distance requested for an out-of-ball coordinate")
    argument = math.sqrt(c) * np.linalg.norm(x - y, axis=-1) / np.sqrt(left * right)
    return 2 / math.sqrt(c) * np.arcsinh(argument)


def cross_class_negatives(ids, labels, count, seed):
    """Same selected object identities across models, stages and curvatures."""
    result = np.empty((len(ids), count), dtype=np.int64)
    for position, (sample_id, label) in enumerate(zip(ids, labels)):
        if sample_id < 0:
            raise ValueError("Canonical object IDs must be nonnegative")
        pool = np.flatnonzero(labels != label)
        if len(pool) < count:
            raise ValueError("Too few different-class objects for distinct negatives")
        generator = np.random.default_rng(np.random.SeedSequence([int(seed), int(sample_id), 673]))
        result[position] = generator.choice(pool, size=count, replace=False)
    if np.any(labels[result] == labels[:, None]):
        raise AssertionError("A diagnostic negative has the anchor class")
    return result


def safe_name(text):
    result = re.sub(r"[^A-Za-z0-9_-]+", "_", str(text)).strip("_")
    return result or "observation"


@dataclass
class Observation:
    metadata: dict
    ids: np.ndarray
    labels: np.ndarray
    arrays: dict
    source_sha256: str
    checks: dict


def load_observation(entry):
    required = ("label", "stage", "cache", "kind", "c", "gamma", "delta", "epoch")
    if any(key not in entry for key in required):
        raise ValueError("Each observation requires " + ", ".join(required))
    if any(token in str(entry["label"]) for token in ("/", "\\", ":")):
        raise ValueError("Use a public display label, not a private path or host")
    source = Path(entry["cache"])
    source_hash = file_sha256(source)
    c = float(entry["c"])
    if min(c, float(entry["gamma"]), float(entry["delta"])) <= 0:
        raise ValueError("Curvature and both margin parameters must be positive")
    arrays = {}
    depth_errors = []
    with np.load(source, allow_pickle=False) as cache:
        ids = np.asarray(cache["sample_ids"], dtype=np.int64)
        labels = np.asarray(cache["labels"], dtype=np.int64).reshape(-1)
        order = np.argsort(ids, kind="stable")
        if ids.ndim != 1 or len(ids) != len(labels) or len(np.unique(ids)) != len(ids):
            raise ValueError("Object IDs must be one-dimensional and unique")
        ids, labels = ids[order], labels[order]
        condition = str(entry.get("condition", "native_eval"))
        if "condition" in cache:
            cached_condition = str(np.asarray(cache["condition"]).item())
            if "condition" in entry and cached_condition != condition:
                raise ValueError("The supplied forward condition differs from its cache")
            condition = cached_condition
        for count in PART_COUNTS:
            for role in ("whole", "part"):
                mu_key = f"{role}_mu_{count}"
                if mu_key not in cache and role == "whole":
                    mu_key = "whole_mu"
                depth_key = f"{role}_depth_{count}"
                if depth_key not in cache and role == "whole":
                    depth_key = "whole_depth"
                vectors = np.asarray(cache[mu_key], dtype=np.float64)[order]
                depth = np.asarray(cache[depth_key], dtype=np.float64).reshape(-1)[order]
                if vectors.ndim != 2 or len(vectors) != len(ids) or depth.shape != ids.shape:
                    raise ValueError("All embeddings and depths must match the object IDs")
                if not np.isfinite(vectors).all() or not np.isfinite(depth).all() or np.any(depth < 0):
                    raise ValueError("Non-finite or negative depth")
                recomputed_depth = ball_depth(vectors, c)
                depth_errors.append(float(np.max(np.abs(recomputed_depth - depth))))
                arrays[f"{role}_mu_{count}"] = vectors
                arrays[f"{role}_depth_{count}"] = depth
    if file_sha256(source) != source_hash:
        raise AssertionError("A source cache changed while it was being read")
    if max(depth_errors) > .001:
        raise ValueError("Depths disagree with coordinates/curvature by more than .001")
    public_metadata = {key: entry[key] for key in required if key != "cache"}
    public_metadata.update(c=c, gamma=float(entry["gamma"]), delta=float(entry["delta"]),
                           condition=condition)
    for key in ("input_protocol", "checkpoint_sha256", "source_commit"):
        if key in entry:
            public_metadata[key] = entry[key]
    checks = {"sample_count": len(ids), "canonical_ids_sha256": array_sha256(ids),
              "labels_sha256": array_sha256(labels), "source_cache_sha256": source_hash,
              "depth_coordinate_max_abs_error": max(depth_errors),
              "depth_source": ("historical float64 coordinate-derived depth" if entry["kind"] == "original" and condition.startswith("native_eval")
                               else "cached native model dist0; coordinates verified in float64"),
              "source_cache_unchanged": True, "sort": "ascending canonical sample_id"}
    return Observation(public_metadata, ids, labels, arrays, source_hash, checks)


def analyze_observation(observation, negatives, tolerance):
    arrays, metadata = observation.arrays, observation.metadata
    negative_count = negatives.shape[1]
    records, class_records, object_records = [], [], {}
    masks_strict, masks_basic, masks_contrastive, masks_joint = [], [], [], []
    for count in PART_COUNTS:
        whole, part = arrays[f"whole_mu_{count}"], arrays[f"part_mu_{count}"]
        whole_depth, part_depth = arrays[f"whole_depth_{count}"], arrays[f"part_depth_{count}"]
        margin = metadata["gamma"] / count
        gap, positive = whole_depth - part_depth, ball_distance(whole, part, metadata["c"])
        slack = gap - margin
        strict, basic, ordered = slack >= -tolerance, gap >= .9 * margin - tolerance, gap > 0
        hinges = np.maximum(-slack, 0)
        negative_distance = np.empty(negatives.shape, dtype=np.float64)
        for start in range(0, len(whole), 256):
            stop = min(start + 256, len(whole))
            negative_distance[start:stop] = ball_distance(whole[start:stop, None, :],
                part[negatives[start:stop]], metadata["c"])
        contrastive_slack = negative_distance - positive[:, None] - metadata["delta"]
        pair_pass = contrastive_slack >= -tolerance
        object_rate = pair_pass.mean(axis=1)
        contrastive_ninety = pair_pass.sum(axis=1) >= math.ceil(.9 * negative_count)
        joint = strict & contrastive_ninety
        record = {**metadata, "part_count": count, "objects": len(gap),
                  "radial_margin": margin, "distance_tolerance": tolerance,
                  "radial_strict_without_tolerance_count": int(np.sum(slack >= 0)),
                  "radial_strict_without_tolerance_percent": float(100 * np.mean(slack >= 0)),
                  "radial_margin_met_count": int(strict.sum()),
                  "radial_margin_met_percent": float(100 * strict.mean()),
                  "radial_90pct_margin_met_count": int(basic.sum()),
                  "radial_90pct_margin_met_percent": float(100 * basic.mean()),
                  "whole_deeper_count": int(ordered.sum()),
                  "whole_deeper_percent": float(100 * ordered.mean()),
                  "whole_V8_cap_percent": float(100 * np.mean(whole_depth >= 4.5999)) if metadata["kind"] == "v8" else None,
                  "part_V8_cap_percent": float(100 * np.mean(part_depth >= 4.5999)) if metadata["kind"] == "v8" else None,
                  "whole_deep_enough_even_if_part_at_origin_percent": float(100 * np.mean(whole_depth >= margin - tolerance)),
                  "negative_parts_per_object": negative_count,
                  "negative_pairs": int(pair_pass.size),
                  "contrastive_pair_pass_count": int(pair_pass.sum()),
                  "contrastive_pair_pass_percent": float(100 * pair_pass.mean()),
                  "contrastive_object_90pct_pass_count": int(contrastive_ninety.sum()),
                  "contrastive_object_90pct_pass_percent": float(100 * contrastive_ninety.mean()),
                  "joint_radial_and_contrastive_90pct_count": int(joint.sum()),
                  "joint_radial_and_contrastive_90pct_percent": float(100 * joint.mean())}
        for values, prefix in ((whole_depth, "whole_depth"), (part_depth, "part_depth"),
                               (gap, "depth_gap"), (slack, "radial_slack"),
                               (hinges, "radial_hinge"), (positive, "positive_distance"),
                               (object_rate, "contrastive_object_pass_fraction"),
                               (contrastive_slack.reshape(-1), "contrastive_slack"),
                               (np.maximum(-contrastive_slack, 0).reshape(-1), "contrastive_hinge")):
            record.update(describe(values, prefix))
        records.append(record)
        for class_id in np.unique(observation.labels):
            selected = observation.labels == class_id
            class_record = {**metadata, "part_count": count, "class_id": int(class_id),
                "class_name": MODELNET40_NAMES[class_id] if 0 <= class_id < 40 else str(class_id),
                "objects": int(selected.sum()), "radial_margin": margin,
                "radial_margin_met_count": int(strict[selected].sum()),
                "radial_margin_met_percent": float(100 * strict[selected].mean()),
                "radial_90pct_margin_met_percent": float(100 * basic[selected].mean()),
                "whole_deeper_percent": float(100 * ordered[selected].mean()),
                "contrastive_pair_pass_percent": float(100 * pair_pass[selected].mean()),
                "contrastive_object_90pct_pass_percent": float(100 * contrastive_ninety[selected].mean()),
                "joint_radial_and_contrastive_90pct_percent": float(100 * joint[selected].mean())}
            class_record.update(describe(gap[selected], "depth_gap"))
            class_record.update(describe(hinges[selected], "radial_hinge"))
            class_records.append(class_record)
        for key, value in {"depth_gap": gap, "radial_slack": slack, "radial_hinge": hinges,
                           "radial_met": strict, "radial_basic90": basic,
                           "contrastive_pass_fraction": object_rate,
                           "contrastive_90_met": contrastive_ninety, "joint_met": joint,
                           "positive_distance": positive}.items():
            object_records[f"{key}_{count}"] = value
        masks_strict.append(strict)
        masks_basic.append(basic)
        masks_contrastive.append(contrastive_ninety)
        masks_joint.append(joint)
    all_three = {**metadata, "objects": len(observation.ids), "part_counts": "/".join(map(str, PART_COUNTS))}
    for name, masks in (("radial_margin", masks_strict), ("radial_90pct_margin", masks_basic),
                        ("contrastive_90pct", masks_contrastive), ("joint", masks_joint)):
        all_met = np.logical_and.reduce(masks)
        all_three[f"all_three_{name}_count"] = int(all_met.sum())
        all_three[f"all_three_{name}_percent"] = float(100 * all_met.mean())
    object_records["negative_positions"] = negatives
    return records, class_records, all_three, object_records


def save_figure(figure, figures_dir, basename, index):
    figure.savefig(figures_dir / (basename + ".png"), dpi=180, bbox_inches="tight")
    figure.savefig(figures_dir / (basename + ".svg"), bbox_inches="tight")
    plt.close(figure)
    index.append({"stem": basename, "png": "figures/" + basename + ".png",
                  "svg": "figures/" + basename + ".svg"})


def condition_title(condition):
    return {"native_eval": "clean/eval：模型自身保存的BN",
            "BN_current_batch_only": "仅BN使用当前批次：冻结诊断"}.get(condition, condition)


def stage_title(stage):
    return {"last": "训练终点", "best": "各自best", "e300": "e300"}.get(stage, stage)


def grouped_observations(observations):
    groups = {}
    for observation in observations:
        meta = observation.metadata
        groups.setdefault((meta["condition"], str(meta["stage"])), []).append(observation)
    return groups


def plot_group(group, results, figures_dir, suffix, index):
    first = group[0].metadata
    title = condition_title(first["condition"]) + " · " + stage_title(str(first["stage"]))
    lookup = {(str(row["label"]), int(row["part_count"])): row for row in results}
    width = .75 / len(group)
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.3), sharey=True)
    for position, observation in enumerate(group):
        name = observation.metadata["label"]
        values = [lookup[name, count] for count in PART_COUNTS]
        x = np.arange(3) - .375 + width * (position + .5)
        color = COLORS[position % len(COLORS)]
        axes[0].bar(x, [row["radial_margin_met_percent"] for row in values], width=width,
                    color=color, label=name)
        axes[1].bar(x, [row["radial_90pct_margin_met_percent"] for row in values], width=width, color=color)
        for axis, key in ((axes[0], "radial_margin_met_percent"),
                          (axes[1], "radial_90pct_margin_met_percent")):
            for place, row in zip(x, values):
                value = row[key]
                axis.text(place, value + 1.1, format(value, '.2f' if value < 1 else '.1f'), ha="center", va="bottom", fontsize=8)
    for axis, heading in zip(axes, ("达到完整径向margin（含1e−4数值容差）", "达到90%径向margin")):
        axis.set(xticks=np.arange(3), xticklabels=[f"N={n}" for n in PART_COUNTS],
                 ylim=(0, 108), ylabel="对象比例（%）", title=heading)
        axis.grid(axis="y", alpha=.2)
        axis.set_axisbelow(True)
    axes[0].legend(fontsize=8)
    figure.suptitle(title + "\n每对象独立检验 d0(whole)−d0(part) ≥ γ/N；不是平均深度判据", fontsize=12)
    figure.tight_layout(rect=(0, 0, 1, .88))
    save_figure(figure, figures_dir, "01_radial_margin_satisfaction" + suffix, index)

    figure, axes = plt.subplots(len(group), 3, figsize=(12, 3.35 * len(group)), squeeze=False)
    for row_index, observation in enumerate(group):
        name, gamma = observation.metadata["label"], observation.metadata["gamma"]
        all_values = np.concatenate([observation.arrays[f"{role}_depth_{count}"]
            for role in ("whole", "part") for count in PART_COUNTS])
        maximum = math.ceil(float(all_values.max()) * 2) / 2 + .1
        for column, count in enumerate(PART_COUNTS):
            axis = axes[row_index, column]
            whole, part = observation.arrays[f"whole_depth_{count}"], observation.arrays[f"part_depth_{count}"]
            axis.hexbin(whole, part, gridsize=42, extent=(0, maximum, 0, maximum), mincnt=1,
                        bins="log", cmap="Blues", linewidths=0, rasterized=True)
            margin = gamma / count
            line = np.linspace(0, maximum, 200)
            axis.plot(line, line, "--", color="#8A949E", lw=.8, label="深度相同")
            axis.plot(line[line >= margin], line[line >= margin] - margin,
                      color="#D1495B", lw=1.5, label="Rhier达标边界")
            record = lookup[name, count]
            axis.set(xlim=(0, maximum), ylim=(0, maximum), xlabel="whole原点双曲深度", ylabel="part原点双曲深度",
                     title=f"{name} · N={count} · γ/N={margin:.3f}\n线下达标 {record['radial_margin_met_percent']:.2f}%")
            axis.set_aspect("equal", adjustable="box")
            if row_index == 0 and column == 0:
                axis.legend(fontsize=7, loc="upper left")
    figure.suptitle(title + "\n完整对象对的二维密度；红线下满足径向正则，灰线仅表示whole/part同深", fontsize=12)
    figure.tight_layout(rect=(0, 0, 1, .92))
    save_figure(figure, figures_dir, "02_whole_part_scatter" + suffix, index)

    figure, axes = plt.subplots(1, 3, figsize=(13, 4.1), sharey=True)
    for position, observation in enumerate(group):
        for axis, count in zip(axes, PART_COUNTS):
            gap = observation.arrays[f"whole_depth_{count}"] - observation.arrays[f"part_depth_{count}"]
            ordered = np.sort(gap - observation.metadata["gamma"] / count)
            axis.plot(ordered, np.arange(1, len(ordered) + 1) / len(ordered) * 100,
                      color=COLORS[position % len(COLORS)], label=observation.metadata["label"], lw=1.6)
    for axis, count in zip(axes, PART_COUNTS):
        axis.axvline(0, color="#D1495B", linestyle="--", lw=1.2)
        axis.set(title=f"N={count}", xlabel="径向slack = whole深度 − part深度 − γ/N", ylabel="累计对象比例（%）", ylim=(0, 100))
        axis.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    figure.suptitle(title + "\nslack≥0才完整满足；slack<0的绝对值就是Rhier hinge", fontsize=12)
    figure.tight_layout(rect=(0, 0, 1, .86))
    save_figure(figure, figures_dir, "03_radial_slack_ecdf" + suffix, index)

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.3), sharey=True)
    for position, observation in enumerate(group):
        name = observation.metadata["label"]
        values = [lookup[name, count] for count in PART_COUNTS]
        x = np.arange(3) - .375 + width * (position + .5)
        for axis, key in ((axes[0], "contrastive_object_90pct_pass_percent"),
                          (axes[1], "joint_radial_and_contrastive_90pct_percent")):
            axis.bar(x, [row[key] for row in values], width=width,
                     color=COLORS[position % len(COLORS)], label=name)
            for place, row in zip(x, values):
                value = row[key]
                axis.text(place, value + 1.1, format(value, '.2f' if value < 1 else '.1f'), ha="center", va="bottom", fontsize=8)
    for axis, heading in zip(axes, ("至少18/20异类负part满足Rcontr", "同时满足完整Rhier与左图Rcontr判据")):
        axis.set(xticks=np.arange(3), xticklabels=[f"N={n}" for n in PART_COUNTS],
                 ylim=(0, 108), ylabel="对象比例（%）", title=heading)
        axis.grid(axis="y", alpha=.2)
        axis.set_axisbelow(True)
    axes[0].legend(fontsize=8)
    figure.suptitle(title + "\n共同20个异类负part/对象；受控几何诊断，不是训练flip回放", fontsize=12)
    figure.tight_layout(rect=(0, 0, 1, .88))
    save_figure(figure, figures_dir, "04_contrastive_and_joint" + suffix, index)


def validate_outputs(report_dir):
    text_suffixes = {".json", ".csv", ".svg"}
    for path in report_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in text_suffixes:
            contents = path.read_text(encoding="utf-8-sig")
            if re.search(r"\b(?:\d{1,3}\.){3}\d{1,3}\b|(?i:[a-z]:\\(?:Users|Hycore)\\)|/mnt/[^\s]+", contents):
                raise AssertionError("A public artifact contains a private path or host")


def self_check():
    c = .1
    depths = np.array([1., 2., 3.])
    vectors = np.zeros((3, 2))
    vectors[:, 0] = np.tanh(math.sqrt(c) * depths / 2) / math.sqrt(c)
    np.testing.assert_allclose(ball_depth(vectors, c), depths, atol=1e-12)
    np.testing.assert_allclose(ball_distance(vectors, vectors[::-1], c), np.abs(depths - depths[::-1]), atol=1e-12)
    ids, labels = np.arange(60), np.repeat(np.arange(3), 20)
    selected = cross_class_negatives(ids, labels, 20, 22)
    np.testing.assert_array_equal(selected, cross_class_negatives(ids, labels, 20, 22))
    if np.any(labels[selected] == labels[:, None]) or any(len(np.unique(row)) != 20 for row in selected):
        raise AssertionError("Negative sampling check failed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--private-out-dir", type=Path)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    self_check()
    if args.self_check and args.config is None:
        print("Analytic geometry and reproducible cross-class negative checks passed")
        return
    if args.config is None or args.out_dir is None:
        parser.error("--config and --out-dir are required")
    config = json.loads(args.config.read_text(encoding="utf-8-sig"))
    observations = [load_observation(entry) for entry in config["models"]]
    if not observations:
        raise ValueError("The config contains no observations")
    key_tuples = [(row.metadata["label"], row.metadata["stage"], row.metadata["condition"]) for row in observations]
    if len(set(key_tuples)) != len(key_tuples):
        raise ValueError("Observation labels/stages/conditions must be unique")
    tolerance = float(config.get("tolerance", DEFAULT_TOLERANCE))
    if tolerance != DEFAULT_TOLERANCE:
        raise ValueError("This report uses a fixed 1e-4 numerical tolerance")
    seed, negative_count = int(config.get("seed", 22)), int(config.get("negative_count", 20))
    if negative_count != 20 or seed != 22:
        raise ValueError("This audit uses a fixed 20-negative diagnostic panel and seed22")
    if args.private_out_dir:
        args.private_out_dir.mkdir(parents=True, exist_ok=False)
    figures_dir, tables_dir = args.out_dir / "figures", args.out_dir / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    records, class_records, all_three_records, checks = [], [], [], []
    references, negative_panels = {}, {}
    for observation in observations:
        condition = observation.metadata["condition"]
        if condition in references:
            reference = references[condition]
            np.testing.assert_array_equal(observation.ids, reference.ids)
            np.testing.assert_array_equal(observation.labels, reference.labels)
        else:
            references[condition] = observation
            negative_panels[condition] = cross_class_negatives(observation.ids, observation.labels, negative_count, seed)
        negatives = negative_panels[condition]
        result, per_class, three, objects = analyze_observation(observation, negatives, tolerance)
        records.extend(result)
        class_records.extend(per_class)
        all_three_records.append(three)
        identity = {key: observation.metadata[key] for key in ("label", "stage", "condition")}
        checks.append({**identity, **observation.checks,
                       "negative_object_ids_sha256": array_sha256(observation.ids[negatives]),
                       "negative_classes_all_different": True, "negative_objects_distinct_per_anchor": True})
        observation.arrays.update(objects)
        if args.private_out_dir:
            stem = "__".join(safe_name(identity[key]) for key in ("label", "stage", "condition"))
            np.savez_compressed(args.private_out_dir / (stem + ".npz"),
                sample_ids=observation.ids, labels=observation.labels, **objects)
        print(f"Analyzed {observation.metadata['label']} {observation.metadata['stage']} {condition}: {len(observation.ids)} objects", flush=True)
    write_csv(tables_dir / "paired_margin_summary.csv", records)
    write_csv(tables_dir / "paired_margin_by_class.csv", class_records)
    write_csv(tables_dir / "paired_all_three_sizes.csv", all_three_records)
    summary = {"scope": "frozen clean whole/part geometry; not historical training satisfaction",
        "radial_rule": "d0(whole)-d0(part) >= gamma/N", "tolerance": tolerance,
        "radial_90pct_rule": "gap >= 0.9*gamma/N - tolerance",
        "contrastive_rule": "d(whole,negative_part)-d(whole,own_part) >= delta",
        "contrastive_negative_protocol": "20 distinct cross-class objects per anchor, sampled with seed22 and canonical identity; common across models and stages for each condition",
        "contrastive_object_rule": "at least18/20 negative pairs meet the contrastive margin with tolerance",
        "joint_rule": "full radial margin AND contrastive object rule",
        "negative_count": negative_count, "seed": seed,
        "N_weighting": "separate N=200/400/600; no pooling to imply the training N distribution",
        "curvature_note": "each observation uses its own curvature and gamma/delta; absolute distances differ across curvatures",
        "checks": checks, "by_N": records, "all_three_N": all_three_records,
        "optimizer_updates": 0, "backward_calls": 0, "encoder_calls_in_this_analysis": 0,
        "test_forward_calls": 0}
    write_json(tables_dir / "paired_margin_summary.json", summary)
    configure_fonts()
    figure_index = []
    groups = grouped_observations(observations)
    preferred = next((key for key in groups if key[0] == "native_eval" and key[1] in ("last", "e300")), next(iter(groups)))
    for key, group in groups.items():
        if key[0] == "native_eval_panel":
            continue
        suffix = "" if key == preferred else "_" + safe_name("_".join(key))
        group_rows = [row for row in records if row["condition"] == key[0] and str(row["stage"]) == key[1]]
        plot_group(group, group_rows, figures_dir, suffix, figure_index)
    write_json(tables_dir / "figure_index.json", figure_index)
    for entry, observation in zip(config["models"], observations):
        if file_sha256(entry["cache"]) != observation.source_sha256:
            raise AssertionError("A source cache changed during the analysis")
    validate_outputs(args.out_dir)
    write_json(tables_dir / "paired_margin_validation.json", {
        "analytic_distance_check": "passed", "negative_sampling_reproducible": True,
        "same_objects_and_labels_within_each_condition": True,
        "source_hashes_preserved": True, "public_object_ids_or_paths": False,
        "observations": len(observations), "sample_sizes_by_condition": {key: len(value.ids) for key, value in references.items()},
        "figure_pairs": len(figure_index), "tolerance": tolerance})
    print(json.dumps({"observations": len(observations), "figure_pairs": len(figure_index),
                      "public_tables": 6, "status": "completed"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
