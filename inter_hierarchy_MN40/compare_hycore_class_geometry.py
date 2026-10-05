"""Read-only, matched clean whole/local-patch audit of one HyCoRe checkpoint.

This is an inference diagnostic, not a replay of training telemetry. Whole input
is the raw first 1024 HDF5 points, without augmentation or part overwrite. A
fixed split/ID/seed selects one centre; 200/400/600-point Euclidean kNN patches
are nested around that same centre. These patches have no semantic part labels.
All checkpoints and inference batch sizes receive the same input coordinates.

The model remains in eval mode, with no backward/optimizer calls. FP32 source
inference is followed by FP64 c=1 geometry; no extra clipping is applied. The
source architecture retains its fixed 512 first-stage FPS request for patches
smaller than 512. Per-size summaries use one patch per object, never pooled
patches with an ambiguous denominator. Cached embeddings are optional.

Pure NumPy functions below are importable without Torch/native FPS for CPU
checks and analysis of saved instance caches.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import glob
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time

import numpy as np

REPO = Path(__file__).resolve().parents[1]
SOURCE = REPO / "classification_ModelNet40"
CLASS_NAMES = (
    "airplane", "bathtub", "bed", "bench", "bookshelf", "bottle", "bowl", "car",
    "chair", "cone", "cup", "curtain", "desk", "door", "dresser", "flower_pot",
    "glass_box", "guitar", "keyboard", "lamp", "laptop", "mantel", "monitor",
    "night_stand", "person", "piano", "plant", "radio", "range_hood", "sink",
    "sofa", "stairs", "stool", "table", "tent", "toilet", "tv_stand", "vase",
    "wardrobe", "xbox",
)
NATIVE_NEAR_RADIUS = .99599


def utc_now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(value):
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(str(value.shape).encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                     allow_nan=False) + "\n", encoding="utf-8")


def vector_geometry(features):
    """Raw radius and exact c1 distance from origin, without projection."""
    features = np.asarray(features, dtype=np.float64)
    if features.ndim != 2 or not np.isfinite(features).all():
        raise ValueError("Embedding must be a finite [N,D] array")
    radius = np.linalg.norm(features, axis=1)
    if (radius >= 1).any():
        raise ValueError("Raw embedding lies outside/at the c1 ball boundary")
    return radius, 2 * np.arctanh(radius)


def matched_distances(x, y):
    """Exact c1 rowwise distance, algebraically equivalent to source ball.dist.

    d(x,y)=2 asinh(||x-y||/sqrt((1-||x||^2)(1-||y||^2))). FP64 subtracts
    rows directly, avoiding cancellation in the dot-product squared distance.
    """
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if x.shape != y.shape or x.ndim != 2:
        raise ValueError("Matched embeddings must share [N,D] shape")
    xr, _ = vector_geometry(x)
    yr, _ = vector_geometry(y)
    return 2 * np.arcsinh(np.linalg.norm(x - y, axis=1) /
                         np.sqrt((1 - xr * xr) * (1 - yr * yr)))


def paired_geometry(whole, part, part_size):
    """Per-object, one-patch metrics; all hinge values are unweighted."""
    if not 1 <= int(part_size) <= 1024:
        raise ValueError("part_size must be in [1,1024]")
    whole, part = np.asarray(whole, dtype=np.float64), np.asarray(part, dtype=np.float64)
    wr, wd = vector_geometry(whole)
    pr, pd = vector_geometry(part)
    if whole.shape != part.shape:
        raise ValueError("whole/part shapes must match")
    norms = wr * pr
    direction_defined = norms > 1e-15
    cosine = np.divide(np.einsum("ij,ij->i", whole, part), norms,
                       out=np.ones_like(norms), where=direction_defined)
    # Undefined angles use 0 in compact arrays plus an explicit boolean mask;
    # summaries exclude those rows rather than interpreting 0 as alignment.
    angle = np.degrees(np.arccos(np.clip(cosine, -1., 1.)))
    angle[~direction_defined] = 0
    gap = wd - pd
    return {"part_r": pr, "part_d0": pd, "depth_gap": gap,
            "rhier": np.maximum(1000. / part_size - gap, 0),
            "whole_part_d": matched_distances(whole, part),
            "whole_part_angle_deg": angle, "direction_defined": direction_defined}


def describe(values):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if not np.isfinite(values).all():
        raise ValueError("Summary values must be finite")
    if len(values) == 0:
        return {"count": 0, "mean": None, "p10": None, "median": None, "p90": None}
    quantiles = np.percentile(values, [10, 50, 90])
    return {"count": int(len(values)), "mean": float(values.mean()),
            "p10": float(quantiles[0]), "median": float(quantiles[1]),
            "p90": float(quantiles[2])}


def classification_summary(labels, predictions, ce):
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    predictions = np.asarray(predictions, dtype=np.int64).reshape(-1)
    ce = np.asarray(ce, dtype=np.float64).reshape(-1)
    if not (labels.shape == predictions.shape == ce.shape) or not len(labels):
        raise ValueError("Nonempty classification arrays must align")
    counts = np.bincount(labels, minlength=40)
    correct = np.bincount(labels, weights=(labels == predictions), minlength=40)
    observed = counts > 0
    accuracy = np.divide(correct, counts, out=np.zeros(40), where=observed)
    return {"count": int(len(labels)), "correct": int(correct.sum()),
            "oa_percent": float(100 * correct.sum() / counts.sum()),
            "aa_percent_observed_classes": float(100 * accuracy[observed].mean()),
            "observed_class_count": int(observed.sum()),
            "smoothed_ce_sample_mean": float(ce.mean()),
            "per_class": [{"class_id": i, "class_name": CLASS_NAMES[i],
                           "count": int(counts[i]), "correct": int(correct[i]),
                           "accuracy_percent": float(100 * accuracy[i]) if counts[i] else None}
                          for i in range(40)]}


def deterministic_part_indices(points, sample_ids, split, seed, part_sizes):
    """ID-stable centre and nested nearest-neighbour indices with stable ties.

    Does not reproduce source random draws or destructive part overwrite. It
    isolates representation differences using identical clean diagnostic input.
    The centre distribution is uniform over the 1024 original point positions.
    """
    points = np.asarray(points)
    sample_ids = np.asarray(sample_ids, dtype=np.int64)
    if points.shape != (len(sample_ids), 1024, 3) or split not in ("train", "test"):
        raise ValueError("Expect [N,1024,3] and split train/test")
    sizes = sorted(set(int(size) for size in part_sizes))
    if not sizes or sizes[0] < 24 or sizes[-1] > 1024:
        raise ValueError("Source PointMLP requires diagnostic sizes in [24,1024]")
    centers = np.empty(len(points), dtype=np.int64)
    indices = {size: np.empty((len(points), size), dtype=np.int64) for size in sizes}
    tie_order = np.arange(1024)
    for row, sample_id in enumerate(sample_ids):
        generator = np.random.default_rng(np.random.SeedSequence(
            [int(seed), 0 if split == "train" else 1, int(sample_id)]))
        centre = int(generator.integers(0, 1024))
        centers[row] = centre
        xyz = np.asarray(points[row], dtype=np.float64)
        distance = np.einsum("ij,ij->i", xyz - xyz[centre], xyz - xyz[centre])
        order = np.lexsort((tie_order, distance))
        for size in sizes:
            indices[size][row] = order[:size]
    return centers, indices


def rows_and_summary(cache, part_sizes):
    """Make long-form CSV rows and pooled summary from a compact cache."""
    labels = cache["labels"]
    whole_summary = classification_summary(labels, cache["whole_prediction"], cache["whole_ce"])
    rows = []
    summary = {"whole_classification": whole_summary,
               "whole_r": describe(cache["whole_r"]),
               "whole_d0": describe(cache["whole_d0"]),
               "whole_near_native_percent": float(100 * np.mean(cache["whole_r"] >= NATIVE_NEAR_RADIUS)),
               "parts": {}, "all_finite": True,
               "unit": "c1 hyperbolic distance; angle in degrees; rates in percent",
               "patch_prediction_interpretation": "Agreement with the source object's class label, not semantic-part accuracy",
               "denominator": "whole: one observation per object; each part size: one fixed-centre patch per object; no cross-size pooling"}

    def add_stats(row, prefix, values):
        for name, value in describe(values).items():
            if name != "count":
                row[f"{prefix}_{name}"] = value

    for cls in range(40):
        mask = labels == cls
        row = {"class_id": cls, "class_name": CLASS_NAMES[cls],
               "representation": "whole", "num_points": 1024,
               "object_count": int(mask.sum()), "observation_count": int(mask.sum()),
               "correct_whole": int(np.sum(cache["whole_prediction"][mask] == cls)),
               "whole_accuracy_percent": float(100 * np.mean(cache["whole_prediction"][mask] == cls)) if mask.any() else None,
               "whole_smoothed_ce": float(cache["whole_ce"][mask].mean()) if mask.any() else None}
        add_stats(row, "r", cache["whole_r"][mask])
        add_stats(row, "d0", cache["whole_d0"][mask])
        row["near_native_percent"] = float(100 * np.mean(cache["whole_r"][mask] >= NATIVE_NEAR_RADIUS)) if mask.any() else None
        rows.append(row)
    for size in part_sizes:
        prefix = f"part_{size}_"
        part_summary = {name: describe(cache[prefix + name]) for name in
                        ("r", "d0", "depth_gap", "rhier", "whole_part_d")}
        defined = cache[prefix + "direction_defined"]
        part_summary["angle_deg"] = describe(cache[prefix + "whole_part_angle_deg"][defined])
        part_summary["direction_defined_count"] = int(defined.sum())
        part_summary["near_native_percent"] = float(100 * np.mean(cache[prefix + "r"] >= NATIVE_NEAR_RADIUS))
        part_summary["rhier_active_percent"] = float(100 * np.mean(cache[prefix + "rhier"] > 0))
        part_summary["whole_deeper_percent"] = float(100 * np.mean(cache[prefix + "depth_gap"] > 0))
        part_summary["rhier_margin"] = 1000. / size
        part_summary["auxiliary_class_label_agreement_percent"] = float(100 * np.mean(cache[prefix + "prediction"] == labels))
        part_summary["object_count"] = int(len(labels))
        summary["parts"][str(size)] = part_summary
        for cls in range(40):
            mask = labels == cls
            row = {"class_id": cls, "class_name": CLASS_NAMES[cls],
                   "representation": "local_knn_patch", "num_points": int(size),
                   "object_count": int(mask.sum()), "observation_count": int(mask.sum()),
                   "rhier_margin": 1000. / size,
                   "auxiliary_class_label_agreement_percent": float(100 * np.mean(cache[prefix + "prediction"][mask] == cls)) if mask.any() else None}
            for metric in ("r", "d0", "depth_gap", "rhier", "whole_part_d"):
                add_stats(row, metric, cache[prefix + metric][mask])
            angle_mask = mask & cache[prefix + "direction_defined"]
            add_stats(row, "angle_deg", cache[prefix + "whole_part_angle_deg"][angle_mask])
            row["direction_defined_count"] = int(angle_mask.sum())
            row["near_native_percent"] = float(100 * np.mean(cache[prefix + "r"][mask] >= NATIVE_NEAR_RADIUS)) if mask.any() else None
            row["rhier_active_percent"] = float(100 * np.mean(cache[prefix + "rhier"][mask] > 0)) if mask.any() else None
            row["whole_deeper_percent"] = float(100 * np.mean(cache[prefix + "depth_gap"][mask] > 0)) if mask.any() else None
            rows.append(row)
    return rows, summary


def write_csv(path, rows):
    fields = list(dict.fromkeys(field for row in rows for field in row))
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_split(data_dir, split, limit):
    import h5py
    # Match data.load_data: glob order, not a newly sorted convention. Record
    # exact file order and data digest so different filesystems cannot silently
    # align different object IDs. Each model on the same server sees this order.
    filenames = glob.glob(str(Path(data_dir) / f"ply_data_{split}*.h5"))
    if not filenames:
        raise FileNotFoundError(f"No {split} HDF5 shards found in {data_dir}")
    points, labels, shards = [], [], []
    for filename in filenames:
        with h5py.File(filename, "r") as shard:
            cloud = shard["data"][:, :1024, :].astype(np.float32)
            label = shard["label"][:].astype(np.int64).reshape(-1)
        if cloud.shape != (len(label), 1024, 3):
            raise ValueError("Source HDF5 must provide XYZ and at least1024 points")
        points.append(cloud)
        labels.append(label)
        shards.append({"name": Path(filename).name, "count": int(len(label))})
    points, labels = np.concatenate(points), np.concatenate(labels)
    count_full = len(labels)
    if limit:
        points, labels = points[:limit], labels[:limit]
    if not np.isfinite(points).all() or ((labels < 0) | (labels >= 40)).any():
        raise ValueError("Invalid source data/labels")
    ids = np.arange(len(labels), dtype=np.int64)
    return points, labels, ids, {"split": split, "source_shards_in_glob_order": shards,
                                "full_source_count": count_full, "evaluated_count": len(labels),
                                "complete_split": len(labels) == count_full,
                                "points1024_sha256": array_sha256(points),
                                "labels_sha256": array_sha256(labels),
                                "ids_sha256": array_sha256(ids)}


def inference_split(model, points, labels, ids, split, args, torch, device):
    centers, part_indices = deterministic_part_indices(points, ids, split, args.seed, args.part_sizes)
    points_before = array_sha256(points)
    count = len(ids)
    cache = {"ids": ids, "labels": labels, "part_center_index": centers}
    for name, dtype in (("whole_prediction", np.int64), ("whole_ce", np.float64),
                        ("whole_r", np.float64), ("whole_d0", np.float64)):
        cache[name] = np.empty(count, dtype=dtype)
    if args.save_features:
        cache["whole_mu"] = np.empty((count, 256), dtype=np.float32)
    for size in args.part_sizes:
        for name in ("r", "d0", "depth_gap", "rhier", "whole_part_d", "whole_part_angle_deg"):
            cache[f"part_{size}_{name}"] = np.empty(count, dtype=np.float64)
        cache[f"part_{size}_direction_defined"] = np.empty(count, dtype=bool)
        cache[f"part_{size}_prediction"] = np.empty(count, dtype=np.int64)
        if args.save_features:
            cache[f"part_{size}_mu"] = np.empty((count, 256), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, count, args.batch_size):
            stop = min(start + args.batch_size, count)
            clean = np.ascontiguousarray(points[start:stop])
            cloud = torch.from_numpy(clean).to(device)
            whole_mu, logits = model(cloud.transpose(1, 2).contiguous())
            if whole_mu.shape != (stop - start, 256) or logits.shape != (stop - start, 40):
                raise RuntimeError("Unexpected model output shapes")
            if not torch.isfinite(whole_mu).all() or not torch.isfinite(logits).all():
                raise RuntimeError("Non-finite whole inference")
            whole = whole_mu.detach().cpu().numpy()
            log_probs = torch.log_softmax(logits, dim=1)
            gold = torch.from_numpy(labels[start:stop]).to(device)
            # Exact source epsilon=.2 smoothing over39 incorrect classes.
            gold_log = log_probs.gather(1, gold[:, None]).squeeze(1)
            ce = -(.8 * gold_log + (.2 / 39) * (log_probs.sum(1) - gold_log))
            cache["whole_prediction"][start:stop] = logits.argmax(1).cpu().numpy()
            cache["whole_ce"][start:stop] = ce.cpu().numpy().astype(np.float64)
            wr, wd = vector_geometry(whole)
            cache["whole_r"][start:stop], cache["whole_d0"][start:stop] = wr, wd
            if args.save_features:
                cache["whole_mu"][start:stop] = whole
            for size in args.part_sizes:
                indices = part_indices[size][start:stop]
                patch = clean[np.arange(stop - start)[:, None], indices]
                part_mu, part_logits = model(torch.from_numpy(np.ascontiguousarray(patch)).to(device).transpose(1, 2).contiguous())
                if part_mu.shape != whole_mu.shape or not torch.isfinite(part_mu).all() or not torch.isfinite(part_logits).all():
                    raise RuntimeError("Invalid patch inference")
                part = part_mu.detach().cpu().numpy()
                metrics = paired_geometry(whole, part, size)
                for name, values in metrics.items():
                    target = name[5:] if name in ("part_r", "part_d0") else name
                    cache[f"part_{size}_{target}"][start:stop] = values
                cache[f"part_{size}_prediction"][start:stop] = part_logits.argmax(1).cpu().numpy()
                if args.save_features:
                    cache[f"part_{size}_mu"][start:stop] = part
            if start == 0 or stop == count or (start // args.batch_size + 1) % 50 == 0:
                print(f"{args.run_label}: {split} {stop}/{count}", flush=True)
    if array_sha256(points) != points_before:
        raise RuntimeError("Clean whole source points changed during patch extraction")
    for key, value in cache.items():
        if not np.isfinite(value).all():
            raise RuntimeError(f"Non-finite cached {key}")
    return cache, {"part_centers_sha256": array_sha256(centers),
                   "part_indices_sha256_by_size": {str(size): array_sha256(part_indices[size]) for size in args.part_sizes},
                   "clean_whole_input_unchanged": True}


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--run-label", required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True, help="New directory; never overwrite an existing result")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--split", nargs="+", choices=("train", "test"), default=["train", "test"])
    parser.add_argument("--part-sizes", nargs="+", type=int, default=[200, 400, 600])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=22)
    parser.add_argument("--save-features", action="store_true", help="Keep private FP32 whole_mu and part_<size>_mu arrays for CPU matched-batch analysis")
    parser.add_argument("--expected-epoch", type=int, help="External epoch declaration recorded separately; does not rewrite source zero-based epoch")
    parser.add_argument("--limit", type=int, default=0, help="Positive: partial startup diagnostic only; zero: full splits")
    args = parser.parse_args(argv)
    args.part_sizes = sorted(set(args.part_sizes))
    args.split = list(dict.fromkeys(args.split))
    if args.batch_size < 1 or args.seed < 0 or args.limit < 0 or min(args.part_sizes) < 24 or max(args.part_sizes) > 1024:
        parser.error("Need positive batch, nonnegative seed/limit, and part sizes24..1024")
    if args.expected_epoch is not None and args.expected_epoch < 1:
        parser.error("expected-epoch must be a positive external declaration")
    args.checkpoint = args.checkpoint.resolve(strict=True)
    args.data_dir = args.data_dir.resolve(strict=True)
    args.output_dir = args.output_dir.resolve()
    if args.output_dir.exists():
        parser.error("output-dir must be NEW")
    return args


def checkpoint_metadata(saved):
    def scalar(value):
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        if hasattr(value, "numel") and value.numel() == 1:
            return value.item()
        return None
    return {key: scalar(saved.get(key)) for key in
            ("epoch", "completed_epochs", "acc", "best_acc", "format", "wrapper_protocol") if key in saved}


def cpu_self_check():
    """Meaningful geometry/input-alignment checks, no Torch import or GPU."""
    x = np.array([[.2, 0.], [.7, 0.], [.99599998, 0.]], dtype=np.float32)
    y = np.array([[.1, 0.], [.3, 0.], [.5, 0.]], dtype=np.float32)
    _xr, xd = vector_geometry(x)
    _yr, yd = vector_geometry(y)
    np.testing.assert_allclose(matched_distances(x, y), xd - yd, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(matched_distances(x, y), matched_distances(y, x), atol=0)
    np.testing.assert_array_equal(matched_distances(x, x), np.zeros(3))
    paired = paired_geometry(x, y, 400)
    np.testing.assert_allclose(paired["rhier"], np.maximum(2.5 - (xd - yd), 0))
    np.testing.assert_allclose(paired["whole_part_angle_deg"], 0)
    undefined = paired_geometry(np.zeros((1, 2)), np.ones((1, 2)) * .1, 200)
    assert not undefined["direction_defined"][0]
    cloud = np.random.default_rng(123).normal(size=(3, 1024, 3)).astype(np.float32)
    ids = np.array([9, 42, 17])
    centers, indices = deterministic_part_indices(cloud, ids, "train", 22, [200, 400, 600])
    for row in range(3):
        single_centers, single_indices = deterministic_part_indices(cloud[row:row + 1], ids[row:row + 1], "train", 22, [200, 400, 600])
        assert centers[row] == single_centers[0]
        for size in (200, 400, 600):
            np.testing.assert_array_equal(indices[size][row], single_indices[size][0])
        np.testing.assert_array_equal(indices[200][row], indices[600][row, :200])
    counts = classification_summary(np.array([0, 0, 1]), np.array([0, 1, 1]), np.array([1., 2., 3.]))
    np.testing.assert_allclose(counts["oa_percent"], 200 / 3)
    np.testing.assert_allclose(counts["aa_percent_observed_classes"], 75)
    cache = {"labels": np.array([0, 0, 1]), "whole_prediction": np.array([0, 1, 1]),
             "whole_ce": np.array([1., 2., 3.]), "whole_r": _xr, "whole_d0": xd}
    for key, value in paired.items():
        cache["part_400_" + (key[5:] if key in ("part_r", "part_d0") else key)] = value
    cache["part_400_prediction"] = np.array([0, 0, 1])
    rows, summary = rows_and_summary(cache, [400])
    assert len(rows) == 80 and summary["parts"]["400"]["object_count"] == 3
    assert rows[2]["object_count"] == 0 and rows[2]["r_mean"] is None
    return {"status": "passed", "checks": ["radial source-compatible distance", "distance symmetry/self-zero",
             "hinge/depth/angular statistics", "undefined-angle mask", "part ID/batch-size invariance",
             "nested patches", "OA/AA denominators", "40-class rows and missing-class handling"]}


def main(argv=None):
    args = arguments(argv)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    manifest = {"format": "hycore-clean-class-geometry-1", "status": "starting",
                "started_utc": utc_now(), "run_label": args.run_label,
                "audit_script_sha256": sha256_file(__file__),
                "checkpoint": str(args.checkpoint), "checkpoint_sha256": sha256_file(args.checkpoint),
                "checkpoint_size_bytes": args.checkpoint.stat().st_size,
                "expected_epoch_external_declaration": args.expected_epoch,
                "command": sys.argv, "config": {key: (str(value) if isinstance(value, Path) else value) for key, value in vars(args).items()},
                "method": {"architecture": "Hype_pointMLP", "curvature": 1., "embedding_dim": 256,
                           "mode": "eval/no_grad", "whole": "raw first1024 HDF points; no augmentation; no overwrite",
                           "patch": "one fixed ID/seed centre, Euclidean kNN sorted by distance then source-point-index; nested sizes",
                           "semantic_part_labels": False, "geometry_precision": "float64 from unchanged float32 embeddings",
                           "near_native_radius_threshold": NATIVE_NEAR_RADIUS,
                           "rhier": "max(1000/Npart - (d0(whole)-d0(part)),0), unweighted",
                           "training_telemetry_equivalent": False,
                           "source_fps": "native FPS; unchanged first512 request even when patch has fewer points",
                           "fps_initial_point": "Repository CUDA kernel sampling_gpu.cu starts from source point index0 (not the unused Python random-start fallback)",
                           "determinism_boundary": "Fixed input and seed, eval BN, cudnn deterministic/benchmark off; no claim of bitwise equality across GPU architectures, kernel versions or inference batch sizes"}}
    try:
        manifest["code_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    except (OSError, subprocess.SubprocessError):
        manifest["code_commit"] = None
    write_json(args.output_dir / "manifest.json", manifest)
    try:
        import torch
        sys.path.insert(0, str(SOURCE))
        from models.pointmlp import Hype_pointMLP

        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        device = torch.device(args.device)
        if device.type != "cuda" or not torch.cuda.is_available():
            raise ValueError("Source PointMLP native FPS requires an available CUDA device")
        torch.cuda.set_device(device)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        if not isinstance(saved, dict) or not isinstance(saved.get("net"), dict) or not saved["net"]:
            raise ValueError("Require original-compatible nonempty checkpoint['net'] state dictionary")
        model = Hype_pointMLP().to(device)
        state = {key[7:] if key.startswith("module.") else key: value for key, value in saved["net"].items()}
        model.load_state_dict(state, strict=True)
        for name in ("manifold", "manifold2"):
            curvature = float(getattr(model, name).c.detach())
            if not math.isclose(curvature, 1., abs_tol=1e-5):
                raise ValueError(f"Loaded {name} curvature {curvature} differs from c1")
        model.eval()
        model.requires_grad_(False)
        buffers_before = {name: buffer.detach().cpu().clone() for name, buffer in model.named_buffers()}
        manifest["checkpoint_fields_unmodified"] = checkpoint_metadata(saved)
        if args.expected_epoch is not None and isinstance(saved.get("completed_epochs"), int) and saved["completed_epochs"] != args.expected_epoch:
            raise ValueError("External expected epoch differs from explicit completed_epochs; raw epoch is not inferred")
        manifest["runtime"] = {"torch": torch.__version__, "numpy": np.__version__,
                               "cuda": torch.version.cuda, "device": str(device),
                               "gpu_name": torch.cuda.get_device_name(device),
                               "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}
        manifest["status"] = "running"
        write_json(args.output_dir / "manifest.json", manifest)
        reports = {}
        for split in args.split:
            split_started = time.monotonic()
            split_dir = args.output_dir / split
            split_dir.mkdir(exist_ok=False)
            points, labels, ids, identity = load_split(args.data_dir, split, args.limit)
            cache, patch_identity = inference_split(model, points, labels, ids, split, args, torch, device)
            rows, summary = rows_and_summary(cache, args.part_sizes)
            summary.update({"run_label": args.run_label, "checkpoint_sha256": manifest["checkpoint_sha256"],
                            "checkpoint_fields_unmodified": manifest["checkpoint_fields_unmodified"],
                            "data_identity": identity, "patch_identity": patch_identity,
                            "split_seconds": time.monotonic() - split_started,
                            "features_saved": args.save_features,
                            "scope": "full clean split" if identity["complete_split"] else "partial startup diagnostic; not full-split result"})
            np.savez_compressed(split_dir / "instances.npz", **cache)
            write_csv(split_dir / "classes.csv", rows)
            write_json(split_dir / "summary.json", summary)
            reports[split] = summary
            # Detect buffer mutation immediately after each split as well as at
            # the end, including BN running stats/counters and manifold buffers.
            current_buffers = dict(model.named_buffers())
            changed = [name for name, before in buffers_before.items()
                       if name not in current_buffers or not torch.equal(before, current_buffers[name].detach().cpu())]
            if changed or current_buffers.keys() != buffers_before.keys():
                raise RuntimeError(f"Eval model buffers changed: {changed}")
            print(f"{args.run_label}: {split} OA={summary['whole_classification']['oa_percent']:.6f}% AA={summary['whole_classification']['aa_percent_observed_classes']:.6f}%", flush=True)
        after_hash = sha256_file(args.checkpoint)
        if after_hash != manifest["checkpoint_sha256"]:
            raise RuntimeError("Checkpoint file changed while audit was running")
        manifest.update({"status": "completed", "finished_utc": utc_now(),
                         "wall_seconds": time.monotonic() - started,
                         "bn_and_all_buffers_unchanged": True, "all_finite": True,
                         "checkpoint_unchanged": True,
                         "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device),
                         "splits": reports})
        write_json(args.output_dir / "manifest.json", manifest)
    except BaseException as error:
        manifest.update({"status": "failed", "finished_utc": utc_now(),
                         "wall_seconds": time.monotonic() - started,
                         "error": f"{type(error).__name__}: {error}"})
        write_json(args.output_dir / "manifest.json", manifest)
        raise


if __name__ == "__main__":
    main()
