"""Read-only matched clean whole export of complete historical H20/B0 epochs40.

Requires the immutable named e40 checkpoints and the shared immutable e20
prefix. Only canonical training IDs8856 are forwarded, in eval batches32,
using raw first1024 XYZ without augmentation, shuffling, crops or overwrite.
No test file, validation forward, proxy GPU allocation, backward or update.
Pairwise geometry is exact FP64 over every unordered within-class pair.
The output caches and full object identities are private run artifacts.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

# Set before importing NumPy/Torch so this diagnostic does not occupy the
# training job's CPU pool. The caller selects the already inspected idle GPU.
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from inter_hierarchy_MN40.v6_balanced_b0.contract import (
    CHECKPOINT_VERSION as B0_FORMAT, SOURCE_VERSION as H20_FORMAT,
    derive_b0_config, validate_checkpoint_contract,
)

NAMES = "airplane bathtub bed bench bookshelf bottle bowl car chair cone cup curtain desk door dresser flower_pot glass_box guitar keyboard lamp laptop mantel monitor night_stand person piano plant radio range_hood sink sofa stairs stool table tent toilet tv_stand vase wardrobe xbox".split()
EXPECTED_SOURCE = {
    "architecture": "Hype_pointMLP", "num_classes": 40, "seed": 22,
    "epochs": 300, "steps_per_epoch": 200, "warmup_epochs": 20,
    "global_batch": 64, "world_size": 2, "classes_per_rank": 16,
    "instances_per_class": 2, "c": 1, "D": 256, "P": 512,
    "sample_K": 20, "proxy_K": 20, "T": 50, "margin": .1, "tau": .1,
    "lambda_hier_after_warmup": .5, "label_smoothing": .2,
    "alpha_contrastive": .01, "alpha_radial": .01, "contrastive_margin": 4.,
    "whole_count_range": [800, 1024], "child_count_range": [200, 600],
    "part_overwrites_whole": True, "FPS_first_stage": 512,
    "precision": "FP32", "self_negative": False,
    "extra_HIER_tangent_cap": False, "HIER_backward_hook": False,
    "smoke": False, "evaluation_batch_limit": None,
    "workers": 2, "eval_batch_size": 32, "train_eval_every": 10,
    "sampler": "class-uniform; class-internal replacement; contiguous class blocks",
    "radial_margin": "1000 / child_count",
    "BN": "ordinary local32 training; child and whole update; rank0 buffer broadcast",
    "self_exclusion_scope": "index i=k in sample and proxy graphs",
    "proxy_initialization": "random tangent",
    "model_optimizer": {"name": "RiemannianSGD", "lr": .1, "min_lr": .005,
                        "momentum": .9, "weight_decay": 2e-4, "norm_clip": 1.},
    "proxy_optimizer": {"name": "AdamW", "lr": .01, "min_lr": .0005,
                        "betas": [.9, .999], "eps": 1e-8, "weight_decay": .01,
                        "gradient_clipping": None},
    "schedule": "both 300-epoch cosine; proxy paused during first20 epochs; no restart",
    "validation": "clean no augmentation; best OA then lower smoothed CE",
}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            result.update(block)
    return result.hexdigest()


def array_hash(value):
    value = np.ascontiguousarray(value)
    result = hashlib.sha256()
    result.update(str(value.dtype).encode())
    result.update(str(value.shape).encode())
    result.update(value.tobytes())
    return result.hexdigest()


def state_hash(state):
    result = hashlib.sha256()
    for name, tensor in state.items():
        result.update(name.encode())
        result.update(array_hash(tensor.detach().cpu().numpy()).encode())
    return result.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_csv(path, rows):
    import csv
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def ids_from(saved, key, count):
    raw = np.asarray(saved.get(key, []))
    if raw.ndim != 1 or raw.dtype.kind not in "iu" or len(raw) != count:
        raise ValueError(f"{key} must have exactly{count} integer IDs")
    ids = raw.astype(np.int64)
    if np.any(ids < 0) or np.any(ids >= 9840) or np.any(np.diff(ids) <= 0):
        raise ValueError(f"{key} must be sorted, unique canonical train-shard IDs")
    return ids


def split_sha(train, valid):
    result = hashlib.sha256()
    result.update(np.asarray(train, dtype=np.int64).tobytes())
    result.update(np.asarray(valid, dtype=np.int64).tobytes())
    return result.hexdigest()


def validate_complete(saved, format_name, epoch):
    if saved.get("format") != format_name or type(saved.get("epoch")) is not int or type(saved.get("completed_epochs")) is not int:
        raise ValueError("Unsupported complete checkpoint format or epoch type")
    if saved["epoch"] != epoch or saved["completed_epochs"] != epoch:
        raise ValueError(f"Require the complete named epoch{epoch} checkpoint")
    for flag in ("model_selection_only", "diagnostic_only", "continuation_smoke_not_resumable"):
        if saved.get(flag, False) is not False:
            raise ValueError("Selection/smoke/diagnostic checkpoint rejected: " + flag)
    required = {"net", "optimizer", "scheduler", "rank_states", "best", "best_net",
                "initialization", "initialized_model_sha256", "commit", "training_config",
                "split_sha256", "train_ids", "validation_ids"}
    if format_name == H20_FORMAT:
        required |= {"proxy", "proxy_optimizer", "proxy_scheduler"}
    elif format_name == B0_FORMAT:
        required |= {"fork_source"}
        if any(key in saved for key in ("proxy", "proxy_optimizer", "proxy_scheduler")):
            raise ValueError("B0 checkpoint must not contain a live proxy arm")
    if not required.issubset(saved) or not saved["net"] or not saved["best_net"]:
        raise ValueError("Incomplete production checkpoint")
    train = ids_from(saved, "train_ids", 8856)
    valid = ids_from(saved, "validation_ids", 984)
    if not np.array_equal(np.sort(np.concatenate((train, valid))), np.arange(9840)):
        raise ValueError("Train/validation IDs must partition canonical9840 exactly")
    if saved["split_sha256"] != split_sha(train, valid):
        raise ValueError("Checkpoint split checksum differs from its own IDs")
    states = saved["rank_states"]
    if len(states) != 2 or [row.get("rank") for row in states] != [0, 1]:
        raise ValueError("Require ordered complete rank0/rank1 states")
    for row in states:
        if not {"BN_buffers", "rng", "sampler"}.issubset(row):
            raise ValueError("Incomplete rank state")
        if not {"python", "numpy", "torch", "cuda"}.issubset(row["rng"]):
            raise ValueError("Incomplete saved RNG")
        if row["sampler"].get("epoch") != epoch - 1:
            raise ValueError("Saved sampler epoch mismatch")
    schedule = saved["scheduler"]
    if (schedule.get("last_epoch"), schedule.get("T_max"), schedule.get("eta_min")) != (epoch, 300, .005):
        raise ValueError("Model scheduler must retain the original300-epoch cosine")
    expected_lr = .005 + (.1 - .005) * (1 + math.cos(math.pi * epoch / 300)) / 2
    optimizer = saved["optimizer"]
    if not optimizer.get("state") or not optimizer.get("param_groups"):
        raise ValueError("Missing complete optimizer and momentum state")
    for group in optimizer["param_groups"]:
        if abs(group.get("lr", -1) - expected_lr) > 1e-12 or group.get("momentum") != .9 or group.get("weight_decay") != 2e-4:
            raise ValueError("Model optimizer protocol mismatch")
    if not isinstance(saved["best"], dict) or not 1 <= saved["best"].get("epoch", 0) <= epoch:
        raise ValueError("Missing source validation-selection history")
    return train, valid


def validate_triplet(h20, b0, prefix, prefix_digest):
    """The two e40 models differ; match protocol/lineage, never state values."""
    train, valid = validate_complete(h20, H20_FORMAT, 40)
    bt, bv = validate_complete(b0, B0_FORMAT, 40)
    pt, pv = validate_complete(prefix, H20_FORMAT, 20)
    for at, av in ((bt, bv), (pt, pv)):
        if not np.array_equal(train, at) or not np.array_equal(valid, av):
            raise ValueError("Historical prefix/H20/B0 train/validation IDs differ")
    reference = h20["training_config"]
    for key, expected in EXPECTED_SOURCE.items():
        if reference.get(key) != expected:
            raise ValueError("Unsupported H20 source protocol: " + key)
    if prefix["training_config"] != reference:
        raise ValueError("H20e40 and common e20 source configs differ")
    config = derive_b0_config(reference)
    if b0["training_config"] != config:
        raise ValueError("B0 differs from the explicit source objective whitelist")
    validate_checkpoint_contract(prefix, config, reference, h20["split_sha256"], train.tolist(), valid.tolist())
    validate_checkpoint_contract(b0, config, reference, h20["split_sha256"], train.tolist(), valid.tolist())
    initialization = h20["initialized_model_sha256"]
    if prefix["initialized_model_sha256"] != initialization or b0["initialized_model_sha256"] != initialization:
        raise ValueError("Random initialization identity differs between arms")
    if h20["initialization"] != prefix["initialization"]:
        raise ValueError("Historical H20 initialization provenance differs from its prefix")
    if prefix["initialization"] != {"method": "random", "seed": 22, "source_checkpoint": None, "teacher": None}:
        raise ValueError("Historical source must retain random initialization provenance")
    b0_initial = b0["initialization"]
    if (b0_initial.get("method"), b0_initial.get("seed"), b0_initial.get("teacher")) != ("complete V6-H20 e20 fork", 22, None) or not b0_initial.get("source_checkpoint"):
        raise ValueError("B0 must retain complete e20 fork provenance")
    if h20["commit"] != prefix["commit"]:
        raise ValueError("Historical H20 source commit differs from its e20 prefix")
    fork = b0["fork_source"]
    expected_fork = {"sha256": prefix_digest, "source_commit": prefix["commit"],
                     "completed_epochs": 20, "source_initialized_model_sha256": initialization}
    if any(fork.get(key) != value for key, value in expected_fork.items()) or not fork.get("path"):
        raise ValueError("B0 does not fork the explicitly supplied immutable e20 checkpoint")
    if b0_initial["source_checkpoint"] != fork["path"]:
        raise ValueError("B0 initialization and fork provenance paths disagree internally")
    return train, valid, {"same_training_ids": True, "same_validation_ids": True,
        "same_split_sha256": h20["split_sha256"], "same_initialization_sha256": initialization,
        "shared_e20_checkpoint_sha256": prefix_digest, "source_configuration": reference,
        "b0_configuration": config, "b0_fork_source": fork,
        "configuration_changes": {key: {"H20": reference.get(key), "B0": config[key]}
                                  for key in config if reference.get(key) != config[key]},
        "bitwise_paired_training_claim": False,
        "interpretation": "Historical same-protocol e20 continuations; HIER disabled aftere20 is the intended objective difference. Historical H20 is not a promised bitwise replay."}


def source_split(labels):
    # Exact pure-NumPy implementation from hier_proxy_scratch_v4.train.
    rng = np.random.default_rng(22 + 1701)
    train, valid = [], []
    for label in np.unique(labels):
        ids = rng.permutation(np.flatnonzero(labels == label))
        count = max(1, int(round(len(ids) * .1)))
        if len(ids) - count < 8:
            raise ValueError("Insufficient source class size")
        valid.extend(ids[:count].tolist())
        train.extend(ids[count:].tolist())
    return np.asarray(sorted(train), dtype=np.int64), np.asarray(sorted(valid), dtype=np.int64)


def load_training(data_dir, train, valid):
    import h5py
    paths = sorted(Path(data_dir).resolve(strict=True).glob("ply_data_train*.h5"))
    if not paths:
        raise FileNotFoundError("No canonical training HDF5 shards")
    points, labels, shard_names, shard_rows, info = [], [], [], [], []
    for path in paths:
        before = digest(path)
        with h5py.File(path, "r") as shard:
            if shard["data"].ndim != 3 or shard["data"].shape[1] < 1024 or shard["data"].shape[2] != 3:
                raise ValueError("Require source XYZ with at least1024 points")
            cloud = shard["data"][:, :1024, :].astype(np.float32)
            label = shard["label"][:].astype(np.int64).reshape(-1)
        if cloud.shape != (len(label), 1024, 3) or digest(path) != before:
            raise ValueError("HDF5 shard changed during load or shape mismatch")
        points.append(cloud); labels.append(label)
        shard_names.extend([path.name] * len(label)); shard_rows.extend(range(len(label)))
        info.append({"name": path.name, "path": str(path), "sha256": before,
                     "count": int(len(label)), "bytes": path.stat().st_size})
    points, labels = np.concatenate(points), np.concatenate(labels)
    if points.shape != (9840, 1024, 3) or not np.isfinite(points).all() or not np.array_equal(np.unique(labels), np.arange(40)):
        raise ValueError("Require canonical ModelNet40 training9840 and all40 labels")
    expected_train, expected_valid = source_split(labels)
    if not np.array_equal(train, expected_train) or not np.array_equal(valid, expected_valid):
        raise ValueError("Checkpoint IDs do not match source seed22 stratified split in sorted shards")
    selected = np.ascontiguousarray(points[train])
    gold = labels[train]
    names = np.asarray(shard_names)[train]
    rows = np.asarray(shard_rows, dtype=np.int64)[train]
    identity = {"training_shards_sorted": info, "canonical_id": "offset in lexicographically sorted ply_data_train*.h5; also shard name and row",
                "full_training_count": 9840, "forwarded_train_id_count": 8856,
                "whole_points": 1024, "train_ids_sha256": array_hash(train),
                "labels_sha256": array_hash(gold), "whole_clean1024_sha256": array_hash(selected),
                "full_training_labels_sha256": array_hash(labels),
                "test_files_read": False, "validation_forwarded": False}
    return selected, gold, names, rows, identity


def describe(values):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite summary values")
    if not len(values):
        return {"count": 0, "mean": None, "std": None, "p01": None, "p10": None,
                "median": None, "p90": None, "p99": None}
    quantiles = np.quantile(values, [.01, .1, .5, .9, .99])
    return {"count": int(len(values)), "mean": float(values.mean()), "std": float(values.std()),
            **dict(zip(("p01", "p10", "median", "p90", "p99"), map(float, quantiles)))}


def geometry(mu):
    mu = np.asarray(mu, dtype=np.float64)
    radius = np.linalg.norm(mu, axis=1)
    if mu.ndim != 2 or not np.isfinite(mu).all() or np.any(radius >= 1):
        raise ValueError("Require finite raw c1 ball embeddings; no diagnostic projection")
    defined = radius > 1e-15
    directions = np.divide(mu, radius[:, None], out=np.zeros_like(mu), where=defined[:, None])
    return mu, radius, 2 * np.arctanh(radius), directions, defined


def all_unordered_pairs(x, directions, defined):
    """Direct FP64 differences in blocks, avoiding dot-product cancellation."""
    n = len(x)
    hyp, angles = [], []
    squared = np.einsum("ij,ij->i", x, x)
    for start in range(0, n, 16):
        stop = min(start + 16, n)
        rows = np.arange(start, stop)[:, None]
        columns = np.arange(n)[None, :]
        upper = columns > rows
        difference = x[start:stop, None, :] - x[None, :, :]
        euclid = np.linalg.norm(difference, axis=-1)
        denominator = np.sqrt((1 - squared[start:stop, None]) * (1 - squared[None, :]))
        hyp.append((2 * np.arcsinh(euclid / denominator))[upper])
        cosine = np.clip(directions[start:stop] @ directions.T, -1., 1.)
        angle_defined = upper & defined[start:stop, None] & defined[None, :]
        angles.append(np.degrees(np.arccos(cosine))[angle_defined])
    return (np.concatenate(hyp) if hyp else np.empty(0),
            np.concatenate(angles) if angles else np.empty(0))


def summarize(cache):
    x, radius, depth, directions, defined = geometry(cache["mu"])
    labels, pred, ce = cache["labels"], cache["whole_prediction"], cache["whole_ce"]
    classes, rows, pair_hyp, pair_angles = [], [], [], []
    for cls, name in enumerate(NAMES):
        at = np.flatnonzero(labels == cls)
        hyp, angle = all_unordered_pairs(x[at], directions[at], defined[at])
        pair_hyp.append(hyp); pair_angles.append(angle)
        valid_u = directions[at][defined[at]]
        resultant = float(np.linalg.norm(valid_u.mean(0))) if len(valid_u) else None
        item = {"class_id": cls, "class_name": name, "object_count": int(len(at)),
                "unordered_pair_count": int(len(at) * (len(at) - 1) // 2),
                "angular_pair_count": int(len(angle)), "undefined_direction_count": int((~defined[at]).sum()),
                "correct": int((pred[at] == cls).sum()),
                "accuracy_percent": float(100 * np.mean(pred[at] == cls)) if len(at) else None,
                "smoothed_ce": float(ce[at].mean()) if len(at) else None,
                "radius": describe(radius[at]), "depth": describe(depth[at]),
                "within_class_angle_degrees": describe(angle),
                "within_class_hyperbolic_distance": describe(hyp),
                "direction_resultant_length": resultant}
        classes.append(item)
        row = {key: value for key, value in item.items() if not isinstance(value, dict)}
        for metric in ("radius", "depth", "within_class_angle_degrees", "within_class_hyperbolic_distance"):
            row.update({metric + "_" + key: value for key, value in item[metric].items()})
        rows.append(row)
    angles, hyp = np.concatenate(pair_angles), np.concatenate(pair_hyp)
    summary = {"scope": "checkpoint train_ids8856 clean1024 eval32 whole only", "object_count": len(labels),
               "correct": int((pred == labels).sum()), "oa_percent": float(100 * np.mean(pred == labels)),
               "aa_percent": float(np.mean([row["accuracy_percent"] for row in classes])),
               "smoothed_ce": float(ce.mean()), "radius": describe(radius), "depth": describe(depth),
               "pooled_within_class_angle_degrees": describe(angles),
               "pooled_within_class_hyperbolic_distance": describe(hyp),
               "equal_class_mean_angle_degrees": float(np.mean([row["within_class_angle_degrees"]["mean"] for row in classes])),
               "equal_class_mean_hyperbolic_distance": float(np.mean([row["within_class_hyperbolic_distance"]["mean"] for row in classes])),
               "equal_class_mean_depth": float(np.mean([row["depth"]["mean"] for row in classes])),
               "classes": classes,
               "geometry": {"curvature": 1., "precision": "FP64 from unchanged FP32 mu",
                            "pairs": "every unordered within-class pair once; self-pairs excluded; no downsampling",
                            "pooled_pair_weight": "class weight n_c(n_c-1)/2; differs from equal-class mean",
                            "depth_denominator": "one observation per training object",
                            "origin_dependence": "depth and directional angle use each model's learned c1 origin; rotationally invariant, not invariant to all origin-changing isometries",
                            "cross_model_coordinates": "No coordinate subtraction, alignment or cross-model proxy retrieval"}}
    return summary, rows, radius, depth


def comparison(h20, b0, hc, bc):
    rows = []
    for a, b in zip(h20["classes"], b0["classes"]):
        row = {"class_id": a["class_id"], "class_name": a["class_name"], "object_count": a["object_count"],
               "h20_accuracy_percent": a["accuracy_percent"], "b0_accuracy_percent": b["accuracy_percent"],
               "delta_h20_minus_b0_accuracy_points": a["accuracy_percent"] - b["accuracy_percent"]}
        for metric in ("depth", "radius", "within_class_angle_degrees", "within_class_hyperbolic_distance"):
            for statistic in ("mean", "median"):
                row[f"h20_{metric}_{statistic}"] = a[metric][statistic]
                row[f"b0_{metric}_{statistic}"] = b[metric][statistic]
                row[f"delta_h20_minus_b0_{metric}_{statistic}"] = a[metric][statistic] - b[metric][statistic]
        rows.append(row)
    hcorrect = hc["whole_prediction"] == hc["labels"]
    bcorrect = bc["whole_prediction"] == bc["labels"]
    return {"checkpoint_epoch": 40, "object_count": 8856, "H20_oa_percent": h20["oa_percent"],
            "B0_oa_percent": b0["oa_percent"], "delta_h20_minus_b0_oa_points": h20["oa_percent"] - b0["oa_percent"],
            "prediction_disagreement_percent": float(100 * np.mean(hc["whole_prediction"] != bc["whole_prediction"])),
            "both_correct": int((hcorrect & bcorrect).sum()), "only_H20_correct": int((hcorrect & ~bcorrect).sum()),
            "only_B0_correct": int((~hcorrect & bcorrect).sum()), "both_wrong": int((~hcorrect & ~bcorrect).sum()),
            "classes": rows, "interpretation": "Train classification and within-class geometric differences after20 matched continuation epochs; not test generalization, not independent morphology validation, and not a bitwise paired training experiment."}, rows


def check_named(path, epoch):
    path = Path(path).resolve(strict=True)
    if path.name != f"checkpoint_epoch_{epoch:03d}.pth":
        raise ValueError(f"Require immutable checkpoint_epoch_{epoch:03d}.pth; never last/best")
    return path


def export_one(torch, Hype_pointMLP, saved, arm, output, points, labels, ids, names, shard_rows, common, device):
    output.mkdir(exist_ok=False)
    manifest = {**common, "status": "running", "arm": arm, "started_utc": utc_now(),
                "checkpoint_format": saved["format"], "checkpoint_epoch": saved["epoch"],
                "checkpoint_source_commit": saved["commit"], "training_config": saved["training_config"],
                "optimizer_updates": 0, "validation_forwarded": False, "official_test_read": False,
                "proxy_loaded_to_gpu": False, "batch_size": 32, "input_mode": "clean first1024",
                "native_fps": "unchanged repository CUDA FPS; first index0; no stochastic augmentation",
                "geometry_pairs": "exact all unordered within-class pairs in CPU FP64"}
    write_json(output / "manifest.json", manifest)
    model = Hype_pointMLP().to(device)
    model.load_state_dict(saved["net"], strict=True)
    for name in ("manifold", "manifold2"):
        manifold = getattr(model, name)
        if not math.isclose(float(manifold.c.detach()), 1., abs_tol=1e-5):
            raise ValueError("Checkpoint curvature differs from c1: " + name)
        manifold.requires_grad_(False)
    model.requires_grad_(False); model.eval()
    before = state_hash(model.state_dict())
    buffers = {name: value.detach().cpu().clone() for name, value in model.named_buffers()}
    if before != state_hash(saved["net"]):
        raise RuntimeError("Loaded backbone differs from complete checkpoint net")
    rank0_buffers = saved["rank_states"][0]["BN_buffers"]
    if buffers.keys() != rank0_buffers.keys() or any(not torch.equal(value, rank0_buffers[name].detach().cpu()) for name, value in buffers.items()):
        raise RuntimeError("Checkpoint net does not contain its own complete rank0 BN/buffer state")
    count = len(ids)
    mu = np.empty((count, 256), dtype=np.float32)
    logits = np.empty((count, 40), dtype=np.float32)
    ce = np.empty(count, dtype=np.float64)
    input_before = array_hash(points)
    started = time.monotonic()
    with torch.no_grad():
        for start in range(0, count, 32):
            stop = min(start + 32, count)
            cloud = torch.from_numpy(points[start:stop].copy()).to(device)
            whole, logit = model(cloud.transpose(1, 2).contiguous())
            if whole.shape != (stop - start, 256) or logit.shape != (stop - start, 40) or not torch.isfinite(whole).all() or not torch.isfinite(logit).all():
                raise RuntimeError("Invalid frozen eval output")
            mu[start:stop] = whole.cpu().numpy(); logits[start:stop] = logit.cpu().numpy()
            logp = torch.log_softmax(logit, -1)
            gold = torch.as_tensor(labels[start:stop], device=device)
            correct_logp = logp.gather(1, gold[:, None]).squeeze(1)
            ce[start:stop] = (-.8 * correct_logp - (.2 / 39) * (logp.sum(1) - correct_logp)).cpu().double().numpy()
            if start == 0 or stop == count or (start // 32 + 1) % 50 == 0:
                print(json.dumps({"arm": arm, "forwarded": stop, "total": count}), flush=True)
    if state_hash(model.state_dict()) != before or any(not torch.equal(buffers[name], value.detach().cpu()) for name, value in model.named_buffers()):
        raise RuntimeError("Frozen eval changed model/BN state")
    if any(parameter.grad is not None for parameter in model.parameters()) or array_hash(points) != input_before:
        raise RuntimeError("Backward or input mutation detected")
    cache = {"sample_ids": ids, "ids": ids, "labels": labels, "mu": mu, "whole_mu": mu,
             "logits": logits, "whole_prediction": logits.argmax(1), "whole_ce": ce,
             "canonical_shard_names": names, "canonical_shard_rows": shard_rows}
    summary, rows, radius, depth = summarize(cache)
    cache.update(radius=radius, depth=depth)
    cache_path = output / "whole_cache.npz"
    np.savez_compressed(cache_path, **cache)
    write_csv(output / "classes.csv", rows)
    summary.update(checkpoint=common["checkpoint"], data_identity=common["data_identity"],
                   arm=arm, checkpoint_format=saved["format"], checkpoint_epoch=40,
                   backbone_state_sha256=before, inference_seconds=time.monotonic() - started)
    write_json(output / "summary.json", summary)
    manifest.update(status="completed", finished_utc=utc_now(), wall_seconds=time.monotonic() - started,
                    backbone_initial_sha256=before, backbone_final_sha256=state_hash(model.state_dict()),
                    bn_and_all_buffers_unchanged=True, weights_unchanged=True, input_unchanged=True,
                    gradients_absent=True, cache_sha256=digest(cache_path),
                    cache_mu_sha256=array_hash(mu), logits_sha256=array_hash(logits),
                    oa_percent=summary["oa_percent"], aa_percent=summary["aa_percent"])
    write_json(output / "manifest.json", manifest)
    del model
    torch.cuda.empty_cache()
    return cache, summary


def self_test():
    train, valid = np.arange(8856), np.arange(8856, 9840)
    reference = {**EXPECTED_SOURCE, "audit_epochs": [40]}
    def synthetic(epoch):
        lr = .005 + (.1 - .005) * (1 + math.cos(math.pi * epoch / 300)) / 2
        return {"format": H20_FORMAT, "epoch": epoch, "completed_epochs": epoch,
                "model_selection_only": False, "net": {"w": 1}, "proxy": {"p": 1}, "best_net": {"w": 1},
                "optimizer": {"state": {0: {"momentum_buffer": 1}}, "param_groups": [{"lr": lr, "momentum": .9, "weight_decay": 2e-4}]},
                "proxy_optimizer": {}, "proxy_scheduler": {},
                "scheduler": {"last_epoch": epoch, "T_max": 300, "eta_min": .005},
                "rank_states": [{"rank": rank, "BN_buffers": {}, "rng": dict.fromkeys(("python", "numpy", "torch", "cuda")), "sampler": {"epoch": epoch - 1}} for rank in range(2)],
                "best": {"epoch": 10}, "initialization": {"method": "random", "seed": 22, "source_checkpoint": None, "teacher": None}, "initialized_model_sha256": "init",
                "commit": "source", "training_config": reference, "train_ids": train,
                "validation_ids": valid, "split_sha256": split_sha(train, valid)}
    h20, prefix = synthetic(40), synthetic(20)
    b0 = copy.deepcopy(h20)
    b0["format"] = B0_FORMAT; b0["training_config"] = derive_b0_config(reference)
    b0["initialization"] = {"method": "complete V6-H20 e20 fork", "seed": 22, "source_checkpoint": "prefix/checkpoint_epoch_020.pth", "teacher": None}
    for key in ("proxy", "proxy_optimizer", "proxy_scheduler"):
        del b0[key]
    b0["fork_source"] = {"sha256": "prefix", "path": "prefix/checkpoint_epoch_020.pth", "source_commit": "source", "completed_epochs": 20, "source_initialized_model_sha256": "init"}
    validate_triplet(h20, b0, prefix, "prefix")
    for key, value in (("epoch", 39), ("format", H20_FORMAT), ("diagnostic_only", True), ("initialized_model_sha256", "other")):
        wrong = copy.deepcopy(b0); wrong[key] = value
        try:
            validate_triplet(h20, wrong, prefix, "prefix")
        except (ValueError, RuntimeError):
            pass
        else:
            raise AssertionError("Identity gate missed " + key)
    for key, value in (("smoke", True), ("self_negative", True), ("c", .1), ("global_batch", 32)):
        wrong = copy.deepcopy(h20); wrong["training_config"][key] = value
        try:
            validate_triplet(wrong, b0, prefix, "prefix")
        except (ValueError, RuntimeError):
            pass
        else:
            raise AssertionError("Protocol gate missed " + key)
    wrong = copy.deepcopy(b0); wrong["fork_source"]["sha256"] = "different"
    try:
        validate_triplet(h20, wrong, prefix, "prefix")
    except ValueError:
        pass
    else:
        raise AssertionError("Shared prefix checksum gate missed")
    x, r, d, u, defined = geometry(np.array([[.2, 0], [.1, 0], [0, .3]], dtype=np.float32))
    hyp, angle = all_unordered_pairs(x, u, defined)
    assert len(hyp) == 3 and len(angle) == 3
    np.testing.assert_allclose(angle, [0., 90., 90.], atol=1e-8)
    np.testing.assert_allclose(hyp[0], abs(d[0] - d[1]), rtol=1e-12)
    rotation = np.array([[0., -1.], [1., 0.]])
    xr, rr, dr, ur, ddef = geometry(x @ rotation)
    hp, ap = all_unordered_pairs(xr, ur, ddef)
    np.testing.assert_allclose(hp, hyp); np.testing.assert_allclose(ap, angle)
    _, _, _, zu, zd = geometry(np.array([[0., 0.], [.1, 0.]]))
    zh, za = all_unordered_pairs(np.array([[0., 0.], [.1, 0.]]), zu, zd)
    assert len(zh) == 1 and len(za) == 0
    assert describe(np.empty(0))["median"] is None
    print(json.dumps({"self_test": "passed", "checks": ["complete e40 formats and objective whitelist", "shared immutable e20 lineage", "epoch/protocol/smoke/initialization rejection", "all unordered pairs once", "FP64 direct c1 radial distance", "rotation-invariant statistics", "undefined direction excluded from angles"]}))


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h20-checkpoint", type=Path)
    parser.add_argument("--b0-checkpoint", type=Path)
    parser.add_argument("--source-e20-checkpoint", type=Path)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--gpu", type=int, default=0, help="Logical index; caller checks the physical idle GPU before launch")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if not args.self_test and not all((args.h20_checkpoint, args.b0_checkpoint, args.source_e20_checkpoint, args.data_dir, args.run_dir)):
        parser.error("Two e40 checkpoints, shared e20 checkpoint, data-dir and new run-dir required")
    if args.gpu < 0:
        parser.error("GPU logical index must be nonnegative")
    return args


def main():
    args = arguments()
    if args.self_test:
        self_test(); return
    paths = {"H20": check_named(args.h20_checkpoint, 40), "B0": check_named(args.b0_checkpoint, 40),
             "shared_e20": check_named(args.source_e20_checkpoint, 20)}
    if len(set(paths.values())) != 3:
        raise ValueError("Require three distinct immutable checkpoint files")
    args.run_dir.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    source_hashes = {key: digest(path) for key, path in paths.items()}
    source_files = [__file__, ROOT / "inter_hierarchy_MN40/v6_balanced_b0/contract.py",
                    ROOT / "inter_hierarchy_MN40/models/pointmlp.py",
                    ROOT / "inter_hierarchy_MN40/models/ops/nn.py",
                    ROOT / "pointnet2_ops_lib/pointnet2_ops/_ext-src/src/sampling_gpu.cu"]
    manifest = {"format": "v6-b0-matched-whole-export-1", "status": "starting", "started_utc": utc_now(),
                "diagnostic_only": True, "main_training": False, "optimizer_updates": 0,
                "command_argv": [sys.executable, *sys.argv], "checkpoint_epoch": 40,
                "source_checkpoints": {key: {"path": str(path), "sha256": source_hashes[key], "bytes": path.stat().st_size} for key, path in paths.items()},
                "source_manifest": [{"path": str(path), "sha256": digest(path)} for path in source_files],
                "official_test_read": False, "validation_forwarded": False,
                "inference_batch_size": 32, "input_mode": "clean1024; unchanged source ordering",
                "proxy_cross_coordinate_retrieval": False}
    for key, command in (("commit", ["git", "rev-parse", "HEAD"]), ("branch", ["git", "branch", "--show-current"])):
        manifest[key] = subprocess.check_output(command, cwd=ROOT, text=True).strip()
    write_json(args.run_dir / "manifest.json", manifest)
    try:
        import torch
        torch.set_num_threads(2)
        saved = {key: torch.load(path, map_location="cpu", weights_only=False) for key, path in paths.items()}
        train, valid, gate = validate_triplet(saved["H20"], saved["B0"], saved["shared_e20"], source_hashes["shared_e20"])
        manifest["identity_gate"] = gate
        points, labels, names, shard_rows, data_identity = load_training(args.data_dir, train, valid)
        manifest["data_identity"] = data_identity
        sys.path.insert(0, str(ROOT / "pointnet2_ops_lib"))
        sys.path.insert(0, str(ROOT / "inter_hierarchy_MN40"))
        from models.pointmlp import Hype_pointMLP
        if not torch.cuda.is_available():
            raise ValueError("Source PointMLP native FPS requires CUDA")
        device = torch.device("cuda", args.gpu)
        torch.cuda.set_device(device)
        torch.random.default_generator.manual_seed(22)
        torch.cuda.default_generators[device.index].manual_seed(22)
        torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
        torch.cuda.reset_peak_memory_stats(device)
        manifest.update(status="running", runtime={"torch": torch.__version__, "numpy": np.__version__,
                        "cuda": torch.version.cuda, "device": str(device), "gpu_name": torch.cuda.get_device_name(device),
                        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                        "cpu_threads": 2})
        write_json(args.run_dir / "manifest.json", manifest)
        caches, summaries = {}, {}
        for arm in ("H20", "B0"):
            common = {"checkpoint": manifest["source_checkpoints"][arm], "data_identity": data_identity,
                      "commit": manifest["commit"], "branch": manifest["branch"],
                      "source_manifest": manifest["source_manifest"], "runtime": manifest["runtime"],
                      "identity_gate": gate, "diagnostic_only": True}
            caches[arm], summaries[arm] = export_one(torch, Hype_pointMLP, saved[arm], arm, args.run_dir / arm,
                                                    points, labels, train, names, shard_rows, common, device)
        paired, rows = comparison(summaries["H20"], summaries["B0"], caches["H20"], caches["B0"])
        write_csv(args.run_dir / "matched_classes.csv", rows)
        write_json(args.run_dir / "matched_summary.json", paired)
        if any(digest(paths[key]) != value for key, value in source_hashes.items()):
            raise RuntimeError("An immutable source checkpoint changed during export")
        if any(digest(row["path"]) != row["sha256"] for row in data_identity["training_shards_sorted"]):
            raise RuntimeError("A training HDF5 shard changed during export")
        if any(digest(row["path"]) != row["sha256"] for row in manifest["source_manifest"]):
            raise RuntimeError("Source code changed during export")
        manifest.update(status="completed", finished_utc=utc_now(), wall_seconds=time.monotonic() - started,
                        checkpoint_files_unchanged=True, hdf5_files_unchanged=True, source_files_unchanged=True,
                        matched_summary=paired, peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device),
                        private_artifacts=["H20/whole_cache.npz", "B0/whole_cache.npz"],
                        report_files=["matched_summary.json", "matched_classes.csv", "H20/classes.csv", "B0/classes.csv"])
        write_json(args.run_dir / "manifest.json", manifest)
        print(json.dumps({"status": "completed", "wall_seconds": manifest["wall_seconds"],
                          "H20_oa": paired["H20_oa_percent"], "B0_oa": paired["B0_oa_percent"]}), flush=True)
    except BaseException as exc:
        manifest.update(status="failed", finished_utc=utc_now(), wall_seconds=time.monotonic() - started,
                        error=f"{type(exc).__name__}: {exc}")
        write_json(args.run_dir / "manifest.json", manifest)
        raise


if __name__ == "__main__":
    main()
