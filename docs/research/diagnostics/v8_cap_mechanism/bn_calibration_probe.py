"""Controlled temporary BN-bank calibration using disjoint training objects.

No parameter, optimizer or test updates. Fresh cumulative BN banks include
whole-only clean, matched whole->whole alias, part->whole and augmentation.
All evaluate exactly the same held-out training objects in direct clean/eval.
This probes BN-bank compatibility, not whether historical joint BN was causal.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time

import h5py
import numpy as np
import torch

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from whole_condition_probe import (  # noqa: E402
    Hype_pointMLP, configure_backbone, depth, get_children_alias,
    sha, state_hash, stats, write_json, capture_rng, restore_rng, rng_equal,
    select_objects, prepare_clouds, nearest_indices,
)


def parameter_fingerprint(model):
    import hashlib
    digest = hashlib.sha256()
    for name, value in sorted(model.named_parameters()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def eval_whole(model, clouds, device, microbatch=32):
    raw, output = [], []

    def hook(module, _inputs, raw_ball):
        raw.append(module.manifold.logmap0(raw_ball).detach().float().cpu())

    handle = model.emb.register_forward_hook(hook, prepend=True)
    model.eval()
    try:
        with torch.no_grad():
            for start in range(0, len(clouds), microbatch):
                points = torch.from_numpy(clouds[start:start + microbatch].copy()).to(device).transpose(1, 2).contiguous()
                mu, _ = model(points, emb=True)
                output.append(mu.detach().float().cpu())
    finally:
        handle.remove()
    tangent = torch.cat(raw)
    mu = torch.cat(output)
    norms, depths = tangent.norm(dim=-1), depth(mu)
    return {"raw_precap_tangent_norm": stats(norms.numpy()), "depth": stats(depths.numpy()),
            "precap_over_2_3_fraction": float((norms > 2.3).float().mean()),
            "cap_depth_fraction": float((depths >= 4.5999).float().mean())}, {
                "raw_precap_tangent_norm": norms.numpy(), "depth": depths.numpy()}


def calibrate(model, points, whole_targets, child_targets, mode, device, microbatch=32):
    model.eval()
    layers = [m for m in model.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
    for module in layers:
        if not module.track_running_stats:
            raise ValueError("Calibration requires tracked BN running statistics")
        module.reset_running_stats()
        module.momentum = None
        module.train(True)
    forward_calls = 0
    with torch.no_grad():
        for start in range(0, len(points), microbatch):
            stop = start + microbatch
            source = torch.from_numpy(points[start:stop].copy()).to(device).transpose(1, 2).contiguous()
            if mode == "whole_only_clean":
                model(source, emb=True)
                forward_calls += 1
            else:
                wtarget = torch.from_numpy(whole_targets[start:stop]).to(device)
                ctarget = torch.from_numpy(child_targets[start:stop]).to(device)
                _, whole, _ = get_children_alias(source, 1024, centers=nearest_indices(source, wtarget))
                _, part, _ = get_children_alias(whole, 400, centers=nearest_indices(whole, ctarget))
                model(whole if mode == "whole_twice_alias_clean" else part, emb=True)
                model(whole, emb=True)
                forward_calls += 2
    return {"BN_layers": len(layers), "local32_forward_calls": forward_calls,
            "num_batches_tracked_unique": sorted({int(m.num_batches_tracked) for m in layers}),
            "running_mean_abs_mean": float(np.mean([float(m.running_mean.abs().mean()) for m in layers])),
            "running_var_mean": float(np.mean([float(m.running_var.mean()) for m in layers])),
            "momentum": None, "running_stats_reset": True,
            "affine_parameters_unchanged": True, "dropout_during_calibration": "eval"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--fixed-ids", required=True, help="Factorial private NPZ sample_ids, or NPY")
    p.add_argument("--label", default="V8_B64_e300_BN_calibration")
    p.add_argument("--calibration-objects", type=int, choices=(640, 1024), default=640)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--microbatch", type=int, default=32)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--seed", type=int, default=22)
    p.add_argument("--banks", default="whole_only_clean,whole_twice_alias_clean,joint_alias_clean,joint_alias_original_aug")
    args = p.parse_args()
    banks = args.banks.split(",")
    if not banks or set(banks) - {"whole_only_clean", "whole_twice_alias_clean", "joint_alias_clean", "joint_alias_original_aug"}:
        p.error("Unknown BN-bank condition")
    if args.microbatch != 32:
        p.error("Use local32 for calibration/evaluation")
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=False)
    started = time.time()
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    hashes = {"checkpoint": sha(args.checkpoint), "fixed_ids": sha(args.fixed_ids)}
    cp = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = configure_backbone(Hype_pointMLP())
    net = {k[7:] if k.startswith("module.") else k: v for k, v in cp["net"].items()}
    model.load_state_dict(net, strict=True)
    model.to(device).eval().requires_grad_(False)
    original_hash, param_hash = state_hash(model), parameter_fingerprint(model)
    original_buffers = {name: value.detach().clone() for name, value in model.named_buffers()}
    original_modes = {name: module.training for name, module in model.named_modules()}
    original_momenta = {name: module.momentum for name, module in model.named_modules()
                        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)}
    rng_before = capture_rng(device)
    data, labels, shard_hashes = [], [], {}
    for path in sorted(Path(args.data_dir).glob("ply_data_train*.h5")):
        with h5py.File(path, "r") as handle:
            data.append(np.asarray(handle["data"], dtype=np.float32)[:, :1024])
            labels.append(np.asarray(handle["label"]).reshape(-1).astype(np.int64))
        shard_hashes[path.name] = sha(path)
    data, labels = np.concatenate(data), np.concatenate(labels)
    if len(data) != 9840:
        raise AssertionError("Expected official9840 training source, never test")
    allowed = np.asarray(cp.get("train_ids", np.arange(9840)), dtype=np.int64)
    loaded = np.load(args.fixed_ids, allow_pickle=False)
    if isinstance(loaded, np.lib.npyio.NpzFile):
        with loaded:
            holdout = loaded["sample_ids"].copy().astype(np.int64)
    else:
        holdout = np.asarray(loaded, dtype=np.int64)
    if (holdout.ndim != 1 or len(holdout) not in (128, 256) or len(np.unique(holdout)) != len(holdout)
            or not np.isin(holdout, allowed).all()):
        raise ValueError("Holdout must contain128/256 unique members of source training pool")
    calibration_pool = allowed[~np.isin(allowed, holdout)]
    calibration_ids = select_objects(labels, calibration_pool, args.calibration_objects // 64, args.seed + 10007)
    if np.intersect1d(calibration_ids, holdout).size:
        raise AssertionError("Calibration and measurement objects overlap")
    center_rng = np.random.RandomState(args.seed + 1701)
    wc = center_rng.randint(0, 1024, size=len(calibration_ids))
    cc = center_rng.randint(0, 1024, size=len(calibration_ids))
    baseline, baseline_arrays = eval_whole(model, data[holdout], device, args.microbatch)
    if state_hash(model) != original_hash:
        raise AssertionError("Baseline evaluation changed source state")
    write_json(output / "progress.json", {"phase": "baseline_complete", "baseline": baseline})
    rows = []

    def restore_source():
        with torch.no_grad():
            for name, buffer in model.named_buffers():
                buffer.copy_(original_buffers[name])
        for name, module in model.named_modules():
            module.training = original_modes[name]
            if name in original_momenta:
                module.momentum = original_momenta[name]
        restore_rng(rng_before, device)

    try:
        for mode in banks:
            restore_source()
            prepared = prepare_clouds(data[calibration_ids], wc, cc,
                                      mode == "joint_alias_original_aug", args.seed + 3001)
            bank = calibrate(model, *prepared, mode, device, args.microbatch)
            if parameter_fingerprint(model) != param_hash:
                raise AssertionError("Calibration changed a trainable parameter")
            bank_hash = state_hash(model)
            calibrated_buffers = {name: value.detach().cpu().clone() for name, value in model.named_buffers()}
            measurement, arrays = eval_whole(model, data[holdout], device, args.microbatch)
            if state_hash(model) != bank_hash:
                raise AssertionError("Holdout evaluation changed the calibrated BN bank")
            row = {"mode": mode, "calibration_objects": len(calibration_ids),
                   "holdout_training_objects": len(holdout), "measurement": measurement,
                   "BN_bank": bank, "temporary_bank_fingerprint": bank_hash,
                   "parameter_fingerprint_unchanged": True,
                   "delta_mean_depth_from_source_bank": measurement["depth"]["mean"] - baseline["depth"]["mean"],
                   "delta_cap_percentage_points_from_source_bank": 100 * (measurement["cap_depth_fraction"] - baseline["cap_depth_fraction"]),
                   "calibration_holdout_intersection": 0, "optimizer_updates": 0, "backward_calls": 0,
                   "holdout_condition": "direct first1024, no augmentation, eval BN, whole-only forward"}
            write_json(output / (mode + ".json"), row)
            torch.save(calibrated_buffers, output / ("private_" + mode + "_BN_bank.pt"))
            np.savez_compressed(output / ("private_" + mode + ".npz"), sample_ids=holdout,
                                calibration_sample_ids=calibration_ids, baseline_depth=baseline_arrays["depth"], **arrays)
            rows.append(row)
            write_json(output / "progress.json", {"phase": "bank_complete", "mode": mode,
                                                  "completed": len(rows), "measurement": measurement})
            print(json.dumps({"mode": mode, "mean_depth": measurement["depth"]["mean"],
                              "cap_fraction": measurement["cap_depth_fraction"]}), flush=True)
    finally:
        restore_source()
    if (state_hash(model) != original_hash or parameter_fingerprint(model) != param_hash
            or not rng_equal(capture_rng(device), rng_before) or any(p.grad is not None for p in model.parameters())):
        raise AssertionError("Final source/RNG/gradient restoration failed")
    for key, value in hashes.items():
        if sha(getattr(args, key)) != value:
            raise AssertionError("An input file changed")
    summary = {"complete": True, "label": args.label, "source_epoch": cp.get("epoch"),
               "source_hashes": hashes, "training_shard_sha256": shard_hashes,
               "official_training_pool": 9840, "eligible_source_training_objects": len(allowed),
               "calibration_objects": len(calibration_ids), "holdout_training_objects": len(holdout),
               "calibration_holdout_intersection": 0, "source_bank_baseline": baseline, "banks": rows,
               "calibration_method": "Reset BN running stats; momentum=None cumulative updates; BN-only train; affine weights unchanged",
               "source_checkpoint_unchanged": True, "source_model_BN_RNG_restored": True,
               "optimizer_updates": 0, "backward_calls": 0, "test_forward_calls": 0,
               "device": str(device), "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
               "seconds": time.time() - started, "finished_utc": datetime.now(timezone.utc).isoformat(),
               "runner_sha256": sha(__file__),
               "limits": "Bounded temporary BN recalibration changes distribution/statistic estimator. Direct whole-only sees one pass; matched whole-twice-alias and joint banks see two passes with identical aliased whole geometry. This does not prove historical exclusivity or classification benefit."}
    write_json(output / "summary.json", summary)
    write_json(output / "progress.json", {"phase": "complete", "completed": len(rows)})


if __name__ == "__main__":
    main()
