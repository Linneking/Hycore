"""Collect bounded paired training views for V5 K diagnostics; no optimization.

The V4 epoch20 checkpoint is a representation reference only. This runner
does not initialize a V5 training run, evaluate validation/test data, or
write model weights. It preserves HyCoRe child overwriting, local32 BN,
fixed FPS requests, and c1 geometry while caching globally ordered64 rows.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

from .diagnose import (
    REPO, Hype_pointMLP, HyCoReTrainingForward, PointClouds,
    SourceClassBatchSampler, checkpoint_sha, freeze_curvature,
    gather_without_grad, load_shards, seed_all, shared_crop_counts,
    split_identity, stratified_split, write_json,
)
from .base_protocol import smoothed_ce

import geoopt
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


@torch.no_grad()
def broadcast_working_buffers(backbone):
    """Reproduce rank0 DDP buffers at every outer child/whole forward.

    An ordinary no-grad DDP forward may disable its next automatic buffer
    broadcast. Explicit broadcasts keep this diagnostic's train-mode BN
    protocol the same as the intended optimizing two-rank implementation.
    Child then whole update each rank's working buffers normally.
    """
    for buffer in backbone.buffers():
        dist.broadcast(buffer, src=0)


def classifier_mu_gradient(backbone, local_mu, local_labels):
    """Gradient of the global64 mean CE with respect to these local32 rows.

    The encoder remains under no_grad. Only a cloned embedding and the
    unchanged classifier are used in autograd.grad, so neither backbone
    nor classifier parameter gradients are accumulated. The local CE mean
    is divided by two; concatenating both local gradients gives the true
    global64-mean gradient, without differentiable-gather/DDP factors.
    """
    with torch.enable_grad():
        mu = local_mu.detach().clone().requires_grad_(True)
        logits = backbone.classifier(mu)
        local_ce = smoothed_ce(logits, local_labels)
        gradient = torch.autograd.grad(local_ce / dist.get_world_size(), mu)[0]
    if not torch.isfinite(gradient).all() or not torch.isfinite(local_ce):
        raise RuntimeError("Nonfinite frozen-classifier CE gradient")
    return gradient.detach(), local_ce.detach(), logits.detach()


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--batches", type=int, default=64)
    parser.add_argument("--views", type=int, default=2)
    parser.add_argument("--seed", type=int, default=22)
    args = parser.parse_args()
    if not 1 <= args.batches <= 64 or args.views != 2:
        parser.error("Bounded collection requires 1..64 batches and exactly two paired views")
    return args


def main():
    args = arguments()
    torch.set_num_threads(2)
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    if int(os.environ["WORLD_SIZE"]) != 2:
        raise RuntimeError("Exactly two ranks required: local32/global64")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", timeout=dt.timedelta(minutes=5))
    started_wall = time.perf_counter()
    if rank == 0:
        args.run_dir.mkdir(parents=True, exist_ok=False)
    dist.barrier()
    manifest = {
        "status": "running", "diagnostic_only": True, "main_training": False,
        "optimizer_steps": 0, "encoder_backward": False,
        "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        "branch": subprocess.check_output(
            ["git", "branch", "--show-current"], cwd=REPO, text=True).strip(),
        "command_argv": [sys.executable, *sys.argv],
        "config": {key: str(value) if isinstance(value, Path) else value
                   for key, value in vars(args).items()},
        "fixed": {
            "c": 1, "D": 256, "P": 512, "T": 50,
            "margin": .1, "tau": .1, "lambda_H_analysis": .5,
            "proxy_initialization_analysis": "random_tangent_seed22",
            "global_batch": 64, "local_encoder_BN": 32,
            "classes_per_rank": 16, "instances_per_class": 2,
            "nominal_steps": 200, "sampler_epoch": 0,
            "part_overwrites_whole": True, "original_fixed_FPS": True,
            "extra_cap": False, "HIER_backward_hook": False,
            "BN_mode": "train; rank0 buffers broadcast before each child/whole outer forward",
            "CE_gradient": "frozen classifier only; global64 mean dCE/dmu; no extra DDP factor",
        },
        "visible_gpus": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "runtime": {"torch": torch.__version__, "geoopt": geoopt.__version__,
                    "numpy": np.__version__},
        "reference_identity": None,
    }
    try:
        points, labels, shards = load_shards(args.data_dir, "train")
        train_ids, val_ids = stratified_split(labels, args.seed, .1)
        manifest.update(
            split_sha256=split_identity(train_ids, val_ids),
            train_size=len(train_ids), validation_size=len(val_ids),
            validation_read=False, test_read=False, training_shards=shards,
        )
        seed_all(args.seed)
        backbone = Hype_pointMLP().to(device)
        reference = torch.load(args.reference, map_location="cpu", weights_only=False)
        state = {key.removeprefix("module."): value
                 for key, value in reference["net"].items()}
        backbone.load_state_dict(state, strict=True)
        freeze_curvature(backbone)
        identity = {
            "sha256": checkpoint_sha(args.reference) if rank == 0 else None,
            "epoch": reference.get("epoch"), "path": str(args.reference),
            "role": "V4 representation reference, not V5 initialization or performance baseline",
        }
        identities = [identity if rank == 0 else None]
        dist.broadcast_object_list(identities, src=0)
        identity = identities[0]
        manifest["reference_identity"] = identity
        del reference, state
        # Initial DDP construction broadcasts parameters and buffers once.
        # Subsequent buffer broadcasts are explicit, even under no_grad.
        model = DDP(HyCoReTrainingForward(backbone), device_ids=[local_rank],
                    broadcast_buffers=False)
        model.train()
        dataset = PointClouds(points[train_ids], labels[train_ids], 1024,
                              "train", args.seed)
        sampler = SourceClassBatchSampler(labels[train_ids], rank=rank,
                                         seed=args.seed, steps=200)
        sampler.set_epoch(0)
        manifest["sampler_identity"] = sampler.state_dict()
        device_observations = [None, None]
        dist.all_gather_object(device_observations, {
            "rank": rank, "local_rank": local_rank,
            "device_name": torch.cuda.get_device_name(device),
            "device_total_MiB": torch.cuda.get_device_properties(device).total_memory / 1024 ** 2,
        })
        manifest["rank_devices"] = device_observations
        if rank == 0:
            write_json(args.run_dir / "manifest.json", manifest)

        bn_layers = [module for module in backbone.modules()
                     if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)]
        rows, cases, seen_plans = [], [], set()
        torch.cuda.reset_peak_memory_stats(device)
        for step in range(args.batches):
            plan = sampler.global_plan(step, epoch=0)
            if plan.indices in seen_plans:
                raise RuntimeError("Bounded collection received a repeated global batch plan")
            seen_plans.add(plan.indices)
            expected_ids = torch.as_tensor(train_ids[list(plan.indices)], device=device)
            expected_labels = torch.tensor(plan.labels, device=device)
            pair_ids = None
            for view in range(args.views):
                view_seed = args.seed + step * 1009 + view * 50000017 + rank * 1000003
                seed_all(view_seed)
                # Fetch fresh copies from the same IDs for each augmentation.
                # Do not reuse the alias-mutated tensor from the other view.
                fetched = [dataset[index] for index in plan.rank_indices(rank)]
                cloud = torch.stack([item[0] for item in fetched]).to(device)
                local_labels = torch.tensor([item[1] for item in fetched], device=device)
                local_ids = torch.as_tensor(
                    train_ids[[item[2] for item in fetched]], device=device)
                whole_count, child_count = shared_crop_counts(
                    args.seed + view * 50000017, 0, step)
                broadcast_working_buffers(backbone)
                before_bn = [int(module.num_batches_tracked) for module in bn_layers]
                dist.barrier()
                torch.cuda.synchronize(device)
                started = time.perf_counter()
                with torch.no_grad():
                    local_out = model(cloud, whole_count, child_count)
                del cloud
                ce_grad, local_ce, recomputed_logits = classifier_mu_gradient(
                    backbone, local_out["mu"], local_labels)
                logit_difference = (recomputed_logits - local_out["logits"]).abs().max()
                if float(logit_difference) > 1e-6:
                    raise RuntimeError("Frozen-classifier CE gradient did not match original logits")
                values = {
                    "mu": gather_without_grad(local_out["mu"].detach()),
                    "nu": gather_without_grad(local_out["nu"].detach()),
                    "gold": gather_without_grad(local_labels),
                    "ids": gather_without_grad(local_ids),
                    "ce_mu_grad": gather_without_grad(ce_grad),
                }
                local_ces = gather_without_grad(local_ce.reshape(1))
                counts = gather_without_grad(torch.tensor(
                    [[whole_count, child_count]], device=device))
                if bool((counts != counts[0]).any()):
                    raise RuntimeError("Paired-view crop counts differ between ranks")
                if not torch.equal(values["ids"], expected_ids):
                    raise RuntimeError("Gathered instance order differs from global sampler plan")
                if not torch.equal(values["gold"], expected_labels):
                    raise RuntimeError("Gathered labels differ from global sampler plan")
                if bool((values["gold"] == values["gold"].flip(0)).any()):
                    raise RuntimeError("Global64 child-flip negatives share the anchor class")
                if view == 0:
                    pair_ids = values["ids"].detach().clone()
                elif not torch.equal(pair_ids, values["ids"]):
                    raise RuntimeError("Paired views must have identical global IDs/order")
                if any(not torch.isfinite(value).all() for value in values.values()):
                    raise RuntimeError("Nonfinite cached representation or CE gradient")
                if any(parameter.grad is not None for parameter in backbone.parameters()):
                    raise RuntimeError("Read-only feature collection accumulated parameter gradients")
                bn_delta = [int(module.num_batches_tracked) - before
                            for module, before in zip(bn_layers, before_bn)]
                if any(delta != 2 for delta in bn_delta):
                    raise RuntimeError("Original child/whole BN must update twice per view")
                input_stats = local_out["input_stats"]
                if not all(input_stats[key] for key in (
                        "child_aliases_whole", "whole_aliases_source", "whole_changed_by_child")):
                    raise RuntimeError("Original HyCoRe part-overwrite behavior was not preserved")
                torch.cuda.synchronize(device)
                elapsed = torch.tensor(time.perf_counter() - started, device=device)
                dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
                observations = [None, None]
                dist.all_gather_object(observations, {
                    "rank": rank, "input": input_stats, "BN_update_deltas": bn_delta,
                    "peak_allocated_MiB": torch.cuda.max_memory_allocated(device) / 1024 ** 2,
                    "classifier_logit_max_difference": float(logit_difference),
                })
                row = {
                    "step": step, "view": view,
                    "seconds_max_rank": float(elapsed), "global_batch": 64,
                    "whole_count": whole_count, "child_count": child_count,
                    "ce_global_mean": float(local_ces.mean()),
                    "rank_observations": observations,
                }
                if rank == 0:
                    rows.append(row)
                    cases.append({
                        "stage": "v4_epoch20_reference", "step": step, "view": view,
                        "pair_id": f"v4_epoch20_reference:{step}",
                        "source_identity": identity,
                        "whole_count": whole_count, "child_count": child_count,
                        "ce_global_mean": row["ce_global_mean"],
                        **{key: value.detach().cpu().clone() for key, value in values.items()},
                    })
                    print(json.dumps({
                        "step": step, "view": view, "seconds": float(elapsed),
                        "child_count": child_count, "ce": row["ce_global_mean"],
                    }, allow_nan=False), flush=True)
                del local_out, values, ce_grad, recomputed_logits

        dist.barrier()
        # End collectives before rank0 serializes its CPU cache and hashes
        # the source file. Other ranks must not wait during CPU analysis.
        dist.destroy_process_group()
        del model, backbone, dataset, points
        torch.cuda.empty_cache()
        if rank == 0:
            final_sha = checkpoint_sha(args.reference)
            if final_sha != identity["sha256"]:
                raise RuntimeError("Reference checkpoint changed during read-only collection")
            feature_cache = {
                "identity": {
                    "commit": manifest["commit"], "branch": manifest["branch"],
                    "split_sha256": manifest["split_sha256"],
                    "source": "paired augmented training subsets only; restored HyCoRe overwrite/BN/FPS",
                    "reference_identity": identity,
                    "CE_gradient_semantics": manifest["fixed"]["CE_gradient"],
                },
                "config": manifest["fixed"], "proxy_count": 512, "cases": cases,
            }
            torch.save(feature_cache, args.run_dir / "training_features.pt")
            times = [row["seconds_max_rank"] for row in rows[4:]]
            write_json(args.run_dir / "collection.json", {
                "diagnostic_only": True, "optimizer_steps": 0,
                "global_plans": args.batches, "paired_views": args.views,
                "cached_cases": len(cases), "rows": rows,
                "median_view_seconds_after4": statistics.median(times) if times else None,
                "timing_limit": "two encoder forwards, frozen-classifier gradient and gathers; excludes input fetch and analysis",
                "reference_file_sha_unchanged": True,
            })
            manifest.update(
                status="completed", finished_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                wall_seconds=time.perf_counter() - started_wall,
                cached_cases=len(cases), source_file_sha_unchanged=True,
            )
            write_json(args.run_dir / "manifest.json", manifest)
            print("PAIRED_TRAIN_FEATURE_COLLECTION_COMPLETED", flush=True)
    except BaseException as exc:
        if rank == 0:
            manifest.update(status="failed", error=repr(exc),
                            finished_utc=dt.datetime.now(dt.timezone.utc).isoformat())
            write_json(args.run_dir / "manifest.json", manifest)
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
