"""V6 HyCoRe B64: two local32 ranks, full shuffled/drop-last training.

The only objective is released HyCoRe CE + .01 contrastive + .01 radial.
Every production epoch uses153 disjoint global64 batches from9840 objects,
then evaluates the complete official2468-object test split. Model selection
uses strict improvement of test OA rounded to three decimals, as in the
source. No hierarchy proxy, relation mining or extra optimizer is created.

This is a dual-GPU adaptation, not a bitwise rerun of the single-GPU source:
global crops are shared, RNG streams are rank-specific, ordinary BN is local32,
and the shuffle uses a private Python RNG rather than DataLoader randperm.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from ..hier_proxy_scratch_v5.diagnose import (
    REPO, Hype_pointMLP, PointClouds, checkpoint_sha, freeze_curvature,
    load_shards, seed_all,
)
from ..hier_proxy_scratch_v5.base_protocol import HyCoReTrainingForward
from ..hier_proxy_scratch_v5.distributed import (
    gather_with_grad, gather_without_grad, shared_crop_counts,
)
from ..hier_proxy_scratch_v5.telemetry import EpochTelemetry
from ..hier_proxy_scratch_v6.train import (
    worker_seed, synchronized_assert, gradient_norm, finite_tensors,
    append_jsonl, atomic_copy, cpu_state,
    checkpoint_rank_state,
)
from hier_proxy_scratch_v3.train import atomic_json, atomic_torch, capture_rng, restore_rng
from hier_proxy_scratch_v4.train import state_hash
from hutil import cal_loss, hype_triplet_losses
from models.manifolds import PoincareBall
from .sampling import SourceShuffleBatchSampler

import geoopt
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader


CHECKPOINT_VERSION = "hycore-v6-b64-shuffle-source-1"
EXPECTED_TRAIN_SIZE = 9840
EXPECTED_TEST_SIZE = 2468
GLOBAL_BATCH = 64
PRODUCTION_STEPS = 153
PRODUCTION_DRAWS = GLOBAL_BATCH * PRODUCTION_STEPS


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=22)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--workers", type=int, default=4,
                        help="Workers per rank; two ranks default to8 total")
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--train-eval-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=20)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-steps", type=int, default=2)
    parser.add_argument("--smoke-eval-batches", type=int, default=2)
    args = parser.parse_args()
    if args.workers < 0 or min(args.eval_batch_size, args.train_eval_every, args.save_every) < 1:
        parser.error("workers must be nonnegative; evaluation/save settings positive")
    if args.seed != 22:
        parser.error("Reviewed B64 initialization/sampling seed is22")
    if args.smoke:
        if not (1 <= args.epochs <= 2 and 1 <= args.smoke_steps <= 2
                and 1 <= args.smoke_eval_batches <= 2):
            parser.error("Smoke limits: epochs<=2, steps<=2, clean-train eval batches<=2")
    elif args.epochs != 300:
        parser.error("Reviewed production training budget is300epochs")
    return args


def fixed_configuration(args):
    return {
        "architecture": "Hype_pointMLP", "num_classes": 40,
        "seed": args.seed, "epochs": args.epochs,
        "steps_per_epoch": args.smoke_steps if args.smoke else PRODUCTION_STEPS,
        "production_steps_per_epoch": PRODUCTION_STEPS,
        "global_batch": GLOBAL_BATCH, "local_batch": 32, "world_size": 2,
        "dataset": "complete original ModelNet40 train9840; official test2468",
        "sampler": "one global shuffle without replacement; contiguous rank halves; drop_last",
        "production_draws_per_epoch": PRODUCTION_DRAWS,
        "production_dropped_per_epoch": EXPECTED_TRAIN_SIZE - PRODUCTION_DRAWS,
        "c": 1, "D": 256, "HIER": False, "proxy": None,
        "label_smoothing": .2, "alpha_contrastive": .01, "alpha_radial": .01,
        "contrastive_margin": 4., "radial_margin": "1000 / child_count",
        "intra_hinge": "released torch.max(x, zeros_like(x)), including tie gradient",
        "negative": "global child.flip(0); same-class negatives allowed and counted",
        "whole_count_range": [800, 1024], "child_count_range": [200, 600],
        "part_overwrites_whole": True, "FPS_first_stage": 512,
        "BN": "ordinary local32; child and whole update; DDP rank0 buffer broadcast",
        "precision": "FP32", "extra_tangent_cap": False, "extra_backward_hook": False,
        "model_optimizer": {"name": "RiemannianSGD", "lr": .1, "min_lr": .005,
                            "momentum": .9, "weight_decay": 2e-4, "norm_clip": 1.},
        "schedule": "cosine for full epoch budget; no warmup or restart",
        "evaluation": "every production epoch full clean official test; first1024 points",
        "model_selection": "strict greater test OA after source %.3f rounding; no CE tie-break",
        "reported_ce": "row-weighted CE and original rounded mean-of-batch CE, separately",
        "eval_batch_size": args.eval_batch_size, "workers_per_rank": args.workers,
        "train_eval_every": args.train_eval_every, "save_every": args.save_every,
        "smoke": args.smoke,
        "smoke_test": "at most2batches; partial engineering check only" if args.smoke else None,
        "evaluation_batch_limit": args.smoke_eval_batches if args.smoke else None,
        "rng_protocol": "V6 epoch/step/rank streams and shared counts; probes restore RNG/BN",
        "source_single_gpu_rng_bitwise_equivalent": False,
    }


def global_source_losses(local_outputs, local_labels):
    """Released losses over the rank-ordered global64, including label collisions.

    Differentiable gather backward SUM cancels DDP parameter-gradient averaging.
    There is no extra world-size multiplier and no rejection of random-batch
    same-class flip negatives. The old balanced-batch helper rejects these.
    """
    mu = gather_with_grad(local_outputs["mu"])
    nu = gather_with_grad(local_outputs["nu"])
    logits = gather_with_grad(local_outputs["logits"])
    gold = gather_without_grad(local_labels.reshape(-1).to(logits.device))
    if len(gold) != GLOBAL_BATCH:
        raise RuntimeError(f"Expected global64; got{len(gold)}")
    whole_count, child_count = int(local_outputs["whole_count"]), int(local_outputs["child_count"])
    counts = gather_without_grad(torch.tensor([[whole_count, child_count]], device=mu.device))
    if bool((counts != counts[0]).any()):
        raise RuntimeError("Whole and child counts must agree across both ranks")
    whole_radius, part_radius, positive, negative, contrastive, radial = hype_triplet_losses(
        mu, nu, hier_margin=1000. / child_count, contr_margin=4., ball_dim=256)
    ce = cal_loss(logits, gold)
    # Preserve the source expression's left-to-right floating-point grouping.
    base = ce + .01 * contrastive + .01 * radial
    with torch.no_grad():
        ball = PoincareBall(c=1., dim=256)
        whole_depth, part_depth = ball.dist0(mu).mean(), ball.dist0(nu).mean()
    return {
        "mu": mu, "nu": nu, "logits": logits, "gold": gold,
        "base": base, "ce": ce, "intra_contrastive": contrastive,
        "intra_radial": radial, "positive_distance": positive, "negative_distance": negative,
        "whole_radius": whole_radius, "part_radius": part_radius,
        "whole_depth": whole_depth, "part_depth": part_depth,
        "whole_count": whole_count, "child_count": child_count,
    }


@torch.no_grad()
def evaluate_clean_source(backbone, points, labels, args, device, prefix):
    """Original clean input/CE, full precision plus source rounded aggregates."""
    backbone.eval()
    dataset = PointClouds(points, labels, 1024, "clean", args.seed)
    loader = DataLoader(dataset, batch_size=args.eval_batch_size, shuffle=False,
                        num_workers=args.workers, pin_memory=True, worker_init_fn=worker_seed,
                        generator=torch.Generator().manual_seed(args.seed + 9000001))
    confusion = torch.zeros(40, 40, dtype=torch.int64)
    per_class_loss = torch.zeros(40, dtype=torch.float64)
    count, loss_sum, batch_loss_sum, batches = 0, 0., 0., 0
    for batch, (cloud, gold, _ids) in enumerate(loader):
        if args.smoke and batch >= args.smoke_eval_batches:
            break
        _mu, logits = backbone(cloud.to(device).transpose(1, 2).contiguous())
        if not torch.isfinite(logits).all():
            raise RuntimeError("Nonfinite clean-evaluation logits")
        targets = torch.zeros_like(logits).scatter_(1, gold.to(device)[:, None], 1)
        targets = targets * .8 + (1 - targets) * (.2 / 39)
        per_row_loss = -(targets * torch.log_softmax(logits, -1)).sum(-1).cpu()
        batch_loss_sum += float(cal_loss(logits, gold.to(device)))
        prediction = logits.argmax(-1).cpu()
        confusion += torch.bincount(gold * 40 + prediction, minlength=1600).reshape(40, 40)
        per_class_loss.scatter_add_(0, gold, per_row_loss.double())
        count += len(gold)
        loss_sum += float(per_row_loss.sum())
        batches += 1
    if count == 0:
        raise RuntimeError("Clean evaluation observed no examples")
    counts, correct = confusion.sum(1), confusion.diagonal()
    present = counts > 0
    oa = 100. * float(correct.sum()) / count
    aa = 100. * float((correct[present].double() / counts[present]).mean())
    return {
        prefix + "_oa": oa, prefix + "_aa": aa,
        prefix + "_oa_source_rounded": source_rounded(oa),
        prefix + "_aa_source_rounded": source_rounded(aa),
        prefix + "_ce": loss_sum / count,
        prefix + "_ce_source_batch_mean": batch_loss_sum / batches,
        prefix + "_ce_source_rounded": source_rounded(batch_loss_sum / batches),
        prefix + "_count": count, prefix + "_batches": batches,
        prefix + "_class_count": counts.tolist(),
        prefix + "_class_accuracy": [100. * float(correct[c]) / int(counts[c])
                                    if int(counts[c]) else None for c in range(40)],
        prefix + "_class_ce": [float(per_class_loss[c]) / int(counts[c])
                              if int(counts[c]) else None for c in range(40)],
        prefix + "_confusion": confusion.tolist(), prefix + "_smoke_partial": args.smoke,
        prefix + "_partial_not_reportable": args.smoke,
    }


def evaluate_preserving_buffers(backbone, points, labels, args, device, prefix, rank):
    """V6 collective-safe pattern; restore rank0 RNG and all nonparameter buffers."""
    dist.barrier()
    result, failure = None, None
    if rank == 0:
        rng = capture_rng()
        buffers = {name: value.detach().clone() for name, value in backbone.named_buffers()}
        try:
            result = evaluate_clean_source(backbone, points, labels, args, device, prefix)
            if any(not torch.equal(value, buffers[name]) for name, value in backbone.named_buffers()):
                raise RuntimeError("Clean evaluation unexpectedly changed a model buffer")
            result[prefix + "_buffers_unchanged"] = True
        except BaseException as exc:
            failure = repr(exc)
        finally:
            restore_rng(rng)
            with torch.no_grad():
                for name, buffer in backbone.named_buffers():
                    buffer.copy_(buffers[name])
    shared = [result, failure]
    dist.broadcast_object_list(shared, src=0)
    if shared[1] is not None:
        raise RuntimeError("Rank0 clean evaluation failed: " + shared[1])
    backbone.train()
    dist.barrier()
    return shared[0]


@torch.no_grad()
def parameter_replica_difference(parameters, *, gradients=False):
    """Compare parameters/gradients only: local32 BN buffers need not agree."""
    values = [parameter.grad if gradients else parameter for parameter in parameters]
    if gradients:
        presence = torch.tensor([value is not None for value in values],
                                device=parameters[0].device, dtype=torch.int32)
        all_presence = gather_without_grad(presence)
        synchronized_assert(torch.equal(all_presence[:len(presence)], all_presence[len(presence):]),
                            "Gradient presence differs between DDP ranks", parameters[0].device)
    observed = [value.detach().float().reshape(-1) for value in values if value is not None]
    if not observed:
        return 0.
    vector = torch.cat(observed)
    twin = vector.clone()
    dist.broadcast(twin, src=0)
    difference = (twin - vector).abs().max()
    dist.all_reduce(difference, op=dist.ReduceOp.MAX)
    return float(difference)


def source_rounded(value):
    return float("%.3f" % float(value))


def main():
    args = arguments()
    torch.set_num_threads(2)
    rank, local_rank = int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"])
    if int(os.environ["WORLD_SIZE"]) != 2:
        raise RuntimeError("V6 B64 requires exactly two ranks")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", timeout=dt.timedelta(minutes=10))
    started_wall = time.perf_counter()
    config = fixed_configuration(args)
    manifest = {
        "status": "starting", "main_training": not args.smoke, "diagnostic_only": args.smoke,
        "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        "branch": subprocess.check_output(["git", "branch", "--show-current"], cwd=REPO, text=True).strip(),
        "command_argv": [sys.executable, *sys.argv],
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "training_config": config, "visible_gpus": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "physical_gpus": os.environ.get("HYCORE_PHYSICAL_GPUS", os.environ.get("CUDA_VISIBLE_DEVICES")),
        "physical_gpu_uuids": os.environ.get("HYCORE_PHYSICAL_GPU_UUIDS"),
        "runtime": {"torch": torch.__version__, "geoopt": geoopt.__version__, "numpy": np.__version__},
        "initialization": {"method": "random", "seed": args.seed, "source_checkpoint": None, "teacher": None},
        "official_test_read": False, "official_test_evaluations": 0,
        "official_test_full_evaluations": 0,
        "validation_size": 0, "resume": False,
    }
    owns_run_dir = False
    try:
        startup_error = [None]
        if rank == 0:
            try:
                args.run_dir.mkdir(parents=True, exist_ok=False)
                owns_run_dir = True
            except BaseException as exc:
                startup_error[0] = repr(exc)
        dist.broadcast_object_list(startup_error, src=0)
        if startup_error[0] is not None:
            raise RuntimeError("New run directory creation failed: " + startup_error[0])
        dist.barrier()
        points, labels, shards = load_shards(args.data_dir, "train")
        test_points, test_labels, test_shards = load_shards(args.data_dir, "test")
        synchronized_assert(len(labels) == EXPECTED_TRAIN_SIZE and len(test_labels) == EXPECTED_TEST_SIZE,
                            "Expected original train9840/test2468 dataset", device)
        synchronized_assert(points.shape[1:] == (2048, 3) and test_points.shape[1:] == (2048, 3)
                            and np.isfinite(points).all() and np.isfinite(test_points).all()
                            and np.array_equal(np.unique(labels), np.arange(40))
                            and np.array_equal(np.unique(test_labels), np.arange(40)),
                            "Unexpected ModelNet40 point/label schema", device)
        train_ids = np.arange(len(labels), dtype=np.int64)
        ids_sha = hashlib.sha256(train_ids.tobytes()).hexdigest()
        manifest.update(
            train_size=len(labels), test_size=len(test_labels), train_ids_sha256=ids_sha,
            training_shards=shards, test_shards=test_shards, official_test_read=True,
            official_test_read_reason="authorized source every-epoch test; smoke forward is explicitly partial",
            train_labels_sha256=hashlib.sha256(labels.tobytes()).hexdigest(),
            test_labels_sha256=hashlib.sha256(test_labels.tobytes()).hexdigest(),
            train_class_sizes=np.bincount(labels, minlength=40).tolist(),
            test_class_sizes=np.bincount(test_labels, minlength=40).tolist(),
        )
        seed_all(args.seed)
        backbone = Hype_pointMLP().to(device)
        freeze_curvature(backbone)
        synchronized_assert(all(bool(torch.isclose(manifold.c.detach(),
            torch.ones_like(manifold.c), atol=1e-6, rtol=0).all())
            for manifold in (backbone.manifold, backbone.manifold2)),
            "Both backbone manifolds must retain c=1", device)
        initial_hash = state_hash(backbone)
        manifest["initialized_model_sha256"] = initial_hash
        model = DDP(HyCoReTrainingForward(backbone), device_ids=[local_rank], broadcast_buffers=True)
        model_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        optimizer = geoopt.optim.RiemannianSGD(model_parameters, lr=.1, momentum=.9, weight_decay=2e-4)
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=.005)
        dataset = PointClouds(points, labels, 1024, "train", args.seed)
        sampler = SourceShuffleBatchSampler(train_ids, rank=rank, seed=args.seed,
                                           global_batch=GLOBAL_BATCH, world_size=2)
        synchronized_assert(len(sampler) == PRODUCTION_STEPS, "Sampler must produce153full global batches", device)
        devices = [None, None]
        dist.all_gather_object(devices, {
            "rank": rank, "local_rank": local_rank, "name": torch.cuda.get_device_name(device),
            "uuid": str(getattr(torch.cuda.get_device_properties(device), "uuid", "unavailable")),
            "total_MiB": torch.cuda.get_device_properties(device).total_memory / 1024 ** 2,
        })
        manifest.update(rank_devices=devices, sampler_identity=sampler.state_dict(),
                        status="running", completed_epochs=0)
        if rank == 0:
            atomic_json(args.run_dir / "manifest.json", manifest)
        best, best_net = None, None
        cumulative_seen = set()

        for epoch_index in range(args.epochs):
            epoch = epoch_index + 1
            model.train()
            sampler.set_epoch(epoch_index)
            epoch_seed = args.seed + epoch_index * 1009 + rank * 1000003
            seed_all(epoch_seed)
            loader = DataLoader(dataset, batch_sampler=sampler, num_workers=args.workers,
                                worker_init_fn=worker_seed, pin_memory=True,
                                generator=torch.Generator().manual_seed(epoch_seed))
            telemetry = EpochTelemetry(train_ids, labels, num_classes=40) if rank == 0 else None
            learning_rates = {"model": optimizer.param_groups[0]["lr"]}
            epoch_started = time.perf_counter()
            torch.cuda.reset_peak_memory_stats(device)
            expected_steps = args.smoke_steps if args.smoke else len(sampler)
            steps_done = 0
            for step, (cloud, local_gold, subset_ids) in enumerate(loader):
                if step >= expected_steps:
                    break
                step_seed = args.seed + epoch_index * 10000019 + step * 1009 + rank * 1000003
                seed_all(step_seed)
                cloud, local_gold = cloud.to(device), local_gold.to(device)
                local_ids = torch.as_tensor(train_ids[subset_ids.numpy()], device=device)
                whole_count, child_count = shared_crop_counts(args.seed, epoch_index, step)
                optimizer.zero_grad(set_to_none=True)
                bn_layers = [module for module in backbone.modules()
                             if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)]
                bn_before = [int(module.num_batches_tracked) for module in bn_layers]
                started = time.perf_counter()
                local_out = model(cloud, whole_count, child_count)
                values = global_source_losses(local_out, local_gold)
                ids = gather_without_grad(local_ids)
                planned_ids = torch.tensor(sampler.global_plan(step).ids, device=device, dtype=ids.dtype)
                synchronized_assert(torch.equal(ids, planned_ids),
                                    "Gathered batch order differs from sampler global plan", device)
                synchronized_assert(len(torch.unique(ids)) == GLOBAL_BATCH,
                                    "Global shuffled batch contains repeated ID", device)
                loss = values["base"]
                synchronized_assert(finite_tensors([loss, values["mu"], values["nu"], values["logits"]]),
                                    "Nonfinite HyCoRe objective/embeddings", device)
                loss.backward()
                synchronized_assert(finite_tensors(parameter.grad for parameter in model_parameters),
                                    "Nonfinite model gradients", device)
                replica_check = step in (0, expected_steps - 1)
                gradient_difference = parameter_replica_difference(model_parameters, gradients=True) if replica_check else None
                if replica_check:
                    synchronized_assert(gradient_difference <= 1e-7,
                                        "DDP model gradients differ between ranks", device)
                bn_delta = [int(module.num_batches_tracked) - before
                            for module, before in zip(bn_layers, bn_before)]
                synchronized_assert(bool(bn_delta) and all(delta == 2 for delta in bn_delta),
                                    "Every BN layer must update twice per training step", device)
                synchronized_assert(all(local_out["input_stats"][key] for key in
                    ("child_aliases_whole", "whole_aliases_source", "whole_changed_by_child")),
                    "Original part-overwrite behavior changed", device)
                preclip_norm = torch.nn.utils.clip_grad_norm_(model_parameters, 1.)
                postclip_norm = gradient_norm(model_parameters)
                synchronized_assert(bool(torch.isfinite(preclip_norm)) and bool(torch.isfinite(postclip_norm))
                                    and float(postclip_norm) <= 1.00001,
                                    "Nonfinite or excessive postclip gradient norm", device)
                optimizer.step()
                synchronized_assert(finite_tensors(model_parameters), "Nonfinite updated model parameters", device)
                parameter_difference = parameter_replica_difference(model_parameters) if replica_check else None
                if replica_check:
                    synchronized_assert(parameter_difference <= 1e-7,
                                        "DDP updated model parameters differ between ranks", device)
                synchronized_assert(all(bool(torch.isclose(manifold.c.detach(),
                    torch.ones_like(manifold.c), atol=1e-6, rtol=0).all())
                    for manifold in (backbone.manifold, backbone.manifold2)),
                    "Backbone curvature changed from c=1", device)
                torch.cuda.synchronize(device)
                elapsed = torch.tensor(time.perf_counter() - started, device=device)
                dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
                rank_peaks = [None, None]
                dist.all_gather_object(rank_peaks, {
                    "rank": rank, "peak_allocated_MiB": torch.cuda.max_memory_allocated(device) / 1024 ** 2,
                    "BN_update_min": min(bn_delta), "BN_update_max": max(bn_delta),
                    "input": local_out["input_stats"], "local_batch_class_count": int(local_gold.unique().numel()),
                })
                same_class_negative_count = int((values["gold"] == values["gold"].flip(0)).sum())
                scalars = {
                    "loss": float(loss.detach()), "base": float(values["base"].detach()),
                    "ce": float(values["ce"].detach()),
                    "intra_contrastive": float(values["intra_contrastive"].detach()),
                    "intra_radial": float(values["intra_radial"].detach()),
                    "weighted_intra_contrastive": .01 * float(values["intra_contrastive"].detach()),
                    "weighted_intra_radial": .01 * float(values["intra_radial"].detach()),
                    "intra_positive_distance": float(values["positive_distance"].detach()),
                    "intra_negative_distance": float(values["negative_distance"].detach()),
                    "whole_mean_radius": float(values["whole_radius"].detach()),
                    "part_mean_radius": float(values["part_radius"].detach()),
                    "whole_mean_depth": float(values["whole_depth"]), "part_mean_depth": float(values["part_depth"]),
                    "model_preclip_norm": float(preclip_norm), "model_postclip_norm": float(postclip_norm),
                    "model_clip_applied": float(preclip_norm > 1.),
                    "model_clip_scale": min(1., 1. / (float(preclip_norm) + 1e-6)),
                    "step_seconds_max_rank": float(elapsed),
                    "peak_allocated_MiB_max_rank": max(item["peak_allocated_MiB"] for item in rank_peaks),
                    "whole_count": whole_count, "child_count": child_count,
                    "global_batch_class_count": int(values["gold"].unique().numel()),
                    "same_class_flip_negative_count": same_class_negative_count,
                    "same_class_flip_negative_fraction": same_class_negative_count / GLOBAL_BATCH,
                    "same_instance_flip_negative_count": int((ids == ids.flip(0)).sum()),
                    "lambda_hier": 0.,
                    "model_gradient_replica_max_difference": gradient_difference,
                    "model_parameter_replica_max_difference": parameter_difference,
                }
                if rank == 0:
                    telemetry.update(ids=ids, labels=values["gold"], logits=values["logits"],
                                     mu=values["mu"], nu=values["nu"], scalar_metrics=scalars)
                    cumulative_seen.update(ids.detach().cpu().tolist())
                    atomic_json(args.run_dir / "heartbeat.json", {
                        "status": "running", "epoch": epoch, "step": step + 1,
                        "steps_per_epoch": expected_steps, "phase": "HyCoRe_base_only",
                        "updated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                        "lr": learning_rates, "scalars": scalars, "rank_observations": rank_peaks,
                    })
                    append_jsonl(args.run_dir / "steps.jsonl", {
                        "epoch": epoch, "step": step + 1, "ids": ids.detach().cpu().tolist(), **scalars})
                    if step == 0 or (step + 1) % 20 == 0 or step + 1 == expected_steps:
                        print(json.dumps({"epoch": epoch, "step": step + 1, "loss": scalars["loss"],
                                          "lr": learning_rates}, allow_nan=False), flush=True)
                steps_done += 1
                del local_out, values, loss
            synchronized_assert(steps_done == expected_steps, "Epoch did not complete declared update count", device)
            plan_summary = sampler.epoch_summary(include_ids=True)
            synchronized_assert(plan_summary["steps"] == PRODUCTION_STEPS
                                and plan_summary["draws"] == PRODUCTION_DRAWS
                                and plan_summary["unique"] == PRODUCTION_DRAWS
                                and plan_summary["dropped_count"] == 48
                                and plan_summary["padding_count"] == 0
                                and plan_summary["repeated_draws"] == 0,
                                "Full production sampler coverage differs from153x64drop-last", device)
            training_summary = telemetry.summary() if rank == 0 else None
            coverage_okay = True
            if rank == 0:
                coverage = training_summary["sample_coverage"]
                coverage_okay = coverage["draws"] == expected_steps * GLOBAL_BATCH and (
                    coverage["unique_ids"] == expected_steps * GLOBAL_BATCH) and coverage["repeated_draws"] == 0
            synchronized_assert(coverage_okay, "Observed epoch unique coverage differs from actual sampler draw plan", device)

            test_metrics = None
            clean_train = None
            test_metrics = evaluate_preserving_buffers(backbone, test_points, test_labels, args, device, "test", rank)
            if not args.smoke:
                synchronized_assert(test_metrics["test_count"] == EXPECTED_TEST_SIZE,
                                    "Official test evaluation must cover all2468 rows", device)
                manifest["official_test_full_evaluations"] += 1
            manifest["official_test_evaluations"] += 1
            if args.smoke or epoch % args.train_eval_every == 0 or epoch == args.epochs:
                clean_train = evaluate_preserving_buffers(backbone, points, labels, args, device, "train_eval", rank)
                if not args.smoke:
                    synchronized_assert(clean_train["train_eval_count"] == EXPECTED_TRAIN_SIZE,
                                        "Clean train evaluation must cover all9840 rows", device)
            scheduler.step()
            # Smoke exercises selection/saving on a partial subset, explicitly
            # labelled nonreportable and never used to initialize production.
            improved = test_metrics is not None and (best is None or
                test_metrics["test_oa_source_rounded"] > best["test_oa_source_rounded"])
            if improved:
                best = {"epoch": epoch, "test_oa": test_metrics["test_oa"],
                        "test_oa_source_rounded": test_metrics["test_oa_source_rounded"],
                        "test_aa": test_metrics["test_aa"], "test_ce": test_metrics["test_ce"],
                        "selected_by": "source-rounded test OA strictgreater"}
                if rank == 0:
                    best_net = cpu_state(backbone)
            rank_states = [None, None]
            dist.all_gather_object(rank_states, checkpoint_rank_state(backbone, sampler, rank))
            checkpoint_error = None
            if rank == 0:
                row = {
                    "epoch": epoch, "phase": "HyCoRe_base_only", "lr_used": learning_rates,
                    "lr_next": {"model": optimizer.param_groups[0]["lr"]},
                    "epoch_wall_seconds": time.perf_counter() - epoch_started,
                    "telemetry": training_summary, "test": test_metrics, "clean_train": clean_train,
                    "best": best, "best_updated": improved,
                    "sampler": {key: value for key, value in plan_summary.items() if key != "plans"},
                    "actual_optimizer_steps": steps_done,
                    "cumulative_unique_training_ids": len(cumulative_seen),
                    "cumulative_training_fraction": len(cumulative_seen) / EXPECTED_TRAIN_SIZE,
                    "official_test_partial_not_reportable": args.smoke,
                }
                payload = {
                    "format": CHECKPOINT_VERSION, "model_selection_only": False,
                    "completed_epochs": epoch, "epoch": epoch, "net": backbone.state_dict(),
                    "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                    "rank_states": rank_states, "training_config": config,
                    "train_ids_sha256": ids_sha, "train_ids": train_ids,
                    "validation_ids": np.empty(0, dtype=np.int64),
                    "commit": manifest["commit"], "branch": manifest["branch"],
                    "initialization": manifest["initialization"], "initialized_model_sha256": initial_hash,
                    "best": best, "best_net": best_net, "metrics": row,
                    "saved_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                }
                try:
                    atomic_json(args.run_dir / f"sampler_epoch_{epoch:03d}.json", plan_summary)
                    last_path = args.run_dir / "last.pth"
                    atomic_torch(last_path, payload)
                    saved_files = ["last.pth"]
                    if improved:
                        atomic_copy(last_path, args.run_dir / "best.pth")
                        saved_files.append("best.pth")
                    if epoch % args.save_every == 0:
                        name = f"checkpoint_epoch_{epoch:03d}.pth"
                        atomic_copy(last_path, args.run_dir / name)
                        saved_files.append(name)
                    row["checkpoint"] = {
                        "epoch": epoch, "files": saved_files, "last_sha256": checkpoint_sha(last_path),
                        "last_bytes": last_path.stat().st_size, "saved_utc": payload["saved_utc"],
                    }
                    atomic_json(args.run_dir / f"metrics_epoch_{epoch:03d}.json", row)
                    append_jsonl(args.run_dir / "metrics.jsonl", row)
                    manifest.update(status="running", completed_epochs=epoch, best=best,
                                    latest_checkpoint=row["checkpoint"],
                                    updated_utc=dt.datetime.now(dt.timezone.utc).isoformat())
                    atomic_json(args.run_dir / "manifest.json", manifest)
                    print(json.dumps({"epoch": epoch, "test_oa": test_metrics["test_oa"] if test_metrics else None,
                                      "best": best, "coverage": training_summary["sample_coverage"]["fraction"]},
                                     allow_nan=False), flush=True)
                except BaseException as exc:
                    checkpoint_error = repr(exc)
            error = [checkpoint_error]
            dist.broadcast_object_list(error, src=0)
            if error[0] is not None:
                raise RuntimeError("Epoch checkpoint failed: " + error[0])
            dist.barrier()

        dist.barrier()
        if rank == 0:
            finished = dt.datetime.now(dt.timezone.utc).isoformat()
            manifest.update(status="completed", completed_epochs=args.epochs, finished_utc=finished,
                            wall_seconds=time.perf_counter() - started_wall, best=best,
                            official_test_partial_not_reportable=args.smoke)
            atomic_json(args.run_dir / "manifest.json", manifest)
            atomic_json(args.run_dir / "heartbeat.json", {"status": "completed", "epoch": args.epochs,
                        "updated_utc": finished, "best": best})
            print("V6_B64_HYCORE_TRAINING_COMPLETED", flush=True)
    except BaseException as exc:
        if rank == 0 and owns_run_dir:
            manifest.update(status="failed", error=repr(exc),
                            finished_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                            wall_seconds=time.perf_counter() - started_wall)
            atomic_json(args.run_dir / "manifest.json", manifest)
            atomic_json(args.run_dir / "heartbeat.json", {"status": "failed", "error": repr(exc)})
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
