"""V9 B64 native-c1 control: same base protocol, without a HIER branch or proxies.

Full9840 train pool; official2468 test selects every epoch. Smoke never reads test.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time

from ..hier_proxy_scratch_v5.diagnose import (
    REPO, Hype_pointMLP, PointClouds,
    SourceClassBatchSampler, checkpoint_sha,
    gather_without_grad, load_shards, seed_all, shared_crop_counts,
    split_identity,
)
from .base_protocol import global_base_losses, smoothed_ce, HyCoReTrainingForward
from .geometry import configure_backbone, freeze_curvature
from .telemetry import EpochTelemetry as NativeEpochTelemetry
from hier_proxy_scratch_v3.train import atomic_json, atomic_torch, capture_rng, restore_rng
from hier_proxy_scratch_v4.train import state_hash

import geoopt
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader


CHECKPOINT_VERSION = "hycore-v9-nativec1-full9840-testselect-b64-control-1"
from ..hier_proxy_scratch_v8_fulltrain.protocol import full_training_ids, select_best


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", choices=("B64",), default="B64")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--seed", type=int, default=22)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--steps-per-epoch", type=int, default=200)
    parser.add_argument("--warmup-epochs", type=int, default=0)
    parser.add_argument("--lambda-hier", type=float, default=0.)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--train-eval-every", type=int, default=10)
    parser.add_argument("--audit-epochs", default="1,5,20,40,99,100,160,200,240,300")
    parser.add_argument("--save-every", type=int, default=20)
    parser.add_argument("--skip-final-test", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-eval-batches", type=int, default=2)
    args = parser.parse_args()
    args.audit_epochs = tuple(sorted(set(int(item) for item in args.audit_epochs.split(",") if item.strip())))
    if args.workers < 0 or args.eval_batch_size < 1 or args.train_eval_every < 1 or args.save_every < 1:
        parser.error("workers must be nonnegative; evaluation/save intervals positive")
    if not (args.warmup_epochs == 0 and math.isfinite(args.lambda_hier) and args.lambda_hier == 0.):
        parser.error("V9 B64 control requires W0 and HIER weight zero")
    if args.smoke:
        if not (1 <= args.epochs <= 2 and 1 <= args.steps_per_epoch <= 2 and
                0 <= args.warmup_epochs < args.epochs and 1 <= args.smoke_eval_batches <= 2):
            parser.error("Smoke bounds: epochs<=2, steps<=2, warmup<epochs, eval batches<=2")
        if not args.skip_final_test:
            parser.error("Smoke runs must use --skip-final-test")
    elif (args.epochs, args.steps_per_epoch, args.warmup_epochs, args.seed) != (300, 200, 0, 22):
        parser.error("V9 B64 control requires seed22, 300epochs x200steps, W0")
    if any(epoch < 1 for epoch in args.audit_epochs):
        parser.error("Audit epochs use one-based positive epoch numbers")
    return args


def fixed_configuration(args):
    """Explicit native-c1 configuration; no HIER module is imported or created."""
    name = "V9-B64-C1-CAPnone-GS0-M1000_4-TS-S22"
    return {
        "architecture": "Hype_pointMLP", "num_classes": 40, "control": "B64",
        "seed": args.seed, "epochs": args.epochs, "steps_per_epoch": args.steps_per_epoch,
        "warmup_epochs": 0, "global_batch": 64, "world_size": 2,
        "classes_per_rank": 16, "instances_per_class": 2,
        "sampler": "class-uniform; class-internal replacement; rank-disjoint class blocks",
        "c": 1., "base_c": 1., "native_c": 1., "hier_c": None,
        "D": 256, "P": 0, "sample_K": None, "proxy_K": None, "T": None,
        "margin": None, "tau": None, "lambda_hier_after_warmup": 0.,
        "label_smoothing": .2, "alpha_contrastive": .01, "alpha_radial": .01,
        "contrastive_margin": 4., "radial_gamma": 1000., "radial_margin": "1000 / child_count",
        "whole_count_range": [800, 1024], "child_count_range": [200, 600],
        "part_overwrites_whole": True, "FPS_first_stage": 512,
        "BN": "ordinary local32 training; child and whole update; rank0 buffer broadcast",
        "geometry_domains": {"native_whole_part": 1.},
        "single_curvature_audit_compatible": True,
        "precision": "FP32", "native_extra_cap": None, "native_GS": False,
        "hier_view": None, "hier_view_depth": None, "proxy_tangent_cap": None,
        "HIER_backward_hook": False, "hook_scope": None,
        "intra_hinge": "official torch.maximum, including half gradient at exact zero",
        "self_negative": False, "negative_flip_same_class": "forbidden by class-disjoint B64",
        "proxy_initialization": None, "proxy_parameter_projection": None,
        "hier_distance": None, "intra_distance": "Geoopt PoincareBall c=1",
        "HIER_objective_enabled": False, "HIER_calls_per_step": 0,
        "model_optimizer": {"name": "RiemannianSGD", "lr": .1, "min_lr": .005,
                            "momentum": .9, "weight_decay": 2e-4, "norm_clip": 1.},
        "proxy_optimizer": None,
        "schedule": "model 300-epoch cosine; no proxy optimizer",
        "validation": "none", "canonical_name": name, "display_name": name,
        "training_pool": "official train9840, all objects available to class sampler",
        "selection": "official test2468 every epoch; strictly higher unrounded OA; first tie retained",
        "selection_forward": "eval/no_grad only; no optimizer, scheduler feedback or early stopping",
        "train_eval_every": args.train_eval_every, "eval_batch_size": args.eval_batch_size,
        "workers": args.workers, "audit_epochs": list(args.audit_epochs), "smoke": args.smoke,
        "evaluation_batch_limit": args.smoke_eval_batches if args.smoke else None,
    }


class EpochTelemetry(NativeEpochTelemetry):
    """Report native geometry only; a control has no HIER shell observation."""
    def summary(self):
        result = super().summary()
        result["geometry"]["hier_view_domain"] = {
            "available": False, "c": None, "target_whole_depth": None,
            "hook_applied": False, "reason": "HIER branch and proxies are absent"}
        return result


def control_base_losses(outputs, labels, control="B64"):
    if control != "B64":
        raise ValueError("V9 native control supports only the global64 B64 protocol")
    return global_base_losses(outputs, labels)


def worker_seed(_worker):
    seed = torch.initial_seed() % (2 ** 32)
    random.seed(seed)
    np.random.seed(seed)


def synchronized_assert(condition, message, device):
    """All ranks agree before an exception would strand another collective."""
    okay = torch.tensor(1 if condition else 0, device=device, dtype=torch.int32)
    dist.all_reduce(okay, op=dist.ReduceOp.MIN)
    if int(okay) == 0:
        raise RuntimeError(message)


def gradient_norm(parameters):
    gradients = [parameter.grad.detach().float().norm().square()
                 for parameter in parameters if parameter.grad is not None]
    return torch.stack(gradients).sum().sqrt() if gradients else torch.tensor(0.)


def finite_tensors(values):
    checks = [torch.isfinite(value).all() for value in values if value is not None]
    return bool(torch.stack(checks).all()) if checks else True


def append_jsonl(path, value):
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()


def atomic_copy(source, target):
    temporary = target.with_name(target.name + ".tmp")
    shutil.copyfile(source, temporary)
    os.replace(temporary, target)


def cpu_state(module):
    return {name: value.detach().cpu().clone() for name, value in module.state_dict().items()}


@torch.no_grad()
def evaluate_clean(backbone, points, labels, args, device, prefix):
    """Unwrapped rank0 model; no DDP collective is called during evaluation."""
    backbone.eval()
    dataset = PointClouds(points, labels, 1024, "clean", args.seed)
    loader = DataLoader(dataset, batch_size=args.eval_batch_size, shuffle=False,
                        num_workers=args.workers, pin_memory=True,
                        worker_init_fn=worker_seed,
                        generator=torch.Generator().manual_seed(args.seed + 9000001))
    confusion = torch.zeros(40, 40, dtype=torch.int64)
    per_class_loss = torch.zeros(40, dtype=torch.float64)
    count, loss_sum = 0, 0.
    for batch, (cloud, gold, _ids) in enumerate(loader):
        if args.smoke and batch >= args.smoke_eval_batches:
            break
        _mu, logits = backbone(cloud.to(device).transpose(1, 2).contiguous())
        if not torch.isfinite(logits).all():
            raise RuntimeError("Nonfinite clean-evaluation logits")
        targets = torch.zeros_like(logits).scatter_(1, gold.to(device)[:, None], 1)
        targets = targets * .8 + (1 - targets) * (.2 / 39)
        per_row_loss = -(targets * torch.log_softmax(logits, -1)).sum(-1).cpu()
        prediction = logits.argmax(-1).cpu()
        confusion += torch.bincount(gold * 40 + prediction, minlength=1600).reshape(40, 40)
        per_class_loss.scatter_add_(0, gold, per_row_loss.double())
        count += len(gold)
        loss_sum += float(per_row_loss.sum())
    counts = confusion.sum(1)
    correct = confusion.diagonal()
    present = counts > 0
    return {
        prefix + "_oa": 100. * float(correct.sum()) / count,
        prefix + "_aa": 100. * float((correct[present] / counts[present]).mean()),
        prefix + "_ce": loss_sum / count,
        prefix + "_count": count,
        prefix + "_class_count": counts.tolist(),
        prefix + "_class_accuracy": [100. * float(correct[c]) / int(counts[c])
                                    if int(counts[c]) else None for c in range(40)],
        prefix + "_class_ce": [float(per_class_loss[c]) / int(counts[c])
                              if int(counts[c]) else None for c in range(40)],
        prefix + "_confusion": confusion.tolist(),
        prefix + "_smoke_partial": args.smoke,
    }


def rank0_evaluation(backbone, points, labels, args, device, prefix, rank):
    dist.barrier()
    result, failure = None, None
    if rank == 0:
        rng = capture_rng()
        before_buffers = {name: value.detach().clone() for name, value in backbone.named_buffers()}
        try:
            result = evaluate_clean(backbone, points, labels, args, device, prefix)
            if any(not torch.equal(value, before_buffers[name]) for name, value in backbone.named_buffers()):
                raise RuntimeError("Evaluation changed model/BN buffers")
        except BaseException as exc:
            failure = repr(exc)
        finally:
            restore_rng(rng)
    shared = [result, failure]
    dist.broadcast_object_list(shared, src=0)
    if shared[1] is not None:
        raise RuntimeError("Rank0 evaluation failed: " + shared[1])
    backbone.train()
    dist.barrier()
    return shared[0]


def checkpoint_rank_state(backbone, sampler, rank):
    return {
        "rank": rank, "rng": capture_rng(), "sampler": sampler.state_dict(),
        "BN_buffers": {name: value.detach().cpu().clone()
                       for name, value in backbone.named_buffers()},
    }


def validate_resume(saved, config, split_sha, train_ids, selection_ids, sampler):
    if saved.get("format") != CHECKPOINT_VERSION or saved.get("model_selection_only", False):
        raise RuntimeError("Resume requires a complete V9 B64 training checkpoint")
    if saved.get("training_config") != config or saved.get("split_sha256") != split_sha:
        raise RuntimeError("Resume configuration or split differs from checkpoint")
    if not np.array_equal(saved["train_ids"], train_ids) or not np.array_equal(saved["selection_ids"], selection_ids):
        raise RuntimeError("Resume sample IDs differ from checkpoint")
    if len(saved["rank_states"]) != config["world_size"]:
        raise RuntimeError("Resume checkpoint must contain all rank states")
    sampler.load_state_dict(saved["rank_states"][0]["sampler"])
    if not 0 <= int(saved["completed_epochs"]) <= config["epochs"]:
        raise RuntimeError("Invalid resume epoch")



def main():
    args = arguments()
    torch.set_num_threads(2)
    rank, local_rank = int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"])
    world_size = 2
    if int(os.environ["WORLD_SIZE"]) != world_size:
        raise RuntimeError("Control world size differs from sampler")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", timeout=dt.timedelta(minutes=10))
    started_wall = time.perf_counter()
    if rank == 0:
        args.run_dir.mkdir(parents=True, exist_ok=False)
    dist.barrier()
    config = fixed_configuration(args)
    manifest = {
        "status": "starting", "main_training": not args.smoke,
        "diagnostic_only": args.smoke,
        "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "commit": (os.environ.get("V9_SOURCE_COMMIT") or subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()),
        "branch": (os.environ.get("V9_SOURCE_BRANCH") or subprocess.check_output(["git", "branch", "--show-current"], cwd=REPO, text=True).strip()),
        "command_argv": [sys.executable, *sys.argv],
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "canonical_name": config["canonical_name"], "training_config": config, "visible_gpus": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "physical_gpus": os.environ.get('HYCORE_PHYSICAL_GPUS', os.environ.get('CUDA_VISIBLE_DEVICES')),
        "physical_gpu_uuids": os.environ.get('HYCORE_PHYSICAL_GPU_UUIDS'),
        "runtime": {"torch": torch.__version__, "geoopt": geoopt.__version__, "numpy": np.__version__},
        "initialization": {"method": "random", "seed": args.seed, "source_checkpoint": None, "teacher": None},
        "official_test_read": False, "resume_identity": None,
        "representation_domains": {"whole_part": "native c1, no extra cap/GS",
                                   "HIER_view_and_proxies": "absent"},
    }
    try:
        points, labels, shards = load_shards(args.data_dir, "train")
        train_ids = full_training_ids(labels)
        if args.smoke:
            selection_ids = train_ids[:args.eval_batch_size * args.smoke_eval_batches]
            selection_points, selection_labels = points[selection_ids], labels[selection_ids]
            selection_shards, selection_prefix = [], "smoke_train"
        else:
            selection_points, selection_labels, selection_shards = load_shards(args.data_dir, "test")
            if len(selection_labels) != 2468:
                raise RuntimeError("Expected official test2468")
            selection_ids = np.arange(len(selection_labels), dtype=np.int64)
            selection_prefix = "test"
            manifest["official_test_read"] = True
        split_sha = split_identity(train_ids, selection_ids)
        manifest.update(split_sha256=split_sha, train_size=len(train_ids), validation_size=0,
                        selection_size=len(selection_ids), selection_domain=selection_prefix,
                        training_shards=shards, selection_shards=selection_shards,
                        selection_labels_sha256=hashlib.sha256(selection_labels.tobytes()).hexdigest(),
                        train_labels_sha256=hashlib.sha256(labels.tobytes()).hexdigest())
        seed_all(args.seed)
        backbone = configure_backbone(Hype_pointMLP()).to(device)
        freeze_curvature(backbone)
        initial_hash = state_hash(backbone)
        manifest["initialized_model_sha256"] = initial_hash
        model = DDP(HyCoReTrainingForward(backbone), device_ids=[local_rank], broadcast_buffers=True)
        model_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        optimizer = geoopt.optim.RiemannianSGD(model_parameters, lr=.1, momentum=.9, weight_decay=2e-4)
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=.005)
        dataset = PointClouds(points[train_ids], labels[train_ids], 1024, "train", args.seed)
        sampler = SourceClassBatchSampler(labels[train_ids], rank=rank, seed=args.seed, steps=args.steps_per_epoch)
        start_epoch, best, best_net = 0, None, None
        best_selection_metrics = None
        if args.resume is not None:
            saved = torch.load(args.resume, map_location="cpu", weights_only=False)
            validate_resume(saved, config, split_sha, train_ids, selection_ids, sampler)
            backbone.load_state_dict(saved["net"], strict=True)
            freeze_curvature(backbone)
            optimizer.load_state_dict(saved["optimizer"])
            scheduler.load_state_dict(saved["scheduler"])
            rank_state = saved["rank_states"][rank]
            sampler.load_state_dict(rank_state["sampler"])
            buffers = dict(backbone.named_buffers())
            for name, value in rank_state["BN_buffers"].items():
                buffers[name].copy_(value.to(buffers[name]))
            restore_rng(rank_state["rng"])
            start_epoch = int(saved["completed_epochs"])
            best, best_net = saved["best"], saved["best_net"]
            best_selection_metrics = saved["best_selection_metrics"]
            if best is None or best_net is None or best_selection_metrics is None:
                raise RuntimeError("Resume lacks the selected model and its matching metrics")
            if int(best["epoch"]) > start_epoch:
                raise RuntimeError("Selected model epoch exceeds completed training")
            metric_prefix = "smoke_train" if args.smoke else "test"
            if best[metric_prefix + "_oa"] != best_selection_metrics[metric_prefix + "_oa"]:
                raise RuntimeError("Resume best score and selected metrics differ")
            manifest["initialization"] = saved["initialization"]
            manifest["initialized_model_sha256"] = saved["initialized_model_sha256"]
            manifest["resume_identity"] = {"path": str(args.resume), "sha256": checkpoint_sha(args.resume),
                                            "completed_epochs": start_epoch, "source_commit": saved["commit"]}
            if rank == 0:
                # The selected model is embedded in every complete checkpoint,
                # so resume does not depend on a mutable old best.pth file.
                atomic_torch(args.run_dir / "best.pth", {
                    "format": CHECKPOINT_VERSION, "model_selection_only": True,
                    "net": best_net, "best": best, "epoch": best["epoch"],
                    "metrics": {"test": best_selection_metrics} if not args.smoke else {"smoke_training_selection": best_selection_metrics},
                    "training_config": config, "split_sha256": split_sha,
                    "resume_source": manifest["resume_identity"],
                })
            del saved
        if args.resume is None and rank ==0:
            atomic_torch(args.run_dir / "initialization_e0.pth", {
                "format": CHECKPOINT_VERSION, "model_selection_only": True,
                "initialization_only": True, "net": cpu_state(backbone),
                "training_config": config, "initialization": manifest["initialization"],
                "initialized_model_sha256": initial_hash, "commit": manifest["commit"]})
        devices = [None] * world_size
        dist.all_gather_object(devices, {
            "rank": rank, "local_rank": local_rank,
            "name": torch.cuda.get_device_name(device),
            "uuid": str(getattr(torch.cuda.get_device_properties(device), "uuid", "unavailable")),
            "total_MiB": torch.cuda.get_device_properties(device).total_memory / 1024 ** 2,
        })
        manifest["rank_devices"] = devices
        manifest["sampler_identity"] = sampler.state_dict()
        manifest.update(status="running", completed_epochs=start_epoch)
        if rank == 0:
            atomic_json(args.run_dir / "manifest.json", manifest)

        for epoch_index in range(start_epoch, args.epochs):
            epoch = epoch_index + 1
            active_hier, lambda_hier = False, 0.
            model.train()
            sampler.set_epoch(epoch_index)
            epoch_seed = args.seed + epoch_index * 1009 + rank * 1000003
            seed_all(epoch_seed)
            loader = DataLoader(dataset, batch_sampler=sampler, num_workers=args.workers,
                                worker_init_fn=worker_seed, pin_memory=True,
                                generator=torch.Generator().manual_seed(epoch_seed))
            telemetry = EpochTelemetry(train_ids, labels[train_ids], num_classes=40) if rank == 0 else None
            learning_rates = {"model": optimizer.param_groups[0]["lr"]}
            epoch_started = time.perf_counter()
            torch.cuda.reset_peak_memory_stats(device)
            for step, (cloud, local_gold, subset_ids) in enumerate(loader):
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
                values = control_base_losses(local_out, local_gold, args.control)
                ids = gather_without_grad(local_ids)
                loss = values["base"]
                if step == 0 and epoch in args.audit_epochs and rank == 0:
                    atomic_torch(args.run_dir / f"actual_training_epoch_{epoch:03d}.pth", {
                        "stage": "actual_training_pre_optimizer", "substage": "pre_backward",
                        "epoch": epoch, "step": step + 1, "completed_optimizer_steps": epoch_index * args.steps_per_epoch + step,
                        "commit": manifest["commit"], "training_config": config,
                        "rank_order": [0, 1], "ids": ids.detach().cpu(),
                        "labels": values["gold"].detach().cpu(),
                        "whole_count": whole_count, "child_count": child_count,
                        "native_c": 1., "hier_view_present": False, "proxy_parameter_count": 0,
                        "native_mu": values["mu"].detach().cpu(),
                        "native_nu": values["nu"].detach().cpu(),
                        "BN_counters_before_local_rank0": bn_before,
                        "BN_counters_after_local_rank0": [int(layer.num_batches_tracked) for layer in bn_layers],
                        "input_stats_local_rank0": local_out["input_stats"],
                    })
                synchronized_assert(bool(torch.isfinite(loss)), "Nonfinite objective", device)
                loss.backward()
                synchronized_assert(finite_tensors(parameter.grad for parameter in
                    list(model.parameters())), "Nonfinite gradients", device)
                bn_delta = [int(module.num_batches_tracked) - before
                            for module, before in zip(bn_layers, bn_before)]
                synchronized_assert(all(delta == 2 for delta in bn_delta), "Child/whole BN update must equal2", device)
                synchronized_assert(all(local_out["input_stats"][key] for key in
                    ("child_aliases_whole", "whole_aliases_source", "whole_changed_by_child")),
                    "Original part-overwrite behavior changed", device)
                preclip_norm = torch.nn.utils.clip_grad_norm_(model_parameters, 1.)
                postclip_norm = gradient_norm(model_parameters)
                synchronized_assert(bool(torch.isfinite(preclip_norm)) and bool(torch.isfinite(postclip_norm)),
                                    "Nonfinite clipped norm", device)
                optimizer.step()
                synchronized_assert(finite_tensors(model.parameters()), "Nonfinite updated parameters", device)
                torch.cuda.synchronize(device)
                elapsed = torch.tensor(time.perf_counter() - started, device=device)
                dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
                rank_peaks = [None] * world_size
                dist.all_gather_object(rank_peaks, {
                    "rank": rank, "peak_allocated_MiB": torch.cuda.max_memory_allocated(device) / 1024 ** 2,
                    "BN_update_min": min(bn_delta), "BN_update_max": max(bn_delta),
                    "input": local_out["input_stats"],
                })
                scalars = {
                    "loss": float(loss.detach()), "base": float(values["base"].detach()),
                    "ce": float(values["ce"].detach()), "lambda_hier": 0.,
                    "intra_contrastive": float(values["intra_contrastive"].detach()),
                    "intra_radial": float(values["intra_radial"].detach()),
                    "model_preclip_norm": float(preclip_norm), "model_postclip_norm": float(postclip_norm),
                    "model_clip_applied": float(preclip_norm >1.),
                    "model_clip_scale": min(1., 1. /(float(preclip_norm) +1e-6)),
                    "step_seconds_max_rank": float(elapsed),
                    "peak_allocated_MiB_max_rank": max(item["peak_allocated_MiB"] for item in rank_peaks),
                    "whole_count": whole_count, "child_count": child_count,
                    "whole_depth": float(values["whole_depth"]), "part_depth": float(values["part_depth"]),
                    "native_whole_norm_min": float(values["mu"].detach().norm(dim=-1).min()),
                    "flip_same_class_fraction": float((values["gold"] == values["gold"].flip(0)).float().mean()),
                    "hier_calls": 0., "proxy_parameter_count": 0.,
                }
                if rank == 0:
                    telemetry.update(ids=ids, labels=values["gold"], logits=values["logits"],
                        mu=values["mu"], nu=values["nu"], scalar_metrics=scalars,
                        hier_stats=None, sample_mining=None)
                    heartbeat = {"status": "running", "epoch": epoch, "step": step + 1,
                        "steps_per_epoch": args.steps_per_epoch, "phase": "joint" if active_hier else "base_only",
                        "updated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                        "lr": learning_rates, "scalars": scalars, "rank_observations": rank_peaks}
                    atomic_json(args.run_dir / "heartbeat.json", heartbeat)
                    append_jsonl(args.run_dir / "steps.jsonl", {"epoch": epoch, "step": step + 1,
                        "ids": ids.detach().cpu().tolist(), **scalars})
                    if step == 0 or (step + 1) % 20 == 0 or step + 1 == args.steps_per_epoch:
                        print(json.dumps({"epoch": epoch, "step": step + 1, "loss": scalars["loss"],
                            "phase": heartbeat["phase"], "lr": learning_rates}, allow_nan=False), flush=True)
                del local_out, values, loss
            scheduler.step()
            selection = rank0_evaluation(backbone, selection_points, selection_labels, args, device, selection_prefix, rank)
            clean_train = None
            if epoch % args.train_eval_every == 0 or epoch == args.epochs:
                clean_train = rank0_evaluation(backbone, points[train_ids], labels[train_ids], args, device, "train_eval", rank)
            candidate = select_best(best, selection, selection_prefix, epoch)
            improved = candidate is not best
            best = candidate
            if improved:
                best_selection_metrics = selection
            if improved and rank == 0:
                best_net = cpu_state(backbone)
            rank_states = [None] * world_size
            dist.all_gather_object(rank_states, checkpoint_rank_state(backbone, sampler, rank))
            checkpoint_error = None
            if rank == 0:
                row = {"epoch": epoch, "phase": "joint" if active_hier else "base_only",
                    "lambda_hier": lambda_hier, "lr_used": learning_rates,
                    "lr_next": {"model": optimizer.param_groups[0]["lr"]},
                    "epoch_wall_seconds": time.perf_counter() - epoch_started,
                    "telemetry": telemetry.summary(), "structure": {"available": False, "reason": "no HIER proxies"},
                    "selection_domain": selection_prefix,
                    "test": selection if not args.smoke else None,
                    "smoke_training_selection": selection if args.smoke else None,
                    "clean_train": clean_train, "best": best, "best_updated": improved}
                payload = {
                    "format": CHECKPOINT_VERSION, "model_selection_only": False,
                    "completed_epochs": epoch, "epoch": epoch, "net": backbone.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "rank_states": rank_states,
                    "training_config": config, "split_sha256": split_sha,
                    "train_ids": train_ids, "selection_ids": selection_ids,
                    "selection_domain": selection_prefix, "validation_size": 0,
                    "commit": manifest["commit"], "branch": manifest["branch"],
                    "initialization": manifest["initialization"],
                    "initialized_model_sha256": manifest["initialized_model_sha256"],
                    "best": best, "best_net": best_net,
                    "best_selection_metrics": best_selection_metrics, "metrics": row,
                    "saved_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                }
                try:
                    last_path = args.run_dir / "last.pth"
                    atomic_torch(last_path, payload)
                    digest = checkpoint_sha(last_path)
                    saved_files = ["last.pth"]
                    if improved:
                        atomic_copy(last_path, args.run_dir / "best.pth")
                        saved_files.append("best.pth")
                    if epoch % args.save_every == 0 or epoch in (1, 5, 99):
                        name = f"checkpoint_epoch_{epoch:03d}.pth"
                        atomic_copy(last_path, args.run_dir / name)
                        saved_files.append(name)
                    row["checkpoint"] = {"epoch": epoch, "files": saved_files,
                        "last_sha256": digest, "last_bytes": last_path.stat().st_size,
                        "saved_utc": payload["saved_utc"]}
                    atomic_json(args.run_dir / f"metrics_epoch_{epoch:03d}.json", row)
                    append_jsonl(args.run_dir / "metrics.jsonl", row)
                    manifest.update(status="running", completed_epochs=epoch, best=best,
                        latest_checkpoint=row["checkpoint"], updated_utc=dt.datetime.now(dt.timezone.utc).isoformat())
                    atomic_json(args.run_dir / "manifest.json", manifest)
                    print(json.dumps({"epoch": epoch, "selection_domain": selection_prefix,
                        "selection_oa": selection[selection_prefix + "_oa"], "best": best},
                        allow_nan=False), flush=True)
                except BaseException as exc:
                    checkpoint_error = repr(exc)
            error = [checkpoint_error]
            dist.broadcast_object_list(error, src=0)
            if error[0] is not None:
                raise RuntimeError("Epoch checkpoint failed: " + error[0])
            dist.barrier()

        test_metrics, final_error = None, None
        dist.barrier()
        if rank == 0 and not args.smoke:
            try:
                selected = torch.load(args.run_dir / "best.pth", map_location="cpu", weights_only=False)
                if selected["epoch"] != best["epoch"] or selected["best"] != best:
                    raise RuntimeError("Selected checkpoint epoch/identity differs from best")
                test_metrics = selected["metrics"]["test"]
                if test_metrics["test_oa"] != best["test_oa"]:
                    raise RuntimeError("Selected checkpoint metrics differ from best score")
                if test_metrics["test_count"] != 2468:
                    raise RuntimeError("Selected checkpoint lacks complete official test metrics")
                atomic_json(args.run_dir / "final_test.json", {
                    "selected_by": "test OA strictly higher; first tie retained",
                    "test_used_for_selection": True, "fresh_final_forward": False, "best": best,
                    "best_checkpoint_sha256": checkpoint_sha(args.run_dir / "best.pth"),
                    "test_shards": selection_shards, **test_metrics})
            except BaseException as exc:
                final_error = repr(exc)
        error = [final_error]
        dist.broadcast_object_list(error, src=0)
        if error[0] is not None:
            raise RuntimeError("Final test failed: " + error[0])
        dist.barrier()
        dist.destroy_process_group()
        if rank == 0:
            manifest.update(status="completed", completed_epochs=args.epochs,
                finished_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                wall_seconds=time.perf_counter() - started_wall, best=best,
                final_test=test_metrics, final_summary_skipped=args.smoke, test_used_for_selection=not args.smoke)
            atomic_json(args.run_dir / "manifest.json", manifest)
            atomic_json(args.run_dir / "heartbeat.json", {"status": "completed",
                "epoch": args.epochs, "updated_utc": manifest["finished_utc"], "best": best})
            print("V9_FULL9840_TESTSELECT_B64_CONTROL_COMPLETED", flush=True)
    except BaseException as exc:
        if rank == 0:
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
