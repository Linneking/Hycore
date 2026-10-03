"""Single-GPU B32 HyCoRe stability diagnostic, not the matched global64 B0.

Every nominal epoch shuffles the full training subset and takes only its
first 200 x 32 rows. This budget cap does not traverse the whole subset.
The released whole/part aliasing, training BN, fixed FPS and local child
flip are retained. Same-class flip negatives are measured, not repaired.
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
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "pointnet2_ops_lib"))
sys.path.insert(0, str(REPO / "inter_hierarchy_MN40"))
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import geoopt
import numpy as np
import torch
from torch.nn import functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Sampler

from models.pointmlp import Hype_pointMLP
from hier_proxy_v2_online.train import PointClouds, load_shards, seed_all
from hier_proxy_scratch_v3.train import (
    atomic_json, atomic_torch, capture_rng, freeze_curvature,
    git_checkout_state, restore_rng,
)
from hier_proxy_scratch_v4.train import split_identity, state_hash, stratified_split
from .base_protocol import HyCoReTrainingForward, intra_losses, smoothed_ce
from .distributed import shared_crop_counts
from .telemetry import EpochTelemetry


PROTOCOL = "v5_hycore_b32_random_permutation_budget_cap"
SCHEMA = 1


def utc_now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_args(args):
    return {key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()}


class BudgetPermutationSampler(Sampler):
    """Private epoch RNG, no replacement and no class balancing."""

    def __init__(self, size, steps=200, seed=22, batch_size=32):
        self.size, self.steps = int(size), int(steps)
        self.seed, self.batch_size, self.epoch = int(seed), int(batch_size), 0
        if min(self.size, self.steps, self.batch_size) < 1:
            raise ValueError("sampler sizes must be positive")
        if self.steps * self.batch_size > self.size:
            raise ValueError("budget exceeds available distinct training instances")

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def indices(self):
        generator = torch.Generator().manual_seed(self.seed + 1009 * self.epoch)
        return torch.randperm(self.size, generator=generator)[:len(self)].tolist()

    def __iter__(self):
        return iter(self.indices())

    def __len__(self):
        return self.steps * self.batch_size

    def state_dict(self):
        return {"kind": "torch_randperm_private_epoch_rng_v1", "size": self.size,
                "steps": self.steps, "batch_size": self.batch_size,
                "seed": self.seed, "epoch": self.epoch,
                "seed_formula": "seed + 1009 * epoch", "replacement": False,
                "class_balance": False, "first_rows_only": len(self)}


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--resume", type=Path,
                        help="Complete epoch-boundary B32 checkpoint; output must be a NEW directory")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--steps-per-epoch", type=int, default=200)
    parser.add_argument("--seed", type=int, default=22)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--train-eval-every", type=int, default=10)
    parser.add_argument("--audit-epochs", default="21,40,100,160,200")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-eval-batches", type=int, default=2,
                        help="Bounded clean-eval batches in smoke only; production always evaluates all")
    parser.add_argument("--skip-final-test", action="store_true", help="Smoke only")
    args = parser.parse_args()
    try:
        args.audit_epochs = sorted(set(int(item) for item in args.audit_epochs.split(",") if item.strip()))
    except ValueError:
        parser.error("--audit-epochs must be comma-separated positive epoch numbers")
    if min(args.epochs, args.steps_per_epoch, args.eval_batch_size,
           args.train_eval_every, args.smoke_eval_batches) < 1 or args.workers < 0:
        parser.error("epochs, steps and evaluation sizes must be positive; workers nonnegative")
    if any(epoch < 1 for epoch in args.audit_epochs):
        parser.error("audit epoch numbers must be positive")
    if not args.smoke and (args.epochs != 200 or args.steps_per_epoch != 200):
        parser.error("Authorized production budget is exactly200 epochs x200 steps; use --smoke for bounded checks")
    if args.smoke and (args.epochs > 2 or args.steps_per_epoch > 6):
        parser.error("Smoke is bounded to at most2 epochs and6 steps per epoch")
    if args.skip_final_test and not args.smoke:
        parser.error("--skip-final-test is only allowed with --smoke")
    args.data_dir = args.data_dir.resolve(strict=True)
    args.run_dir = args.run_dir.resolve()
    if args.run_dir.exists():
        parser.error("--run-dir must be new, including resumed runs")
    if args.resume:
        args.resume = args.resume.resolve(strict=True)
    return args


def _finite(name, value):
    if not bool(torch.isfinite(value).all()):
        raise FloatingPointError("Nonfinite " + name)


def _comparison(a, b):
    denominator = a.norm() * b.norm()
    return {"norm_ratio": float(a.norm() / b.norm()) if float(b.norm()) else None,
            "cosine": float(torch.dot(a, b) / denominator) if float(denominator) else None}


def parameter_audit(backbone, ce, weighted_intra):
    """Read CE/intra gradients on the working graph; no second BN forward."""
    named = [(name, value) for name, value in backbone.named_parameters() if value.requires_grad]
    parameters = [value for _, value in named]
    vectors = {}
    for term, loss in (("ce", ce), ("weighted_intra", weighted_intra)):
        gradients = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
        vector = torch.cat([(torch.zeros_like(value) if grad is None else grad.detach()).reshape(-1)
                            for value, grad in zip(parameters, gradients)])
        _finite(term + " parameter audit", vector)
        vectors[term] = vector
    groups = {"all_model": [], "shared_encoder": [], "euclidean_features": [],
              "mobius_embedding": [], "classifier": []}
    offset = 0
    for name, value in named:
        indices = torch.arange(offset, offset + value.numel(), device=value.device)
        groups["all_model"].append(indices)
        if name.startswith("classifier."):
            groups["classifier"].append(indices)
        else:
            groups["shared_encoder"].append(indices)
            groups["mobius_embedding" if name.startswith("emb.") else "euclidean_features"].append(indices)
        offset += value.numel()
    report = {"semantics": "Single-card local32 full parameter gradients; working graph, no extra forward or optimizer step",
              "hierarchy": None, "BN_extra_updates": 0, "groups": {}}
    for group, pieces in groups.items():
        if not pieces:
            continue
        indices = torch.cat(pieces)
        ce_vector, intra = vectors["ce"][indices], vectors["weighted_intra"][indices]
        report["groups"][group] = {"parameter_elements": len(indices),
            "ce_norm": float(ce_vector.norm()), "weighted_intra_norm": float(intra.norm()),
            "intra_vs_ce": _comparison(intra, ce_vector),
            "base_norm": float((ce_vector + intra).norm()), "hierarchy": None}
    return report


@torch.no_grad()
def evaluate(backbone, dataset, args, device, phase):
    """Deterministic no-augmentation evaluation; no child forward or BN update."""
    saved_rng = capture_rng()
    previous_mode = backbone.training
    backbone.eval()
    confusion = torch.zeros((40, 40), dtype=torch.int64)
    class_loss = torch.zeros(40, dtype=torch.float64)
    count, loss_sum = 0, 0.0
    bn_before = [value.num_batches_tracked.clone() for value in backbone.modules()
                 if isinstance(value, torch.nn.modules.batchnorm._BatchNorm)]
    try:
        generator = torch.Generator().manual_seed(args.seed + 7717)
        loader = DataLoader(dataset, batch_size=args.eval_batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True, generator=generator)
        for batch_index, (cloud, labels, _) in enumerate(loader):
            if args.smoke and batch_index >= args.smoke_eval_batches:
                break
            labels_gpu = labels.to(device, non_blocking=True)
            mu, logits = backbone(cloud.to(device, non_blocking=True).transpose(1, 2).contiguous())
            _finite(phase + " mu", mu); _finite(phase + " logits", logits)
            logp = F.log_softmax(logits, dim=-1)
            target = torch.full_like(logits, .2 / (logits.shape[1] - 1))
            target.scatter_(1, labels_gpu[:, None], .8)
            per_row_ce = -(target * logp).sum(-1)
            pred = logits.argmax(-1).cpu()
            confusion += torch.bincount(labels * 40 + pred, minlength=1600).reshape(40, 40)
            class_loss.index_add_(0, labels, per_row_ce.double().cpu())
            count += len(labels); loss_sum += float(per_row_ce.sum())
        support = confusion.sum(1)
        present = support > 0
        accuracy = confusion.diagonal().double() / support.clamp_min(1)
        report = {"phase": phase, "examples": count, "expected_examples": len(dataset),
            "complete_dataset": count == len(dataset), "smoke_only": args.smoke,
            "ce": loss_sum / count, "oa": 100 * float(confusion.diagonal().sum()) / count,
            "aa": 100 * float(accuracy[present].mean()),
            "class_count": support.tolist(), "class_accuracy": (100 * accuracy).tolist(),
            "class_ce": [float(class_loss[i] / support[i]) if support[i] else None for i in range(40)],
            "confusion": confusion.tolist(), "augmentation": False, "BN_updated": False}
        bn_after = [value.num_batches_tracked for value in backbone.modules()
                    if isinstance(value, torch.nn.modules.batchnorm._BatchNorm)]
        if any(not torch.equal(before, after) for before, after in zip(bn_before, bn_after)):
            raise RuntimeError("Clean evaluation modified BN counters")
        return report
    finally:
        backbone.train(previous_mode)
        restore_rng(saved_rng)


def checkpoint_state(backbone, optimizer, scheduler, args, epoch, split_sha,
                     train_ids, validation_ids, sampler, best, best_net, initial_sha):
    return {"schema": SCHEMA, "protocol": PROTOCOL, "epoch": epoch,
        "net": backbone.state_dict(), "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(), "rng": capture_rng(),
        "config": json_args(args), "split_sha256": split_sha,
        "train_ids": train_ids, "validation_ids": validation_ids,
        "sampler_identity": sampler.state_dict(), "best": best, "best_net": best_net,
        "initial_model_sha256": initial_sha, "source_run_dir": str(args.run_dir),
        "checkpoint_boundary": "after full nominal epoch, validation and scheduler.step",
        **git_checkout_state()}


def validate_resume(state, args, split_sha, train_ids, validation_ids):
    if state.get("schema") != SCHEMA or state.get("protocol") != PROTOCOL:
        raise ValueError("Resume requires this B32 protocol's complete checkpoint")
    if state.get("selection_only") or any(key not in state for key in
            ("net", "optimizer", "scheduler", "rng", "best_net")):
        raise ValueError("Selection-only checkpoint cannot resume training; use last or periodic checkpoint")
    if state.get("split_sha256") != split_sha:
        raise ValueError("Resume split identity mismatch")
    for key, expected in (("train_ids", train_ids), ("validation_ids", validation_ids)):
        if not np.array_equal(np.asarray(state[key]), expected):
            raise ValueError("Resume " + key + " mismatch")
    old = state["config"]
    for key in ("epochs", "steps_per_epoch", "seed", "workers", "eval_batch_size",
                "train_eval_every", "audit_epochs", "smoke", "smoke_eval_batches", "skip_final_test"):
        if old.get(key) != json_args(args)[key]:
            raise ValueError("Resume config mismatch: " + key)
    if not 0 < int(state["epoch"]) < args.epochs:
        raise ValueError("Resume checkpoint must have at least one unfinished epoch")
    sampler = state.get("sampler_identity", {})
    if sampler.get("kind") != "torch_randperm_private_epoch_rng_v1" or sampler.get("batch_size") != 32:
        raise ValueError("Resume sampler identity mismatch")


def main():
    args = arguments()
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise RuntimeError("B32 diagnostic is single-card only")
    if not torch.cuda.is_available():
        raise RuntimeError("B32 training requires CUDA")
    torch.set_num_threads(2)
    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)
    args.run_dir.mkdir(parents=True, exist_ok=False)
    manifest = {"status": "starting", "protocol": PROTOCOL, "main_training": not args.smoke,
        "stability_diagnostic": True, "matched_global64_HIER_control": False,
        "started_utc": utc_now(), "pid": os.getpid(), "command_argv": [sys.executable, *sys.argv],
        "config": json_args(args), **git_checkout_state(),
        "visible_gpus": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "physical_gpus": os.environ.get('HYCORE_PHYSICAL_GPUS', os.environ.get('CUDA_VISIBLE_DEVICES')),
        "physical_gpu_uuids": os.environ.get('HYCORE_PHYSICAL_GPU_UUIDS'),
        "gpu": {"logical_index": 0, "name": torch.cuda.get_device_name(0),
                "total_memory_bytes": torch.cuda.get_device_properties(0).total_memory},
        "runtime": {"python": sys.version, "torch": torch.__version__,
                    "geoopt": geoopt.__version__, "numpy": np.__version__},
        "teacher_identity": None, "historical_checkpoint_loaded": False,
        "fixed": {"c": 1, "D": 256, "batch_size": 32, "BN_batch": 32,
            "CE_epsilon": .2, "contr_alpha": .01, "radial_alpha": .01,
            "contr_margin": 4, "radial_margin": "1000 / child_points",
            "whole_points": [800, 1024], "child_points": [200, 600],
            "part_overwrites_whole": True, "BN_updates_per_step": 2, "FPS_first_centers": 512,
            "negative": "local32 child.flip(0), unchanged same-class collisions allowed",
            "optimizer": "RiemannianSGD", "lr": .1, "min_lr": .005,
            "momentum": .9, "weight_decay": 2e-4, "model_clip_norm": 1,
            "hierarchy": None, "precision": "float32",
            "sampler": "ordinary epoch permutation, first steps*32, no replacement or class balance",
            "crop_count_adaptation": "Private seed/epoch/step inclusive uniforms via shared_crop_counts; crop-center stream separately seeded per step"},
        "test_read": False, "source_checkpoint_identity": None}
    atomic_json(args.run_dir / "manifest.json", manifest)
    started = time.monotonic()
    try:
        points, labels, shards = load_shards(args.data_dir, "train")
        train_ids, validation_ids = stratified_split(labels, args.seed, .1)
        split_sha = split_identity(train_ids, validation_ids)
        if len(train_ids) != 8856 or len(validation_ids) != 984:
            raise ValueError("Expected the fixed ModelNet40 8856/984 split")
        seed_all(args.seed)
        backbone = Hype_pointMLP().to(device)
        freeze_curvature(backbone)
        initial_sha = state_hash(backbone)
        model = HyCoReTrainingForward(backbone)
        optimizer = geoopt.optim.RiemannianSGD(
            [value for value in backbone.parameters() if value.requires_grad],
            lr=.1, momentum=.9, weight_decay=2e-4)
        scheduler = CosineAnnealingLR(optimizer, args.epochs, eta_min=.005)
        train_labels = labels[train_ids]
        train_data = PointClouds(points[train_ids], train_labels, 1024, "train", args.seed)
        val_data = PointClouds(points[validation_ids], labels[validation_ids], 1024, "eval", args.seed)
        clean_train = PointClouds(points[train_ids], train_labels, 1024, "eval", args.seed)
        sampler = BudgetPermutationSampler(len(train_ids), args.steps_per_epoch, args.seed)
        best = {"epoch": None, "oa": -1.0, "ce": None}
        best_net = None
        start_epoch = 0
        if args.resume:
            state = torch.load(args.resume, map_location="cpu", weights_only=False)
            validate_resume(state, args, split_sha, train_ids, validation_ids)
            if state["initial_model_sha256"] != initial_sha:
                raise ValueError("Resume initial-model identity mismatch")
            backbone.load_state_dict(state["net"], strict=True)
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
            best = state["best"]; start_epoch = int(state["epoch"])
            best_net = state["best_net"]
            # Use the best state embedded at this boundary, never a later
            # source-run best file. This supports old periodic checkpoints.
            atomic_torch(args.run_dir / "best_checkpoint.pth", {
                "schema": SCHEMA, "protocol": PROTOCOL, "selection_only": True,
                "epoch": best["epoch"], "best": best, "net": best_net,
                "source_resume_sha256": file_sha256(args.resume)})
            manifest["source_checkpoint_identity"] = {"path": str(args.resume),
                "sha256": file_sha256(args.resume), "completed_epoch": start_epoch,
                "embedded_best_epoch": best["epoch"], "role": "complete same-protocol B32 resume"}
            restore_rng(state["rng"])
            del state
        manifest.update(status="running", split_sha256=split_sha, train_size=len(train_ids),
            validation_size=len(validation_ids), training_shards=shards,
            initial_model_sha256=initial_sha, sampler_identity=sampler.state_dict(),
            nominal_epoch_draws=len(sampler), nominal_epoch_unique_coverage=len(sampler) / len(train_ids),
            omitted_training_rows_per_epoch=len(train_ids) - len(sampler),
            source_model_identity="random_seed" + str(args.seed), start_epoch=start_epoch)
        atomic_json(args.run_dir / "manifest.json", manifest)
        bn_layers = [value for value in backbone.modules()
                     if isinstance(value, torch.nn.modules.batchnorm._BatchNorm)]
        torch.cuda.reset_peak_memory_stats(device)
        with (args.run_dir / "steps.jsonl").open("x", encoding="utf-8", buffering=1) as steps_stream:
            for epoch in range(start_epoch, args.epochs):
                epoch_start = time.monotonic()
                torch.cuda.reset_peak_memory_stats(device)
                seed_all(args.seed + 1009 * epoch)
                sampler.set_epoch(epoch)
                generator = torch.Generator().manual_seed(args.seed + 1009 * epoch + 19)
                loader = DataLoader(train_data, sampler=sampler, batch_size=32, drop_last=True,
                    num_workers=args.workers, generator=generator, pin_memory=True)
                telemetry = EpochTelemetry(train_ids, train_labels, num_classes=40)
                model.train()
                lr_used = optimizer.param_groups[0]["lr"]
                completed_steps = 0
                for step, (cloud, gold, subset_ids) in enumerate(loader):
                    step_start = time.monotonic()
                    seed_all(args.seed + epoch * 1000003 + step * 1009)
                    whole_count, child_count = shared_crop_counts(args.seed, epoch, step)
                    optimizer.zero_grad(set_to_none=True)
                    bn_before = [value.num_batches_tracked.clone() for value in bn_layers]
                    output = model(cloud.to(device, non_blocking=True), whole_count, child_count)
                    gold_gpu = gold.to(device, non_blocking=True)
                    ce = smoothed_ce(output["logits"], gold_gpu)
                    intra = intra_losses(output["mu"], output["nu"], child_count)
                    weighted_intra = .01 * (intra["intra_contrastive"] + intra["intra_radial"])
                    loss = ce + weighted_intra
                    for name, value in (("loss", loss), ("mu", output["mu"]),
                                        ("nu", output["nu"]), ("logits", output["logits"])):
                        _finite(name, value)
                    audit = None
                    if step == 0 and epoch + 1 in args.audit_epochs:
                        audit = parameter_audit(backbone, ce, weighted_intra)
                    bn_deltas = [int(value.num_batches_tracked - before)
                                 for value, before in zip(bn_layers, bn_before)]
                    if not bn_deltas or any(delta != 2 for delta in bn_deltas):
                        raise RuntimeError("Every training BN must update exactly twice")
                    if not (output["input_stats"]["child_aliases_whole"] and
                            output["input_stats"]["whole_aliases_source"]):
                        raise RuntimeError("Original part/whole alias was lost")
                    loss.backward()
                    preclip = torch.nn.utils.clip_grad_norm_(backbone.parameters(), 1.0)
                    _finite("model gradient norm", preclip)
                    postclip = torch.sqrt(sum(value.grad.detach().square().sum()
                        for value in backbone.parameters() if value.grad is not None))
                    _finite("postclip gradient norm", postclip)
                    optimizer.step()
                    if not bool(torch.stack([torch.isfinite(value).all()
                                             for value in backbone.parameters()]).all()):
                        raise FloatingPointError("Nonfinite updated model parameter")
                    collisions = int((gold == gold.flip(0)).sum())
                    scalar = {"loss": float(loss.detach()), "ce": float(ce.detach()),
                        "weighted_intra": float(weighted_intra.detach()),
                        **{key: float(value.detach()) for key, value in intra.items()},
                        "model_preclip_norm": float(preclip), "model_postclip_norm": float(postclip),
                        "model_clip_triggered": int(float(preclip) > 1.0),
                        "negative_class_collisions": collisions,
                        "negative_class_collision_fraction": collisions / 32,
                        "lr": lr_used, "step_seconds": time.monotonic() - step_start}
                    global_ids = train_ids[subset_ids.numpy()]
                    telemetry.update(ids=global_ids, labels=gold, logits=output["logits"],
                        mu=output["mu"], nu=output["nu"], scalar_metrics=scalar)
                    row = {"epoch": epoch + 1, "step": step + 1, "time_utc": utc_now(),
                        "ids": global_ids.tolist(), "metrics": scalar, "input": output["input_stats"],
                        "BN_update_deltas": bn_deltas, "parameter_gradient": audit,
                        "hierarchy": None, "peak_allocated_MiB": torch.cuda.max_memory_allocated(device) / 2**20}
                    steps_stream.write(json.dumps(row, allow_nan=False) + "\n")
                    atomic_json(args.run_dir / "heartbeat.json", {"status": "training",
                        "epoch": epoch + 1, "step": step + 1, "time_utc": utc_now(),
                        "pid": os.getpid(), "loss": scalar["loss"]})
                    completed_steps += 1
                    del output, intra, weighted_intra, ce, loss, audit
                if completed_steps != args.steps_per_epoch:
                    raise RuntimeError("Nominal B32 epoch did not complete its exact step budget")
                val = evaluate(backbone, val_data, args, device, "validation")
                clean = evaluate(backbone, clean_train, args, device, "train_clean") if (
                    (epoch + 1) % args.train_eval_every == 0) else None
                is_best = val["oa"] > best["oa"] or (
                    val["oa"] == best["oa"] and (best["ce"] is None or val["ce"] < best["ce"]))
                if is_best:
                    best = {"epoch": epoch + 1, "oa": val["oa"], "ce": val["ce"], "aa": val["aa"]}
                    best_net = {key: value.detach().cpu().clone()
                                for key, value in backbone.state_dict().items()}
                scheduler.step()
                epoch_report = {"epoch": epoch + 1, "protocol": PROTOCOL,
                    "sampler": sampler.state_dict(), "training": telemetry.summary(),
                    "validation": val, "train_clean": clean, "best": best,
                    "lr_used": lr_used, "lr_next": optimizer.param_groups[0]["lr"],
                    "epoch_seconds": time.monotonic() - epoch_start,
                    "peak_allocated_MiB": torch.cuda.max_memory_allocated(device) / 2**20,
                    "test_read": False, "smoke_only": args.smoke}
                state = checkpoint_state(backbone, optimizer, scheduler, args, epoch + 1,
                    split_sha, train_ids, validation_ids, sampler, best, best_net, initial_sha)
                checkpoint_paths = [args.run_dir / "last_checkpoint.pth"]
                if is_best:
                    checkpoint_paths.append(args.run_dir / "best_checkpoint.pth")
                if (epoch + 1) % 20 == 0 or args.smoke:
                    checkpoint_paths.append(args.run_dir / f"checkpoint_epoch_{epoch + 1:03d}.pth")
                checkpoint_records = []
                for checkpoint_path in checkpoint_paths:
                    atomic_torch(checkpoint_path, state)
                    checkpoint_records.append({"epoch": epoch + 1, "path": str(checkpoint_path),
                        "size_bytes": checkpoint_path.stat().st_size,
                        "sha256": file_sha256(checkpoint_path), "written_utc": utc_now()})
                epoch_report["checkpoints"] = checkpoint_records
                atomic_json(args.run_dir / f"metrics_epoch_{epoch + 1:03d}.json", epoch_report)
                del state
                manifest.update(completed_epochs=epoch + 1, best=best, updated_utc=utc_now(),
                                latest_checkpoint_records=checkpoint_records)
                atomic_json(args.run_dir / "manifest.json", manifest)
                atomic_json(args.run_dir / "heartbeat.json", {"status": "epoch_completed",
                    "epoch": epoch + 1, "step": args.steps_per_epoch, "time_utc": utc_now(),
                    "pid": os.getpid(), "validation_oa": val["oa"]})
                print(json.dumps({"epoch": epoch + 1,
                    "train_oa": epoch_report['training']['training']['oa'],
                    "train_aa": epoch_report['training']['training']['aa_observed_classes'],
                    "validation_oa": val["oa"], "validation_ce": val["ce"], "best": best}), flush=True)
        if not args.skip_final_test:
            selected = torch.load(args.run_dir / "best_checkpoint.pth", map_location="cpu", weights_only=False)
            backbone.load_state_dict(selected["net"], strict=True)
            test_points, test_labels, test_shards = load_shards(args.data_dir, "test")
            test_data = PointClouds(test_points, test_labels, 1024, "eval", args.seed)
            final = evaluate(backbone, test_data, args, device, "test")
            final.update(selected_epoch=selected["epoch"], checkpoint_sha256=file_sha256(
                args.run_dir / "best_checkpoint.pth"), test_shards=test_shards,
                selection="validation OA, then lower validation CE; official test read once")
            atomic_json(args.run_dir / "final_test.json", final)
            manifest["test_read"] = True
        manifest.update(status="completed", finished_utc=utc_now(),
            wall_seconds=time.monotonic() - started, best=best,
            peak_allocated_MiB=torch.cuda.max_memory_allocated(device) / 2**20)
        atomic_json(args.run_dir / "manifest.json", manifest)
        atomic_json(args.run_dir / "heartbeat.json", {"status": "completed", "time_utc": utc_now(),
            "epoch": args.epochs, "pid": os.getpid()})
    except BaseException as error:
        manifest.update(status="failed", failed_utc=utc_now(), error=repr(error),
            wall_seconds=time.monotonic() - started)
        atomic_json(args.run_dir / "manifest.json", manifest)
        atomic_json(args.run_dir / "heartbeat.json", {"status": "failed", "time_utc": utc_now(),
            "pid": os.getpid(), "error": repr(error)})
        raise


if __name__ == "__main__":
    main()
