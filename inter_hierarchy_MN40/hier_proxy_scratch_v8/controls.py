"""V8 B64 and B0-32 controls: identical V8 geometry; no HIER or proxy updates.

Validation selects; official test is evaluated once after300epochs.
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
    REPO, Hype_pointMLP, HyCoReTrainingForward, PointClouds,
    SourceClassBatchSampler, checkpoint_sha,
    gather_without_grad, load_shards, seed_all, shared_crop_counts,
    split_identity, stratified_split,
)
from .base_protocol import global_base_losses, smoothed_ce, intra_losses
from ..hier_proxy_scratch_v5.distributed import broadcast_parameters, sync_parameter_gradients
from .geometry import configure_backbone, freeze_curvature, proxy_update_stats
from .telemetry import EpochTelemetry
from hier_proxy_scratch_v3.train import atomic_json, atomic_torch, capture_rng, restore_rng
from hier_proxy_scratch_v4.train import state_hash

import geoopt
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader


CHECKPOINT_VERSION = "hycore-v8-c01-controls-1"


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", choices=("B64", "B0-32"), required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--seed", type=int, default=22)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--steps-per-epoch", type=int)
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
    if args.steps_per_epoch is None:
        args.steps_per_epoch = 200 if args.control == "B64" else 276
    args.audit_epochs = tuple(sorted(set(int(item) for item in args.audit_epochs.split(",") if item.strip())))
    if args.workers < 0 or args.eval_batch_size < 1 or args.train_eval_every < 1 or args.save_every < 1:
        parser.error("workers must be nonnegative; evaluation/save intervals positive")
    if not (args.warmup_epochs == 0 and math.isfinite(args.lambda_hier) and args.lambda_hier == 0.):
        parser.error("V8 controls require W0 and HIER weight zero")
    if args.smoke:
        if not (1 <= args.epochs <= 2 and 1 <= args.steps_per_epoch <= 3 and
                0 <= args.warmup_epochs < args.epochs and 1 <= args.smoke_eval_batches <= 2):
            parser.error("Smoke bounds: epochs<=2, steps<=3, warmup<epochs, eval batches<=2")
        if not args.skip_final_test:
            parser.error("Smoke runs must use --skip-final-test")
    elif (args.epochs, args.steps_per_epoch, args.warmup_epochs, args.seed) != (300, 200 if args.control == "B64" else 276, 0, 22):
        parser.error("Controls require seed22, 300epochs, B64:200/B0-32:276steps, W0")
    if any(epoch < 1 for epoch in args.audit_epochs):
        parser.error("Audit epochs use one-based positive epoch numbers")
    return args


def fixed_configuration(args):
    from .train import fixed_configuration as main_configuration
    config = main_configuration(args)
    b64 = args.control == "B64"
    config.update(control=args.control, canonical_name="V8-" + args.control + ("-C01" if not b64 else ""),
        global_batch=64 if b64 else 32, world_size=2 if b64 else 1,
        classes_per_rank=16 if b64 else None, instances_per_class=2 if b64 else None,
        sampler=config["sampler"] if b64 else "shuffle without replacement; drop_last; local32",
        lambda_hier_after_warmup=0., P=0, sample_K=None, proxy_K=None, T=None,
        proxy_optimizer=None, proxy_initialization=None, proxy_parameter_projection=None,
        HIER_objective_enabled=False, HIER_calls_per_step=0,
        schedule="model 300-epoch cosine; no proxy optimizer",
        negative_flip_same_class="forbidden by class-disjoint B64" if b64 else "allowed as original HyCoRe; measured")
    return config


class ShuffleBatchSampler:
    """A complete reproducible shuffled epoch, without repeating leftover IDs."""
    def __init__(self, size, seed, steps=None, rank=0):
        if rank !=0 or size <32:
            raise ValueError("B0-32 requires rank0 and at least32 objects")
        self.size, self.seed, self.epoch = int(size), int(seed), 0
        self.steps = self.size //32 if steps is None else int(steps)
        if not 1 <= self.steps <= self.size //32:
            raise ValueError("invalid shuffle epoch length")
    def set_epoch(self, epoch):
        self.epoch = int(epoch)
    def __len__(self):
        return self.steps
    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch *1009)
        ids = torch.randperm(self.size, generator=generator).tolist()
        for step in range(self.steps):
            yield ids[step *32:(step+1) *32]
    def state_dict(self):
        return dict(kind="shuffle32", size=self.size, seed=self.seed, epoch=self.epoch, steps=self.steps)
    def load_state_dict(self, state):
        expected = dict(kind="shuffle32", size=self.size, seed=self.seed, epoch=state["epoch"], steps=self.steps)
        if state != expected:
            raise RuntimeError("Shuffle sampler identity differs")
        self.epoch = int(state["epoch"])


def control_base_losses(outputs, labels, control):
    if control == "B64":
        return global_base_losses(outputs, labels)
    if control != "B0-32" or len(labels) !=32:
        raise ValueError("B0-32 requires exactly32 rows")
    mu, nu, logits = outputs["mu"], outputs["nu"], outputs["logits"]
    values = intra_losses(mu, nu, int(outputs["child_count"]))
    ce = smoothed_ce(logits, labels)
    return dict(mu=mu, nu=nu, logits=logits, gold=labels, ce=ce,
        base=ce + .01 *values["intra_contrastive"] + .01 *values["intra_radial"], **values)


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
        try:
            result = evaluate_clean(backbone, points, labels, args, device, prefix)
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


def validate_resume(saved, config, split_sha, train_ids, val_ids, sampler):
    if saved.get("format") != CHECKPOINT_VERSION or saved.get("model_selection_only", False):
        raise RuntimeError("Resume requires a complete V8 training checkpoint")
    if saved.get("training_config") != config or saved.get("split_sha256") != split_sha:
        raise RuntimeError("Resume configuration or split differs from checkpoint")
    if not np.array_equal(saved["train_ids"], train_ids) or not np.array_equal(saved["validation_ids"], val_ids):
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
    world_size = 2 if args.control == "B64" else 1
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
        "commit": (os.environ.get("V8_SOURCE_COMMIT") or subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()),
        "branch": (os.environ.get("V8_SOURCE_BRANCH") or subprocess.check_output(["git", "branch", "--show-current"], cwd=REPO, text=True).strip()),
        "command_argv": [sys.executable, *sys.argv],
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "canonical_name": config["canonical_name"], "training_config": config, "visible_gpus": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "physical_gpus": os.environ.get('HYCORE_PHYSICAL_GPUS', os.environ.get('CUDA_VISIBLE_DEVICES')),
        "physical_gpu_uuids": os.environ.get('HYCORE_PHYSICAL_GPU_UUIDS'),
        "runtime": {"torch": torch.__version__, "geoopt": geoopt.__version__, "numpy": np.__version__},
        "initialization": {"method": "random", "seed": args.seed, "source_checkpoint": None, "teacher": None},
        "official_test_read": False, "resume_identity": None,
    }
    try:
        points, labels, shards = load_shards(args.data_dir, "train")
        train_ids, val_ids = stratified_split(labels, args.seed, .1)
        split_sha = split_identity(train_ids, val_ids)
        manifest.update(split_sha256=split_sha, train_size=len(train_ids), validation_size=len(val_ids),
                        training_shards=shards, train_labels_sha256=hashlib.sha256(labels.tobytes()).hexdigest())
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
        sampler = (SourceClassBatchSampler(labels[train_ids], rank=rank, seed=args.seed, steps=args.steps_per_epoch)
                   if args.control == "B64" else ShuffleBatchSampler(len(train_ids), args.seed, args.steps_per_epoch, rank))
        start_epoch, best, best_net = 0, None, None
        if args.resume is not None:
            saved = torch.load(args.resume, map_location="cpu", weights_only=False)
            validate_resume(saved, config, split_sha, train_ids, val_ids, sampler)
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
            validation = rank0_evaluation(backbone, points[val_ids], labels[val_ids], args, device, "val", rank)
            clean_train = None
            if epoch % args.train_eval_every == 0 or epoch == args.epochs:
                clean_train = rank0_evaluation(backbone, points[train_ids], labels[train_ids], args, device, "train_eval", rank)
            improved = best is None or validation["val_oa"] > best["val_oa"] or (
                validation["val_oa"] == best["val_oa"] and validation["val_ce"] < best["val_ce"])
            if improved:
                best = {"epoch": epoch, "val_oa": validation["val_oa"], "val_ce": validation["val_ce"],
                        "val_aa": validation["val_aa"]}
                if rank == 0:
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
                    "validation": validation,
                    "clean_train": clean_train, "best": best, "best_updated": improved}
                payload = {
                    "format": CHECKPOINT_VERSION, "model_selection_only": False,
                    "completed_epochs": epoch, "epoch": epoch, "net": backbone.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "rank_states": rank_states,
                    "training_config": config, "split_sha256": split_sha,
                    "train_ids": train_ids, "validation_ids": val_ids,
                    "commit": manifest["commit"], "branch": manifest["branch"],
                    "initialization": manifest["initialization"],
                    "initialized_model_sha256": manifest["initialized_model_sha256"],
                    "best": best, "best_net": best_net, "metrics": row,
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
                    print(json.dumps({"epoch": epoch, "val_oa": validation["val_oa"],
                        "val_ce": validation["val_ce"], "best": best}, allow_nan=False), flush=True)
                except BaseException as exc:
                    checkpoint_error = repr(exc)
            error = [checkpoint_error]
            dist.broadcast_object_list(error, src=0)
            if error[0] is not None:
                raise RuntimeError("Epoch checkpoint failed: " + error[0])
            dist.barrier()

        test_metrics, final_error = None, None
        dist.barrier()
        if rank == 0 and not args.skip_final_test:
            try:
                selected = torch.load(args.run_dir / "best.pth", map_location="cpu", weights_only=False)
                backbone.load_state_dict(selected["net"], strict=True)
                test_points, test_labels, test_shards = load_shards(args.data_dir, "test")
                manifest["official_test_read"] = True
                test_metrics = evaluate_clean(backbone, test_points, test_labels, args, device, "test")
                atomic_json(args.run_dir / "final_test.json", {
                    "selected_by": "validation OA then lower CE", "best": best,
                    "best_checkpoint_sha256": checkpoint_sha(args.run_dir / "best.pth"),
                    "test_shards": test_shards, **test_metrics})
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
                final_test=test_metrics, final_test_skipped=args.skip_final_test)
            atomic_json(args.run_dir / "manifest.json", manifest)
            atomic_json(args.run_dir / "heartbeat.json", {"status": "completed",
                "epoch": args.epochs, "updated_utc": manifest["finished_utc"], "best": best})
            print("V8_C01_CONTROL_TRAINING_COMPLETED", flush=True)
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
