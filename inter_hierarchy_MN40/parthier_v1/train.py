"""Fresh PartHier v1, native c1, nested200/400/600, Q2048, global64.

Full runs use 300x200 and the user's historical TS protocol. Bounded smoke
and diagnostic runs read training data only and retain the 300-epoch schedule.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
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
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

from models.pointmlp import Hype_pointMLP
from hier_proxy_v2_online.train import load_shards, PointClouds
from hier_proxy_scratch_v3.train import atomic_json, atomic_torch, capture_rng, restore_rng
from hier_proxy_scratch_v4.train import state_hash, split_identity
from ..hier_proxy_scratch_v5.sampler import SourceClassBatchSampler
from ..hier_proxy_scratch_v5.distributed import gather_without_grad, shared_crop_counts
from ..hier_proxy_scratch_v9.train import rank0_evaluation, cpu_state, worker_seed
from .base_protocol import HyCoReTrainingForward, global_base_losses, configure_backbone, PART_COUNTS
from .loss import PartHierarchyLoss

VERSION = "parthier-v1-nativec1-nested200400600-q2048-1"


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--mode", choices=("smoke", "diagnostic", "full"), default="full")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--steps-per-epoch", type=int, default=200)
    p.add_argument("--queue-size", type=int, choices=(0, 2048), default=2048)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--eval-batch-size", type=int, default=32)
    p.add_argument("--save-on-cpu", action="store_true")
    p.add_argument("--activation-checkpoint", action="store_true")
    p.add_argument("--checkpoint-policy", choices=("all", "small_medium"), default="all")
    p.add_argument("--diagnostic-max-whole", action="store_true")
    p.add_argument("--resume", type=Path)
    p.add_argument("--seed", type=int, default=4780)
    p.add_argument("--save-every", type=int, default=20)
    a = p.parse_args()
    if a.save_on_cpu and a.activation_checkpoint:
        p.error("choose one activation-memory strategy")
    if a.checkpoint_policy != "all" and not a.activation_checkpoint:
        p.error("selective policy requires activation checkpointing")
    if a.diagnostic_max_whole and a.mode != "diagnostic":
        p.error("maximum-whole capacity probe is diagnostic-only")
    if a.seed != 4780 or min(a.epochs, a.steps_per_epoch, a.eval_batch_size, a.save_every) < 1 or a.workers < 0:
        p.error("seed4780 and positive run intervals required")
    if a.mode == "full" and (a.epochs, a.steps_per_epoch, a.queue_size) != (300, 200, 2048):
        p.error("full training requires fresh300x200/Q2048")
    if a.mode == "smoke" and (a.epochs > 2 or a.steps_per_epoch > 3):
        p.error("smoke bounded to2x3")
    if a.mode == "diagnostic" and (a.epochs > 10 or a.epochs * a.steps_per_epoch > 400):
        p.error("diagnostics bounded to400 updates/10 epochs")
    # Evaluation helper compatibility: smoke selects a tiny training-only pool.
    a.smoke, a.smoke_eval_batches = a.mode != "full", 2
    return a


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def configuration(a):
    return {
        "version": VERSION, "mode": a.mode, "seed": a.seed,
        "epochs": a.epochs, "steps_per_epoch": a.steps_per_epoch,
        "schedule_epochs": 300, "global_batch": 64, "world_size": 2,
        "classes_per_rank": 16, "instances_per_class": 2,
        "sampler": "class-uniform; within-class replacement; no cyclic queue",
        "native_c": 1., "hier_c": 1., "D": 256,
        "extra_cap": False, "GS": False, "fixed_whole_depth": None,
        "whole_counts": [800, 1024], "part_counts": list(PART_COUNTS),
        "crop": "one frozen whole; one center/order; overwrite600 once; independent prefix clones",
        "backbone_passes_per_step": 4, "BN": "ordinary local32, part200/400/600 then whole",
        "intra": {"gamma": 1000., "delta": 4., "alpha_per_scale": .01, "aggregation": "sum_three_scales"},
        "hier": {"K": 20, "T": 50, "margin": .1, "tau": .1, "weight": .5,
                 "self_k": False, "scale_mask": False, "exclude_query_source": False,
                 "collision": "A=B zero; divide by all draws", "proxy_term": False},
        "order": {"weight": .1, "margin": .2, "terms": ["A<min(i,j)", "B<A"],
                  "third_descendant": "diagnostic_only", "collision": "same_mask_all_draws",
                  "depth_ST": "weighted_candidate_depth; hard forward"},
        "chain": {"margin": .2, "aggregation": "three_adjacent_hinges_sum_then_batch_mean"},
        "queue": {"capacity": a.queue_size, "item": "part", "detach": True,
                  "order": "global object row then scale; FIFO", "enqueue": "after optimizer, pre-update forward features",
                  "source_metadata": ["object_id", "scale", "step"], "deduplicate": False},
        "model_optimizer": {"name": "RiemannianSGD", "lr": .1, "eta_min": .005,
                            "momentum": .9, "weight_decay": 2e-4, "norm_clip": 1.},
        "proxy_optimizer": None, "warmup": 0, "label_smoothing": .2,
        "selection": "official2468 test each epoch; strict OA, retain first tie" if a.mode == "full" else "training-only sanity; no selection",
        "early_stopping": False, "workers": a.workers, "eval_batch_size": a.eval_batch_size,
        "save_on_cpu": a.save_on_cpu, "activation_checkpoint": a.activation_checkpoint,
        "checkpoint_BN": "restore running buffers on recompute" if a.activation_checkpoint else None,
        "checkpoint_scope": "pointmlp_euclidean_trunk" if a.activation_checkpoint else None,
        "checkpoint_policy": a.checkpoint_policy,
        "diagnostic_max_whole": a.diagnostic_max_whole,
        "precision": "FP32",
    }


def jsonl(path, data):
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(data, allow_nan=False) + "\n")


def numeric_leaves(data, prefix=""):
    result = {}
    for key, value in data.items():
        path = prefix + str(key)
        if isinstance(value, dict):
            result.update(numeric_leaves(value, path + "."))
        elif isinstance(value, (float, int)) and not isinstance(value, bool):
            result[path] = value
        elif isinstance(value, list) and all(isinstance(x, (float, int)) for x in value):
            result.update({path + "." + str(i): v for i, v in enumerate(value)})
    return result


def gradient_norm(parameters):
    terms = [p.grad.detach().float().square().sum() for p in parameters if p.grad is not None]
    return torch.stack(terms).sum().sqrt() if terms else torch.tensor(0.)


def assert_finite(parameters):
    if not all(bool(torch.isfinite(p).all()) and (p.grad is None or bool(torch.isfinite(p.grad).all())) for p in parameters):
        raise FloatingPointError("nonfinite model/gradient")


def replica_error(parameters):
    error = 0.
    for p in parameters:
        twin = p.detach().clone()
        dist.broadcast(twin, src=0)
        error = max(error, float((p.detach() - twin).abs().max()))
    result = torch.tensor(error, device=parameters[0].device)
    dist.all_reduce(result, op=dist.ReduceOp.MAX)
    return float(result)


def queue_digest(hierarchy):
    h = hashlib.sha256()
    for key in ("features", "object_ids", "scales", "steps"):
        h.update(hierarchy.queue_state()[key].numpy().tobytes())
    return h.hexdigest()


def array_sha(value):
    value = np.ascontiguousarray(value)
    return hashlib.sha256(memoryview(value).cast("B")).hexdigest()


def synchronized_assert(condition, message, device):
    good = torch.tensor(int(bool(condition)), device=device)
    dist.all_reduce(good, op=dist.ReduceOp.MIN)
    if not int(good):
        raise RuntimeError(message)


def main():
    a = arguments()
    torch.set_num_threads(2)
    rank, local = int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"])
    if int(os.environ["WORLD_SIZE"]) != 2:
        raise RuntimeError("global64 requires two local32 ranks")
    torch.cuda.set_device(local)
    device = torch.device("cuda", local)
    dist.init_process_group("nccl", timeout=dt.timedelta(minutes=15))
    config = configuration(a)
    manifest = {"format": VERSION, "status": "starting", "training_config": config,
                "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                "commit": os.environ.get("PARTHIER_SOURCE_COMMIT") or subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
                "branch": os.environ.get("PARTHIER_SOURCE_BRANCH") or subprocess.check_output(["git", "branch", "--show-current"], cwd=REPO, text=True).strip(),
                "visible_gpus": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "gpu_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
                "gpu_uuids": os.environ.get("HYCORE_PHYSICAL_GPU_UUIDS"),
                "runtime": {"python": sys.version, "torch": torch.__version__, "geoopt": geoopt.__version__},
                "command": [sys.executable, *sys.argv], "teacher": None, "source_checkpoint": None,
                "official_test_read": False, "main_training": a.mode == "full"}
    run_created = False
    try:
        creation = [None]
        if rank == 0:
            try:
                a.run_dir.mkdir(parents=True, exist_ok=False)
                run_created = True
            except BaseException as exc:
                creation[0] = repr(exc)
        dist.broadcast_object_list(creation, src=0)
        if creation[0] is not None:
            raise RuntimeError("fresh run directory creation failed: " + creation[0])
        points, labels, shards = load_shards(a.data_dir, "train")
        if len(labels) != 9840:
            raise RuntimeError("expected full9840 training pool")
        train_ids = np.arange(len(labels), dtype=np.int64)
        if a.mode == "full":
            eval_points, eval_labels, eval_shards = load_shards(a.data_dir, "test")
            if len(eval_labels) != 2468:
                raise RuntimeError("expected full2468 official test")
            manifest["official_test_read"] = True
            prefix = "test"
        else:
            eval_points, eval_labels, eval_shards = points[:64], labels[:64], []
            prefix = "smoke_train"
        manifest.update(train_size=len(labels), selection_size=len(eval_labels), training_shards=shards,
                        selection_shards=eval_shards, selection_domain=prefix,
                        train_labels_sha256=hashlib.sha256(labels.tobytes()).hexdigest())
        data_identity = {"train_points": array_sha(points), "train_labels": array_sha(labels),
                         "selection_points": array_sha(eval_points), "selection_labels": array_sha(eval_labels)}
        manifest["data_identity"] = data_identity
        seed_all(a.seed)
        backbone = configure_backbone(Hype_pointMLP()).to(device)
        initial_sha = state_hash(backbone)
        manifest["initialized_model_sha256"] = initial_sha
        model = DDP(HyCoReTrainingForward(backbone, save_on_cpu=a.save_on_cpu,
                    activation_checkpoint=a.activation_checkpoint, checkpoint_policy=a.checkpoint_policy),
                    device_ids=[local], broadcast_buffers=True)
        parameters = [p for p in model.parameters() if p.requires_grad]
        optimizer = geoopt.optim.RiemannianSGD(parameters, lr=.1, momentum=.9, weight_decay=2e-4)
        scheduler = CosineAnnealingLR(optimizer, T_max=300, eta_min=.005)
        hierarchy = PartHierarchyLoss(queue_size=a.queue_size).to(device)
        dataset = PointClouds(points, labels, 1024, "train", a.seed)
        sampler = SourceClassBatchSampler(labels, rank=rank, seed=a.seed, steps=a.steps_per_epoch)
        start_epoch, global_step, best, best_net, best_metrics = 0, 0, None, None, None
        if a.resume:
            saved = torch.load(a.resume, map_location="cpu", weights_only=False)
            if (saved.get("format") != VERSION or saved["training_config"] != config
                    or saved.get("data_identity") != data_identity or len(saved["rank_states"]) != 2
                    or not np.array_equal(saved["train_ids"], train_ids)):
                raise RuntimeError("resume identity/config differs")
            backbone.load_state_dict(saved["net"], strict=True)
            optimizer.load_state_dict(saved["optimizer"])
            scheduler.load_state_dict(saved["scheduler"])
            hierarchy.load_queue_state(saved["queue"])
            current = saved["rank_states"][rank]
            for name, value in backbone.named_buffers():
                value.copy_(current["buffers"][name].to(device))
            sampler.load_state_dict(current["sampler"])
            restore_rng(current["rng"])
            start_epoch, global_step, best = saved["completed_epochs"], saved["global_step"], saved["best"]
            best_net, best_metrics = saved["best_net"], saved["best_metrics"]
            manifest["resume_source"] = str(a.resume)
            manifest["initialized_model_sha256"] = saved["initialized_model_sha256"]
            if rank == 0 and best_net is not None:
                atomic_torch(a.run_dir / "best.pth", {"format": VERSION, "net": best_net,
                    "best": best, "metrics": best_metrics, "training_config": config,
                    "model_selection_only": True})
        elif rank == 0:
            atomic_torch(a.run_dir / "initialization_e0.pth", {"format": VERSION, "net": cpu_state(backbone),
                "training_config": config, "model_sha256": initial_sha, "seed": a.seed, "source_checkpoint": None})
        if rank == 0:
            manifest.update(status="running", completed_epochs=start_epoch)
            atomic_json(a.run_dir / "manifest.json", manifest)
        for epoch_index in range(start_epoch, a.epochs):
            epoch, epoch_started = epoch_index + 1, time.perf_counter()
            sampler.set_epoch(epoch_index)
            seed_all(a.seed + epoch_index * 1009 + rank * 1000003)
            loader = DataLoader(dataset, batch_sampler=sampler, num_workers=a.workers, pin_memory=True,
                worker_init_fn=worker_seed, generator=torch.Generator().manual_seed(a.seed + epoch_index * 1009 + rank * 1000003))
            model.train()
            torch.cuda.reset_peak_memory_stats(device)
            sums, correct, seen, unique_ids = {}, 0, 0, set()
            for step, (cloud, gold, subset_ids) in enumerate(loader):
                step_started = time.perf_counter()
                seed_all(a.seed + epoch_index * 10000019 + step * 1009 + rank * 1000003)
                whole_count, _unused_part_count = shared_crop_counts(a.seed, epoch_index, step)
                if a.diagnostic_max_whole:
                    whole_count = 1024
                ids = gather_without_grad(torch.as_tensor(subset_ids, device=device).long())
                optimizer.zero_grad(set_to_none=True)
                bn_layers = [m for m in backbone.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
                before_bn = [int(m.num_batches_tracked) for m in bn_layers]
                local_out = model(cloud.to(device), whole_count)
                values = global_base_losses(local_out, gold.to(device))
                weighted, stats = hierarchy(values["mu"], values["parts"], values["gold"], ids,
                    step=global_step, seed=a.seed + epoch_index * 10000019 + step * 1009 + 70000027)
                loss = values["base"] + weighted
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("nonfinite objective")
                loss.backward()
                assert_finite(parameters)
                bn_delta = [int(m.num_batches_tracked) - n for m, n in zip(bn_layers, before_bn)]
                synchronized_assert(all(n == 4 for n in bn_delta), "BN must update exactly4 times", device)
                preclip = float(torch.nn.utils.clip_grad_norm_(parameters, 1.))
                postclip = float(gradient_norm(parameters))
                optimizer.step()
                assert_finite(parameters)
                hierarchy.enqueue(values["parts"], ids, global_step)
                global_step += 1
                check_replica = step in (0, a.steps_per_epoch - 1)
                replica = replica_error(parameters) if check_replica else None
                if replica is not None and replica > 1e-6:
                    raise RuntimeError("replica parameters differ")
                crop = local_out["input_stats"]
                synchronized_assert(all(crop[k] for k in ("parts_nested", "part600_matches_whole_prefix", "whole_suffix_unchanged")), "crop invariant failed", device)
                row = {"epoch": epoch, "step": step + 1, "global_step": global_step,
                       "loss": float(loss), "ce": float(values["ce"]),
                       "intra_radial": float(values["intra_radial"]), "intra_contrastive": float(values["intra_contrastive"]),
                       "preclip_norm": preclip, "postclip_norm": postclip, "bn_delta": 4,
                       "lr": optimizer.param_groups[0]["lr"], "queue_size_after": len(hierarchy.queue),
                       "replica_error": replica, "whole_count": whole_count, "hierarchy": stats,
                       "per_scale": {k: {n: float(v) for n, v in val.items() if n != "child_count"} for k, val in values["per_scale"].items()}}
                for key, number in numeric_leaves(row).items():
                    sums[key] = sums.get(key, 0.) + number
                correct += int((values["logits"].argmax(-1) == values["gold"]).sum())
                seen += len(ids)
                unique_ids.update(ids.detach().cpu().tolist())
                if rank == 0:
                    jsonl(a.run_dir / "steps.jsonl", row)
                    if step == 0 and epoch in (1, 2, 5, 20, 100, 200, 300):
                        atomic_torch(a.run_dir / f"actual_training_epoch_{epoch:03d}.pth", {
                            "stage": "pre_update_features", "epoch": epoch, "global_step": global_step - 1,
                            "mu": values["mu"].detach().cpu(), "parts": values["parts"].detach().cpu(),
                            "ids": ids.cpu(), "labels": values["gold"].cpu(), "crop": crop,
                            "crop_scope": "rank0 local rows0:32; features and IDs global rank order",
                            "training_config": config})
                    if step == 0 or (step + 1) % 20 == 0:
                        atomic_json(a.run_dir / "heartbeat.json", {"status": "running", "epoch": epoch,
                            "step": step + 1, "global_step": global_step, "loss": float(loss),
                            "updated_utc": dt.datetime.now(dt.timezone.utc).isoformat()})
                        print(json.dumps({"epoch": epoch, "step": step + 1, "loss": float(loss),
                                          "queue": len(hierarchy.queue), "peak_MiB": torch.cuda.max_memory_allocated() / 2**20}), flush=True)
                del loss, weighted, local_out, values
                torch.cuda.synchronize(device)
                free_bytes, total_bytes = torch.cuda.mem_get_info(device)
                jsonl(a.run_dir / f"resources_rank_{rank}.jsonl", {
                    "rank": rank, "epoch": epoch, "step": step + 1, "global_step": global_step,
                    "whole_count": whole_count, "queue_size_after": len(hierarchy.queue),
                    "seconds": time.perf_counter() - step_started,
                    "allocated_MiB": torch.cuda.memory_allocated(device) / 2**20,
                    "reserved_MiB": torch.cuda.memory_reserved(device) / 2**20,
                    "peak_allocated_MiB": torch.cuda.max_memory_allocated(device) / 2**20,
                    "peak_reserved_MiB": torch.cuda.max_memory_reserved(device) / 2**20,
                    "free_MiB": free_bytes / 2**20, "total_MiB": total_bytes / 2**20,
                })
            scheduler.step()
            q_shas = [None, None]
            dist.all_gather_object(q_shas, queue_digest(hierarchy))
            if q_shas[0] != q_shas[1]:
                raise RuntimeError("queue replicas differ")
            metrics = {"epoch": epoch, "updates": a.steps_per_epoch, "global_step": global_step,
                       "train_oa": 100 * correct / seen, "train_seen": seen, "unique_train_ids": len(unique_ids),
                       "step_means": {k: v / a.steps_per_epoch for k, v in sums.items()},
                       "queue_sha256": q_shas[0], "queue_size": len(hierarchy.queue),
                       "peak_memory_MiB": torch.cuda.max_memory_allocated() / 2**20}
            if a.mode != "diagnostic":
                metrics.update(rank0_evaluation(backbone, eval_points, eval_labels, a, device, prefix, rank))
                if a.mode == "full" and metrics["test_count"] != 2468:
                    raise RuntimeError("full official test count differs from2468")
            better = a.mode == "full" and (best is None or metrics["test_oa"] > best["oa"])
            if better:
                best = {"epoch": epoch, "oa": metrics["test_oa"], "aa": metrics["test_aa"]}
                best_net, best_metrics = cpu_state(backbone), dict(metrics)
            states = [None, None]
            dist.all_gather_object(states, {"rank": rank, "rng": capture_rng(), "sampler": sampler.state_dict(),
                "buffers": {n: v.detach().cpu().clone() for n, v in backbone.named_buffers()}})
            metrics["wall_seconds"] = time.perf_counter() - epoch_started
            if rank == 0:
                saved = {"format": VERSION, "net": cpu_state(backbone), "optimizer": optimizer.state_dict(),
                         "scheduler": scheduler.state_dict(), "queue": hierarchy.queue_state(),
                         "training_config": config, "rank_states": states, "train_ids": train_ids,
                         "completed_epochs": epoch, "global_step": global_step, "best": best,
                         "best_net": best_net, "best_metrics": best_metrics, "data_identity": data_identity,
                         "initialized_model_sha256": manifest["initialized_model_sha256"],
                         "commit": manifest["commit"], "metrics": metrics}
                atomic_torch(a.run_dir / "last.pth", saved)
                if epoch == 1 or epoch % a.save_every == 0:
                    atomic_torch(a.run_dir / f"checkpoint_epoch_{epoch:03d}.pth", saved)
                if better:
                    atomic_torch(a.run_dir / "best.pth", saved)
                atomic_json(a.run_dir / f"metrics_epoch_{epoch:03d}.json", metrics)
                jsonl(a.run_dir / "metrics.jsonl", metrics)
                manifest.update(completed_epochs=epoch, best=best, global_step=global_step)
                atomic_json(a.run_dir / "manifest.json", manifest)
                if epoch == 1:
                    reloaded = torch.load(a.run_dir / "last.pth", map_location="cpu", weights_only=False)
                    acceptance = {"updates": metrics["updates"], "selection_count": metrics.get(prefix + "_count", 0),
                        "BN_updates_per_step": 4, "rank_states": len(reloaded["rank_states"]),
                        "queue_equal_across_ranks": True, "queue_size": len(hierarchy.queue),
                        "finite_weights": all(bool(torch.isfinite(v).all()) for v in reloaded["net"].values()),
                        "schedule_Tmax": reloaded["scheduler"]["T_max"], "actual_optimizer": "RiemannianSGD",
                        "source_commit": manifest["commit"], "data_identity": data_identity,
                        "diagnostic_only": a.mode != "full"}
                    if not acceptance["finite_weights"] or acceptance["rank_states"] != 2 or acceptance["schedule_Tmax"] != 300:
                        raise RuntimeError("first epoch saved state acceptance failed")
                    atomic_json(a.run_dir / "first_epoch_acceptance.json", acceptance)
                print(json.dumps({"epoch_completed": epoch, "wall_seconds": metrics["wall_seconds"], "best": best}), flush=True)
            dist.barrier()
        if rank == 0:
            manifest.update(status="completed", finished_utc=dt.datetime.now(dt.timezone.utc).isoformat())
            atomic_json(a.run_dir / "manifest.json", manifest)
            atomic_json(a.run_dir / "heartbeat.json", {"status": "completed", "completed_epochs": a.epochs, "global_step": global_step})
    except BaseException as exc:
        if rank == 0 and run_created:
            manifest.update(status="failed", error=repr(exc), failed_utc=dt.datetime.now(dt.timezone.utc).isoformat())
            atomic_json(a.run_dir / "manifest.json", manifest)
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
