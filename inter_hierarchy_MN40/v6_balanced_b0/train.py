"""V6 matched balanced64 B0: inherit historical H20 epochs1-20 and continue21-300.

No HIER forward or proxy update. Exact CE/intra, data, sampler, local32 BN,
alias/FPS and original300-epoch cosine are retained. New output directories
are required; smoke updates remain explicitly nonreportable/nonresumable.
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
    SourceClassBatchSampler, checkpoint_sha, freeze_curvature,
    gather_without_grad, load_shards, seed_all, shared_crop_counts,
    split_identity, stratified_split,
)
from ..hier_proxy_scratch_v5.base_protocol import global_base_losses, smoothed_ce
from ..hier_proxy_scratch_v6.train import fixed_configuration as source_configuration
from ..hycore_b64_v6.train import parameter_replica_difference
from .contract import CHECKPOINT_VERSION, SOURCE_VERSION, derive_b0_config, validate_checkpoint_contract
from ..hier_proxy_scratch_v5.telemetry import EpochTelemetry
from hier_proxy_scratch_v3.train import atomic_json, atomic_torch, capture_rng, restore_rng
from hier_proxy_scratch_v4.train import state_hash

import geoopt
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir",type=Path,required=True)
    parser.add_argument("--run-dir",type=Path,required=True)
    parser.add_argument("--source-checkpoint",type=Path,required=True)
    parser.add_argument("--source-sha256",help="Optional independently verified source identity")
    parser.add_argument("--resume",type=Path)
    parser.add_argument("--seed",type=int,default=22)
    parser.add_argument("--epochs",type=int,default=300)
    parser.add_argument("--steps-per-epoch",type=int,default=200)
    parser.add_argument("--workers",type=int,default=2)
    parser.add_argument("--eval-batch-size",type=int,default=32)
    parser.add_argument("--train-eval-every",type=int,default=10)
    parser.add_argument("--save-every",type=int,default=20)
    parser.add_argument("--skip-final-test",action="store_true")
    parser.add_argument("--fork-smoke",action="store_true")
    parser.add_argument("--smoke-epochs",type=int,default=1)
    parser.add_argument("--smoke-steps",type=int,default=2)
    parser.add_argument("--smoke-eval-batches",type=int,default=2)
    args=parser.parse_args()
    if (args.seed,args.epochs,args.steps_per_epoch,args.workers,args.eval_batch_size,args.train_eval_every)!=(22,300,200,2,32,10):
        parser.error("Matched V6 requires seed22/epochs300/steps200/workers2/eval32/train-eval10")
    if args.save_every<1:
        parser.error("save interval must be positive")
    if args.fork_smoke and (args.resume is not None or not args.skip_final_test or
        not 1<=args.smoke_epochs<=2 or not 1<=args.smoke_steps<=3 or not 1<=args.smoke_eval_batches<=2):
        parser.error("Smoke: no resume, skip final test, epochs<=2, steps<=3, eval batches<=2")
    args.warmup_epochs=20 # Historical source metadata; never repeated.
    args.audit_epochs=(21,40,100,160,200,240,300)
    args.smoke=args.fork_smoke
    return args

def source_reference(args):
    reference_args=argparse.Namespace(**vars(args))
    reference_args.smoke=False
    return source_configuration(reference_args)

def fixed_configuration(args):
    return derive_b0_config(source_reference(args))

def nested_equal(left,right):
    if isinstance(left,torch.Tensor) or isinstance(right,torch.Tensor):
        return isinstance(left,torch.Tensor) and isinstance(right,torch.Tensor) and torch.equal(left.cpu(),right.cpu())
    if isinstance(left,np.ndarray) or isinstance(right,np.ndarray):
        return isinstance(left,np.ndarray) and isinstance(right,np.ndarray) and np.array_equal(left,right)
    if isinstance(left,dict) and isinstance(right,dict):
        return left.keys()==right.keys() and all(nested_equal(left[k],right[k]) for k in left)
    if isinstance(left,(tuple,list)) and isinstance(right,(tuple,list)):
        return type(left) is type(right) and len(left)==len(right) and all(nested_equal(a,b) for a,b in zip(left,right))
    return type(left) is type(right) and left==right

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


def validate_resume(saved,config,reference,split_sha,train_ids,val_ids,sampler):
    validate_checkpoint_contract(saved,config,reference,split_sha,train_ids.tolist(),val_ids.tolist())
    for state in saved["rank_states"]:
        sampler.load_state_dict(state["sampler"])

def main():
    args = arguments()
    torch.set_num_threads(2)
    rank, local_rank = int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"])
    if int(os.environ["WORLD_SIZE"]) != 2:
        raise RuntimeError("V6 balanced B0 requires exactly two ranks")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", timeout=dt.timedelta(minutes=10))
    started_wall = time.perf_counter()
    owns_run_dir=False
    config=fixed_configuration(args)
    reference=source_reference(args)
    manifest = {
        "status": "starting", "main_training": not args.smoke,
        "diagnostic_only": args.smoke,
        "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        "branch": subprocess.check_output(["git", "branch", "--show-current"], cwd=REPO, text=True).strip(),
        "command_argv": [sys.executable, *sys.argv],
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "training_config": config, "visible_gpus": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "physical_gpus": os.environ.get('HYCORE_PHYSICAL_GPUS', os.environ.get('CUDA_VISIBLE_DEVICES')),
        "physical_gpu_uuids": os.environ.get('HYCORE_PHYSICAL_GPU_UUIDS'),
        "runtime": {"torch": torch.__version__, "geoopt": geoopt.__version__, "numpy": np.__version__},
        "initialization": {"method": "complete V6-H20 e20 fork", "seed": args.seed, "source_checkpoint": str(args.source_checkpoint), "teacher": None},
        "official_test_read": False, "official_test_evaluations":0, "resume_identity": None,
        "objective_delta":"HIER weight0; no proxy/mining/update; CE/intra unchanged",
        "inherited_prefix_epochs":20, "new_warmup_epochs":0,
        "historical_H20_bitwise_paired":False, "smoke_partial_not_reportable":args.fork_smoke,
    }
    try:
        startup_error=[None]
        if rank==0:
            try:
                args.run_dir.mkdir(parents=True,exist_ok=False)
                owns_run_dir=True
            except BaseException as exc:
                startup_error[0]=repr(exc)
        dist.broadcast_object_list(startup_error,src=0)
        if startup_error[0] is not None:
            raise RuntimeError("New output directory creation failed: "+startup_error[0])
        dist.barrier()
        points, labels, shards = load_shards(args.data_dir, "train")
        train_ids, val_ids = stratified_split(labels, args.seed, .1)
        split_sha = split_identity(train_ids, val_ids)
        synchronized_assert(len(train_ids)==8856 and len(val_ids)==984 and
                            points.shape[1:]==(2048,3) and np.isfinite(points).all(),
                            "ModelNet40 train8856/validation984 schema required",device)
        manifest.update(split_sha256=split_sha, train_size=len(train_ids), validation_size=len(val_ids),
                        training_shards=shards, train_labels_sha256=hashlib.sha256(labels.tobytes()).hexdigest())
        seed_all(args.seed)
        backbone = Hype_pointMLP().to(device)
        freeze_curvature(backbone)
        initial_hash = state_hash(backbone)
        manifest["initialized_model_sha256"] = initial_hash
        model = DDP(HyCoReTrainingForward(backbone), device_ids=[local_rank], broadcast_buffers=True)
        # HIER initialization has its own CPU Generator, consuming no base RNG.
        # No dummy proxy allocation/forward is required after complete-state restore.
        model_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        optimizer = geoopt.optim.RiemannianSGD(model_parameters, lr=.1, momentum=.9, weight_decay=2e-4)
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=.005)
        dataset = PointClouds(points[train_ids], labels[train_ids], 1024, "train", args.seed)
        sampler = SourceClassBatchSampler(labels[train_ids], rank=rank, seed=args.seed, steps=args.steps_per_epoch)
        source_digest=checkpoint_sha(args.source_checkpoint)
        if args.source_sha256 is not None and source_digest != args.source_sha256:
            raise RuntimeError("Source checkpoint differs from independently verified SHA256")
        prefix=torch.load(args.source_checkpoint,map_location="cpu",weights_only=False)
        validate_resume(prefix,config,reference,split_sha,train_ids,val_ids,sampler)
        if prefix["format"]!=SOURCE_VERSION or int(prefix["completed_epochs"])!=20:
            raise RuntimeError("Source must be complete historical V6-H20 epoch20")
        fork_identity={"path":str(args.source_checkpoint),"sha256":source_digest,
                       "source_commit":prefix["commit"],"completed_epochs":20,
                       "source_initialized_model_sha256":prefix["initialized_model_sha256"]}
        if prefix["initialized_model_sha256"]!=initial_hash:
            raise RuntimeError("Random initialization identity differs from historical source")
        saved=prefix
        if args.resume is not None:
            saved=torch.load(args.resume,map_location="cpu",weights_only=False)
            validate_resume(saved,config,reference,split_sha,train_ids,val_ids,sampler)
            if saved["format"]!=CHECKPOINT_VERSION or saved.get("fork_source")!=fork_identity:
                raise RuntimeError("Resume must retain exact epoch20 fork identity")
        backbone.load_state_dict(saved["net"],strict=True)
        freeze_curvature(backbone)
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        rank_state=saved["rank_states"][rank]
        sampler.load_state_dict(rank_state["sampler"])
        buffers=dict(backbone.named_buffers())
        if buffers.keys()!=rank_state["BN_buffers"].keys():
            raise RuntimeError("Rank buffer names differ from model")
        for name,value in rank_state["BN_buffers"].items():
            buffers[name].copy_(value.to(buffers[name]))
        restore_rng(rank_state["rng"])
        checks={
            "model_parameters_exact":all(torch.equal(value.detach().cpu(),saved["net"][name]) for name,value in backbone.named_parameters()),
            "rank_buffers_exact":all(torch.equal(value.detach().cpu(),rank_state["BN_buffers"][name]) for name,value in backbone.named_buffers()),
            "curvature_c1":all(bool(torch.isclose(m.c.detach(),torch.ones_like(m.c),atol=1e-6,rtol=0).all()) for m in (backbone.manifold,backbone.manifold2)),
            "optimizer_exact":nested_equal(optimizer.state_dict(),saved["optimizer"]),
            "scheduler_exact":nested_equal(scheduler.state_dict(),saved["scheduler"]),
            "sampler_exact":sampler.state_dict()==rank_state["sampler"],
            "rng_exact":nested_equal(capture_rng(),rank_state["rng"]),
            "initialization_identity_exact":initial_hash==prefix["initialized_model_sha256"],
        }
        synchronized_assert(all(checks.values()),"Complete-state restoration audit failed",device)
        restored=[None,None]
        dist.all_gather_object(restored,{"rank":rank,"checks":checks,"rank_model_state_sha256":state_hash(backbone)})
        synchronized_assert(parameter_replica_difference(model_parameters)<=1e-7,"Restored parameter replicas differ",device)
        start_epoch=int(saved["completed_epochs"])
        best,best_net=saved["best"],saved["best_net"]
        manifest.update(initialized_model_sha256=prefix["initialized_model_sha256"],
                        source_initialization=prefix["initialization"],fork_source=fork_identity,
                        restore_audit=restored,inherited_best=prefix["best"],
                        resume_identity={"path":str(args.resume or args.source_checkpoint),
                            "sha256":checkpoint_sha(args.resume or args.source_checkpoint),
                            "completed_epochs":start_epoch,"source_commit":saved["commit"]})
        if rank==0:
            atomic_json(args.run_dir/"restore_audit.json",{"rank_audits":restored,"fork_source":fork_identity,
                        "source_configuration":reference,"target_configuration":config,
                        "first_continuation_epoch":start_epoch+1,"base_rng_dummy_proxy_consumption_required":False})
            atomic_torch(args.run_dir/"best.pth",{"format":CHECKPOINT_VERSION,"model_selection_only":True,
                        "net":best_net,"best":best,"epoch":best["epoch"],"training_config":config,
                        "split_sha256":split_sha,"fork_source":fork_identity,
                        "selection_scope":"shared prefix plus continuation"})
        del saved,prefix
        devices = [None, None]
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

        stop_epoch=min(args.epochs,start_epoch+args.smoke_epochs) if args.fork_smoke else args.epochs
        expected_steps=args.smoke_steps if args.fork_smoke else args.steps_per_epoch
        for epoch_index in range(start_epoch,stop_epoch):
            epoch = epoch_index + 1
            lambda_hier = 0.
            model.train()
            sampler.set_epoch(epoch_index)
            epoch_seed = args.seed + epoch_index * 1009 + rank * 1000003
            seed_all(epoch_seed)
            loader = DataLoader(dataset, batch_sampler=sampler, num_workers=args.workers,
                                worker_init_fn=worker_seed, pin_memory=True,
                                generator=torch.Generator().manual_seed(epoch_seed))
            telemetry = EpochTelemetry(train_ids, labels[train_ids], num_classes=40) if rank == 0 else None
            learning_rates = {"model":optimizer.param_groups[0]["lr"]}
            epoch_started = time.perf_counter()
            torch.cuda.reset_peak_memory_stats(device)
            steps_done=0
            for step, (cloud, local_gold, subset_ids) in enumerate(loader):
                if step>=expected_steps:
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
                values = global_base_losses(local_out, local_gold)
                ids = gather_without_grad(local_ids)
                loss = values["base"]
                synchronized_assert(bool(torch.isfinite(loss)), "Nonfinite objective", device)
                loss.backward()
                synchronized_assert(finite_tensors(parameter.grad for parameter in
                    model.parameters()), "Nonfinite gradients", device)
                replica_check=step in (0,expected_steps-1)
                gradient_difference=parameter_replica_difference(model_parameters,gradients=True) if replica_check else None
                if replica_check:
                    synchronized_assert(gradient_difference<=1e-7,"DDP gradient replicas differ",device)
                bn_delta = [int(module.num_batches_tracked) - before
                            for module, before in zip(bn_layers, bn_before)]
                synchronized_assert(bool(bn_delta) and all(delta == 2 for delta in bn_delta), "Child/whole BN update must equal2", device)
                synchronized_assert(all(local_out["input_stats"][key] for key in
                    ("child_aliases_whole", "whole_aliases_source", "whole_changed_by_child")),
                    "Original part-overwrite behavior changed", device)
                preclip_norm = torch.nn.utils.clip_grad_norm_(model_parameters, 1.)
                postclip_norm = gradient_norm(model_parameters)
                synchronized_assert(bool(torch.isfinite(preclip_norm)) and bool(torch.isfinite(postclip_norm)) and float(postclip_norm)<=1.00001,
                                    "Nonfinite clipped norm", device)
                optimizer.step()
                synchronized_assert(finite_tensors(model_parameters),"Nonfinite updated parameters",device)
                synchronized_assert(all(bool(torch.isclose(m.c.detach(),torch.ones_like(m.c),atol=1e-6,rtol=0).all()) for m in (backbone.manifold,backbone.manifold2)),
                                    "Fixed curvature changed from c1",device)
                parameter_difference=parameter_replica_difference(model_parameters) if replica_check else None
                if replica_check:
                    synchronized_assert(parameter_difference<=1e-7,"DDP parameter replicas differ",device)
                torch.cuda.synchronize(device)
                elapsed = torch.tensor(time.perf_counter() - started, device=device)
                dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
                rank_peaks = [None, None]
                dist.all_gather_object(rank_peaks, {
                    "rank": rank, "peak_allocated_MiB": torch.cuda.max_memory_allocated(device) / 1024 ** 2,
                    "BN_update_min": min(bn_delta), "BN_update_max": max(bn_delta),
                    "input": local_out["input_stats"],
                })
                scalars = {
                    "loss": float(loss.detach()), "base": float(values["base"].detach()),
                    "ce": float(values["ce"].detach()), "lambda_hier": lambda_hier,
                    "intra_contrastive": float(values["intra_contrastive"].detach()),
                    "intra_radial": float(values["intra_radial"].detach()),
                    "weighted_intra_contrastive":.01*float(values["intra_contrastive"].detach()),
                    "weighted_intra_radial":.01*float(values["intra_radial"].detach()),
                    "whole_mean_depth":float(values["whole_depth"]),"part_mean_depth":float(values["part_depth"]),
                    "whole_mean_radius":float(values["whole_radius"]),"part_mean_radius":float(values["part_radius"]),
                    "model_gradient_replica_max_difference":gradient_difference,
                    "model_parameter_replica_max_difference":parameter_difference,
                    "model_preclip_norm": float(preclip_norm), "model_postclip_norm": float(postclip_norm),
                    "model_clip_applied": float(preclip_norm > 1.),
                    "model_clip_scale": min(1., 1. / (float(preclip_norm) + 1e-6)),
                    "step_seconds_max_rank": float(elapsed),
                    "peak_allocated_MiB_max_rank": max(item["peak_allocated_MiB"] for item in rank_peaks),
                    "whole_count": whole_count, "child_count": child_count,
                    "global_batch_class_count":int(values["gold"].unique().numel()),
                    "same_instance_flip_negative_count":int((ids==ids.flip(0)).sum()),
                }
                if rank == 0:
                    telemetry.update(ids=ids, labels=values["gold"], logits=values["logits"],
                        mu=values["mu"], nu=values["nu"], scalar_metrics=scalars,
                        hier_stats=None,sample_mining=None)
                    heartbeat = {"status": "running", "epoch": epoch, "step": step + 1,
                        "steps_per_epoch": expected_steps, "phase": "V6_balanced_B0",
                        "updated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                        "lr": learning_rates, "scalars": scalars, "rank_observations": rank_peaks}
                    atomic_json(args.run_dir / "heartbeat.json", heartbeat)
                    append_jsonl(args.run_dir / "steps.jsonl", {"epoch": epoch, "step": step + 1,
                        "ids": ids.detach().cpu().tolist(), **scalars})
                    if step == 0 or (step + 1) % 20 == 0 or step + 1 == expected_steps:
                        print(json.dumps({"epoch": epoch, "step": step + 1, "loss": scalars["loss"],
                            "phase": heartbeat["phase"], "lr": learning_rates}, allow_nan=False), flush=True)
                steps_done+=1
                del local_out,values,loss
            synchronized_assert(steps_done==expected_steps,"Epoch update budget differs",device)
            scheduler.step()
            validation = rank0_evaluation(backbone, points[val_ids], labels[val_ids], args, device, "val", rank)
            clean_train = None
            if args.fork_smoke or epoch % args.train_eval_every == 0 or epoch == args.epochs:
                clean_train = rank0_evaluation(backbone, points[train_ids], labels[train_ids], args, device, "train_eval", rank)
            improved = not args.fork_smoke and (best is None or validation["val_oa"] > best["val_oa"] or (
                validation["val_oa"] == best["val_oa"] and validation["val_ce"] < best["val_ce"]))
            synchronized_assert(args.fork_smoke or validation["val_count"]==984,
                                "Production validation must cover984 objects",device)
            synchronized_assert(args.fork_smoke or clean_train is None or clean_train["train_eval_count"]==8856,
                                "Production clean train must cover8856 objects",device)
            if improved:
                best = {"epoch": epoch, "val_oa": validation["val_oa"], "val_ce": validation["val_ce"],
                        "val_aa": validation["val_aa"]}
                if rank == 0:
                    best_net = cpu_state(backbone)
            rank_states = [None, None]
            dist.all_gather_object(rank_states, checkpoint_rank_state(backbone, sampler, rank))
            checkpoint_error = None
            if rank == 0:
                row = {"epoch": epoch, "phase": "V6_balanced_B0",
                    "lambda_hier": lambda_hier, "lr_used": learning_rates,
                    "lr_next": {"model": optimizer.param_groups[0]["lr"]},
                    "epoch_wall_seconds": time.perf_counter() - epoch_started,
                    "telemetry":telemetry.summary(),"structure":None,
                    "actual_optimizer_steps":steps_done,"smoke_partial_not_reportable":args.fork_smoke,
                    "validation": validation,
                    "clean_train": clean_train, "best": best, "best_updated": improved}
                payload = {
                    "format": CHECKPOINT_VERSION, "model_selection_only": False,
                    "completed_epochs": epoch, "epoch": epoch, "net": backbone.state_dict(),
                    "optimizer":optimizer.state_dict(),"scheduler":scheduler.state_dict(),
                    "rank_states":rank_states,"fork_source":fork_identity,
                    "diagnostic_only":args.fork_smoke,"continuation_smoke_not_resumable":args.fork_smoke,
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
                    if not args.fork_smoke and (epoch % args.save_every==0 or epoch in (99,200,300)):
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
                if len(test_labels)!=2468:
                    raise RuntimeError("Official test must contain2468 objects")
                manifest["official_test_read"]=True
                manifest["official_test_evaluations"]+=1
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
            manifest.update(status="completed", completed_epochs=stop_epoch,
                finished_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                wall_seconds=time.perf_counter() - started_wall, best=best,
                final_test=test_metrics, final_test_skipped=args.skip_final_test)
            atomic_json(args.run_dir / "manifest.json", manifest)
            atomic_json(args.run_dir / "heartbeat.json", {"status": "completed",
                "epoch": stop_epoch, "updated_utc": manifest["finished_utc"], "best": best})
            print("V6_BALANCED_B0_COMPLETED",flush=True)
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
