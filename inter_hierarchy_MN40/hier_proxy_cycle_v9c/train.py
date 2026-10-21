"""Paired cycle-B64 ablation: original c.1/cap2.3/GS/half-margin base.

HIER-on uses a differentiable fixed-depth4.6 view forked before base GS.
Production is300x300 updates and test selection; smoke never reads test.
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

from ..hier_proxy_scratch_v5.diagnose import (REPO, Hype_pointMLP, PointClouds,
    checkpoint_sha, gather_without_grad, seed_all, split_identity)
from ..hier_proxy_scratch_v5.distributed import broadcast_parameters, sync_parameter_gradients
from ..hier_proxy_scratch_v9.controls import (worker_seed, synchronized_assert,
    gradient_norm, finite_tensors, append_jsonl, atomic_copy, cpu_state, rank0_evaluation)
from ..hier_proxy_scratch_v6.structure_monitor import StructureAccumulator, radial_angular_gradient
from ..hier_proxy_scratch_v8_fulltrain.protocol import full_training_ids, select_best
from ..hycore_c01_gs_halfmargin_20261020.train import state_digest
from .base_protocol import global_base_losses, smoothed_ce, HyCoReTrainingForward
from .geometry import (configure_backbone, freeze_curvature, proxy_update_stats,
    whole_to_hier, assert_hier_shell, HIER_DEPTH)
from .sampler import ClassCycleBatchSampler
from .telemetry import EpochTelemetry
from .parameter_audit import audit_parameter_gradients
from hier_proxy_scratch_v3.train import atomic_json, atomic_torch, capture_rng, restore_rng
from hier_proxy_scratch_v4.train import state_hash

import geoopt
import h5py
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

CHECKPOINT_VERSION = "hycore-cycle-v9c-c01-cap23-gs1-m5002-testselect-1"
EXPECTED_ORIGINAL_INITIAL_DIGEST = "8926765a4842652b751873fb89aa37d24d9b0f4f2d831d853b9cc0c1cd46472d"
EXPECTED_TRAIN_LABELS_SHA = "2c1052c604c4379a4b8452ddf3e7beee2e5471269eba8bd660d5f2c630613d89"
EXPECTED_TEST_LABELS_SHA = "51a870948420ac011793a54e04d0b0c7341809a3b0295dfba1c7ba450131a576"
TRAIN_SHARD_ORDER = tuple("ply_data_train%d.h5" % n for n in (1, 4, 3, 2, 0))


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hier", choices=("off", "on"), required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--seed", type=int, default=4780)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--steps-per-epoch", type=int, default=300)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--train-eval-every", type=int, default=10)
    parser.add_argument("--audit-epochs", default="1,5,20,40,60,80,99,100,120,140,160,180,200,220,240,260,280,300")
    parser.add_argument("--save-every", type=int, default=20)
    parser.add_argument("--skip-final-test", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-eval-batches", type=int, default=1)
    args = parser.parse_args(argv)
    args.audit_epochs = tuple(sorted(set(int(x) for x in args.audit_epochs.split(",") if x.strip())))
    if args.workers < 0 or min(args.eval_batch_size, args.train_eval_every, args.save_every) < 1:
        parser.error("workers nonnegative; evaluation/save intervals positive")
    if args.seed != 4780:
        parser.error("Both arms require original seed4780")
    if args.smoke:
        if not (1 <= args.epochs <= 2 and 1 <= args.steps_per_epoch <= 2 and
                1 <= args.smoke_eval_batches <= 2 and args.skip_final_test):
            parser.error("Smoke: <=2 epochs x2 train steps, train-only, skip-final-test")
    elif (args.epochs, args.steps_per_epoch, args.workers, args.eval_batch_size) != (300, 300, 8, 16):
        parser.error("Production requires300x300, workers8, eval16")
    elif args.skip_final_test:
        parser.error("Production records selected official test")
    if any(epoch < 1 for epoch in args.audit_epochs):
        parser.error("Audit epochs are positive one-based")
    return args


def fixed_configuration(args):
    on = args.hier == "on"
    name = ("ORIG-HIER64-K20-W0-C0.1-CAP2.3-GSbase1-GShier1-H4.6-M500_2-CYCLE300-TS-S4780" if on else "ORIG-B64-C0.1-CAP2.3-GS1-M500_2-CYCLE300-TS-S4780")
    return {
        "architecture": "Hype_pointMLP", "num_classes": 40, "seed": args.seed,
        "epochs": args.epochs, "steps_per_epoch": args.steps_per_epoch,
        "world_size": 2, "global_batch": 64, "classes_per_rank": 16, "instances_per_class": 2,
        "sampler": "five-step balanced class omission; shuffled per-class cyclic queues; cross-epoch cursor",
        "sampler_epoch_reset": False, "c": .1, "base_c": .1, "native_c": .1,
        "hier_c": .1 if on else None, "D": 256, "P": 512 if on else 0, "hier_enabled": on,
        "warmup_epochs": 0, "sample_K": 20 if on else None, "proxy_K": 20 if on else None,
        "T": 50 if on else None, "margin": .1 if on else None, "tau": .1 if on else None,
        "lambda_hier_after_warmup": .5 if on else 0.,
        "label_smoothing": .2, "alpha_contrastive": .01, "alpha_radial": .01,
        "contrastive_margin": 2., "radial_gamma": 500., "radial_margin": "500 / child_count",
        "whole_count_range": [800, 1024], "child_count_range": [200, 600],
        "part_overwrites_whole": True, "FPS_first_stage": 512,
        "BN": "ordinary local32; part then whole; rank0 buffer broadcast; no SyncBN",
        "geometry_domains": {"native_whole_part": .1, **({"hier_view_proxy": .1} if on else {})},
        "precision": "FP32", "native_extra_cap": 2.3, "native_GS": True,
        "hier_view": "direction-preserving fixed depth from cap-after/pre-GS whole" if on else None,
        "hier_view_depth": 4.6 if on else None, "proxy_tangent_cap": 2.3 if on else None,
        "HIER_backward_hook": on,
        "hook_scope": "one base GS per whole/part; one pre-base-GS HIER-view GS; one proxy GS" if on else "one shared GS per whole/part",
        "intra_hinge": "official torch.maximum, half gradient at exact zero",
        "self_negative": False, "proxy_parameter_projection": "none; official forward cap only" if on else None,
        "proxy_initialization": "private seeded N(0,I)/sqrt(D)*2.3*.9" if on else None,
        "HIER_objective_enabled": on, "HIER_calls_per_step": int(on),
        "model_optimizer": {"name": "RiemannianSGD", "lr": .1, "min_lr": .005,
            "momentum": .9, "weight_decay": 2e-4, "norm_clip": 1.},
        "proxy_optimizer": {"name": "AdamW", "lr": .01, "min_lr": .0005, "betas": [.9, .999],
            "eps": 1e-8, "weight_decay": .01, "gradient_clipping": None} if on else None,
        "schedule": "300-epoch cosine; no restart; enabled HIER active update1",
        "random_streams": "continuous main RNG; persistent private sampler/data/crop/Gumbel; no per-step reseed",
        "train_shard_order": list(TRAIN_SHARD_ORDER), "initialization": "original CPU construction seed4780",
        "cudnn_benchmark": True, "cudnn_deterministic": True,
        "canonical_name": name, "display_name": name, "validation": "none",
        "training_pool": "official train9840",
        "selection": "official test2468 every epoch; strictly higher unrounded OA; first tie retained",
        "selection_forward": "eval/no_grad; preserves RNG/BN; no early stopping",
        "train_eval_every": args.train_eval_every, "eval_batch_size": args.eval_batch_size,
        "workers": args.workers, "audit_epochs": list(args.audit_epochs), "save_every": args.save_every,
        "smoke": args.smoke, "evaluation_batch_limit": args.smoke_eval_batches if args.smoke else None}


def array_sha256(array):
    return hashlib.sha256(memoryview(np.ascontiguousarray(array))).hexdigest()


def load_shards(data_dir, partition):
    paths = sorted(Path(data_dir).resolve(strict=True).glob("ply_data_" + partition + "*.h5"))
    if not paths:
        raise FileNotFoundError("No ModelNet40 " + partition + " shards")
    if partition == "train":
        order = tuple(json.loads(os.environ.get("HYCORE_TRAIN_SHARD_ORDER", json.dumps(TRAIN_SHARD_ORDER))))
        by_name = {path.name: path for path in paths}
        if order != TRAIN_SHARD_ORDER or set(order) != set(by_name):
            raise ValueError("Training shard order must match original1,4,3,2,0")
        paths = [by_name[name] for name in order]
    chunks, gold = [], []
    for path in paths:
        with h5py.File(path, "r") as handle:
            chunks.append(handle["data"][:].astype(np.float32))
            gold.append(handle["label"][:].astype(np.int64).reshape(-1))
    return np.concatenate(chunks), np.concatenate(gold), [path.name for path in paths]


def make_generators(seed, rank, device, hier_enabled):
    return {
        "loader": torch.Generator().manual_seed(seed + 100001 + rank * 1000003),
        "crop": torch.Generator().manual_seed(seed + 200001),
        "gumbel": torch.Generator(device=device).manual_seed(seed + 300001) if hier_enabled else None}


def capture_generators(generators):
    return {name: gen.get_state().cpu().clone() if gen is not None else None for name, gen in generators.items()}


def restore_generators(generators, states):
    if set(generators) != set(states):
        raise RuntimeError("Resume private RNG streams differ")
    for name, gen in generators.items():
        if (gen is None) != (states[name] is None):
            raise RuntimeError("Resume HIER RNG presence differs")
        if gen is not None:
            gen.set_state(states[name].cpu())


def shared_crop_counts(generator, device):
    values = torch.tensor([int(torch.randint(800, 1025, (), generator=generator)),
                           int(torch.randint(200, 601, (), generator=generator))], dtype=torch.int64, device=device)
    dist.broadcast(values, src=0)
    return int(values[0]), int(values[1])


def checkpoint_rank_state(backbone, sampler, rank, generators):
    return {"rank": rank, "rng": capture_rng(), "sampler": sampler.state_dict(),
        "private_generators": capture_generators(generators),
        "BN_buffers": {name: value.detach().cpu().clone() for name, value in backbone.named_buffers()}}


def validate_resume(saved, config, split_sha, train_ids, selection_ids):
    if saved.get("format") != CHECKPOINT_VERSION or saved.get("model_selection_only", False):
        raise RuntimeError("Resume requires full cycle-B64 checkpoint")
    if saved.get("training_config") != config or saved.get("split_sha256") != split_sha:
        raise RuntimeError("Resume configuration/split differs")
    if not np.array_equal(saved["train_ids"], train_ids) or not np.array_equal(saved["selection_ids"], selection_ids):
        raise RuntimeError("Resume IDs differ")
    if len(saved["rank_states"]) != 2 or not 0 <= int(saved["completed_epochs"]) <= config["epochs"]:
        raise RuntimeError("Resume rank states/epoch invalid")
    best, metrics = saved.get("best"), saved.get("best_selection_metrics")
    prefix = "smoke_train" if config["smoke"] else "test"
    if best is None or saved.get("best_net") is None or metrics is None:
        raise RuntimeError("Resume lacks selected weights/matching metrics")
    if best[prefix + "_oa"] != metrics[prefix + "_oa"] or int(best["epoch"]) > saved["completed_epochs"]:
        raise RuntimeError("Resume selected model/metrics identity mismatch")


def _optional_state(module):
    return None if module is None else module.state_dict()


def main():
    args = arguments()
    enabled = args.hier == "on"
    torch.set_num_threads(2)
    rank, local_rank = int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"])
    if int(os.environ["WORLD_SIZE"]) != 2:
        raise RuntimeError("Cycle-B64 requires exactly two ranks")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", timeout=dt.timedelta(minutes=10))
    started_wall = time.perf_counter()
    if rank == 0:
        args.run_dir.mkdir(parents=True, exist_ok=False)
    dist.barrier()
    config = fixed_configuration(args)
    manifest = {
        "completed_optimizer_steps": 0, "status": "starting", "main_training": not args.smoke, "diagnostic_only": args.smoke,
        "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "commit": os.environ.get("V9C_SOURCE_COMMIT") or os.environ.get("V9_SOURCE_COMMIT") or subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        "branch": os.environ.get("V9C_SOURCE_BRANCH") or os.environ.get("V9_SOURCE_BRANCH") or subprocess.check_output(["git", "branch", "--show-current"], cwd=REPO, text=True).strip(),
        "command_argv": [sys.executable, *sys.argv],
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "canonical_name": config["canonical_name"], "training_config": config,
        "visible_gpus": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "physical_gpus": os.environ.get("HYCORE_PHYSICAL_GPUS", os.environ.get("CUDA_VISIBLE_DEVICES")),
        "physical_gpu_uuids": os.environ.get("HYCORE_PHYSICAL_GPU_UUIDS"),
        "runtime": {"torch": torch.__version__, "geoopt": geoopt.__version__, "numpy": np.__version__},
        "initialization": {"method": "original CPU random initialization", "seed": args.seed, "source_checkpoint": None, "teacher": None},
        "official_test_read": False, "resume_identity": None,
        "representation_domains": {"whole_part": "c.1 cap2.3 baseGS once", "HIER_view_and_proxies": "c.1 wholedepth4.6 GS once on separate fork" if enabled else "absent"}}
    try:
        points, labels, shards = load_shards(args.data_dir, "train")
        if array_sha256(labels) != EXPECTED_TRAIN_LABELS_SHA:
            raise RuntimeError("Training label rows differ from original reference")
        train_ids = full_training_ids(labels)
        if args.smoke:
            selection_ids = train_ids[:args.eval_batch_size * args.smoke_eval_batches]
            selection_points, selection_labels = points[selection_ids], labels[selection_ids]
            selection_shards, selection_prefix = [], "smoke_train"
        else:
            selection_points, selection_labels, selection_shards = load_shards(args.data_dir, "test")
            if len(selection_labels) != 2468 or array_sha256(selection_labels) != EXPECTED_TEST_LABELS_SHA:
                raise RuntimeError("Expected original official test2468 row identity")
            selection_ids = np.arange(len(selection_labels), dtype=np.int64)
            selection_prefix = "test"
            manifest["official_test_read"] = True
        split_sha = split_identity(train_ids, selection_ids)
        manifest.update(split_sha256=split_sha, train_size=len(train_ids), validation_size=0,
            selection_size=len(selection_ids), selection_domain=selection_prefix,
            training_shards=shards, selection_shards=selection_shards,
            selection_labels_sha256=array_sha256(selection_labels), train_labels_sha256=array_sha256(labels),
            train_points_sha256=array_sha256(points),
            training_shard_sha256={name: checkpoint_sha(args.data_dir / name) for name in shards})
        seed_all(args.seed)
        backbone = configure_backbone(Hype_pointMLP())
        freeze_curvature(backbone)
        initial_digest = state_digest(backbone)
        if initial_digest != EXPECTED_ORIGINAL_INITIAL_DIGEST:
            raise RuntimeError("Initial state differs from original seed4780 c.1/cap/GS: " + initial_digest)
        initial_hash = state_hash(backbone)
        manifest.update(initialized_model_sha256=initial_hash, original_initial_state_digest=initial_digest)
        backbone = backbone.to(device)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True
        # Preserve continuous main RNG. Rank-local streams start once only;
        # neither training epochs nor individual steps reseed them.
        random.seed(args.seed + rank * 1000003)
        np.random.seed(args.seed + rank * 1000003)
        torch.cuda.manual_seed(args.seed + rank * 1000003)
        model = DDP(HyCoReTrainingForward(backbone, hier_enabled=enabled), device_ids=[local_rank], broadcast_buffers=True)
        model_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        optimizer = geoopt.optim.RiemannianSGD(model_parameters, lr=.1, momentum=.9, weight_decay=2e-4)
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=.005)
        proxy, proxy_optimizer, proxy_scheduler = None, None, None
        if enabled:
            from ..hier_proxy_scratch_v8.hier_loss import HIERLoss
            proxy = HIERLoss(seed=args.seed).to(device)
            proxy.assert_geometry()
            broadcast_parameters(proxy)
            proxy_optimizer = torch.optim.AdamW(proxy.parameters(), lr=.01, betas=(.9, .999), eps=1e-8, weight_decay=.01)
            proxy_scheduler = CosineAnnealingLR(proxy_optimizer, T_max=args.epochs, eta_min=.0005)
        generators = make_generators(args.seed, rank, device, enabled)
        dataset = PointClouds(points[train_ids], labels[train_ids], 1024, "train", args.seed)
        sampler = ClassCycleBatchSampler(labels[train_ids], rank=rank, seed=args.seed, steps=args.steps_per_epoch)
        start_epoch, best, best_net, best_proxy, best_selection_metrics = 0, None, None, None, None
        if args.resume is not None:
            saved = torch.load(args.resume, map_location="cpu", weights_only=False)
            validate_resume(saved, config, split_sha, train_ids, selection_ids)
            if saved.get("train_points_sha256") != manifest["train_points_sha256"]:
                raise RuntimeError("Resume training data rows differ")
            backbone.load_state_dict(saved["net"], strict=True)
            freeze_curvature(backbone)
            optimizer.load_state_dict(saved["optimizer"])
            scheduler.load_state_dict(saved["scheduler"])
            if enabled:
                proxy.load_state_dict(saved["proxy"], strict=True)
                proxy.assert_geometry()
                proxy_optimizer.load_state_dict(saved["proxy_optimizer"])
                proxy_scheduler.load_state_dict(saved["proxy_scheduler"])
            elif any(saved.get(key) is not None for key in ("proxy", "proxy_optimizer", "proxy_scheduler", "best_proxy")):
                raise RuntimeError("No-HIER resume contains hierarchy state")
            rank_state = saved["rank_states"][rank]
            sampler.load_state_dict(rank_state["sampler"])
            buffers = dict(backbone.named_buffers())
            for name, value in rank_state["BN_buffers"].items():
                buffers[name].copy_(value.to(buffers[name]))
            restore_rng(rank_state["rng"])
            restore_generators(generators, rank_state["private_generators"])
            start_epoch = int(saved["completed_epochs"])
            best, best_net = saved["best"], saved["best_net"]
            best_proxy, best_selection_metrics = saved.get("best_proxy"), saved["best_selection_metrics"]
            manifest["initialization"] = saved["initialization"]
            manifest["initialized_model_sha256"] = saved["initialized_model_sha256"]
            manifest["resume_identity"] = {"path": str(args.resume), "sha256": checkpoint_sha(args.resume),
                "completed_epochs": start_epoch, "source_commit": saved["commit"]}
            if rank == 0:
                atomic_torch(args.run_dir / "best.pth", {
                    "format": CHECKPOINT_VERSION, "model_selection_only": True, "net": best_net, "proxy": best_proxy,
                    "best": best, "epoch": best["epoch"],
                    "metrics": {"test": best_selection_metrics} if not args.smoke else {"smoke_training_selection": best_selection_metrics},
                    "training_config": config, "split_sha256": split_sha, "resume_source": manifest["resume_identity"]})
            del saved
        elif rank == 0:
            atomic_torch(args.run_dir / "initialization_e0.pth", {
                "format": CHECKPOINT_VERSION, "model_selection_only": True, "initialization_only": True,
                "net": cpu_state(backbone), "proxy": cpu_state(proxy) if enabled else None,
                "training_config": config, "initialization": manifest["initialization"],
                "initialized_model_sha256": initial_hash, "original_initial_state_digest": initial_digest, "commit": manifest["commit"]})
        devices = [None, None]
        dist.all_gather_object(devices, {"rank": rank, "local_rank": local_rank,
            "name": torch.cuda.get_device_name(device),
            "uuid": str(getattr(torch.cuda.get_device_properties(device), "uuid", "unavailable")),
            "total_MiB": torch.cuda.get_device_properties(device).total_memory / 1024 ** 2})
        manifest["rank_devices"] = devices
        manifest.update(status="running", completed_epochs=start_epoch)
        if rank == 0:
            atomic_json(args.run_dir / "manifest.json", manifest)

        for epoch_index in range(start_epoch, args.epochs):
            epoch = epoch_index + 1
            model.train()
            sampler.set_epoch(epoch_index)
            loader = DataLoader(dataset, batch_sampler=sampler, num_workers=args.workers,
                worker_init_fn=worker_seed, pin_memory=True, generator=generators["loader"])
            telemetry = EpochTelemetry(train_ids, labels[train_ids], num_classes=40, hier_enabled=enabled) if rank == 0 else None
            structure_telemetry = StructureAccumulator() if enabled and rank == 0 else None
            learning_rates = {"model": optimizer.param_groups[0]["lr"], "proxy": proxy_optimizer.param_groups[0]["lr"] if enabled else None}
            epoch_started = time.perf_counter()
            torch.cuda.reset_peak_memory_stats(device)
            step_count = 0
            for step, (cloud, local_gold, subset_ids) in enumerate(loader):
                step_count += 1
                cloud_sha = array_sha256(cloud.numpy()) if epoch == 1 and step == 0 else None
                cloud, local_gold = cloud.to(device), local_gold.to(device)
                local_ids = torch.as_tensor(train_ids[subset_ids.numpy()], device=device)
                whole_count, child_count = shared_crop_counts(generators["crop"], device)
                optimizer.zero_grad(set_to_none=True)
                if enabled:
                    proxy_optimizer.zero_grad(set_to_none=True)
                if enabled and step == 0 and epoch in args.audit_epochs:
                    audit_rng, audit_generator_state = capture_rng(), capture_generators(generators)
                    try:
                        audit = audit_parameter_gradients(model, proxy, cloud, local_gold, local_ids,
                            whole_count, child_count, topk=20, seed=args.seed + epoch_index * 10000019, lambda_hier=.5)
                    finally:
                        restore_rng(audit_rng)
                        restore_generators(generators, audit_generator_state)
                    if rank == 0:
                        atomic_json(args.run_dir / f"parameter_audit_epoch_{epoch:03d}.json", audit)
                    optimizer.zero_grad(set_to_none=True)
                    proxy_optimizer.zero_grad(set_to_none=True)
                bn_layers = [module for module in backbone.modules() if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)]
                bn_before = [int(module.num_batches_tracked) for module in bn_layers]
                started = time.perf_counter()
                local_out = model(cloud, whole_count, child_count)
                values = global_base_losses(local_out, local_gold)
                ids = gather_without_grad(local_ids)
                hier_mu, hier_loss, hier_stats, sample_mining, structure_stats = None, None, None, None, None
                proxy_cap_stats, proxy_updates, proxy_grad_stats = {}, {}, {}
                if enabled:
                    hier_mu = whole_to_hier(values["hier_source_mu"])
                    assert_hier_shell(hier_mu)
                    relation_seed = args.seed + (epoch_index * args.steps_per_epoch + step) * 10007
                    hier_loss, hier_stats = proxy(hier_mu, values["gold"], topk=20, seed=relation_seed,
                        data_ids=ids, generator=generators["gumbel"], return_mining=True, exclude_self_negative=True)
                    sample_mining, structure_stats = hier_stats.pop("_sample_mining"), hier_stats.pop("structure")
                    synchronized_assert(all(hier_stats[key]["self_k_triplets"] == 0 for key in ("sample", "proxy")),
                        "Sample/proxy self-k must both be zero", device)
                    loss = values["base"] + .5 * hier_loss
                else:
                    loss = values["base"]
                if step == 0 and epoch in args.audit_epochs and rank == 0:
                    snapshot = {"stage": "actual_training_pre_optimizer", "epoch": epoch, "step": step + 1,
                        "completed_optimizer_steps": epoch_index * args.steps_per_epoch + step,
                        "commit": manifest["commit"], "training_config": config, "rank_order": [0, 1],
                        "ids": ids.detach().cpu(), "labels": values["gold"].detach().cpu(),
                        "whole_count": whole_count, "child_count": child_count, "native_c": .1,
                        "hier_view_present": enabled, "proxy_parameter_count": 512 * 256 if enabled else 0,
                        "native_mu": values["mu"].detach().cpu(), "native_nu": values["nu"].detach().cpu(),
                        "BN_counters_before_local_rank0": bn_before,
                        "BN_counters_after_local_rank0": [int(layer.num_batches_tracked) for layer in bn_layers],
                        "input_stats_local_rank0": local_out["input_stats"]}
                    if enabled:
                        snapshot.update(hier_c=.1, hier_target_depth=HIER_DEPTH,
                            hier_source_mu=values["hier_source_mu"].detach().cpu(), hier_mu=hier_mu.detach().cpu(),
                            proxy_tangent=proxy.tangent_proxies.detach().cpu().clone(),
                            proxy_ball=proxy.proxies().detach().cpu(), structure=structure_stats)
                    atomic_torch(args.run_dir / f"actual_training_epoch_{epoch:03d}.pth", snapshot)
                synchronized_assert(bool(torch.isfinite(loss)), "Nonfinite objective", device)
                loss.backward()
                all_parameters = model_parameters + (list(proxy.parameters()) if enabled else [])
                synchronized_assert(finite_tensors(parameter.grad for parameter in all_parameters), "Nonfinite gradients", device)
                if enabled:
                    sync_parameter_gradients(proxy.parameters())
                    proxy_grad_stats = radial_angular_gradient(proxy.tangent_proxies, proxy.tangent_proxies.grad)
                bn_delta = [int(module.num_batches_tracked) - before for module, before in zip(bn_layers, bn_before)]
                synchronized_assert(all(delta == 2 for delta in bn_delta), "Part/whole BN update must equal2", device)
                synchronized_assert(all(local_out["input_stats"][key] for key in
                    ("child_aliases_whole", "whole_aliases_source", "whole_changed_by_child")), "Part-overwrite behavior changed", device)
                preclip_norm = torch.nn.utils.clip_grad_norm_(model_parameters, 1.)
                postclip_norm = gradient_norm(model_parameters)
                synchronized_assert(bool(torch.isfinite(preclip_norm)) and bool(torch.isfinite(postclip_norm)), "Nonfinite clipped norm", device)
                optimizer.step()
                replica_difference = torch.tensor(0., device=device)
                if enabled:
                    before_proxy = proxy.tangent_proxies.detach().clone()
                    proxy_optimizer.step()
                    proxy_cap_stats = proxy.project_after_step(proxy_optimizer)
                    proxy_updates = proxy_update_stats(before_proxy, proxy.tangent_proxies)
                    twin = proxy.tangent_proxies.detach().clone()
                    dist.broadcast(twin, src=0)
                    replica_difference = (twin - proxy.tangent_proxies.detach()).abs().max()
                    dist.all_reduce(replica_difference, op=dist.ReduceOp.MAX)
                    synchronized_assert(float(replica_difference) <= 1e-7, "Proxy replicas diverged", device)
                synchronized_assert(finite_tensors(all_parameters), "Nonfinite updated parameters", device)
                torch.cuda.synchronize(device)
                elapsed = torch.tensor(time.perf_counter() - started, device=device)
                dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
                rank_peaks = [None, None]
                dist.all_gather_object(rank_peaks, {"rank": rank,
                    "peak_allocated_MiB": torch.cuda.max_memory_allocated(device) / 1024 ** 2,
                    "BN_update_min": min(bn_delta), "BN_update_max": max(bn_delta),
                    "augmented_cloud_sha256": cloud_sha, "input": local_out["input_stats"]})
                scalars = {
                    "loss": float(loss.detach()), "base": float(values["base"].detach()), "ce": float(values["ce"].detach()),
                    "lambda_hier": .5 if enabled else 0., "intra_contrastive": float(values["intra_contrastive"].detach()),
                    "intra_radial": float(values["intra_radial"].detach()),
                    "model_preclip_norm": float(preclip_norm), "model_postclip_norm": float(postclip_norm),
                    "model_clip_applied": float(preclip_norm > 1.), "model_clip_scale": min(1., 1. / (float(preclip_norm) + 1e-6)),
                    "step_seconds_max_rank": float(elapsed),
                    "peak_allocated_MiB_max_rank": max(item["peak_allocated_MiB"] for item in rank_peaks),
                    "whole_count": whole_count, "child_count": child_count,
                    "whole_depth": float(values["whole_depth"]), "part_depth": float(values["part_depth"]),
                    "native_whole_norm_min": float(values["mu"].detach().norm(dim=-1).min()),
                    "hier_calls": int(enabled), "proxy_parameter_count": 512 * 256 if enabled else 0,
                    "flip_same_class_fraction": float((values["gold"] == values["gold"].flip(0)).float().mean())}
                if enabled:
                    scalars.update(hier_monitored_loss=float(hier_loss.detach()),
                        proxy_replica_max_difference=float(replica_difference),
                        hier_whole_depth=hier_stats["geometry"]["whole"]["depth_median"],
                        hier_whole_depth_max=hier_stats["geometry"]["whole"]["depth_max"],
                        **{"proxy_cap_" + key: value for key, value in proxy_cap_stats.items()},
                        **{"proxy_" + key: value for key, value in proxy_updates.items() if key != "semantics"},
                        **{"proxy_gradient_" + key: value for key, value in proxy_grad_stats.items() if key != "semantics"})
                if rank == 0:
                    if epoch == 1 and step == 0:
                        manifest["first_batch_identity"] = {
                            "ids": ids.detach().cpu().tolist(), "ids_sha256": array_sha256(ids.detach().cpu().numpy()),
                            "labels_sha256": array_sha256(values["gold"].detach().cpu().numpy()),
                            "whole_count": whole_count, "child_count": child_count,
                            "rank_augmented_cloud_sha256": [item["augmented_cloud_sha256"] for item in rank_peaks]}
                        atomic_json(args.run_dir / "first_batch_identity.json", manifest["first_batch_identity"])
                        atomic_json(args.run_dir / "manifest.json", manifest)
                    if enabled:
                        structure_telemetry.update(structure_stats)
                    telemetry.update(ids=ids, labels=values["gold"], logits=values["logits"], mu=values["mu"], nu=values["nu"],
                        scalar_metrics=scalars, hier_stats=hier_stats, sample_mining=sample_mining)
                    heartbeat = {"status": "running", "epoch": epoch, "step": step + 1, "steps_per_epoch": args.steps_per_epoch,
                        "optimizer_steps": epoch_index * args.steps_per_epoch + step + 1, "completed_optimizer_steps": epoch_index * args.steps_per_epoch + step + 1,
                        "phase": "joint" if enabled else "base_only", "updated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                        "lr": learning_rates, "scalars": scalars, "rank_observations": rank_peaks}
                    atomic_json(args.run_dir / "heartbeat.json", heartbeat)
                    append_jsonl(args.run_dir / "steps.jsonl", {"epoch": epoch, "step": step + 1,
                        "ids": ids.detach().cpu().tolist(), "hier": hier_stats, **scalars})
                    if step == 0 or (step + 1) % 20 == 0 or step + 1 == args.steps_per_epoch:
                        print(json.dumps({"epoch": epoch, "step": step + 1, "loss": scalars["loss"],
                            "phase": heartbeat["phase"], "lr": learning_rates}, allow_nan=False), flush=True)
                del local_out, values, loss, hier_mu, hier_loss, sample_mining, hier_stats, structure_stats
            if step_count != args.steps_per_epoch:
                raise RuntimeError("Unexpected update count")
            scheduler.step()
            if enabled:
                proxy_scheduler.step()
            selection = rank0_evaluation(backbone, selection_points, selection_labels, args, device, selection_prefix, rank)
            clean_train = None
            if epoch % args.train_eval_every == 0 or epoch == args.epochs:
                clean_train = rank0_evaluation(backbone, points[train_ids], labels[train_ids], args, device, "train_eval", rank)
            candidate = select_best(best, selection, selection_prefix, epoch)
            improved = candidate is not best
            best = candidate
            if improved:
                best_selection_metrics = selection
                if rank == 0:
                    best_net, best_proxy = cpu_state(backbone), cpu_state(proxy) if enabled else None
            rank_states = [None, None]
            dist.all_gather_object(rank_states, checkpoint_rank_state(backbone, sampler, rank, generators))
            checkpoint_error = None
            if rank == 0:
                row = {"epoch": epoch, "phase": "joint" if enabled else "base_only", "lambda_hier": .5 if enabled else 0.,
                    "lr_used": learning_rates,
                    "lr_next": {"model": optimizer.param_groups[0]["lr"], "proxy": proxy_optimizer.param_groups[0]["lr"] if enabled else None},
                    "epoch_wall_seconds": time.perf_counter() - epoch_started, "optimizer_steps": epoch * args.steps_per_epoch, "completed_optimizer_steps": epoch * args.steps_per_epoch,
                    "epoch_optimizer_steps": step_count, "telemetry": telemetry.summary(),
                    "structure": structure_telemetry.summary() if enabled else {"available": False, "reason": "no HIER branch or proxies"},
                    "selection_domain": selection_prefix, "test": selection if not args.smoke else None,
                    "smoke_training_selection": selection if args.smoke else None,
                    "clean_train": clean_train, "best": best, "best_updated": improved}
                payload = {"format": CHECKPOINT_VERSION, "model_selection_only": False, "completed_epochs": epoch, "completed_optimizer_steps": epoch * args.steps_per_epoch, "epoch": epoch,
                    "net": backbone.state_dict(), "proxy": _optional_state(proxy), "optimizer": optimizer.state_dict(),
                    "proxy_optimizer": _optional_state(proxy_optimizer), "scheduler": scheduler.state_dict(),
                    "proxy_scheduler": _optional_state(proxy_scheduler), "rank_states": rank_states,
                    "training_config": config, "split_sha256": split_sha, "train_ids": train_ids, "selection_ids": selection_ids,
                    "selection_domain": selection_prefix, "validation_size": 0, "train_points_sha256": manifest["train_points_sha256"],
                    "commit": manifest["commit"], "branch": manifest["branch"], "initialization": manifest["initialization"],
                    "initialized_model_sha256": manifest["initialized_model_sha256"], "original_initial_state_digest": initial_digest,
                    "best": best, "best_net": best_net, "best_proxy": best_proxy,
                    "best_selection_metrics": best_selection_metrics, "metrics": row, "saved_utc": dt.datetime.now(dt.timezone.utc).isoformat()}
                try:
                    last_path = args.run_dir / "last.pth"
                    atomic_torch(last_path, payload)
                    digest, saved_files = checkpoint_sha(last_path), ["last.pth"]
                    if improved:
                        atomic_copy(last_path, args.run_dir / "best.pth")
                        saved_files.append("best.pth")
                    if epoch % args.save_every == 0 or epoch in (1, 5, 99):
                        name = f"checkpoint_epoch_{epoch:03d}.pth"
                        atomic_copy(last_path, args.run_dir / name)
                        saved_files.append(name)
                    row["checkpoint"] = {"epoch": epoch, "files": saved_files, "last_sha256": digest,
                        "last_bytes": last_path.stat().st_size, "saved_utc": payload["saved_utc"]}
                    atomic_json(args.run_dir / f"metrics_epoch_{epoch:03d}.json", row)
                    append_jsonl(args.run_dir / "metrics.jsonl", row)
                    manifest.update(status="running", completed_epochs=epoch, optimizer_steps=epoch * args.steps_per_epoch, completed_optimizer_steps=epoch * args.steps_per_epoch,
                        best=best, latest_checkpoint=row["checkpoint"], updated_utc=dt.datetime.now(dt.timezone.utc).isoformat())
                    atomic_json(args.run_dir / "manifest.json", manifest)
                    print(json.dumps({"epoch": epoch, "selection_domain": selection_prefix,
                        "selection_oa": selection[selection_prefix + "_oa"], "best": best}, allow_nan=False), flush=True)
                except BaseException as exc:
                    checkpoint_error = repr(exc)
            error = [checkpoint_error]
            dist.broadcast_object_list(error, src=0)
            if error[0] is not None:
                raise RuntimeError("Epoch checkpoint failed: " + error[0])
            dist.barrier()
        final_error, test_metrics = None, None
        dist.barrier()
        if rank == 0 and not args.smoke:
            try:
                selected = torch.load(args.run_dir / "best.pth", map_location="cpu", weights_only=False)
                if selected["epoch"] != best["epoch"] or selected["best"] != best:
                    raise RuntimeError("Selected checkpoint best identity differs")
                test_metrics = selected["metrics"]["test"]
                if test_metrics["test_count"] != 2468 or test_metrics["test_oa"] != best["test_oa"]:
                    raise RuntimeError("Selected checkpoint official test identity differs")
                atomic_json(args.run_dir / "final_test.json", {"selected_by": "strictly higher unrounded test OA; first tie retained",
                    "test_used_for_selection": True, "fresh_final_forward": False, "best": best,
                    "best_checkpoint_sha256": checkpoint_sha(args.run_dir / "best.pth"), "test_shards": selection_shards, **test_metrics})
            except BaseException as exc:
                final_error = repr(exc)
        error = [final_error]
        dist.broadcast_object_list(error, src=0)
        if error[0] is not None:
            raise RuntimeError("Final test summary failed: " + error[0])
        dist.barrier()
        if rank == 0:
            manifest.update(status="completed", completed_epochs=args.epochs, optimizer_steps=args.epochs * args.steps_per_epoch, completed_optimizer_steps=args.epochs * args.steps_per_epoch,
                finished_utc=dt.datetime.now(dt.timezone.utc).isoformat(), wall_seconds=time.perf_counter() - started_wall,
                best=best, final_test=test_metrics, final_summary_skipped=args.smoke, test_used_for_selection=not args.smoke)
            atomic_json(args.run_dir / "manifest.json", manifest)
            atomic_json(args.run_dir / "heartbeat.json", {"status": "completed", "epoch": args.epochs,
                "updated_utc": manifest["finished_utc"], "best": best})
            print("V9C_CYCLE_B64_" + args.hier.upper() + "_COMPLETED", flush=True)
    except BaseException as exc:
        if rank == 0:
            manifest.update(status="failed", error=repr(exc), finished_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                wall_seconds=time.perf_counter() - started_wall)
            atomic_json(args.run_dir / "manifest.json", manifest)
            atomic_json(args.run_dir / "heartbeat.json", {"status": "failed", "error": repr(exc)})
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
