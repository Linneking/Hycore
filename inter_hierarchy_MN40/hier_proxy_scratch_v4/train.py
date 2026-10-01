"""Matched B0/H3/H5, 200 total epochs including an exact shared 20-epoch prefix.

Whole input is not overwritten by child sampling. Part BN uses batch
statistics without updating running statistics; only whole updates them.
The original HyCoRe radial/contrastive expressions and label smoothing stay.
Checkpoint selection uses a deterministic validation split; official test
is read only once after the fixed training budget, without early stopping.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
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

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "pointnet2_ops_lib"))
sys.path.insert(0, str(HERE.parent))

import geoopt
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

from hutil import cal_loss, knn, hype_triplet_losses
from models.pointmlp import Hype_pointMLP
from hier_proxy_scratch_v3.train import (
    atomic_json, atomic_torch, capture_rng, restore_rng, freeze_curvature,
    git_checkout_state, initialize_proxies_from_features, model_optimizer,
)
from hier_proxy_v2_online.train import (
    PointClouds, load_shards, seed_all, initialization_features,
    assert_manifold_contract, sha256, triplet_columns,
)
from hier_proxy_v2_online.diagnostics import fixed_rng
from hier_proxy_scratch_v4.geometry import (
    ProxyHierarchy, DEFAULT_MARGIN, DEFAULT_TAU, source_cap_intra_audit,
)
from hier_proxy_scratch_v4.mining import mine_batch_triplets
from hier_proxy_scratch_v4.sampler import EpochCoveringClassBatchSampler


def utc_now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arm", choices=("B0", "H3", "H5"), required=True)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--shared-warmup-dir", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--warmup-epochs", type=int, default=20)
    p.add_argument("--ramp-epochs", type=int, default=20)
    p.add_argument("--seed", type=int, default=22)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--sample-t", type=int, default=64)
    p.add_argument("--proxy-t", type=int, default=32)
    p.add_argument("--proxy-count", type=int, default=256)
    p.add_argument("--proxy-k", type=int, default=20)
    p.add_argument("--gradient-mode", choices=("euclidean", "riemannian"), default="euclidean")
    p.add_argument("--inter-weight", type=float, default=0.03)
    p.add_argument("--proxy-weight", type=float, default=0.03)
    p.add_argument("--margin", type=float, default=DEFAULT_MARGIN)
    p.add_argument("--tau", type=float, default=DEFAULT_TAU)
    p.add_argument("--proxy-lr", type=float, default=0.005)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--min-lr", type=float, default=0.005)
    p.add_argument("--weight-decay", type=float, default=2e-4)
    p.add_argument("--validation-fraction", type=float, default=0.1)
    p.add_argument("--save-every", type=int, default=20)
    p.add_argument("--max-batches", type=int, default=0, help="smoke only; zero is full epoch")
    p.add_argument("--skip-final-test", action="store_true", help="smoke only")
    p.add_argument("--device", default="cuda:0")
    a = p.parse_args()
    if not 0 < a.warmup_epochs < a.epochs or not 0 < a.validation_fraction < 0.3:
        p.error("invalid epoch budget or validation fraction")
    if min(a.sample_t, a.proxy_t, a.proxy_k, a.save_every, a.ramp_epochs) < 1:
        p.error("counts must be positive")
    if a.proxy_k >= a.proxy_count or a.workers < 0 or a.max_batches < 0:
        p.error("invalid proxy neighborhood/workers/batch limit")
    if any(not math.isfinite(v) or v <= 0 for v in
           (a.lr, a.proxy_lr, a.tau, a.margin, a.inter_weight, a.proxy_weight)):
        p.error("rates, margins and weights must be finite and positive")
    a.batch_size = 32
    a.num_points = 1024
    a.teacher_views = 1
    a.k = 3 if a.arm != "H5" else 5
    return a


def json_args(args):
    return {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}


def stratified_split(labels, seed=22, fraction=0.1):
    rng = np.random.default_rng(seed + 1701)
    train, validation = [], []
    for label in np.unique(labels):
        indices = rng.permutation(np.flatnonzero(labels == label))
        count = max(1, int(round(len(indices) * fraction)))
        if len(indices) - count < 8:
            raise ValueError("each training class needs eight distinct instances")
        validation.extend(indices[:count].tolist())
        train.extend(indices[count:].tolist())
    return np.asarray(sorted(train)), np.asarray(sorted(validation))


def split_identity(train, validation):
    digest = hashlib.sha256()
    digest.update(train.astype(np.int64).tobytes())
    digest.update(validation.astype(np.int64).tobytes())
    return digest.hexdigest()


def sample_subcloud(data, kmin, kmax):
    """Same random-center kNN construction, independent output storage."""
    count = random.randint(kmin, min(kmax, data.shape[-1]))
    neighbors = knn(data[:, :3], count)
    centers = torch.tensor([random.randrange(data.shape[-1]) for _ in range(len(data))],
                           device=data.device)
    selected = neighbors[torch.arange(len(data), device=data.device), centers]
    cloud = data.gather(2, selected[:, None].expand(-1, data.shape[1], -1)).contiguous()
    return cloud, count


@contextmanager
def part_batch_statistics_only(model):
    layers = [m for m in model.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
    previous = [m.track_running_stats for m in layers]
    try:
        for m in layers:
            m.track_running_stats = False
        yield
    finally:
        for m, flag in zip(layers, previous):
            m.track_running_stats = flag


def base_forward(model, points, labels, device):
    source = points.to(device).transpose(1, 2).contiguous()
    whole, _ = sample_subcloud(source, 800, 1024)
    child, count = sample_subcloud(whole, 200, 600)
    with part_batch_statistics_only(model):
        nu, _ = model(child, emb=True)
    mu, logits = model(whole)
    _, _, dp, dn, contrastive, radial = hype_triplet_losses(
        mu, nu, hier_margin=1000.0 / count, contr_margin=4, ball_dim=256)
    ce = cal_loss(logits, labels.to(device).reshape(-1))
    base = ce + 0.01 * (contrastive + radial)
    return {"mu": mu, "logits": logits, "base": base, "ce": ce,
            "intra_contrastive": contrastive, "intra_radial": radial,
            "positive_distance": dp, "negative_distance": dn,
            "whole_radius": mu.detach().norm(dim=-1).mean(),
            "part_radius": nu.detach().norm(dim=-1).mean()}


def make_loader(dataset, labels, args, epoch):
    sampler = EpochCoveringClassBatchSampler(labels, seed=args.seed)
    sampler.set_epoch(epoch)
    generator = torch.Generator().manual_seed(args.seed + epoch * 1009)
    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=args.workers,
                        generator=generator, pin_memory=True)
    return loader, sampler


@torch.no_grad()
def evaluate(model, loader, device, prefix="val"):
    model.eval()
    matrix = torch.zeros(40, 40, dtype=torch.long)
    total_loss, count = 0.0, 0
    for points, labels, _ in loader:
        _, logits = model(points.to(device).transpose(1, 2).contiguous())
        total_loss += float(cal_loss(logits, labels.to(device))) * len(labels)
        pred = logits.argmax(-1).cpu()
        matrix += torch.bincount(labels * 40 + pred, minlength=1600).reshape(40, 40)
        count += len(labels)
    return {prefix + "_loss": total_loss / count,
            prefix + "_oa": 100 * float(matrix.diagonal().sum()) / count,
            prefix + "_aa": 100 * float((matrix.diagonal() / matrix.sum(1).clamp_min(1)).mean()),
            prefix + "_class_accuracy": (100 * matrix.diagonal() / matrix.sum(1).clamp_min(1)).tolist()}


def state_hash(model):
    h = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def train_epoch(model, proxy, dataset, labels, optimizer, proxy_optimizer, args, epoch, device):
    model.train()
    seed_all(args.seed + epoch * 1009)
    loader, sampler = make_loader(dataset, labels, args, epoch)
    seen, sums, confusion = set(), {}, torch.zeros(40, 40, dtype=torch.long)
    rows = 0
    first_step = {}
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats(device)
    scale = min(1.0, (epoch - args.warmup_epochs + 1) / args.ramp_epochs) if proxy is not None else 0.0
    for b, (points, gold, ids) in enumerate(loader):
        if args.max_batches and b >= args.max_batches:
            break
        if bool((gold == gold.flip(0)).any()):
            raise RuntimeError("flip negative shares class")
        optimizer.zero_grad(set_to_none=True)
        if proxy_optimizer is not None:
            proxy_optimizer.zero_grad(set_to_none=True)
        values = base_forward(model, points, gold, device)
        loss = values["base"]
        numeric = {key: float(value.detach()) for key, value in values.items()
                   if key not in ("mu", "logits")}
        if proxy is not None:
            mu = proxy.ball.projx(values["mu"].float())
            projected = (mu.detach() - values["mu"].detach()).norm(dim=-1) > 1e-6
            mined = mine_batch_triplets(mu, gold, k=args.k, t_per_anchor=args.sample_t,
                                       seed=args.seed + epoch * 1000003 + b)
            # HIER randomness never consumes the base/data RNG stream.
            with fixed_rng(args.seed + epoch * 1000003 + b, device):
                inter, inter_stats = proxy.sample_loss(mu, triplet_columns(mined["inter"]),
                                                       tau=args.tau, margin=args.margin)
                ploss, proxy_stats = proxy.proxy_loss(k=args.proxy_k, t_per_anchor=args.proxy_t,
                                                      tau=args.tau, margin=args.margin)
            loss = loss + scale * (args.inter_weight * inter + args.proxy_weight * ploss)
            numeric.update(inter=float(inter.detach()), proxy=float(ploss.detach()),
                           weighted_inter=float((scale * args.inter_weight * inter).detach()),
                           weighted_proxy=float((scale * args.proxy_weight * ploss).detach()),
                           projected_samples=float(projected.sum()))
            numeric.update({"mine_" + k: float(v) for k, v in mined["stats"].items()})
            numeric.update({"inter_" + k: float(v) for k, v in inter_stats.items()})
            numeric.update({"proxy_" + k: float(v) for k, v in proxy_stats.items()})
            if b == 0:
                gb = torch.autograd.grad(values["base"], values["mu"], retain_graph=True)[0]
                gi = torch.autograd.grad(inter, values["mu"], retain_graph=True)[0]
                first_step["first_base_mu_grad_norm"] = float(gb.norm())
                first_step["first_inter_mu_grad_norm"] = float(gi.norm())
                first_step["first_weighted_inter_base_grad_ratio"] = float(scale * args.inter_weight * gi.norm() / gb.norm().clamp_min(1e-20))
                first_step["first_base_inter_grad_cosine"] = float(torch.nn.functional.cosine_similarity(gb.flatten(), gi.flatten(), dim=0))
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"nonfinite loss epoch={epoch+1} batch={b}")
        loss.backward()
        # Independent parameter groups: proxy count cannot scale backbone clipping.
        model_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not bool(torch.isfinite(model_norm)):
            raise FloatingPointError("nonfinite model gradient")
        optimizer.step()
        if proxy_optimizer is not None:
            proxy_norm = torch.nn.utils.clip_grad_norm_(proxy.parameters(), 1.0)
            if not bool(torch.isfinite(proxy_norm)):
                raise FloatingPointError("nonfinite proxy gradient")
            proxy_optimizer.step()
            numeric["proxy_grad_norm"] = float(proxy_norm)
        numeric.update(loss=float(loss.detach()), model_grad_norm=float(model_norm))
        for k, value in numeric.items():
            if not math.isfinite(value):
                raise FloatingPointError(f"nonfinite diagnostic {k}")
            sums[k] = sums.get(k, 0.0) + value
        pred = values["logits"].detach().argmax(-1).cpu()
        confusion += torch.bincount(gold * 40 + pred, minlength=1600).reshape(40, 40)
        seen.update(ids.tolist())
        rows += 1
    if not rows:
        raise RuntimeError("no batches")
    result = {"train_" + k: v / rows for k, v in sums.items()}
    result.update(first_step)
    result.update(train_oa=100 * float(confusion.diagonal().sum()) / int(confusion.sum()),
                  train_unique_samples=len(seen), train_unique_coverage=len(seen) / len(dataset),
                  train_batches=rows, train_seconds=time.monotonic() - started,
                  peak_memory_mb=torch.cuda.max_memory_allocated(device) / 1024**2,
                  sampler_plan=sampler.stats, hier_scale=scale)
    if proxy is not None:
        total = sums.get("inter_triplets", 0)
        result["train_inter_collision_rate"] = sums.get("inter_collisions", 0) / max(total, 1)
        result["train_inter_active_rate"] = sums.get("inter_active_triplets", 0) / max(total, 1)
    assert_manifold_contract(model, optimizer)
    return result


def wait_for_shared(args):
    ready = args.shared_warmup_dir / "shared_ready.json"
    while not ready.is_file():
        manifest = args.shared_warmup_dir / "manifest.json"
        if manifest.is_file() and json.loads(manifest.read_text())["status"] == "failed":
            raise RuntimeError("shared warmup producer failed")
        time.sleep(15)
    info = json.loads(ready.read_text())
    checkpoint = args.shared_warmup_dir / "shared_warmup.pth"
    if sha256(checkpoint) != info["checkpoint_sha256"]:
        raise RuntimeError("shared checkpoint checksum mismatch")
    if sha256(args.shared_warmup_dir / "initial_mu.pt") != info["features_sha256"]:
        raise RuntimeError("shared feature checksum mismatch")
    return torch.load(checkpoint, map_location="cpu", weights_only=False), info


def ensure_gpu_idle_before_allocation():
    physical = os.environ.get("HYCORE_PHYSICAL_GPU", os.environ.get("CUDA_VISIBLE_DEVICES"))
    if physical is None or not physical.isdigit():
        raise RuntimeError("require a single physical CUDA_VISIBLE_DEVICES")
    from hier_proxy_scratch_v4.launch_matrix import wait_for_idle_gpu
    return wait_for_idle_gpu(
        int(physical), poll_seconds=15,
        on_wait=lambda _: print("waiting for assigned GPU to become completely idle", flush=True),
    )


def main():
    args = arguments()
    args.run_dir = args.run_dir.resolve()
    args.shared_warmup_dir = args.shared_warmup_dir.resolve()
    args.data_dir = args.data_dir.resolve(strict=True)
    if args.run_dir.exists():
        raise FileExistsError("new run directory required")
    args.run_dir.mkdir(parents=True)
    device = torch.device(args.device)
    points, labels, shards = load_shards(args.data_dir, "train")
    train_ids, validation_ids = stratified_split(labels, args.seed, args.validation_fraction)
    split_sha = split_identity(train_ids, validation_ids)
    np.savez(args.run_dir / "split_ids.npz", train=train_ids, validation=validation_ids)
    manifest = {"status": "starting" if args.arm == "B0" else "waiting_shared_warmup",
                "started_utc": utc_now(), "args": json_args(args), "command": sys.argv,
                "seed": args.seed, "split_sha256": split_sha,
                "source_checkpoint": "random initialization + shared v4 warmup",
                "teacher": None, "train_count": len(train_ids), "validation_count": len(validation_ids),
                "train_shards": shards, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "gpu_physical": os.environ.get("HYCORE_PHYSICAL_GPU", os.environ.get("CUDA_VISIBLE_DEVICES")),
                "gpu_uuid": os.environ.get("HYCORE_PHYSICAL_GPU_UUID"),
                "torch": torch.__version__, "geoopt": geoopt.__version__,
                "fixed_budget_no_early_stopping": True,
                "whole_output_cap": None, "curvature": 1, "distance_units": "native d1",
                "intra_compatibility": source_cap_intra_audit(), **git_checkout_state()}
    path = args.run_dir / "manifest.json"
    atomic_json(path, manifest)
    try:
        source, ready = (None, None) if args.arm == "B0" else wait_for_shared(args)
        ensure_gpu_idle_before_allocation()
        seed_all(args.seed)
        model = Hype_pointMLP().to(device)
        freeze_curvature(model)
        optimizer, scheduler = model_optimizer(model, args)
        start = 0
        if source is not None:
            expected = source["args"]
            for key in ("seed", "epochs", "warmup_epochs", "lr", "min_lr", "weight_decay", "validation_fraction", "workers", "max_batches"):
                if expected[key] != getattr(args, key):
                    raise ValueError("shared training mismatch: " + key)
            if source["split_sha256"] != split_sha or source["epoch"] != args.warmup_epochs:
                raise ValueError("shared split/epoch mismatch")
            model.load_state_dict(source["net"])
            optimizer.load_state_dict(source["optimizer"])
            scheduler.load_state_dict(source["scheduler"])
            restore_rng(source["rng"])
            start = source["epoch"]
            manifest.update(shared_warmup=ready, shared_model_sha256=source["model_sha256"])
            for previous in args.shared_warmup_dir.glob("metrics_epoch_*.json"):
                number = int(previous.stem.rsplit("_", 1)[1])
                if number <= start:
                    shutil.copyfile(previous, args.run_dir / previous.name)
        proxy = proxy_optimizer = proxy_scheduler = None
        if args.arm != "B0":
            reference = torch.load(args.shared_warmup_dir / "initial_mu.pt", map_location="cpu", weights_only=False)
            proxy = ProxyHierarchy(args.proxy_count, 256, seed=args.seed,
                                   gradient_mode=args.gradient_mode).to(device)
            initialization = initialize_proxies_from_features(proxy, reference)
            atomic_json(args.run_dir / "proxy_initialization.json", initialization)
            proxy_optimizer = geoopt.optim.RiemannianSGD(proxy.parameters(), lr=args.proxy_lr,
                                                         momentum=0.9, weight_decay=0)
            proxy_scheduler = CosineAnnealingLR(proxy_optimizer, args.epochs - args.warmup_epochs,
                                                eta_min=args.proxy_lr * 0.1)
        train_points, train_labels = points[train_ids], labels[train_ids]
        dataset = PointClouds(train_points, train_labels, 1024, "train", args.seed)
        validation_loader = DataLoader(PointClouds(points[validation_ids], labels[validation_ids], 1024, "eval", args.seed),
                                       batch_size=32, shuffle=False, num_workers=args.workers)
        best_oa = source.get("best_val_oa", -1) if source else -1
        if source and source.get("best_net") is not None:
            atomic_torch(args.run_dir / "best.pth", {"net": source["best_net"], "epoch": source["best_epoch"], "val_oa": best_oa})
        best_net = source.get("best_net") if source else None
        best_epoch = source.get("best_epoch", 0) if source else 0
        manifest.update(status="running", running_utc=utc_now(), gpu_name=torch.cuda.get_device_name(device),
                        initial_model_sha256=state_hash(model), actual_start_epoch=start)
        atomic_json(path, manifest)
        for epoch in range(start, args.epochs):
            lr = optimizer.param_groups[0]["lr"]
            row = {"epoch": epoch + 1, "arm": args.arm, "lr": lr,
                   "proxy_lr": proxy_optimizer.param_groups[0]["lr"] if proxy_optimizer else None,
                   **train_epoch(model, proxy, dataset, train_labels, optimizer, proxy_optimizer, args, epoch, device)}
            with fixed_rng(args.seed + 9020, device):
                row.update(evaluate(model, validation_loader, device))
            scheduler.step()
            if proxy_scheduler:
                proxy_scheduler.step()
            if row["val_oa"] > best_oa:
                best_oa, best_epoch = row["val_oa"], epoch + 1
                best_net = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                atomic_torch(args.run_dir / "best.pth", {"net": best_net, "epoch": best_epoch, "val_oa": best_oa,
                                                         "split_sha256": split_sha})
            atomic_json(args.run_dir / f"metrics_epoch_{epoch+1:03d}.json", row)
            state = {"net": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                     "epoch": epoch + 1, "args": json_args(args), "rng": capture_rng(), "split_sha256": split_sha,
                     "metrics": row, "best_val_oa": best_oa, "best_epoch": best_epoch, "best_net": best_net,
                     "proxy": proxy.state_dict() if proxy else None,
                     "proxy_optimizer": proxy_optimizer.state_dict() if proxy_optimizer else None,
                     "proxy_scheduler": proxy_scheduler.state_dict() if proxy_scheduler else None}
            atomic_torch(args.run_dir / "last.pth", state)
            if (epoch + 1) % args.save_every == 0:
                atomic_torch(args.run_dir / f"checkpoint_epoch_{epoch+1:03d}.pth", state)
            if args.arm == "B0" and epoch + 1 == args.warmup_epochs:
                state["model_sha256"] = state_hash(model)
                atomic_torch(args.run_dir / "shared_warmup.pth", state)
                with fixed_rng(args.seed + 9040, device):
                    features = initialization_features(model, train_points, train_labels, args, device)
                atomic_torch(args.run_dir / "initial_mu.pt", features)
                atomic_json(args.run_dir / "shared_ready.json", {
                    "epoch": epoch + 1, "checkpoint_sha256": sha256(args.run_dir / "shared_warmup.pth"),
                    "features_sha256": sha256(args.run_dir / "initial_mu.pt"),
                    "model_sha256": state["model_sha256"], "split_sha256": split_sha, "ready_utc": utc_now()})
            manifest.update(last_epoch=epoch + 1, best_val_oa=best_oa, best_epoch=best_epoch)
            atomic_json(path, manifest)
            print("epoch:", json.dumps(row, ensure_ascii=False), flush=True)
        if not args.skip_final_test:
            best = torch.load(args.run_dir / "best.pth", map_location="cpu", weights_only=False)
            model.load_state_dict(best["net"])
            test_points, test_labels, test_shards = load_shards(args.data_dir, "test")
            test_loader = DataLoader(PointClouds(test_points, test_labels, 1024, "eval", args.seed), batch_size=32,
                                     shuffle=False, num_workers=args.workers)
            report = {"selected_epoch": best["epoch"], "selection": "maximum validation OA; fixed budget completed",
                      "test_shards": test_shards, "split_sha256": split_sha, **evaluate(model, test_loader, device, "test")}
            atomic_json(args.run_dir / "final_test.json", report)
        manifest["status"] = "completed"
    except Exception as error:
        manifest.update(status="failed", error=repr(error))
        raise
    finally:
        manifest["finished_utc"] = utc_now()
        atomic_json(path, manifest)


if __name__ == "__main__":
    main()
