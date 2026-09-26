"""Frozen-backbone HyCoRe + HIER proxy experiment on ModelNet40.

Run B0/B1/B2 from the verified original HyCoRe checkpoint. Every arm uses the
full ModelNet40 train split and a fixed 25-epoch schedule. The official test
split is evaluated once on the final model. ``calibrate`` builds the shared
reference cache and reports gradient scales without updating any weights.
"""

from __future__ import annotations

import argparse
import csv
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

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "pointnet2_ops_lib"))
sys.path.insert(0, str(HERE.parent))

import geoopt
import h5py
import numpy as np
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset

from hutil import cal_loss, get_children_np, hype_triplet_losses
from models.pointmlp import Hype_pointMLP
from v2.sampler import ClassBalancedBatchSampler
from hier_proxy_v1.mining import mine_batch_triplets
from hier_proxy_v1.proxy_hierarchy import ProxyHierarchy


ORIGINAL_FOLDER = "Hype_PointNet-Offv_pointmlp_hycore_var-4780"
ALPHA_INTRA = 0.01
MAX_BETA = 10.0
FROZEN_NAMES = (
    "embedding", "local_grouper_list", "pre_blocks_list",
    "pos_blocks_list", "proj",
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("train", "calibrate"), default="calibrate")
    parser.add_argument("--arm", choices=("B0", "B1", "B2"), default="B2")
    parser.add_argument("--pretrained", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=22)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--classes-per-batch", type=int, default=5)
    parser.add_argument("--samples-per-class", type=int, default=8)
    parser.add_argument("--num-points", type=int, default=1024)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--teacher-views", type=int, default=3)
    parser.add_argument("--reference-cache", type=Path,
                        help="fixed cache created by a calibration run; required for B1/B2 training")
    parser.add_argument("--calibration-batches", type=int, default=8)
    parser.add_argument("--calibration-json", type=Path,
                        help="read suggested beta coefficients from calibration.json")
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--max-batches", type=int, default=0,
                        help="smoke run: limit train batches per epoch; 0 means full epoch")
    parser.add_argument("--skip-final-test", action="store_true",
                        help="for smoke runs only; official test remains untouched")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--proxy-lr", type=float, default=5e-4)
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=2e-4)
    parser.add_argument("--beta-in", type=float)
    parser.add_argument("--beta-out", type=float)
    parser.add_argument("--beta-p", type=float)
    parser.add_argument("--k-in", type=int, default=3)
    parser.add_argument("--proxy-count", type=int, default=128)
    parser.add_argument("--proxy-k", type=int, default=8)
    parser.add_argument("--proxy-triples-per-anchor", type=int, default=2)
    parser.add_argument("--tau", type=float, default=0.1)
    parser.add_argument("--margin", type=float, default=0.1)
    args = parser.parse_args()
    if args.epochs < 1 or args.num_points != 1024 or args.teacher_views < 1:
        parser.error("epochs and teacher views must be positive; HyCoRe uses 1024 points")
    if args.classes_per_batch < 2 or args.samples_per_class < args.k_in + 2:
        parser.error("need >=2 classes and >=k_in+2 distinct instances per class")
    if args.calibration_batches < 1 or args.save_every < 1 or args.max_batches < 0:
        parser.error("calibration batches and save interval must be positive")
    if args.proxy_k >= args.proxy_count or args.proxy_k < 2:
        parser.error("proxy K must be in [2, proxy count)")
    if args.lr <= 0 or args.proxy_lr <= 0 or args.min_lr < 0:
        parser.error("learning rates must be positive and min LR nonnegative")
    if args.mode == "train":
        if args.calibration_json is not None:
            with args.calibration_json.open("r", encoding="utf-8") as stream:
                saved = json.load(stream)["suggested_betas"]
            for field, key in (("beta_in", "in"), ("beta_out", "out"), ("beta_p", "p")):
                if getattr(args, field) is None:
                    setattr(args, field, saved.get(key))
        required = {"B0": (), "B1": ("beta_in", "beta_p"),
                    "B2": ("beta_in", "beta_out", "beta_p")}[args.arm]
        for field in required:
            if getattr(args, field) is None:
                parser.error("--" + field.replace("_", "-") + " required for " + args.arm)
        if args.arm != "B0" and args.reference_cache is None:
            parser.error("B1/B2 training requires --reference-cache from calibration")
    for field in ("beta_in", "beta_out", "beta_p"):
        value = getattr(args, field)
        if value is not None and (not math.isfinite(value) or value < 0 or value > MAX_BETA):
            parser.error(field + f" must be finite and between 0 and {MAX_BETA}")
    return args


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True,
        text=True, check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def write_json(path: Path, value: object) -> None:
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)


def load_model(path: Path, device: torch.device) -> tuple[torch.nn.Module, dict]:
    path = path.resolve(strict=True)
    if path.name != "best_checkpoint.pth" or path.parent.name != ORIGINAL_FOLDER:
        raise ValueError("--pretrained must be the original HyCoRe best checkpoint")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or checkpoint.get("epoch") != 229:
        raise ValueError("original HyCoRe checkpoint must report epoch 229")
    if abs(float(checkpoint.get("acc", float("nan"))) - 94.044) > 0.1:
        raise ValueError("checkpoint OA does not match original HyCoRe inventory")
    state = checkpoint.get("net")
    if not isinstance(state, dict):
        raise ValueError("checkpoint has no model state")
    state = {key.removeprefix("module."): value for key, value in state.items()}
    model = Hype_pointMLP().to(device)
    model.load_state_dict(state, strict=True)
    return model, checkpoint


def freeze_euclidean_backbone(model: torch.nn.Module) -> list[torch.nn.Module]:
    frozen = [getattr(model, name) for name in FROZEN_NAMES]
    for module in frozen:
        module.requires_grad_(False)
        module.eval()
    for module in (model.emb, model.classifier):
        module.requires_grad_(True)
    return frozen


def training_mode(model: torch.nn.Module, frozen: list[torch.nn.Module]) -> None:
    model.train()
    for module in frozen:
        module.eval()  # Also freeze BatchNorm running statistics.


def load_shards(data_dir: Path, partition: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    paths = sorted(data_dir.resolve(strict=True).glob("ply_data_" + partition + "*.h5"))
    if not paths:
        raise FileNotFoundError("no ModelNet40 " + partition + " HDF5 shards in " + str(data_dir))
    point_chunks, label_chunks = [], []
    for path in paths:
        with h5py.File(path, "r") as stream:
            point_chunks.append(stream["data"][:].astype(np.float32))
            label_chunks.append(stream["label"][:].astype(np.int64).reshape(-1))
    return np.concatenate(point_chunks), np.concatenate(label_chunks), [p.name for p in paths]


class PointClouds(Dataset):
    def __init__(self, points: np.ndarray, labels: np.ndarray, num_points: int,
                 mode: str, seed: int, view: int = 0):
        self.points = points
        self.labels = labels
        self.num_points = num_points
        self.mode = mode
        self.seed = seed
        self.view = view

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        points = self.points[index, :self.num_points].copy()
        if self.mode == "train":
            scale = np.random.uniform(2 / 3, 3 / 2, size=3)
            shift = np.random.uniform(-0.2, 0.2, size=3)
            points = (points * scale + shift).astype(np.float32)
            np.random.shuffle(points)
        elif self.mode == "reference":
            rng = np.random.default_rng(self.seed + self.view * len(self.labels) + index)
            scale = rng.uniform(2 / 3, 3 / 2, size=3)
            shift = rng.uniform(-0.2, 0.2, size=3)
            points = (points * scale + shift).astype(np.float32)
            rng.shuffle(points)
        return torch.from_numpy(points), int(self.labels[index]), int(index)


@torch.no_grad()
def reference_features(model: torch.nn.Module, points: np.ndarray,
                       labels: np.ndarray, args: argparse.Namespace,
                       device: torch.device) -> torch.Tensor:
    model.eval()
    tangent_sum = torch.zeros(len(labels), 256, dtype=torch.float32)
    for view in range(args.teacher_views):
        dataset = PointClouds(points, labels, args.num_points, "reference", args.seed, view)
        loader = DataLoader(dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=0, pin_memory=True)
        for point_batch, _, sample_id in loader:
            mu, _ = model(point_batch.to(device).transpose(1, 2))
            tangent_sum[sample_id] += model.manifold2.logmap0(mu).cpu()
    average = tangent_sum / args.teacher_views
    return model.manifold2.expmap0(average.to(device)).cpu()


def load_reference_cache(path: Path, source_sha256: str, shards: list[str],
                         labels: np.ndarray, num_points: int) -> torch.Tensor:
    cached = torch.load(path.resolve(strict=True), map_location="cpu", weights_only=False)
    if not isinstance(cached, dict):
        raise ValueError("reference cache must be a dictionary")
    if cached.get("source_sha256") != source_sha256:
        raise ValueError("reference cache source checkpoint hash mismatch")
    if cached.get("train_shards_sorted") != shards or cached.get("num_points") != num_points:
        raise ValueError("reference cache dataset identity mismatch")
    if not torch.equal(torch.as_tensor(cached.get("labels")), torch.as_tensor(labels)):
        raise ValueError("reference cache label ordering mismatch")
    reference = cached.get("mu")
    if not isinstance(reference, torch.Tensor) or reference.shape != (len(labels), 256):
        raise ValueError("reference cache has wrong feature shape")
    if not bool(torch.isfinite(reference).all()):
        raise ValueError("reference cache has non-finite features")
    return reference


def triplet_columns(rows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return rows[:, 0], rows[:, 1], rows[:, 2]


def forward_losses(model: torch.nn.Module, proxy: ProxyHierarchy | None,
                   point_batch: torch.Tensor, labels: torch.Tensor,
                   ids: torch.Tensor, reference: torch.Tensor | None,
                   args: argparse.Namespace, epoch: int, batch_index: int,
                   device: torch.device) -> dict:
    data = point_batch.to(device, non_blocking=True).transpose(1, 2)
    labels = labels.to(device, non_blocking=True).reshape(-1)
    _, _, n_points = get_children_np(data.clone(), kmin=800, kmax=1024)
    margin, pos_data, _ = get_children_np(
        data.clone(), starting=n_points, kmin=200, kmax=600,
    )
    pos_mu, _ = model(pos_data, emb=True)
    mu, logits = model(data)
    _, _, _, _, triplet, hierarchy = hype_triplet_losses(
        mu, pos_mu, hier_margin=margin, contr_margin=4, ball_dim=256,
    )
    ce = cal_loss(logits, labels)
    intra = triplet + hierarchy
    zero = mu.sum() * 0.0
    losses = {"ce": ce, "intra": intra, "mu": mu, "logits": logits,
              "in": zero, "out": zero, "proxy": zero,
              "mining_stats": {}, "in_stats": {}, "out_stats": {}, "proxy_stats": {}}
    if proxy is None:
        return losses
    if reference is None:
        raise ValueError("proxy experiment requires fixed original-checkpoint cache")
    ref_batch = reference[ids.long()].to(device, non_blocking=True)
    mined = mine_batch_triplets(
        ref_batch, labels, k_in=args.k_in,
        seed=args.seed + epoch * 1000003 + batch_index, hard_ratio=0.5,
    )
    in_loss, in_stats = proxy.sample_loss(
        mu, triplet_columns(mined["in"]), tau=args.tau, margin=args.margin,
    )
    out_loss, out_stats = proxy.sample_loss(
        mu, triplet_columns(mined["out"]), tau=args.tau, margin=args.margin,
    )
    proxy_loss, proxy_stats = proxy.proxy_loss(
        k=args.proxy_k, t_per_anchor=args.proxy_triples_per_anchor,
        tau=args.tau, margin=args.margin,
    )
    losses.update({"in": in_loss, "out": out_loss, "proxy": proxy_loss,
                   "mining_stats": mined["stats"], "in_stats": in_stats,
                   "out_stats": out_stats, "proxy_stats": proxy_stats})
    return losses


def configured_betas(args: argparse.Namespace) -> tuple[float, float, float]:
    if args.arm == "B0":
        return 0.0, 0.0, 0.0
    if args.arm == "B1":
        return float(args.beta_in), 0.0, float(args.beta_p)
    return float(args.beta_in), float(args.beta_out), float(args.beta_p)


def norm_of_gradient(loss: torch.Tensor, tensor: torch.Tensor) -> torch.Tensor:
    gradient = torch.autograd.grad(loss, tensor, retain_graph=True, allow_unused=True)[0]
    return torch.zeros_like(tensor) if gradient is None else gradient


def calibrate(losses: dict, proxy: ProxyHierarchy) -> dict:
    base = losses["ce"] + ALPHA_INTRA * losses["intra"]
    mu, parameter = losses["mu"], proxy.tangent_proxies
    g_base_mu = norm_of_gradient(base, mu)
    g_in_mu = norm_of_gradient(losses["in"], mu)
    g_out_mu = norm_of_gradient(losses["out"], mu)
    g_in_p = norm_of_gradient(losses["in"], parameter)
    g_out_p = norm_of_gradient(losses["out"], parameter)
    g_p_p = norm_of_gradient(losses["proxy"], parameter)
    n_base = float(g_base_mu.norm())
    n_in = float(g_in_mu.norm())
    n_out = float(g_out_mu.norm())
    beta_in = 0.2 * n_base / n_in if n_in > 1e-12 else None
    beta_out = 0.1 * n_base / n_out if n_out > 1e-12 else None
    sample_proxy_gradient = (0 if beta_in is None else beta_in * g_in_p)
    sample_proxy_gradient = sample_proxy_gradient + (0 if beta_out is None else beta_out * g_out_p)
    n_sample_proxy = float(sample_proxy_gradient.norm()) if isinstance(sample_proxy_gradient, torch.Tensor) else 0.0
    n_proxy_only = float(g_p_p.norm())
    beta_p = 0.25 * n_sample_proxy / n_proxy_only if n_proxy_only > 1e-12 else None
    return {
        "raw_loss": {name: float(losses[name].detach()) for name in ("ce", "intra", "in", "out", "proxy")},
        "gradient_norm": {"base_mu": n_base, "in_mu": n_in, "out_mu": n_out,
                          "in_proxy": float(g_in_p.norm()), "out_proxy": float(g_out_p.norm()),
                          "proxy_only_proxy": n_proxy_only},
        "suggested_betas": {"in": beta_in, "out": beta_out, "p": beta_p},
        "effective_gradient_ratio": {
            "in_to_base_mu": beta_in * n_in / n_base if beta_in is not None and n_base > 1e-12 else None,
            "out_to_base_mu": beta_out * n_out / n_base if beta_out is not None and n_base > 1e-12 else None,
            "proxy_only_to_sample_proxy": beta_p * n_proxy_only / n_sample_proxy
            if beta_p is not None and n_sample_proxy > 1e-12 else None,
        },
        "rule": "in: 20% base mu gradient; out: 10% base mu gradient; proxy-only: 25% weighted sample proxy gradient",
        "mining": losses["mining_stats"], "in": losses["in_stats"],
        "out": losses["out_stats"], "proxy": losses["proxy_stats"],
    }


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader,
             device: torch.device, prefix: str) -> dict[str, float]:
    model.eval()
    loss_sum, total = 0.0, 0
    confusion = torch.zeros(40, 40, dtype=torch.long)
    for points, labels, _ in loader:
        labels = labels.reshape(-1).long()
        _, logits = model(points.to(device).transpose(1, 2))
        loss_sum += float(cal_loss(logits, labels.to(device))) * len(labels)
        predictions = logits.argmax(-1).cpu()
        total += len(labels)
        for gold, pred in zip(labels.tolist(), predictions.tolist()):
            confusion[gold, pred] += 1
    count_by_class = confusion.sum(-1)
    valid = count_by_class > 0
    aa = (confusion.diagonal()[valid].float() / count_by_class[valid]).mean()
    return {prefix + "_loss": loss_sum / total,
            prefix + "_oa": 100 * float(confusion.diagonal().sum()) / total,
            prefix + "_aa": 100 * float(aa)}


def train_epoch(model: torch.nn.Module, proxy: ProxyHierarchy | None,
                frozen: list[torch.nn.Module], loader: DataLoader,
                sampler: ClassBalancedBatchSampler, optimizer: torch.optim.Optimizer,
                reference: torch.Tensor | None, args: argparse.Namespace,
                epoch: int, device: torch.device) -> dict[str, float]:
    training_mode(model, frozen)
    if proxy is not None:
        proxy.train()
    sampler.set_epoch(epoch)
    start = time.monotonic()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    sums = {name: 0.0 for name in ("loss", "ce", "intra", "in", "out", "proxy",
                                     "in_triplets", "out_triplets", "proxy_triplets",
                                     "in_active", "out_active", "proxy_active",
                                     "in_valid", "out_valid", "proxy_valid",
                                     "in_collisions", "out_collisions", "proxy_collisions",
                                     "eligible_in", "eligible_out", "proxy_eligible",
                                     "grad_norm")}
    correct, total, batches = 0, 0, 0
    b_in, b_out, b_p = configured_betas(args)
    for batch_index, (points, labels, ids) in enumerate(loader):
        if args.max_batches and batch_index >= args.max_batches:
            break
        optimizer.zero_grad(set_to_none=True)
        values = forward_losses(model, proxy, points, labels, ids, reference,
                                args, epoch, batch_index, device)
        loss = (values["ce"] + ALPHA_INTRA * values["intra"] +
                b_in * values["in"] + b_out * values["out"] + b_p * values["proxy"])
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"non-finite loss at epoch {epoch}, batch {batch_index}")
        loss.backward()
        params = [p for group in optimizer.param_groups for p in group["params"]]
        grad_norm = torch.nn.utils.clip_grad_norm_(params, 1.0)
        if not bool(torch.isfinite(grad_norm)):
            raise FloatingPointError(f"non-finite gradient at epoch {epoch}, batch {batch_index}")
        optimizer.step()
        for name in ("ce", "intra", "in", "out", "proxy"):
            sums[name] += float(values[name].detach())
        sums["loss"] += float(loss.detach())
        sums["grad_norm"] += float(grad_norm)
        mine = values["mining_stats"]
        sums["eligible_in"] += mine.get("eligible_in_anchors", 0)
        sums["eligible_out"] += mine.get("eligible_out_anchors", 0)
        for branch in ("in", "out", "proxy"):
            stats = values[branch + "_stats"]
            sums[branch + "_triplets"] += stats.get("triplets", 0)
            sums[branch + "_active"] += stats.get("active_triplets", 0)
            sums[branch + "_valid"] += stats.get("valid_triplets", 0)
            sums[branch + "_collisions"] += stats.get("collisions", 0)
        sums["proxy_eligible"] += values["proxy_stats"].get("eligible_anchors", 0)
        labels = labels.reshape(-1)
        correct += int((values["logits"].detach().argmax(-1).cpu() == labels).sum())
        total += len(labels)
        batches += 1
    result = {"train_" + key: value / batches for key, value in sums.items()}
    for branch in ("in", "out", "proxy"):
        result["train_" + branch + "_active_rate"] = (
            sums[branch + "_active"] / sums[branch + "_triplets"]
            if sums[branch + "_triplets"] else 0.0
        )
    result["train_oa"] = 100 * correct / total
    result["train_seconds"] = time.monotonic() - start
    result["peak_memory_mb"] = (torch.cuda.max_memory_allocated(device) / (1024 ** 2)
                                 if device.type == "cuda" else 0.0)
    return result


def save_checkpoint(path: Path, model: torch.nn.Module, proxy: ProxyHierarchy | None,
                    optimizer: torch.optim.Optimizer, epoch: int,
                    row: dict, args: argparse.Namespace) -> None:
    state = {"net": model.state_dict(), "proxy": proxy.state_dict() if proxy else None,
             "optimizer": optimizer.state_dict(), "epoch": epoch, "metrics": row,
             "args": {key: str(value) if isinstance(value, Path) else value
                      for key, value in vars(args).items()}}
    torch.save(state, path)


def main() -> None:
    args = arguments()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)
    seed_all(args.seed)
    model, source = load_model(args.pretrained, device)
    frozen = freeze_euclidean_backbone(model)
    train_points, train_labels, train_shards = load_shards(args.data_dir, "train")
    if min(np.bincount(train_labels)) < args.samples_per_class:
        raise ValueError("class-balanced batch needs more instances per training class")
    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    manifest = {"status": "running", "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                "git_commit": git_commit(), "source_checkpoint": str(args.pretrained.resolve()),
                "source_checkpoint_sha256": sha256(args.pretrained),
                "source_epoch": source["epoch"], "source_acc": source["acc"],
                "train_shards_sorted": train_shards, "train_count": len(train_labels),
                "train_labels_sha256": hashlib.sha256(train_labels.tobytes()).hexdigest(),
                "seed": args.seed,
                "device": str(device),
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                "torch": torch.__version__, "torch_cuda": torch.version.cuda,
                "geoopt": getattr(geoopt, "__version__", "unknown"),
                "numpy": np.__version__,
                "args": {key: str(value) if isinstance(value, Path) else value
                         for key, value in vars(args).items()}}
    write_json(run_dir / "manifest.json", manifest)
    try:
        train_dataset = PointClouds(train_points, train_labels, args.num_points, "train", args.seed)
        batch_size = args.classes_per_batch * args.samples_per_class
        args.batch_size = batch_size
        sampler = ClassBalancedBatchSampler(
            train_labels, args.classes_per_batch,
            args.samples_per_class, args.seed,
        )
        train_loader = DataLoader(
            train_dataset, batch_sampler=sampler,
            num_workers=args.workers, pin_memory=True,
            persistent_workers=args.workers > 0,
            generator=torch.Generator().manual_seed(args.seed + 101),
        )
        use_proxy = args.arm != "B0" or args.mode == "calibrate"
        proxy = None
        reference = None
        if use_proxy:
            if args.mode == "calibrate" and args.reference_cache is None:
                reference = reference_features(
                    model, train_points, train_labels, args, device,
                )
                torch.save({"mu": reference, "labels": torch.as_tensor(train_labels),
                            "source_sha256": manifest["source_checkpoint_sha256"],
                            "train_shards_sorted": train_shards,
                            "num_points": args.num_points,
                            "views": args.teacher_views}, run_dir / "reference_mu.pt")
                manifest["reference_cache"] = str(run_dir / "reference_mu.pt")
                manifest["reference_cache_sha256"] = sha256(run_dir / "reference_mu.pt")
            else:
                reference = load_reference_cache(
                    args.reference_cache, manifest["source_checkpoint_sha256"],
                    train_shards, train_labels, args.num_points,
                )
                manifest["reference_cache"] = str(args.reference_cache.resolve())
                manifest["reference_cache_sha256"] = sha256(args.reference_cache)
            proxy = ProxyHierarchy(
                num_proxies=args.proxy_count, dim=256, c=1, seed=args.seed,
            ).to(device)
            init_stats = proxy.initialize(reference.to(device))
            write_json(run_dir / "proxy_initialization.json", init_stats)
            print("proxy initialization:", init_stats, flush=True)

        if args.mode == "calibrate":
            seed_all(args.seed)
            training_mode(model, frozen)
            per_batch = []
            for batch_index, (points, labels, ids) in enumerate(train_loader):
                if batch_index >= args.calibration_batches:
                    break
                values = forward_losses(model, proxy, points, labels, ids, reference,
                                        args, 0, batch_index, device)
                per_batch.append(calibrate(values, proxy))
                del values
            keys = ("in", "out", "p")
            suggested = {}
            unsafe = {}
            for key in keys:
                values = [r["suggested_betas"][key] for r in per_batch
                          if r["suggested_betas"][key] is not None]
                median = float(np.median(values)) if values else None
                unsafe[key] = median is None or median > MAX_BETA or not math.isfinite(median)
                suggested[key] = None if unsafe[key] else median
            gradient_keys = per_batch[0]["gradient_norm"]
            ratio_keys = per_batch[0]["effective_gradient_ratio"]
            report = {"batches": len(per_batch), "suggested_betas": suggested,
                      "unsafe_coefficients": unsafe,
                      "median_gradient_norm": {
                          key: float(np.median([r["gradient_norm"][key] for r in per_batch]))
                          for key in gradient_keys},
                      "median_effective_gradient_ratio": {
                          key: float(np.median([r["effective_gradient_ratio"][key]
                                                for r in per_batch
                                                if r["effective_gradient_ratio"][key] is not None]))
                          if any(r["effective_gradient_ratio"][key] is not None for r in per_batch)
                          else None for key in ratio_keys},
                      "rule": per_batch[0]["rule"], "per_batch": per_batch}
            write_json(run_dir / "calibration.json", report)
            print("calibration:", json.dumps(report, sort_keys=True), flush=True)
            manifest["status"] = "completed_calibration"
            return

        parameter_groups = [{"params": [p for p in model.parameters() if p.requires_grad]}]
        if proxy is not None:
            parameter_groups.append({"params": list(proxy.parameters()), "lr": args.proxy_lr})
        optimizer = geoopt.optim.RiemannianSGD(
            parameter_groups, lr=args.lr, momentum=0.9,
            weight_decay=args.weight_decay,
        )
        scheduler = CosineAnnealingLR(optimizer, args.epochs, eta_min=args.min_lr)
        seed_all(args.seed)
        fields = ["epoch", "lr", "proxy_lr", "train_loss", "train_ce", "train_intra", "train_in",
                  "train_out", "train_proxy", "train_oa", "train_seconds",
                  "train_in_triplets", "train_out_triplets", "train_proxy_triplets",
                  "train_in_active", "train_out_active", "train_proxy_active",
                  "train_in_active_rate", "train_out_active_rate", "train_proxy_active_rate",
                  "train_in_valid", "train_out_valid", "train_proxy_valid",
                  "train_in_collisions", "train_out_collisions", "train_proxy_collisions",
                  "train_eligible_in", "train_eligible_out", "train_proxy_eligible",
                  "train_grad_norm", "peak_memory_mb"]
        metrics_path = run_dir / "metrics.csv"
        with metrics_path.open("w", newline="", encoding="utf-8") as stream:
            csv.DictWriter(stream, fieldnames=fields).writeheader()
        for epoch in range(args.epochs):
            learning_rate = optimizer.param_groups[0]["lr"]
            proxy_learning_rate = optimizer.param_groups[1]["lr"] if proxy else 0.0
            train_metrics = train_epoch(model, proxy, frozen, train_loader, sampler,
                                        optimizer, reference, args, epoch, device)
            row = {"epoch": epoch, "lr": learning_rate,
                   "proxy_lr": proxy_learning_rate, **train_metrics}
            with metrics_path.open("a", newline="", encoding="utf-8") as stream:
                csv.DictWriter(stream, fieldnames=fields).writerow(row)
            print("epoch:", json.dumps(row, sort_keys=True), flush=True)
            save_checkpoint(run_dir / "last.pth", model, proxy, optimizer, epoch, row, args)
            if (epoch + 1) % args.save_every == 0 and epoch + 1 < args.epochs:
                save_checkpoint(run_dir / f"epoch_{epoch + 1:03d}.pth",
                                model, proxy, optimizer, epoch, row, args)
            scheduler.step()

        final_test = None
        test_shards = None
        if not args.skip_final_test:
            test_points, test_labels, test_shards = load_shards(args.data_dir, "test")
            test_dataset = PointClouds(test_points, test_labels, args.num_points, "eval", args.seed)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                     num_workers=args.workers, pin_memory=True)
            final_test = evaluate(model, test_loader, device, "test")
        summary = {"final_epoch": args.epochs - 1, "test": final_test,
                   "test_shards_sorted": test_shards, "smoke_batches": args.max_batches,
                   "finished_utc": dt.datetime.now(dt.timezone.utc).isoformat()}
        write_json(run_dir / "summary.json", summary)
        print("final:", json.dumps(summary, sort_keys=True), flush=True)
        manifest["status"] = "completed"
    except Exception as error:
        manifest["status"] = "failed"
        manifest["error"] = repr(error)
        raise
    finally:
        manifest["finished_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
        write_json(run_dir / "manifest.json", manifest)


if __name__ == "__main__":
    main()
