"""Online-mined HyCoRe + HIER diagnostic experiment on ModelNet40.

B1/B2 start from the verified epoch-229 HyCoRe checkpoint. Initial features
set the proxy scale once; every training triplet is mined from the current
student embedding. Per-epoch test curves are engineering diagnostics only and
must not be used for checkpoint selection or paper claims.
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
from hier_proxy_v2_online.mining import mine_batch_triplets
from hier_proxy_v2_online.proxy_hierarchy import ProxyHierarchy
from hier_proxy_v2_online.diagnostics import (
    fixed_rng, fixed_triplet_metrics, proxy_geometry,
)


ORIGINAL_FOLDER = "Hype_PointNet-Offv_pointmlp_hycore_var-4780"
ALPHA_INTRA = 0.01
MAX_BETA = 10.0
MAX_RAW_RADIUS = 1.01
FROZEN_NAMES = (
    "embedding", "local_grouper_list", "pre_blocks_list",
    "pos_blocks_list", "proj",
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("train", "calibrate"), default="calibrate")
    parser.add_argument("--arm", choices=("B1", "B2"), default="B2")
    parser.add_argument("--pretrained", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=22)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--classes-per-batch", type=int, default=5)
    parser.add_argument("--samples-per-class", type=int, default=8)
    parser.add_argument("--num-points", type=int, default=1024)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--teacher-views", type=int, default=3)
    parser.add_argument("--initialization-cache", type=Path,
                        help="original-checkpoint features used only to set initial proxy scale")
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
    parser.add_argument("--hier-scale", type=float, default=1.0,
                        help="common multiplier on enabled HIER terms")
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
        required = {"B1": ("beta_in", "beta_p"),
                    "B2": ("beta_in", "beta_out", "beta_p")}[args.arm]
        for field in required:
            if getattr(args, field) is None:
                parser.error("--" + field.replace("_", "-") + " required for " + args.arm)
        if args.initialization_cache is None:
            parser.error("training requires --initialization-cache from calibration")
    if not math.isfinite(args.hier_scale) or not 0 < args.hier_scale <= 4:
        parser.error("--hier-scale must be finite and in (0, 4]")
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


def git_branch() -> str:
    result = subprocess.run(
        ["git", "branch", "--show-current"], cwd=REPO, capture_output=True,
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
    # MobiusLayer owns its manifold as a child module. The recursive call
    # above also toggles Geoopt's isp_c curvature parameters, unintentionally
    # changing the ball radius during training unless they are re-frozen.
    model.manifold.requires_grad_(False)
    model.manifold2.requires_grad_(False)
    assert_manifold_contract(model)
    return frozen


def assert_manifold_contract(model: torch.nn.Module,
                             optimizer: torch.optim.Optimizer | None = None) -> None:
    curvature_ids = set()
    for name in ("manifold", "manifold2"):
        manifold = getattr(model, name)
        c = float(manifold.c.detach())
        if not math.isfinite(c) or abs(c - 1.0) > 1e-5:
            raise RuntimeError(f"{name} curvature changed from c=1: {c:.8f}")
        for parameter in manifold.parameters():
            curvature_ids.add(id(parameter))
            if parameter.requires_grad:
                raise RuntimeError(f"{name} curvature is unexpectedly trainable")
    if optimizer is not None:
        optimizer_ids = {id(parameter) for group in optimizer.param_groups
                         for parameter in group["params"]}
        if curvature_ids & optimizer_ids:
            raise RuntimeError("curvature parameter entered optimizer")


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
def initialization_features(model: torch.nn.Module, points: np.ndarray,
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


def load_initialization_cache(path: Path, source_sha256: str, shards: list[str],
                              labels: np.ndarray, num_points: int) -> torch.Tensor:
    cached = torch.load(path.resolve(strict=True), map_location="cpu", weights_only=False)
    if not isinstance(cached, dict):
        raise ValueError("initialization cache must be a dictionary")
    if cached.get("source_sha256") != source_sha256:
        raise ValueError("initialization cache source checkpoint hash mismatch")
    if cached.get("train_shards_sorted") != shards or cached.get("num_points") != num_points:
        raise ValueError("initialization cache dataset identity mismatch")
    if not torch.equal(torch.as_tensor(cached.get("labels")), torch.as_tensor(labels)):
        raise ValueError("initialization cache label ordering mismatch")
    features = cached.get("mu")
    if not isinstance(features, torch.Tensor) or features.shape != (len(labels), 256):
        raise ValueError("initialization cache has wrong feature shape")
    if not bool(torch.isfinite(features).all()):
        raise ValueError("initialization cache has non-finite features")
    return features


def triplet_columns(rows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return rows[:, 0], rows[:, 1], rows[:, 2]


def forward_losses(model: torch.nn.Module, proxy: ProxyHierarchy,
                   point_batch: torch.Tensor, labels: torch.Tensor,
                   ids: torch.Tensor,
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
    # Full-batch training encountered raw embeddings outside the strict
    # interior check; the cause may include model/Geoopt numerical behavior.
    # The explicit HIER distance formula requires an interior point.
    # Project only the HIER branch; leave classification and intra unchanged.
    raw_radius = mu.detach().norm(dim=-1)
    if not bool(torch.isfinite(raw_radius).all()) or float(raw_radius.max()) > MAX_RAW_RADIUS:
        raise FloatingPointError(
            f"invalid HyCoRe embedding radius; max={float(raw_radius.max()):.8f}, "
            f"dtype={mu.dtype}, manifold_c={float(model.manifold.c):.8f}, "
            f"limit={MAX_RAW_RADIUS:.3f}"
        )
    hier_mu = proxy.ball.projx(mu.float())
    projected_radius = hier_mu.detach().norm(dim=-1)
    if not bool(torch.isfinite(projected_radius).all()) or float(projected_radius.max()) >= 1:
        raise FloatingPointError("HIER projection did not produce interior points")
    projected_samples = int(
        ((hier_mu.detach() - mu.detach()).norm(dim=-1) > 1e-6).sum()
    )
    # The selector is the *current* student representation. Detaching makes
    # the discrete top-k decision non-differentiable; the loss below still
    # receives the live mu and can update both student and proxies.
    mined = mine_batch_triplets(
        hier_mu.detach(), labels, k_in=args.k_in,
        seed=args.seed + epoch * 1000003 + batch_index, hard_ratio=0.5,
    )
    mined["stats"].update(raw_radius_max=float(raw_radius.max()),
                          projected_radius_max=float(projected_radius.max()),
                          raw_outside_samples=int((raw_radius >= 1).sum()),
                          projected_samples=projected_samples)
    in_loss, in_stats = proxy.sample_loss(
        hier_mu, triplet_columns(mined["in"]), tau=args.tau, margin=args.margin,
    )
    out_loss, out_stats = proxy.sample_loss(
        hier_mu, triplet_columns(mined["out"]), tau=args.tau, margin=args.margin,
    )
    proxy_loss, proxy_stats = proxy.proxy_loss(
        k=args.proxy_k, t_per_anchor=args.proxy_triples_per_anchor,
        tau=args.tau, margin=args.margin,
    )
    losses.update({"in": in_loss, "out": out_loss, "proxy": proxy_loss,
                   "hier_mu": hier_mu,
                   "mining_stats": mined["stats"], "in_stats": in_stats,
                   "out_stats": out_stats, "proxy_stats": proxy_stats})
    return losses


def configured_betas(args: argparse.Namespace) -> tuple[float, float, float]:
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
    # Calibrate the shared proxy coefficient for B1. B2 uses the same value,
    # so its sole additional objective is beta_out * L_out.
    sample_proxy_gradient = (0 if beta_in is None else beta_in * g_in_p)
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
        "rule": "in: 20% base mu gradient; out: 10% base mu gradient; shared proxy coefficient: 25% weighted B1 in-proxy gradient",
        "mining": losses["mining_stats"], "in": losses["in_stats"],
        "out": losses["out_stats"], "proxy": losses["proxy_stats"],
    }


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader,
             device: torch.device, prefix: str,
             predictions_path: Path | None = None) -> dict[str, float]:
    model.eval()
    loss_sum, total = 0.0, 0
    confusion = torch.zeros(40, 40, dtype=torch.long)
    sample_ids, sample_labels, sample_logits = [], [], []
    for points, labels, ids in loader:
        labels = labels.reshape(-1).long()
        _, logits = model(points.to(device).transpose(1, 2))
        loss_sum += float(cal_loss(logits, labels.to(device))) * len(labels)
        predictions = logits.argmax(-1).cpu()
        if predictions_path is not None:
            sample_ids.append(ids.reshape(-1).cpu().numpy())
            sample_labels.append(labels.cpu().numpy())
            sample_logits.append(logits.detach().cpu().numpy())
        total += len(labels)
        for gold, pred in zip(labels.tolist(), predictions.tolist()):
            confusion[gold, pred] += 1
    count_by_class = confusion.sum(-1)
    valid = count_by_class > 0
    aa = (confusion.diagonal()[valid].float() / count_by_class[valid]).mean()
    if predictions_path is not None:
        np.savez_compressed(
            predictions_path,
            sample_id=np.concatenate(sample_ids),
            label=np.concatenate(sample_labels),
            logits=np.concatenate(sample_logits),
        )
    return {prefix + "_loss": loss_sum / total,
            prefix + "_oa": 100 * float(confusion.diagonal().sum()) / total,
            prefix + "_aa": 100 * float(aa)}


def probe_neighbour_edges(mu: torch.Tensor, labels: torch.Tensor,
                          proxy: ProxyHierarchy, k: int) -> set[tuple[int, int]]:
    """Directed same-class top-k edges for a fixed class-balanced probe."""
    with torch.no_grad():
        distance = proxy.ball.dist(mu[:, None, :], mu[None, :, :]).cpu()
        classes = labels.reshape(-1).cpu().tolist()
        edges: set[tuple[int, int]] = set()
        for i, label in enumerate(classes):
            peers = [j for j, other in enumerate(classes) if other == label and j != i]
            peers.sort(key=lambda j: (float(distance[i, j]), j))
            edges.update((i, j) for j in peers[:k])
        return edges


def edge_jaccard(left: set[tuple[int, int]],
                 right: set[tuple[int, int]]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


@torch.no_grad()
def probe_distance_stats(mu: torch.Tensor, labels: torch.Tensor,
                         proxy: ProxyHierarchy) -> dict[str, float]:
    """Track class-in spread and separation on identical probe instances."""
    distances = proxy.ball.dist(mu[:, None, :], mu[None, :, :])
    labels = labels.reshape(-1).to(mu.device)
    same = labels[:, None] == labels[None, :]
    same.fill_diagonal_(False)
    different = ~same
    different.fill_diagonal_(False)
    within = distances[same]
    between = distances[different]
    return {
        "probe_within_distance_mean": float(within.mean()),
        "probe_within_distance_std": float(within.std(unbiased=False)),
        "probe_between_distance_mean": float(between.mean()),
        "probe_between_to_within_ratio": float(between.mean() / within.mean().clamp_min(1e-12)),
    }


def fixed_probe_data(model: torch.nn.Module, proxy: ProxyHierarchy,
                     points: np.ndarray, labels: np.ndarray,
                     args: argparse.Namespace, device: torch.device
                     ) -> tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                                tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                                torch.Tensor, dict, set[tuple[int, int]]]:
    """Select fixed IDs/views/triplets once; never use these for optimization."""
    sampler = ClassBalancedBatchSampler(labels, args.classes_per_batch,
                                        args.samples_per_class, args.seed)
    sampler.set_epoch(0)
    indices = next(iter(sampler))
    ids = torch.as_tensor(indices, dtype=torch.long)
    gold = torch.as_tensor(labels[indices], dtype=torch.long)
    views = []
    for view in (0, 1):
        dataset = PointClouds(points, labels, args.num_points,
                              "reference", args.seed, view)
        views.append(torch.stack([dataset[int(i)][0] for i in indices]))
    model.eval()
    proxy.eval()
    with fixed_rng(args.seed + 9001, device), torch.no_grad():
        initial_mu, _ = model(views[0].to(device).transpose(1, 2))
        initial_mu = proxy.ball.projx(initial_mu)
    mined = mine_batch_triplets(initial_mu.detach(), gold.to(device),
                                k_in=args.k_in, seed=args.seed + 701)
    if mined["in"].shape[0] == 0:
        raise RuntimeError("fixed probe has no class-in triplets")
    initial_edges = probe_neighbour_edges(initial_mu, gold, proxy, args.k_in)
    first = (views[0], gold, ids)
    second = (views[1], gold, ids)
    return first, second, initial_mu.detach().cpu(), mined, initial_edges


def probe_epoch(model: torch.nn.Module, proxy: ProxyHierarchy,
                first: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                second: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                initial_mu: torch.Tensor, fixed_triplets: dict,
                initial_edges: set[tuple[int, int]],
                previous_edges: set[tuple[int, int]],
                initial_tangent: torch.Tensor,
                occupancy_reference: torch.Tensor,
                args: argparse.Namespace, device: torch.device
                ) -> tuple[dict[str, float], set[tuple[int, int]], dict]:
    """Repeat the same diagnostic after each epoch without changing RNG state."""
    model.eval()
    proxy.eval()
    with fixed_rng(args.seed + 9001, device), torch.no_grad():
        current_mu, _ = model(first[0].to(device).transpose(1, 2))
        current_mu = proxy.ball.projx(current_mu)
    with fixed_rng(args.seed + 9001, device), torch.no_grad():
        second_mu, _ = model(second[0].to(device).transpose(1, 2))
        second_mu = proxy.ball.projx(second_mu)
    edges = probe_neighbour_edges(current_mu, first[1], proxy, args.k_in)
    other_edges = probe_neighbour_edges(second_mu, second[1], proxy, args.k_in)
    result: dict[str, float] = {
        "probe_knn_jaccard_initial": edge_jaccard(initial_edges, edges),
        "probe_knn_jaccard_previous": edge_jaccard(previous_edges, edges),
        "probe_knn_jaccard_views": edge_jaccard(edges, other_edges),
    }
    result.update(probe_distance_stats(current_mu, first[1], proxy))
    with fixed_rng(args.seed + 9002, device):
        values = forward_losses(model, proxy, *first, args, 0, 0, device)
        in_loss, in_stats = proxy.sample_loss(
            values["hier_mu"], triplet_columns(fixed_triplets["in"]),
            tau=args.tau, margin=args.margin,
        )
        out_loss, out_stats = proxy.sample_loss(
            values["hier_mu"], triplet_columns(fixed_triplets["out"]),
            tau=args.tau, margin=args.margin,
        )
        base = values["ce"] + ALPHA_INTRA * values["intra"]
        b_in, b_out, b_p = configured_betas(args)
        parameter = proxy.tangent_proxies
        gradients = {
            "base_mu": norm_of_gradient(base, values["mu"]),
            "in_mu": norm_of_gradient(in_loss, values["mu"]),
            "out_mu": norm_of_gradient(out_loss, values["mu"]),
            "in_proxy": norm_of_gradient(in_loss, parameter),
            "out_proxy": norm_of_gradient(out_loss, parameter),
            "proxy_only_proxy": norm_of_gradient(values["proxy"], parameter),
        }
        result.update({
            "probe_in_loss": float(in_loss.detach()),
            "probe_out_loss": float(out_loss.detach()),
            "probe_proxy_loss": float(values["proxy"].detach()),
            "probe_base_mu_grad": float(gradients["base_mu"].norm()),
            "probe_weighted_in_mu_grad": float((args.hier_scale * b_in * gradients["in_mu"]).norm()),
            "probe_weighted_out_mu_grad": float((args.hier_scale * b_out * gradients["out_mu"]).norm()),
            "probe_weighted_in_proxy_grad": float((args.hier_scale * b_in * gradients["in_proxy"]).norm()),
            "probe_weighted_out_proxy_grad": float((args.hier_scale * b_out * gradients["out_proxy"]).norm()),
            "probe_weighted_proxy_only_grad": float((args.hier_scale * b_p * gradients["proxy_only_proxy"]).norm()),
            "probe_weighted_total_proxy_grad": float((args.hier_scale * (
                b_in * gradients["in_proxy"] + b_out * gradients["out_proxy"] +
                b_p * gradients["proxy_only_proxy"])).norm()),
        })
        for branch, stats in (("in", in_stats), ("out", out_stats)):
            valid = stats["valid_triplets"]
            result["probe_" + branch + "_valid"] = float(valid)
            result["probe_" + branch + "_margin_satisfied_rate"] = (
                1.0 - stats["active_triplets"] / valid if valid else 0.0
            )
    result.update(fixed_triplet_metrics(
        proxy, initial_tangent, initial_mu, current_mu.detach(),
        triplet_columns(fixed_triplets["in"]),
        seed=args.seed + 9003, tau=args.tau, margin=args.margin,
    ))
    geometry, snapshot = proxy_geometry(proxy, initial_tangent, occupancy_reference)
    result.update({key: float(value) for key, value in geometry.items()})
    return result, edges, snapshot


def train_epoch(model: torch.nn.Module, proxy: ProxyHierarchy | None,
                 frozen: list[torch.nn.Module], loader: DataLoader,
                 sampler: ClassBalancedBatchSampler, optimizer: torch.optim.Optimizer,
                 args: argparse.Namespace,
                epoch: int, device: torch.device) -> dict[str, float]:
    training_mode(model, frozen)
    if proxy is not None:
        proxy.train()
    sampler.set_epoch(epoch)
    start = time.monotonic()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    sums = {name: 0.0 for name in ("loss", "ce", "intra", "in", "out", "proxy",
                                     "base", "weighted_in", "weighted_out", "weighted_proxy",
                                     "in_triplets", "out_triplets", "proxy_triplets",
                                     "in_active", "out_active", "proxy_active",
                                     "in_valid", "out_valid", "proxy_valid",
                                     "in_collisions", "out_collisions", "proxy_collisions",
                                     "eligible_in", "eligible_out", "proxy_eligible",
                                     "projected_samples", "raw_outside_samples",
                                     "grad_norm")}
    correct, total, batches = 0, 0, 0
    max_raw_radius = 0.0
    max_projected_radius = 0.0
    b_in, b_out, b_p = configured_betas(args)
    for batch_index, (points, labels, ids) in enumerate(loader):
        if args.max_batches and batch_index >= args.max_batches:
            break
        optimizer.zero_grad(set_to_none=True)
        try:
            values = forward_losses(model, proxy, points, labels, ids,
                                    args, epoch, batch_index, device)
        except Exception as error:
            raise RuntimeError(
                f"epoch={epoch + 1}, batch_index={batch_index}: {error}"
            ) from error
        base = values["ce"] + ALPHA_INTRA * values["intra"]
        weighted_in = args.hier_scale * b_in * values["in"]
        weighted_out = args.hier_scale * b_out * values["out"]
        weighted_proxy = args.hier_scale * b_p * values["proxy"]
        loss = base + weighted_in + weighted_out + weighted_proxy
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
        for name, value in (("base", base), ("weighted_in", weighted_in),
                            ("weighted_out", weighted_out),
                            ("weighted_proxy", weighted_proxy)):
            sums[name] += float(value.detach())
        sums["loss"] += float(loss.detach())
        sums["grad_norm"] += float(grad_norm)
        mine = values["mining_stats"]
        if batch_index % 50 == 0:
            print("batch_radius:", json.dumps({
                "epoch": epoch + 1, "batch_index": batch_index,
                "raw_radius_max": mine["raw_radius_max"],
                "projected_radius_max": mine["projected_radius_max"],
                "raw_outside_samples": mine["raw_outside_samples"],
                "projected_samples": mine["projected_samples"],
            }, sort_keys=True), flush=True)
        sums["eligible_in"] += mine.get("eligible_in_anchors", 0)
        sums["eligible_out"] += mine.get("eligible_out_anchors", 0)
        sums["projected_samples"] += mine.get("projected_samples", 0)
        sums["raw_outside_samples"] += mine.get("raw_outside_samples", 0)
        max_raw_radius = max(max_raw_radius, mine.get("raw_radius_max", 0.0))
        max_projected_radius = max(max_projected_radius,
                                   mine.get("projected_radius_max", 0.0))
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
    result["train_max_raw_radius"] = max_raw_radius
    result["train_max_projected_radius"] = max_projected_radius
    for branch in ("in", "out", "proxy"):
        result["train_" + branch + "_active_rate"] = (
            sums[branch + "_active"] / sums[branch + "_triplets"]
            if sums[branch + "_triplets"] else 0.0
        )
    result["train_oa"] = 100 * correct / total
    assert_manifold_contract(model, optimizer)
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
                "git_commit": git_commit(), "git_branch": git_branch(),
                "command": sys.argv, "protocol": "online-student-mining-diagnostic",
                "source_checkpoint": str(args.pretrained.resolve()),
                "source_checkpoint_sha256": sha256(args.pretrained),
                "source_epoch": source["epoch"], "source_acc": source["acc"],
                "train_shards_sorted": train_shards, "train_count": len(train_labels),
                "train_labels_sha256": hashlib.sha256(train_labels.tobytes()).hexdigest(),
                "seed": args.seed,
                "device": str(device),
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                "model_parameter_dtype": str(next(model.parameters()).dtype),
                "torch": torch.__version__, "torch_cuda": torch.version.cuda,
                "geoopt": getattr(geoopt, "__version__", "unknown"),
                "model_manifold_c": float(model.manifold.c),
                "model_manifold2_c": float(model.manifold2.c),
                "hier_proxy_c": 1.0,
                "max_raw_radius_guard": MAX_RAW_RADIUS,
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
        if args.initialization_cache is None:
            initial_features = initialization_features(
                model, train_points, train_labels, args, device,
            )
            cache_path = run_dir / "initial_mu.pt"
            torch.save({"mu": initial_features, "labels": torch.as_tensor(train_labels),
                        "source_sha256": manifest["source_checkpoint_sha256"],
                        "train_shards_sorted": train_shards,
                        "num_points": args.num_points,
                        "views": args.teacher_views}, cache_path)
        else:
            cache_path = args.initialization_cache.resolve(strict=True)
            initial_features = load_initialization_cache(
                cache_path, manifest["source_checkpoint_sha256"],
                train_shards, train_labels, args.num_points,
            )
        manifest["initialization_cache"] = str(cache_path)
        manifest["initialization_cache_sha256"] = sha256(cache_path)
        manifest["initialization_cache_use"] = "proxy-scale-only; never mined for training triplets"
        proxy = ProxyHierarchy(
            num_proxies=args.proxy_count, dim=256, c=1, seed=args.seed,
        ).to(device)
        init_stats = proxy.initialize(initial_features.to(device))
        write_json(run_dir / "proxy_initialization.json", init_stats)
        initial_tangent = proxy.tangent_proxies.detach().clone()
        initial_proxy_points = proxy.proxies().detach().clone()
        torch.save({"tangent": initial_tangent.cpu(),
                    "ball": initial_proxy_points.cpu()}, run_dir / "proxy_epoch_000.pt")
        print("proxy initialization:", init_stats, flush=True)

        if args.mode == "calibrate":
            seed_all(args.seed)
            training_mode(model, frozen)
            per_batch = []
            for batch_index, (points, labels, ids) in enumerate(train_loader):
                if batch_index >= args.calibration_batches:
                    break
                values = forward_losses(model, proxy, points, labels, ids,
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

        parameter_groups = [{"params": [p for p in model.parameters() if p.requires_grad]},
                            {"params": list(proxy.parameters()), "lr": args.proxy_lr}]
        optimizer = geoopt.optim.RiemannianSGD(
            parameter_groups, lr=args.lr, momentum=0.9,
            weight_decay=args.weight_decay,
        )
        assert_manifold_contract(model, optimizer)
        scheduler = CosineAnnealingLR(optimizer, args.epochs, eta_min=args.min_lr)
        seed_all(args.seed)
        first, second, initial_mu, fixed_triplets, initial_edges = fixed_probe_data(
            model, proxy, train_points, train_labels, args, device,
        )
        torch.save({"first": first, "second": second,
                    "triplets": {key: value.cpu() if isinstance(value, torch.Tensor) else value
                                 for key, value in fixed_triplets.items()},
                    "initial_mu": initial_mu}, run_dir / "fixed_probe.pt")
        occupancy_ids = np.concatenate([
            np.flatnonzero(train_labels == label)[:10]
            for label in np.unique(train_labels)
        ])
        occupancy_reference = initial_features[occupancy_ids]
        manifest["probe_ids"] = first[2].tolist()
        manifest["occupancy_reference_ids"] = occupancy_ids.tolist()
        test_loader = None
        test_shards = None
        if not args.skip_final_test:
            test_points, test_labels, test_shards = load_shards(args.data_dir, "test")
            test_dataset = PointClouds(test_points, test_labels, args.num_points,
                                       "eval", args.seed)
            test_loader = DataLoader(test_dataset, batch_size=batch_size,
                                     shuffle=False, num_workers=args.workers,
                                     pin_memory=True)
        baseline_probe, previous_edges, baseline_snapshot = probe_epoch(
            model, proxy, first, second, initial_mu, fixed_triplets,
            initial_edges, initial_edges, initial_tangent,
            occupancy_reference, args, device,
        )
        baseline_snapshot["tangent"] = proxy.tangent_proxies.detach().cpu()
        torch.save(baseline_snapshot, run_dir / "proxy_diagnostic_epoch_000.pt")
        baseline_test = {"test_loss": None, "test_oa": None, "test_aa": None}
        if test_loader is not None:
            with fixed_rng(args.seed + 9020, device):
                baseline_test = evaluate(
                    model, test_loader, device, "test",
                    run_dir / "predictions_epoch_000.npz",
                )
        write_json(run_dir / "baseline.json", {"epoch": 0, **baseline_test,
                                                 **baseline_probe})
        metrics_path = run_dir / "metrics.csv"
        fields = None
        for epoch in range(args.epochs):
            epoch_start = time.monotonic()
            learning_rate = optimizer.param_groups[0]["lr"]
            proxy_learning_rate = optimizer.param_groups[1]["lr"]
            train_metrics = train_epoch(model, proxy, frozen, train_loader, sampler,
                                        optimizer, args, epoch, device)
            train_finished = time.monotonic()
            epoch_number = epoch + 1
            probe_metrics, previous_edges, snapshot = probe_epoch(
                model, proxy, first, second, initial_mu, fixed_triplets,
                initial_edges, previous_edges, initial_tangent,
                occupancy_reference, args, device,
            )
            snapshot["tangent"] = proxy.tangent_proxies.detach().cpu()
            torch.save(snapshot, run_dir / f"proxy_diagnostic_epoch_{epoch_number:03d}.pt")
            test_metrics = {"test_loss": None, "test_oa": None, "test_aa": None}
            if test_loader is not None:
                with fixed_rng(args.seed + 9020, device):
                    test_metrics = evaluate(
                        model, test_loader, device, "test",
                        run_dir / f"predictions_epoch_{epoch_number:03d}.npz",
                    )
            row = {"epoch": epoch_number, "lr": learning_rate,
                   "proxy_lr": proxy_learning_rate,
                   "hier_scale": args.hier_scale,
                   "beta_in": args.beta_in,
                   "beta_out_effective": configured_betas(args)[1],
                   "beta_p": args.beta_p,
                   **train_metrics, **probe_metrics, **test_metrics,
                   "diagnostic_seconds": time.monotonic() - train_finished,
                   "epoch_wall_seconds": time.monotonic() - epoch_start}
            if fields is None:
                fields = list(row)
                with metrics_path.open("w", newline="", encoding="utf-8") as stream:
                    csv.DictWriter(stream, fieldnames=fields).writeheader()
            with metrics_path.open("a", newline="", encoding="utf-8") as stream:
                csv.DictWriter(stream, fieldnames=fields).writerow(row)
            print("epoch:", json.dumps(row, sort_keys=True), flush=True)
            save_checkpoint(run_dir / "last.pth", model, proxy, optimizer, epoch, row, args)
            torch.save({"emb": model.emb.state_dict(),
                        "classifier": model.classifier.state_dict(),
                        "proxy": proxy.state_dict(), "epoch": epoch_number},
                       run_dir / f"head_epoch_{epoch_number:03d}.pt")
            if (epoch + 1) % args.save_every == 0 and epoch + 1 < args.epochs:
                save_checkpoint(run_dir / f"epoch_{epoch + 1:03d}.pth",
                                model, proxy, optimizer, epoch, row, args)
            scheduler.step()

        summary = {"final_epoch": args.epochs, "test": test_metrics,
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
