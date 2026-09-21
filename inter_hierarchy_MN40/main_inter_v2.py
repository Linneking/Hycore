"""Corrected teacher-guided inter-sample hierarchy experiment.

This deliberately lives beside the legacy experiment so previous results and
training behaviour stay reproducible.  A fixed pretrained HyCoRe model supplies
multi-view teacher directions.  The student is fine-tuned with the original
HyCoRe losses and, optionally, an exact-LCA within-class ranking objective.
"""

import argparse
import csv
import datetime
import json
import logging
import os
import random
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "pointnet2_ops_lib"))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "2")
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

import geoopt
import numpy as np
import sklearn.metrics as metrics
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from data import ModelNet40
from hutil import cal_loss, get_children_np, hype_triplet_losses
from models.pointmlp import Hype_pointMLP
from models.manifolds import PoincareBall
from v2.geometry import equal_radius_leaves
from v2.geometry import pairwise_ball_distance
from v2.losses import gather_teacher_similarity, lca_ranking_loss
from v2.sampler import ClassBalancedBatchSampler


def parse_args():
    p = argparse.ArgumentParser("HyCoRe inter hierarchy v2")
    p.add_argument("--pretrained", required=True)
    p.add_argument("--run_dir", required=True)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=40)
    p.add_argument("--classes_per_batch", type=int, default=5)
    p.add_argument("--samples_per_class", type=int, default=8)
    p.add_argument("--num_points", type=int, default=1024)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=22)
    p.add_argument("--learning_rate", type=float, default=0.005)
    p.add_argument("--min_lr", type=float, default=0.0001)
    p.add_argument("--weight_decay", type=float, default=2e-4)
    p.add_argument("--alpha", type=float, default=0.01)
    p.add_argument("--beta_inter", type=float, default=0.005)
    p.add_argument("--leaf_radius", type=float, default=0.90)
    p.add_argument("--rank_margin", type=float, default=0.05)
    p.add_argument("--teacher_gap", type=float, default=0.02)
    p.add_argument("--neighbor_k", type=int, default=3)
    p.add_argument("--lca_mode", choices=("exact", "gromov"), default="exact")
    p.add_argument("--teacher_views", type=int, default=3)
    p.add_argument("--teacher_mode", choices=("cosine", "hyperbolic"),
                   default="hyperbolic")
    p.add_argument("--eval_only", action="store_true")
    p.add_argument("--smoke_batches", type=int, default=0,
                   help="limit train batches per epoch; 0 means the full epoch")
    return p.parse_args()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_pretrained(net, path):
    checkpoint = torch.load(path, map_location="cpu")
    state = checkpoint.get("net", checkpoint.get("model", checkpoint))
    state = {k[7:] if k.startswith("module.") else k: v for k, v in state.items()}
    net.load_state_dict(state, strict=True)
    return checkpoint


@torch.no_grad()
def collect_teacher(net, loader, size, device, views, ball, mode):
    """Average teacher features over views in the appropriate geometry."""
    net.eval()
    feature_sum = None
    labels = torch.empty(size, dtype=torch.long)
    for _ in range(views):
        for data, label, idx in loader:
            data = data.to(device).permute(0, 2, 1)
            mu, _ = net(data)
            feat = (F.normalize(mu, dim=-1) if mode == "cosine"
                    else ball.logmap0(mu)).cpu()
            if feature_sum is None:
                feature_sum = torch.zeros(size, feat.shape[1], dtype=feat.dtype)
            feature_sum[idx.long()] += feat
            labels[idx.long()] = label.reshape(-1).long()
    mean = feature_sum / float(views)
    if mode == "cosine":
        mean = F.normalize(mean, dim=-1)
    else:
        mean = ball.expmap0(mean.to(device)).cpu()
    return mean, labels


@torch.no_grad()
def evaluate(net, loader, device):
    net.eval()
    losses, ys, preds = [], [], []
    start = time.monotonic()
    for data, label, _ in loader:
        data = data.to(device).permute(0, 2, 1)
        label = label.to(device).reshape(-1)
        _, logits = net(data)
        losses.append(float(cal_loss(logits, label).item()))
        ys.append(label.cpu().numpy())
        preds.append(logits.argmax(1).cpu().numpy())
    y, pred = np.concatenate(ys), np.concatenate(preds)
    return {
        "test_loss": float(np.mean(losses)),
        "test_oa": 100.0 * metrics.accuracy_score(y, pred),
        "test_aa": 100.0 * metrics.balanced_accuracy_score(y, pred),
        "test_seconds": time.monotonic() - start,
    }


def train_epoch(net, loader, sampler, optimizer, teacher, ball, args, epoch, device):
    net.train()
    sampler.set_epoch(epoch)
    torch.cuda.reset_peak_memory_stats()
    start = time.monotonic()
    sums = {k: 0.0 for k in ("loss", "task", "hycore", "inter", "grad_norm",
                                      "inter_grad_norm")}
    triplets = satisfied = 0
    weighted_gap = 0.0
    ys, preds = [], []
    batches = 0
    for batch_idx, (data, label, sample_id) in enumerate(loader):
        if args.smoke_batches and batch_idx >= args.smoke_batches:
            break
        data = data.to(device).permute(0, 2, 1)
        label = label.to(device).reshape(-1)
        optimizer.zero_grad(set_to_none=True)

        mar_par, _, n_points = get_children_np(data.clone(), kmin=800, kmax=1024)
        mar, pos_data, _ = get_children_np(data.clone(), starting=n_points, kmin=200, kmax=600)
        pos_mu, _ = net(pos_data, emb=True)
        parent_mu, logits = net(data)
        _, _, _, _, t_loss, h_loss = hype_triplet_losses(
            parent_mu, pos_mu, hier_margin=mar, contr_margin=4, ball_dim=256)
        task_loss = cal_loss(logits, label)

        if args.beta_inter > 0:
            if args.teacher_mode == "cosine":
                teacher_sim = gather_teacher_similarity(teacher, sample_id)
            else:
                teacher_batch = teacher[sample_id.long()].to(device)
                teacher_sim = -pairwise_ball_distance(ball, teacher_batch, teacher_batch)
            leaves = equal_radius_leaves(parent_mu, args.leaf_radius)
            inter_loss, rank_stats = lca_ranking_loss(
                leaves, label, teacher_sim, ball, neighbor_k=args.neighbor_k,
                margin=args.rank_margin, min_teacher_gap=args.teacher_gap,
                lca_mode=args.lca_mode)
        else:
            inter_loss = parent_mu.sum() * 0.0
            from v2.losses import RankingStats
            rank_stats = RankingStats(0, 0, 0.0)

        hycore_loss = t_loss + h_loss
        loss = task_loss + args.alpha * hycore_loss + args.beta_inter * inter_loss
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at epoch={epoch} batch={batch_idx}")
        if args.beta_inter > 0 and rank_stats.triplets:
            inter_grad = torch.autograd.grad(
                args.beta_inter * inter_loss, parent_mu, retain_graph=True,
                allow_unused=False)[0]
            inter_grad_norm = float(inter_grad.norm().item())
        else:
            inter_grad_norm = 0.0
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        optimizer.step()

        sums["loss"] += float(loss.item())
        sums["task"] += float(task_loss.item())
        sums["hycore"] += float(hycore_loss.item())
        sums["inter"] += float(inter_loss.item())
        sums["grad_norm"] += float(grad_norm.item())
        sums["inter_grad_norm"] += inter_grad_norm
        triplets += rank_stats.triplets
        satisfied += rank_stats.satisfied
        weighted_gap += rank_stats.mean_gap * rank_stats.triplets
        ys.append(label.detach().cpu().numpy())
        preds.append(logits.detach().argmax(1).cpu().numpy())
        batches += 1

    y, pred = np.concatenate(ys), np.concatenate(preds)
    out = {k: v / max(batches, 1) for k, v in sums.items()}
    out.update({
        "train_oa": 100.0 * metrics.accuracy_score(y, pred),
        "train_aa": 100.0 * metrics.balanced_accuracy_score(y, pred),
        "triplets": triplets,
        "rank_satisfaction": satisfied / triplets if triplets else 0.0,
        "rank_gap": weighted_gap / triplets if triplets else 0.0,
        "train_seconds": time.monotonic() - start,
        "peak_memory_mb": torch.cuda.max_memory_allocated() / (1024 ** 2),
    })
    return out


def main():
    args = parse_args()
    os.makedirs(args.run_dir, exist_ok=False)
    seed_everything(args.seed)
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    with open(os.path.join(args.run_dir, "args.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)

    log = logging.getLogger("inter_v2")
    log.setLevel(logging.INFO)
    handler = logging.FileHandler(os.path.join(args.run_dir, "run.log"))
    handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    log.addHandler(handler)
    def report(msg):
        print(msg, flush=True)
        log.info(msg)

    train_set = ModelNet40(partition="train", num_points=args.num_points)
    test_set = ModelNet40(partition="test", num_points=args.num_points)
    sampler = ClassBalancedBatchSampler(
        train_set.label, args.classes_per_batch, args.samples_per_class, args.seed)
    train_loader = DataLoader(train_set, batch_sampler=sampler, num_workers=args.workers,
                              pin_memory=True, persistent_workers=args.workers > 0)
    teacher_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.workers, pin_memory=True)

    net = Hype_pointMLP().to(device)
    checkpoint = load_pretrained(net, args.pretrained)
    report(f"loaded={args.pretrained} epoch={checkpoint.get('epoch', 'unknown')} ")
    initial = evaluate(net, test_loader, device)
    report("initial_eval=" + json.dumps(initial, sort_keys=True))
    if args.eval_only:
        with open(os.path.join(args.run_dir, "summary.json"), "w") as f:
            json.dump(initial, f, indent=2)
        return

    ball = PoincareBall(c=1.0, dim=256).to(device)
    if args.beta_inter > 0:
        report(f"collecting fixed {args.teacher_mode} teacher over "
               f"{args.teacher_views} augmented views")
        teacher, teacher_labels = collect_teacher(
            net, teacher_loader, len(train_set), device, args.teacher_views,
            ball, args.teacher_mode)
        assert torch.equal(teacher_labels, torch.as_tensor(train_set.label).reshape(-1))
        torch.save({"features": teacher, "labels": teacher_labels,
                    "views": args.teacher_views}, os.path.join(args.run_dir, "teacher.pt"))
    else:
        teacher = None
        report("beta_inter=0: skipped teacher extraction for exact zero-increment control")

    optimizer = geoopt.optim.RiemannianSGD(
        net.parameters(), lr=args.learning_rate, momentum=0.9,
        weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, args.epochs, eta_min=args.min_lr)
    fields = ["epoch", "lr", "loss", "task", "hycore", "inter", "grad_norm",
              "inter_grad_norm",
              "train_oa", "train_aa", "triplets", "rank_satisfaction", "rank_gap",
              "train_seconds", "peak_memory_mb", "test_loss", "test_oa", "test_aa",
              "test_seconds"]
    metrics_path = os.path.join(args.run_dir, "metrics.csv")
    with open(metrics_path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=fields).writeheader()

    best_oa = initial["test_oa"]
    for epoch in range(args.epochs):
        train_metrics = train_epoch(net, train_loader, sampler, optimizer, teacher,
                                    ball, args, epoch, device)
        test_metrics = evaluate(net, test_loader, device)
        row = {"epoch": epoch, "lr": optimizer.param_groups[0]["lr"],
               **train_metrics, **test_metrics}
        with open(metrics_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=fields).writerow(row)
        report("epoch=" + json.dumps(row, sort_keys=True))
        is_best = test_metrics["test_oa"] >= best_oa
        best_oa = max(best_oa, test_metrics["test_oa"])
        state = {"net": net.state_dict(), "epoch": epoch, "metrics": row,
                 "args": vars(args), "optimizer": optimizer.state_dict()}
        torch.save(state, os.path.join(args.run_dir, "last.pth"))
        if is_best:
            torch.save(state, os.path.join(args.run_dir, "best.pth"))
        scheduler.step()
    with open(os.path.join(args.run_dir, "summary.json"), "w") as f:
        json.dump({"initial": initial, "best_test_oa": best_oa,
                   "finished": datetime.datetime.now().isoformat()}, f, indent=2)


if __name__ == "__main__":
    main()
