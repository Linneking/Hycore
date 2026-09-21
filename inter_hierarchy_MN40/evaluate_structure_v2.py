"""Deterministic held-out structural evaluation for v2 checkpoints."""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "pointnet2_ops_lib"))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "2")
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

import numpy as np
import torch
from scipy.stats import spearmanr
from torch.utils.data import DataLoader

from data import ModelNet40
from models.manifolds import PoincareBall
from models.pointmlp import Hype_pointMLP
from v2.geometry import equal_radius_leaves, exact_lca_depth, pairwise_ball_distance
from v2.losses import lca_ranking_loss
from v2.sampler import ClassBalancedBatchSampler


def args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--teacher", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--batches", type=int, default=50)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--leaf_radius", type=float, default=.9)
    p.add_argument("--neighbor_k", type=int, default=3)
    p.add_argument("--rank_margin", type=float, default=.05)
    p.add_argument("--teacher_gap", type=float, default=.02)
    return p.parse_args()


def load(net, path):
    # These are checkpoints produced locally by this experiment and include
    # optimizer/NumPy scalar state, so PyTorch 2.6+ needs explicit full loading.
    obj = torch.load(path, map_location="cpu", weights_only=False)
    state = obj.get("net", obj.get("model", obj))
    state = {k[7:] if k.startswith("module.") else k: v for k, v in state.items()}
    net.load_state_dict(state, strict=True)


@torch.no_grad()
def main():
    a = args()
    device = torch.device("cuda:0")
    ball = PoincareBall(c=1., dim=256).to(device)
    net = Hype_pointMLP().to(device)
    load(net, a.checkpoint)
    net.eval()
    teacher = torch.load(a.teacher, map_location="cpu")["features"]

    dataset = ModelNet40(partition="train", num_points=1024)
    dataset.partition = "heldout"  # disable random train-time translation/shuffle
    sampler = ClassBalancedBatchSampler(
        dataset.label, 5, 8, seed=20260922, batches_per_epoch=a.batches)
    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=a.workers,
                        pin_memory=True)

    total_triplets = total_satisfied = 0
    weighted_gap = weighted_loss = 0.0
    teacher_pairs, student_pairs = [], []
    for data, label, sample_id in loader:
        data = data.to(device).permute(0, 2, 1)
        label = label.to(device).reshape(-1)
        mu, _ = net(data)
        leaves = equal_radius_leaves(mu, a.leaf_radius)
        teacher_batch = teacher[sample_id.long()].to(device)
        teacher_sim = -pairwise_ball_distance(ball, teacher_batch, teacher_batch)
        loss, stats = lca_ranking_loss(
            leaves, label, teacher_sim, ball, neighbor_k=a.neighbor_k,
            margin=a.rank_margin, min_teacher_gap=a.teacher_gap, lca_mode="exact")
        total_triplets += stats.triplets
        total_satisfied += stats.satisfied
        weighted_gap += stats.mean_gap * stats.triplets
        weighted_loss += float(loss.item()) * stats.triplets

        depth = exact_lca_depth(ball, leaves, leaves)
        same = label[:, None].eq(label[None, :])
        upper = torch.triu(torch.ones_like(same), diagonal=1).bool()
        mask = same & upper
        teacher_pairs.append(teacher_sim[mask].cpu().numpy())
        student_pairs.append(depth[mask].cpu().numpy())

    teacher_pairs = np.concatenate(teacher_pairs)
    student_pairs = np.concatenate(student_pairs)
    rho = float(spearmanr(teacher_pairs, student_pairs).statistic)
    result = {
        "checkpoint": a.checkpoint,
        "batches": a.batches,
        "pairs": int(teacher_pairs.size),
        "triplets": total_triplets,
        "rank_satisfaction": total_satisfied / total_triplets if total_triplets else 0.,
        "rank_gap": weighted_gap / total_triplets if total_triplets else 0.,
        "ranking_loss": weighted_loss / total_triplets if total_triplets else 0.,
        "spearman": rho,
    }
    with open(a.output, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
