"""Restored HyCoRe operators and a global64 CE/intra objective.

The pointMLP backbone is unchanged. Child sampling overwrites the prefix of
the whole tensor through a view, and both child and whole passes update BN
normally. Each rank encodes32 rows; gathering makes CE, intra and HIER use
the same64 rows. The contrastive negative is global child.flip(0).
"""
from __future__ import annotations

import random

import torch
from torch import nn
import torch.nn.functional as F

from .distributed import gather_with_grad, gather_without_grad


def knn_indices(points, count):
    """Exact squared-distance/topk construction from original hutil.knn."""
    inner = -2 * torch.matmul(points.transpose(2, 1), points)
    squared = torch.sum(points ** 2, dim=1, keepdim=True)
    distances = -squared - inner - squared.transpose(2, 1)
    return distances.topk(k=count, dim=-1)[1]


def get_children_alias(data, count, *, centers=None):
    """Original get_children_np with an externally shared global point count.

    Returns a view and overwrites data[:,:,:count], exactly as the released
    HyCoRe source does. The advanced-index RHS is copied before assignment.
    """
    if data.ndim != 3 or data.shape[1] < 3 or not 1 <= count <= data.shape[-1]:
        raise ValueError("expected [batch,channels,points] and valid crop count")
    neighbors = knn_indices(data[:, :3], count)
    child = data[:, :, :count]
    if centers is None:
        centers = [random.randint(0, data.shape[-1] - 1) for _ in range(len(data))]
    if len(centers) != len(data):
        raise ValueError("one crop center is required per instance")
    for row, center in enumerate(centers):
        child[row, :, :] = data[row, :, neighbors[row, int(center), :]]
    return 1000.0 / count, child, count - 1


class HyCoReTrainingForward(nn.Module):
    """Wrap both backbone calls inside a single outer DDP forward."""

    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, points, whole_count, child_count):
        if points.ndim != 3 or points.shape[-1] != 3:
            raise ValueError("input points must have shape [local_batch,N,3]")
        source = points.transpose(1, 2).contiguous()
        _, whole, _ = get_children_alias(source, int(whole_count))
        whole_before_child = whole.detach().clone()
        _, child, _ = get_children_alias(whole, int(child_count))
        # No BN override: shared backbone running statistics update twice.
        nu, _ = self.backbone(child, emb=True)
        mu, logits = self.backbone(whole)
        stats = {
            "whole_point_count": int(whole_count),
            "child_point_count": int(child_count),
            "child_aliases_whole": child.untyped_storage().data_ptr() == whole.untyped_storage().data_ptr(),
            "whole_aliases_source": whole.untyped_storage().data_ptr() == source.untyped_storage().data_ptr(),
            "whole_changed_by_child": bool(not torch.equal(whole, whole_before_child)),
            # Original pointMLP requests512 first-stage FPS centers even when
            # a child has fewer points. Keep the operator/configuration intact.
            "fps_first_stage_requested": 512,
            "fps_child_below_requested": int(child_count) < 512,
            "fps_child_minimum_repeated_indices": max(0, 512 - int(child_count)),
        }
        return {"mu": mu, "nu": nu, "logits": logits,
                "whole_count": int(whole_count), "child_count": int(child_count),
                "input_stats": stats}


def smoothed_ce(logits, labels, epsilon=0.2):
    """Original HyCoRe smoothing: eps is distributed over OTHER classes."""
    labels = labels.contiguous().view(-1)
    classes = logits.shape[1]
    one_hot = torch.zeros_like(logits).scatter(1, labels[:, None], 1)
    target = one_hot * (1 - epsilon) + (1 - one_hot) * epsilon / (classes - 1)
    return -(target * F.log_softmax(logits, dim=1)).sum(dim=1).mean()


def intra_losses(mu, nu, child_count):
    """Original hype_triplet_losses, using inherited Geoopt c1 distances."""
    from geoopt import PoincareBall

    ball = PoincareBall(c=1.0).to(device=mu.device, dtype=mu.dtype)
    negative = nu.flip(0)
    positive_distance = ball.dist(mu, nu)
    negative_distance = ball.dist(mu, negative)
    whole_depth, child_depth = ball.dist0(mu), ball.dist0(nu)
    contrastive = F.relu(positive_distance - negative_distance + 4).mean()
    radial = F.relu(-whole_depth + child_depth + 1000.0 / child_count).mean()
    return {"intra_contrastive": contrastive, "intra_radial": radial,
            "positive_distance": positive_distance.mean(),
            "negative_distance": negative_distance.mean(),
            "whole_depth": whole_depth.detach().mean(),
            "part_depth": child_depth.detach().mean()}


def global_base_losses(local_outputs, local_labels, expected_global_batch=64):
    """Compute every base term on the same gathered64 rows as inter.

    All ranks call this in the same order. No world-size correction belongs
    here: differentiable gather backward SUM and DDP averaging cancel it.
    """
    mu = gather_with_grad(local_outputs["mu"])
    nu = gather_with_grad(local_outputs["nu"])
    logits = gather_with_grad(local_outputs["logits"])
    labels = gather_without_grad(local_labels.reshape(-1).to(logits.device))
    if len(labels) != expected_global_batch:
        raise ValueError(f"expected global{expected_global_batch}, got {len(labels)}")
    if bool((labels == labels.flip(0)).any()):
        raise RuntimeError("global flip negative shares the anchor class")
    child_count = int(local_outputs["child_count"])
    # Fail loudly if externally supplied counts differ across ranks.
    whole_count = int(local_outputs.get("whole_count", 1024))
    counts = gather_without_grad(torch.tensor([[whole_count, child_count]], device=mu.device))
    if bool((counts != counts[0]).any()):
        raise RuntimeError("whole/child counts must be shared across the global batch")
    ce = smoothed_ce(logits, labels)
    values = intra_losses(mu, nu, child_count)
    base = ce + 0.01 * values["intra_contrastive"] + 0.01 * values["intra_radial"]
    return {"mu": mu, "nu": nu, "logits": logits, "gold": labels,
            "base": base, "ce": ce, **values,
            "whole_radius": mu.detach().norm(dim=-1).mean(),
            "part_radius": nu.detach().norm(dim=-1).mean(),
            "global_batch_size": len(labels),
            "whole_count": whole_count, "child_count": child_count}
