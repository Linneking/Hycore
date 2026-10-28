"""Segmentation feature pooling around the unchanged HyCoRe triplet-loss kernel.

The kernel in ``native`` retains the official batch reversal and ``torch.max``
hinges. Only the task-specific construction of whole/part inputs is new here.
This module returns the two regularizers; cross entropy belongs to the trainer.
"""

import torch
from torch import nn

from .native import PoincareBall, hype_triplet_losses


def choose_parts(features, generator, n_min=200, n_max=600):
    """Select a feature-neighborhood independently within every point cloud.

    Use an independent CPU generator, one center per cloud, and one shared
    uniformly sampled size per batch. Sorting is discrete; gathered values keep
    their gradients. Stable ties prefer the lower point index, after explicitly
    including the sampled center. No input features are modified.
    """
    if features.ndim != 3:
        raise ValueError("Expected features [B,N,F]")
    batch, points, channels = features.shape
    if not batch or not channels or not 1 <= n_min <= n_max <= points:
        raise ValueError((batch, points, channels, n_min, n_max))
    size = int(torch.randint(n_min, n_max + 1, (), generator=generator))
    centers = torch.randint(points, (batch,), generator=generator).to(features.device)
    rows = torch.arange(batch, device=features.device)
    with torch.no_grad():
        detached = features.detach()
        center_values = detached[rows, centers]
        distances = (detached - center_values[:, None]).square().sum(-1)
        distances[rows, centers] = -1.0
        indices = distances.argsort(dim=-1, stable=True)[:, :size]
    selected = features.gather(1, indices[:, :, None].expand(-1, -1, channels))
    return selected.mean(dim=1), indices, centers, size


def flip_negative_indices(categories):
    """Official reversal, intentionally without class filtering or resampling."""
    labels = categories.reshape(-1)
    rows = torch.arange(len(labels), device=labels.device)
    return rows.flip(0)


def relation_losses(ball, whole, part, gamma, delta):
    """Call the original loss kernel with the segmentation radial direction."""
    if whole.ndim != 2 or whole.shape != part.shape:
        raise ValueError("Whole and part must be equal-sized [B,F] tensors")
    native_values = hype_triplet_losses(
        whole, part, hier_margin=gamma, contr_margin=delta,
        ball_dim=whole.shape[-1], one_child=False, opposite_hier=True)
    contrast, radial = native_values[4], native_values[5]
    with torch.no_grad():
        whole_depth, part_depth = ball.dist0(whole), ball.dist0(part)
        positive = ball.dist(whole, part)
        negative = ball.dist(whole, part.flip(0))
        radial_values = whole_depth - part_depth + gamma
        contrast_values = positive - negative + delta
        metric_tensors = {
            "hier": radial.detach(), "contr": contrast.detach(),
            "whole_depth": whole_depth.mean(), "part_depth": part_depth.mean(),
            "depth_gap": (part_depth - whole_depth).mean(),
            "hier_satisfied": (radial_values <= 1e-6).float().mean(),
            "contr_satisfied": (contrast_values <= 1e-6).float().mean(),
            "positive_distance": positive.mean(), "negative_distance": negative.mean(),
            "whole_projected": (whole.norm(dim=-1) >= .996 - 1e-6).float().mean(),
            "part_projected": (part.norm(dim=-1) >= .996 - 1e-6).float().mean(),
        }
        # Transfer the small diagnostics vector once instead of synchronizing
        # CUDA separately for every logged scalar.
        values = torch.stack(tuple(metric_tensors.values())).cpu().tolist()
        info = dict(zip(metric_tensors, values))
        info.update({
            # Compatibility field: all reversed pairs are used by the source.
            # This does not assert that their semantic classes differ.
            "valid_negative": len(whole), "objects": len(whole),
            "negative_policy": "official_batch_flip_without_class_filter",
        })
    return radial, contrast, info


class HyCoReSegLoss(nn.Module):
    def __init__(self, alpha=.01, beta=.01, gamma_base=1000., delta=4.,
                 part_min=200, part_max=600, seed=4780, feature_dim=None):
        super().__init__()
        if alpha < 0 or beta < 0 or gamma_base < 0 or delta < 0:
            raise ValueError("Loss coefficients and margins must be nonnegative")
        if not 1 <= part_min <= part_max:
            raise ValueError("Invalid part size interval")
        self.feature_dim = feature_dim
        self.ball = PoincareBall(c=1.0, dim=feature_dim or 512)
        self.alpha, self.beta = alpha, beta
        self.gamma_base, self.delta = gamma_base, delta
        self.part_min, self.part_max = part_min, part_max
        self.generator = torch.Generator(device="cpu").manual_seed(seed + 100003)

    def forward(self, features, categories):
        if features.ndim != 3 or categories.numel() != features.shape[0]:
            raise ValueError("Expected H[B,N,F] and one category per cloud")
        if self.feature_dim is not None and features.shape[-1] != self.feature_dim:
            raise ValueError("Feature width differs from configured regularizer")
        if self.alpha == 0 and self.beta == 0:
            return features.sum() * 0.0, {"disabled": True}
        width = features.shape[-1]
        if self.ball.coord_dim != width:
            self.ball = PoincareBall(c=1.0, dim=width)
        self.ball = self.ball.to(device=features.device, dtype=features.dtype)
        pooled, _, _, size = choose_parts(
            features, self.generator, self.part_min, self.part_max)
        whole_tangent = features.mean(dim=1)
        whole = self.ball.expmap0(whole_tangent)
        part = self.ball.expmap0(pooled)
        gamma = self.gamma_base / size
        radial, contrast, info = relation_losses(self.ball, whole, part, gamma, self.delta)
        with torch.no_grad():
            labels = categories.reshape(-1)
            negative = flip_negative_indices(labels)
            rows = torch.arange(len(labels), device=labels.device)
            point_norms = features.detach().norm(dim=-1)
            metric_tensors = {
                "same_class_pair_fraction": (labels == labels[negative]).float().mean(),
                "self_pair_fraction": (rows == negative).float().mean(),
                "feature_norm_mean": point_norms.mean(), "feature_norm_max": point_norms.max(),
                "whole_tangent_norm_mean": whole_tangent.norm(dim=-1).mean(),
                "part_tangent_norm_mean": pooled.norm(dim=-1).mean(),
            }
            values = torch.stack(tuple(metric_tensors.values())).cpu().tolist()
            info.update(dict(zip(metric_tensors, values)))
            info.update(part_points=size, gamma=gamma, delta=self.delta, feature_dim=width)
        return self.alpha * contrast + self.beta * radial, info
