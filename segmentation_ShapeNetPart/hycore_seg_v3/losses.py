"""Mean-then-EXP regularization with explicit in-batch, different-class negatives.

The original v2/native flip kernel remains unchanged.  This version explicitly
gathers the sampled negative part in the real contrastive distance expression.
Index selection is discrete; positive and negative embedding paths keep their
ordinary autograd graph.  No additional point-cloud encoding is performed.
"""

import hashlib

import torch
from torch import nn

from segmentation_ShapeNetPart.hycore_seg_v2.losses import choose_parts
from segmentation_ShapeNetPart.hycore_seg_v2.native import PoincareBall


def sample_negative_indices(categories, generator):
    """Uniformly choose one different-class *sample* per anchor, with reuse.

    Returns (indices, candidate_counts, valid_mask) on categories.device.
    Invalid anchors have index -1.  The independent generator must be a CPU
    generator; no global Torch, data, part-selection or dropout RNG is consumed.
    """
    labels = categories.detach().reshape(-1).to(device="cpu", dtype=torch.long)
    if not len(labels):
        raise ValueError("Cannot sample negatives from an empty batch")
    if generator.device.type != "cpu":
        raise ValueError("Negative sampling requires an independent CPU generator")
    indices = torch.full((len(labels),), -1, dtype=torch.long)
    counts = torch.zeros(len(labels), dtype=torch.long)
    for anchor in range(len(labels)):
        candidates = torch.nonzero(labels != labels[anchor], as_tuple=False).flatten()
        counts[anchor] = len(candidates)
        if len(candidates):
            draw = int(torch.randint(len(candidates), (), generator=generator))
            indices[anchor] = candidates[draw]
    device = categories.device
    return indices.to(device), counts.to(device), (counts > 0).to(device)


def _scalar_metrics(metrics):
    """Transfer scalar diagnostics together; values in the result are JSON-safe."""
    present = {key: value for key, value in metrics.items() if value is not None}
    values = torch.stack(tuple(present.values())).detach().cpu().tolist() if present else []
    result = dict(zip(present, values))
    result.update({key: None for key, value in metrics.items() if value is None})
    return result


def relation_losses(ball, whole, part, gamma, delta, negative_indices, valid_mask=None):
    """Return (unweighted radial, unweighted contrast, JSON diagnostics).

    Rcontr is averaged only over valid anchors.  A batch without candidates has
    a differentiable zero contrast term; every anchor still contributes to
    Rhier.  ``torch.max`` preserves the original kernel's hinge-at-zero rule.
    """
    if whole.ndim != 2 or whole.shape != part.shape or not len(whole):
        raise ValueError("Whole and part must be nonempty equal-sized [B,F] tensors")
    indices = negative_indices.to(device=part.device, dtype=torch.long).reshape(-1)
    if indices.numel() != len(whole):
        raise ValueError("One negative index is required per anchor")
    valid = indices >= 0 if valid_mask is None else valid_mask.to(device=part.device, dtype=torch.bool).reshape(-1)
    if valid.numel() != len(whole):
        raise ValueError("One validity flag is required per anchor")
    if ((indices[valid] < 0) | (indices[valid] >= len(part))).any():
        raise ValueError("A valid negative index is out of range")

    whole_depth, part_depth = ball.dist0(whole), ball.dist0(part)
    positive = ball.dist(whole, part)
    radial_values = whole_depth - part_depth + gamma
    radial = torch.max(radial_values, torch.zeros_like(radial_values)).mean()
    valid_count = int(valid.sum().item())
    if valid_count:
        # Neither the gathered part nor the anchor is detached.
        negative = ball.dist(whole[valid], part[indices[valid]])
        contrast_values = positive[valid] - negative + delta
        contrast = torch.max(contrast_values, torch.zeros_like(contrast_values)).mean()
    else:
        negative = contrast_values = None
        contrast = (whole.sum() + part.sum()) * 0.0

    with torch.no_grad():
        # Geoopt v0.5.1's standard numerical ball projection, not an extra cap.
        eps = 4e-3 if whole.dtype == torch.float32 else 1e-5
        radius = (1 - eps) / ball.c.sqrt()
        metrics = {
            "hier": radial.detach(), "contr": contrast.detach(),
            "whole_depth": whole_depth.mean(), "part_depth": part_depth.mean(),
            "depth_gap": (part_depth - whole_depth).mean(),
            "hier_satisfied": (radial_values <= 1e-6).float().mean(),
            "contr_satisfied": ((contrast_values <= 1e-6).float().mean()
                                if valid_count else None),
            "positive_distance": positive.mean(),
            "negative_distance": negative.mean() if valid_count else None,
            "whole_radius_mean": whole.norm(dim=-1).mean(),
            "part_radius_mean": part.norm(dim=-1).mean(),
            "whole_boundary_fraction": (whole.norm(dim=-1) >= radius - 1e-6).float().mean(),
            "part_boundary_fraction": (part.norm(dim=-1) >= radius - 1e-6).float().mean(),
        }
        info = _scalar_metrics(metrics)
        info.update(valid_negative=valid_count, objects=len(whole),
                    negative_valid_rate=valid_count / len(whole),
                    negative_policy="uniform_different_class_sample_with_reuse",
                    no_negative_candidates=valid_count == 0)
    return radial, contrast, info


class HyCoReSegLoss(nn.Module):
    def __init__(self, alpha=.01, beta=.01, gamma_base=1000., delta=4.,
                 part_min=200, part_max=600, seed=4780, feature_dim=256):
        super().__init__()
        if alpha < 0 or beta < 0 or gamma_base < 0 or delta < 0:
            raise ValueError("Loss coefficients and margins must be nonnegative")
        if not 1 <= part_min <= part_max:
            raise ValueError("Invalid part size interval")
        if feature_dim is not None and feature_dim < 1:
            raise ValueError("Feature width must be positive")
        self.feature_dim = feature_dim
        self.ball = PoincareBall(c=1.0, dim=feature_dim or 256)
        self.alpha, self.beta = alpha, beta
        self.gamma_base, self.delta = gamma_base, delta
        self.part_min, self.part_max = part_min, part_max
        self.generator = torch.Generator(device="cpu").manual_seed(seed + 100003)
        self.negative_generator = torch.Generator(device="cpu").manual_seed(seed + 200003)
        self.monitor = None
        self.last_components = {}

    def _capture(self, name, tensor, tangent_input=None):
        if self.monitor is not None:
            self.monitor.capture(name, tensor, tangent_input=tangent_input)

    def forward(self, features, categories):
        self.last_components = {}
        if features.ndim != 3 or categories.numel() != features.shape[0]:
            raise ValueError("Expected H[B,N,F] and one category per cloud")
        if self.feature_dim is not None and features.shape[-1] != self.feature_dim:
            raise ValueError("Feature width differs from configured regularizer")
        if self.alpha == 0 and self.beta == 0:
            zero = features.sum() * 0.0
            self.last_components = {"contr": zero, "hier": zero}
            return zero, {"disabled": True}
        width = features.shape[-1]
        if self.ball.coord_dim != width:
            self.ball = PoincareBall(c=1.0, dim=width)
        self.ball = self.ball.to(device=features.device, dtype=features.dtype)

        pooled, part_indices, centers, size = choose_parts(
            features, self.generator, self.part_min, self.part_max)
        whole_tangent = features.mean(dim=1)
        self._capture("whole_tangent", whole_tangent)
        self._capture("part_tangent", pooled)
        whole = self.ball.expmap0(whole_tangent)
        part = self.ball.expmap0(pooled)
        self._capture("whole_exp", whole, tangent_input=whole_tangent)
        self._capture("part_exp", part, tangent_input=pooled)

        indices, counts, valid = sample_negative_indices(categories, self.negative_generator)
        gamma = self.gamma_base / size
        radial, contrast, info = relation_losses(
            self.ball, whole, part, gamma, self.delta, indices, valid)
        weighted_contrast, weighted_radial = self.alpha * contrast, self.beta * radial
        self.last_components = {"contr": weighted_contrast, "hier": weighted_radial}

        with torch.no_grad():
            labels_cpu = categories.detach().reshape(-1).to(device="cpu", dtype=torch.long)
            indices_cpu, counts_cpu, valid_cpu = indices.cpu(), counts.cpu(), valid.cpu()
            selected = indices_cpu[valid_cpu]
            negative_labels = torch.full_like(indices_cpu, -1)
            negative_labels[valid_cpu] = labels_cpu[selected]
            if len(selected):
                if not bool((negative_labels[valid_cpu] != labels_cpu[valid_cpu]).all()):
                    raise AssertionError("A sampled negative has the anchor's class")
                rows = torch.arange(len(labels_cpu))
                same_class = float((negative_labels[valid_cpu] == labels_cpu[valid_cpu]).float().mean())
                self_pair = float((selected == rows[valid_cpu]).float().mean())
                hits = torch.bincount(selected, minlength=len(labels_cpu))
                unique_negatives = int((hits > 0).sum())
                max_reuse = int(hits.max())
            else:
                same_class = self_pair = None
                unique_negatives = max_reuse = 0
                hits = torch.zeros(len(labels_cpu), dtype=torch.long)
            norm_metrics = {
                "feature_norm_mean": features.detach().norm(dim=-1).mean(),
                "feature_norm_max": features.detach().norm(dim=-1).max(),
                "whole_tangent_norm_mean": whole_tangent.norm(dim=-1).mean(),
                "whole_tangent_norm_max": whole_tangent.norm(dim=-1).max(),
                "part_tangent_norm_mean": pooled.norm(dim=-1).mean(),
                "part_tangent_norm_max": pooled.norm(dim=-1).max(),
            }
            info.update(_scalar_metrics(norm_metrics))
            index_bytes = part_indices.detach().to(device="cpu", dtype=torch.int64).contiguous().numpy().tobytes()
            info.update({
                "anchor_classes": labels_cpu.tolist(),
                "negative_indices": indices_cpu.tolist(),
                "negative_classes": negative_labels.tolist(),
                "candidate_counts": counts_cpu.tolist(),
                "negative_valid_mask": valid_cpu.tolist(),
                "negative_legal_rate": 1.0 if len(selected) else None,
                "same_class_pair_fraction": same_class,
                "self_pair_fraction": self_pair,
                "negative_unique_count": unique_negatives,
                "negative_reuse_count": len(selected) - unique_negatives,
                "negative_max_reuse": max_reuse,
                "negative_hit_counts": hits.tolist(),
                "contrast_denominator": len(selected),
                "hier_denominator": len(labels_cpu),
                "no_candidate_anchors": int((~valid_cpu).sum()),
                "part_centers": centers.detach().cpu().tolist(),
                "part_indices_sha256": hashlib.sha256(index_bytes).hexdigest(),
                "part_indices_shape": list(part_indices.shape),
                "part_points": size, "gamma": gamma, "delta": self.delta,
                "feature_dim": width,
            })
        return weighted_contrast + weighted_radial, info
