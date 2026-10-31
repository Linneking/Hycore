"""Experiment I selection and H-compatible geometry at the 128-D M output."""

import hashlib

import torch
from torch import nn

from segmentation_ShapeNetPart.hycore_seg_v2.losses import choose_parts
from segmentation_ShapeNetPart.hycore_seg_v2.native import PoincareBall
from segmentation_ShapeNetPart.hycore_seg_v3.losses import (
    _scalar_metrics,
    relation_losses,
    sample_negative_indices,
)


class HyCoReSegLoss(nn.Module):
    """Pick H-style feature neighbors, then regularize separately encoded µ128."""

    def __init__(self, alpha=.01, beta=.01, gamma_base=1000., delta=4.,
                 part_min=200, part_max=600, seed=4780, feature_dim=128):
        super().__init__()
        if (alpha, beta) != (.01, .01):
            raise ValueError("Experiment I requires both regularizers at .01")
        if gamma_base < 0 or delta < 0 or not 1 <= part_min <= part_max:
            raise ValueError("Invalid margins or part size interval")
        if feature_dim != 128:
            raise ValueError("Experiment I regularizes the 128-D first-M output")
        self.alpha, self.beta = float(alpha), float(beta)
        self.gamma_base, self.delta = float(gamma_base), float(delta)
        self.part_min, self.part_max = int(part_min), int(part_max)
        self.feature_dim = int(feature_dim)
        self.ball = PoincareBall(c=1., dim=feature_dim)
        # Keep H's independent sampling streams and their checkpoint keys.
        self.generator = torch.Generator(device="cpu").manual_seed(seed + 100003)
        self.negative_generator = torch.Generator(device="cpu").manual_seed(seed + 200003)
        self.monitor = None
        self.last_components = {}
        self.regularizers_active = True

    def select_parts(self, whole_features):
        """Return H's stable feature-neighbor indices, not a pooled feature."""
        if whole_features.ndim != 3 or whole_features.shape[-1] != 256:
            raise ValueError("Part selection expects whole post-Conv9 [B,N,256]")
        with torch.no_grad():
            _, indices, centers, size = choose_parts(
                whole_features.detach(), self.generator, self.part_min, self.part_max)
        return indices, centers, size

    def forward(self, whole_mu, part_mu, categories, part_size,
                part_indices, part_centers, whole_features, part_features):
        self.last_components = {}
        if whole_mu.ndim != 2 or part_mu.shape != whole_mu.shape or whole_mu.shape[-1] != 128:
            raise ValueError("Expected paired whole/part ball embeddings [B,128]")
        if categories.numel() != len(whole_mu):
            raise ValueError("Expected one category per whole/part pair")
        if whole_features.ndim != 3 or whole_features.shape[-1] != 256:
            raise ValueError("Expected whole post-Conv9 [B,N,256]")
        if part_features.ndim != 3 or part_features.shape != (len(whole_mu), part_size, 256):
            raise ValueError("Expected reencoded part [B,Npart,256]")
        if part_indices.shape != (len(whole_mu), part_size) or part_centers.shape != (len(whole_mu),):
            raise ValueError("Part indices/centers do not match embeddings")
        if not self.part_min <= part_size <= self.part_max:
            raise ValueError("Part size is outside the configured interval")
        if ((part_indices < 0) | (part_indices >= whole_features.shape[1])).any():
            raise ValueError("Part index out of range")

        ball = self.ball.to(device=whole_mu.device, dtype=whole_mu.dtype)
        indices, counts, valid = sample_negative_indices(categories, self.negative_generator)
        gamma = self.gamma_base / part_size
        radial, contrast, info = relation_losses(
            ball, whole_mu, part_mu, gamma, self.delta, indices, valid)
        weighted_contrast = self.alpha * contrast
        weighted_radial = self.beta * radial
        self.last_components = {"contr": weighted_contrast, "hier": weighted_radial}

        with torch.no_grad():
            labels_cpu = categories.detach().reshape(-1).to(device="cpu", dtype=torch.long)
            negative_cpu = indices.cpu()
            valid_cpu = valid.cpu()
            selected = negative_cpu[valid_cpu]
            negative_labels = torch.full_like(negative_cpu, -1)
            negative_labels[valid_cpu] = labels_cpu[selected]
            if len(selected):
                if not bool((negative_labels[valid_cpu] != labels_cpu[valid_cpu]).all()):
                    raise AssertionError("A sampled negative has the anchor class")
                rows = torch.arange(len(labels_cpu))
                same_class = float((negative_labels[valid_cpu] == labels_cpu[valid_cpu]).float().mean())
                self_pair = float((selected == rows[valid_cpu]).float().mean())
                hits = torch.bincount(selected, minlength=len(labels_cpu))
                unique_negatives = int((hits > 0).sum())
                max_reuse = int(hits.max())
            else:
                same_class = self_pair = None
                hits = torch.zeros(len(labels_cpu), dtype=torch.long)
                unique_negatives = max_reuse = 0

            norms = {
                "feature_norm_mean": whole_features.detach().norm(dim=-1).mean(),
                "feature_norm_max": whole_features.detach().norm(dim=-1).max(),
                "part_feature_norm_mean": part_features.detach().norm(dim=-1).mean(),
                "part_feature_norm_max": part_features.detach().norm(dim=-1).max(),
                "whole_tangent_norm_mean": whole_features.detach().mean(dim=1).norm(dim=-1).mean(),
                "part_tangent_norm_mean": part_features.detach().mean(dim=1).norm(dim=-1).mean(),
            }
            info.update(_scalar_metrics(norms))
            index_bytes = part_indices.detach().to(device="cpu", dtype=torch.int64).contiguous().numpy().tobytes()
            info.update({
                "anchor_classes": labels_cpu.tolist(),
                "negative_indices": negative_cpu.tolist(),
                "negative_classes": negative_labels.tolist(),
                "candidate_counts": counts.cpu().tolist(),
                "negative_valid_mask": valid_cpu.tolist(),
                "negative_legal_rate": 1. if len(selected) else None,
                "same_class_pair_fraction": same_class,
                "self_pair_fraction": self_pair,
                "negative_unique_count": unique_negatives,
                "negative_reuse_count": len(selected) - unique_negatives,
                "negative_max_reuse": max_reuse,
                "negative_hit_counts": hits.tolist(),
                "contrast_denominator": len(selected),
                "hier_denominator": len(labels_cpu),
                "no_candidate_anchors": int((~valid_cpu).sum()),
                "part_centers": part_centers.detach().cpu().tolist(),
                "part_indices_sha256": hashlib.sha256(index_bytes).hexdigest(),
                "part_indices_shape": list(part_indices.shape),
                "part_points": int(part_size),
                "gamma": gamma,
                "delta": self.delta,
                "feature_dim": 128,
                "part_selection_feature_dim": 256,
                "regularizer_feature_location": "after_emb2_whole_and_reencoded_part",
                "part_selection_stage": "after_conv9_bn9_leaky_whole",
                "regularizers_active": True,
                "regularizer_diagnostics_only": False,
                "alpha": self.alpha,
                "beta": self.beta,
                "active_weighted_contr": float(weighted_contrast.detach()),
                "active_weighted_hier": float(weighted_radial.detach()),
            })
        return weighted_contrast + weighted_radial, info
