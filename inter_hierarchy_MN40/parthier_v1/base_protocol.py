"""Native-c1 HyCoRe with one frozen-source, nested 200/400/600 crop.

The 600-point crop overwrites the whole prefix once. Three independent input
copies are encoded, small to large, followed by the overwritten whole. No cap,
fixed-depth mapping or extra gradient hook is added. Optional non-reentrant
checkpointing restores BN buffers after recompute, retaining four effective
BN updates without storing all four backbone activation graphs.
"""
from __future__ import annotations

from contextlib import nullcontext
import random

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from ..hier_proxy_scratch_v5.distributed import gather_with_grad, gather_without_grad
from ..hier_proxy_scratch_v9.base_protocol import knn_indices, intra_losses, smoothed_ce
from ..hier_proxy_scratch_v9.geometry import configure_backbone, freeze_curvature


PART_COUNTS = (200, 400, 600)
CROP_VERSION = "parthier-v1-frozen-whole-single600-prefix-1"


def _centers(value, batch, count, device):
    if value is None:
        value = [random.randint(0, count - 1) for _ in range(batch)]
    centers = torch.as_tensor(value, device=device)
    if centers.ndim != 1 or len(centers) != batch:
        raise ValueError("one crop center is required per instance")
    if centers.dtype == torch.bool or centers.is_floating_point() or centers.is_complex():
        raise TypeError("crop centers must be integer indices")
    centers = centers.to(dtype=torch.long)
    if bool(((centers < 0) | (centers >= count)).any()):
        raise ValueError("crop center outside the source point set")
    return centers


def _ordered_nearest(source, centers, count):
    # Retain the original HyCoRe squared-distance/topk operator and its tie
    # behavior. Select the center row once; all smaller crops use its prefix.
    neighbors = knn_indices(source[:, :3], count)
    rows = torch.arange(len(source), device=source.device)
    return neighbors[rows, centers]


def _gather_points(source, indices):
    return source.gather(2, indices[:, None, :].expand(-1, source.shape[1], -1))


def sample_nested_parts(points, whole_count, *, centers=None, whole_centers=None):
    """Return overwritten whole, independent nested inputs and source indices.

    ``centers`` addresses the frozen whole, before the 600-prefix overwrite.
    ``whole_centers`` addresses the original input. Explicit centers support
    reproducible input audits; normal training draws each center exactly once.
    Metadata contains indices only, never a second raw point-cloud payload.
    """
    if points.ndim != 3 or points.shape[-1] != 3 or len(points) == 0:
        raise ValueError("input points must have shape [local_batch,N,3]")
    if not points.is_floating_point():
        raise TypeError("input points must be floating point")
    if isinstance(whole_count, bool) or int(whole_count) != whole_count:
        raise ValueError("whole_count must be an integer")
    whole_count = int(whole_count)
    if not PART_COUNTS[-1] <= whole_count <= points.shape[1]:
        raise ValueError("whole_count must contain 600 points and fit the input")

    source = points.transpose(1, 2).contiguous()
    whole_centers = _centers(whole_centers, len(source), source.shape[-1], source.device)
    whole_indices = _ordered_nearest(source, whole_centers, whole_count)
    selected_whole = _gather_points(source, whole_indices)
    whole = source[:, :, :whole_count]
    whole.copy_(selected_whole)

    # A detached view would still share mutable storage. This independent
    # snapshot is the only source used for the complete 600-neighbor selection.
    frozen_whole = whole.clone()
    centers = _centers(centers, len(source), whole_count, source.device)
    part_indices = _ordered_nearest(frozen_whole, centers, PART_COUNTS[-1])
    selected600 = _gather_points(frozen_whole, part_indices)
    whole[:, :, :PART_COUNTS[-1]].copy_(selected600)
    part_inputs = tuple(whole[:, :, :count].clone() for count in PART_COUNTS)

    stats = {
        "crop_version": CROP_VERSION,
        "whole_point_count": whole_count,
        "part_point_counts": list(PART_COUNTS),
        "single_part_overwrite_count": 1,
        "parts_nested": all(torch.equal(part_inputs[i], part_inputs[-1][:, :, :count])
                            for i, count in enumerate(PART_COUNTS)),
        "part600_matches_whole_prefix": bool(torch.equal(part_inputs[-1], whole[:, :, :600])),
        "whole_suffix_unchanged": bool(torch.equal(whole[:, :, 600:], frozen_whole[:, :, 600:])),
        "whole_changed_by_part600": bool(not torch.equal(whole, frozen_whole)),
        "parts_alias_whole": any(part.untyped_storage().data_ptr() == whole.untyped_storage().data_ptr()
                                 for part in part_inputs),
        "whole_aliases_source": whole.untyped_storage().data_ptr() == source.untyped_storage().data_ptr(),
        "whole_centers": whole_centers.detach(),
        "part_centers": centers.detach(),
        "whole_indices": whole_indices.detach(),
        "part_indices": part_indices.detach(),
        "part_source_indices": whole_indices.gather(1, part_indices).detach(),
        "fps_first_stage_requested": 512,
        "fps_minimum_repeated_indices_by_scale": {str(count): max(0, 512 - count)
                                                  for count in PART_COUNTS},
    }
    return whole, part_inputs, stats


class _PreserveBatchNormBuffers:
    """Reusable recompute context, including exceptional/early-stop exits.

    Snapshots are taken on each entry, rather than at checkpoint creation:
    backward begins after all four ordinary forward updates have completed.
    Recompute uses temporary clones so its backward-saved running buffers
    are never modified by a subsequent restoration copy. Finally the module
    regains its original buffer objects; no outer DDP forward occurs here.
    The stack also makes nested/repeated context entries safe.
    """

    def __init__(self, backbone):
        self.backbone = backbone
        self._snapshots = []

    def __enter__(self):
        states = []
        try:
            for module in self.backbone.modules():
                if isinstance(module, nn.modules.batchnorm._BatchNorm):
                    for name in ("running_mean", "running_var", "num_batches_tracked"):
                        original = getattr(module, name)
                        if original is not None:
                            temporary = original.detach().clone()
                            states.append((module, name, original))
                            setattr(module, name, temporary)
        except BaseException:
            # An allocation failure during entry must not strand a partially
            # replaced buffer set; __exit__ is not called when __enter__ fails.
            for module, name, original in reversed(states):
                setattr(module, name, original)
            raise
        self._snapshots.append(states)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for module, name, original in reversed(self._snapshots.pop()):
            setattr(module, name, original)
        return False


class HyCoReTrainingForward(nn.Module):
    """Keep all four shared-backbone calls inside one outer DDP forward."""

    def __init__(self, backbone, *, save_on_cpu=False, activation_checkpoint=False):
        super().__init__()
        if save_on_cpu and activation_checkpoint:
            raise ValueError("save_on_cpu and activation_checkpoint are mutually exclusive")
        self.backbone = backbone
        self.save_on_cpu = bool(save_on_cpu)
        self.activation_checkpoint = bool(activation_checkpoint)

    def _checkpoint_contexts(self):
        return nullcontext(), _PreserveBatchNormBuffers(self.backbone)

    def _backbone_forward(self, points, *, emb=False):
        forward = (lambda value: self.backbone(value, emb=True)) if emb else self.backbone
        if self.activation_checkpoint and torch.is_grad_enabled():
            # Non-reentrant checkpointing supports ordinary point inputs with
            # requires_grad=False and preserves parameter gradients. Cropping
            # remains outside this region; only Torch RNG must be replayed.
            return checkpoint(forward, points, use_reentrant=False,
                              preserve_rng_state=True, context_fn=self._checkpoint_contexts)
        return forward(points)

    def forward(self, points, whole_count, *, centers=None, whole_centers=None):
        whole, part_inputs, stats = sample_nested_parts(
            points, whole_count, centers=centers, whole_centers=whole_centers)
        scope = (torch.autograd.graph.save_on_cpu(pin_memory=points.is_cuda)
                 if self.save_on_cpu else nullcontext())
        # The CPU-offload and BN-preserving checkpoint paths are exclusive.
        with scope:
            parts = torch.stack([self._backbone_forward(part, emb=True)[0] for part in part_inputs], dim=1)
            mu, logits = self._backbone_forward(whole)
        stats["backbone_forward_calls"] = 4
        stats["activation_save_on_cpu"] = self.save_on_cpu
        stats["activation_checkpoint"] = self.activation_checkpoint
        return {"mu": mu, "parts": parts, "logits": logits,
                "whole_count": int(whole_count), "input_stats": stats}


def global_base_losses(local_outputs, local_labels, expected_global_batch=64):
    """Gather live features and add all three original intra terms at .01.

    Every rank computes the same global mean. Gather backward SUM and DDP
    parameter-gradient averaging cancel; no world-size factor belongs here.
    The contrastive negative is global flip separately for each part scale.
    """
    mu = gather_with_grad(local_outputs["mu"])
    parts = gather_with_grad(local_outputs["parts"])
    logits = gather_with_grad(local_outputs["logits"])
    labels = gather_without_grad(local_labels.reshape(-1).to(logits.device))
    if len(labels) != expected_global_batch:
        raise ValueError(f"expected global{expected_global_batch}, got {len(labels)}")
    if mu.ndim != 2 or parts.shape != (len(mu), len(PART_COUNTS), mu.shape[-1]):
        raise ValueError("parts must have shape [global_batch,3,embedding_dimension]")
    if logits.ndim != 2 or len(logits) != len(mu) or len(labels) != len(mu):
        raise ValueError("whole, parts, logits and labels must share the global batch")
    if bool((labels == labels.flip(0)).any()):
        raise RuntimeError("global flip negative shares the anchor class")
    whole_count = int(local_outputs["whole_count"])
    counts = gather_without_grad(torch.tensor([[whole_count, *PART_COUNTS]], device=mu.device))
    if bool((counts != counts[0]).any()):
        raise RuntimeError("whole/part counts must be shared across the global batch")
    per_scale = {str(count): {"child_count": count, **intra_losses(mu, parts[:, index], count)}
                 for index, count in enumerate(PART_COUNTS)}
    contrastive = sum(item["intra_contrastive"] for item in per_scale.values())
    radial = sum(item["intra_radial"] for item in per_scale.values())
    ce = smoothed_ce(logits, labels)
    base = ce + .01 * contrastive + .01 * radial
    return {"mu": mu, "parts": parts, "logits": logits, "gold": labels,
            "base": base, "ce": ce, "intra_contrastive": contrastive,
            "intra_radial": radial, "per_scale": per_scale,
            "whole_depth": per_scale["200"]["whole_depth"],
            "part_depth": torch.stack([item["part_depth"] for item in per_scale.values()]).mean(),
            "whole_radius": mu.detach().norm(dim=-1).mean(),
            "part_radius": parts.detach().norm(dim=-1).mean(),
            "global_batch_size": len(labels), "whole_count": whole_count,
            "part_counts": PART_COUNTS, "input_stats": local_outputs.get("input_stats", {})}
