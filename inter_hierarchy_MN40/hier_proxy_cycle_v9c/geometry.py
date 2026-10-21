"""C0.1/cap2.3 base and a separately scaled fixed-shell HIER branch.

The embedding hook caps once, then the base path gets one inverse-metric GS.
A wrapper may temporarily capture the cap result BEFORE GS for HIER. The
HIER branch normalizes that result and applies its own GS exactly once.
"""
from __future__ import annotations

import math
import torch
from torch import nn

from ..hier_proxy_scratch_v8.geometry import (
    InverseMetric, official_distance, proxy_update_stats, to_ball,
)

C = NATIVE_C = HIER_VIEW_C = 0.1
HIER_DEPTH = 4.6
HIER_TANGENT_CAP = 2.3
HIER_RADIUS = math.tanh(math.sqrt(C) * HIER_DEPTH / 2) / math.sqrt(C)
HIER_SHELL_GS = (1 - C * HIER_RADIUS ** 2) ** 2 / 4
MIN_DIRECTION_NORM = 1e-8


def _validate_points(points):
    if not isinstance(points, torch.Tensor):
        raise TypeError("Geometry expects a Tensor")
    if points.dtype not in (torch.float32, torch.float64):
        raise TypeError("Cycle geometry requires FP32 or FP64")
    if points.ndim != 2 or points.shape[0] == 0 or points.shape[1] < 2:
        raise ValueError("Geometry expects nonempty [batch, dimension>=2]")
    if not bool(torch.isfinite(points).all()):
        raise ValueError("Geometry contains nonfinite values")


class FinalGeometry(nn.Module):
    """Forward hook; graph capture is opt-in and cleared by the DDP wrapper."""
    def __init__(self):
        super().__init__()
        self.capture_enabled = False
        self.pre_gs = None

    def forward(self, module, inputs, mu):
        capped = to_ball(module.manifold.logmap0(mu), c=C,
                         clip_r=HIER_TANGENT_CAP, hook=False)
        if self.capture_enabled:
            if self.pre_gs is not None:
                raise RuntimeError("Stale pre-GS graph capture")
            self.pre_gs = capped
        return InverseMetric.apply(capped, C)

    def clear(self):
        self.capture_enabled = False
        self.pre_gs = None


def freeze_curvature(backbone):
    for manifold in (backbone.manifold, backbone.manifold2,
                     backbone.emb.manifold, backbone.classifier.manifold):
        manifold.requires_grad_(False)
        if not math.isclose(float(manifold.c), C, rel_tol=1e-5):
            raise RuntimeError("Every backbone operation must use c=.1")


def configure_backbone(backbone):
    """Exact ORIG c.1/cap/GS mapping, without consuming random numbers."""
    if any(hasattr(backbone, name) for name in ("v8_geometry", "v9_geometry", "v9c_geometry")):
        raise RuntimeError("Final geometry must be installed exactly once")
    if backbone.emb._forward_hooks or backbone.classifier._forward_hooks:
        raise RuntimeError("Unexpected existing embedding/classifier hook")
    first = type(backbone.manifold)(dim=512, c=C)
    second = type(backbone.manifold2)(dim=256, c=C)
    first.requires_grad_(False)
    second.requires_grad_(False)
    backbone.manifold, backbone.manifold2 = first, second
    backbone.emb.manifold = first
    backbone.classifier.manifold = second
    backbone.v9c_geometry = FinalGeometry()
    backbone.emb.register_forward_hook(backbone.v9c_geometry)
    freeze_curvature(backbone)
    return backbone


def whole_to_hier(pre_gs_whole):
    """Build depth4.6 shell from PRE-GS coordinates, with one branch GS.

    Input must be the captured cap result, never the base-path GS output.
    Input/output positions coincide numerically with base where already on
    the cap, but their autograd paths differ. No detach or straight-through.
    """
    _validate_points(pre_gs_whole)
    radius = pre_gs_whole.norm(dim=-1, keepdim=True)
    if bool((radius <= MIN_DIRECTION_NORM).any()):
        raise ValueError("HIER direction undefined for norm<=1e-8")
    if bool((radius >= 1 / math.sqrt(C)).any()):
        raise ValueError("Source lies outside c=.1 ball")
    shell = pre_gs_whole / radius * HIER_RADIUS
    return InverseMetric.apply(shell, C)


def native_depth(points):
    _validate_points(points)
    scaled = math.sqrt(C) * points.norm(dim=-1)
    if bool((scaled >= 1).any()):
        raise ValueError("Point lies outside c=.1 ball")
    return 2 / math.sqrt(C) * torch.atanh(scaled)


hier_depth = native_depth


def assert_hier_shell(points):
    _validate_points(points)
    target = torch.full_like(points.norm(dim=-1), HIER_RADIUS)
    if not torch.allclose(points.norm(dim=-1), target, rtol=2e-6, atol=2e-6):
        raise RuntimeError("HIER whole view must lie on depth4.6 shell")


@torch.no_grad()
def native_geometry_stats(points):
    h = native_depth(points)
    r = points.norm(dim=-1)
    if bool((h > HIER_DEPTH + 5e-5).any()):
        raise RuntimeError("Base embedding exceeds cap2.3 depth")
    return {"c": C, "points": len(points), "depth_min": float(h.min()),
            "depth_mean": float(h.mean()), "depth_median": float(h.median()),
            "depth_max": float(h.max()), "radius_min": float(r.min()),
            "radius_max": float(r.max()), "space": "base_hycore_c01_capped",
            "extra_cap": True, "tangent_cap": HIER_TANGENT_CAP,
            "inverse_metric_hook": True,
            "cap_fraction": float((h >= 4.5999).float().mean())}


@torch.no_grad()
def hier_geometry_stats(points):
    assert_hier_shell(points)
    result = native_geometry_stats(points)
    result.update(space="hier_shell_view", target_depth=HIER_DEPTH,
                  inverse_metric_factor=HIER_SHELL_GS,
                  source="cap_after_pre_base_GS; separate GS once")
    return result
