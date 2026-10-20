"""V9 keeps native HyCoRe geometry and builds a separate HIER shell view.

Native whole/part: c=1, no additional output cap or inverse-metric hook.
HIER whole view: preserve direction, discard native radius, use c=.1/depth4.6.
The branch remains differentiable: its Jacobian removes direct radial gradients.
"""
from __future__ import annotations

import math

import torch

from ..hier_proxy_scratch_v8.geometry import (
    InverseMetric, official_distance, proxy_update_stats, to_ball,
)

NATIVE_C = 1.0
HIER_VIEW_C = 0.1
HIER_DEPTH = 4.6
HIER_TANGENT_CAP = 2.3
HIER_RADIUS = math.tanh(math.sqrt(HIER_VIEW_C) * HIER_DEPTH / 2) / math.sqrt(HIER_VIEW_C)
HIER_SHELL_GS = (1 - HIER_VIEW_C * HIER_RADIUS ** 2) ** 2 / 4
MIN_DIRECTION_NORM = 1e-8


def _validate_points(points):
    if not isinstance(points, torch.Tensor):
        raise TypeError("Geometry expects a Tensor")
    if points.dtype not in (torch.float32, torch.float64):
        raise TypeError("V9 geometry requires FP32 or FP64; mixed precision is disabled")
    if points.ndim != 2 or points.shape[0] == 0 or points.shape[1] < 2:
        raise ValueError("Geometry expects a nonempty [batch, dimension>=2] Tensor")
    if not bool(torch.isfinite(points).all()):
        raise ValueError("Geometry input contains nonfinite values")


def _manifolds(backbone):
    return (backbone.manifold, backbone.manifold2,
            backbone.emb.manifold, backbone.classifier.manifold)


def freeze_curvature(backbone):
    """Freeze existing native c=1 manifolds; never replace them or attach a hook."""
    for manifold in _manifolds(backbone):
        if not math.isclose(float(manifold.c), NATIVE_C, rel_tol=1e-6, abs_tol=1e-7):
            raise RuntimeError("V9 native HyCoRe manifold must retain c=1")
        manifold.requires_grad_(False)


def configure_backbone(backbone):
    """Validate an unmodified native backbone without changing its forward graph."""
    if hasattr(backbone, "v8_geometry") or hasattr(backbone, "v9_geometry"):
        raise RuntimeError("V9 native backbone must not carry an extra output mapping")
    if backbone.emb._forward_hooks or backbone.classifier._forward_hooks:
        raise RuntimeError("V9 native emb/classifier must not have an output hook")
    freeze_curvature(backbone)
    return backbone


def whole_to_hier(mu):
    """Map native c=1 whole directions to the HIER c=.1/depth4.6 shell.

    For r>0, d(view)/d(mu) = HIER_RADIUS/r * (I-u*u.T), followed by one GS.
    Do not detach either mu or its norm and do not install this on the backbone.
    Exactly/near-zero vectors have no reliable direction and are rejected.
    """
    _validate_points(mu)
    radius = mu.norm(dim=-1, keepdim=True)
    if bool((radius <= MIN_DIRECTION_NORM).any()):
        raise ValueError("HIER shell direction is undefined for native norm<=1e-8")
    if bool((radius >= 1 / math.sqrt(NATIVE_C)).any()):
        raise ValueError("Native whole must lie inside the c=1 Poincare ball")
    shell = (mu / radius) * HIER_RADIUS
    if not bool(torch.isfinite(shell).all()):
        raise RuntimeError("Nonfinite HIER shell view")
    return InverseMetric.apply(shell, HIER_VIEW_C)


def _origin_depth(points, c):
    _validate_points(points)
    scaled_radius = math.sqrt(c) * points.norm(dim=-1)
    if bool((scaled_radius >= 1).any()):
        raise ValueError("Point lies outside the specified Poincare ball")
    return 2 / math.sqrt(c) * torch.atanh(scaled_radius)


def native_depth(points):
    return _origin_depth(points, NATIVE_C)


def hier_depth(points):
    return _origin_depth(points, HIER_VIEW_C)


def assert_hier_shell(points):
    _validate_points(points)
    radius = points.norm(dim=-1)
    target = torch.full_like(radius, HIER_RADIUS)
    if not torch.allclose(radius, target, rtol=2e-6, atol=2e-6):
        raise RuntimeError("V9 HIER whole view must lie on the depth4.6 shell")


def _geometry_stats(points, c):
    radius = points.norm(dim=-1)
    values = _origin_depth(points, c)
    return {"c": c, "points": len(points), "depth_min": float(values.min()),
            "depth_mean": float(values.mean()), "depth_median": float(values.median()),
            "depth_max": float(values.max()), "radius_min": float(radius.min()),
            "radius_max": float(radius.max())}


@torch.no_grad()
def native_geometry_stats(points):
    report = _geometry_stats(points, NATIVE_C)
    report.update(space="native_hycore", extra_cap=False, inverse_metric_hook=False)
    return report


@torch.no_grad()
def hier_geometry_stats(points):
    assert_hier_shell(points)
    report = _geometry_stats(points, HIER_VIEW_C)
    report.update(space="hier_shell_view", target_depth=HIER_DEPTH,
                  inverse_metric_hook=True, inverse_metric_factor=HIER_SHELL_GS)
    return report