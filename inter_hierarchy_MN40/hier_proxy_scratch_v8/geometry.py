"""HIER-CVPR23 mapping/numerics; curvature is per operation, never global.

Source: sung-yeon-kim/HIER-CVPR23, commit 3986a744a1a54fd357e307d1cb3f2e81910b9ffc.
Final shared mu is capped once; the original point encoder is unchanged.
"""
from __future__ import annotations
import math
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

C, CLIP_R = .1, 2.3


class Artanh(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        x = x.clamp(-1 + 1e-5, 1 - 1e-5)
        ctx.save_for_backward(x)
        return (torch.log1p(x) - torch.log1p(-x)) * .5

    @staticmethod
    def backward(ctx, grad):
        x, = ctx.saved_tensors
        return grad / (1 - x.square())


class InverseMetric(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, c):
        ctx.save_for_backward(x)
        ctx.c = float(c)
        return x

    @staticmethod
    def backward(ctx, grad):
        x, = ctx.saved_tensors
        return grad * (1 - ctx.c * x.square().sum(-1, keepdim=True)).square() / 4, None


def to_ball(tangent, c=C, clip_r=CLIP_R, hook=True):
    norm = tangent.norm(dim=-1, keepdim=True) + 1e-5
    tangent = tangent * (clip_r / norm).clamp(max=1.)
    norm = tangent.norm(dim=-1, keepdim=True).clamp_min(1e-5)
    result = (math.sqrt(c) * norm).clamp(-15, 15).tanh() * tangent / (math.sqrt(c) * norm)
    radius = result.norm(dim=-1, keepdim=True).clamp_min(1e-5)
    result = result * ((1 - 1e-3) / math.sqrt(c) / radius).clamp(max=1.)
    return InverseMetric.apply(result, c) if hook else result


def depth(x, c=C):
    return 2 / math.sqrt(c) * torch.atanh((math.sqrt(c) * x.norm(dim=-1)).clamp(max=1 - 1e-7))


class FinalGeometry(nn.Module):
    def forward(self, module, inputs, mu):
        # Geoopt mu already belongs to c=.1; use its logmap then official cap/map.
        tangent = module.manifold.logmap0(mu)
        return to_ball(tangent)


def freeze_curvature(backbone):
    for manifold in (backbone.manifold, backbone.manifold2, backbone.emb.manifold,
                     backbone.classifier.manifold):
        manifold.requires_grad_(False)
        if not math.isclose(float(manifold.c), C, rel_tol=1e-5):
            raise RuntimeError("V8 manifold curvature differs from .1")


def configure_backbone(backbone):
    if hasattr(backbone, "v8_geometry"):
        raise RuntimeError("Final geometry must be installed exactly once")
    backbone.manifold = type(backbone.manifold)(dim=512, c=C)
    backbone.manifold2 = type(backbone.manifold2)(dim=256, c=C)
    backbone.emb.manifold = backbone.manifold
    backbone.classifier.manifold = backbone.manifold2
    backbone.v8_geometry = FinalGeometry()
    backbone.emb.register_forward_hook(backbone.v8_geometry)
    freeze_curvature(backbone)
    return backbone


@torch.no_grad()
def assert_output_geometry(points):
    if not bool(torch.isfinite(points).all()) or not bool((depth(points) <= 4.60005).all()):
        raise RuntimeError("V8 output exceeds the finite depth cap")


def _distance_block(x, y, c):
    # Literal released batched Mobius addition including denominator +1e-5.
    xy = (-x) @ y.T
    x2, y2 = x.square().sum(-1, keepdim=True), y.square().sum(-1)[None, :]
    num = (1 + 2 * c * xy + c * y2)[..., None] * (-x[:, None, :])
    num = num + (1 - c * x2)[:, :, None] * y
    denominator = 1 + 2 * c * xy + c * c * x2 * y2
    difference = num / (denominator[..., None] + 1e-5)
    return 2 / math.sqrt(c) * Artanh.apply(math.sqrt(c) * difference.norm(dim=-1))


def official_distance(x, y=None, c=C, block=32):
    y = x if y is None else y
    parts = []
    for start in range(0, len(x), block):
        a = x[start:start + block]
        if torch.is_grad_enabled() and (a.requires_grad or y.requires_grad):
            value = checkpoint(lambda u, v: _distance_block(u, v, c), a, y,
                               use_reentrant=False, preserve_rng_state=False)
        else:
            value = _distance_block(a, y, c)
        parts.append(value)
    return torch.cat(parts, 0)


@torch.no_grad()
def geometry_stats(points):
    r, h = points.norm(dim=-1), depth(points)
    factor = (1 - C * r.square()).square() / 4
    return {"c": C, "points": len(points), "depth_min": float(h.min()),
            "depth_median": float(h.median()), "depth_max": float(h.max()),
            "radius_max": float(r.max()), "cap_fraction": float((h >= 4.5999).float().mean()),
            "inverse_metric_factor_min": float(factor.min()),
            "inverse_metric_factor_median": float(factor.median())}


@torch.no_grad()
def proxy_update_stats(before, after):
    from ..hier_proxy_scratch_v6.structure_monitor import proxy_update_stats as old_stats
    report = old_stats(before, after)
    report["raw_ball_radius_change_mean"] = float(((math.sqrt(C) * after.norm(dim=-1)).tanh()
                                                  - (math.sqrt(C) * before.norm(dim=-1)).tanh()).mean() / math.sqrt(C))
    report["mapped_ball_radius_change_mean"] = float((to_ball(after, hook=False).norm(dim=-1)
                                                     - to_ball(before, hook=False).norm(dim=-1)).mean())
    report["semantics"] = "Actual AdamW tangent displacement; mapped radius includes official forward cap, c=.1"
    return report
