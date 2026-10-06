"""Prevent a flat expmap numerical cap by projecting optimized parameters."""
from __future__ import annotations
import math
import torch

MAX_NUMERICAL_PROXY_DEPTH = 2.0 * math.atanh(.999)


def validate_geometry(initial_depth, max_depth):
    if not (math.isfinite(initial_depth) and math.isfinite(max_depth) and
            0 < initial_depth <= max_depth < MAX_NUMERICAL_PROXY_DEPTH):
        raise ValueError("require 0 < initial depth <= max depth < numerical proxy depth")


@torch.no_grad()
def initialize_exact_depth(parameter, depth):
    if not math.isfinite(depth) or depth <= 0:
        raise ValueError("proxy depth must be finite and positive")
    norms = parameter.norm(dim=-1, keepdim=True)
    if bool((norms <= 0).any()):
        raise ValueError("proxy direction has zero norm")
    parameter.mul_((depth / 2.) / norms)


def calibrated_initial_depth(median_depth, requested=5., inside_ratio=.9):
    if not (math.isfinite(median_depth) and median_depth > 0 and
            math.isfinite(requested) and requested > 0 and 0 < inside_ratio < 1):
        raise ValueError("invalid whole median or initialization rule")
    return min(requested, inside_ratio * median_depth)


@torch.no_grad()
def project_proxy_parameter(parameter, optimizer, max_depth, moment_policy="retain"):
    """Standard projected AdamW; optional removal of outward first moment.

    Adam velocity has opposite sign to exp_avg. The optional policy removes
    negative radial exp_avg only on projected rows; exp_avg_sq is unchanged.
    """
    if moment_policy not in ("retain", "remove_outward"):
        raise ValueError("unknown proxy first-moment policy")
    validate_geometry(min(1., max_depth), max_depth)
    before = parameter.norm(dim=-1)
    cap = max_depth / 2.
    hit = before > cap
    parameter.mul_((cap / before.clamp_min(1e-15)).clamp(max=1.)[:, None])
    removed = 0
    state = optimizer.state.get(parameter, {}) if optimizer is not None else {}
    if moment_policy == "remove_outward" and "exp_avg" in state:
        first = state["exp_avg"]
        unit = parameter / parameter.norm(dim=-1, keepdim=True).clamp_min(1e-15)
        radial = (first * unit).sum(-1)
        outward = hit & (radial < 0)
        first.sub_(torch.where(outward, radial, torch.zeros_like(radial))[:, None] * unit)
        removed = int(outward.sum())
    after = parameter.norm(dim=-1)
    return {"hit_count": int(hit.sum()), "hit_fraction": float(hit.float().mean()),
            "pre_max_tangent_norm": float(before.max()),
            "post_max_tangent_norm": float(after.max()),
            "post_median_tangent_norm": float(after.median()),
            "post_max_depth": float(2 * after.max()),
            "removed_outward_first_moment_count": removed}


@torch.no_grad()
def assert_parameter_safe(parameter, max_depth, tolerance=1e-6):
    if not bool(torch.isfinite(parameter).all()) or float(parameter.norm(dim=-1).max()) > max_depth / 2. + tolerance:
        raise RuntimeError("V7 proxy parameter violates finite tangent-depth cap")
