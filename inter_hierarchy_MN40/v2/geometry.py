"""Numerically safe Poincare-ball geometry used by the v2 inter loss.

The legacy implementation passed two ``[N, D]`` tensors to ``ball.dist`` and
assumed the result was pairwise. Geoopt instead applies broadcasting in the
usual PyTorch sense, so equal shapes produce ``[N]`` elementwise distances.
All pairwise functions below expand the sample axes explicitly.
"""

from __future__ import annotations

import torch


def pairwise_ball_distance(ball, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Return all Poincare distances with shape ``[len(x), len(y)]``."""
    if x.ndim != 2 or y.ndim != 2 or x.shape[-1] != y.shape[-1]:
        raise ValueError("x and y must be [N,D] and [M,D] tensors")
    return ball.dist(x[:, None, :], y[None, :, :])


def gromov_product_matrix(ball, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Compute ``(x_i|y_j)_0`` for every pair.

    This is a tree-branch-depth approximation, not the exact Poincare LCA.
    """
    dx = ball.dist0(x).reshape(-1, 1)
    dy = ball.dist0(y).reshape(1, -1)
    return 0.5 * (dx + dy - pairwise_ball_distance(ball, x, y))


def poincare_to_klein(x: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    x2 = x.square().sum(dim=-1, keepdim=True)
    return 2.0 * x / (1.0 + x2).clamp_min(eps)


def klein_to_poincare(k: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    k2 = k.square().sum(dim=-1, keepdim=True).clamp_max(1.0 - eps)
    return k / (1.0 + torch.sqrt((1.0 - k2).clamp_min(eps)))


def exact_lca(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Closest-to-origin point on each Poincare geodesic joining x and y.

    Poincare geodesics become straight chords in the Klein model. Hyperbolic
    distance from the origin is monotone in the Klein norm, so the required
    point is the Euclidean projection of the origin onto each chord segment.
    ``x`` and ``y`` may have any broadcast-compatible leading dimensions.
    """
    kx = poincare_to_klein(x, eps)
    ky = poincare_to_klein(y, eps)
    delta = ky - kx
    denom = delta.square().sum(dim=-1, keepdim=True)
    t = -(kx * delta).sum(dim=-1, keepdim=True) / denom.clamp_min(eps)
    t = t.clamp(0.0, 1.0)
    # Coincident endpoints have an undefined chord parameter; either endpoint
    # is the LCA, so force t=0 without introducing NaNs.
    t = torch.where(denom > eps, t, torch.zeros_like(t))
    return klein_to_poincare(kx + t * delta, eps)


def exact_lca_depth(ball, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Pairwise exact LCA depths, shape ``[len(x), len(y)]``."""
    lca = exact_lca(x[:, None, :], y[None, :, :])
    return ball.dist0(lca)


def equal_radius_leaves(
    z: torch.Tensor,
    radius: torch.Tensor | float,
    eps: float = 4e-3,
) -> torch.Tensor:
    """Create same-direction leaves on a shared Euclidean Poincare radius."""
    if z.ndim != 2:
        raise ValueError("z must be [N,D]")
    r = torch.as_tensor(radius, dtype=z.dtype, device=z.device)
    r = r.clamp(min=1e-4, max=1.0 - eps)
    direction = z / z.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return direction * r
