"""Fixed-c=1 HIER geometry for a shared, untruncated HyCoRe whole embedding.

Distances use native ``d_1`` units. The default margin and selection
temperature are the source HIER values (0.1) multiplied by sqrt(0.1).
This is a distance-unit conversion for correspondingly scaled points, not
a claim that the different base networks follow equivalent trajectories.

The optional backward preconditioner is on the HIER branch only. Its forward
is the identity, so neither CE/intra values nor their gradients are changed.
Proxy coordinates receive the preconditioner before the expmap Jacobian is
propagated to ordinary tangent parameters. RSGD on those ordinary parameters
does not implicitly provide this embedding-coordinate preconditioner.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

import torch

from hier_proxy_v2_online.proxy_hierarchy import ProxyHierarchy as _ProxyHierarchy


CURVATURE = 1.0
SOURCE_CURVATURE = 0.1
SOURCE_MARGIN = 0.1
SOURCE_TEMPERATURE = 0.1
DEFAULT_MARGIN = SOURCE_MARGIN * math.sqrt(SOURCE_CURVATURE)
DEFAULT_TAU = SOURCE_TEMPERATURE * math.sqrt(SOURCE_CURVATURE)
GRADIENT_MODES = ("euclidean", "riemannian")


def riemannian_factor(points: torch.Tensor, c: float = CURVATURE) -> torch.Tensor:
    """Return inverse conformal metric, one scalar for each point."""
    if c <= 0:
        raise ValueError("curvature magnitude must be positive")
    return (1.0 - c * points.square().sum(-1, keepdim=True)).square() / 4.0


class _HIERGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, points: torch.Tensor, c: float) -> torch.Tensor:
        ctx.save_for_backward(points)
        ctx.c = c
        return points.view_as(points)

    @staticmethod
    def backward(ctx, gradient: torch.Tensor):
        (points,) = ctx.saved_tensors
        return riemannian_factor(points, ctx.c) * gradient, None


def hier_gradient(
    points: torch.Tensor, mode: str = "euclidean", c: float = CURVATURE
) -> torch.Tensor:
    """Identity forward; choose ordinary or metric-preconditioned backward."""
    if mode not in GRADIENT_MODES:
        raise ValueError(f"gradient mode must be one of {GRADIENT_MODES}")
    return _HIERGradient.apply(points, c) if mode == "riemannian" else points


def radius_gradient_audit(
    radii: Sequence[float] = (0.0, 0.5, 0.6214253856, 0.8, 0.98, 0.996)
) -> list[dict[str, float]]:
    """Predicted gradient suppression at relative ball radii, all at c=1."""
    result = []
    for radius in radii:
        if not 0 <= radius < 1:
            raise ValueError("relative ball radii must be in [0,1)")
        result.append({
            "relative_ball_radius": float(radius),
            "radial_depth_d1": 2.0 * math.atanh(radius),
            "riemannian_to_euclidean_factor": (1.0 - radius * radius) ** 2 / 4.0,
        })
    return result


def source_cap_intra_audit(part_counts: Sequence[int] = (200, 600)) -> dict:
    """Show why copying the source final output cap conflicts with Rhier.

    This concerns a cap on FINAL shared mu. Clipping before another Mobius
    layer does not necessarily bound the final radius by the same number.
    """
    tangent_cap = 2.3 * math.sqrt(SOURCE_CURVATURE)
    max_depth = 2.0 * tangent_cap
    lower_bounds = []
    for count in part_counts:
        if count < 1:
            raise ValueError("part point counts must be positive")
        lower_bounds.append({
            "part_points": int(count),
            "intra_radial_margin": 1000.0 / count,
            "minimum_hinge_even_with_part_at_origin": max(0.0, 1000.0 / count - max_depth),
        })
    return {
        "c": CURVATURE,
        "source_c": SOURCE_CURVATURE,
        "source_tangent_cap": 2.3,
        "equivalent_c1_tangent_cap": tangent_cap,
        "equivalent_relative_ball_cap": math.tanh(tangent_cap),
        "maximum_whole_radial_depth_d1": max_depth,
        "intra_lower_bounds": lower_bounds,
        "whole_output_capped_in_v4": False,
    }


class ProxyHierarchy(_ProxyHierarchy):
    """Source max-cost/Gumbel/three-hinge core with explicit c=1 backward.

    ``gradient_mode`` controls whole-coordinate HIER gradients. Proxy
    coordinates can be controlled separately using ``proxy_gradient_mode``;
    by default both use the same choice. ``proxies()`` continues to return
    actual coordinates for detached mining, initialization and diagnostics.
    Only live loss paths are preconditioned.
    """

    def __init__(
        self,
        num_proxies: int = 256,
        dim: int = 256,
        c: float = CURVATURE,
        seed: int = 22,
        max_ball_fraction: float = 0.98,
        gradient_mode: str = "euclidean",
        proxy_gradient_mode: Optional[str] = None,
    ) -> None:
        if abs(c - CURVATURE) > 1e-12:
            raise ValueError("v4 requires fixed curvature c=1")
        proxy_mode = gradient_mode if proxy_gradient_mode is None else proxy_gradient_mode
        if gradient_mode not in GRADIENT_MODES or proxy_mode not in GRADIENT_MODES:
            raise ValueError(f"gradient modes must be one of {GRADIENT_MODES}")
        super().__init__(num_proxies, dim, CURVATURE, seed, max_ball_fraction)
        self.gradient_mode = gradient_mode
        self.proxy_gradient_mode = proxy_mode

    def _sample_loss_with_modes(
        self,
        mu: torch.Tensor,
        triplets: Sequence[torch.Tensor],
        tau: float,
        margin: float,
        generator: Optional[torch.Generator],
        gradient_mode: str,
        proxy_gradient_mode: str,
    ) -> tuple[torch.Tensor, dict[str, float | int]]:
        if mu.ndim != 2 or mu.shape[1] != self.dim:
            raise ValueError(f"mu must be [N,{self.dim}]")
        whole = hier_gradient(mu, gradient_mode, self.c)
        proxies = hier_gradient(self.proxies(), proxy_gradient_mode, self.c)
        distances = self.ball.dist(whole[:, None, :], proxies[None, :, :])
        return self._loss_from_distances(distances, triplets, tau, margin, generator)

    def sample_loss(
        self,
        mu: torch.Tensor,
        triplets: Sequence[torch.Tensor],
        tau: float = DEFAULT_TAU,
        margin: float = DEFAULT_MARGIN,
        generator: Optional[torch.Generator] = None,
    ) -> tuple[torch.Tensor, dict[str, float | int]]:
        return self._sample_loss_with_modes(
            mu, triplets, tau, margin, generator,
            self.gradient_mode, self.proxy_gradient_mode,
        )

    def proxy_loss(
        self,
        k: int = 20,
        t_per_anchor: int = 50,
        tau: float = DEFAULT_TAU,
        margin: float = DEFAULT_MARGIN,
        generator: Optional[torch.Generator] = None,
    ) -> tuple[torch.Tensor, dict[str, float | int]]:
        triplets, mining_stats = self.mine_proxy_triplets(k, t_per_anchor, generator)
        proxies = hier_gradient(self.proxies(), self.proxy_gradient_mode, self.c)
        distances = self.ball.dist(proxies[:, None, :], proxies[None, :, :])
        loss, stats = self._loss_from_distances(distances, triplets, tau, margin, generator)
        return loss, {**stats, **mining_stats}

    def gradient_mode_audit(
        self,
        mu: torch.Tensor,
        triplets: Sequence[torch.Tensor],
        tau: float = DEFAULT_TAU,
        margin: float = DEFAULT_MARGIN,
        seed: int = 109,
    ) -> dict[str, float | int]:
        """Compare both backward paths on identical geometry/Gumbel samples.

        Does not accumulate gradients or consume global RNG. The mu leaf is
        detached so this audit does not retain or backpropagate the backbone.
        Proxy gradients are measured AFTER the expmap Jacobian in both paths.
        """
        modes = {}
        for mode in GRADIENT_MODES:
            audit_mu = mu.detach().clone().requires_grad_(True)
            generator = torch.Generator(device=audit_mu.device).manual_seed(seed)
            loss, stats = self._sample_loss_with_modes(
                audit_mu, triplets, tau, margin, generator, mode, mode,
            )
            g_mu, g_tangent = torch.autograd.grad(
                loss, (audit_mu, self.tangent_proxies), allow_unused=False,
            )
            modes[mode] = {
                "loss": float(loss.detach()),
                "mu_norm": float(g_mu.detach().norm()),
                "proxy_tangent_norm": float(g_tangent.detach().norm()),
                "collisions": int(stats["collisions"]),
                "active_triplets": int(stats["active_triplets"]),
            }
        factors = riemannian_factor(mu.detach(), self.c).flatten()
        raw, preconditioned = modes["euclidean"], modes["riemannian"]
        return {
            "raw_loss": raw["loss"],
            "riemannian_loss": preconditioned["loss"],
            "mu_raw_grad_norm": raw["mu_norm"],
            "mu_riemannian_grad_norm": preconditioned["mu_norm"],
            "mu_riemannian_to_raw_ratio": preconditioned["mu_norm"] / max(raw["mu_norm"], 1e-30),
            "proxy_tangent_raw_grad_norm": raw["proxy_tangent_norm"],
            "proxy_tangent_riemannian_grad_norm": preconditioned["proxy_tangent_norm"],
            "proxy_tangent_riemannian_to_raw_ratio": preconditioned["proxy_tangent_norm"] / max(raw["proxy_tangent_norm"], 1e-30),
            "mu_factor_min": float(factors.min()),
            "mu_factor_median": float(factors.median()),
            "mu_factor_max": float(factors.max()),
            "collisions": raw["collisions"],
            "active_triplets": raw["active_triplets"],
        }
