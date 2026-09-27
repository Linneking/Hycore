"""Deterministic, read-only diagnostics for online HIER proxy experiments.

All returned scalar metrics are plain Python numbers for CSV/JSON logging.
The accompanying snapshot contains detached CPU tensors for later inspection.
The fixed probe is diagnostic only: its triplets are not used for training.
"""

from __future__ import annotations

from contextlib import contextmanager
import math
import random
from typing import Iterator, Sequence, TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from .proxy_hierarchy import ProxyHierarchy


@contextmanager
def fixed_rng(seed: int, device: torch.device | str) -> Iterator[None]:
    """Seed a probe temporarily and restore Python, NumPy and Torch RNGs.

    CUDA states on every visible GPU are preserved when a CUDA probe is used
    or CUDA has already been initialized. This context is not thread-safe.
    """
    if not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA probe requested but CUDA is unavailable")

    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    preserve_cuda = torch.cuda.is_available() and (
        device.type == "cuda" or torch.cuda.is_initialized()
    )
    cuda_states = torch.cuda.get_rng_state_all() if preserve_cuda else None
    try:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if preserve_cuda:
            torch.cuda.manual_seed_all(seed)
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


def _initial_proxy_points(proxy: "ProxyHierarchy", initial_tangent: torch.Tensor) -> torch.Tensor:
    """Apply exactly the same clipping and map as ProxyHierarchy.proxies()."""
    if (torch.is_tensor(initial_tangent)
            and initial_tangent.device == proxy.tangent_proxies.device
            and initial_tangent.data_ptr() == proxy.tangent_proxies.data_ptr()):
        raise ValueError("initial_tangent must be a detached clone captured before training")
    tangent = torch.as_tensor(
        initial_tangent,
        device=proxy.tangent_proxies.device,
        dtype=proxy.tangent_proxies.dtype,
    )
    if tangent.shape != proxy.tangent_proxies.shape:
        raise ValueError("initial_tangent shape must match proxy.tangent_proxies")
    if not bool(torch.isfinite(tangent).all()):
        raise ValueError("initial_tangent contains NaN or Inf")
    norm = tangent.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    scale = (proxy.max_tangent_norm / norm).clamp(max=1.0)
    return proxy.ball.projx(proxy.ball.expmap0(tangent * scale))


def _spacing(proxy: "ProxyHierarchy", positions: torch.Tensor) -> torch.Tensor:
    matrix = proxy.ball.dist(positions[:, None, :], positions[None, :, :])
    matrix.fill_diagonal_(float("inf"))
    return matrix.min(dim=-1).values


@torch.no_grad()
def proxy_geometry(
    proxy: "ProxyHierarchy",
    initial_tangent: torch.Tensor,
    fixed_reference_mu: torch.Tensor | None = None,
    *,
    chunk_size: int = 256,
) -> tuple[dict[str, float | int], dict[str, torch.Tensor]]:
    """Measure real Poincare displacement and fixed-sample proxy assignments.

    ``fixed_reference_mu`` must contain the same unaugmented embeddings at
    every epoch if assignment changes are to be interpreted longitudinally.
    Assignments use nearest hyperbolic distance, not sampled HIER ancestors.
    Large reference sets are processed in chunks to bound GPU memory.
    """
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    initial = _initial_proxy_points(proxy, initial_tangent)
    current = proxy.proxies()
    zero = torch.zeros_like(current)
    displacement = proxy.ball.dist(initial, current)
    initial_radius = proxy.ball.dist(zero, initial)
    current_radius = proxy.ball.dist(zero, current)
    initial_spacing = _spacing(proxy, initial)
    current_spacing = _spacing(proxy, current)
    relative_displacement = displacement / initial_spacing.clamp_min(1e-12)
    for values in (displacement, initial_radius, current_radius,
                   initial_spacing, current_spacing, relative_displacement):
        if not bool(torch.isfinite(values).all()):
            raise FloatingPointError("non-finite proxy geometry")

    metrics: dict[str, float | int] = {
        "proxy_displacement_mean": float(displacement.mean()),
        "proxy_displacement_median": float(displacement.median()),
        "proxy_displacement_max": float(displacement.max()),
        "proxy_displacement_over_initial_spacing_median": float(
            relative_displacement.median()
        ),
        "proxy_fraction_moved_over_tenth_spacing": float(
            (relative_displacement > 0.1).float().mean()
        ),
        "proxy_radius_mean": float(current_radius.mean()),
        "proxy_radius_std": float(current_radius.std(unbiased=False)),
        "proxy_radius_change_mean": float((current_radius - initial_radius).mean()),
        "proxy_nearest_spacing_mean": float(current_spacing.mean()),
        "proxy_nearest_spacing_min": float(current_spacing.min()),
        "proxy_nearest_spacing_change_mean": float(
            (current_spacing - initial_spacing).mean()
        ),
    }
    snapshot = {
        "initial_ball": initial.detach().cpu(),
        "current_ball": current.detach().cpu(),
        "displacement": displacement.detach().cpu(),
        "relative_displacement": relative_displacement.detach().cpu(),
        "initial_radius": initial_radius.detach().cpu(),
        "current_radius": current_radius.detach().cpu(),
        "initial_nearest_spacing": initial_spacing.detach().cpu(),
        "current_nearest_spacing": current_spacing.detach().cpu(),
    }

    if fixed_reference_mu is not None:
        if fixed_reference_mu.ndim != 2 or fixed_reference_mu.shape[1] != proxy.dim:
            raise ValueError(f"fixed_reference_mu must be [N,{proxy.dim}]")
        if fixed_reference_mu.shape[0] == 0:
            raise ValueError("fixed_reference_mu must be nonempty")
        old_counts = torch.zeros(proxy.num_proxies, dtype=torch.long, device=current.device)
        new_counts = torch.zeros_like(old_counts)
        initial_assignments = []
        current_assignments = []
        current_nearest_distances = []
        changed = 0
        nearest_sum = 0.0
        count = int(fixed_reference_mu.shape[0])
        for start in range(0, count, chunk_size):
            mu = fixed_reference_mu[start:start + chunk_size].to(
                device=current.device, dtype=current.dtype
            )
            if not bool(torch.isfinite(mu).all()):
                raise ValueError("fixed_reference_mu contains NaN or Inf")
            old_dist = proxy.ball.dist(mu[:, None, :], initial[None, :, :])
            new_dist = proxy.ball.dist(mu[:, None, :], current[None, :, :])
            old_assign = old_dist.argmin(dim=-1)
            nearest, new_assign = new_dist.min(dim=-1)
            old_counts += torch.bincount(old_assign, minlength=proxy.num_proxies)
            new_counts += torch.bincount(new_assign, minlength=proxy.num_proxies)
            initial_assignments.append(old_assign.cpu())
            current_assignments.append(new_assign.cpu())
            current_nearest_distances.append(nearest.cpu())
            changed += int((old_assign != new_assign).sum())
            nearest_sum += float(nearest.sum())
        frequencies = new_counts.float() / count
        nonempty = frequencies[frequencies > 0]
        entropy = float(-(nonempty * nonempty.log()).sum())
        metrics.update({
            "proxy_reference_count": count,
            "proxy_occupied": int((new_counts > 0).sum()),
            "proxy_occupancy_max_fraction": float(new_counts.max()) / count,
            "proxy_occupancy_entropy_norm": (
                entropy / math.log(proxy.num_proxies)
                if proxy.num_proxies > 1 else 0.0
            ),
            "proxy_assignment_change_fraction": changed / count,
            "proxy_nearest_distance_mean": nearest_sum / count,
        })
        snapshot["initial_occupancy"] = old_counts.cpu()
        snapshot["current_occupancy"] = new_counts.cpu()
        snapshot["initial_assignment"] = torch.cat(initial_assignments)
        snapshot["current_assignment"] = torch.cat(current_assignments)
        snapshot["current_nearest_distance"] = torch.cat(current_nearest_distances)
    return metrics, snapshot


@torch.no_grad()
def fixed_triplet_metrics(
    proxy: "ProxyHierarchy",
    initial_tangent: torch.Tensor,
    initial_mu: torch.Tensor,
    current_mu: torch.Tensor,
    triplets: Sequence[torch.Tensor],
    *,
    seed: int = 22,
    tau: float = 0.1,
    margin: float = 0.1,
) -> dict[str, float | int]:
    """Evaluate fixed triplets with initial/current student × proxy states.

    The four combinations use identical Gumbel random draws. A difference
    between s0_p0 and s0_p1 isolates proxy movement with fixed embeddings;
    s0_p0 versus s1_p0 isolates student movement with fixed proxies. These
    comparisons remain descriptive because sampled ancestor choices may flip.
    """
    if not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    initial_p = _initial_proxy_points(proxy, initial_tangent)
    current_p = proxy.proxies()
    device = current_p.device
    for mu in (initial_mu, current_mu):
        if mu.ndim != 2 or mu.shape[1] != proxy.dim:
            raise ValueError(f"student mu must be [N,{proxy.dim}]")
        if mu.shape[0] != initial_mu.shape[0]:
            raise ValueError("initial_mu and current_mu need equal sample count")
    result: dict[str, float | int] = {}
    for student_key, mu in (("s0", initial_mu), ("s1", current_mu)):
        mu = mu.to(device=device, dtype=current_p.dtype)
        for proxy_key, positions in (("p0", initial_p), ("p1", current_p)):
            distances = proxy.ball.dist(mu[:, None, :], positions[None, :, :])
            generator = torch.Generator(device=device).manual_seed(seed)
            loss, stats = proxy._loss_from_distances(
                distances, triplets, tau, margin, generator
            )
            prefix = f"probe_{student_key}_{proxy_key}_"
            valid = stats["valid_triplets"]
            result[prefix + "loss"] = float(loss)
            result[prefix + "triplets"] = int(stats["triplets"])
            result[prefix + "valid_rate"] = (
                valid / stats["triplets"] if stats["triplets"] else 0.0
            )
            result[prefix + "active_rate"] = float(stats["active_rate"])
            result[prefix + "margin_satisfied_rate_valid"] = (
                1.0 - stats["active_triplets"] / valid if valid else 0.0
            )
    return result
