"""HIER-style learnable ancestor proxies for the HyCoRe inter experiment.

This module reproduces the *implemented* HIER gHHC objective: pair and triple
proxies are sampled independently with hard straight-through Gumbel softmax;
their three hinge terms are zeroed when both samples choose the same proxy,
then averaged over *all* sampled triplets. The source paper describes a
conditional triple-proxy choice, which the released code does not implement.

The corrections here are explicit: sampled negative indices never equal the
anchor or positive, and proxy mutual-kNN never contains self. Mining decisions
are detached; gradients flow through the selected proxy distances.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from geoopt import PoincareBall


Triplets = tuple[torch.Tensor, torch.Tensor, torch.Tensor]


class ProxyHierarchy(nn.Module):
    """Global unlabeled hyperbolic proxies in a fixed-curvature Poincare ball.

    ``tangent_proxies`` are ordinary trainable Euclidean parameters. Mapping
    them through ``expmap0`` avoids a Riemannian optimizer and ensures every
    forward proxy lies strictly within the ball. ``initialize`` should be
    called once with original-checkpoint whole embeddings before training.
    """

    def __init__(
        self,
        num_proxies: int = 128,
        dim: int = 256,
        c: float = 1.0,
        seed: int = 22,
        max_ball_fraction: float = 0.98,
    ) -> None:
        super().__init__()
        if num_proxies < 3 or dim < 1 or c <= 0:
            raise ValueError("need at least three proxies, positive dim and c")
        if not 0 < max_ball_fraction < 1:
            raise ValueError("max_ball_fraction must be in (0,1)")
        self.num_proxies = int(num_proxies)
        self.dim = int(dim)
        self.c = float(c)
        self.seed = int(seed)
        self.max_ball_fraction = float(max_ball_fraction)
        # Geoopt's PoincareBall infers dimension from each tensor; unlike the
        # repository's own wrapper it does not accept a ``dim`` constructor arg.
        self.ball = PoincareBall(c=self.c)
        self.tangent_proxies = nn.Parameter(torch.zeros(num_proxies, dim))

    @property
    def max_tangent_norm(self) -> float:
        return math.atanh(self.max_ball_fraction) / math.sqrt(self.c)

    @torch.no_grad()
    def initialize(self, reference_mu: torch.Tensor) -> dict[str, float]:
        """Initialize at half the median reference tangent norm, seed fixed.

        The magnitude is capped so the initial Euclidean ball radius is at
        most 0.8 times the curvature-dependent boundary radius. Directions
        come from a local CPU generator, leaving global experiment RNG alone.
        """
        if reference_mu.ndim != 2 or reference_mu.shape[1] != self.dim:
            raise ValueError(f"reference_mu must be [N,{self.dim}]")
        if reference_mu.shape[0] == 0:
            raise ValueError("reference_mu must be nonempty")
        if not bool(torch.isfinite(reference_mu).all()):
            raise ValueError("reference_mu contains NaN or Inf")
        boundary = 1.0 / math.sqrt(self.c)
        if not bool((reference_mu.norm(dim=-1) < boundary).all()):
            raise ValueError("reference_mu has points outside the Poincare ball")
        median_norm = float(self.ball.logmap0(reference_mu).norm(dim=-1).median())
        init_cap = math.atanh(0.8) / math.sqrt(self.c)
        target = min(max(0.5 * median_norm, 0.05), init_cap)
        generator = torch.Generator(device="cpu").manual_seed(self.seed)
        directions = torch.randn(
            self.num_proxies, self.dim, generator=generator, dtype=torch.float32
        )
        directions = F.normalize(directions, dim=-1)
        values = directions.to(
            device=self.tangent_proxies.device, dtype=self.tangent_proxies.dtype
        ) * target
        self.tangent_proxies.copy_(values)
        return {
            "reference_median_tangent_norm": median_norm,
            "init_tangent_norm": target,
            "init_ball_norm": float(self.proxies().norm(dim=-1).median()),
        }

    def proxies(self) -> torch.Tensor:
        # Tangent clipping prevents optimization from driving expmap0 into
        # a numerically saturated ball-boundary regime.
        norm = self.tangent_proxies.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        scale = (self.max_tangent_norm / norm).clamp(max=1.0)
        return self.ball.projx(self.ball.expmap0(self.tangent_proxies * scale))

    @staticmethod
    def _triplets(
        triplets: Sequence[torch.Tensor], n: int, device: torch.device
    ) -> Triplets:
        if len(triplets) != 3:
            raise ValueError("triplets must be (i,j,k)")
        i, j, k = (torch.as_tensor(t, dtype=torch.long, device=device).reshape(-1)
                   for t in triplets)
        if not (i.numel() == j.numel() == k.numel()):
            raise ValueError("triplet index arrays must have equal length")
        if i.numel():
            for t in (i, j, k):
                if bool(((t < 0) | (t >= n)).any()):
                    raise ValueError("triplet index out of range")
            if bool(((i == j) | (i == k) | (j == k)).any()):
                raise ValueError("triplets must have three distinct indices")
        return i, j, k

    @staticmethod
    def _hard_gumbel(
        logits: torch.Tensor, generator: Optional[torch.Generator]
    ) -> torch.Tensor:
        if generator is None:
            return F.gumbel_softmax(logits, tau=1.0, hard=True, dim=-1)
        # Equivalent hard straight-through sample with an explicit RNG for
        # deterministic unit tests and fixed-seed calibration.
        u = torch.rand(
            logits.shape, dtype=logits.dtype, device=logits.device,
            generator=generator,
        ).clamp(min=1e-6, max=1 - 1e-6)
        soft = F.softmax(logits - torch.log(-torch.log(u)), dim=-1)
        hard = F.one_hot(soft.argmax(dim=-1), num_classes=logits.shape[-1]).to(soft)
        return hard - soft.detach() + soft

    def _loss_from_distances(
        self,
        distances: torch.Tensor,
        triplets: Sequence[torch.Tensor],
        tau: float,
        margin: float,
        generator: Optional[torch.Generator],
    ) -> tuple[torch.Tensor, dict[str, float | int]]:
        if tau <= 0 or margin < 0:
            raise ValueError("tau must be positive and margin nonnegative")
        i, j, k = self._triplets(triplets, distances.shape[0], distances.device)
        count = int(i.numel())
        if count == 0:
            zero = distances.sum() * 0.0
            return zero, {
                "triplets": 0, "collisions": 0, "valid_triplets": 0,
                "active_triplets": 0, "collision_rate": 0.0,
                "active_rate": 0.0,
            }

        di, dj, dk = distances[i], distances[j], distances[k]
        pair_cost = torch.maximum(di, dj)
        triple_cost = torch.maximum(pair_cost, dk)
        p_ij = self._hard_gumbel(-pair_cost / tau, generator)
        p_ijk = self._hard_gumbel(-triple_cost / tau, generator)
        valid = p_ij.argmax(dim=-1) != p_ijk.argmax(dim=-1)

        i_pair, i_triple = (di * p_ij).sum(-1), (di * p_ijk).sum(-1)
        j_pair, j_triple = (dj * p_ij).sum(-1), (dj * p_ijk).sum(-1)
        k_pair, k_triple = (dk * p_ij).sum(-1), (dk * p_ijk).sum(-1)
        h_i = F.relu(i_pair - i_triple + margin)
        h_j = F.relu(j_pair - j_triple + margin)
        h_k = F.relu(k_triple - k_pair + margin)
        raw = h_i + h_j + h_k
        loss = (raw * valid.to(raw.dtype)).mean()
        with torch.no_grad():
            collisions = count - int(valid.sum())
            active = int(((raw > 0) & valid).sum())
        return loss, {
            "triplets": count,
            "collisions": collisions,
            "valid_triplets": count - collisions,
            "active_triplets": active,
            "collision_rate": collisions / count,
            "active_rate": active / count,
        }

    def sample_loss(
        self,
        mu: torch.Tensor,
        triplets: Sequence[torch.Tensor],
        tau: float = 0.1,
        margin: float = 0.1,
        generator: Optional[torch.Generator] = None,
    ) -> tuple[torch.Tensor, dict[str, float | int]]:
        """HIER loss from student whole embeddings to global proxies."""
        if mu.ndim != 2 or mu.shape[1] != self.dim:
            raise ValueError(f"mu must be [N,{self.dim}]")
        p = self.proxies()
        distances = self.ball.dist(mu[:, None, :], p[None, :, :])
        return self._loss_from_distances(distances, triplets, tau, margin, generator)

    @torch.no_grad()
    def mine_proxy_triplets(
        self,
        k: int = 8,
        t_per_anchor: int = 50,
        generator: Optional[torch.Generator] = None,
    ) -> tuple[Triplets, dict[str, int]]:
        """Mutual-kNN proxy triples, with all three indices distinct."""
        if k < 1 or k >= self.num_proxies:
            raise ValueError("proxy k must satisfy 1 <= k < num_proxies")
        if t_per_anchor < 1:
            raise ValueError("t_per_anchor must be positive")
        p = self.proxies()
        distances = self.ball.dist(p[:, None, :], p[None, :, :]).detach()
        distances.fill_diagonal_(float("inf"))
        nearest = distances.topk(k=k, dim=-1, largest=False).indices
        knn = torch.zeros_like(distances, dtype=torch.bool)
        knn.scatter_(dim=-1, index=nearest, value=True)
        mutual = knn & knn.T
        diag = torch.eye(self.num_proxies, device=p.device, dtype=torch.bool)
        mutual &= ~diag
        negative = ~mutual & ~diag
        # Match the released HIER requirement of >1 mutual positives per
        # anchor. Unlike its negative pool, our pool excludes self.
        eligible = (mutual.sum(-1) > 1) & negative.any(-1)
        anchors = eligible.nonzero(as_tuple=False).flatten()
        if anchors.numel() == 0:
            empty = torch.empty(0, device=p.device, dtype=torch.long)
            return (empty, empty, empty), {
                "eligible_anchors": 0, "mutual_pairs": int(mutual.sum().item()),
            }
        pos = torch.multinomial(
            mutual[anchors].float(), t_per_anchor, replacement=True,
            generator=generator,
        ).reshape(-1)
        neg = torch.multinomial(
            negative[anchors].float(), t_per_anchor, replacement=True,
            generator=generator,
        ).reshape(-1)
        i = anchors.repeat_interleave(t_per_anchor)
        return (i, pos, neg), {
            "eligible_anchors": int(anchors.numel()),
            "mutual_pairs": int(mutual.sum().item()),
        }

    def proxy_loss(
        self,
        k: int = 8,
        t_per_anchor: int = 50,
        tau: float = 0.1,
        margin: float = 0.1,
        generator: Optional[torch.Generator] = None,
    ) -> tuple[torch.Tensor, dict[str, float | int]]:
        """HIER gHHC applied to the unlabeled proxy–proxy graph."""
        triplets, mining_stats = self.mine_proxy_triplets(
            k=k, t_per_anchor=t_per_anchor, generator=generator
        )
        p = self.proxies()
        distances = self.ball.dist(p[:, None, :], p[None, :, :])
        loss, stats = self._loss_from_distances(
            distances, triplets, tau, margin, generator
        )
        return loss, {**stats, **mining_stats}
