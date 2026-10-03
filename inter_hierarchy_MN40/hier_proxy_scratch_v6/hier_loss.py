"""V6 HIER: V5 geometry/loss with self-negative candidates excluded.

The released max-cost, independent hard Gumbel samples, three hinges and
collision-masked mean are retained. Extra tangent clipping and the HIER
inverse-metric backward hook are disabled; their predicted effects are
reported without modifying the live geometry. Numerical ball projection
still protects expmap output and is reported separately from tangent clip.
"""

from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F

from ..hier_proxy_scratch_v5.relations import mine_reciprocal_triplets, mine_sample_triplets, poincare_distance
from .structure_monitor import ancestor_summary


def expmap0_c1(tangent: torch.Tensor) -> torch.Tensor:
    norm = tangent.norm(dim=-1, keepdim=True).clamp_min(1e-15)
    raw = torch.tanh(norm) * tangent / norm
    # Released HIER pmath.project uses max norm 1-1e-3. This is ball
    # numerical safety, not the disabled clip_r=2.3 on tangent features.
    radius = raw.norm(dim=-1, keepdim=True).clamp_min(1e-15)
    return raw * (0.999 / radius).clamp(max=1.0)


@torch.no_grad()
def shadow_geometry(points: torch.Tensor, clip_r: float = 2.3) -> dict:
    """Predict source-style tangent clipping and inverse-metric factors."""
    radius = points.norm(dim=-1)
    tangent_radius = torch.atanh(radius.clamp(max=1 - torch.finfo(points.dtype).eps))
    factor = (1 - points.square().sum(-1)).square() / 4
    return {
        "points": len(points), "extra_tangent_clip_enabled": False,
        "riemannian_backward_enabled": False, "shadow_clip_r": clip_r,
        "shadow_clip_count": int((tangent_radius > clip_r).sum()),
        "shadow_clip_fraction": float((tangent_radius > clip_r).float().mean()),
        "ball_radius_min": float(radius.min()), "ball_radius_median": float(radius.median()),
        "ball_radius_max": float(radius.max()), "tangent_radius_max": float(tangent_radius.max()),
        "inverse_metric_factor_min": float(factor.min()),
        "inverse_metric_factor_median": float(factor.median()),
        "inverse_metric_factor_max": float(factor.max()),
    }


def _hard_gumbel(logits: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
    if generator is None:
        return F.gumbel_softmax(logits, dim=-1, hard=True)
    # The explicit-generator path consumes Exp(1) noise, as PyTorch's
    # F.gumbel_softmax does; default path remains literally the source API.
    noise = -torch.empty_like(logits).exponential_(generator=generator).log()
    soft = F.softmax(logits + noise, dim=-1)
    hard = F.one_hot(soft.argmax(-1), logits.shape[-1]).to(soft)
    return hard - soft.detach() + soft


def ghhc_loss(
    distances: torch.Tensor, triplets: torch.Tensor,
    tau: float = 0.1, margin: float = 0.1,
    generator: torch.Generator | None = None,
    return_details: bool = False,
) -> tuple:
    """Released compute_gHHC; k=i is explicitly accepted."""
    if tau <= 0 or margin < 0:
        raise ValueError("tau must be positive and margin nonnegative")
    triplets = torch.as_tensor(triplets, device=distances.device, dtype=torch.long).reshape(-1, 3)
    count = len(triplets)
    if count == 0:
        result = (distances.sum() * 0, {
            "triplets": 0, "collisions": 0, "collision_rate": 0.0,
            "noncollision_triplets": 0, "active_triplets": 0,
            "active_fraction_all_draws": 0.0, "active_fraction_noncollision": 0.0,
            "active_self_k_triplets": 0, "active_distinct_k_triplets": 0,
            "active_i_hinges": 0, "active_j_hinges": 0, "active_k_hinges": 0,
            "noncollision_self_k_triplets": 0, "self_k_loss_contribution": 0.0,
            "distinct_k_loss_contribution": 0.0,
        })
        if return_details:
            empty = distances.sum(-1)[:0]
            return (*result, {"per_draw_loss": empty, "raw_per_draw_loss": empty,
                              "masked_hinges": empty.reshape(0, 1).expand(0, 3),
                              "self_k_mask": torch.empty(0, dtype=torch.bool, device=distances.device),
                              "pair_proxy_idx": triplets[:, 0], "triple_proxy_idx": triplets[:, 0]})
        return result
    if bool(((triplets < 0) | (triplets >= distances.shape[0])).any()):
        raise ValueError("triplet index outside distance rows")
    i, j, k = triplets.unbind(-1)
    di, dj, dk = distances[i], distances[j], distances[k]
    pair_cost = torch.maximum(di, dj)
    triple_cost = torch.maximum(pair_cost, dk)
    pair_prob = _hard_gumbel(-pair_cost / tau, generator)
    triple_prob = _hard_gumbel(-triple_cost / tau, generator)
    pair_idx, triple_idx = pair_prob.argmax(-1), triple_prob.argmax(-1)
    valid = pair_idx != triple_idx
    hi = F.relu((di * pair_prob).sum(-1) - (di * triple_prob).sum(-1) + margin)
    hj = F.relu((dj * pair_prob).sum(-1) - (dj * triple_prob).sum(-1) + margin)
    hk = F.relu((dk * triple_prob).sum(-1) - (dk * pair_prob).sum(-1) + margin)
    raw = hi + hj + hk
    per_draw = raw * valid.to(raw.dtype)
    loss = per_draw.mean()
    with torch.no_grad():
        active = valid & (raw > 0)
        noncollision = int(valid.sum())
        active_count = int(active.sum())
        stats = {
            "triplets": count, "collisions": count - noncollision,
            "collision_rate": (count - noncollision) / count,
            "noncollision_triplets": noncollision, "active_triplets": active_count,
            "active_fraction_all_draws": active_count / count,
            "active_fraction_noncollision": active_count / noncollision if noncollision else 0.0,
            "active_self_k_triplets": int((active & (i == k)).sum()),
            "active_distinct_k_triplets": int((active & (i != k)).sum()),
            "active_i_hinges": int((valid & (hi > 0)).sum()),
            "active_j_hinges": int((valid & (hj > 0)).sum()),
            "active_k_hinges": int((valid & (hk > 0)).sum()),
            "noncollision_self_k_triplets": int((valid & (i == k)).sum()),
            "self_k_loss_contribution": float(per_draw[i == k].sum() / count),
            "distinct_k_loss_contribution": float(per_draw[i != k].sum() / count),
        }
    if return_details:
        return loss, stats, {
            "per_draw_loss": per_draw, "raw_per_draw_loss": raw,
            "masked_hinges": torch.stack((hi, hj, hk), dim=-1) * valid[:, None],
            "self_k_mask": i == k, "pair_proxy_idx": pair_idx, "triple_proxy_idx": triple_idx,
        }
    return loss, stats


class HIERLoss(nn.Module):
    def __init__(self, num_proxies: int = 512, dim: int = 256, c: float = 1.0,
                 margin: float = 0.1, tau: float = 0.1, seed: int = 22,
                 clip_r: float = 2.3):
        super().__init__()
        if c != 1 or num_proxies < 3 or dim < 1:
            raise ValueError("V6 requires c=1, at least three proxies, positive D")
        self.num_proxies, self.dim, self.c = num_proxies, dim, c
        self.margin, self.tau, self.clip_r = margin, tau, clip_r
        generator = torch.Generator(device="cpu").manual_seed(seed)
        values = torch.randn(num_proxies, dim, generator=generator) / math.sqrt(dim) * clip_r * 0.9
        self.tangent_proxies = nn.Parameter(values)

    def proxies(self) -> torch.Tensor:
        return expmap0_c1(self.tangent_proxies)

    def sample_loss(self, mu: torch.Tensor, triplets: torch.Tensor,
                    generator: torch.Generator | None = None,
                    return_details: bool = False) -> tuple:
        if len(triplets) and bool((triplets[:, 0] == triplets[:, 2]).any()):
            raise ValueError("V6 excludes sample self-k triplets")
        return ghhc_loss(poincare_distance(mu, self.proxies()), triplets,
                         tau=self.tau, margin=self.margin, generator=generator,
                         return_details=return_details)

    def proxy_loss(self, topk: int = 8, t_per_anchor: int = 50, seed: int = 23,
                   exclude_self_negative: bool = True,
                   generator: torch.Generator | None = None,
                   return_details: bool = False) -> tuple:
        if not exclude_self_negative:
            raise ValueError("V6 requires proxy self-k exclusion")
        proxies = self.proxies()
        distances = poincare_distance(proxies)
        mined = mine_reciprocal_triplets(torch.exp(-distances.detach()), topk,
                                        t_per_anchor, seed, exclude_self_negative)
        loss, stats, details = ghhc_loss(distances, mined["triplets"], self.tau,
                                         self.margin, generator, return_details=True)
        if mined["stats"]["self_k_triplets"]:
            raise RuntimeError("V6 proxy mining produced self-k")
        if return_details:
            return loss, {**mined["stats"], **stats}, {
                "triplets": mined["triplets"], "details": details, "points": proxies}
        return loss, {**mined["stats"], **stats}

    def forward(self, mu: torch.Tensor, labels: torch.Tensor, topk: int = 8,
                proxy_topk: int | None = None, seed: int = 22,
                data_ids: torch.Tensor | None = None, exclude_self_negative: bool = True,
                t_per_anchor: int = 50, generator: torch.Generator | None = None,
                return_mining: bool = False, return_components: bool = False,
                monitor_structure: bool = True) -> tuple[torch.Tensor, dict]:
        if not exclude_self_negative:
            raise ValueError("V6 requires sample and proxy self-k exclusion")
        if mu.ndim != 2 or mu.shape[1] != self.dim:
            raise ValueError(f"mu must be [N,{self.dim}]")
        sample = mine_sample_triplets(mu, labels, topk=topk, t_per_anchor=t_per_anchor,
                                     seed=seed, exclude_self_negative=exclude_self_negative,
                                     data_ids=data_ids)
        sample_loss, sample_stats, sample_details = self.sample_loss(
            mu, sample["triplets"], generator, return_details=True)
        proxy_loss, proxy_stats, proxy_details = self.proxy_loss(topk if proxy_topk is None else proxy_topk,
                                                t_per_anchor, seed + 1,
                                                exclude_self_negative, generator, return_details=True)
        with torch.no_grad():
            tangent_norm = self.tangent_proxies.norm(dim=-1)
            raw_radius = torch.tanh(tangent_norm)
            geometry = {
                "whole": shadow_geometry(mu.detach(), self.clip_r),
                "proxy": shadow_geometry(self.proxies().detach(), self.clip_r),
                "proxy_tangent_shadow_clip_count": int((tangent_norm > self.clip_r).sum()),
                "proxy_tangent_shadow_clip_fraction": float((tangent_norm > self.clip_r).float().mean()),
                "proxy_numerical_ball_project_count": int((raw_radius > 0.999).sum()),
                "proxy_numerical_ball_project_fraction": float((raw_radius > 0.999).float().mean()),
            }
        stats = {
            "sample": {**sample["stats"], **sample_stats}, "proxy": proxy_stats,
            "sample_loss": float(sample_loss.detach()), "proxy_loss": float(proxy_loss.detach()),
            "geometry": geometry,
        }
        if monitor_structure:
            stats["structure"] = {
                "sample": ancestor_summary(mu, proxy_details["points"], sample["triplets"], sample_details),
                "proxy": ancestor_summary(proxy_details["points"], proxy_details["points"],
                                           proxy_details["triplets"], proxy_details["details"])}
        if return_components:
            stats["_components"] = {"sample_loss": sample_loss, "proxy_loss": proxy_loss,
                                    "sample_details": sample_details,
                                    "proxy_details": proxy_details["details"]}
        if return_mining:
            stats["_sample_mining"] = sample
        return sample_loss + proxy_loss, stats
