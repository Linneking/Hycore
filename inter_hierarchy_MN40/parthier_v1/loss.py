"""PartHier v1: native c=1 parts as shared, live ancestor candidates.

No new geometry map, inverse-metric hook, proxy parameters, or optimizer.
Whole triplets use the V5 reciprocal miner; ancestor selection retains V6
independent hard Gumbel draws and its collision-masked all-draw mean.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from ..hier_proxy_scratch_v5.relations import mine_sample_triplets
from ..hier_proxy_scratch_v6.hier_loss import _hard_gumbel
from ..hier_proxy_scratch_v8.geometry import official_distance
from .feature_queue import PartFeatureQueue


def origin_depth(points):
    """Native c=1 origin distance; no extra cap or gradient hook."""
    return 2 * torch.atanh(points.norm(dim=-1))


def _selection_losses(distances, triplets, whole_depth, candidate_depth, *,
                      tau=.1, margin=.1, radial_margin=.2, generator=None):
    """V6 GHHC plus radial order using the same two ST hard selections.

    Keeping the depth selection as prob @ depth retains the selection
    path, while its forward value is the actual hard candidate's depth.
    This surrogate weights candidate depths, not a synthetic soft point.
    The proposal's two order terms are retained; B<k is diagnostic only.
    """
    triplets = torch.as_tensor(triplets, device=distances.device, dtype=torch.long).reshape(-1, 3)
    count = len(triplets)
    if not count:
        zero = distances.sum() * 0 + whole_depth.sum() * 0 + candidate_depth.sum() * 0
        indices = triplets[:, 0]
        details = {"pair_idx": indices, "triple_idx": indices,
                   "valid": indices.bool(), "hier_raw": candidate_depth[:0],
                   "hier_per_draw": candidate_depth[:0],
                   "order_hinges": distances.new_empty((0, 2)),
                   "order_per_draw": candidate_depth[:0]}
        return zero, zero, details
    if bool(((triplets < 0) | (triplets >= len(distances))).any()):
        raise ValueError("triplet index outside whole rows")
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
    depth_a, depth_b = pair_prob @ candidate_depth, triple_prob @ candidate_depth
    order_hinges = torch.stack((
        F.relu(depth_a - torch.minimum(whole_depth[i], whole_depth[j]) + radial_margin),
        F.relu(depth_b - depth_a + radial_margin),
    ), dim=-1)
    order_per_draw = order_hinges.sum(-1) * valid.to(order_hinges.dtype)
    details = {"pair_idx": pair_idx, "triple_idx": triple_idx, "valid": valid,
               "hier_raw": raw, "hier_per_draw": per_draw,
               "order_hinges": order_hinges, "order_per_draw": order_per_draw}
    return per_draw.mean(), order_per_draw.mean(), details


def _depth_summary(values):
    return {"count": int(values.numel()), "min": float(values.min()) if values.numel() else 0.,
            "mean": float(values.mean()) if values.numel() else 0.,
            "max": float(values.max()) if values.numel() else 0.}


class PartHierarchyLoss(nn.Module):
    def __init__(self, queue_size=2048, lambda_hier=.5, lambda_order=.1,
                 radial_margin=.2, grain_margin=.2):
        super().__init__()
        if min(lambda_hier, lambda_order, radial_margin, grain_margin) < 0:
            raise ValueError("weights and radial/grain margins must be nonnegative")
        self.queue = PartFeatureQueue(queue_size)
        self.lambda_hier, self.lambda_order = float(lambda_hier), float(lambda_order)
        self.radial_margin, self.grain_margin = float(radial_margin), float(grain_margin)
        self.topk, self.t_per_anchor, self.tau, self.margin = 20, 50, .1, .1

    def enqueue(self, parts, object_ids, step):
        """Enqueue detached features from the completed training forward."""
        self.queue.enqueue(parts, object_ids, step)

    def queue_state(self):
        return self.queue.state_dict()

    def load_queue_state(self, state):
        self.queue.load_state_dict(state)

    def forward(self, mu, parts, labels, object_ids, step: int, seed: int):
        if mu.ndim != 2 or mu.shape[1] < 2 or len(mu) < self.topk:
            raise ValueError("whole features must be [B>=20,D>=2] for K20")
        if parts.shape != (len(mu), 3, mu.shape[1]) or parts.device != mu.device or parts.dtype != mu.dtype:
            raise ValueError("parts must be aligned [B,3,D] with whole device/dtype")
        ids = PartFeatureQueue._validate(parts, object_ids, step)
        labels = torch.as_tensor(labels, device=mu.device)
        if labels.ndim != 1 or len(labels) != len(mu):
            raise ValueError("one label is required per global whole row")
        if not bool(torch.isfinite(mu).all()) or not bool((mu.square().sum(-1) < 1).all()):
            raise ValueError("whole features must be finite c=1 ball points")
        current = parts.reshape(-1, mu.shape[1])
        current_count = len(current)
        candidate_ids = ids.repeat_interleave(3)
        candidate_scales = torch.arange(3, device=mu.device).repeat(len(mu))
        candidate_steps = torch.full((current_count,), step, dtype=torch.long, device=mu.device)
        queue = self.queue.entries(device=mu.device, dtype=mu.dtype)
        if len(self.queue):
            if queue["features"].shape[1] != mu.shape[1]:
                raise ValueError("queue embedding dimension differs from current batch")
            if bool((queue["steps"] > step).any()):
                raise ValueError("queue contains future features relative to training step")
            candidates = torch.cat((current, queue["features"]), dim=0)
            candidate_ids = torch.cat((candidate_ids, queue["object_ids"]))
            candidate_scales = torch.cat((candidate_scales, queue["scales"]))
            candidate_steps = torch.cat((candidate_steps, queue["steps"]))
        else:
            candidates = current
        mined = mine_sample_triplets(mu.detach(), labels, topk=self.topk,
            t_per_anchor=self.t_per_anchor, seed=seed, exclude_self_negative=True, data_ids=ids)
        triplets = mined["triplets"]
        if len(triplets) and bool((triplets[:, 0] == triplets[:, 2]).any()):
            raise RuntimeError("PartHier must exclude sample self-k")
        distances = official_distance(mu, candidates, c=1.)
        whole_depth, part_depth = origin_depth(mu), origin_depth(parts)
        candidate_depth = origin_depth(candidates)
        generator = torch.Generator(device=mu.device).manual_seed(seed)
        hier, order, detail = _selection_losses(distances, triplets, whole_depth,
            candidate_depth, tau=self.tau, margin=self.margin,
            radial_margin=self.radial_margin, generator=generator)
        chain_hinges = torch.stack((
            F.relu(part_depth[:, 0] - part_depth[:, 1] + self.grain_margin),
            F.relu(part_depth[:, 1] - part_depth[:, 2] + self.grain_margin),
            F.relu(part_depth[:, 2] - whole_depth + self.grain_margin),
        ), dim=-1)
        chain = chain_hinges.sum(-1).mean()
        weighted = self.lambda_hier * hier + self.lambda_order * (order + chain)
        if not bool(torch.isfinite(torch.stack((hier, order, chain, weighted))).all()):
            raise FloatingPointError("nonfinite PartHier loss")
        with torch.no_grad():
            valid, a, b = detail["valid"], detail["pair_idx"], detail["triple_idx"]
            draws, effective = len(triplets), int(valid.sum())
            order_hinges = detail["order_hinges"]
            roles = {}
            for name, selected in (("pair", a), ("triple", b)):
                chosen = selected[valid]
                n = len(chosen)
                roles[name] = {
                    "noncollision_draws": n,
                    "scale_counts": [int((candidate_scales[chosen] == scale).sum()) for scale in range(3)],
                    "current_count": int((chosen < current_count).sum()),
                    "queue_count": int((chosen >= current_count).sum()),
                    "current_fraction": float((chosen < current_count).float().mean()) if n else 0.,
                    "queue_fraction": float((chosen >= current_count).float().mean()) if n else 0.,
                    "unique_candidates": int(chosen.unique().numel()),
                    "depth": _depth_summary(candidate_depth[chosen]),
                    "age": _depth_summary((step - candidate_steps[chosen]).to(mu.dtype)),
                }
            if draws:
                i, j, k = triplets.unbind(-1)
                pair_own = (candidate_ids[a] == ids[i]) | (candidate_ids[a] == ids[j])
                triple_own = pair_own.new_zeros(draws)
                for descendant in (i, j, k):
                    triple_own |= candidate_ids[b] == ids[descendant]
                pair_reverse = candidate_depth[a] >= torch.minimum(whole_depth[i], whole_depth[j])
                triple_reverse = candidate_depth[b] >= candidate_depth[a]
                third_reverse = candidate_depth[b] >= whole_depth[k]
            else:
                pair_own = triple_own = pair_reverse = triple_reverse = third_reverse = valid
            def fraction(mask):
                return float(mask[valid].float().mean()) if effective else 0.
            stats = {
                "sample": {**mined["stats"], "collisions": draws - effective,
                    "collision_rate": (draws - effective) / draws if draws else 0.,
                    "noncollision_triplets": effective,
                    "effective_fraction_all_draws": effective / draws if draws else 0.,
                    "active_triplets": int((valid & (detail["hier_raw"] > 0)).sum())},
                "hier_loss": float(hier), "order_loss": float(order), "chain_loss": float(chain),
                "weighted_inter": float(weighted), "finite": True,
                "c": 1., "extra_cap": False, "inverse_metric_hook": False,
                "current_candidates": current_count, "queue_candidates": len(self.queue),
                "candidate_count": len(candidates), "roles": roles,
                "depth": {"whole": _depth_summary(whole_depth),
                    "part_scales": [_depth_summary(part_depth[:, g]) for g in range(3)]},
                "order": {"denominator": "all_draws; collisions_zero",
                    "valid_draws": effective,
                    "all_satisfied_fraction_noncollision": fraction((order_hinges == 0).all(-1)),
                    "term_satisfied_fraction_noncollision": [fraction(order_hinges[:, term] == 0) for term in range(2)],
                    "pair_not_shallower_fraction": fraction(pair_reverse),
                    "triple_not_shallower_than_pair_fraction": fraction(triple_reverse),
                    "triple_not_shallower_than_k_fraction": fraction(third_reverse),
                    "triple_k_margin_satisfied_fraction": fraction((candidate_depth[b] + self.radial_margin <= whole_depth[triplets[:, 2]]) if draws else valid),
                    "pair_source_in_descendants_fraction": fraction(pair_own),
                    "triple_source_in_descendants_fraction": fraction(triple_own)},
                "chain": {"all_satisfied_fraction": float((chain_hinges == 0).all(-1).float().mean()),
                    "term_satisfied_fraction": [(float((chain_hinges[:, term] == 0).float().mean())) for term in range(3)]},
            }
        return weighted, stats
