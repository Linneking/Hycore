"""Released HIER relation selection, with explicit diagnostic denominators.

K includes self. Same-label similarities receive +1. An anchor needs at
least two nonself mutual neighbours. j and k are drawn independently with
replacement; the source negative pool includes i itself. Empty pools are
handled safely, a robustness correction to the released concatenation.
"""

from __future__ import annotations

import numpy as np
import torch


def poincare_distance(x: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
    """Live pairwise d_1, including finite gradients for coincident points."""
    y = x if y is None else y
    if x.ndim != 2 or y.ndim != 2 or x.shape[1] != y.shape[1]:
        raise ValueError("points must be [N,D] and [M,D]")
    if not x.is_floating_point() or not y.is_floating_point():
        raise ValueError("points must be floating point")
    for z in (x, y):
        if not bool(torch.isfinite(z).all()) or not bool((z.square().sum(-1) < 1).all()):
            raise ValueError("points must be finite and strictly inside the c=1 ball")
    denominator = torch.sqrt(
        (1 - x.square().sum(-1))[:, None] * (1 - y.square().sum(-1))[None, :]
    )
    # cdist has a defined zero derivative at identical rows. sqrt of a
    # manually formed squared distance produces NaN gradients there.
    euclidean = torch.cdist(x, y, compute_mode="donot_use_mm_for_euclid_dist")
    return 2 * torch.asinh(euclidean / denominator)


@torch.no_grad()
def mine_reciprocal_triplets(
    similarity: torch.Tensor,
    topk: int = 8,
    t_per_anchor: int = 50,
    seed: int = 22,
    exclude_self_negative: bool = False,
    labels: torch.Tensor | None = None,
    data_ids: torch.Tensor | None = None,
) -> dict:
    """Return ``triplets[T,3]``, ``mutual[N,N]`` and coverage ``stats``.

    Coverage of positive/negative pairs is directed and is conditioned on
    eligible anchors. Candidate-triple coverage has the full eligible
    Cartesian-product denominator; repeat rate instead uses actual draws.
    Position coverage and unique data-ID coverage have separate fields.
    A local NumPy RandomState reproduces source choice order without
    consuming a caller's global RNG or depending on the CUDA rank.
    """
    if similarity.ndim != 2 or similarity.shape[0] != similarity.shape[1]:
        raise ValueError("similarity must be square")
    n = similarity.shape[0]
    if not 1 <= topk <= n or t_per_anchor < 1:
        raise ValueError("require 1 <= topk <= N and t_per_anchor >= 1")
    if not bool(torch.isfinite(similarity).all()):
        raise ValueError("similarity contains NaN or Inf")
    targets = None if labels is None else torch.as_tensor(labels).reshape(-1).cpu().tolist()
    ids = list(range(n)) if data_ids is None else torch.as_tensor(data_ids).reshape(-1).cpu().tolist()
    if len(ids) != n or (targets is not None and len(targets) != n):
        raise ValueError("label/data-ID lengths must match N")
    nearest = torch.topk(similarity.detach(), topk, dim=-1).indices
    directed = torch.zeros_like(similarity, dtype=torch.bool)
    directed.scatter_(1, nearest, True)
    mutual = directed & directed.T
    mutual.fill_diagonal_(False)
    negative = ~mutual
    if exclude_self_negative:
        negative.fill_diagonal_(False)
    eligible = (mutual.sum(-1) >= 2) & negative.any(-1)
    rng = np.random.RandomState(seed)
    rows = []
    positive_candidates, negative_candidates = set(), set()
    candidate_triples, expected_unique_triples = 0, 0.0
    eligible_positions = eligible.nonzero(as_tuple=False).flatten().cpu().tolist()
    for i in eligible_positions:
        positives = mutual[i].nonzero(as_tuple=False).flatten().cpu().numpy()
        negatives = negative[i].nonzero(as_tuple=False).flatten().cpu().numpy()
        js = rng.choice(positives, t_per_anchor, replace=True)
        ks = rng.choice(negatives, t_per_anchor, replace=True)
        rows.extend((i, int(j), int(k)) for j, k in zip(js, ks))
        positive_candidates.update((i, int(j)) for j in positives)
        negative_candidates.update((i, int(k)) for k in negatives)
        combinations = len(positives) * len(negatives)
        candidate_triples += combinations
        expected_unique_triples += combinations * (1 - (1 - 1 / combinations) ** t_per_anchor)
    triples = torch.tensor(rows, dtype=torch.long, device=similarity.device).reshape(-1, 3)
    unique_rows = set(rows)
    positive_seen = {(i, j) for i, j, _ in rows}
    negative_seen = {(i, k) for i, _, k in rows}
    positions_seen = {position for row in rows for position in row}
    all_ids = set(ids)
    role_ids = {ids[position] for position in positions_seen}
    unique_id_triples = {(ids[i], ids[j], ids[k]) for i, j, k in rows}
    count = len(rows)
    self_k = sum(i == k for i, _, k in rows)
    degree = mutual.sum(-1).cpu().tolist()
    stats = {
        "batch_size": n, "topk_including_self": topk, "t_per_anchor": t_per_anchor,
        "exclude_self_negative": bool(exclude_self_negative),
        "directed_self_neighbours": int(directed.diagonal().sum()),
        "eligible_anchors": len(eligible_positions),
        "anchor_coverage": len(eligible_positions) / n,
        "eligible_anchor_data_ids": len({ids[i] for i in eligible_positions}),
        "anchor_data_id_coverage": len({ids[i] for i in eligible_positions}) / len(all_ids),
        "mutual_positive_edges": int(mutual.sum()),
        "mutual_degree_min": min(degree), "mutual_degree_max": max(degree),
        "mutual_degree_mean": sum(degree) / n,
        "mutual_degree_histogram": {str(d): degree.count(d) for d in sorted(set(degree))},
        "anchors_with_zero_positive": sum(d == 0 for d in degree),
        "anchors_with_one_positive": sum(d == 1 for d in degree),
        "triplets": count, "unique_triplets": len(unique_rows),
        "repeated_triplets": count - len(unique_rows),
        "repeat_fraction": (count - len(unique_rows)) / count if count else 0.0,
        "candidate_triplets": candidate_triples,
        "candidate_triplet_coverage": len(unique_rows) / candidate_triples if candidate_triples else 0.0,
        "expected_unique_triplets": expected_unique_triples,
        "candidate_positive_pairs": len(positive_candidates),
        "candidate_negative_pairs": len(negative_candidates),
        "unique_positive_pairs": len(positive_seen), "unique_negative_pairs": len(negative_seen),
        "positive_pair_coverage": len(positive_seen) / len(positive_candidates) if positive_candidates else 0.0,
        "negative_pair_coverage": len(negative_seen) / len(negative_candidates) if negative_candidates else 0.0,
        "any_role_positions": len(positions_seen), "any_role_position_coverage": len(positions_seen) / n,
        "batch_unique_data_ids": len(all_ids), "any_role_unique_data_ids": len(role_ids),
        "any_role_data_id_coverage": len(role_ids) / len(all_ids),
        "unique_data_id_triples": len(unique_id_triples),
        "duplicate_data_id_draws": count - len(unique_id_triples),
        "self_k_triplets": self_k, "self_k_fraction": self_k / count if count else 0.0,
        "equal_data_id_ik_triplets": sum(ids[i] == ids[k] for i, _, k in rows),
        "equal_data_id_ij_triplets": sum(ids[i] == ids[j] for i, j, _ in rows),
    }
    if targets is not None:
        stats.update({
            "classes": len(set(targets)),
            "same_class_j_triplets": sum(targets[i] == targets[j] for i, j, _ in rows),
            "same_class_k_triplets": sum(targets[i] == targets[k] for i, _, k in rows),
            "all_same_class_triplets": sum(targets[i] == targets[j] == targets[k] for i, j, k in rows),
            "same_class_nonself_k_triplets": sum(i != k and targets[i] == targets[k] for i, _, k in rows),
            "same_class_positive_candidates": sum(targets[i] == targets[j] for i, j in positive_candidates),
            "cross_class_positive_candidates": sum(targets[i] != targets[j] for i, j in positive_candidates),
            "same_class_negative_candidates": sum(targets[i] == targets[k] for i, k in negative_candidates),
            "cross_class_negative_candidates": sum(targets[i] != targets[k] for i, k in negative_candidates),
        })
    return {"triplets": triples, "mutual": mutual, "negative": negative, "eligible": eligible, "stats": stats}


@torch.no_grad()
def mine_sample_triplets(mu: torch.Tensor, labels: torch.Tensor, **kwargs) -> dict:
    labels = torch.as_tensor(labels, device=mu.device).reshape(-1)
    similarity = torch.exp(-poincare_distance(mu.detach()))
    similarity[labels[:, None] == labels[None, :]] += 1
    return mine_reciprocal_triplets(similarity, labels=labels, **kwargs)
