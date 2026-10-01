"""Detached HIER relation selection for a single unified inter objective.

Class priority reproduces the ordering of HIER's ``exp(-distance) + same``
scores without floating-point ties at large distances. Self is excluded
before top-k. The negative set is the complement of *mutual* neighbours,
not the complement of the directed top-k. All draws independently resample
both j and k with replacement; indices within one triple remain distinct.
"""

from __future__ import annotations

import random

import torch


@torch.no_grad()
def pairwise_poincare_distance_c1(mu: torch.Tensor) -> torch.Tensor:
    """Curvature-one distances used only for discrete relation selection."""
    if mu.ndim != 2 or not mu.is_floating_point():
        raise ValueError("mu must be a floating [batch, dimension] tensor")
    z = mu.detach()
    if not bool(torch.isfinite(z).all()):
        raise ValueError("mu contains NaN or Inf")
    norm_sq = z.square().sum(-1)
    if not bool((norm_sq < 1).all()):
        raise ValueError("mu must be strictly inside the curvature-one ball")
    denominator = (1 - norm_sq).clamp_min(torch.finfo(z.dtype).eps)
    delta_sq = (z[:, None, :] - z[None, :, :]).square().sum(-1)
    distances = 2 * torch.asinh(torch.sqrt(
        (delta_sq / (denominator[:, None] * denominator[None, :])).clamp_min(0)))
    distances.fill_diagonal_(0)
    return distances


def mine_batch_triplets(
    mu: torch.Tensor,
    labels: torch.Tensor,
    k: int = 3,
    t_per_anchor: int = 32,
    seed: int = 0,
    min_mutual_positives: int = 2,
) -> dict:
    """Draw mixed within/cross-class HIER triples from the same live batch.

    The returned index tensor selects live embeddings in the loss; no graph
    is retained by ranking. The default eligibility threshold (>1 mutual
    positives) matches the released HIER implementation. Excluding self is
    a deliberate correctness fix relative to that implementation.

    ``unique_triplet_fraction`` measures repetition among actual draws.
    ``candidate_triplet_coverage`` measures the distinct draws divided by
    the complete eligible (i,j,k) candidate space, a different denominator.
    """
    for name, value in (("k", k), ("t_per_anchor", t_per_anchor),
                        ("min_mutual_positives", min_mutual_positives)):
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    distances = pairwise_poincare_distance_c1(mu).cpu().tolist()
    n = len(distances)
    targets = torch.as_tensor(labels).detach().reshape(-1).cpu().tolist()
    if len(targets) != n:
        raise ValueError("labels length must match batch size")
    if n > 1 and k >= n:
        raise ValueError("k must be smaller than batch size")
    directed = []
    for i in range(n):
        candidates = [j for j in range(n) if j != i]
        candidates.sort(key=lambda j: (targets[j] != targets[i], distances[i][j], j))
        directed.append(candidates[:k])
    sets = [set(row) for row in directed]
    mutual = [[j for j in directed[i] if i in sets[j]] for i in range(n)]
    rng = random.Random(seed)
    rows = []
    candidate_total = 0
    expected_unique = 0.0
    eligible_count = 0
    eligible_same_negative = 0
    eligible_cross_negative = 0
    positives_seen = set()
    negatives_seen = set()
    eligible_positive_total = 0
    eligible_negative_total = 0
    eligible_same_positive_total = 0
    eligible_cross_positive_total = 0
    for i, positives in enumerate(mutual):
        positive_set = set(positives)
        negatives = [index for index in range(n)
                     if index != i and index not in positive_set]
        if len(positives) < min_mutual_positives or not negatives:
            continue
        eligible_count += 1
        eligible_positive_total += len(positives)
        eligible_negative_total += len(negatives)
        eligible_same_negative += sum(targets[index] == targets[i] for index in negatives)
        eligible_cross_negative += sum(targets[index] != targets[i] for index in negatives)
        eligible_same_positive_total += sum(targets[index] == targets[i] for index in positives)
        eligible_cross_positive_total += sum(targets[index] != targets[i] for index in positives)
        combinations = len(positives) * len(negatives)
        candidate_total += combinations
        expected_unique += combinations * (1 - (1 - 1 / combinations) ** t_per_anchor)
        for _ in range(t_per_anchor):
            j, negative = rng.choice(positives), rng.choice(negatives)
            rows.append((i, j, negative))
            positives_seen.add((i, j))
            negatives_seen.add((i, negative))
    triples = torch.tensor(rows, dtype=torch.long, device=mu.device).reshape(-1, 3)
    mutual_mask = torch.zeros((n, n), dtype=torch.bool, device=mu.device)
    for i, positives in enumerate(mutual):
        if positives:
            mutual_mask[i, positives] = True
    count = len(rows)
    unique_count = len(set(rows))
    same_count = sum(targets[i] == targets[negative] for i, _, negative in rows)
    stats = {
        "batch_size": n,
        "classes": len(set(targets)),
        "k": k,
        "t_per_anchor": t_per_anchor,
        "min_mutual_positives": min_mutual_positives,
        "mutual_positive_edges": sum(map(len, mutual)),
        "anchors_with_mutual_positive": sum(bool(row) for row in mutual),
        "eligible_anchors": eligible_count,
        "anchor_coverage": eligible_count / n if n else 0.0,
        "triplets": count,
        "unique_triplets": unique_count,
        "repeated_triplets": count - unique_count,
        "unique_triplet_fraction": unique_count / count if count else 0.0,
        "candidate_triplets": candidate_total,
        "candidate_triplet_coverage": unique_count / candidate_total if candidate_total else 0.0,
        "expected_unique_triplets": expected_unique,
        "expected_candidate_coverage": expected_unique / candidate_total if candidate_total else 0.0,
        "positive_pair_coverage": len(positives_seen) / eligible_positive_total if eligible_positive_total else 0.0,
        "negative_pair_coverage": len(negatives_seen) / eligible_negative_total if eligible_negative_total else 0.0,
        "same_class_triplets": same_count,
        "cross_class_triplets": count - same_count,
        "same_class_fraction": same_count / count if count else 0.0,
        "eligible_same_negative_candidates": eligible_same_negative,
        "eligible_cross_negative_candidates": eligible_cross_negative,
        "eligible_same_positive_candidates": eligible_same_positive_total,
        "eligible_cross_positive_candidates": eligible_cross_positive_total,
    }
    return {"inter": triples, "stats": stats, "mutual": mutual_mask}
