"""Mine HIER triplets from the current student embeddings in a balanced batch.

Neighbour ranking is discrete and uses ``mu.detach()``. The returned indices
select the live ``mu`` in the loss, so gradients still reach the student.
Only class labels and current hyperbolic distances determine candidates.
"""

from __future__ import annotations

import random

import torch


def pairwise_poincare_distance_c1(mu: torch.Tensor) -> torch.Tensor:
    """Pairwise distances inside the curvature-one Poincare ball, detached."""
    if mu.ndim != 2:
        raise ValueError("mu must have shape [batch, dimension]")
    with torch.no_grad():
        z = mu.detach()
        if not bool(torch.isfinite(z).all()):
            raise ValueError("mu contains NaN or Inf")
        norm_sq = z.square().sum(dim=-1)
        if not bool((norm_sq < 1).all()):
            raise ValueError("mu must lie strictly inside the unit Poincare ball")
        delta_sq = (z[:, None, :] - z[None, :, :]).square().sum(dim=-1)
        denom = (1 - norm_sq).clamp_min(torch.finfo(z.dtype).eps)
        ratio = delta_sq / (denom[:, None] * denom[None, :])
        distance = 2 * torch.asinh(torch.sqrt(ratio.clamp_min(0)))
        distance.fill_diagonal_(0)
        return distance


def mine_batch_triplets(
    mu: torch.Tensor,
    labels: torch.Tensor,
    k_in: int = 3,
    seed: int = 0,
    hard_ratio: float = 0.5,
) -> dict:
    """Choose at most one within-class and cross-class triplet per anchor.

    ``j`` is sampled from mutual same-class top-``k_in`` neighbours. The
    within-class ``k`` is sampled outside the anchor's directed top-k. For
    about ``hard_ratio`` of eligible cross-class anchors, ``k`` is the nearest
    other-class sample; it is uniform among other-class samples otherwise.
    A fixed seed makes a fixed embedding deterministic. It does not freeze
    relationships when the student changes their distance ranking.
    """
    if not isinstance(k_in, int) or k_in < 1:
        raise ValueError("k_in must be a positive integer")
    if not 0 <= hard_ratio <= 1:
        raise ValueError("hard_ratio must be in [0, 1]")
    if mu.ndim != 2:
        raise ValueError("mu must have shape [batch, dimension]")
    labels_flat = torch.as_tensor(labels).reshape(-1)
    n = mu.shape[0]
    if labels_flat.numel() != n:
        raise ValueError("labels length must match batch size")
    targets = [int(v) for v in labels_flat.detach().cpu().tolist()]
    empty = torch.empty((0, 3), dtype=torch.long, device=mu.device)
    if n == 0:
        return {
            "in": empty,
            "out": empty.clone(),
            "stats": {
                "batch_size": 0, "classes": 0, "mutual_positive_edges": 0,
                "anchors_with_mutual_positive": 0,
                "eligible_in_anchors": 0, "eligible_out_anchors": 0,
                "in_triplets": 0, "out_triplets": 0,
                "in_anchor_coverage": 0.0, "out_anchor_coverage": 0.0,
                "out_hard_count": 0, "out_random_count": 0,
            },
        }

    rng = random.Random(seed)
    distances = pairwise_poincare_distance_c1(mu).cpu().tolist()
    class_to_indices: dict[int, list[int]] = {}
    for index, label in enumerate(targets):
        class_to_indices.setdefault(label, []).append(index)

    directed: list[list[int]] = []
    for i, label in enumerate(targets):
        same = [j for j in class_to_indices[label] if j != i]
        same.sort(key=lambda j: (distances[i][j], j))
        directed.append(same[:k_in])
    directed_sets = [set(neighbours) for neighbours in directed]

    in_triplets: list[tuple[int, int, int]] = []
    out_candidates: list[tuple[int, int, list[int]]] = []
    mutual_edge_count = 0
    anchors_with_mutual_positive = 0
    for i, label in enumerate(targets):
        positives = [j for j in directed[i] if i in directed_sets[j]]
        mutual_edge_count += len(positives)
        if not positives:
            continue
        anchors_with_mutual_positive += 1
        j = rng.choice(positives)
        same_outside_topk = [
            k for k in class_to_indices[label]
            if k != i and k not in directed_sets[i]
        ]
        if same_outside_topk:
            in_triplets.append((i, j, rng.choice(same_outside_topk)))
        different_class = [k for k in range(n) if targets[k] != label]
        if different_class:
            out_candidates.append((i, j, different_class))

    hard_count = min(len(out_candidates), int(len(out_candidates) * hard_ratio + 0.5))
    hard_positions = set(rng.sample(range(len(out_candidates)), hard_count))
    out_triplets: list[tuple[int, int, int]] = []
    for position, (i, j, candidates) in enumerate(out_candidates):
        if position in hard_positions:
            k = min(candidates, key=lambda index: (distances[i][index], index))
        else:
            k = rng.choice(candidates)
        out_triplets.append((i, j, k))

    def as_tensor(rows: list[tuple[int, int, int]]) -> torch.Tensor:
        return torch.tensor(rows, dtype=torch.long, device=mu.device).reshape(-1, 3)

    stats = {
        "batch_size": n,
        "classes": len(class_to_indices),
        "mutual_positive_edges": mutual_edge_count,
        "anchors_with_mutual_positive": anchors_with_mutual_positive,
        "eligible_in_anchors": len(in_triplets),
        "eligible_out_anchors": len(out_triplets),
        "in_triplets": len(in_triplets),
        "out_triplets": len(out_triplets),
        "in_anchor_coverage": len(in_triplets) / n,
        "out_anchor_coverage": len(out_triplets) / n,
        "out_hard_count": hard_count,
        "out_random_count": len(out_triplets) - hard_count,
    }
    return {"in": as_tensor(in_triplets), "out": as_tensor(out_triplets), "stats": stats}
