"""Teacher-guided within-class LCA ranking objectives."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .geometry import exact_lca_depth, gromov_product_matrix


@dataclass
class RankingStats:
    triplets: int
    satisfied: int
    mean_gap: float

    @property
    def coverage(self):
        return float(self.triplets > 0)

    @property
    def satisfaction_rate(self) -> float:
        return self.satisfied / self.triplets if self.triplets else 0.0


def gather_teacher_similarity(
    teacher_features: torch.Tensor,
    batch_ids: torch.Tensor,
) -> torch.Tensor:
    """Cosine matrix aligned to *batch order*, never sorted global IDs."""
    ids = batch_ids.detach().to(device="cpu", dtype=torch.long)
    features = F.normalize(teacher_features[ids], dim=-1)
    return features @ features.T


def _zero_like(z: torch.Tensor) -> torch.Tensor:
    # Retain a valid zero-gradient path so callers may always call backward.
    return z.sum() * 0.0


def lca_ranking_loss(
    leaves: torch.Tensor,
    labels: torch.Tensor,
    teacher_similarity: torch.Tensor,
    ball,
    *,
    neighbor_k: int = 3,
    margin: float = 0.05,
    min_teacher_gap: float = 0.02,
    lca_mode: str = "exact",
) -> tuple[torch.Tensor, RankingStats]:
    """Rank teacher-near pairs deeper than teacher-far pairs.

    For each anchor, positives are mutual top-k neighbours within its class.
    Negatives are drawn from the lower half of class similarities. The
    teacher-similarity gap weights each hinge term, avoiding dependence on the
    absolute scale of a particular teacher representation.
    """
    n = leaves.shape[0]
    if teacher_similarity.shape != (n, n):
        raise ValueError("teacher_similarity must align with leaves in batch order")
    if lca_mode == "exact":
        depth = exact_lca_depth(ball, leaves, leaves)
    elif lca_mode == "gromov":
        depth = gromov_product_matrix(ball, leaves, leaves)
    else:
        raise ValueError(f"unknown lca_mode: {lca_mode}")

    terms = []
    weights = []
    satisfied = 0
    gaps = []
    labels = labels.reshape(-1)
    sim = teacher_similarity.to(device=leaves.device, dtype=leaves.dtype)

    for cls in labels.unique():
        idx = torch.nonzero(labels == cls, as_tuple=False).flatten()
        m = idx.numel()
        if m < 3:
            continue
        local = sim[idx][:, idx].clone()
        local.fill_diagonal_(-torch.inf)
        k = min(neighbor_k, m - 2)
        top = local.topk(k, dim=1).indices
        mutual = torch.zeros((m, m), dtype=torch.bool, device=leaves.device)
        rows = torch.arange(m, device=leaves.device)[:, None].expand_as(top)
        mutual[rows, top] = True
        mutual = mutual & mutual.T

        for a in range(m):
            positives = torch.nonzero(mutual[a], as_tuple=False).flatten()
            if positives.numel() == 0:
                continue
            finite_vals = local[a][torch.isfinite(local[a])]
            cutoff = finite_vals.median()
            negatives = torch.nonzero(local[a] <= cutoff, as_tuple=False).flatten()
            negatives = negatives[negatives != a]
            if negatives.numel() == 0:
                continue
            for p in positives:
                # Deterministic hardest far sample in this batch.
                neg_scores = local[a, negatives]
                q = negatives[neg_scores.argmax()]
                teacher_gap = local[a, p] - local[a, q]
                if not torch.isfinite(teacher_gap) or teacher_gap <= min_teacher_gap:
                    continue
                i, j, kidx = idx[a], idx[p], idx[q]
                student_gap = depth[i, j] - depth[i, kidx]
                weight = teacher_gap.detach().clamp(max=1.0)
                terms.append(weight * F.relu(margin - student_gap))
                weights.append(weight)
                satisfied += int((student_gap.detach() > 0).item())
                gaps.append(float(student_gap.detach().item()))

    if not terms:
        return _zero_like(leaves), RankingStats(0, 0, 0.0)
    # Normalize confidence weights so the objective scale stays comparable
    # across teachers while higher-confidence relations still count more.
    loss = torch.stack(terms).sum() / torch.stack(weights).sum().clamp_min(1e-12)
    return loss, RankingStats(
        len(terms), satisfied, sum(gaps) / len(gaps)
    )
