"""Read-only HIER triplet mining and random-proxy diagnostics.

This module does not update model weights.  The proxy probe deliberately uses
randomly initialized proxies; its loss statistics are *not* a trained result.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class GridConfig:
    name: str
    mode: str
    sample_k: int
    proxy_k: int
    label_bonus: float = 0.0


DEFAULT_GRID = (
    GridConfig("H01_bonus1_k04", "hier", 4, 4, 1.0),
    GridConfig("H02_bonus1_k08", "hier", 8, 8, 1.0),
    GridConfig("H03_bonus1_k12", "hier", 12, 12, 1.0),
    GridConfig("H04_bonus1_k20", "hier", 20, 20, 1.0),
    GridConfig("H05_bonus1_k30", "hier", 30, 30, 1.0),
    GridConfig("H06_bonus0_k08", "hier", 8, 8, 0.0),
    GridConfig("H07_bonus0_k20", "hier", 20, 20, 0.0),
    GridConfig("C01_within_k02", "within", 2, 20),
    GridConfig("C02_within_k04", "within", 4, 20),
)


def poincare_dist_matrix(x: torch.Tensor, y: torch.Tensor | None = None, c: float = 1.0) -> torch.Tensor:
    """Pairwise hyperbolic distances for points inside a curvature-c ball."""
    if c <= 0:
        raise ValueError("curvature magnitude c must be positive")
    same_tensor = y is None or y.data_ptr() == x.data_ptr()
    if y is None:
        y = x
    if x.ndim != 2 or y.ndim != 2 or x.shape[1] != y.shape[1]:
        raise ValueError("expected matrices with the same feature dimension")
    x2 = x.square().sum(dim=1)
    y2 = y.square().sum(dim=1)
    if torch.any(c * x2 >= 1) or torch.any(c * y2 >= 1):
        raise ValueError("embedding lies outside the Poincare ball")
    sq = (x2[:, None] + y2[None, :] - 2 * (x @ y.T)).clamp_min(0)
    denominator = ((1 - c * x2)[:, None] * (1 - c * y2)[None, :]).clamp_min(1e-12)
    argument = (1 + 2 * c * sq / denominator).clamp_min(1)
    distances = torch.acosh(argument) / math.sqrt(c)
    if same_tensor:
        distances.fill_diagonal_(0)
    return distances


def random_proxy_bank(count: int, dimension: int, seed: int, device: torch.device,
                      c: float = 1.0, clip_r: float = 2.3) -> torch.Tensor:
    """HIER-style random tangent initialization, mapped at the origin."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    tangent = torch.randn(count, dimension, generator=generator, dtype=torch.float32)
    tangent = tangent.to(device) * (clip_r * 0.9 / math.sqrt(dimension))
    norm = tangent.norm(dim=1, keepdim=True).clamp_min(1e-12)
    return torch.tanh(math.sqrt(c) * norm) * tangent / (math.sqrt(c) * norm)


def balanced_batches(labels: np.ndarray, seed: int, count: int,
                     classes_per_batch: int = 5, samples_per_class: int = 8) -> list[np.ndarray]:
    """Fixed, class-balanced 5x8 batches; samples may recur across batches."""
    labels = np.asarray(labels).reshape(-1)
    by_class = {int(cls): np.flatnonzero(labels == cls) for cls in np.unique(labels)}
    if len(by_class) < classes_per_batch:
        raise ValueError("not enough classes")
    if any(len(pool) < samples_per_class for pool in by_class.values()):
        raise ValueError("a class has fewer samples than requested")
    generator = np.random.default_rng(seed)
    classes = np.asarray(sorted(by_class))
    result = []
    for _ in range(count):
        selected = generator.choice(classes, classes_per_batch, replace=False)
        batch = np.concatenate([generator.choice(by_class[int(cls)], samples_per_class,
                                                 replace=False) for cls in selected])
        generator.shuffle(batch)
        result.append(batch.astype(np.int64))
    return result


def mine_pools(distances: torch.Tensor, labels: np.ndarray, config: GridConfig) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Construct HIER mutual-kNN pools or explicit within-class comparison pools.

    Faithful HIER includes self in top-K, then removes it from positives.  Its
    negative pool includes self, matching the released code for measurement.
    """
    n = len(distances)
    labels = np.asarray(labels).reshape(-1)
    if distances.shape != (n, n) or len(labels) != n:
        raise ValueError("distance/label shape mismatch")
    if config.sample_k <= 0 or config.sample_k > n:
        raise ValueError("sample_k outside batch size")
    if config.mode == "hier":
        same = torch.as_tensor(labels[:, None] == labels[None, :], device=distances.device)
        score = torch.exp(-distances) + config.label_bonus * same
        nearest = torch.topk(score, config.sample_k, dim=1).indices
        adjacency = torch.zeros((n, n), dtype=torch.bool, device=distances.device)
        adjacency.scatter_(1, nearest, True)
        mutual = (adjacency & adjacency.T).cpu().numpy()
        np.fill_diagonal(mutual, False)
        positives = [np.flatnonzero(mutual[i]) for i in range(n)]
        negatives = [np.flatnonzero(~mutual[i]) for i in range(n)]
    elif config.mode == "within":
        array = distances.detach().cpu().numpy()
        adjacency = np.zeros((n, n), dtype=bool)
        for i in range(n):
            eligible = np.flatnonzero((labels == labels[i]) & (np.arange(n) != i))
            nearest = eligible[np.argsort(array[i, eligible], kind="stable")[:config.sample_k]]
            adjacency[i, nearest] = True
        mutual = adjacency & adjacency.T
        positives = [np.flatnonzero(mutual[i]) for i in range(n)]
        negatives = [np.flatnonzero((labels == labels[i]) & ~mutual[i] &
                                    (np.arange(n) != i)) for i in range(n)]
    else:
        raise ValueError(f"unknown mining mode: {config.mode}")
    return positives, negatives


def draw_triplets(positives: list[np.ndarray], negatives: list[np.ndarray],
                  seed: int, per_anchor: int = 50) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample with replacement, retaining HIER's >1-positive anchor guard."""
    generator = np.random.default_rng(seed)
    anchors, nearby, distant = [], [], []
    for i, (pos, neg) in enumerate(zip(positives, negatives)):
        if len(pos) <= 1 or len(neg) == 0:
            continue
        anchors.append(np.full(per_anchor, i, dtype=np.int64))
        nearby.append(generator.choice(pos, per_anchor, replace=True))
        distant.append(generator.choice(neg, per_anchor, replace=True))
    if not anchors:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty, empty
    return np.concatenate(anchors), np.concatenate(nearby), np.concatenate(distant)


def mining_counts(distances: torch.Tensor, labels: np.ndarray,
                  positives: list[np.ndarray], negatives: list[np.ndarray],
                  triplets: tuple[np.ndarray, np.ndarray, np.ndarray]) -> dict[str, float]:
    """Report both exact candidate-pool expectations and sampled outcomes."""
    labels = np.asarray(labels).reshape(-1)
    i, j, k = triplets
    eligible = [a for a, (pos, neg) in enumerate(zip(positives, negatives))
                if len(pos) > 1 and len(neg) > 0]
    result: dict[str, float] = {
        "anchors": len(labels), "eligible_anchors": len(eligible), "triplets": len(i),
        "expected_j_same_sum": 0.0, "expected_k_same_other_sum": 0.0,
        "expected_k_cross_sum": 0.0, "expected_k_self_sum": 0.0,
        "positive_pool_sum": 0.0, "negative_pool_sum": 0.0,
        "j_same": 0, "j_cross": 0, "k_same_other": 0, "k_cross": 0,
        "k_self": 0, "ordered": 0, "unique_triplets": 0,
    }
    for a in eligible:
        pos, neg = positives[a], negatives[a]
        result["expected_j_same_sum"] += float(np.mean(labels[pos] == labels[a]))
        result["expected_k_same_other_sum"] += float(np.mean((labels[neg] == labels[a]) & (neg != a)))
        result["expected_k_cross_sum"] += float(np.mean(labels[neg] != labels[a]))
        result["expected_k_self_sum"] += float(np.mean(neg == a))
        result["positive_pool_sum"] += len(pos)
        result["negative_pool_sum"] += len(neg)
    if len(i):
        j_same = labels[i] == labels[j]
        k_same = labels[i] == labels[k]
        k_self = i == k
        result.update({
            "j_same": int(j_same.sum()), "j_cross": int((~j_same).sum()),
            "k_same_other": int((k_same & ~k_self).sum()),
            "k_cross": int((~k_same).sum()), "k_self": int(k_self.sum()),
            "unique_triplets": len(set(zip(i.tolist(), j.tolist(), k.tolist()))),
        })
        matrix = distances.detach().cpu().numpy()
        result["ordered"] = int((matrix[i, j] < matrix[i, k]).sum())
    return result


def _gumbel_argmax(logits: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    uniform = torch.rand(logits.shape, device=logits.device, generator=generator)
    uniform = uniform.clamp(1e-7, 1 - 1e-7)
    return (logits - torch.log(-torch.log(uniform))).argmax(dim=1)


def proxy_probe(cp_distance: torch.Tensor, triplets: tuple[np.ndarray, np.ndarray, np.ndarray],
                seed: int, tau: float = 0.1, margin: float = 0.1,
                proxy_nodes: bool = False, chunk_size: int = 2048) -> dict[str, float]:
    """Evaluate HIER's hard proxy choices and hinges without backpropagation."""
    i, j, k = triplets
    n = len(i)
    result = {"triplets": n, "collisions": 0, "pre_active": 0,
              "effective": 0, "hinge_1_sum": 0.0, "hinge_2_sum": 0.0,
              "hinge_3_sum": 0.0, "masked_loss_sum": 0.0,
              "pair_endpoint": 0, "triple_endpoint": 0}
    if n == 0:
        return result
    generator = torch.Generator(device=cp_distance.device).manual_seed(seed)
    for start in range(0, n, chunk_size):
        stop = min(n, start + chunk_size)
        a = torch.as_tensor(i[start:stop], dtype=torch.long, device=cp_distance.device)
        b = torch.as_tensor(j[start:stop], dtype=torch.long, device=cp_distance.device)
        c = torch.as_tensor(k[start:stop], dtype=torch.long, device=cp_distance.device)
        da, db, dc = cp_distance[a], cp_distance[b], cp_distance[c]
        pair_max = torch.maximum(da, db)
        pair_proxy = _gumbel_argmax(-pair_max / tau, generator)
        triple_proxy = _gumbel_argmax(-torch.maximum(pair_max, dc) / tau, generator)
        row = torch.arange(len(a), device=cp_distance.device)
        first = (da[row, pair_proxy] - da[row, triple_proxy] + margin).relu()
        second = (db[row, pair_proxy] - db[row, triple_proxy] + margin).relu()
        third = (dc[row, triple_proxy] - dc[row, pair_proxy] + margin).relu()
        total = first + second + third
        collision = pair_proxy == triple_proxy
        result["collisions"] += int(collision.sum().item())
        result["pre_active"] += int((total > 0).sum().item())
        result["effective"] += int(((total > 0) & ~collision).sum().item())
        result["hinge_1_sum"] += float(first.sum().item())
        result["hinge_2_sum"] += float(second.sum().item())
        result["hinge_3_sum"] += float(third.sum().item())
        result["masked_loss_sum"] += float((total * (~collision)).sum().item())
        if proxy_nodes:
            result["pair_endpoint"] += int(((pair_proxy == a) | (pair_proxy == b)).sum().item())
            result["triple_endpoint"] += int(((triple_proxy == a) |
                                               (triple_proxy == b) |
                                               (triple_proxy == c)).sum().item())
    return result
