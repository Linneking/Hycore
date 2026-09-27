"""CPU checks that online mining follows current student geometry."""

import math
import os
import sys

import torch
from geoopt import PoincareBall

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from hier_proxy_v2_online.mining import mine_batch_triplets, pairwise_poincare_distance_c1


def _topk(mu: torch.Tensor, labels: torch.Tensor, anchor: int, k: int = 3) -> set[int]:
    distances = pairwise_poincare_distance_c1(mu)[anchor].tolist()
    same = [i for i in range(len(labels)) if i != anchor and labels[i] == labels[anchor]]
    return set(sorted(same, key=lambda i: (distances[i], i))[:k])


def test_rank_crossing_updates_selected_triplet_with_fixed_seed():
    # For anchor 0, moving index 4 inside top-3 pushes index 3 out.
    # There are five class-0 samples, so each state has one eligible k.
    labels = torch.tensor([0] * 5 + [1] * 5)
    initial = torch.tensor([
        [0.00], [0.05], [0.10], [0.15], [0.40],
        [-0.20], [-0.30], [-0.40], [-0.50], [-0.60],
    ], requires_grad=True)
    updated = initial.detach().clone()
    updated[4, 0] = 0.12
    updated.requires_grad_(True)
    assert _topk(initial, labels, 0) == {1, 2, 3}
    assert _topk(updated, labels, 0) == {1, 2, 4}

    before = mine_batch_triplets(initial, labels, k_in=3, seed=22)
    after = mine_batch_triplets(updated, labels, k_in=3, seed=22)
    before_anchor = next(row for row in before["in"].tolist() if row[0] == 0)
    after_anchor = next(row for row in after["in"].tolist() if row[0] == 0)
    assert before_anchor[2] == 4
    assert after_anchor[2] == 3
    assert before_anchor != after_anchor
    assert torch.equal(before["in"], mine_batch_triplets(initial, labels, seed=22)["in"])
    assert torch.equal(after["in"], mine_batch_triplets(updated, labels, seed=22)["in"])
    assert not before["in"].requires_grad and not after["in"].requires_grad
    assert initial.grad is None and updated.grad is None


def test_balanced_5x8_respects_mutual_neighbours_and_class_rules():
    generator = torch.Generator().manual_seed(91)
    points = []
    labels = []
    for cls in range(5):
        angle = 2 * math.pi * cls / 5
        center = torch.tensor([0.45 * math.cos(angle), 0.45 * math.sin(angle)])
        points.append(center + 0.065 * torch.randn(8, 2, generator=generator))
        labels.extend([cls] * 8)
    mu = torch.cat(points).requires_grad_(True)
    labels = torch.tensor(labels)
    result = mine_batch_triplets(mu, labels, k_in=3, seed=22)
    same_seed = mine_batch_triplets(mu, labels, k_in=3, seed=22)
    assert torch.equal(result["in"], same_seed["in"])
    assert torch.equal(result["out"], same_seed["out"])
    assert result["stats"]["batch_size"] == 40
    assert result["stats"]["classes"] == 5
    assert result["stats"]["in_triplets"] > 0
    assert result["stats"]["out_triplets"] > 0
    assert result["stats"]["out_hard_count"] == int(len(result["out"]) * 0.5 + 0.5)

    knn = [_topk(mu, labels, anchor) for anchor in range(len(labels))]
    for branch in ("in", "out"):
        triples = result[branch]
        assert triples.dtype == torch.long and triples.shape[1] == 3
        assert not triples.requires_grad
        assert len(set(triples[:, 0].tolist())) == len(triples)
        for i, j, k in triples.tolist():
            assert len({i, j, k}) == 3
            assert labels[i] == labels[j]
            assert j in knn[i] and i in knn[j]
            if branch == "in":
                assert labels[i] == labels[k] and k not in knn[i]
            else:
                assert labels[i] != labels[k]
    assert mu.grad is None


def test_all_hard_cross_class_negatives_are_nearest():
    mu = torch.tensor([
        [0.00], [0.05], [0.10], [0.15], [0.40],
        [-0.20], [-0.30], [-0.40], [-0.50], [-0.60],
    ])
    labels = torch.tensor([0] * 5 + [1] * 5)
    result = mine_batch_triplets(mu, labels, seed=42, hard_ratio=1.0)
    distances = pairwise_poincare_distance_c1(mu).tolist()
    assert result["stats"]["out_hard_count"] == len(result["out"])
    for i, _, k in result["out"].tolist():
        other = [candidate for candidate in range(len(labels)) if labels[candidate] != labels[i]]
        assert k == min(other, key=lambda candidate: (distances[i][candidate], candidate))


def test_distance_detaches_and_rejects_boundary():
    mu = torch.tensor([[0.0, 0.0], [0.5, 0.0]], requires_grad=True)
    distance = pairwise_poincare_distance_c1(mu)
    assert not distance.requires_grad
    assert torch.allclose(distance, distance.T)
    assert torch.allclose(distance.diag(), torch.zeros(2))
    assert torch.allclose(distance[0, 1], torch.tensor(math.log(3)), atol=1e-6)
    try:
        mine_batch_triplets(torch.tensor([[1.0, 0.0]]), torch.tensor([0]))
    except ValueError as error:
        assert "ball" in str(error)
    else:
        raise AssertionError("boundary embedding should be rejected")


def test_projected_hier_branch_accepts_float32_boundary_and_keeps_gradient():
    # HyCoRe occasionally rounds a valid Mobius output to norm exactly one.
    # The HIER-only projection restores an interior point before ranking.
    raw = torch.tensor([[1.0, 0.0], [0.90, 0.02]], requires_grad=True)
    ball = PoincareBall(c=1)
    projected = ball.projx(raw)
    distances = pairwise_poincare_distance_c1(projected)
    assert torch.isfinite(distances).all()
    assert float(projected.norm(dim=-1).max()) < 1.0
    ball.dist(projected[0], projected[1]).backward()
    assert raw.grad is not None and torch.isfinite(raw.grad).all()


if __name__ == "__main__":
    for test in (
        test_rank_crossing_updates_selected_triplet_with_fixed_seed,
        test_balanced_5x8_respects_mutual_neighbours_and_class_rules,
        test_all_hard_cross_class_negatives_are_nearest,
        test_distance_detaches_and_rejects_boundary,
        test_projected_hier_branch_accepts_float32_boundary_and_keeps_gradient,
    ):
        test()
        print(f"PASS {test.__name__}")
