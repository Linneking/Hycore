"""CPU-only tests for the two HIER triplet-mining branches."""

import math
import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from hier_proxy_v1.mining import mine_batch_triplets, pairwise_poincare_distance_c1


def _balanced_points(classes: int, per_class: int):
    generator = torch.Generator().manual_seed(91)
    points = []
    labels = []
    for cls in range(classes):
        angle = 2 * math.pi * cls / classes
        center = torch.tensor([0.45 * math.cos(angle), 0.45 * math.sin(angle)])
        points.append(center + 0.065 * torch.randn(per_class, 2, generator=generator))
        labels.extend([cls] * per_class)
    return torch.cat(points).requires_grad_(True), torch.tensor(labels)


def test_distance_has_c1_geometry_and_detaches():
    points = torch.tensor([[0.0, 0.0], [0.5, 0.0]], requires_grad=True)
    distance = pairwise_poincare_distance_c1(points)
    assert distance.shape == (2, 2)
    assert torch.allclose(distance, distance.T, atol=1e-7)
    assert torch.allclose(distance.diag(), torch.zeros(2))
    assert torch.allclose(distance[0, 1], torch.tensor(math.log(3)), atol=1e-6)
    assert not distance.requires_grad


def _check_balanced(classes: int, per_class: int):
    points, labels = _balanced_points(classes, per_class)
    result = mine_batch_triplets(points, labels, k_in=3, seed=22)
    again = mine_batch_triplets(points, labels, k_in=3, seed=22)
    assert torch.equal(result["in"], again["in"])
    assert torch.equal(result["out"], again["out"])
    assert result["stats"] == again["stats"]
    assert result["stats"]["classes"] == classes
    assert result["stats"]["batch_size"] == classes * per_class
    assert result["stats"]["in_triplets"] > 0
    assert result["stats"]["out_triplets"] > 0
    assert result["stats"]["out_hard_count"] + result["stats"]["out_random_count"] == result["stats"]["out_triplets"]
    assert result["stats"]["out_hard_count"] == int(result["stats"]["out_triplets"] * 0.5 + 0.5)

    distances = pairwise_poincare_distance_c1(points).tolist()
    topk = []
    for i in range(len(labels)):
        same = [j for j in range(len(labels)) if j != i and labels[j] == labels[i]]
        same.sort(key=lambda j: (distances[i][j], j))
        topk.append(set(same[:3]))

    for branch in ("in", "out"):
        triples = result[branch]
        assert triples.dtype == torch.long
        assert triples.shape[1] == 3
        assert not triples.requires_grad
        assert len(set(triples[:, 0].tolist())) == len(triples)  # one per anchor
        for i, j, k in triples.tolist():
            assert len({i, j, k}) == 3
            assert j in topk[i] and i in topk[j]
            assert labels[j] == labels[i]
            if branch == "in":
                assert labels[k] == labels[i]
                assert k not in topk[i]
            else:
                assert labels[k] != labels[i]

    # Every inter-class anchor with an intra-class triplet shares its chosen j.
    out_positive = {i: j for i, j, _ in result["out"].tolist()}
    assert all(out_positive[i] == j for i, j, _ in result["in"].tolist())


def test_balanced_4x10():
    _check_balanced(4, 10)


def test_balanced_5x8():
    _check_balanced(5, 8)


def test_hard_inter_class_negative_is_nearest():
    points, labels = _balanced_points(5, 8)
    result = mine_batch_triplets(points, labels, seed=42, hard_ratio=1.0)
    distance = pairwise_poincare_distance_c1(points).tolist()
    assert result["stats"]["out_hard_count"] == len(result["out"])
    for i, _, k in result["out"].tolist():
        other = [idx for idx in range(len(labels)) if labels[idx] != labels[i]]
        expected = min(other, key=lambda idx: (distance[i][idx], idx))
        assert k == expected


def test_missing_candidate_skips_branch():
    # All three non-self neighbours are in top-3; no same-class k exists.
    points = torch.tensor([[0.1, 0.0], [0.2, 0.0], [0.3, 0.0], [0.4, 0.0]])
    one_class = mine_batch_triplets(points, torch.zeros(4, dtype=torch.long))
    assert one_class["in"].shape == (0, 3)
    assert one_class["out"].shape == (0, 3)
    # Distinct classes cannot supply a same-class positive j.
    unique = mine_batch_triplets(points, torch.arange(4))
    assert unique["in"].shape == (0, 3)
    assert unique["out"].shape == (0, 3)


def test_empty_and_invalid_embeddings():
    empty = mine_batch_triplets(torch.empty(0, 2), torch.empty(0, dtype=torch.long))
    assert empty["stats"]["batch_size"] == 0
    assert empty["in"].shape == (0, 3)
    try:
        mine_batch_triplets(torch.tensor([[1.0, 0.0]]), torch.tensor([0]))
    except ValueError as error:
        assert "ball" in str(error)
    else:
        raise AssertionError("boundary point should be rejected")


if __name__ == "__main__":
    for test in (
        test_distance_has_c1_geometry_and_detaches,
        test_balanced_4x10,
        test_balanced_5x8,
        test_hard_inter_class_negative_is_nearest,
        test_missing_candidate_skips_branch,
        test_empty_and_invalid_embeddings,
    ):
        test()
        print(f"PASS {test.__name__}")
