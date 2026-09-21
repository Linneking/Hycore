import os
import sys

import torch
from geoopt import PoincareBall

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from v2.geometry import (
    equal_radius_leaves,
    exact_lca,
    exact_lca_depth,
    gromov_product_matrix,
    pairwise_ball_distance,
)
from v2.losses import gather_teacher_similarity, lca_ranking_loss


def _points():
    return torch.tensor(
        [[0.45, 0.05], [0.42, 0.12], [-0.35, 0.18], [0.05, -0.40]],
        dtype=torch.float64,
        requires_grad=True,
    )


def test_pairwise_distance_and_gromov_invariants():
    ball = PoincareBall(c=1.0)
    z = _points()
    d = pairwise_ball_distance(ball, z, z)
    assert d.shape == (4, 4)
    assert torch.allclose(d, d.T, atol=1e-9)
    assert torch.allclose(d.diag(), torch.zeros(4, dtype=z.dtype), atol=1e-9)
    g = gromov_product_matrix(ball, z, z)
    assert torch.allclose(g, g.T, atol=1e-9)
    assert torch.allclose(g.diag(), ball.dist0(z), atol=1e-9)


def test_exact_lca_is_symmetric_finite_and_differentiable():
    ball = PoincareBall(c=1.0)
    z = _points()
    a = exact_lca(z[0], z[2])
    b = exact_lca(z[2], z[0])
    assert torch.allclose(a, b, atol=1e-9)
    depths = exact_lca_depth(ball, z, z)
    assert torch.isfinite(depths).all()
    depths.sum().backward()
    assert z.grad is not None and torch.isfinite(z.grad).all()


def test_equal_radius_leaves_are_inside_ball():
    z = _points().detach()
    leaves = equal_radius_leaves(z, 0.8)
    norms = leaves.norm(dim=-1)
    assert torch.allclose(norms, torch.full_like(norms, 0.8), atol=1e-9)
    assert bool((norms < 1).all())


def test_teacher_similarity_preserves_batch_order():
    teacher = torch.eye(5, dtype=torch.float64)
    ids = torch.tensor([3, 1, 4])
    actual = gather_teacher_similarity(teacher, ids)
    expected = teacher[ids] @ teacher[ids].T
    assert torch.equal(actual, expected)


def test_ranking_loss_permutation_invariance_and_empty_case():
    ball = PoincareBall(c=1.0)
    z = equal_radius_leaves(_points(), 0.7)
    labels = torch.tensor([0, 0, 0, 0])
    teacher = torch.tensor(
        [[1.0, .95, .20, .10], [.95, 1.0, .15, .05],
         [.20, .15, 1.0, .90], [.10, .05, .90, 1.0]], dtype=z.dtype
    )
    loss, stats = lca_ranking_loss(z, labels, teacher, ball, neighbor_k=1)
    perm = torch.tensor([2, 0, 3, 1])
    loss2, stats2 = lca_ranking_loss(
        z[perm], labels[perm], teacher[perm][:, perm], ball, neighbor_k=1
    )
    assert torch.allclose(loss, loss2, atol=1e-9)
    assert stats.triplets == stats2.triplets
    empty, empty_stats = lca_ranking_loss(
        z[:2], labels[:2], teacher[:2, :2], ball, neighbor_k=1
    )
    assert empty_stats.triplets == 0
    empty.backward()


def test_better_student_order_reduces_ranking_loss():
    ball = PoincareBall(c=1.0)
    labels = torch.zeros(3, dtype=torch.long)
    teacher = torch.tensor(
        [[1.0, .9, .1], [.9, 1.0, .2], [.1, .2, 1.0]], dtype=torch.float64)
    bad = torch.tensor([[.7, 0.0], [-.65, .10], [.68, .12]], dtype=torch.float64)
    good = torch.tensor([[.7, 0.0], [.68, .12], [-.65, .10]], dtype=torch.float64)
    bad_loss, _ = lca_ranking_loss(bad, labels, teacher, ball, neighbor_k=1)
    good_loss, _ = lca_ranking_loss(good, labels, teacher, ball, neighbor_k=1)
    assert good_loss < bad_loss


if __name__ == "__main__":
    tests = [
        test_pairwise_distance_and_gromov_invariants,
        test_exact_lca_is_symmetric_finite_and_differentiable,
        test_equal_radius_leaves_are_inside_ball,
        test_teacher_similarity_preserves_batch_order,
        test_ranking_loss_permutation_invariance_and_empty_case,
        test_better_student_order_reduces_ranking_loss,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
