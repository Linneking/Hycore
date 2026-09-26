"""Focused numerical and sampling tests for the HIER proxy objective."""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from hier_proxy_v1.proxy_hierarchy import ProxyHierarchy


def _reference(dim: int = 6) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(904)
    tangent = torch.randn(48, dim, generator=generator) * 0.06
    return ProxyHierarchy(num_proxies=16, dim=dim).ball.expmap0(tangent)


def test_proxy_initialization_is_deterministic_and_inside_ball():
    reference = _reference()
    a = ProxyHierarchy(num_proxies=16, dim=6, seed=42)
    b = ProxyHierarchy(num_proxies=16, dim=6, seed=42)
    meta_a, meta_b = a.initialize(reference), b.initialize(reference)
    assert meta_a == meta_b
    assert torch.allclose(a.tangent_proxies, b.tangent_proxies)
    assert bool((a.proxies().norm(dim=-1) < 1).all())
    assert torch.allclose(
        a.tangent_proxies.norm(dim=-1),
        torch.full((16,), meta_a["init_tangent_norm"]), atol=1e-6,
    )
    with torch.no_grad():
        a.tangent_proxies.mul_(1000)
    assert bool((a.proxies().norm(dim=-1) < 1).all())


def test_sample_loss_has_finite_student_and_proxy_gradients():
    hierarchy = ProxyHierarchy(num_proxies=32, dim=6, seed=42)
    hierarchy.initialize(_reference())
    tangent = torch.nn.Parameter(torch.randn(6, 6) * 0.08)
    mu = hierarchy.ball.expmap0(tangent)
    triplets = (
        torch.tensor([0, 0, 1, 1, 2, 2, 3, 3] * 8),
        torch.tensor([1, 2, 2, 3, 3, 4, 4, 5] * 8),
        torch.tensor([3, 4, 4, 5, 5, 0, 0, 1] * 8),
    )
    loss, stats = hierarchy.sample_loss(mu, triplets, margin=1.0, tau=0.2)
    assert stats["triplets"] == 64
    assert stats["valid_triplets"] > 0
    assert torch.isfinite(loss)
    loss.backward()
    assert tangent.grad is not None and torch.isfinite(tangent.grad).all()
    assert hierarchy.tangent_proxies.grad is not None
    assert torch.isfinite(hierarchy.tangent_proxies.grad).all()
    assert float(hierarchy.tangent_proxies.grad.abs().sum()) > 0


def test_proxy_miner_excludes_self_and_returns_finite_loss():
    hierarchy = ProxyHierarchy(num_proxies=32, dim=6, seed=29)
    hierarchy.initialize(_reference())
    i, j, k = hierarchy.mine_proxy_triplets(k=8, t_per_anchor=2)[0]
    assert i.numel() > 0
    assert bool(((i != j) & (i != k) & (j != k)).all())
    loss, stats = hierarchy.proxy_loss(k=8, t_per_anchor=2, margin=1.0)
    assert stats["triplets"] > 0
    assert torch.isfinite(loss)
    loss.backward()
    assert hierarchy.tangent_proxies.grad is not None
    assert torch.isfinite(hierarchy.tangent_proxies.grad).all()


def test_empty_triplets_return_differentiable_zero():
    hierarchy = ProxyHierarchy(num_proxies=8, dim=6)
    hierarchy.initialize(_reference())
    tangent = torch.nn.Parameter(torch.randn(3, 6) * 0.02)
    mu = hierarchy.ball.expmap0(tangent)
    empty = torch.empty(0, dtype=torch.long)
    loss, stats = hierarchy.sample_loss(mu, (empty, empty, empty))
    assert stats["triplets"] == 0 and loss.item() == 0
    loss.backward()
    assert tangent.grad is not None and torch.isfinite(tangent.grad).all()
    assert hierarchy.tangent_proxies.grad is not None


def test_invalid_triplets_are_rejected():
    hierarchy = ProxyHierarchy(num_proxies=8, dim=6)
    hierarchy.initialize(_reference())
    mu = _reference()[:3]
    try:
        hierarchy.sample_loss(mu, ([0], [1], [0]))
    except ValueError:
        pass
    else:
        raise AssertionError("self-negative triplet was accepted")


def test_collision_mask_keeps_all_triplets_in_denominator():
    hierarchy = ProxyHierarchy(num_proxies=3, dim=2)
    distances = torch.tensor(
        [[2.0, 0.0, 3.0], [2.0, 0.0, 3.0],
         [0.0, 2.0, 3.0], [1.0, 1.0, 0.0]], requires_grad=True,
    )
    assignments = [
        torch.tensor([[1., 0., 0.], [1., 0., 0.]]),
        torch.tensor([[0., 1., 0.], [1., 0., 0.]]),
    ]
    hierarchy._hard_gumbel = lambda logits, generator: assignments.pop(0)
    loss, stats = hierarchy._loss_from_distances(
        distances, ([0, 0], [1, 2], [2, 3]),
        tau=0.1, margin=1.0, generator=None,
    )
    assert stats["triplets"] == 2 and stats["collisions"] == 1
    assert torch.allclose(loss, torch.tensor(4.5))


if __name__ == "__main__":
    tests = [
        test_proxy_initialization_is_deterministic_and_inside_ball,
        test_sample_loss_has_finite_student_and_proxy_gradients,
        test_proxy_miner_excludes_self_and_returns_finite_loss,
        test_empty_triplets_return_differentiable_zero,
        test_invalid_triplets_are_rejected,
        test_collision_mask_keeps_all_triplets_in_denominator,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
