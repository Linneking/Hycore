"""Numerical gates for shared c=1 and branch-specific HIER backward."""

from __future__ import annotations

import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import geoopt
import torch

from hier_proxy_scratch_v4.geometry import (
    DEFAULT_MARGIN, DEFAULT_TAU, ProxyHierarchy, hier_gradient,
    radius_gradient_audit, riemannian_factor, source_cap_intra_audit,
)


def test_hier_precondition_is_identity_and_does_not_touch_base_gradient():
    points = torch.tensor([[0.8, 0.0], [0.0, 0.5]], dtype=torch.float64, requires_grad=True)
    branch = hier_gradient(points, "riemannian")
    assert torch.equal(branch, points)
    base_weights = torch.tensor([[2.0, -1.0], [0.5, 3.0]], dtype=points.dtype)
    hier_weights = torch.tensor([[1.0, 4.0], [-2.0, 1.0]], dtype=points.dtype)
    total = (points * base_weights).sum() + (branch * hier_weights).sum()
    total.backward()
    expected = base_weights + riemannian_factor(points.detach()) * hier_weights
    assert torch.allclose(points.grad, expected, atol=1e-12, rtol=1e-12)
    assert torch.allclose(points.grad - base_weights,
                          riemannian_factor(points.detach()) * hier_weights)


def test_c1_distance_units_preserve_source_hinge_and_gumbel_logits():
    ball_old = geoopt.PoincareBall(c=0.1).double()
    ball_new = geoopt.PoincareBall(c=1.0).double()
    # Use actual represented curvature, avoiding float32 constructor rounding.
    scale = math.sqrt(float(ball_old.c) / float(ball_new.c))
    x_old = torch.tensor([[0.1, 0.3], [-0.2, 0.4], [0.25, -0.15]], dtype=torch.float64)
    p_old = torch.tensor([[0.35, 0.0], [-0.25, -0.2], [0.0, 0.55]], dtype=torch.float64)
    old = ball_old.dist(x_old[:, None, :], p_old[None, :, :])
    new = ball_new.dist((scale * x_old)[:, None, :], (scale * p_old)[None, :, :])
    assert torch.allclose(new, scale * old, atol=2e-9, rtol=2e-9)
    old_pair, new_pair = torch.maximum(old[0], old[1]), torch.maximum(new[0], new[1])
    assert torch.allclose(-new_pair / (0.1 * scale), -old_pair / 0.1, atol=2e-8, rtol=2e-8)
    old_hinge = torch.relu(old[0, 0] - old[0, 1] + 0.1)
    new_hinge = torch.relu(new[0, 0] - new[0, 1] + 0.1 * scale)
    assert torch.allclose(new_hinge / scale, old_hinge, atol=2e-9, rtol=2e-9)
    assert abs(DEFAULT_MARGIN - 0.0316227766016838) < 1e-14
    assert DEFAULT_TAU == DEFAULT_MARGIN


def _hierarchy() -> ProxyHierarchy:
    hierarchy = ProxyHierarchy(num_proxies=16, dim=4, seed=42).double()
    generator = torch.Generator().manual_seed(490)
    reference = hierarchy.ball.expmap0(torch.randn(48, 4, generator=generator, dtype=torch.float64) * 0.15)
    hierarchy.initialize(reference)
    return hierarchy


def _triplets():
    return (
        torch.tensor([0, 0, 1, 1, 2, 2, 3, 3] * 8),
        torch.tensor([1, 2, 2, 3, 3, 4, 4, 5] * 8),
        torch.tensor([3, 4, 4, 5, 5, 0, 0, 1] * 8),
    )


def test_gradient_audit_has_identical_forward_and_finite_backward():
    hierarchy = _hierarchy()
    mu = hierarchy.ball.expmap0(torch.randn(6, 4, dtype=torch.float64) * 0.2)
    report = hierarchy.gradient_mode_audit(mu, _triplets(), margin=1.0, tau=0.2, seed=109)
    assert report["raw_loss"] == report["riemannian_loss"]
    assert report["active_triplets"] > 0
    assert report["mu_raw_grad_norm"] > 0
    assert 0 < report["mu_riemannian_to_raw_ratio"] <= 0.25 + 1e-10
    assert report["proxy_tangent_raw_grad_norm"] > 0
    assert math.isfinite(report["proxy_tangent_riemannian_grad_norm"])
    # autograd.grad diagnostics never leave accumulated training gradients.
    assert hierarchy.tangent_proxies.grad is None


def test_shared_whole_is_not_capped_and_source_cap_conflicts_with_intra():
    hierarchy = _hierarchy()
    mu = torch.zeros(6, 4, dtype=torch.float64, requires_grad=True)
    with torch.no_grad():
        mu[:, 0] = torch.tensor([0.8, 0.85, 0.9, 0.95, 0.97, 0.99], dtype=mu.dtype)
    before = mu.detach().clone()
    loss, _ = hierarchy.sample_loss(mu, _triplets(), margin=1.0, tau=0.2,
                                   generator=torch.Generator().manual_seed(109))
    assert torch.equal(mu.detach(), before)
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(mu.grad).all()
    report = source_cap_intra_audit()
    assert report["whole_output_capped_in_v4"] is False
    assert report["maximum_whole_radial_depth_d1"] < 1000 / 600
    assert all(row["minimum_hinge_even_with_part_at_origin"] > 0
               for row in report["intra_lower_bounds"])
    assert radius_gradient_audit([0.996])[0]["riemannian_to_euclidean_factor"] < 2e-5


def test_proxy_precondition_still_backpropagates_through_tangent_mapping():
    hierarchy = _hierarchy()
    hierarchy.proxy_gradient_mode = "riemannian"
    loss, stats = hierarchy.proxy_loss(k=5, t_per_anchor=8, margin=1.0, tau=0.2,
                                      generator=torch.Generator().manual_seed(39))
    assert stats["triplets"] > 0
    assert torch.isfinite(loss)
    loss.backward()
    assert hierarchy.tangent_proxies.grad is not None
    assert torch.isfinite(hierarchy.tangent_proxies.grad).all()
    assert float(hierarchy.tangent_proxies.grad.norm()) > 0


if __name__ == "__main__":
    tests = [
        test_hier_precondition_is_identity_and_does_not_touch_base_gradient,
        test_c1_distance_units_preserve_source_hinge_and_gumbel_logits,
        test_gradient_audit_has_identical_forward_and_finite_backward,
        test_shared_whole_is_not_capped_and_source_cap_conflicts_with_intra,
        test_proxy_precondition_still_backpropagates_through_tangent_mapping,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
