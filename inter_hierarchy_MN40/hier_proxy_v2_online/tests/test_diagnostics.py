"""Checks that fixed diagnostics do not perturb training randomness or state."""

from __future__ import annotations

import os
import math
import random
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from hier_proxy_v2_online.diagnostics import (
    fixed_rng,
    fixed_triplet_metrics,
    proxy_geometry,
)
from hier_proxy_v2_online.proxy_hierarchy import ProxyHierarchy


def _numpy_states_equal(left: tuple, right: tuple) -> bool:
    return (
        left[0] == right[0]
        and np.array_equal(left[1], right[1])
        and left[2:] == right[2:]
    )


def test_fixed_rng_repeats_probe_and_restores_all_cpu_streams_on_error():
    python_before = random.getstate()
    numpy_before = np.random.get_state()
    torch_before = torch.random.get_rng_state()
    try:
        with fixed_rng(76, "cpu"):
            first = (random.random(), np.random.random(), torch.rand(3))
            raise ZeroDivisionError
    except ZeroDivisionError:
        pass
    else:
        raise AssertionError("the deliberate probe exception was not raised")
    assert random.getstate() == python_before
    assert _numpy_states_equal(np.random.get_state(), numpy_before)
    assert torch.equal(torch.random.get_rng_state(), torch_before)
    with fixed_rng(76, "cpu"):
        second = (random.random(), np.random.random(), torch.rand(3))
    assert first[0] == second[0]
    assert first[1] == second[1]
    assert torch.equal(first[2], second[2])


def test_fixed_rng_restores_every_visible_cuda_stream():
    if not torch.cuda.is_available():
        return
    before = torch.cuda.get_rng_state_all()
    with fixed_rng(81, "cuda:0"):
        first = torch.rand(4, device="cuda:0")
    after = torch.cuda.get_rng_state_all()
    assert all(torch.equal(a, b) for a, b in zip(before, after))
    with fixed_rng(81, "cuda:0"):
        second = torch.rand(4, device="cuda:0")
    assert torch.equal(first, second)


def test_proxy_geometry_reports_hyperbolic_movement_and_assignment_change():
    proxy = ProxyHierarchy(num_proxies=3, dim=2)
    initial_tangent = torch.tensor([[-0.5, 0.0], [0.0, 0.0], [0.5, 0.0]])
    with torch.no_grad():
        proxy.tangent_proxies.copy_(initial_tangent)
    try:
        proxy_geometry(proxy, proxy.tangent_proxies)
    except ValueError:
        pass
    else:
        raise AssertionError("live proxy parameters cannot be the initial snapshot")
    fixed_mu = proxy.ball.expmap0(torch.tensor([[-0.45, 0.0], [0.49, 0.0]]))
    initial_metrics, _ = proxy_geometry(proxy, initial_tangent, fixed_mu)
    assert initial_metrics["proxy_displacement_max"] < 1e-4
    assert initial_metrics["proxy_assignment_change_fraction"] == 0.0
    with torch.no_grad():
        proxy.tangent_proxies[0] = torch.tensor([0.6, 0.0])
    metrics, snapshot = proxy_geometry(proxy, initial_tangent, fixed_mu)
    chunked_metrics, _ = proxy_geometry(proxy, initial_tangent, fixed_mu, chunk_size=1)
    expected = float(proxy.ball.dist(snapshot["initial_ball"][0],
                                     snapshot["current_ball"][0]))
    assert abs(float(snapshot["displacement"][0]) - expected) < 1e-5
    assert metrics["proxy_displacement_max"] > 0.1
    assert metrics["proxy_assignment_change_fraction"] == 0.5
    for key, value in metrics.items():
        assert math.isclose(chunked_metrics[key], value, rel_tol=1e-6, abs_tol=1e-6)
    assert int(snapshot["current_occupancy"].sum()) == 2
    assert snapshot["initial_assignment"].tolist() == [0, 2]
    assert snapshot["current_assignment"].tolist() == [1, 2]
    assert metrics["proxy_occupied"] == 2
    assert all(torch.isfinite(value).all() for value in snapshot.values()
               if value.is_floating_point())


def test_fixed_triplet_four_corners_use_the_same_random_draws():
    proxy = ProxyHierarchy(num_proxies=8, dim=2, seed=33)
    tangent = torch.tensor([
        [-0.4, 0.0], [-0.3, 0.3], [0.0, 0.4], [0.3, 0.3],
        [0.4, 0.0], [0.3, -0.3], [0.0, -0.4], [-0.3, -0.3],
    ])
    with torch.no_grad():
        proxy.tangent_proxies.copy_(tangent)
    mu = proxy.ball.expmap0(tangent[:4] * 0.9)
    triplets = (torch.tensor([0, 1, 2, 3]),
                torch.tensor([1, 2, 3, 0]),
                torch.tensor([2, 3, 0, 1]))
    result = fixed_triplet_metrics(proxy, tangent, mu, mu, triplets, seed=44)
    suffixes = ("loss", "triplets", "valid_rate", "active_rate",
                "margin_satisfied_rate_valid")
    for suffix in suffixes:
        values = [result[f"probe_{student}_{ancestor}_{suffix}"]
                  for student in ("s0", "s1") for ancestor in ("p0", "p1")]
        assert values.count(values[0]) == 4
    assert result["probe_s0_p0_triplets"] == 4


if __name__ == "__main__":
    tests = (
        test_fixed_rng_repeats_probe_and_restores_all_cpu_streams_on_error,
        test_fixed_rng_restores_every_visible_cuda_stream,
        test_proxy_geometry_reports_hyperbolic_movement_and_assignment_change,
        test_fixed_triplet_four_corners_use_the_same_random_draws,
    )
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
