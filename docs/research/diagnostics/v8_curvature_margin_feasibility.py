"""CPU-only analytic V8 geometry audit; no data, weights, or training.

Positive c denotes curvature magnitude (sectional curvature is -c).
Inputs follow Geoopt expmap0 and the HIER tangent-cap convention.
The joint feasibility bound assumes ALL radial hinges vanish, common N,
and the same origin-depth cap for whole and part. It is necessary only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path


def radius(depth: float, c: float) -> float:
    return math.tanh(math.sqrt(c) * depth / 2) / math.sqrt(c)


def depth_from_radius(r: float, c: float) -> float:
    return 2 * math.atanh(math.sqrt(c) * r) / math.sqrt(c)


def polar_distance(a: float, b: float, angle: float, c: float) -> float:
    # Hyperbolic cosine law; a,b are origin distances, angle is in radians.
    sc = math.sqrt(c)
    value = math.cosh(sc * a) * math.cosh(sc * b)
    value -= math.sinh(sc * a) * math.sinh(sc * b) * math.cos(angle)
    return math.acosh(max(1.0, value)) / sc


def poincare_distance(a: float, b: float, angle: float, c: float) -> float:
    # Independent coordinate formula checks the polar calculation.
    r, t = radius(a, c), radius(b, c)
    squared = r * r + t * t - 2 * r * t * math.cos(angle)
    value = 1 + 2 * c * squared / ((1 - c * r * r) * (1 - c * t * t))
    return math.acosh(max(1.0, value)) / math.sqrt(c)


def policy(name: str, gamma: float, delta: float, cap: float, c: float = .1) -> dict:
    rows = []
    impossible_radial = []
    impossible_joint = []
    for n in range(200, 601):
        m = gamma / n
        if m > cap:
            impossible_radial.append(n)
        # d(w,p_neg)-d(w,p_pos) <= 2*(cap-m) if all radial hinges zero.
        if delta > 2 * (cap - m):
            impossible_joint.append(n)
        if n in (200, 300, 400, 500, 600):
            part_max = max(0.0, cap - m)
            rows.append({
                "N": n, "radial_margin": m,
                "part_depth_ceiling_if_radial_zero_and_whole_at_cap": cap - m,
                "contrastive_gap_upper_bound_if_all_radial_zero": 2 * (cap - m) if m <= cap else None,
                "minimum_contrastive_hinge_if_all_radial_zero": max(0, delta - 2 * (cap - m)) if m <= cap else None,
                "orthogonal_negative_gap_in_radial_tight_construction":
                    polar_distance(cap, part_max, math.pi / 2, c) - m
                    if m <= cap else None,
            })
    return {"name": name, "gamma": gamma, "delta": delta, "depth_cap": cap, "c": c,
            "radial_impossible_N": impossible_radial,
            "joint_zero_impossible_N_necessary_bound": impossible_joint,
            "joint_impossible_crop_fraction_under_uniform_integer_N": len(impossible_joint) / 401,
            "rows": rows}


def audit() -> dict:
    old_cap = 2 * math.atanh(.996)  # Geoopt 0.5.1 FP32 projection, c=1.
    new_cap = 2 * 2.3
    scale = new_cap / old_cap
    geometry = []
    max_inverse_error = 0.
    for c in (1., .5, .1, .01):
        for tangent_norm in (2.07, 2.3):
            d = 2 * tangent_norm
            r = radius(d, c)
            recovered_depth = depth_from_radius(r, c)
            inverse_error = abs(recovered_depth - d)
            max_inverse_error = max(max_inverse_error, inverse_error)
            assert inverse_error < 1e-12
            geometry.append({"c": c, "tangent_norm": tangent_norm,
                "ball_boundary": 1 / math.sqrt(c), "coordinate_radius": r,
                "normalized_coordinate_radius": math.sqrt(c) * r,
                "origin_depth": recovered_depth,
                "inverse_metric_gradient_factor": (1 - c * r * r)**2 / 4})
    rng = random.Random(20261015)
    max_error = 0.
    for _ in range(500):
        c = rng.choice((1., .5, .1, .01))
        a, b, angle = rng.uniform(.1, new_cap), rng.uniform(.1, new_cap), rng.uniform(.01, math.pi)
        max_error = max(max_error, abs(polar_distance(a, b, angle, c) - poincare_distance(a, b, angle, c)))
    assert max_error < 1e-10
    for c in (1., .1):
        assert abs(polar_distance(4.6, 2.1, 0., c) - 2.5) < 1e-12
        assert abs(polar_distance(4.6, 2.1, math.pi, c) - 6.7) < 1e-12
    for g, d in ((500., 2.), (500., 4.), (1000 * scale, 4.)):
        m = g / 200
        p = new_cap - m
        gap = polar_distance(new_cap, p, math.pi, .1) - m
        assert abs(gap - 2 * (new_cap - m)) < 1e-12
    change_c = []
    for angle_deg in (0, 30, 60, 90, 120, 180):
        angle = math.radians(angle_deg)
        old = polar_distance(4.6, 2.1, angle, 1.)
        new = polar_distance(4.6, 2.1, angle, .1)
        change_c.append({"whole_depth": 4.6, "part_depth": 2.1,
            "angle_degrees": angle_deg, "distance_c1": old,
            "distance_c01": new, "ratio_c01_over_c1": new / old})
    curves = [policy("unscaled", 1000, 4, new_cap),
              policy("illustrative_2p3_over_4_not_unit_consistent", 575, 4, new_cap),
              policy("depth_budget_scale_radial_only", 1000 * scale, 4, new_cap),
              policy("depth_budget_scale_both_margins", 1000 * scale, 4 * scale, new_cap),
              policy("radial500_contrastive4", 500, 4, new_cap),
              policy("half_scale_both_candidate", 500, 2, new_cap),
              policy("original_c1_projection_budget", 1000, 4, old_cap, 1.)]
    # Feasibility count depends only on depth budgets, not the angle construction.
    assert curves[0]["radial_impossible_N"] == list(range(200, 218))
    assert curves[2]["joint_zero_impossible_N_necessary_bound"] == list(range(200, 285))
    assert curves[3]["joint_zero_impossible_N_necessary_bound"] == curves[-1]["joint_zero_impossible_N_necessary_bound"]
    assert curves[4]["joint_zero_impossible_N_necessary_bound"] == []
    assert curves[5]["joint_zero_impossible_N_necessary_bound"] == []
    orthogonal_candidate_gaps = []
    for n in range(200, 601):
        m = 500 / n
        gap = polar_distance(new_cap, new_cap - m, math.pi / 2, .1) - m
        orthogonal_candidate_gaps.append(gap)
        assert gap > 2
    return {"format": "v8-curvature-margin-analytic-audit-v1",
        "date": "2026-10-15", "scope": "CPU float64 scalar formulas only; no training or source data",
        "old_projection_epsilon_fp32": .004, "old_projection_depth_cap_c1": old_cap,
        "hier_tangent_cap": 2.3, "hier_origin_depth_cap": new_cap,
        "depth_budget_scale": scale,
        "max_gamma_if_only_radial_scaled_and_delta4_joint_bound": 200 * (new_cap - 4 / 2),
        "max_common_margin_scale_for_joint_bound": 2 * new_cap / (2 * 5 + 4),
        "geometry": geometry, "fixed_depth_curvature_change": change_c,
        "policies": curves, "checks": {"random_formula_cases": 500,
            "independent_distance_formula_max_error": max_error,
            "inverse_map_cases": len(geometry),
            "inverse_map_max_depth_error": max_inverse_error,
            "candidate_orthogonal_zero_hinge_constructions": 401,
            "candidate_orthogonal_contrastive_gap_minimum": min(orthogonal_candidate_gaps),
            "inverse_map_and_tight_triangle_constructions": "passed",
            "crop_range_and_bound_regressions": "passed"},
        "limitations": ["Only necessary joint bound; direction may make feasibility stricter",
            "All radial hinges zero assumed; violated bound does not invalidate soft-loss training",
            "Geoopt projection limit is an implementation budget, not intrinsic hyperbolic-space limit",
            "Analytic HIER cap ignores <=1e-5 clipping epsilon and floating-point roundoff",
            "No statement about trained embeddings, Adam/SGD displacement, or classification quality"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Use a new output file; prior audit results are immutable")
    result = audit()
    result["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "scale": result["depth_budget_scale"],
                      "checks": result["checks"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
