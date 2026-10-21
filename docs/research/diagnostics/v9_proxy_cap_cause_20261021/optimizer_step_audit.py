"""Read-only AdamW step accounting; no extra forward/backward or optimizer step.

Call capture_before(parameter, optimizer) after gradient synchronization, then
observe_after(capture, parameter) immediately after the real AdamW step.  All
counterfactual components share the actual updated second moment. They are an
algebraic attribution of this one step, not independent training interventions.
"""
from __future__ import annotations

import torch


@torch.no_grad()
def capture_before(parameter, optimizer):
    groups = [group for group in optimizer.param_groups
              if any(value is parameter for value in group["params"])]
    if len(groups) != 1 or parameter.grad is None:
        raise ValueError("Expected one proxy group and a present dense gradient")
    group = groups[0]
    if group.get("amsgrad", False) or group.get("maximize", False):
        raise ValueError("Audit supports the actual V9 non-AMSGrad minimizing AdamW")
    state = optimizer.state.get(parameter, {})
    step = state.get("step", 0)
    step = int(step.item()) if isinstance(step, torch.Tensor) else int(step)
    # Copies are private and detached; neither optimizer nor RNG is changed.
    return {
        "parameter": parameter.detach().clone(),
        "gradient": parameter.grad.detach().clone(),
        "exp_avg": state.get("exp_avg", torch.zeros_like(parameter)).detach().clone(),
        "exp_avg_sq": state.get("exp_avg_sq", torch.zeros_like(parameter)).detach().clone(),
        "step_before": step, "lr": float(group["lr"]),
        "betas": tuple(map(float, group["betas"])), "eps": float(group["eps"]),
        "weight_decay": float(group["weight_decay"]),
    }


def _summary(value):
    value = value.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
    if not bool(torch.isfinite(value).all()):
        raise ValueError("Nonfinite audit statistic")
    return {"mean": float(value.mean()), "min": float(value.min()),
            "median": float(value.median()), "max": float(value.max())}


@torch.no_grad()
def observe_after(snapshot, parameter_after, *, include_per_proxy=False, optimizer=None):
    p, g, m, v = (snapshot[key].double() for key in
                  ("parameter", "gradient", "exp_avg", "exp_avg_sq"))
    after = parameter_after.detach().double()
    radius = p.norm(dim=-1)
    if bool((radius == 0).any()):
        raise ValueError("Zero proxy has undefined radial direction")
    u = p / radius[:, None]
    beta1, beta2 = snapshot["betas"]
    step = snapshot["step_before"] + 1
    bc1, bc2 = 1 - beta1 ** step, 1 - beta2 ** step
    denominator = ((beta2 * v + (1 - beta2) * g.square()) / bc2).sqrt() + snapshot["eps"]
    scalar = -snapshot["lr"] / bc1
    g_radial = (g * u).sum(-1)
    m_radial = (m * u).sum(-1)
    history_radial = scalar * beta1 * m_radial[:, None] * u / denominator
    history_tangent = scalar * beta1 * (m - m_radial[:, None] * u) / denominator
    current_radial = scalar * (1 - beta1) * g_radial[:, None] * u / denominator
    current_tangent = scalar * (1 - beta1) * (g - g_radial[:, None] * u) / denominator
    decay = -snapshot["lr"] * snapshot["weight_decay"] * p
    predicted = history_radial + history_tangent + current_radial + current_tangent + decay
    actual = after - p
    radial_step = (actual * u).sum(-1)
    angular_step = actual - radial_step[:, None] * u
    norm_change = after.norm(dim=-1) - radius
    tiny = torch.finfo(torch.float64).tiny
    per_proxy = {
        "before_norm": radius, "after_norm": after.norm(dim=-1),
        "signed_radial_gradient": g_radial,
        "radial_gradient_fraction": g_radial.abs() / g.norm(dim=-1).clamp_min(tiny),
        "gradient_norm": g.norm(dim=-1), "history_m_radial": m_radial,
        "history_m_radial_fraction": m_radial.abs() / m.norm(dim=-1).clamp_min(tiny),
        "history_radial_m_step": (history_radial * u).sum(-1),
        "history_tangent_m_step": (history_tangent * u).sum(-1),
        "current_radial_g_step": (current_radial * u).sum(-1),
        "current_tangent_g_step": (current_tangent * u).sum(-1),
        "weight_decay_step": (decay * u).sum(-1),
        "predicted_radial_step": (predicted * u).sum(-1),
        "actual_radial_step": radial_step, "actual_angular_step": angular_step.norm(dim=-1),
        "actual_step_norm": actual.norm(dim=-1), "exact_norm_change": norm_change,
        "finite_step_excess": norm_change - radial_step,
        "prediction_error_norm": (predicted - actual).norm(dim=-1),
        "prediction_error_max_coordinate": (predicted - actual).abs().amax(dim=-1),
        "eps_dominated_coordinate_fraction": ((denominator - snapshot["eps"]) < snapshot["eps"]).double().mean(-1),
        "newly_over_cap": ((radius <= 2.3) & (after.norm(dim=-1) > 2.3)).double(),
        "returned_inside_cap": ((radius > 2.3) & (after.norm(dim=-1) <= 2.3)).double(),
    }
    report = {"step_before": snapshot["step_before"], "lr": snapshot["lr"],
              "eps": snapshot["eps"], "weight_decay": snapshot["weight_decay"],
              "counts": {"before_over_cap": int((radius > 2.3).sum()),
                         "after_over_cap": int((after.norm(dim=-1) > 2.3).sum()),
                         "newly_over_cap": int(per_proxy["newly_over_cap"].sum()),
                         "returned_inside_cap": int(per_proxy["returned_inside_cap"].sum()),
                         "current_gradient_wants_outward": int((g_radial < 0).sum()),
                         "actual_step_outward": int((norm_change > 0).sum())},
              "all": {key: _summary(value) for key, value in per_proxy.items()},
              "by_before_cap": {}}
    if optimizer is not None:
        state_after = optimizer.state[parameter_after]
        expected_m = beta1 * m + (1 - beta1) * g
        expected_v = beta2 * v + (1 - beta2) * g.square()
        actual_m = state_after["exp_avg"].detach().double()
        actual_v = state_after["exp_avg_sq"].detach().double()
        report["state_invariants"] = {
            "step_after": int(state_after["step"].item()),
            "expected_step_after": step,
            "m_max_abs_error": float((actual_m - expected_m).abs().max()),
            "v_max_abs_error": float((actual_v - expected_v).abs().max()),
            "squared_norm_identity_max_error": float((
                after.square().sum(-1) - p.square().sum(-1)
                - 2 * radius * radial_step - actual.square().sum(-1)).abs().max()),
        }
    for name, mask in (("inside", radius <= 2.3), ("outside", radius > 2.3)):
        report["by_before_cap"][name] = {"count": int(mask.sum())}
        if bool(mask.any()):
            report["by_before_cap"][name].update(
                {key: _summary(value[mask]) for key, value in per_proxy.items()})
    if include_per_proxy:
        # Intended only for sparse private snapshots, never Git or public logs.
        report["_per_proxy"] = {key: value.cpu() for key, value in per_proxy.items()}
    return report


# Explicit aliases for runners which use before_step/after_step terminology.
before_step = capture_before


@torch.no_grad()
def after_step(before, parameter, optimizer, *, include_snapshot=False):
    """Return (report, private_cpu_snapshot); report includes _per_proxy tensors.

    Pop _per_proxy before writing JSON. Include private matrices only for sparse
    steps; the default returns None as the second tuple member.
    """
    report = observe_after(before, parameter, include_per_proxy=True, optimizer=optimizer)
    private = None
    if include_snapshot:
        private = {key: (value.detach().cpu().clone() if isinstance(value, torch.Tensor) else value)
                   for key, value in before.items()}
        private["parameter_after"] = parameter.detach().cpu().clone()
        for key in ("step", "exp_avg", "exp_avg_sq"):
            private[key + "_after"] = optimizer.state[parameter][key].detach().cpu().clone()
    return report, private
