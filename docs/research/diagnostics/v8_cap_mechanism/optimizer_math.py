"""Read-only radial/angular accounting for Euclidean optimizer parameters.

The parameters here are *raw tangent parameters*, not Poincare-ball coordinates
or hyperbolic depths.  A gradient's radial direction need not match an AdamW
update's radial direction.  Even a perfectly tangential finite update increases
the Euclidean norm to second order.  These functions expose both effects without
changing parameters, gradients, optimizer state, or the autograd graph.

All measurements use detached float64 copies on the input device.  Thus tiny
changes in stored float32 parameters are measured accurately, but rounding that
already happened in those parameters cannot be undone.  The last dimension is
the vector dimension; preceding dimensions index independent vectors.
"""

from typing import Dict, Iterable, Union

import torch


def _vectors(value: torch.Tensor, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or not value.is_floating_point():
        raise TypeError(f"{name} must be a floating-point torch.Tensor")
    if value.ndim == 0 or value.shape[-1] == 0:
        raise ValueError(f"{name} must have a nonempty final vector dimension")
    result = value.detach().to(dtype=torch.float64)
    if not bool(torch.isfinite(result).all()):
        raise ValueError(f"{name} contains nonfinite values")
    return result


def _pair(left: torch.Tensor, right: torch.Tensor, names):
    a, b = _vectors(left, names[0]), _vectors(right, names[1])
    if a.shape != b.shape or a.device != b.device:
        raise ValueError("the two vector tensors must have matching shapes and devices")
    return a, b


def _direction(value: torch.Tensor):
    norm = torch.linalg.vector_norm(value, dim=-1)
    defined = norm > 0
    denominator = torch.where(defined, norm, torch.ones_like(norm))
    return norm, value / denominator.unsqueeze(-1), defined


def _ratio(numerator: torch.Tensor, denominator: torch.Tensor):
    safe = torch.where(denominator > 0, denominator, torch.ones_like(denominator))
    return torch.where(denominator > 0, numerator / safe, torch.zeros_like(numerator))


def decompose_gradient(parameter: torch.Tensor, gradient: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Return per-vector raw-gradient geometry relative to ``parameter``.

    Positive ``signed_radial_gradient`` means the gradient points outward;
    ordinary gradient descent would therefore move inward to first order.
    ``radial_gradient_fraction`` is a norm ratio, not an energy fraction.
    Zero parameters have undefined radial directions (flagged explicitly);
    their radial component is conventionally zero and their gradient is angular.
    """
    p, g = _pair(parameter, gradient, ("parameter", "gradient"))
    radius, direction, defined = _direction(p)
    radial = (g * direction).sum(dim=-1)
    angular = g - radial.unsqueeze(-1) * direction
    gnorm = torch.linalg.vector_norm(g, dim=-1)
    angular_norm = torch.linalg.vector_norm(angular, dim=-1)
    return {
        "parameter_norm": radius,
        "gradient_norm": gnorm,
        "signed_radial_gradient": radial,
        "radial_gradient_norm": radial.abs(),
        "angular_gradient_norm": angular_norm,
        "signed_radial_gradient_fraction": _ratio(radial, gnorm),
        "radial_gradient_fraction": _ratio(radial.abs(), gnorm),
        "angular_gradient_fraction": _ratio(angular_norm, gnorm),
        "direction_defined": defined,
    }


def decompose_update(before: torch.Tensor, after: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Return exact finite-update accounting for each raw parameter vector.

    With r=||before||, u=before/r, delta=after-before, a=<u,delta>,
    and b=||delta-a*u||, the identities are

        ||after|| - r = a + second_order_excess
        ||after||**2 - r**2 = 2*r*a + ||delta||**2.

    ``signed_radial_step`` is a first-order projection, not an exact radius
    change.  ``second_order_excess`` is nonnegative up to roundoff, by convexity
    of the norm.  Its name assumes a nonzero starting vector and a small step;
    at zero use the explicit ``direction_defined`` flag.  Nothing is clipped to
    hide numerical residuals.  Weight decay, momentum and Adam preconditioning
    are all included when ``after`` comes from the real optimizer step.
    """
    old, new = _pair(before, after, ("before", "after"))
    radius, direction, defined = _direction(old)
    new_radius = torch.linalg.vector_norm(new, dim=-1)
    delta = new - old
    radial = (delta * direction).sum(dim=-1)
    angular = delta - radial.unsqueeze(-1) * direction
    step_norm = torch.linalg.vector_norm(delta, dim=-1)
    exact_change = new_radius - radius
    return {
        "before_norm": radius,
        "after_norm": new_radius,
        "step_norm": step_norm,
        "signed_radial_step": radial,
        "angular_step_norm": torch.linalg.vector_norm(angular, dim=-1),
        "signed_radial_step_fraction": _ratio(radial, step_norm),
        "exact_norm_change": exact_change,
        "second_order_excess": exact_change - radial,
        "squared_norm_change": new_radius.square() - radius.square(),
        "squared_first_order_radial": 2 * radius * radial,
        "squared_step": step_norm.square(),
        "direction_defined": defined,
    }


def summarize(values: Union[torch.Tensor, Iterable[float]],
              quantiles=(0., .01, .05, .25, .5, .75, .95, .99, 1.)) -> Dict:
    """Return JSON-safe descriptive statistics, explicitly counting nonfinites.

    Quantiles are linear-interpolated and computed over finite entries only;
    empty/all-nonfinite inputs produce ``None`` statistics.  ``std`` is the
    population standard deviation (correction=0), not a sample estimate.
    """
    if isinstance(values, torch.Tensor):
        vector = values.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
    else:
        vector = torch.as_tensor(list(values), dtype=torch.float64).reshape(-1)
    q = tuple(float(item) for item in quantiles)
    if any(not 0 <= item <= 1 for item in q):
        raise ValueError("quantiles must be between zero and one")
    finite = vector[torch.isfinite(vector)]
    result = {
        "count": int(vector.numel()),
        "finite_count": int(finite.numel()),
        "nonfinite_count": int(vector.numel() - finite.numel()),
        "mean": None,
        "std": None,
        "min": None,
        "max": None,
        "quantiles": {f"q{value:g}": None for value in q},
    }
    if finite.numel():
        result.update(mean=float(finite.mean()), std=float(finite.std(correction=0)),
                      min=float(finite.min()), max=float(finite.max()))
        result["quantiles"] = {
            f"q{value:g}": float(torch.quantile(finite, value)) for value in q
        }
    return result
