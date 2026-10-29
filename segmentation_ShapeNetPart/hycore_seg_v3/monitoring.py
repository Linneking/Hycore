"""Sampled, observational geometry and gradient summaries for Pseg3 v3.

No forward pass, RNG draw, parameter mutation, retain_grad, or .grad assignment
is performed here. Call gradient_summary before the trainer's single backward.
All returned payloads contain scalar aggregates, never activations or raw grads.
"""

import math
from collections.abc import Mapping

import torch


_BALL_NAMES = {"point_exp", "m1", "m2", "m3", "whole_exp", "part_exp"}
_QUANTILES = (("p10", .1), ("p50", .5), ("p90", .9), ("p95", .95), ("p99", .99))


def _scalar(value):
    value = float(value)
    return value if math.isfinite(value) else None


def _norm(tensor):
    # Accumulate norms in double precision; the original tensor is unchanged.
    return torch.linalg.vector_norm(tensor.detach(), ord=2, dim=-1, dtype=torch.float64)


def _stats(values):
    flat = values.detach().reshape(-1).to(dtype=torch.float64)
    finite = torch.isfinite(flat)
    clean = flat[finite]
    result = {"count": flat.numel(), "finite_count": clean.numel(),
              "finite_fraction": float(finite.double().mean()) if flat.numel() else None}
    if not clean.numel():
        result.update({key: None for key in ("mean", "min", "max", *dict(_QUANTILES))})
        return result
    result.update(mean=_scalar(clean.mean()), min=_scalar(clean.min()), max=_scalar(clean.max()))
    quantiles = torch.quantile(clean, clean.new_tensor([q for _, q in _QUANTILES])).cpu().tolist()
    result.update({name: _scalar(value) for (name, _), value in zip(_QUANTILES, quantiles)})
    return result


def projection_radius(dtype, curvature=1.):
    """Geoopt stereographic default: float32 eps=.004, otherwise eps=1e-5.

    This is a diagnostic threshold, not a replacement for Geoopt's projection.
    The v3 protocol uses FP32/c=1. Source policy matches pinned Geoopt math._project.
    """
    if curvature <= 0:
        raise ValueError("Poincare curvature magnitude must be positive")
    eps = 4e-3 if dtype == torch.float32 else 1e-5
    return (1. - eps) / math.sqrt(curvature)


def _depth(radius, curvature):
    scaled = radius * math.sqrt(curvature)
    valid = (scaled >= 0) & (scaled < 1) & torch.isfinite(scaled)
    safe = torch.where(valid, scaled, torch.zeros_like(scaled))
    depth = 2. * torch.atanh(safe) / math.sqrt(curvature)
    return torch.where(valid, depth, torch.full_like(depth, float("nan")))


@torch.no_grad()
def tensor_summary(name, tensor, tangent_input=None, curvature=1., boundary_tolerance=1e-6):
    """Summarize one original graph node without retaining its raw contents.

    For an EXP node, tanh(sqrt(c)*||u||)/sqrt(c) supplies an analytic preproject
    radius. Its threshold crossing is labelled analytic, since it is recomputed
    from the same tangent input, not intercepted from Geoopt's actual project.
    M nodes expose only a postproject near-boundary fraction.
    """
    if not isinstance(tensor, torch.Tensor) or tensor.ndim < 1:
        raise ValueError("Expected a tensor with a final feature dimension")
    if boundary_tolerance < 0:
        raise ValueError("Boundary tolerance must be nonnegative")
    detached = tensor.detach()
    radius = _norm(detached)
    finite = torch.isfinite(detached)
    result = {"shape": list(detached.shape), "dtype": str(detached.dtype),
              "finite": bool(finite.all()),
              "finite_fraction": float(finite.double().mean()) if detached.numel() else None,
              "norm": _stats(radius)}
    if detached.ndim == 3:
        result["point_norm"] = result["norm"]
        result["object_mean_norm"] = _stats(_norm(detached.mean(dim=1, dtype=torch.float64)))
    is_ball = name in _BALL_NAMES or name.endswith("_exp") or tangent_input is not None
    if is_ball:
        maximum = projection_radius(detached.dtype, curvature)
        result.update(
            curvature=float(curvature), projection_radius=maximum,
            boundary_tolerance=float(boundary_tolerance),
            radius=_stats(radius), normalized_radius=_stats(radius / maximum),
            depth=_stats(_depth(radius, curvature)),
            near_boundary_fraction=float((radius >= maximum - boundary_tolerance).double().mean())
            if radius.numel() else None,
            near_boundary_count=int((radius >= maximum - boundary_tolerance).sum()),
            vector_count=radius.numel(),
            outside_numerical_radius_fraction=float((radius > maximum + boundary_tolerance).double().mean())
            if radius.numel() else None,
            projection_event_evidence="postproject_near_boundary_only",
        )
    if tangent_input is not None:
        if tangent_input.shape != tensor.shape:
            raise ValueError("EXP tangent input and output must have equal shapes")
        tangent_norm = _norm(tangent_input)
        pre_radius = torch.tanh(math.sqrt(curvature) * tangent_norm) / math.sqrt(curvature)
        crosses = pre_radius > result["projection_radius"]
        result.update(
            tangent_norm=_stats(tangent_norm),
            preprojection_radius=_stats(pre_radius),
            analytic_projection_count=int(crosses.sum()),
            analytic_projection_rate=float(crosses.double().mean()) if crosses.numel() else None,
            analytic_radial_reduction=_stats((pre_radius - radius).clamp_min(0)),
            projection_event_evidence="analytic_same_input_preproject_radius",
        )
    return result


@torch.no_grad()
def gradient_tensor_summary(gradient, node=None):
    """Separate unused gradients from exact zero, including radial/tangent norms.

    The radial direction is the original node's vector direction. At zero node
    norm it is set to zero, leaving the entire gradient in the tangent residual.
    """
    if gradient is None:
        return {"status": "none", "l2": None, "rms": None, "finite": None,
                "radial_l2": None, "tangential_l2": None,
                "gradient_metric": "ambient_euclidean_coordinates"}
    grad = gradient.detach()
    finite = torch.isfinite(grad)
    total = torch.linalg.vector_norm(grad.reshape(-1), dtype=torch.float64)
    all_finite = bool(finite.all())
    result = {"status": "nonzero" if bool((grad != 0).any()) else "zero",
              "gradient_metric": "ambient_euclidean_coordinates",
              "numel": grad.numel(), "finite": all_finite,
              "finite_fraction": float(finite.double().mean()) if grad.numel() else None,
              "zero_fraction": float((grad == 0).double().mean()) if grad.numel() else None,
              "l2": _scalar(total),
              "rms": _scalar(total / math.sqrt(grad.numel())) if grad.numel() else None,
              "max_abs": _scalar(grad.abs().max()) if grad.numel() else None}
    if grad.ndim >= 1:
        result["vector_gradient_norm"] = _stats(_norm(grad))
    if grad.ndim == 3:
        result["point_gradient_norm"] = result["vector_gradient_norm"]
        result["object_mean_gradient_norm"] = _stats(_norm(grad.mean(dim=1, dtype=torch.float64)))
    if node is not None and node.shape == grad.shape and grad.ndim >= 1:
        # Chunk double-precision decomposition to avoid several full-size
        # [B,N,F] double buffers on an already occupied training GPU.
        rows = grad.reshape(-1, grad.shape[-1])
        locations = node.detach().reshape_as(rows)
        signed_blocks, radial_squares, tangent_squares, undefined = [], [], [], []
        for start in range(0, len(rows), 4096):
            vector = rows[start:start + 4096].to(torch.float64)
            location = locations[start:start + 4096].to(torch.float64)
            location_norm = torch.linalg.vector_norm(location, dim=-1)
            unit = location / location_norm.unsqueeze(-1).clamp_min(1e-30)
            signed = (vector * unit).sum(dim=-1)
            radial = signed.unsqueeze(-1) * unit
            signed_blocks.append(signed)
            radial_squares.append(radial.square().sum())
            tangent_squares.append((vector - radial).square().sum())
            undefined.append((location_norm == 0).sum())
        if signed_blocks:
            result.update(radial_l2=_scalar(torch.stack(radial_squares).sum().sqrt()),
                          tangential_l2=_scalar(torch.stack(tangent_squares).sum().sqrt()),
                          radial_signed=_stats(torch.cat(signed_blocks)),
                          radial_direction_undefined_count=int(torch.stack(undefined).sum()))
    return result


def _named_tensors(parameters):
    items = parameters.items() if isinstance(parameters, Mapping) else parameters
    result = []
    for index, item in enumerate(items):
        if isinstance(item, torch.Tensor):
            name, parameter = str(index), item
        else:
            name, parameter = item
        if not isinstance(parameter, torch.Tensor):
            raise TypeError("Parameter groups must contain tensors or (name, tensor) pairs")
        result.append((str(name), parameter))
    return result


@torch.no_grad()
def _parameter_summary(named, gradients):
    none_count, zero_count, nonzero_count, numel = 0, 0, 0, 0
    squared, finite = None, True
    per_parameter = {}
    for name, parameter in named:
        grad = gradients.get(id(parameter))
        stats = gradient_tensor_summary(grad)
        per_parameter[name] = stats
        if grad is None:
            none_count += 1
            continue
        zero_count += int(stats["status"] == "zero")
        nonzero_count += int(stats["status"] == "nonzero")
        numel += grad.numel()
        value = grad.detach().to(torch.float64).square().sum()
        squared = value if squared is None else squared + value
        finite = finite and stats["finite"]
    total = None if squared is None else squared.sqrt()
    return {"status": "none" if squared is None else ("nonzero" if nonzero_count else "zero"),
            "l2": _scalar(total) if total is not None else None,
            "rms": _scalar(total / math.sqrt(numel)) if numel else None,
            "finite": finite if squared is not None else None,
            "numel_with_gradient": numel, "parameter_tensors": len(named),
            "none_count": none_count, "zero_count": zero_count, "nonzero_count": nonzero_count,
            "parameters": per_parameter}


class GeometryMonitor:
    def __init__(self, curvature=1., boundary_tolerance=1e-6):
        self.curvature = float(curvature)
        self.boundary_tolerance = float(boundary_tolerance)
        projection_radius(torch.float32, self.curvature)
        self.begin(False, epoch=0, step=0)

    def begin(self, enabled, epoch=0, step=0):
        self.enabled = bool(enabled)
        self.epoch, self.step = int(epoch), int(step)
        self.nodes, self._tangents = {}, {}
        return self

    def capture(self, name, tensor, tangent_input=None):
        if self.enabled:
            if not isinstance(tensor, torch.Tensor):
                raise TypeError("Monitor capture requires a tensor")
            self.nodes[str(name)] = tensor
            if tangent_input is not None:
                self._tangents[str(name)] = tangent_input
        return tensor

    def summarize(self):
        return {"enabled": self.enabled, "epoch": self.epoch, "step": self.step,
                "geometry": {name: tensor_summary(name, tensor, self._tangents.get(name),
                                                  self.curvature, self.boundary_tolerance)
                             for name, tensor in self.nodes.items()}}

    def gradient_summary(self, losses, parameter_groups=None):
        result = {"enabled": self.enabled, "epoch": self.epoch, "step": self.step,
                  "gradients": {}}
        if not self.enabled:
            return result
        groups = {name: _named_tensors(parameters)
                  for name, parameters in (parameter_groups or {}).items()}
        unique = {}
        for tensor in [*self.nodes.values(), *(p for named in groups.values() for _, p in named)]:
            if tensor.requires_grad:
                unique[id(tensor)] = tensor
        targets = list(unique.values())
        for name, loss in losses.items():
            if not isinstance(loss, torch.Tensor) or loss.numel() != 1:
                raise ValueError("Each monitored loss must be a scalar tensor")
            grads = (torch.autograd.grad(loss, targets, allow_unused=True, retain_graph=True,
                                         create_graph=False)
                     if targets and loss.requires_grad else [None] * len(targets))
            gradients = {id(tensor): gradient for tensor, gradient in zip(targets, grads)}
            result["gradients"][str(name)] = {
                "loss": _scalar(loss.detach()), "loss_requires_grad": loss.requires_grad,
                "nodes": {node_name: gradient_tensor_summary(gradients.get(id(tensor)), tensor)
                          for node_name, tensor in self.nodes.items()},
                "parameter_groups": {group_name: _parameter_summary(named, gradients)
                                     for group_name, named in groups.items()},
            }
        return result

    def clear(self):
        """Release references to sampled graph nodes after the trainer logs them."""
        self.nodes.clear()
        self._tangents.clear()
