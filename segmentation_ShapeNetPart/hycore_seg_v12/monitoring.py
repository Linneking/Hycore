"""TAN score diagnostics plus scalar gradient alignment at U.

Only active weighted losses are compared. These are ambient U-coordinate
observations, not the full-parameter RSGD update or a parameter mutation.
"""
import itertools
import math

import torch

from ..hycore_seg_v10.monitoring import GeometryMonitor as _GeometryMonitor


def pair_summary(first, second):
    if first is None or second is None:
        return {"status": "inactive_or_unused", "cosine": None,
                "dot": None, "u_virtual_negative_first_directional_derivative_of_second": None}
    if first.shape != second.shape:
        raise ValueError("Gradient pair shapes must match")
    a, b = first.detach().reshape(-1), second.detach().reshape(-1)
    terms = torch.zeros(3, dtype=torch.float64, device=a.device)
    for start in range(0, a.numel(), 1048576):
        x = a[start:start + 1048576].double()
        y = b[start:start + 1048576].double()
        terms += torch.stack(((x * y).sum(), x.square().sum(), y.square().sum()))
    dot, square_a, square_b = terms.cpu().tolist()
    if not all(math.isfinite(v) for v in (dot, square_a, square_b)):
        return {"status": "nonfinite", "cosine": None, "dot": None,
                "u_virtual_negative_first_directional_derivative_of_second": None}
    denom = math.sqrt(square_a) * math.sqrt(square_b)
    return {"status": "nonzero" if denom else "zero_norm",
            "cosine": max(-1., min(1., dot / denom)) if denom else None,
            "dot": dot,
            "u_virtual_negative_first_directional_derivative_of_second": -dot,
            "first_l2": math.sqrt(square_a), "second_l2": math.sqrt(square_b)}


class GeometryMonitor(_GeometryMonitor):
    def gradient_summary(self, losses, parameter_groups=None):
        result = super().gradient_summary(losses, parameter_groups)
        if not self.enabled:
            return result
        node = self.nodes.get("h")
        gradients = {}
        for name, loss in losses.items():
            gradient = (torch.autograd.grad(loss, node, allow_unused=True,
                                            retain_graph=True, create_graph=False)[0]
                        if node is not None and node.requires_grad and loss.requires_grad else None)
            gradients[name] = gradient.detach() if gradient is not None else None
        result["gradient_pairs_at_u"] = {
            "coordinate_space": "ambient_euclidean_U_post_conv9",
            "loss_units": "CE5_over5_and_active_weighted_regularizers",
            "interpretation": "virtual_U_gradient_directions_not_full_parameter_RSGD_updates",
            "pairs": {str(a) + "__" + str(b): pair_summary(gradients[a], gradients[b])
                      for a, b in itertools.combinations(losses, 2)},
        }
        return result
