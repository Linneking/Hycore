"""CPU-only local Jacobian checks; no data, full model, or optimizer."""

import json
from pathlib import Path

import torch
import geoopt

from segmentation_ShapeNetPart.hycore_seg_v2.native import PoincareBall, MobiusLayer
from segmentation_ShapeNetPart.hycore_seg_v3.losses import relation_losses


def scalar(x):
    return float(x.detach().double())


def main():
    assert not torch.cuda.is_initialized()
    result = {"torch": torch.__version__, "geoopt": geoopt.__version__,
              "scope": "local CPU graph only; no full-model or checkpoint measurement",
              "depth_composition": [], "seg_hinge": [], "mobius_after_exp": []}
    original_dtype = torch.get_default_dtype()
    for dtype in (torch.float32, torch.float64):
        torch.set_default_dtype(dtype)
        ball = PoincareBall(c=1., dim=3).to(dtype=dtype)
        direction = torch.tensor([.6, .8, 0.], dtype=dtype)
        for radius in (1., 3., 4., 10.):
            for project in (False, True):
                u = (radius * direction).detach().requires_grad_(True)
                z = ball.expmap0(u, project=project)
                depth = ball.dist0(z)
                grad, = torch.autograd.grad(depth, u)
                result["depth_composition"].append({
                    "dtype": str(dtype), "tangent_radius": radius, "project": project,
                    "output_radius": scalar(z.norm()), "depth": scalar(depth),
                    "signed_radial_gradient": scalar(grad @ direction),
                    "gradient_l2": scalar(grad.norm())})
        for whole_r, part_r in ((1., 1.2), (10.5, 10.8)):
            whole_u = torch.tensor([[whole_r, 0., 0.]], dtype=dtype, requires_grad=True)
            part_u = torch.tensor([[0., part_r, 0.]], dtype=dtype, requires_grad=True)
            whole, part = ball.expmap0(whole_u), ball.expmap0(part_u)
            radial, _, _ = relation_losses(ball, whole, part, 2., 4., torch.tensor([0]))
            weighted = .01 * radial
            gw, gp, guw, gup = torch.autograd.grad(weighted, (whole, part, whole_u, part_u))
            result["seg_hinge"].append({
                "dtype": str(dtype), "whole_tangent_radius": whole_r,
                "part_tangent_radius": part_r, "weighted_radial_loss": scalar(weighted),
                "whole_output_gradient_dot_outward": scalar((gw * whole / whole.norm()).sum()),
                "part_output_gradient_dot_outward": scalar((gp * part / part.norm()).sum()),
                "whole_tangent_gradient_l2": scalar(guw.norm()),
                "part_tangent_gradient_l2": scalar(gup.norm())})
        torch.manual_seed(4780)
        layer = MobiusLayer(3, 2, ball).to(dtype=dtype)
        u = (10. * direction).detach().requires_grad_(True)
        z = ball.expmap0(u)
        mu = layer(z)
        loss = ball.dist0(mu)
        gu, gw = torch.autograd.grad(loss, (u, layer._weight))
        radial = (gu * direction).sum()
        tangential = gu - radial * direction
        result["mobius_after_exp"].append({
            "dtype": str(dtype), "input_radius": scalar(z.norm()),
            "mobius_output_radius": scalar(mu.norm()), "loss": scalar(loss),
            "u_gradient_l2": scalar(gu.norm()),
            "u_gradient_radial": scalar(radial),
            "u_gradient_tangential_l2": scalar(tangential.norm()),
            "weight_gradient_l2": scalar(gw.norm())})
    torch.set_default_dtype(original_dtype)
    result["cuda_initialized"] = torch.cuda.is_initialized()
    path = Path(__file__).with_name("local_geometry_probe.json")
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
