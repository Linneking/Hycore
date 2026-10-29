"""CPU whole/part radial hinge with one shared Mobius mapping; no training."""

import json
from pathlib import Path

import torch
import torch.nn.functional as F
import geoopt

from segmentation_ShapeNetPart.hycore_seg_v2.native import PoincareBall, MobiusLayer


def norm(x):
    return float(torch.linalg.vector_norm(x.detach(), dtype=torch.float64))


def split_gradient(gradient, location):
    unit = location.detach() / location.detach().norm(dim=-1, keepdim=True)
    radial = (gradient * unit).sum(dim=-1, keepdim=True) * unit
    return {"l2": norm(gradient), "radial_l2": norm(radial),
            "tangential_l2": norm(gradient - radial)}


def run_case(kind):
    torch.manual_seed(4780)
    width = 512 if kind == "official_512_to_256_initial_layer" else 2
    ball = PoincareBall(c=1., dim=width)
    if width == 2:
        directions = torch.tensor([[.8, .6], [.6, .8]])
    else:
        generator = torch.Generator().manual_seed(23901)
        directions = F.normalize(F.leaky_relu(torch.randn(2, width, generator=generator), .2), dim=-1)
    u = (directions * torch.tensor([[10.5], [10.8]])).requires_grad_(True)
    x = ball.expmap0(u)
    parameters = []
    if kind == "direct_exp":
        mu = x
    elif kind == "official_512_to_256_initial_layer":
        layer = MobiusLayer(512, 256, ball)
        mu = layer(x)
        parameters = [layer._weight, layer._bias]
    else:
        diagonal = [.2, .4] if kind == "anisotropic_interior_output" else [2., 2.]
        matrix = torch.diag(torch.tensor(diagonal)).requires_grad_(True)
        mu = ball.mobius_matvec(matrix, x)
        parameters = [matrix]
    depths = ball.dist0(mu)
    # Original classification orientation; both branches share the same mapping.
    hinge = depths[1] - depths[0] + 1000. / 400.
    loss = .01 * torch.relu(hinge)
    gu, gx, gmu, *gp = torch.autograd.grad(loss, (u, x, mu, *parameters))
    result = {"case": kind, "u_norm": u.detach().norm(dim=-1).tolist(),
              "x_radius": x.detach().norm(dim=-1).tolist(),
              "mu_radius": mu.detach().norm(dim=-1).tolist(),
              "depth_whole_part": depths.detach().tolist(), "weighted_hinge": float(loss.detach()),
              "H_gradient": split_gradient(gu, u), "X_gradient": split_gradient(gx, x),
              "mu_gradient": split_gradient(gmu, mu),
              "parameter_gradient_l2": [norm(g) for g in gp]}
    assert hinge > 0
    assert result["H_gradient"]["radial_l2"] < 1e-5
    if kind == "direct_exp":
        assert norm(gu) < 1e-6
    elif kind == "anisotropic_interior_output":
        assert norm(gu) > 1e-5 and norm(gp[0]) > 1e-3
        assert (mu.detach().norm(dim=-1) < .99).all()
    elif kind == "output_also_projected":
        assert norm(gu) < 1e-6 and norm(gp[0]) < 1e-5
    else:
        assert norm(gp[0]) > 1e-3 and norm(gu) > 1e-5
    return result


def main():
    assert not torch.cuda.is_initialized()
    result = {"scope": "controlled CPU pair probe, not a trained DGCNN checkpoint or full network",
              "torch": torch.__version__, "geoopt": geoopt.__version__,
              "gamma": 2.5, "beta": .01, "seed": 4780,
              "pair_order": ["whole", "part"], "cases": []}
    for kind in ("direct_exp", "anisotropic_interior_output", "output_also_projected",
                 "official_512_to_256_initial_layer"):
        result["cases"].append(run_case(kind))
    result["cuda_initialized"] = torch.cuda.is_initialized()
    assert not result["cuda_initialized"]
    Path(__file__).with_suffix(".json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
