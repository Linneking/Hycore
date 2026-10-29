"""CPU operator inspection: actual inherited methods and local backward behavior."""

import importlib.util
import inspect
import json
from pathlib import Path

import geoopt
import torch
from geoopt.manifolds.stereographic import math as gmath

from segmentation_ShapeNetPart.hycore_seg_v2.native import PoincareBall


def main():
    assert not torch.cuda.is_initialized()
    ball = PoincareBall(c=1., dim=2)
    result = {"scope": "local CPU operators only; no full model, data, checkpoint or optimizer",
              "torch": torch.__version__, "geoopt": geoopt.__version__,
              "method_identity": {}, "projection": [], "exp_depth": []}
    methods = ("expmap0", "projx", "mobius_matvec", "dist0", "dist", "transp0", "egrad2rgrad", "inner")
    for name in methods:
        active = getattr(type(ball), name)
        inherited = active is getattr(geoopt.PoincareBall, name)
        result["method_identity"][name] = {
            "same_function_as_geoopt": inherited, "module": active.__module__,
            "qualname": active.__qualname__, "line": inspect.getsourcelines(active)[1]}
        assert inherited == (name != "inner")

    for values in ((1.2, 0.), (.8, .6)):
        x = torch.tensor(values, dtype=torch.float32, requires_grad=True)
        jac = torch.autograd.functional.jacobian(ball.projx, x)
        unit = x.detach() / x.detach().norm()
        tangent = torch.stack((-unit[1], unit[0]))
        radial_product, tangent_product = jac @ unit, jac @ tangent
        result["projection"].append({
            "input": list(values), "output": ball.projx(x).detach().tolist(),
            "jacobian": jac.tolist(), "jacobian_times_radial": radial_product.tolist(),
            "jacobian_times_tangent": tangent_product.tolist()})
        assert radial_product.norm() < 1e-5 and tangent_product.norm() > .5

    for project in (False, True):
        u = torch.tensor([4., 0.], dtype=torch.float32, requires_grad=True)
        z = ball.expmap0(u, project=project)
        d = ball.dist0(z)
        gz, gu = torch.autograd.grad(d, (z, u))
        result["exp_depth"].append({
            "project": project, "output": z.detach().tolist(), "depth": float(d.detach()),
            "depth_gradient_at_output": gz.tolist(), "depth_gradient_at_input": gu.tolist()})
        assert (gu.norm() < 1e-5) if project else (gu.norm() > 1.9)

    original = Path(__file__).parents[3] / "classification_ModelNet40/models/ops/pmath.py"
    spec = importlib.util.spec_from_file_location("audit_only_unused_hycore_pmath", original)
    pmath = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pmath)
    result["artanh_outside_clamp"] = {}
    for name, function in (("active_geoopt", gmath.artanh), ("unused_hycore_pmath", pmath.artanh)):
        x = torch.tensor(1.0001, dtype=torch.float64, requires_grad=True)
        y = function(x)
        grad, = torch.autograd.grad(y, x)
        result["artanh_outside_clamp"][name] = {"input": float(x.detach()),
            "forward": float(y.detach()), "gradient": float(grad)}
    assert result["artanh_outside_clamp"]["active_geoopt"]["gradient"] == 0.
    assert result["artanh_outside_clamp"]["unused_hycore_pmath"]["gradient"] > 1e4
    result["cuda_initialized"] = torch.cuda.is_initialized()
    assert not result["cuda_initialized"]
    Path(__file__).with_suffix(".json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
