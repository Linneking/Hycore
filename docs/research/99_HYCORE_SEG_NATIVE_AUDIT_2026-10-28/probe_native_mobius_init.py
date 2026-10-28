"""CPU-only initialization probe of the original HyCoRe layer definitions.

Usage: python probe_native_mobius_init.py <checkout-root>
Does not construct a backbone, use a GPU, update parameters, or save checkpoints.
"""

import ast
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
import torch
import geoopt
from torch import nn
from torch.nn import init
from torch.nn.parameter import Parameter
from geoopt import ManifoldParameter

torch.set_num_threads(2)
root = Path(sys.argv[1])
ball_rel = "classification_ModelNet40/models/manifolds/poincareball.py"
layer_rel = "classification_ModelNet40/models/ops/manifold_layers.py"
ball_path, layer_path = root / ball_rel, root / layer_rel
spec = importlib.util.spec_from_file_location("_original_hycore_ball_probe", ball_path)
ball_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ball_module)
Ball = ball_module.PoincareBall

# Execute the exact two source class definitions without importing unrelated models.
tree = ast.parse(layer_path.read_text())
classes = [n for n in tree.body if isinstance(n, ast.ClassDef)
           and n.name in {"RiemannianLayer", "MobiusLayer"}]
assert len(classes) == 2
namespace = {"torch": torch, "nn": nn, "init": init, "math": math,
             "Parameter": Parameter, "ManifoldParameter": ManifoldParameter}
exec(compile(ast.Module(body=classes, type_ignores=[]), str(layer_path), "exec"), namespace)
Layer = namespace["MobiusLayer"]
widths = [(256, 256), (256, 128), (128, 50)]


def build(seed):
    torch.manual_seed(seed)
    return nn.ModuleList([Layer(a, b, Ball(dim=a, c=1.0)) for a, b in widths])


def digest(model):
    h = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        h.update(name.encode())
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


layers = build(4780)
initial_digest = digest(layers)
assert initial_digest == digest(build(4780))
initialization = []
for layer, (a, b) in zip(layers, widths):
    bound = 4 / math.sqrt(a)
    assert tuple(layer._weight.shape) == (b, a)
    assert tuple(layer._bias.shape) == (b, 1)
    assert bool(torch.isfinite(layer.weight).all())
    assert float(layer._bias.detach().abs().max()) <= bound
    initialization.append({
        "in_features": a, "out_features": b,
        "raw_weight_std": float(layer._weight.detach().std()),
        "expected_weight_std": 1 / math.sqrt(3 * a),
        "bias_bound": bound,
        "observed_bias_abs_max": float(layer._bias.detach().abs().max()),
        "effective_weight_finite": True,
    })

cases = []
for scale in (.02, 1.0):
    layers.zero_grad(set_to_none=True)
    generator = torch.Generator().manual_seed(4781)
    features = (torch.randn(4, 8, 256, generator=generator) * scale).requires_grad_()
    x = layers[0].manifold.expmap0(features)
    input_ball_norm = x.detach().norm(dim=-1)
    outputs = []
    for layer, (_, b) in zip(layers, widths):
        x = layer(x)
        assert tuple(x.shape) == (4, 8, b)
        assert bool(torch.isfinite(x).all())
        outputs.append({"shape": list(x.shape), "max_ball_norm": float(x.detach().norm(dim=-1).max())})
    labels = torch.arange(32) % 50
    ce = torch.nn.functional.cross_entropy(x.reshape(-1, 50), labels)
    ce.backward()
    assert torch.isfinite(ce) and bool(torch.isfinite(features.grad).all())
    gradients = []
    for layer in layers:
        row = {}
        for name in ("_weight", "_bias"):
            g = getattr(layer, name).grad
            assert g is not None and bool(torch.isfinite(g).all()) and float(g.norm()) > 0
            row[name] = float(g.norm())
        gradients.append(row)
    cases.append({
        "synthetic_feature_std": scale,
        "input_ball_max_norm": float(input_ball_norm.max()),
        "input_near_projection_fraction": float((input_ball_norm >= .996 - 1e-6).float().mean()),
        "outputs": outputs, "ce": float(ce.detach()),
        "input_gradient_norm": float(features.grad.norm()),
        "parameter_gradient_norms": gradients,
        "finite_forward_backward": True,
    })

assert digest(layers) == initial_digest  # backward only: no optimizer update
assert not torch.cuda.is_initialized()
print(json.dumps({
    "passed": True, "scope": "original_layers_only_cpu_synthetic_probe",
    "seed": 4780, "dtype": "float32", "device": "cpu",
    "environment": {"torch": torch.__version__, "geoopt": geoopt.__version__},
    "sources": [{"file": rel, "sha256_lf": hashlib.sha256((root/rel).read_text().replace("\r\n", "\n").encode()).hexdigest()}
                for rel in (ball_rel, layer_rel)],
    "source_loading": "unchanged geometry module and exact source AST class definitions",
    "head_parameters": sum(p.numel() for p in layers.parameters() if p.requires_grad),
    "initial_state_sha256": initial_digest, "same_seed_repeat_identical": True,
    "no_optimizer_update": True, "cuda_initialized": False,
    "initialization": initialization, "cases": cases,
    "not_validated": ["full_backbone_initialization", "real_feature_distribution", "GPU_numerics", "training_convergence", "paper_score"],
}, ensure_ascii=False))
