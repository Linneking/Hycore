"""CPU gates for parameter projection, W0 and the V7 resume contract."""
from pathlib import Path
import argparse
import ast
import math
import random
import sys
from unittest.mock import patch
import numpy as np
import torch

from inter_hierarchy_MN40.hier_proxy_scratch_v7.hier_loss import HIERLoss, expmap0_c1
from inter_hierarchy_MN40.hier_proxy_scratch_v7.proxy_safety import (
    calibrated_initial_depth, project_proxy_parameter, assert_parameter_safe,
)


def train_functions(*names):
    """Read standalone contracts without importing the CUDA point backbone."""
    path = Path(__file__).resolve().parents[1] / "train.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    nodes = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.Assign))
             and (isinstance(node, ast.FunctionDef) and node.name in names or
                  isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and
                  t.id == "CHECKPOINT_VERSION" for t in node.targets))]
    namespace = dict(argparse=argparse, Path=Path, math=math, np=np, torch=torch)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


def test_exact_initial_depth_and_calibrated_inside_rule():
    hierarchy = HIERLoss(num_proxies=32, dim=8, initial_depth=5.)
    assert torch.allclose(hierarchy.tangent_proxies.norm(dim=-1), torch.full((32,), 2.5), atol=1e-6)
    assert torch.allclose(hierarchy.proxies().norm(dim=-1), torch.full((32,), math.tanh(2.5)), atol=1e-6)
    assert calibrated_initial_depth(4.) == 3.6
    assert calibrated_initial_depth(6.) == 5.
    hierarchy.set_initial_depth(3.6)
    assert torch.allclose(hierarchy.tangent_proxies.norm(dim=-1), torch.full((32,), 1.8), atol=1e-6)


def test_projection_preserves_direction_and_adam_state():
    parameter = torch.nn.Parameter(torch.tensor([[4., 0.], [0., 2.]]))
    optimizer = torch.optim.AdamW([parameter], lr=.01)
    first = torch.tensor([[-1., 2.], [3., 4.]])
    second = torch.tensor([[1., 4.], [9., 16.]])
    optimizer.state[parameter] = {"exp_avg": first.clone(), "exp_avg_sq": second.clone()}
    stats = project_proxy_parameter(parameter, optimizer, 6.)
    assert stats["hit_count"] == 1
    assert torch.equal(parameter, torch.tensor([[3., 0.], [0., 2.]]))
    assert torch.equal(optimizer.state[parameter]["exp_avg"], first)
    assert torch.equal(optimizer.state[parameter]["exp_avg_sq"], second)
    assert_parameter_safe(parameter, 6.)


def test_projected_parameter_has_inward_radial_gradient():
    safe = torch.tensor([[3., 0.]], dtype=torch.float64, requires_grad=True)
    flat = torch.tensor([[5., 0.]], dtype=torch.float64, requires_grad=True)
    safe_depth = 2 * torch.atanh(expmap0_c1(safe).norm(dim=-1)).sum()
    flat_depth = 2 * torch.atanh(expmap0_c1(flat).norm(dim=-1)).sum()
    safe_gradient, = torch.autograd.grad(safe_depth, safe)
    flat_gradient, = torch.autograd.grad(flat_depth, flat)
    assert float(safe_gradient[0, 0]) > 1.99
    assert float(flat_gradient.abs().max()) < 1e-8
    parameter = torch.nn.Parameter(safe.detach().clone())
    optimizer = torch.optim.AdamW([parameter], lr=.01, weight_decay=.01)
    (2 * torch.atanh(expmap0_c1(parameter).norm(dim=-1))).sum().backward()
    optimizer.step()
    project_proxy_parameter(parameter, optimizer, 6.)
    assert float(parameter.norm()) < 3.


def test_optional_moment_policy_only_removes_outward_first_moment():
    parameter = torch.nn.Parameter(torch.tensor([[4., 0.], [0., 4.]]))
    optimizer = torch.optim.AdamW([parameter], lr=.01)
    second = torch.ones_like(parameter)
    optimizer.state[parameter] = {"exp_avg": torch.tensor([[-2., 3.], [4., 2.]]),
                                  "exp_avg_sq": second.clone()}
    stats = project_proxy_parameter(parameter, optimizer, 6., "remove_outward")
    assert stats["removed_outward_first_moment_count"] == 1
    assert torch.equal(optimizer.state[parameter]["exp_avg"], torch.tensor([[0., 3.], [4., 2.]]))
    assert torch.equal(optimizer.state[parameter]["exp_avg_sq"], second)


def test_w0_from_first_epoch_and_strict_resume_identity():
    functions = train_functions("arguments", "fixed_configuration", "validate_resume")
    with patch.object(sys, "argv", ["train", "--data-dir", "data", "--run-dir", "run"]):
        args = functions["arguments"]()
    config = functions["fixed_configuration"](args)
    assert args.warmup_epochs == 0 and args.lambda_hier == .1
    assert config["lambda_hier_after_warmup"] == .1
    assert 0 >= args.warmup_epochs  # production first step selects the active branch
    class Sampler:
        def load_state_dict(self, state):
            assert state == {"same": True}
    saved = {"format": functions["CHECKPOINT_VERSION"], "training_config": config,
             "split_sha256": "same", "train_ids": np.array([1, 2]),
             "validation_ids": np.array([3]), "rank_states": [{"sampler": {"same": True}}, {}],
             "completed_epochs": 1}
    functions["validate_resume"](saved, config, "same", np.array([1, 2]), np.array([3]), Sampler())
    for mutation in ({"format": "hycore-hier-v6-h20-selfk300-1"},
                     {"training_config": {**config, "proxy_max_depth": 5.}},
                     {"model_selection_only": True}):
        try:
            functions["validate_resume"]({**saved, **mutation}, config, "same",
                                           np.array([1, 2]), np.array([3]), Sampler())
        except RuntimeError:
            pass
        else:
            raise AssertionError("V7 resume accepted an incompatible checkpoint")
