"""Guard against accidentally optimizing HyCoRe's shared ball curvature."""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import geoopt
import torch

from hier_proxy_v2_online.train import (
    assert_manifold_contract, freeze_euclidean_backbone,
)
from models.pointmlp import Hype_pointMLP


def test_curvature_remains_fixed_after_one_optimizer_step():
    model = Hype_pointMLP()
    freeze_euclidean_backbone(model)
    assert not model.manifold.isp_c.requires_grad
    assert not model.manifold2.isp_c.requires_grad
    trainable = [parameter for parameter in model.parameters()
                 if parameter.requires_grad]
    assert trainable
    optimizer = geoopt.optim.RiemannianSGD(trainable, lr=1e-4)
    assert_manifold_contract(model, optimizer)
    before = (float(model.manifold.c), float(model.manifold2.c))
    loss = sum(parameter.square().mean() for parameter in trainable)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    assert_manifold_contract(model, optimizer)
    after = (float(model.manifold.c), float(model.manifold2.c))
    assert before == after
    assert all(abs(value - 1.0) < 1e-6 for value in after)


if __name__ == "__main__":
    test_curvature_remains_fixed_after_one_optimizer_step()
    print("PASS test_curvature_remains_fixed_after_one_optimizer_step")
