"""CPU Gloo: native-only V9 B64 gradients match one global64 reference.

This runs the actual production control loss/gather route.  Both encoders
receive gradients from CE and the two native c1 HyCoRe terms; no hierarchy
proxy, fixed-shell representation or HIER gradient hook is constructed.
"""
from __future__ import annotations

import os

from geoopt import PoincareBall
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP

from ..controls import control_base_losses


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(6, 8, bias=False)
        self.part_encoder = nn.Linear(6, 8, bias=False)
        self.classifier = nn.Linear(8, 40)

    def forward(self, inputs):
        tangent = self.encoder(inputs) * .35
        ball = PoincareBall(c=1.).to(tangent)
        mu = ball.expmap0(tangent)
        nu = ball.expmap0(self.part_encoder(inputs) * .12)
        return {"mu": mu, "nu": nu, "logits": self.classifier(mu),
                "whole_count": 900, "child_count": 400}


def objective(model, inputs, labels):
    return control_base_losses(model(inputs), labels, "B64")["base"]


def main():
    torch.set_num_threads(1)
    rank = int(os.environ["RANK"])
    dist.init_process_group("gloo")
    if dist.get_world_size() != 2:
        raise RuntimeError("Contract requires exactly two CPU ranks")
    torch.manual_seed(40)
    raw = Toy().double()
    initial = {key: value.clone() for key, value in raw.state_dict().items()}
    model = DDP(raw)
    inputs = torch.randn(64, 6, generator=torch.Generator().manual_seed(13), dtype=torch.float64)
    labels = torch.arange(32).repeat_interleave(2)
    rank_slice = slice(rank * 32, (rank + 1) * 32)
    loss = objective(model, inputs[rank_slice], labels[rank_slice])
    assert bool(torch.isfinite(loss))
    loss.backward()
    gradients = [parameter.grad.clone() for parameter in raw.parameters()]
    for parameter in raw.parameters():
        assert parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
        assert float(parameter.grad.norm()) > 0
        reference = parameter.grad.clone()
        dist.broadcast(reference, 0)
        torch.testing.assert_close(parameter.grad, reference, rtol=1e-9, atol=1e-10)
    dist.barrier()
    dist.destroy_process_group()
    if rank == 0:
        reference = Toy().double()
        reference.load_state_dict(initial, strict=True)
        reference_loss = objective(reference, inputs, labels)
        torch.testing.assert_close(loss, reference_loss, rtol=1e-12, atol=1e-12)
        reference_loss.backward()
        for actual, parameter in zip(gradients, reference.parameters()):
            torch.testing.assert_close(actual, parameter.grad, rtol=2e-7, atol=2e-9)
        print("V9_NATIVE_ONLY_GLOBAL64_DDP_GRADIENT_CONTRACT_PASSED", flush=True)


if __name__ == "__main__":
    main()
