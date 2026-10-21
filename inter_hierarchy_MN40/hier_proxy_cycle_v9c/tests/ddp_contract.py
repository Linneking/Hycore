"""CPU/Gloo: base-GS plus pre-GS shell-GS matches a single global64 loss.

Each rank holds16 distinct classes x2 rows. Autograd gather backward SUM
cancels DDP parameter-gradient averaging; replicated proxy gradients are
explicitly averaged. No CUDA context or image/model extension is required.
"""
from __future__ import annotations

import os
from unittest import mock

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP

from ..base_protocol import global_base_losses
from ..geometry import C, InverseMetric, to_ball, whole_to_hier
from ..parameter_audit import audit_parameter_gradients
from ...hier_proxy_scratch_v8.hier_loss import HIERLoss
from ...hier_proxy_scratch_v5.distributed import sync_parameter_gradients


class Toy(nn.Module):
    def __init__(self, hier_enabled):
        super().__init__()
        self.hier_enabled = bool(hier_enabled)
        self.encoder = nn.Linear(6, 8, bias=False)
        self.classifier = nn.Linear(8, 40)

    def forward(self, x):
        tangent = self.encoder(x)
        source = to_ball(tangent, c=C, hook=False)
        mu = InverseMetric.apply(source, C)
        nu = to_ball(tangent * .4, c=C, hook=True)
        result = {"mu": mu, "nu": nu, "logits": self.classifier(mu),
                  "child_count": 400, "whole_count": 800}
        if self.hier_enabled:
            result["hier_source_mu"] = source
        return result


class Wrapped(nn.Module):
    def __init__(self, hier_enabled):
        super().__init__()
        self.backbone = Toy(hier_enabled)

    def forward(self, x, *_counts):
        return self.backbone(x)


def objective(model, proxy, x, labels):
    values = global_base_losses(model(x), labels)
    if proxy is None:
        if "hier_source_mu" in values:
            raise AssertionError("No-HIER control constructed a HIER source")
        return values["base"]
    hierarchy, _ = proxy(whole_to_hier(values["hier_source_mu"]), values["gold"],
        topk=6, t_per_anchor=3, seed=81, exclude_self_negative=True,
        generator=torch.Generator().manual_seed(81), monitor_structure=False)
    return values["base"] + .5 * hierarchy


def main():
    torch.set_num_threads(1)
    rank = int(os.environ["RANK"])
    if int(os.environ["WORLD_SIZE"]) != 2:
        raise RuntimeError("Exactly two CPU Gloo ranks are required")
    if torch.cuda.is_initialized():
        raise RuntimeError("CPU contract must not initialize CUDA")
    dist.init_process_group("gloo")
    x = torch.randn(64, 6, generator=torch.Generator().manual_seed(13), dtype=torch.float64)
    labels = torch.arange(32).repeat_interleave(2)
    local_x, local_labels = x[rank * 32:(rank + 1) * 32], labels[rank * 32:(rank + 1) * 32]
    results = []
    for enabled in (False, True):
        torch.manual_seed(40)
        raw = Wrapped(enabled).double()
        initial = {key: value.clone() for key, value in raw.state_dict().items()}
        model = DDP(raw)
        # Off branch never constructs a proxy or a fixed-shell representation.
        proxy = HIERLoss(num_proxies=32, dim=8, seed=4780).double() if enabled else None
        if enabled:
            audit = audit_parameter_gradients(model, proxy, local_x, local_labels,
                torch.arange(rank * 32, (rank + 1) * 32), 800, 400,
                topk=6, seed=81, lambda_hier=.5)
            assert audit["BN_buffers_restored"] and audit["RNG_restored"]
            assert audit["groups"]["all_model"]["weighted_hier_norm"] > 0
            loss = objective(model, proxy, local_x, local_labels)
        else:
            with mock.patch(__name__ + ".whole_to_hier", side_effect=AssertionError("HIER-off shell call")), \
                 mock.patch(__name__ + ".HIERLoss", side_effect=AssertionError("HIER-off proxy construction")):
                loss = objective(model, None, local_x, local_labels)
        loss.backward()
        if enabled:
            sync_parameter_gradients(proxy.parameters())
        gradients = [parameter.grad.detach().clone() for parameter in raw.parameters()]
        for gradient in gradients:
            assert bool(torch.isfinite(gradient).all())
            twin = gradient.clone()
            dist.broadcast(twin, src=0)
            torch.testing.assert_close(gradient, twin, atol=1e-10, rtol=1e-9)
        proxy_gradient = proxy.tangent_proxies.grad.detach().clone() if enabled else None
        results.append((enabled, initial, gradients, proxy_gradient, float(loss.detach())))
        del model, raw, proxy
        dist.barrier()
    dist.destroy_process_group()
    if rank == 0:
        for enabled, initial, gradients, proxy_gradient, distributed_loss in results:
            reference = Wrapped(enabled).double()
            reference.load_state_dict(initial)
            proxy_reference = HIERLoss(num_proxies=32, dim=8, seed=4780).double() if enabled else None
            reference_loss = objective(reference, proxy_reference, x, labels)
            torch.testing.assert_close(reference_loss.detach(), reference_loss.new_tensor(distributed_loss),
                                       atol=1e-10, rtol=1e-9)
            reference_loss.backward()
            for actual, parameter in zip(gradients, reference.parameters()):
                torch.testing.assert_close(actual, parameter.grad, atol=2e-9, rtol=2e-7)
            if enabled:
                torch.testing.assert_close(proxy_gradient, proxy_reference.tangent_proxies.grad,
                                           atol=2e-9, rtol=2e-7)
        assert not torch.cuda.is_initialized()
        print("V9C_GLOBAL64_BASE_GS_PRE_GS_SHELL_GS_AND_PROXY_DDP_GRADIENT_CONTRACT_PASSED", flush=True)


if __name__ == "__main__":
    main()
