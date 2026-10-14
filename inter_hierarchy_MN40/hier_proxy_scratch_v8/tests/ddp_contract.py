"""CPU Gloo: actual global64 objective and gradients equal a one-process reference."""
from __future__ import annotations
import os
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from ..geometry import to_ball
from ..hier_loss import HIERLoss
from ..base_protocol import global_base_losses
from ...hier_proxy_scratch_v5.distributed import sync_parameter_gradients
from ..parameter_audit import audit_parameter_gradients


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(6, 8, bias=False)
        self.classifier = nn.Linear(8, 40)

    def forward(self, x):
        tangent = self.encoder(x)
        mu, nu = to_ball(tangent), to_ball(tangent * .4)
        return {'mu': mu, 'nu': nu, 'logits': self.classifier(mu),
                'child_count': 400, 'whole_count': 800}


class Wrapped(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = Toy()
    def forward(self, x, *counts):
        return self.backbone(x)


def objective(model, proxy, x, labels):
    values = global_base_losses(model(x), labels)
    loss_hier, _ = proxy(values['mu'], values['gold'], topk=6, t_per_anchor=3,
                         generator=torch.Generator().manual_seed(81), monitor_structure=False)
    return values['base'] + .5 * loss_hier


def main():
    torch.set_num_threads(1)
    rank = int(os.environ['RANK'])
    dist.init_process_group('gloo')
    torch.manual_seed(40)
    raw = Wrapped().double()
    initial = {k: v.clone() for k, v in raw.state_dict().items()}
    model = DDP(raw)
    proxy = HIERLoss(num_proxies=32, dim=8, seed=22).double()
    x = torch.randn(64, 6, generator=torch.Generator().manual_seed(13), dtype=torch.float64)
    labels = torch.arange(32).repeat_interleave(2)
    local_x = x[rank * 32:(rank + 1) * 32]
    local_labels = labels[rank * 32:(rank + 1) * 32]
    audit = audit_parameter_gradients(model, proxy, local_x, local_labels,
        torch.arange(rank * 32, (rank + 1) * 32), 800, 400, topk=6, seed=81, lambda_hier=.5)
    assert audit['BN_buffers_restored'] and audit['RNG_restored']
    loss = objective(model, proxy, x[rank * 32:(rank + 1) * 32], labels[rank * 32:(rank + 1) * 32])
    loss.backward()
    sync_parameter_gradients(proxy.parameters())
    gradients = [p.grad.clone() for p in raw.parameters()]
    proxy_gradient = proxy.tangent_proxies.grad.clone()
    for parameter in raw.parameters():
        ref = parameter.grad.clone()
        dist.broadcast(ref, 0)
        torch.testing.assert_close(parameter.grad, ref, atol=1e-10, rtol=1e-9)
    dist.barrier()
    dist.destroy_process_group()
    if rank == 0:
        reference = Wrapped().double()
        reference.load_state_dict(initial)
        p_ref = HIERLoss(num_proxies=32, dim=8, seed=22).double()
        objective(reference, p_ref, x, labels).backward()
        for actual, parameter in zip(gradients, reference.parameters()):
            torch.testing.assert_close(actual, parameter.grad, atol=2e-9, rtol=2e-7)
        torch.testing.assert_close(proxy_gradient, p_ref.tangent_proxies.grad, atol=2e-9, rtol=2e-7)
        print('V8_GLOBAL64_DDP_AND_PROXY_GRADIENT_CONTRACT_PASSED', flush=True)


if __name__ == '__main__':
    main()
