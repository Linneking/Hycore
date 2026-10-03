"""Distributed primitives for one global objective over both rank batches.

Every rank evaluates the same global loss. The autograd-aware all-gather
SUMs feature gradients in backward; DDP subsequently averages parameter
gradients. These two operations give the single global-batch parameter
gradient without an extra division or multiplication by world size.
"""
from __future__ import annotations

import random

import torch
import torch.distributed as dist
from torch.distributed.nn.functional import all_gather as autograd_all_gather


def distributed_active():
    return dist.is_available() and dist.is_initialized()


def world_size():
    return dist.get_world_size() if distributed_active() else 1


def rank():
    return dist.get_rank() if distributed_active() else 0


def gather_with_grad(value):
    """Concatenate equal-sized local tensors in global rank order."""
    if not distributed_active():
        return value
    return torch.cat(autograd_all_gather(value.contiguous()), dim=0)


@torch.no_grad()
def gather_without_grad(value):
    if not distributed_active():
        return value
    pieces = [torch.empty_like(value) for _ in range(world_size())]
    dist.all_gather(pieces, value.contiguous())
    return torch.cat(pieces, dim=0)


@torch.no_grad()
def sync_parameter_gradients(parameters):
    """Average explicitly replicated proxy gradients outside DDP.

    The replicas must take identical control-flow branches. Every parameter
    must either have a gradient on all ranks or have none on all ranks.
    """
    if not distributed_active():
        return
    for parameter in parameters:
        present = torch.tensor(int(parameter.grad is not None), device=parameter.device)
        dist.all_reduce(present, op=dist.ReduceOp.SUM)
        if int(present) == 0:
            continue
        if int(present) != world_size():
            raise RuntimeError("proxy gradient exists on only some ranks")
        dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
        parameter.grad.div_(world_size())


@torch.no_grad()
def broadcast_parameters(module, source=0):
    if distributed_active():
        for value in module.state_dict().values():
            dist.broadcast(value, src=source)


def shared_crop_counts(seed, epoch, step):
    """One global step uses the same original whole/part point counts.

    A private deterministic Python RNG avoids consuming either rank's
    augmentation/center-selection stream. No communication/device is needed.
    Counts follow the original inclusive 800..1024 and 200..600 uniforms.
    """
    crop_rng = random.Random(int(seed) + 1000003 * int(epoch) + 1009 * int(step))
    return crop_rng.randint(800, 1024), crop_rng.randint(200, 600)
