"""CPU/Gloo contract for live three-scale ancestors and detached FIFO state.

Run from the repository with::

    python -m torch.distributed.run --standalone --nproc_per_node=2 \
        -m inter_hierarchy_MN40.parthier_v1.tests.ddp_contract

Both ranks evaluate the complete global64 objective. The production gather
backward SUM must cancel DDP's parameter-gradient averaging, without an extra
world-size correction. Two actual optimizer updates are compared with an
independent single-process/global64 execution, including stale step-0 queue
features at step 1. PointMLP/BN fidelity belongs to the separate base tests.

K20 and every production loss operator are retained. Only the number of draws
per eligible anchor is reduced from the checked default 50 to 3 in this bounded
CPU test; this changes neither gather nor DDP gradient normalization.
"""
from __future__ import annotations

import copy
import json
import os

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP

from ..base_protocol import global_base_losses
from ..feature_queue import PartFeatureQueue
from ..loss import PartHierarchyLoss
from ...hier_proxy_scratch_v5.distributed import gather_without_grad


SEED = 4780
WORLD_SIZE = 2
GLOBAL_BATCH = 64
LOCAL_BATCH = 32
DIM = 8
ATOL = 3e-9
RTOL = 3e-7


class Toy(nn.Module):
    """One shared linear encoder receives all three parts and the whole."""

    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(6, DIM, bias=False)
        self.classifier = nn.Linear(DIM, 40)

    def forward(self, views):
        # The toy's small ball coordinates avoid numerical boundary effects;
        # this is not an additional map in the production backbone.
        features = .2 * torch.tanh(self.encoder(views)) / DIM ** .5
        parts, mu = features[:, :3], features[:, 3]
        return {"mu": mu, "parts": parts, "logits": self.classifier(mu),
                "whole_count": 900, "input_stats": {"toy": True}}


def clone_tensors(mapping):
    return {key: value.detach().cpu().clone() for key, value in mapping.items()}


def assert_state_equal(actual, expected, *, serial_reference=False):
    assert actual.keys() == expected.keys()
    for key in actual:
        if isinstance(actual[key], torch.Tensor):
            tolerance = {"atol": ATOL, "rtol": RTOL} if serial_reference and key == "features" else {"atol": 0, "rtol": 0}
            torch.testing.assert_close(actual[key], expected[key], **tolerance)
        else:
            assert actual[key] == expected[key], key


def assert_nested_close(actual, expected):
    if isinstance(actual, dict):
        assert actual.keys() == expected.keys()
        for key in actual:
            assert_nested_close(actual[key], expected[key])
    elif isinstance(actual, (tuple, list)):
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected):
            assert_nested_close(left, right)
    elif isinstance(actual, float):
        assert abs(actual - expected) <= ATOL + RTOL * abs(expected), (actual, expected)
    else:
        assert actual == expected, (actual, expected)


def new_hierarchy():
    hierarchy = PartHierarchyLoss(queue_size=2048).double()
    assert hierarchy.topk == 20 and hierarchy.t_per_anchor == 50
    assert hierarchy.queue.capacity == 2048
    assert hierarchy.lambda_hier == .5 and hierarchy.lambda_order == .1
    assert hierarchy.radial_margin == .2 and hierarchy.grain_margin == .2
    assert sum(parameter.numel() for parameter in hierarchy.parameters()) == 0
    hierarchy.t_per_anchor = 3
    return hierarchy


def inputs_for_step(step):
    generator = torch.Generator().manual_seed(22000 + step)
    views = torch.randn(GLOBAL_BATCH, 4, 6, generator=generator, dtype=torch.float64)
    # Different tensors for all scales ensure a detached or omitted scale can
    # no longer be hidden by an identical-view analytic shortcut.
    views[:, 0] *= .4
    views[:, 1] *= .7
    views[:, 2] *= .9
    labels = torch.arange(32, dtype=torch.long).repeat_interleave(2)
    ids = torch.arange(GLOBAL_BATCH, dtype=torch.long) + step * 1000
    return views, labels, ids


def current_ancestor_gradient(hierarchy, values, ids, step):
    """Verify actual sample-HIER gradients reach every current part scale.

    Whole features are detached and radial/chain weights are disabled solely
    for this probe. Nonzero part gradients must therefore come from ancestor
    distance/ST paths, not the direct HyCoRe or chain supervision.
    autograd.grad stops at the already gathered part tensor and triggers no
    DDP parameter backward or distributed gather backward.
    """
    saved_weights = hierarchy.lambda_hier, hierarchy.lambda_order
    hierarchy.lambda_hier, hierarchy.lambda_order = 1., 0.
    try:
        probe, stats = hierarchy(values["mu"].detach(), values["parts"],
                                  values["gold"], ids, step=step, seed=8100 + step)
        gradient, = torch.autograd.grad(probe, values["parts"])
    finally:
        hierarchy.lambda_hier, hierarchy.lambda_order = saved_weights
    assert stats["sample"]["noncollision_triplets"] > 0
    assert bool(torch.isfinite(gradient).all())
    norms = [float(gradient[:, scale].norm()) for scale in range(3)]
    assert min(norms) > 1e-12, ("ancestor gradient omitted a part scale", norms)
    return norms


def execute_step(model, hierarchy, optimizer, views, labels, ids, step, *, distributed):
    optimizer.zero_grad(set_to_none=True)
    outputs = model(views)
    assert outputs["parts"].requires_grad and outputs["parts"].grad_fn is not None
    values = global_base_losses(outputs, labels)
    global_ids = gather_without_grad(ids)
    assert values["parts"].shape == (GLOBAL_BATCH, 3, DIM)
    assert global_ids.tolist() == inputs_for_step(step)[2].tolist()
    ancestor_norms = current_ancestor_gradient(hierarchy, values, global_ids, step)
    values["parts"].retain_grad()
    inter, stats = hierarchy(values["mu"], values["parts"], values["gold"],
                             global_ids, step=step, seed=8100 + step)
    assert stats["current_candidates"] == GLOBAL_BATCH * 3
    assert stats["queue_candidates"] == step * GLOBAL_BATCH * 3
    assert stats["candidate_count"] == (step + 1) * GLOBAL_BATCH * 3
    assert stats["c"] == 1. and not stats["extra_cap"] and not stats["inverse_metric_hook"]
    queue_before = hierarchy.queue_state()
    loss = values["base"] + inter
    assert torch.isfinite(loss)
    loss.backward()
    raw = model.module if isinstance(model, DDP) else model
    gradients = clone_tensors({name: parameter.grad for name, parameter in raw.named_parameters()})
    part_gradient = values["parts"].grad.detach().clone()
    assert min(float(part_gradient[:, scale].norm()) for scale in range(3)) > 0
    for parameter in raw.parameters():
        assert parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
        if distributed:
            rank0_gradient = parameter.grad.clone()
            dist.broadcast(rank0_gradient, 0)
            torch.testing.assert_close(parameter.grad, rank0_gradient, atol=ATOL, rtol=RTOL)
    # Queue reads and backward must not modify the previous saved features.
    assert_state_equal(hierarchy.queue_state(), queue_before)
    old_parts = values["parts"].detach().clone()
    norm_before_clip = torch.nn.utils.clip_grad_norm_(raw.parameters(), 1.)
    optimizer.step()
    hierarchy.enqueue(values["parts"], global_ids, step)
    state = hierarchy.queue_state()
    assert state["features"].shape == ((step + 1) * GLOBAL_BATCH * 3, DIM)
    expected_features = torch.cat((queue_before["features"].reshape(-1, DIM),
                                   old_parts.reshape(-1, DIM)), dim=0)
    torch.testing.assert_close(state["features"], expected_features, atol=0, rtol=0)
    torch.testing.assert_close(state["object_ids"][-192:], global_ids.repeat_interleave(3), atol=0, rtol=0)
    torch.testing.assert_close(state["scales"][-192:], torch.arange(3).repeat(64), atol=0, rtol=0)
    assert bool((state["steps"][-192:] == step).all())
    entries = hierarchy.queue.entries()
    assert all(not tensor.requires_grad and tensor.grad_fn is None for tensor in entries.values())
    if distributed:
        replicas = [None] * WORLD_SIZE
        dist.all_gather_object(replicas, state)
        for replica in replicas:
            assert_state_equal(state, replica)
    return {"loss": loss.detach().clone(), "gradients": gradients,
            "parameters": clone_tensors(raw.state_dict()), "part_gradient": part_gradient,
            "queue": state, "stats": copy.deepcopy(stats),
            "norm_before_clip": norm_before_clip.detach().clone(),
            "ancestor_gradient_norms": ancestor_norms}


def full_fifo_contract(hierarchy, *, distributed):
    """Fill beyond 2048, check source/scale order, detach and resume identity."""
    queue = PartFeatureQueue(2048)
    queue.load_state_dict(hierarchy.queue_state())
    generator = torch.Generator().manual_seed(8500)
    incoming = (torch.randn(700, 3, DIM, generator=generator, dtype=torch.float64) * .005).requires_grad_()
    ids = torch.arange(700, dtype=torch.long) + 50000
    expected = incoming.detach().reshape(-1, DIM)[-2048:].clone()
    queue.enqueue(incoming, ids, 2)
    entries = queue.entries()
    assert len(queue) == 2048
    torch.testing.assert_close(entries["features"], expected, atol=0, rtol=0)
    torch.testing.assert_close(entries["object_ids"], ids.repeat_interleave(3)[-2048:], atol=0, rtol=0)
    torch.testing.assert_close(entries["scales"], torch.arange(3).repeat(700)[-2048:], atol=0, rtol=0)
    assert bool((entries["steps"] == 2).all())
    assert not entries["features"].requires_grad and entries["features"].grad_fn is None
    # The FIFO must own its storage, independently of the old graph tensors.
    with torch.no_grad():
        incoming.zero_()
    torch.testing.assert_close(entries["features"], expected, atol=0, rtol=0)
    state = queue.state_dict()
    restored = PartFeatureQueue(2048)
    restored.load_state_dict(state)
    assert_state_equal(restored.state_dict(), state)
    state["features"].zero_()
    torch.testing.assert_close(restored.entries()["features"], expected, atol=0, rtol=0)
    if distributed:
        replicas = [None] * WORLD_SIZE
        dist.all_gather_object(replicas, restored.state_dict())
        for replica in replicas:
            assert_state_equal(restored.state_dict(), replica)


def main():
    torch.set_num_threads(1)
    rank = int(os.environ["RANK"])
    assert int(os.environ["WORLD_SIZE"]) == WORLD_SIZE
    dist.init_process_group("gloo")
    torch.manual_seed(SEED)
    raw = Toy().double()
    initial = clone_tensors(raw.state_dict())
    model = DDP(raw)
    hierarchy = new_hierarchy()
    optimizer = torch.optim.SGD(model.parameters(), lr=.03, momentum=.9, weight_decay=2e-4)
    distributed_records = []
    for step in range(2):
        views, labels, ids = inputs_for_step(step)
        rows = slice(rank * LOCAL_BATCH, (rank + 1) * LOCAL_BATCH)
        distributed_records.append(execute_step(model, hierarchy, optimizer,
            views[rows], labels[rows], ids[rows], step, distributed=True))
    full_fifo_contract(hierarchy, distributed=True)
    dist.barrier()
    dist.destroy_process_group()
    if rank != 0:
        return

    # No process group remains: these are true serial/global64 production
    # gather/loss calls, not a mock replacing the distributed primitives.
    reference = Toy().double()
    reference.load_state_dict(initial)
    reference_hierarchy = new_hierarchy()
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=.03,
                                          momentum=.9, weight_decay=2e-4)
    errors = []
    for step, actual in enumerate(distributed_records):
        expected = execute_step(reference, reference_hierarchy, reference_optimizer,
                                *inputs_for_step(step), step, distributed=False)
        torch.testing.assert_close(actual["loss"], expected["loss"], atol=ATOL, rtol=RTOL)
        torch.testing.assert_close(actual["norm_before_clip"], expected["norm_before_clip"],
                                   atol=ATOL, rtol=RTOL)
        torch.testing.assert_close(actual["part_gradient"], expected["part_gradient"],
                                   atol=ATOL, rtol=RTOL)
        for kind in ("gradients", "parameters"):
            for name, value in actual[kind].items():
                torch.testing.assert_close(value, expected[kind][name], atol=ATOL, rtol=RTOL)
                if kind == "gradients":
                    errors.append(float((value - expected[kind][name]).abs().max()))
        assert_state_equal(actual["queue"], expected["queue"], serial_reference=True)
        assert_nested_close(actual["stats"], expected["stats"])
    full_fifo_contract(reference_hierarchy, distributed=False)
    print(json.dumps({"status": "PARTHIER_V1_GLOBAL64_DDP_GRADIENT_CONTRACT_PASSED",
                      "backend": "gloo", "world_size": WORLD_SIZE, "steps": 2,
                      "global_batch": GLOBAL_BATCH, "topk": 20,
                      "production_draws_per_anchor": 50, "contract_draws_per_anchor": 3,
                      "live_part_scales": 3, "queue_capacity": 2048,
                      "step_queue_candidates": [record["stats"]["queue_candidates"]
                                                for record in distributed_records],
                      "max_parameter_gradient_abs_error": max(errors),
                      "ancestor_gradient_norms": [record["ancestor_gradient_norms"]
                                                   for record in distributed_records]},
                     sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
