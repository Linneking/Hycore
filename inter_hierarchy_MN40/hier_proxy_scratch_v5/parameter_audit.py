"""Read-only global parameter-gradient audit for a bounded DDP diagnostic.

A separate no-sync forward is discarded. Working BN buffers are restored,
and callers reset RNG before the real optimizer forward. No .grad buffers
or optimizer states are changed. Autograd gather SUM plus explicit rank
gradient averaging reproduces the single global objective's gradient.
"""
from __future__ import annotations

import torch
import torch.distributed as dist

from .base_protocol import global_base_losses
from .distributed import gather_without_grad


def _comparison(a, b):
    denominator = a.norm() * b.norm()
    return {"norm_ratio": float(a.norm() / b.norm()) if float(b.norm()) else None,
            "cosine": float(torch.dot(a, b) / denominator) if float(denominator) else None}


def audit_parameter_gradients(model, proxy, cloud, labels, ids,
                              whole_count, child_count, *, topk, seed,
                              lambda_hier=.5):
    backbone = model.module.backbone
    named = [(name, parameter) for name, parameter in backbone.named_parameters()
             if parameter.requires_grad]
    parameters = [parameter for _, parameter in named]
    saved_buffers = [(buffer, buffer.detach().clone()) for buffer in backbone.buffers()]
    flat_groups = {"all_model": [], "shared_encoder": [], "euclidean_features": [],
                   "mobius_embedding": [], "classifier": []}
    offset = 0
    for name, parameter in named:
        positions = torch.arange(offset, offset + parameter.numel(), device=cloud.device)
        flat_groups["all_model"].append(positions)
        if name.startswith("classifier."):
            flat_groups["classifier"].append(positions)
        else:
            flat_groups["shared_encoder"].append(positions)
            group = "mobius_embedding" if name.startswith("emb.") else "euclidean_features"
            flat_groups[group].append(positions)
        offset += parameter.numel()
    flat_groups = {key: torch.cat(value) for key, value in flat_groups.items() if value}
    vectors = {}
    try:
        with model.no_sync():
            output = model(cloud, whole_count, child_count)
            values = global_base_losses(output, labels)
            global_ids = gather_without_grad(ids)
            loss_hier, _ = proxy(values["mu"], values["gold"], topk=topk, seed=seed,
                data_ids=global_ids, generator=torch.Generator(device=cloud.device).manual_seed(seed))
            objectives = {"ce": values["ce"], "weighted_intra": values["base"] - values["ce"],
                          "weighted_hier": lambda_hier * loss_hier}
            for name, loss in objectives.items():
                gradients = torch.autograd.grad(loss, parameters, retain_graph=True,
                                                allow_unused=True)
                vector = torch.cat([torch.zeros_like(parameter).reshape(-1) if gradient is None
                                    else gradient.detach().reshape(-1)
                                    for parameter, gradient in zip(parameters, gradients)])
                dist.all_reduce(vector, op=dist.ReduceOp.SUM)
                vector.div_(dist.get_world_size())
                if not torch.isfinite(vector).all():
                    raise RuntimeError("Nonfinite parameter-component gradient")
                vectors[name] = vector
            if any(parameter.grad is not None for parameter in backbone.parameters()):
                raise RuntimeError("Read-only gradient audit accumulated parameter gradients")
        report = {"semantics": "Global64 partial gradients averaged across ranks; no clipping or optimizer step.",
                  "lambda_hier": lambda_hier, "topk": topk,
                  "BN_buffers_restored": True, "groups": {}}
        for group, positions in flat_groups.items():
            ce, intra, hier = (vectors[key][positions] for key in
                               ("ce", "weighted_intra", "weighted_hier"))
            report["groups"][group] = {
                "parameter_elements": len(positions), "ce_norm": float(ce.norm()),
                "weighted_intra_norm": float(intra.norm()), "weighted_hier_norm": float(hier.norm()),
                "hier_vs_ce": _comparison(hier, ce), "hier_vs_intra": _comparison(hier, intra),
                "hier_vs_base": _comparison(hier, ce + intra),
                "base_norm": float((ce + intra).norm()),
                "combined_preclip_norm": float((ce + intra + hier).norm()),
                "fixed_gradient_weight_scenarios": [
                    {"lambda_hier": weight,
                     "hier_vs_base": _comparison(hier * (weight / lambda_hier), ce + intra),
                     "combined_norm": float((ce + intra + hier * (weight / lambda_hier)).norm()),
                     "combined_vs_base": _comparison(ce + intra + hier * (weight / lambda_hier), ce + intra)}
                    for weight in (.5, .1, .05)]}
        return report
    finally:
        with torch.no_grad():
            for buffer, original in saved_buffers:
                buffer.copy_(original)
            # A no_sync forward can suppress the next automatic DDP buffer
            # broadcast. Restore the rank0 buffer state expected immediately
            # before the real optimizer forward as well.
            for buffer in backbone.buffers():
                dist.broadcast(buffer, src=0)
