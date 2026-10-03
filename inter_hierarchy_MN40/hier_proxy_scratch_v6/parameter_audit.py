"""Read-only global parameter-gradient audit for a bounded DDP diagnostic.

A separate no-sync forward is discarded. Working BN buffers are restored,
and callers reset RNG before the real optimizer forward. No .grad buffers
or optimizer states are changed. Autograd gather SUM plus explicit rank
gradient averaging reproduces the single global objective's gradient.
"""
from __future__ import annotations

import random
import numpy as np
import torch
import torch.distributed as dist

from ..hier_proxy_scratch_v5.base_protocol import global_base_losses
from ..hier_proxy_scratch_v5.distributed import gather_without_grad
from .structure_monitor import radial_angular_gradient


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
    saved_rng = (random.getstate(), np.random.get_state(), torch.get_rng_state(),
                 torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None)
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
            # Original HyCoRe crops alias and overwrite their input. The
            # discarded audit must not modify the real optimizer's cloud.
            output = model(cloud.clone(), whole_count, child_count)
            values = global_base_losses(output, labels)
            global_ids = gather_without_grad(ids)
            loss_hier, hierarchy_stats = proxy(values["mu"], values["gold"], topk=topk, seed=seed,
                data_ids=global_ids, exclude_self_negative=True,
                generator=torch.Generator(device=cloud.device).manual_seed(seed),
                return_components=True, monitor_structure=False)
            components = hierarchy_stats.pop("_components")
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
            proxy_components = {}
            proxy_gradients = {}
            for name in ("sample", "proxy"):
                gradient, = torch.autograd.grad(lambda_hier * components[f"{name}_loss"],
                    (proxy.tangent_proxies,), retain_graph=True, allow_unused=True)
                gradient = torch.zeros_like(proxy.tangent_proxies) if gradient is None else gradient.detach()
                dist.all_reduce(gradient, op=dist.ReduceOp.SUM)
                gradient.div_(dist.get_world_size())
                if not torch.isfinite(gradient).all():
                    raise RuntimeError("Nonfinite proxy-component gradient")
                proxy_gradients[name] = gradient
                proxy_components[name] = radial_angular_gradient(proxy.tangent_proxies, gradient)
                detail = components[f"{name}_details"]
                self_mask = detail["self_k_mask"]
                if bool(self_mask.any()):
                    raise RuntimeError("V6 gradient audit found an excluded self-k triplet")
                # Differentiate the actual empty self-k contribution. This is
                # zero because no such draws exist, not because they are solved.
                self_loss = detail["per_draw_loss"][self_mask].sum() / max(1, len(self_mask))
                self_gradient, = torch.autograd.grad(lambda_hier * self_loss,
                    (proxy.tangent_proxies,), retain_graph=True, allow_unused=True)
                self_gradient = torch.zeros_like(proxy.tangent_proxies) if self_gradient is None else self_gradient.detach()
                dist.all_reduce(self_gradient, op=dist.ReduceOp.SUM)
                self_gradient.div_(dist.get_world_size())
                if not bool(torch.isfinite(self_gradient).all()) or float(self_gradient.norm()) != 0.:
                    raise RuntimeError("Excluded V6 self-k contribution has nonzero/nonfinite gradient")
                proxy_components[name]["self_k"] = {
                    "excluded": True, "draw_count": int(self_mask.sum()),
                    "actual_gradient_norm": float(self_gradient.norm()),
                    "semantics": "No self-k draws are constructed; empty actual contribution has zero gradient."}
            proxy_components["combined"] = radial_angular_gradient(
                proxy.tangent_proxies, proxy_gradients["sample"] + proxy_gradients["proxy"])
            if any(parameter.grad is not None for parameter in backbone.parameters()):
                raise RuntimeError("Read-only gradient audit accumulated parameter gradients")
            if proxy.tangent_proxies.grad is not None:
                raise RuntimeError("Read-only gradient audit accumulated proxy gradients")
        report = {"semantics": "Global64 partial gradients averaged across ranks; no clipping or optimizer step.",
                  "lambda_hier": lambda_hier, "topk": topk,
                  "BN_buffers_restored": True, "RNG_restored": True,
                  "exclude_self_negative": True, "proxy_parameter_components": proxy_components,
                  "groups": {}}
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
        random.setstate(saved_rng[0])
        np.random.set_state(saved_rng[1])
        torch.set_rng_state(saved_rng[2])
        if saved_rng[3] is not None:
            torch.cuda.set_rng_state_all(saved_rng[3])
