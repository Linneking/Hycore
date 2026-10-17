"""CPU-only, one-step AdamW moment interventions on private frozen V8 inputs.

The saved parameter, current gradient, second-moment history, update counter and
optimizer hyperparameters stay matched unless the named intervention explicitly
changes them.  Every step updates a fresh clone through real torch.optim.AdamW.
No source checkpoint, optimizer, encoder, dataset split or production run changes.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from proxy_probe import adam_step, balanced, load_case, loss_parts, rows_for_group, write


VARIANTS = ("saved_native", "zero_m", "tangent_m", "isotropic_v", "tangent_m_isotropic_v")


def intervene(state, parameter, variant):
    """Clone a saved AdamW state and apply exactly the requested moment change."""
    if variant not in VARIANTS:
        raise ValueError(f"Unknown intervention {variant}")
    result = copy.deepcopy(state)
    if len(result["state"]) != 1:
        raise ValueError("Expected exactly one saved proxy parameter tensor")
    saved = next(iter(result["state"].values()))
    if variant == "zero_m":
        saved["exp_avg"].zero_()
    if variant in ("tangent_m", "tangent_m_isotropic_v"):
        p = parameter.detach().double()
        direction = p / p.norm(dim=-1, keepdim=True).clamp_min(torch.finfo(torch.float64).tiny)
        moment = saved["exp_avg"].double()
        moment -= (moment * direction).sum(dim=-1, keepdim=True) * direction
        saved["exp_avg"] = moment.to(dtype=saved["exp_avg"].dtype)
    if variant in ("isotropic_v", "tangent_m_isotropic_v"):
        value = saved["exp_avg_sq"]
        saved["exp_avg_sq"] = value.mean(dim=-1, keepdim=True).expand_as(value).clone()
    return result


def digest_state(state):
    """Deterministically detect mutation of the caller's saved state."""
    out = hashlib.sha256()
    for name, saved in sorted(state["state"].items()):
        out.update(str(name).encode())
        for key, value in sorted(saved.items()):
            out.update(key.encode())
            if isinstance(value, torch.Tensor):
                out.update(value.detach().cpu().contiguous().numpy().tobytes())
            else:
                out.update(repr(value).encode())
    out.update(repr(state["param_groups"]).encode())
    return out.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cases", default="v8_e1,v8_e5,v8_e20,v8_e300")
    parser.add_argument("--seeds", default="22,83")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    args.out.mkdir(parents=True, exist_ok=False)
    started = time.time()
    result = dict(format="v8-cpu-moment-controls-1", status="running", device="cpu",
                  source_updates=0, encoder_forwards=0, test_forwards=0,
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  intervention_definitions={
                      "saved_native": "unchanged saved m, v, step and optimizer hyperparameters",
                      "zero_m": "zero saved exp_avg; preserve v and step",
                      "tangent_m": "project saved exp_avg to current raw-parameter tangent plane; preserve v and step",
                      "isotropic_v": "replace each row of saved exp_avg_sq with its row mean; preserve m and step",
                      "tangent_m_isotropic_v": "combine tangent_m and isotropic_v"},
                  gradient_modes=["current_both", "zero_current"],
                  condition="Frozen clean balanced64, full-ST sample+proxy gradients, native scalar .5 each; CPU FP32",
                  limits="One-step counterfactuals from historical state; not a complete-training causal ablation",
                  rows=[], source_integrity=[])
    write(args.out / "summary.json", result)
    for case in args.cases.split(","):
        parameter, pool, labels, state = load_case(args.inputs, case)
        if state is None:
            raise ValueError(f"No saved optimizer state for {case}")
        original_parameter = parameter.clone()
        original_digest = digest_state(state)
        original_state = copy.deepcopy(state)
        for seed in map(int, args.seeds.split(",")):
            positions = balanced(labels, seed)
            whole = torch.as_tensor(pool[positions], device="cpu")
            gold = torch.as_tensor(labels[positions], dtype=torch.long, device="cpu")
            p = parameter.clone().requires_grad_()
            sample, proxy, stats = loss_parts(p, whole, gold, seed=seed)
            gradient, = torch.autograd.grad(sample + proxy, p)
            base = dict(case=case, batch_seed=seed, component="both",
                        sample_loss=float(sample.detach()), proxy_loss=float(proxy.detach()), **stats)
            for gradient_mode in result["gradient_modes"]:
                used_gradient = gradient if gradient_mode == "current_both" else torch.zeros_like(gradient)
                for variant in VARIANTS:
                    changed = intervene(state, parameter, variant)
                    after = adam_step(parameter, used_gradient, changed)
                    result["rows"].extend(rows_for_group(
                        dict(base, gradient_mode=gradient_mode, variant=variant),
                        parameter, used_gradient, after))
            del p, sample, proxy, gradient
        intact = digest_state(state) == original_digest and torch.equal(parameter, original_parameter)
        if not intact:
            raise RuntimeError("Source parameter/state mutation detected")
        for parameter_id, entry in original_state["state"].items():
            for key, value in entry.items():
                current = state["state"][parameter_id][key]
                if isinstance(value, torch.Tensor) and not torch.equal(value, current):
                    raise RuntimeError(f"Source tensor mutated: {key}")
        result["source_integrity"].append(dict(case=case, unchanged=True,
                                               optimizer_state_sha256=original_digest))
        write(args.out / "summary.json", result)
        print(json.dumps(dict(case=case, rows=len(result["rows"]), elapsed_seconds=time.time()-started)), flush=True)
    result.update(status="completed", elapsed_seconds=time.time()-started,
                  private_clone_optimizer_updates=len(args.cases.split(",")) * len(args.seeds.split(","))
                  * len(VARIANTS) * len(result["gradient_modes"]))
    write(args.out / "summary.json", result)


if __name__ == "__main__":
    main()
