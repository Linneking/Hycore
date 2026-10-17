"""Controlled single-GPU proxy-gradient/AdamW replay on private frozen caches."""
from __future__ import annotations
import argparse
import copy
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from optimizer_math import decompose_gradient, decompose_update
from inter_hierarchy_MN40.hier_proxy_scratch_v8.geometry import to_ball, official_distance
from inter_hierarchy_MN40.hier_proxy_scratch_v5.relations import mine_reciprocal_triplets
from inter_hierarchy_MN40.hier_proxy_scratch_v6.hier_loss import ghhc_loss


def write(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False))
    temp.replace(path)


def quant(value):
    v = value.detach().double().reshape(-1)
    return dict(mean=float(v.mean()), median=float(v.median()), min=float(v.min()),
                max=float(v.max()), p10=float(torch.quantile(v, .1)),
                p90=float(torch.quantile(v, .9)))


def balanced(labels, seed):
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    classes, counts = np.unique(labels, return_counts=True)
    classes = classes[counts >= 2]
    chosen = rng.choice(classes, 32, replace=False)
    return np.asarray([i for c in chosen for i in rng.choice(np.flatnonzero(labels == c), 2, replace=False)])


def loss_parts(tangent, whole, labels, seed=22, hook=True, t=50):
    proxy = to_ball(tangent, hook=hook)
    ds = official_distance(whole)
    dp = official_distance(proxy)
    sp = official_distance(whole, proxy)
    similarity = torch.exp(-ds.detach()) + (labels[:, None] == labels[None, :]).to(whole.dtype)
    sample = mine_reciprocal_triplets(similarity, 20, t, seed, True, labels=labels)
    relations = mine_reciprocal_triplets(torch.exp(-dp.detach()), 20, t, seed + 1, True)
    generator = torch.Generator(device=whole.device).manual_seed(seed)
    sample_loss, ss = ghhc_loss(sp, sample["triplets"], .1, .1, generator)
    proxy_loss, ps = ghhc_loss(dp, relations["triplets"], .1, .1, generator)
    return .5 * sample_loss, .5 * proxy_loss, dict(sample_collision=ss["collision_rate"],
                                                proxy_collision=ps["collision_rate"])


def adam_step(u, grad, state, *, lr=None, eps=None, decay=None, scale=1., sgd=False):
    p = torch.nn.Parameter(u.detach().clone())
    if sgd:
        optimizer = torch.optim.SGD([p], lr=.01 if lr is None else lr, weight_decay=.01 if decay is None else decay)
    else:
        optimizer = torch.optim.AdamW([p], lr=.01, eps=1e-8, weight_decay=.01)
        if state:
            optimizer.load_state_dict(copy.deepcopy(state))
        group = optimizer.param_groups[0]
        if lr is not None: group["lr"] = lr
        if eps is not None: group["eps"] = eps
        if decay is not None: group["weight_decay"] = decay
    p.grad = grad.detach().clone() * scale
    optimizer.step()
    return p.detach()


def packed_original(root, ds):
    folder = root / "original_optimizer" / ds
    meta = json.loads((folder / "metadata.json").read_text())
    with np.load(folder / "proxy_adam.npz", allow_pickle=False) as a:
        arrays = {k: a[k].copy() for k in a.files}
    return meta, arrays


def original_optimizer(meta, arrays):
    # Known original lcas group has exactly one tensor; map that state to param0.
    group = dict(meta["proxy_optimizer_group"])
    group["params"] = [0]
    return {"state": {0: {k: torch.as_tensor(arrays[k]) for k in ("step", "exp_avg", "exp_avg_sq")}},
            "param_groups": [group]}


def load_case(root, name):
    if name.startswith("v8"):
        epoch = int(name.split("_e")[1])
        cp = torch.load(root / f"proxy_e{epoch:03d}.pt", map_location="cpu", weights_only=False)
        with np.load(root / f"e{epoch:03d}_whole_cache.npz", allow_pickle=False) as a:
            whole, labels = a["mu"].copy(), a["labels"].copy()
        return cp["proxy"]["tangent_proxies"].float(), whole, labels, cp.get("proxy_optimizer")
    ds = name
    meta, a = packed_original(root, ds)
    with np.load(root / "original_batches" / f"{ds}.npz", allow_pickle=False) as x:
        whole, labels = x["mu"].copy(), x["labels"].copy()
    tangent_key = "tangent" if "tangent" in a else "proxy_tangent"
    return torch.as_tensor(a[tangent_key]).float(), whole, labels, original_optimizer(meta, a)


def rows_for_group(base, u, g, updated):
    grad = decompose_gradient(u, g)
    move = decompose_update(u, updated)
    groups = {"all": torch.ones(len(u), dtype=torch.bool, device=u.device),
              "clipped": u.detach().norm(dim=-1) > 2.3,
              "interior": u.detach().norm(dim=-1) <= 2.3}
    rows = []
    for name, mask in groups.items():
        if not bool(mask.any()): continue
        row = dict(base, group=name, count=int(mask.sum()))
        for key in ("gradient_norm", "signed_radial_gradient", "radial_gradient_fraction"):
            for stat, val in quant(grad[key][mask]).items(): row["grad_" + key + "_" + stat] = val
        for key in ("step_norm", "signed_radial_step", "exact_norm_change", "second_order_excess", "angular_step_norm"):
            for stat, val in quant(move[key][mask]).items(): row[key + "_" + stat] = val
        row["outward_gradient_descent_fraction"] = float((grad["signed_radial_gradient"][mask] < 0).double().mean())
        row["outward_actual_update_fraction"] = float((move["exact_norm_change"][mask] > 0).double().mean())
        rows.append(row)
    return rows


def static(root, out, device, cases):
    rows, gradients = [], []
    for name in cases:
        u_cpu, pool, labels, state = load_case(root, name)
        for batch_seed in (22, 83):
            positions = balanced(labels, batch_seed)
            w = torch.as_tensor(pool[positions], device=device)
            gold = torch.as_tensor(labels[positions], device=device, dtype=torch.long)
            u = u_cpu.to(device).requires_grad_()
            s, p, stats = loss_parts(u, w, gold, seed=batch_seed)
            gs = torch.autograd.grad(s, u, retain_graph=True)[0]
            gp = torch.autograd.grad(p, u)[0]
            for component, g in (("sample", gs), ("proxy", gp), ("both", gs + gp)):
                base = dict(case=name, batch_seed=batch_seed, component=component,
                            sample_loss=float(s.detach()), proxy_loss=float(p.detach()), **stats)
                options = [("fresh_eps8", dict(lr=.01, eps=1e-8), None),
                           ("fresh_eps4", dict(lr=.01, eps=1e-4), None),
                           ("fresh_lr001", dict(lr=.001, eps=1e-8), None),
                           ("sgd", dict(lr=.01, sgd=True), None)]
                if state:
                    options += [("saved_native", {}, state),
                                ("saved_eps4", dict(eps=1e-4), state),
                                ("saved_zero_gradient", dict(scale=0.), state)]
                for variant, params, saved in options:
                    after = adam_step(u, g, saved, **params)
                    rows.extend(rows_for_group(dict(base, variant=variant), u, g, after))
            del u, s, p, gs, gp
        write(out / "static.json", rows)
        print(json.dumps({"phase": "static", "case": name, "rows": len(rows)}), flush=True)
    # Just-inside/outside radial intervention: same directions, same whole objects.
    last_case = [name for name in cases if name.startswith("v8")][-1]
    raw, pool, labels, _ = load_case(root, last_case)
    pos = balanced(labels, 22)
    w, gold = torch.as_tensor(pool[pos], device=device), torch.as_tensor(labels[pos], device=device)
    for radius in (2., 2.29, 2.31):
        u = (raw / raw.norm(dim=-1, keepdim=True) * radius).to(device).requires_grad_()
        s, p, st = loss_parts(u, w, gold)
        for component, l in (("sample", s), ("proxy", p)):
            g = torch.autograd.grad(l, u, retain_graph=True)[0]
            rows.extend(rows_for_group(dict(case=last_case + "_radius", radius=radius, batch_seed=22,
                        component=component, variant="fresh_eps8", sample_loss=float(s.detach()),
                        proxy_loss=float(p.detach()), **st), u, g, adam_step(u, g, None)))
    write(out / "static.json", rows)


def trajectories(root, out, device, steps, cases):
    trials = []
    for name in cases:
        trials += [(name, "both_eps8", "both", 1e-8, .01, True, False),
                   (name, "both_eps4", "both", 1e-4, .01, True, False)]
    first = [name for name in cases if name.startswith("v8")][0]
    last = [name for name in cases if name.startswith("v8")][-1]
    trials += [(first, "sample_eps8", "sample", 1e-8, .01, True, False),
               (first, "proxy_eps8", "proxy", 1e-8, .01, True, False),
               (first, "both_lr001", "both", 1e-8, .001, True, False),
               (first, "both_nohook", "both", 1e-8, .01, False, False),
               (first, "both_whole_shell", "both", 1e-8, .01, True, True),
               (last, "both_whole_shell", "both", 1e-8, .01, True, True)]
    for case, variant, comp, eps, lr, hook, shell in trials:
        _, pool, labels, _ = load_case(root, case)
        dim = pool.shape[1]
        generator = torch.Generator().manual_seed(22)
        initial = torch.randn(512, dim, generator=generator) / math.sqrt(dim) * 2.3 * .9
        u = torch.nn.Parameter(initial.to(device))
        optimizer = torch.optim.AdamW([u], lr=lr, eps=eps, weight_decay=.01)
        history = []
        for step in range(steps + 1):
            if step % 8 == 0 or step == steps:
                radius = u.detach().norm(dim=-1)
                history.append(dict(step=step, cap_count=int((radius > 2.3).sum()),
                    norm_median=float(radius.median()), norm_max=float(radius.max()),
                    case=case, variant=variant, eps=eps, lr=lr, hook=hook, whole_shell=shell))
                write(out / f"trajectory_{case}_{variant}.json", history)
            if step == steps: break
            positions = balanced(labels, 22 + step % 8)
            whole = torch.as_tensor(pool[positions], device=device)
            if shell:
                whole = whole / whole.norm(dim=-1, keepdim=True) * (math.tanh(math.sqrt(.1) * 2.3) / math.sqrt(.1))
            gold = torch.as_tensor(labels[positions], device=device, dtype=torch.long)
            optimizer.zero_grad(set_to_none=True)
            s, p, stats = loss_parts(u, whole, gold, seed=22 + step, hook=hook)
            loss = s if comp == "sample" else p if comp == "proxy" else s + p
            loss.backward()
            before = u.detach().clone()
            grad = decompose_gradient(u, u.grad)
            optimizer.step()
            move = decompose_update(before, u)
            if step % 8 == 0 or step + 1 == steps:
                history[-1].update(loss=float(loss.detach()),
                    gradient_norm_median=float(grad["gradient_norm"].median()),
                    descent_outward_fraction=float((grad["signed_radial_gradient"] < 0).double().mean()),
                    actual_norm_change_mean=float(move["exact_norm_change"].mean()),
                    first_order_radial_mean=float(move["signed_radial_step"].mean()),
                    second_order_excess_mean=float(move["second_order_excess"].mean()))
            del s, p, loss, grad, move
        print(json.dumps({"phase": "trajectory", "case": case, "variant": variant, "final": history[-1]}), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inputs", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--mode", choices=("static", "trajectory", "all"), default="all")
    p.add_argument("--steps", type=int, default=128)
    p.add_argument("--gpu-uuid", required=True)
    p.add_argument("--static-cases", default="v8_e0,v8_e1,v8_e5,v8_e20,v8_e100,v8_e300,cub,cars,sop")
    p.add_argument("--trajectory-cases", default="v8_e5,v8_e300,cub,sop")
    a = p.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != a.gpu_uuid:
        raise RuntimeError("Must pin exactly the checked idle GPU UUID")
    apps = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid",
                                   "--format=csv,noheader"], text=True)
    if a.gpu_uuid in apps:
        raise RuntimeError("Selected GPU already has a compute process")
    a.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    start = time.time()
    write(a.out / "manifest.json", dict(status="running", mode=a.mode, steps=a.steps,
          source_file_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          condition="Frozen clean features, class-balanced64,512proxies,T50,full ST, original scalar.5",
          source_updates=0, test_forwards=0))
    device = torch.device("cuda:0")
    if a.mode in ("static", "all"): static(a.inputs, a.out, device, a.static_cases.split(","))
    if a.mode in ("trajectory", "all"): trajectories(a.inputs, a.out, device, a.steps, a.trajectory_cases.split(","))
    write(a.out / "manifest.json", dict(status="completed", seconds=time.time()-start,
          mode=a.mode, source_updates=0, private_copy_optimizer_updates=a.steps*(2*len(a.trajectory_cases.split(","))+6) if a.mode!="static" else 0,
          test_forwards=0, limits="Frozen encoder/features; not a complete-training causal ablation"))
    print("CAP_PROXY_REPLAY_COMPLETED", flush=True)


if __name__ == "__main__":
    main()

