"""Bounded preflight on training data: c1 gradients, K/T coverage and base BN.

Reference weights are diagnostic only and never become the new training
initialization. Outputs remain run-local; official test data are not read.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from hier_proxy_scratch_v4.train import (
    Hype_pointMLP, ProxyHierarchy, load_shards, seed_all, freeze_curvature,
    stratified_split, split_identity, sample_subcloud, base_forward,
    initialize_proxies_from_features, triplet_columns, atomic_json, sha256,
    part_batch_statistics_only, make_loader, PointClouds, DEFAULT_TAU, DEFAULT_MARGIN,
)
from hier_proxy_scratch_v4.geometry import radius_gradient_audit, source_cap_intra_audit
from hier_proxy_scratch_v4.mining import mine_batch_triplets
from hier_proxy_scratch_v4.launch_matrix import wait_for_idle_gpu
from hier_proxy_v2_online.diagnostics import fixed_rng


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--reference-checkpoint", type=Path)
    p.add_argument("--physical-gpu", type=int, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError("diagnostic output must be new")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(a.physical_gpu):
        raise RuntimeError("physical GPU visibility mismatch")
    wait_for_idle_gpu(a.physical_gpu, timeout_seconds=60)
    seed_all(22)
    started = time.monotonic()
    device = torch.device("cuda:0")
    points, labels, _ = load_shards(a.data_dir, "train")
    train_ids, val_ids = stratified_split(labels)
    dataset = PointClouds(points[train_ids], labels[train_ids], 1024, "reference", 22)
    config = argparse.Namespace(seed=22, workers=0)
    loader, sampler = make_loader(dataset, labels[train_ids], config, 0)
    batches = []
    for i, batch in enumerate(loader):
        batches.append(batch)
        if i == 3:
            break
    model = Hype_pointMLP().to(device)
    freeze_curvature(model)
    report = {"c": 1, "distance_units": "native d1", "margin": DEFAULT_MARGIN,
              "tau": DEFAULT_TAU, "split_sha256": split_identity(train_ids, val_ids),
              "train_count": len(train_ids), "validation_count": len(val_ids),
              "sampler": sampler.stats, "cap_compatibility": source_cap_intra_audit(),
              "predicted_gradient_factors": radius_gradient_audit(), "representations": {}}
    representations = [("random_initialization", None)]
    if a.reference_checkpoint:
        checkpoint = torch.load(a.reference_checkpoint, map_location="cpu", weights_only=False)
        state = {k.removeprefix("module."): v for k, v in checkpoint["net"].items()}
        representations.append(("reference_diagnostic_only", state))
        report["reference_checkpoint_sha256"] = sha256(a.reference_checkpoint)
        report["reference_checkpoint_epoch"] = checkpoint.get("epoch")
    for name, state in representations:
        if state:
            model.load_state_dict(state)
        model.eval()
        embeddings = []
        with torch.no_grad():
            for points_batch, gold, ids in batches:
                mu, _ = model(points_batch.to(device).transpose(1, 2).contiguous())
                embeddings.append((model.manifold2.projx(mu), gold))
            # Enough distinct references for the global 256-node initialization.
            refs = []
            for batch_index, (point_batch, _, _) in enumerate(loader):
                refs.append(model(point_batch.to(device).transpose(1, 2).contiguous())[0].cpu())
                if batch_index == 11:
                    break
        proxy = ProxyHierarchy(256, 256, seed=22).to(device)
        initialization = initialize_proxies_from_features(proxy, torch.cat(refs))
        rows = []
        for k in (3, 4, 5):
            for t in (16, 32, 50, 64, 96):
                summaries = []
                for b, (mu, gold) in enumerate(embeddings):
                    mined = mine_batch_triplets(mu, gold, k=k, t_per_anchor=t, seed=22+b)
                    with fixed_rng(9020+b, device):
                        loss, stats = proxy.sample_loss(mu, triplet_columns(mined["inter"]))
                    summaries.append({**mined["stats"], **stats, "loss": float(loss.detach())})
                row = {"k": k, "t": t}
                for key in summaries[0]:
                    row[key] = float(np.mean([s[key] for s in summaries]))
                row["collision_rate"] = row["collisions"] / max(row["triplets"], 1)
                row["active_rate"] = row["active_triplets"] / max(row["triplets"], 1)
                rows.append(row)
        mu, gold = embeddings[0]
        relation = mine_batch_triplets(mu, gold, k=3, t_per_anchor=64, seed=22)
        gradient = proxy.gradient_mode_audit(mu, triplet_columns(relation["inter"]))
        radius = mu.detach().norm(dim=-1)
        proxy_grid = []
        for budget in (8, 16, 32, 50):
            with fixed_rng(9050, device):
                ploss, pstats = proxy.proxy_loss(k=20, t_per_anchor=budget)
            proxy_grid.append({"t": budget, "loss": float(ploss.detach()), **pstats,
                               "collision_rate": pstats["collisions"] / max(pstats["triplets"], 1),
                               "active_rate": pstats["active_triplets"] / max(pstats["triplets"], 1)})
        report["representations"][name] = {
            "proxy_initialization": initialization, "sample_grid": rows,
            "proxy_grid": proxy_grid,
            "gradient_audit": gradient, "radius_min": float(radius.min()),
            "radius_median": float(radius.median()), "radius_max": float(radius.max())}
        del proxy
    # Finite full-network backward and exact BN-update assertions.
    model.train()
    bn = [m for m in model.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
    point_batch, gold, _ = batches[0]
    data = point_batch.to(device).transpose(1, 2).contiguous()
    original = data.clone()
    whole, _ = sample_subcloud(data, 800, 1024)
    child, _ = sample_subcloud(whole, 200, 600)
    snapshot = [m.num_batches_tracked.clone() for m in bn]
    buffers = [(m.running_mean.clone(), m.running_var.clone()) for m in bn]
    with part_batch_statistics_only(model):
        model(child, emb=True)
    part_unchanged = all(torch.equal(m.num_batches_tracked, n) and
                         torch.equal(m.running_mean, b[0]) and torch.equal(m.running_var, b[1])
                         for m, n, b in zip(bn, snapshot, buffers))
    model(whole)
    whole_once = all(int(m.num_batches_tracked - n) == 1 for m, n in zip(bn, snapshot))
    values = base_forward(model, point_batch, gold, device)
    values["base"].backward()
    finite = all(bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None)
    report["base_gate"] = {"source_input_unchanged": torch.equal(data, original),
                           "part_bn_buffers_unchanged": part_unchanged,
                           "whole_bn_updates_once": whole_once, "base_backward_finite": finite,
                           "base_loss": float(values["base"].detach())}
    if not all((part_unchanged, whole_once, finite, torch.equal(data, original))):
        raise RuntimeError("base protocol gate failed")
    report["seconds"] = time.monotonic() - started
    report["peak_memory_mb"] = torch.cuda.max_memory_allocated(device) / 1024**2
    atomic_json(a.output, report)
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
