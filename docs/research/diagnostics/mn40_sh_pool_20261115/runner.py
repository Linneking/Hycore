"""Matched single-device SH64 updates, with the sample relation pool at 0/32/64."""
from __future__ import annotations

import argparse
import copy
import hashlib
from itertools import islice
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))

import geoopt
import numpy as np
import torch
from torch.utils.data import DataLoader

from inter_hierarchy_MN40.hycore_c01_gs_halfmargin_20261020 import train as ts06
from inter_hierarchy_MN40.hier_proxy_cycle_v9c.geometry import (
    assert_hier_shell, configure_backbone, official_distance, whole_to_hier,
)
from inter_hierarchy_MN40.hier_proxy_scratch_v5.relations import mine_reciprocal_triplets
from inter_hierarchy_MN40.hier_proxy_scratch_v6.hier_loss import ghhc_loss
from inter_hierarchy_MN40.hier_proxy_scratch_v8.hier_loss import HIERLoss

SEED = 4780
EPOCHS = 300
INITIAL_SHA256 = "8926765a4842652b751873fb89aa37d24d9b0f4f2d831d853b9cc0c1cd46472d"
SPLIT_SHA256 = "3bbf0c79b3cb9581caeafe3f7f12caa6861f236f0159512a5998d5e9bcf25105"
DATA_SHA256 = {
    "ply_data_train0.h5": "996e8f2741c5c2662314a79c4e7b461399e2835dff28925046e7e42339f9a74d",
    "ply_data_train1.h5": "0efaab3f3a7903f78037da933682267dfcdc548dc97aadda25bac218a1930276",
    "ply_data_train2.h5": "d705fc9b55b80e0a45103ec6a2938968f1e704c6bec8f2624bebbf4e435bb50a",
    "ply_data_train3.h5": "ee6fb75f9cb85bc36e589e7ebf23b3cd9c42ef00566c29e7a6679e83ed575495",
    "ply_data_train4.h5": "15143c2e796907767bb795d505a22d46d776e710ae55b9487e5a9684732bd1fa",
    "ply_data_test0.h5": "fb1631826ff4102aac03ee8a59ed309f8ce839a210b6da1a40a2e4d45bba9d56",
    "ply_data_test1.h5": "dc002c8e9e168cedd22afbdfa8f1fae19d0f2fafb6efa993e4c88fbcaee8f353",
}


def verify_data(loaded, partition):
    expected = {name: sha for name, sha in DATA_SHA256.items() if f"_{partition}" in name}
    if loaded[partition]["file_sha256"] != expected:
        raise RuntimeError(f"Unexpected ModelNet40 {partition} data")


def data_loaders(data_module, data_dir):
    from inter_hierarchy_MN40.hier_proxy_scratch_v4.train import split_identity, stratified_split

    names = sorted(name for name in DATA_SHA256 if "_train" in name)
    os.environ["HYCORE_TRAIN_SHARD_ORDER"] = json.dumps(names)
    loaded = ts06.install_data_root(data_module, data_dir)
    train = data_module.ModelNet40(1024, "train")
    verify_data(loaded, "train")
    train_ids, val_ids = stratified_split(train.label.reshape(-1), seed=22, fraction=.1)
    if (len(train_ids), len(val_ids)) != (8856, 984) or split_identity(train_ids, val_ids) != SPLIT_SHA256:
        raise RuntimeError("Unexpected VS984 split")
    val = copy.copy(train)
    val.partition = "validation"
    val.data, val.label = train.data[val_ids], train.label[val_ids]
    train.data, train.label = train.data[train_ids], train.label[train_ids]
    # Separate data RNG so model/dropout and the extra HIER path cannot change it.
    train_loader = DataLoader(train, batch_size=32, shuffle=True, drop_last=True,
                              num_workers=8, generator=torch.Generator().manual_seed(SEED))
    val_loader = DataLoader(val, batch_size=16, num_workers=8,
                            generator=torch.Generator().manual_seed(SEED + 1))
    return train_loader, val_loader, loaded


def prepare(original, batch):
    data, label = batch
    whole = data.cuda().permute(0, 2, 1)
    _, whole, n_points = original.get_children_np(whole, kmin=800, kmax=1024)
    margin, part, _ = original.get_children_np(whole, starting=n_points, kmin=200, kmax=600)
    return whole, part, label.cuda().reshape(-1), margin


def forward(net, crop, geometry):
    whole, part, _, _ = crop
    part_mu, _ = net(part, emb=True)
    if geometry is None:
        whole_mu, logits = net(whole)
        return whole_mu, part_mu, logits, None
    geometry.capture_enabled = True
    try:
        whole_mu, logits = net(whole)
        return whole_mu, part_mu, logits, geometry.pre_gs
    finally:
        geometry.clear()


def batchnorm_state(net):
    return [(buffer, buffer.detach().clone())
            for module in net.modules()
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)
            for buffer in (module.running_mean, module.running_var, module.num_batches_tracked)
            if buffer is not None]


def mine_samples(points, labels, seed):
    similarity = torch.exp(-official_distance(points).detach())
    similarity += (labels[:, None] == labels[None, :]).to(points.dtype)
    return mine_reciprocal_triplets(similarity, 20, 50, seed, True, labels=labels)["triplets"]


def hier_backward(proxy, leaves, labels, pool, update, sample_rng, proxy_rng):
    seed = SEED + update * 10007
    if pool == 64:
        views = [(whole_to_hier(torch.cat(leaves)), torch.cat(labels))]
    else:
        views = [(whole_to_hier(leaf), label) for leaf, label in zip(leaves, labels)]
    for points, _ in views:
        assert_hier_shell(points)
    mined = [mine_samples(points.detach(), label, seed + 2 * n)
             for n, (points, label) in enumerate(views)]
    draw_count = sum(map(len, mined))
    active = cross = 0
    sample_value = 0.0
    for (points, _), triplets in zip(views, mined):
        if not len(triplets):
            continue
        loss, stats = ghhc_loss(official_distance(points, proxy.proxies()), triplets,
                                proxy.tau, proxy.margin, sample_rng)
        share = len(triplets) / draw_count
        (0.5 * share * loss).backward()
        sample_value += share * float(loss.detach())
        active += stats["active_triplets"]
        if pool == 64:
            cross = int(((triplets // 32) != (triplets[:, :1] // 32)).any(dim=1).sum())

    distances = official_distance(proxy.proxies())
    proxy_draws = mine_reciprocal_triplets(torch.exp(-distances.detach()), 20, 50, seed + 1, True)["triplets"]
    proxy_loss, proxy_stats = ghhc_loss(distances, proxy_draws, proxy.tau, proxy.margin, proxy_rng)
    (0.5 * proxy_loss).backward()
    return {"sample_draws": draw_count, "sample_active": active, "sample_cross_micro_draws": cross,
            "proxy_draws": len(proxy_draws), "proxy_active": proxy_stats["active_triplets"],
            "sample_loss": sample_value, "proxy_loss": float(proxy_loss.detach())}


def train_step(net, optimizer, original, criterion2, crops, geometry, proxy, proxy_opt,
               pool, update, sample_rng, proxy_rng, check_replay):
    optimizer.zero_grad(set_to_none=True)
    stats = {"sample_draws": 0, "sample_active": 0, "sample_cross_micro_draws": 0,
             "proxy_draws": 0, "proxy_active": 0, "sample_loss": 0., "proxy_loss": 0.}
    leaves = [None, None]
    if proxy is not None:
        proxy_opt.zero_grad(set_to_none=True)
        # Recompute one model microbatch at a time for backpropagation. Restore
        # BN and RNG after the embedding pass so it has no extra training effect.
        bn = batchnorm_state(net)
        cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state()
        with torch.no_grad():
            probe = [forward(net, crop, geometry)[-1] for crop in crops]
            for buffer, saved in bn:
                buffer.copy_(saved)
        torch.set_rng_state(cpu_rng)
        torch.cuda.set_rng_state(cuda_rng)
        leaves = [point.detach().requires_grad_() for point in probe]
        stats = hier_backward(proxy, leaves, [crop[2] for crop in crops], pool,
                              update, sample_rng, proxy_rng)

    base_value = correct = 0
    for crop, leaf in zip(crops, leaves):
        _, _, labels, margin = crop
        whole, part, logits, source = forward(net, crop, geometry)
        *_, contr, radial = criterion2(whole, part, hier_margin=margin, contr_margin=4, ball_dim=256)
        base = original.cal_loss(logits, labels) + 0.01 * contr + 0.01 * radial
        if check_replay and leaf is not None:
            torch.testing.assert_close(source.detach(), leaf.detach(), rtol=1e-5, atol=1e-6)
        if leaf is not None and leaf.grad is not None:
            torch.autograd.backward((0.5 * base, source), (None, leaf.grad))
        else:
            (0.5 * base).backward()
        base_value += 0.5 * float(base.detach())
        correct += int((logits.detach().argmax(-1) == labels).sum())
        # Drop this graph before allocating the next full model forward.
        del whole, part, logits, source, base, contr, radial, _

    torch.nn.utils.clip_grad_norm_(net.parameters(), 1, error_if_nonfinite=True)
    optimizer.step()
    if proxy is not None:
        proxy_opt.step()
        proxy.assert_geometry()
    return {**stats, "base_loss": base_value, "correct": correct}


def fingerprint(raw, crops):
    digest = hashlib.sha256()
    for batch, crop in zip(raw, crops):
        for tensor in (*batch, *crop[:3]):
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        digest.update(str(crop[3]).encode())
    return digest.hexdigest()


def save_best(path, model, optimizer, scheduler, proxy, proxy_opt, proxy_schedule, epoch, metrics):
    state = {"epoch": epoch, "validation": metrics, "model": model.state_dict(),
             "optimizer": optimizer.state_dict(), "schedule": scheduler.state_dict()}
    if proxy is not None:
        state.update(proxy=proxy.state_dict(), proxy_optimizer=proxy_opt.state_dict(),
                     proxy_schedule=proxy_schedule.state_dict())
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=int, choices=(0, 32, 64), required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    manifest = {
        "status": "initializing", "started": ts06.now(), "pid": os.getpid(),
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        "gpu": os.environ["HYCORE_PHYSICAL_GPUS"], "gpu_uuid": os.environ["CUDA_VISIBLE_DEVICES"],
        "pool": args.pool, "microbatch": 32, "micros_per_update": 2,
        "seed": SEED, "split_seed": 22, "split_sha256": SPLIT_SHA256,
        "epochs": 1 if args.smoke else EPOCHS, "updates_per_epoch": 2 if args.smoke else 138,
        "smoke": args.smoke, "teacher": None, "initial_checkpoint": None, "completed_epochs": 0,
        "official_test_opened": False, "selection": "highest_validation_OA_first_tie",
        "versions": {"torch": torch.__version__, "numpy": np.__version__, "geoopt": geoopt.__version__},
    }
    ts06.atomic_json(out / "manifest.json", manifest)
    try:
        os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
        original = ts06.import_original(REPO)
        import data as data_module
        import hutil

        if torch.cuda.device_count() != 1:
            raise RuntimeError("Each arm requires exactly one visible GPU")
        original.progress_bar = lambda *_: None
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        model = configure_backbone(original.Hype_pointMLP())
        manifest["initial_sha256"] = ts06.state_digest(model)
        if manifest["initial_sha256"] != INITIAL_SHA256:
            raise RuntimeError(f"Unexpected model initialization: {manifest['initial_sha256']}")
        net = torch.nn.DataParallel(model.cuda())
        ts06.install_loss_curvature(hutil)
        criterion2 = ts06.half_margin_criterion(original.hype_triplet_losses)
        train, val, loaded = data_loaders(data_module, args.data_dir)
        if len(train) // 2 != 138 or len(train) % 2:
            raise RuntimeError("Unexpected number of microbatches")
        optimizer = geoopt.optim.RiemannianSGD(net.parameters(), lr=.1, momentum=.9, weight_decay=.0002)
        schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS, eta_min=.005)
        geometry = proxy = proxy_opt = proxy_schedule = sample_rng = proxy_rng = None
        if args.pool:
            geometry = model.v9c_geometry
            proxy = HIERLoss(seed=SEED).cuda()
            proxy_opt = torch.optim.AdamW(proxy.parameters(), lr=.005, eps=1e-4, weight_decay=.01)
            proxy_schedule = torch.optim.lr_scheduler.CosineAnnealingLR(proxy_opt, EPOCHS, eta_min=.0005)
            sample_rng = torch.Generator(device="cuda").manual_seed(SEED + 300001)
            proxy_rng = torch.Generator(device="cuda").manual_seed(SEED + 300002)
        manifest.update(status="running", dataset=loaded, gpu_name=torch.cuda.get_device_name(0))
        ts06.atomic_json(out / "manifest.json", manifest)
        best = None
        for epoch in range(1, manifest["epochs"] + 1):
            net.train()
            torch.cuda.reset_peak_memory_stats()
            started = time.monotonic()
            iterator = iter(train)
            sums = {}
            for step in range(1, manifest["updates_per_epoch"] + 1):
                raw = list(islice(iterator, 2))
                crops = [prepare(original, batch) for batch in raw]
                verify = epoch == 1 and step <= 2
                if verify:
                    before = [int(b) for b, _ in batchnorm_state(net) if b.ndim == 0]
                values = train_step(net, optimizer, original, criterion2, crops, geometry, proxy,
                                    proxy_opt, args.pool, (epoch - 1) * 138 + step,
                                    sample_rng, proxy_rng, verify)
                if verify:
                    after = [int(b) for b, _ in batchnorm_state(net) if b.ndim == 0]
                    whole_updates = [a - b for b, a in zip(before, after)]
                    if not whole_updates or set(whole_updates) - {2, 4} or 4 not in whole_updates:
                        raise RuntimeError(f"Unexpected BN counters {whole_updates}")
                    if args.pool and (not values["sample_draws"] or not values["proxy_draws"]
                                      or int(proxy_opt.state[proxy.tangent_proxies]["step"]) != step
                                      or bool(values["sample_cross_micro_draws"]) != (args.pool == 64)):
                        raise RuntimeError(f"Pool/optimizer acceptance failed: {values}")
                    ts06.atomic_json(out / f"update{step:02}.json", {
                        **values, "input_sha256": fingerprint(raw, crops),
                        "batchnorm_updates": whole_updates, "verified_hier_replay": bool(args.pool),
                        "seconds": round(time.monotonic() - started, 2),
                        "peak_GiB": round(torch.cuda.max_memory_allocated() / 2**30, 3)})
                for key, value in values.items():
                    sums[key] = sums.get(key, 0) + value
            del iterator
            validation = original.validate(net, val, original.cal_loss, "cuda")
            schedule.step()
            if proxy_schedule is not None:
                proxy_schedule.step()
            if best is None or validation["acc"] > best["acc"]:
                best = {"epoch": epoch, **validation}
                if not args.smoke:
                    save_best(out / "best.pt", model, optimizer, schedule, proxy, proxy_opt,
                              proxy_schedule, epoch, validation)
            row = {"epoch": epoch, "validation": validation, "best_validation": best,
                   "epoch_seconds": round(time.monotonic() - started, 2),
                   "train_OA": 100 * sums["correct"] / (64 * step), "updates": step,
                   "peak_GiB": round(torch.cuda.max_memory_allocated() / 2**30, 3), **sums}
            with (out / "epochs.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, allow_nan=False) + "\n")
            print(json.dumps(row, allow_nan=False), flush=True)
            manifest.update(completed_epochs=epoch, best_validation=best, last_epoch=row)
            ts06.atomic_json(out / "manifest.json", manifest)

        if not args.smoke:
            checkpoint = torch.load(out / "best.pt", map_location="cpu", weights_only=True)
            model.load_state_dict(checkpoint["model"])
            manifest.update(status="evaluating_locked_model", official_test_opened=True)
            ts06.atomic_json(out / "manifest.json", manifest)
            test = data_module.ModelNet40(1024, "test")
            verify_data(loaded, "test")
            if len(test) != 2468:
                raise RuntimeError("Unexpected ModelNet40 test size")
            metrics = original.validate(net, DataLoader(test, batch_size=16, num_workers=8),
                                        original.cal_loss, "cuda")
            ts06.atomic_json(out / "final_test.json", {"selected_epoch": best["epoch"], **metrics})
        manifest.update(status="complete", finished=ts06.now())
    except Exception as error:
        manifest.update(status="failed", error=repr(error), finished=ts06.now())
        raise
    finally:
        ts06.atomic_json(out / "manifest.json", manifest)


if __name__ == "__main__":
    main()
