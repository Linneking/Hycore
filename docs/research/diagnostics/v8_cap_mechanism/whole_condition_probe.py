"""Read-only factorial test of BN, augmentation and whole/part construction.

Twelve main conditions use the same training objects and original local32
grouping. BN-only training leaves dropout disabled. Each condition restores
checkpoint BN buffers and RNG; no backward, optimizer or test-set forward.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import itertools
import json
import os
from pathlib import Path
import random
import sys
import time

import h5py
import numpy as np
import torch

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[4]
for path in (ROOT, ROOT / "classification_ModelNet40", ROOT / "inter_hierarchy_MN40",
             ROOT / "pointnet2_ops_lib"):
    sys.path.insert(0, str(path))
from models.pointmlp import Hype_pointMLP  # noqa: E402
from inter_hierarchy_MN40.hier_proxy_scratch_v8.geometry import configure_backbone, depth  # noqa: E402
from inter_hierarchy_MN40.hier_proxy_scratch_v8.base_protocol import get_children_alias  # noqa: E402


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            result.update(block)
    return result.hexdigest()


def state_hash(model):
    result = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        result.update(name.encode())
        result.update(value.detach().cpu().contiguous().numpy().tobytes())
    return result.hexdigest()


def stats(values):
    a = np.asarray(values, dtype=np.float64).reshape(-1)
    if not len(a) or not np.isfinite(a).all():
        raise ValueError("Empty/nonfinite measurement")
    qs = np.quantile(a, [0, .01, .05, .25, .5, .75, .95, .99, 1])
    return dict(zip(("min", "p01", "p05", "p25", "median", "p75", "p95", "p99", "max"),
                    map(float, qs)), mean=float(a.mean()), std=float(a.std()), count=len(a))


def write_json(path, value):
    temporary = Path(str(path) + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def capture_rng(device):
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None}


def restore_rng(saved, device):
    random.setstate(saved["python"])
    np.random.set_state(saved["numpy"])
    torch.set_rng_state(saved["torch"])
    if saved["cuda"] is not None:
        torch.cuda.set_rng_state(saved["cuda"], device)


def rng_equal(a, b):
    return (a["python"] == b["python"] and a["numpy"][0] == b["numpy"][0]
            and np.array_equal(a["numpy"][1], b["numpy"][1])
            and a["numpy"][2:] == b["numpy"][2:] and torch.equal(a["torch"], b["torch"])
            and ((a["cuda"] is None and b["cuda"] is None)
                 or (a["cuda"] is not None and torch.equal(a["cuda"], b["cuda"]))))


def select_objects(labels, allowed, panels, seed):
    rng = np.random.RandomState(seed)
    chosen, used = [], set()
    for _ in range(panels):
        classes = np.sort(rng.choice(np.unique(labels[allowed]), 32, replace=False))
        for cls in classes:
            candidates = np.asarray([i for i in allowed[labels[allowed] == cls] if int(i) not in used])
            pair = rng.choice(candidates, 2, replace=False)
            chosen.extend(pair.tolist())
            used.update(map(int, pair))
    return np.asarray(chosen, dtype=np.int64)


def prepare_clouds(clouds, whole_centers, child_centers, augmented, seed):
    """Keep physical crop-center points fixed across all conditions."""
    rng = np.random.RandomState(seed)
    result, whole_targets, child_targets = [], [], []
    for i, original in enumerate(clouds):
        current = original.copy()
        if augmented:
            scale = rng.uniform(2 / 3, 3 / 2, size=3)
            shift = rng.uniform(-.2, .2, size=3)
            current = (current * scale + shift).astype(np.float32)
        whole_targets.append(current[whole_centers[i]].copy())
        child_targets.append(current[child_centers[i]].copy())
        if augmented:
            rng.shuffle(current)
        result.append(current)
    return np.stack(result), np.stack(whole_targets), np.stack(child_targets)


def nearest_indices(source, targets):
    return ((source.transpose(1, 2) - targets[:, None, :]).square().sum(-1)).argmin(-1).tolist()


def condition_forward(model, points, whole_targets, child_targets, construction, device,
                      microbatch=32, child_count=400):
    raw = {"whole": [], "part": []}
    output = {"whole": [], "part": []}
    active_role = [None]

    def capture(module, _inputs, raw_ball):
        raw[active_role[0]].append(module.manifold.logmap0(raw_ball).detach().float().cpu())

    handle = model.emb.register_forward_hook(capture, prepend=True)
    try:
        with torch.no_grad():
            for start in range(0, len(points), microbatch):
                end = start + microbatch
                source = torch.from_numpy(points[start:end].copy()).to(device).transpose(1, 2).contiguous()
                wtarget = torch.from_numpy(whole_targets[start:end]).to(device)
                ctarget = torch.from_numpy(child_targets[start:end]).to(device)
                if construction == "direct_clean1024":
                    whole = source
                else:
                    _, whole, _ = get_children_alias(source, 1024, centers=nearest_indices(source, wtarget))
                before_child = whole.clone()
                parent = whole if construction == "part_alias400" else whole.clone()
                _, child, _ = get_children_alias(parent, child_count, centers=nearest_indices(parent, ctarget))
                changed = not torch.equal(whole, before_child)
                if construction != "part_alias400" and changed:
                    raise AssertionError("Nonalias condition mutated whole")
                active_role[0] = "part"
                nu, _ = model(child, emb=True)
                active_role[0] = "whole"
                mu, _ = model(whole, emb=True)
                output["part"].append(nu.detach().float().cpu())
                output["whole"].append(mu.detach().float().cpu())
    finally:
        handle.remove()
    values = {}
    private = {}
    for role in ("whole", "part"):
        tangent = torch.cat(raw[role])
        ball = torch.cat(output[role])
        norms = tangent.norm(dim=-1)
        depths = depth(ball)
        values[role] = {"raw_precap_tangent_norm": stats(norms.numpy()),
                        "depth": stats(depths.numpy()),
                        "precap_over_2_3_count": int((norms > 2.3).sum()),
                        "precap_over_2_3_fraction": float((norms > 2.3).float().mean()),
                        "output_cap_depth_count": int((depths >= 4.5999).sum()),
                        "output_cap_depth_fraction": float((depths >= 4.5999).float().mean())}
        private[role + "_raw_norm"] = norms.numpy()
        private[role + "_depth"] = depths.numpy()
    return values, private


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--panels", type=int, choices=(2, 4), default=2)
    parser.add_argument("--microbatch", type=int, default=32)
    parser.add_argument("--seed", type=int, default=22)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--fixed-ids", help="Optional prior private NPZ with sample_ids, or NPY, for exact cross-model objects")
    parser.add_argument("--include-full-train", action="store_true")
    args = parser.parse_args()
    if args.microbatch != 32:
        parser.error("Use local32 for the BN comparison")
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=False)
    started = time.time()
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    checkpoint_sha = sha(args.checkpoint)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    net = checkpoint["net"]
    if any(key.startswith("module.") for key in net):
        net = {key[7:] if key.startswith("module.") else key: value for key, value in net.items()}
    model = configure_backbone(Hype_pointMLP())
    model.load_state_dict(net, strict=True)
    model.to(device).eval().requires_grad_(False)
    original_hash = state_hash(model)
    original_buffers = {name: value.detach().clone() for name, value in model.named_buffers()}
    original_modes = {name: module.training for name, module in model.named_modules()}
    rng_before = capture_rng(device)
    data, labels, shard_hashes = [], [], {}
    for path in sorted(Path(args.data_dir).glob("ply_data_train*.h5")):
        with h5py.File(path, "r") as handle:
            data.append(np.asarray(handle["data"], dtype=np.float32)[:, :1024])
            labels.append(np.asarray(handle["label"]).reshape(-1).astype(np.int64))
        shard_hashes[path.name] = sha(path)
    data, labels = np.concatenate(data), np.concatenate(labels)
    if len(data) != 9840:
        raise AssertionError("Expected official9840 training pool")
    allowed = np.asarray(checkpoint.get("train_ids", np.arange(9840)), dtype=np.int64)
    if args.fixed_ids:
        loaded = np.load(args.fixed_ids, allow_pickle=False)
        if isinstance(loaded, np.lib.npyio.NpzFile):
            with loaded:
                ids = loaded["sample_ids"].copy().astype(np.int64)
        else:
            ids = np.asarray(loaded, dtype=np.int64)
        if (ids.ndim != 1 or len(ids) != args.panels * 64 or len(np.unique(ids)) != len(ids)
                or not np.isin(ids, allowed).all()):
            raise ValueError("Fixed IDs must be unique128/256 members of this checkpoint's training pool")
        for start in range(0, len(ids), 64):
            unique, counts = np.unique(labels[ids[start:start + 64]], return_counts=True)
            if len(unique) != 32 or not np.all(counts == 2):
                raise ValueError("Each fixed64 panel must contain32 classes x2")
    else:
        ids = select_objects(labels, allowed, args.panels, args.seed)
    rng = np.random.RandomState(args.seed + 1701)
    whole_centers = rng.randint(0, 1024, size=len(ids))
    child_centers = rng.randint(0, 1024, size=len(ids))
    prepared = {aug: prepare_clouds(data[ids], whole_centers, child_centers, aug == "original_aug", args.seed + 3001)
                for aug in ("clean", "original_aug")}
    modes = ["evalBN", "BN_train_only"]
    if args.include_full_train:
        modes.append("full_model_train_supplement")
    rows = []
    write_json(output / "progress.json", {"label": args.label, "phase": "inputs_validated",
                                          "source_epoch": checkpoint.get("epoch"), "objects": len(ids)})
    try:
        for mode, aug, construction in itertools.product(
                modes, ("clean", "original_aug"),
                ("direct_clean1024", "reorder_only1024", "part_alias400")):
            with torch.no_grad():
                for name, value in model.named_buffers():
                    value.copy_(original_buffers[name])
            restore_rng(rng_before, device)
            model.eval()
            if mode == "BN_train_only":
                for module in model.modules():
                    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                        module.train(True)
            elif mode == "full_model_train_supplement":
                model.train(True)
            condition = f"{mode}__{aug}__{construction}"
            values, private = condition_forward(model, *prepared[aug], construction, device,
                                                microbatch=args.microbatch)
            buffers_changed_during = sum(not torch.equal(value, original_buffers[name])
                                         for name, value in model.named_buffers())
            with torch.no_grad():
                for name, value in model.named_buffers():
                    value.copy_(original_buffers[name])
            restore_rng(rng_before, device)
            for name, module in model.named_modules():
                module.training = original_modes[name]
            restored = state_hash(model) == original_hash and rng_equal(capture_rng(device), rng_before)
            if not restored or any(parameter.grad is not None for parameter in model.parameters()):
                raise AssertionError("State/RNG/gradient restoration failed")
            row = {"label": args.label, "source_epoch": checkpoint.get("epoch"),
                   "condition": condition, "BN_mode": mode, "augmentation": aug,
                   "construction": construction, "training_objects": len(ids),
                   "local_microbatch": args.microbatch, "whole_points": 1024, "part_points": 400,
                   "measurements": values, "buffers_changed_during_condition": buffers_changed_during,
                   "model_BN_RNG_restored": restored, "backward_calls": 0,
                   "optimizer_updates": 0, "test_forward_calls": 0}
            write_json(output / (condition + ".json"), row)
            np.savez_compressed(output / ("private_" + condition + ".npz"), sample_ids=ids,
                                labels=labels[ids], **private)
            rows.append(row)
            write_json(output / "progress.json", {"label": args.label, "phase": "condition_complete",
                                                  "completed": len(rows), "condition": condition})
            print(json.dumps({"label": args.label, "condition": condition,
                              "whole_depth_median": values["whole"]["depth"]["median"],
                              "whole_depth_mean": values["whole"]["depth"]["mean"],
                              "whole_cap_fraction": values["whole"]["output_cap_depth_fraction"]}), flush=True)
    finally:
        with torch.no_grad():
            for name, value in model.named_buffers():
                value.copy_(original_buffers[name])
        restore_rng(rng_before, device)
        for name, module in model.named_modules():
            module.training = original_modes[name]
    if sha(args.checkpoint) != checkpoint_sha or state_hash(model) != original_hash:
        raise AssertionError("Source checkpoint or model changed")
    result = {"complete": True, "label": args.label, "source_epoch": checkpoint.get("epoch"),
              "checkpoint_sha256": checkpoint_sha, "source_training_config": checkpoint.get("training_config"),
              "training_shard_sha256": shard_hashes, "official_train_count": len(data),
              "eligible_checkpoint_training_count": len(allowed), "objects": len(ids),
              "fixed_ids_sha256": sha(args.fixed_ids) if args.fixed_ids else None,
              "selection": "fixed32 classes x2 per global64 panel; unique training objects; local32 grouping",
              "physical_crop_centers_fixed_across_conditions": True,
              "augmentation": "independent xyz scale U(2/3,1.5), shift U(-.2,.2), point-order shuffle; fixed RandomState",
              "part_before_whole": True, "conditions": rows,
              "model_BN_RNG_restored": True, "checkpoint_file_unchanged": True,
              "optimizer_updates": 0, "backward_calls": 0, "test_forward_calls": 0,
              "device": str(device), "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
              "runner_sha256": sha(__file__), "seconds": time.time() - started,
              "finished_utc": datetime.now(timezone.utc).isoformat(),
              "limits": "128/256 fixed training objects, eval or synthetic BN-only conditions; not exact historical training batches/RNG, and not a full-pool estimate."}
    # Configuration supplied by an old checkpoint may contain Path values.
    if result["source_training_config"] is not None:
        result["source_training_config"] = json.loads(json.dumps(result["source_training_config"], default=str))
    write_json(output / "summary.json", result)
    write_json(output / "progress.json", {"label": args.label, "phase": "complete", "completed": len(rows)})


if __name__ == "__main__":
    main()
