"""Matched ScanObjectNN DeLA Part–Whole H0 / HIER training."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import subprocess
import sys
import types
from datetime import datetime, timezone

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from loss import hyperbolic_pair_losses


ROOT = Path(__file__).resolve().parents[4]
TRAIN_SHA = "bd1ebda43155b112fac0ec9335723e66c4885449022002a2a8d370b56376eed5"
TEST_SHA = "f23822ee2f1e1d18331e91334d74adb10c8718d72d503bfa76ebe0a9348a7ea1"
WEIGHT_CANDIDATES = (0.01, 0.02, 0.05)
CALIBRATION_RATIO = 0.05


class WithRows(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        xyz, label = self.dataset[index]
        return xyz, label, index


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_hash(tensor):
    return hashlib.sha256(tensor.detach().contiguous().cpu().numpy().tobytes()).hexdigest()


def state_hash(state):
    result = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        if isinstance(tensor, torch.Tensor):
            result.update(name.encode())
            result.update(tensor_hash(tensor).encode())
    return result.hexdigest()


def save_json(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def save_checkpoint(path, state):
    temporary = path.with_suffix(".pt.tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def source_imports(options):
    sys.path[:0] = [str(options.dela_root / "ScanObjectNN"), str(ROOT)]
    # Use the author's alternate package name to avoid collisions with DeLA's imports.
    modelsnet = types.ModuleType("modelsnet")
    modelsnet.__path__ = [str(ROOT / "classification_ScanObject" / "models")]
    sys.modules["modelsnet"] = modelsnet
    import config
    config.data_path = options.data_root
    from ScanObjectNN import ScanObjectNN
    from model import DelaPartWhole
    from utils.timm.optim import create_optimizer_v2
    from utils.timm.scheduler.cosine_lr import CosineLRScheduler
    return config, ScanObjectNN, DelaPartWhole, create_optimizer_v2, CosineLRScheduler


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def loaders(options, config, dataset):
    train_data = dataset()
    if len(train_data) != 11416:
        raise ValueError(f"Unexpected train size {len(train_data)}")
    train = DataLoader(WithRows(train_data), batch_size=config.batch_size, shuffle=True,
                       drop_last=True, pin_memory=True, num_workers=options.workers,
                       persistent_workers=options.workers > 0,
                       generator=torch.Generator().manual_seed(options.seed + 200_001))
    return train


def model_batch(model, xyz, labels, rows, smoothing, part_count=None):
    xyz = xyz.cuda(non_blocking=True)
    labels = labels.cuda(non_blocking=True).view(-1).long()
    with torch.autocast("cuda", dtype=torch.float16):
        scores, coordinate, whole, part, m = model(xyz, part_count=part_count)
    with torch.autocast("cuda", enabled=False):
        classification = F.cross_entropy(scores.float(), labels, label_smoothing=smoothing)
        l_h, l_c = hyperbolic_pair_losses(whole, part, m, rows, model.manifold)
    return scores, coordinate, classification, l_h, l_c, whole, part, m, labels


def geometry_summary(model, whole, part):
    with torch.no_grad():
        whole_depth = model.manifold.dist0(whole)
        part_depth = model.manifold.dist0(part)
        result = {"whole_depth_mean": float(whole_depth.mean()),
                  "part_depth_mean": float(part_depth.mean()),
                  "whole_deeper_fraction": float((whole_depth > part_depth).float().mean())}
        for view, probes in getattr(model, "last_geometry", {}).items():
            for name, values in probes.items():
                result[f"{view}_{name}_p95"] = float(torch.quantile(values.float().flatten(), 0.95))
        return result


def calibrate(options, config, model, train):
    import itertools
    probes = []
    model.train()
    for xyz, label, rows in itertools.islice(train, 8):
        with torch.no_grad():
            _, coordinate, ce, l_h, l_c, whole, part, m, _ = model_batch(
                model, xyz, label, rows, config.label_smoothing)
        values = dict(ce=float(ce), coordinate=float(coordinate), l_h=float(l_h), l_c=float(l_c), m=m,
                      **geometry_summary(model, whole, part))
        if not all(math.isfinite(value) for value in values.values()):
            raise RuntimeError("Non-finite calibration measurement")
        probes.append(values)
    if len(probes) != 8:
        raise RuntimeError("Calibration requires eight training batches")
    ratio = statistics.median((row["l_h"] + row["l_c"]) / row["ce"] for row in probes)
    if ratio <= 0:
        raise RuntimeError("The HLoss calibration residual is zero")
    weight = min(WEIGHT_CANDIDATES, key=lambda value: abs(math.log(value * ratio / CALIBRATION_RATIO)))
    tangent = max((value for row in probes for key, value in row.items()
                   if key.endswith("tangent_norm_p95")), default=0.)
    # c=1 expmap is tanh in radius. Keep the 95th percentile below the region near the clamp.
    scale = options.input_scale if tangent <= 2.7 else options.input_scale * 2.4 / tangent
    save_json(options.output / "calibration.json", dict(
        seed=options.seed, training_batches=probes, candidates=WEIGHT_CANDIDATES,
        target_ratio=CALIBRATION_RATIO, chosen_lambda_h=weight,
        weighted_ratio=weight * ratio, input_scale=options.input_scale,
        suggested_input_scale=scale))


class HierBranch:
    def __init__(self, seed, epochs):
        from inter_hierarchy_MN40.hier_proxy_scratch_v8.hier_loss import HIERLoss
        self.loss = HIERLoss(seed=seed).cuda()
        self.optimizer = torch.optim.AdamW(self.loss.parameters(), lr=.005, eps=1e-4,
                                           betas=(.9, .999), weight_decay=.01)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, epochs, .0005)
        self.generator = torch.Generator(device="cuda").manual_seed(seed + 300_001)
        self.seed = seed

    def regularize(self, whole, labels, rows, manifold, step):
        from inter_hierarchy_MN40.hier_proxy_scratch_v8.geometry import to_ball
        with torch.autocast("cuda", enabled=False):
            view = to_ball(manifold.logmap0(whole.float()), c=.1, clip_r=2.3, hook=True)
            loss, stats = self.loss(view, labels, topk=20, proxy_topk=20,
                t_per_anchor=50, seed=self.seed + 10_007 * step, data_ids=rows,
                generator=self.generator, monitor_structure=False)
        for name in ("sample", "proxy"):
            if stats[name]["self_k_triplets"]:
                raise RuntimeError("Invalid HIER self-negative")
        return loss, stats

    def checkpoint(self):
        return dict(proxies=self.loss.state_dict(), proxy_optimizer=self.optimizer.state_dict(),
                    proxy_scheduler=self.scheduler.state_dict(), proxy_rng=self.generator.get_state())


def joint_step(loss, scaler, main_optimizer, branch):
    scaler.scale(loss).backward()
    optimizers = [main_optimizer] if branch is None else [main_optimizer, branch.optimizer]
    for optimizer in optimizers:
        scaler.unscale_(optimizer)
    grads = [parameter.grad for optimizer in optimizers for group in optimizer.param_groups
             for parameter in group["params"] if parameter.grad is not None]
    norm = torch.nn.utils.get_total_norm(grads)
    if not bool(torch.isfinite(norm)):
        scaler.update(new_scale=max(1.0, scaler.get_scale() / 2))
        return False
    for optimizer in optimizers:
        scaler.step(optimizer)
    scaler.update()
    return True


@torch.no_grad()
def evaluate(options, config, model, dataset):
    model.eval()
    if file_hash(options.data_root / "main_split" / "test_objectdataset_augmentedrot_scale75.h5") != TEST_SHA:
        raise RuntimeError("ScanObjectNN test data hash mismatch")
    eval_data = dataset(partition="test")
    if len(eval_data) != 2882:
        raise ValueError(f"Unexpected test size {len(eval_data)}")
    loader = DataLoader(eval_data, batch_size=config.batch_size, pin_memory=True,
                        num_workers=options.workers)
    confusion = torch.zeros((15, 15), dtype=torch.int64, device="cuda")
    for xyz, label in loader:
        with torch.autocast("cuda", dtype=torch.float16):
            scores = model(xyz.cuda(non_blocking=True))
        y = label.cuda(non_blocking=True).view(-1).long()
        pred = scores.argmax(1)
        confusion += torch.bincount(y * 15 + pred, minlength=225).view(15, 15)
    correct, counts = confusion.diag(), confusion.sum(1)
    return dict(oa=100 * float(correct.sum() / counts.sum()),
                aa=100 * float((correct / counts).mean()), rows=int(counts.sum()),
                confusion=confusion.cpu().tolist())


def train_pair_arm(options, config, model, train, optimizer, scheduler, dataset):
    branch = HierBranch(options.seed, config.epoch) if options.mode == "hier" else None
    scaler = torch.amp.GradScaler("cuda", init_scale=4096.0)
    initial_sha = state_hash(model.state_dict())
    initial_proxy_sha = tensor_hash(branch.loss.tangent_proxies) if branch else None
    save_json(options.output / "started.json", dict(initial_model_sha=initial_sha,
                                                     initial_proxy_sha=initial_proxy_sha))
    updates = skipped = step = 0
    total_steps = config.epoch * len(train)
    smoke_rows = []
    for epoch in range(1, (1 if options.smoke else config.epoch) + 1):
        model.train()
        metrics = defaultdict(float)
        for batch_index, (xyz, label, rows) in enumerate(train):
            part_count = (384, 511)[batch_index] if options.smoke else None
            scheduler.step(step)
            optimizer.zero_grad(set_to_none=True)
            if branch:
                branch.optimizer.zero_grad(set_to_none=True)
            scores, coordinate, ce, l_h, l_c, whole, part, m, labels = model_batch(
                model, xyz, label, rows, config.label_smoothing, part_count)
            base = ce + coordinate * (0.003 ** (step / total_steps)) / 3 + options.lambda_h * (l_h + l_c)
            whole_rng = tensor_hash(torch.cuda.get_rng_state()) if options.smoke else None
            if branch:
                regularizer, hier_stats = branch.regularize(whole, labels, rows, model.manifold, step)
                loss = base + .5 * regularizer
            else:
                loss = base
                hier_stats = None
            if not bool(torch.isfinite(loss)):
                raise RuntimeError(f"Nonfinite loss at update {step}")
            did_step = joint_step(loss, scaler, optimizer, branch)
            if options.smoke and not did_step:
                raise RuntimeError(f"Overflow during short test at update {step}")
            updates += int(did_step)
            skipped += int(not did_step)
            if branch and (options.smoke or step % 100 == 0):
                branch.loss.assert_geometry()
            if options.smoke or batch_index == 0:
                trace = dict(epoch=epoch, step=step, input_sha=tensor_hash(xyz), ids_sha=tensor_hash(rows),
                             logits_sha=tensor_hash(scores), whole_sha=tensor_hash(whole), part_sha=tensor_hash(part),
                             ce=float(ce.detach()), coordinate=float(coordinate.detach()), l_h=float(l_h.detach()),
                             l_c=float(l_c.detach()), base=float(base.detach()), main_rng=whole_rng, m=m,
                             did_step=did_step, peak_mib=torch.cuda.max_memory_allocated() // (1024 ** 2),
                             **geometry_summary(model, whole.detach(), part.detach()))
                if hier_stats:
                    trace.update(hier_samples=hier_stats["sample"]["active_triplets"],
                                 hier_proxies=hier_stats["proxy"]["active_triplets"],
                                 proxy_sha=tensor_hash(branch.loss.tangent_proxies))
                print(json.dumps(trace, allow_nan=False), flush=True)
                if options.smoke:
                    trace["rng_unchanged"] = whole_rng == tensor_hash(torch.cuda.get_rng_state())
                    trace["model_sha_after"] = state_hash(model.state_dict())
                    trace["bn_sha"] = state_hash({name: value for name, value in model.state_dict().items()
                                                  if "running_" in name or "num_batches_tracked" in name})
                    smoke_rows.append(trace)
                elif step == 0:
                    if not did_step:
                        raise RuntimeError("First formal update overflowed")
                    save_json(options.output / "first_update.json", trace)
            step += 1
            metrics["examples"] += len(labels)
            metrics["correct"] += float((scores.detach().argmax(1) == labels).sum())
            for name, value in (("ce", ce), ("coordinate", coordinate), ("l_h", l_h), ("l_c", l_c)):
                metrics[name] += float(value.detach())
            if hier_stats:
                metrics["hier_sample_loss"] += hier_stats["sample_loss"]
                metrics["hier_proxy_loss"] += hier_stats["proxy_loss"]
                metrics["hier_active_sample"] += hier_stats["sample"]["active_triplets"]
            if options.smoke and batch_index == 1:
                save_json(options.output / "smoke.json", dict(
                    seed=options.seed, input_scale=options.input_scale, lambda_h=options.lambda_h,
                    initial_model_sha=initial_sha, initial_proxy_sha=initial_proxy_sha,
                    updates=updates, rows=smoke_rows))
                return
        if branch:
            branch.scheduler.step()
        if not metrics["examples"]:
            raise RuntimeError("No training data processed")
        row = {name: value / len(train) for name, value in metrics.items()
               if name not in ("examples", "correct")}
        row.update(epoch=epoch, steps=step, updates=updates, skipped=skipped,
                   train_oa=100 * metrics["correct"] / metrics["examples"],
                   peak_mib=torch.cuda.max_memory_allocated() // (1024 ** 2))
        with (options.output / "metrics.jsonl").open("a", encoding="utf-8") as log:
            log.write(json.dumps(row, allow_nan=False) + "\n")
        print(json.dumps(row, allow_nan=False), flush=True)
        state = dict(epoch=epoch, model=model.state_dict(), optimizer=optimizer.state_dict(),
                     scheduler=scheduler.state_dict(), scaler=scaler.state_dict(),
                     updates=updates, skipped=skipped, loader_rng=train.generator.get_state(),
                     random_rng=random.getstate(), numpy_rng=np.random.get_state(),
                     torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state())
        if branch:
            state.update(branch.checkpoint())
        save_checkpoint(options.output / "last.pt", state)
    result = evaluate(options, config, model, dataset)
    save_json(options.output / "test_e250.json", result)
    print(json.dumps(result), flush=True)


def parse_options():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dela-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("h0", "hier"), required=True)
    parser.add_argument("--seed", type=int, default=4780)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--lambda-h", type=float, default=.02)
    parser.add_argument("--input-scale", type=float, default=1.)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--calibrate", action="store_true")
    options = parser.parse_args()
    if options.workers < 0 or not all(math.isfinite(value) and value > 0
                                      for value in (options.lambda_h, options.input_scale)):
        parser.error("Expected nonnegative workers and finite positive weights and scales")
    return options


def main():
    options = parse_options()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Expose exactly one CUDA device to the run")
    options.output.mkdir(parents=True, exist_ok=False)
    for path in (options.dela_root, options.data_root):
        if not path.is_absolute():
            raise ValueError("DeLA and data directories must be absolute")
    train_file = options.data_root / "main_split" / "training_objectdataset_augmentedrot_scale75.h5"
    if file_hash(train_file) != TRAIN_SHA:
        raise RuntimeError("ScanObjectNN training data hash mismatch")
    config, dataset, make_model, make_optimizer, make_scheduler = source_imports(options)
    if config.batch_size != 32 or config.epoch != 250:
        raise ValueError("Expected the DeLA ScanObjectNN author B32 and 250-epoch settings")
    torch.set_float32_matmul_precision("high")
    seed_all(options.seed)
    train = loaders(options, config, dataset)
    model = make_model(config.dela_args, options.seed + 100_001, exp_input_scale=options.input_scale).cuda()
    optimizer = make_optimizer(model, lr=config.learning_rate, weight_decay=.05)
    scheduler = make_scheduler(optimizer, t_initial=config.epoch * len(train),
        lr_min=config.learning_rate / 10000, warmup_t=config.warmup * len(train),
        warmup_lr_init=config.learning_rate / 20)
    revision = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    save_json(options.output / "manifest.json", dict(mode=options.mode, seed=options.seed,
        revision=revision, dela_file_sha=file_hash(options.dela_root / "ScanObjectNN" / "delacls.py"),
        dela_config_sha=file_hash(options.dela_root / "ScanObjectNN" / "config.py"),
        start=datetime.now(timezone.utc).isoformat(), gpu=torch.cuda.get_device_name(),
        torch=str(torch.__version__), cuda=torch.version.cuda, train_sha=TRAIN_SHA,
        batch_size=config.batch_size, batches_per_epoch=len(train), epochs=config.epoch,
        selection="fixed e250; evaluate official test once", initialized_from="random",
        lambda_h=options.lambda_h, input_scale=options.input_scale,
        smoke=options.smoke, calibration=options.calibrate, optimizer=type(optimizer).__name__,
        hier=dict(c=.1, cap=2.3, gs=True, proxy_count=512, sample_k=20, proxy_k=20, t=50,
                  lambda_sample=.5, lambda_proxy=.5, a1=True) if options.mode == "hier" else None))
    if options.calibrate:
        calibrate(options, config, model, train)
    else:
        train_pair_arm(options, config, model, train, optimizer, scheduler, dataset)


if __name__ == "__main__":
    main()
