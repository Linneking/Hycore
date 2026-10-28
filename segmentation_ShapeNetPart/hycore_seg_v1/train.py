"""Auditable single-GPU baseline / HyCoRe-seg training with validation selection."""

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .data import ShapeNetPart
from .metrics import PART_COUNTS, PART_STARTS, SegMetrics
from .model import DGCNNPartSeg
from .utils import (append_json, git_identity, restore_rng, rng_state, seed_all,
                    seed_worker, sha256, state_hash, timestamp, write_json)


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--mode", choices=("baseline", "head", "hycore"), default="baseline")
    parser.add_argument("--protocol", choices=("validation", "antao_testsel"), default="validation")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--schedule-epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--num-points", type=int, default=2048)
    parser.add_argument("--k", type=int, default=40)
    parser.add_argument("--dropout", type=float, default=.5)
    parser.add_argument("--lr", type=float, default=.1)
    parser.add_argument("--min-lr", type=float, default=.001)
    parser.add_argument("--momentum", type=float, default=.9)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=4780)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--physical-gpu", type=int)
    parser.add_argument("--alpha", type=float, default=.01)
    parser.add_argument("--beta", type=float, default=.01)
    parser.add_argument("--gamma-base", type=float, default=1000.)
    parser.add_argument("--delta", type=float, default=4.)
    parser.add_argument("--part-min", type=int, default=200)
    parser.add_argument("--part-max", type=int, default=600)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--limit-train", type=int, default=0)
    parser.add_argument("--limit-val", type=int, default=0)
    parser.add_argument("--no-final-test", action="store_true")
    parser.add_argument("--resume")
    preliminary, _ = parser.parse_known_args()
    if preliminary.config:
        defaults = json.loads(Path(preliminary.config).read_text())
        unknown = set(defaults) - {a.dest for a in parser._actions}
        if unknown:
            raise ValueError(("Unknown configuration entries", unknown))
        parser.set_defaults(**defaults)
    args = parser.parse_args()
    if not 1 <= args.epochs <= args.schedule_epochs or args.num_points < args.k:
        raise ValueError("Invalid training/schedule/point budget")
    if args.mode == "hycore" and args.part_max > args.num_points:
        raise ValueError("Part size exceeds input point count")
    if (args.max_steps or args.limit_train or args.limit_val) and not args.no_final_test:
        raise ValueError("Bounded smoke runs must disable official final test")
    return args


def ensure_idle(physical_gpu):
    if physical_gpu is None or os.environ.get("CUDA_VISIBLE_DEVICES") != str(physical_gpu):
        raise ValueError("Set one physical GPU explicitly using CUDA_VISIBLE_DEVICES")
    query = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,uuid,utilization.gpu,memory.used", "--format=csv,noheader,nounits"], text=True)
    row = next(x.split(",") for x in query.splitlines() if int(x.split(",")[0]) == physical_gpu)
    gpu_uuid, utilization, memory = row[1].strip(), int(row[2]), int(row[3])
    processes = subprocess.check_output([
        "nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"], text=True)
    if gpu_uuid in processes or utilization > 0 or memory > 500:
        raise RuntimeError("Selected GPU is occupied; no training started")
    return {"physical_index": physical_gpu, "uuid": gpu_uuid,
            "preflight_utilization": utilization, "preflight_memory_mib": memory,
            "checked_at": timestamp()}


def loader(dataset, batch, workers, generator, training):
    return DataLoader(dataset, batch_size=batch, shuffle=training, drop_last=training,
                      num_workers=workers, worker_init_fn=seed_worker, generator=generator,
                      persistent_workers=False, pin_memory=torch.cuda.is_available())


def bn_counts(model):
    return {name: int(module.num_batches_tracked) for name, module in model.named_modules()
            if isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d))}


def save_checkpoint(path, value):
    temporary = Path(str(path) + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


@torch.no_grad()
def evaluate(model, data_loader, device, per_object_path=None):
    model.eval()
    before = bn_counts(model)
    raw, masked = SegMetrics(), SegMetrics()
    ce_sum, objects = 0., 0
    records = []
    for points, categories, targets, ids in data_loader:
        points, categories, targets = points.to(device), categories.to(device), targets.to(device)
        one_hot = F.one_hot(categories, 16).float()
        logits = model(points.transpose(1, 2).contiguous(), one_hot).transpose(1, 2).contiguous()
        if not torch.isfinite(logits).all():
            raise FloatingPointError("Nonfinite evaluation logits")
        ce = F.cross_entropy(logits.reshape(-1, 50), targets.reshape(-1))
        predictions = logits.argmax(-1)
        masked_logits = logits.clone()
        for row, category in enumerate(categories.tolist()):
            start, end = PART_STARTS[category], PART_STARTS[category] + PART_COUNTS[category]
            masked_logits[row, :, :start] = -torch.inf
            masked_logits[row, :, end:] = -torch.inf
        pred_np, target_np, cat_np = predictions.cpu().numpy(), targets.cpu().numpy(), categories.cpu().numpy()
        ious = raw.update(pred_np, target_np, cat_np)
        masked.update(masked_logits.argmax(-1).cpu().numpy(), target_np, cat_np)
        ce_sum += float(ce) * len(categories)
        objects += len(categories)
        if per_object_path:
            records.extend({"id": sample_id, "category": int(cat), "shape_iou": float(iou)}
                           for sample_id, cat, iou in zip(ids, cat_np, ious))
    if bn_counts(model) != before:
        raise AssertionError("Evaluation changed BN state")
    result = raw.result()
    result.update(ce=ce_sum / max(1, objects), masked_secondary=masked.result())
    if per_object_path:
        write_json(per_object_path, records)
    return result


def train(args):
    torch.set_num_threads(args.cpu_threads)
    out = Path(args.out)
    identity = git_identity()
    if identity["source_status"]:
        raise RuntimeError("Source directory is dirty; commit before running")
    gpu = ensure_idle(args.physical_gpu) if args.device == "cuda" else None
    device = torch.device(args.device)
    seed_all(args.seed)
    data_root = Path(args.data)
    data_manifest = json.loads((data_root / "manifest.json").read_text())
    if not data_manifest.get("complete"):
        raise ValueError("Data preparation not complete")
    # Verify the complete prepared identities before GPU training.
    train_split, selection_split = ("train", "val") if args.protocol == "validation" else ("trainval", "test")
    if args.protocol == "antao_testsel" and args.no_final_test:
        raise ValueError("Test-selected production must not masquerade as a no-test smoke")
    for split in (train_split, selection_split):
        if sha256(data_root / (split + ".h5")) != data_manifest["splits"][split]["file_sha256"]:
            raise ValueError("Prepared data hash mismatch: " + split)
    train_set = ShapeNetPart(data_root, train_split, args.num_points, args.limit_train or None)
    val_set = ShapeNetPart(data_root, selection_split, args.num_points, args.limit_val or None)
    if args.protocol == "validation" and set(train_set.ids) & set(val_set.ids):
        raise ValueError("Train/validation ID overlap")
    train_generator = torch.Generator().manual_seed(args.seed + 1009)
    eval_generator = torch.Generator().manual_seed(args.seed + 2003)
    train_loader = loader(train_set, args.batch_size, args.workers, train_generator, True)
    val_loader = loader(val_set, args.eval_batch_size, args.workers, eval_generator, False)
    if not len(train_loader):
        raise ValueError("No full training batch")
    model = DGCNNPartSeg("euclidean" if args.mode == "baseline" else "mobius",
                        args.k, dropout=args.dropout).to(device)
    initial_hash = state_hash(model.state_dict())
    common_hash = state_hash({k: v for k, v in model.state_dict().items() if not k.startswith("conv11.")})
    if args.mode == "baseline":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    else:
        import geoopt
        optimizer = geoopt.optim.RiemannianSGD(model.parameters(), lr=args.lr,
                                             momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.schedule_epochs, eta_min=args.min_lr)
    criterion = None
    if args.mode == "hycore" and (args.alpha or args.beta):
        from .losses import HyCoReSegLoss
        criterion = HyCoReSegLoss(args.alpha, args.beta, args.gamma_base, args.delta,
                                 args.part_min, args.part_max, args.seed).to(device)
    start_epoch, best_score, best_ce, best_epoch, updates = 0, -1., 1e30, 0, 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        previous = dict(checkpoint["config"])
        current = vars(args).copy()
        for field in ("out", "resume", "config"):
            previous.pop(field, None)
            current.pop(field, None)
        if previous != current or checkpoint["data_manifest_sha256"] != sha256(data_root / "manifest.json"):
            raise ValueError("Resume configuration or data identity differs")
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        restore_rng(checkpoint["rng"])
        train_generator.set_state(checkpoint["train_generator"].cpu())
        eval_generator.set_state(checkpoint["eval_generator"].cpu())
        if criterion:
            criterion.generator.set_state(checkpoint["part_generator"].cpu())
        start_epoch, best_epoch = checkpoint["epoch"], checkpoint["best_epoch"]
        best_score, best_ce, updates = checkpoint["best_score"], checkpoint["best_ce"], checkpoint["updates"]
        if best_epoch <= start_epoch:
            # Copy the previously chosen model into the new immutable run fork.
            source_best = Path(args.resume).parent / "best.pth"
            source_state = torch.load(source_best, map_location="cpu", weights_only=False)
            if source_state["epoch"] != best_epoch:
                raise ValueError("Missing matching prior best model")
            save_checkpoint(out / "best.pth", source_state)
    manifest = {"format": "hycore_seg_v1", "started": timestamp(), "source": identity,
                "config": vars(args), "seed": args.seed, "gpu": gpu,
                "initialization": "fresh" if not args.resume else {"checkpoint_sha256": sha256(args.resume)},
                "teacher": None, "data_manifest_sha256": sha256(data_root / "manifest.json"),
                "dataset_source": data_manifest["source"], "split_policy": data_manifest["split_policy"],
                "split_counts": {s: v["objects"] for s, v in data_manifest["splits"].items()},
                "environment": {"python": platform.python_version(), "torch": torch.__version__,
                                "cuda": torch.version.cuda, "numpy": np.__version__, "h5py": h5py.__version__},
                "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                "initial_model_sha256": initial_hash, "initial_backbone_sha256": common_hash,
                "selection": ("val_unmasked_instance_miou_then_lower_ce_then_earlier_epoch" if args.protocol == "validation"
                              else "test_unmasked_instance_miou_greater_equal_later_tie_as_antao"),
                "selection_split": selection_split,
                "train_selection_overlap_ids": len(set(train_set.ids) & set(val_set.ids)),
                "regularizer_constructed": criterion is not None}
    write_json(out / "manifest.json", manifest)
    def snapshot(epoch):
        return {"format": "hycore_seg_v1_complete", "epoch": epoch, "model": model.state_dict(),
                "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                "rng": rng_state(), "train_generator": train_generator.get_state(),
                "eval_generator": eval_generator.get_state(),
                "part_generator": criterion.generator.get_state() if criterion else None,
                "config": vars(args), "source": identity, "data_manifest_sha256": manifest["data_manifest_sha256"],
                "updates": updates, "best_epoch": best_epoch, "best_score": best_score, "best_ce": best_ce}
    save_checkpoint(out / "e0.pth", snapshot(start_epoch))
    started = time.monotonic()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for epoch in range(start_epoch + 1, args.epochs + 1):
        model.train()
        total_loss = total_ce = total_reg = 0.
        objects = steps = 0
        training_metrics = SegMetrics()
        before_bn = bn_counts(model)
        first_input_hash, first_ids = None, None
        epoch_start = time.monotonic()
        actual_lr = optimizer.param_groups[0]["lr"]
        for points, categories, targets, ids in train_loader:
            if first_input_hash is None:
                first_input_hash = state_hash({"points": points, "categories": categories, "targets": targets})
                first_ids = list(ids)
            points, categories, targets = points.to(device), categories.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            one_hot = F.one_hot(categories, 16).float()
            if criterion is None:
                logits = model(points.transpose(1, 2).contiguous(), one_hot)
                reg, diagnostics = logits.new_zeros(()), {}
            else:
                logits, features = model(points.transpose(1, 2).contiguous(), one_hot, return_features=True)
                reg, diagnostics = criterion(features, categories)
            logits = logits.transpose(1, 2).contiguous()
            ce = F.cross_entropy(logits.reshape(-1, 50), targets.reshape(-1))
            loss = ce + reg
            if criterion is not None and epoch == 1 and steps == 0:
                shared = model.conv10[0].weight
                rg = torch.autograd.grad(reg, shared, retain_graph=True)[0]
                cg = torch.autograd.grad(ce, shared, retain_graph=True)[0]
                diagnostics.update(reg_shared_gradient_norm=float(rg.detach().norm()),
                                   ce_shared_gradient_norm=float(cg.detach().norm()))
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training loss")
            loss.backward()
            grad_norm = torch.stack([p.grad.detach().square().sum() for p in model.parameters()
                                     if p.grad is not None]).sum().sqrt()
            if not torch.isfinite(grad_norm):
                raise FloatingPointError("Nonfinite parameter gradient")
            optimizer.step()
            steps, updates = steps + 1, updates + 1
            size = len(categories)
            total_loss += float(loss.detach()) * size
            total_ce += float(ce.detach()) * size
            total_reg += float(reg.detach()) * size
            objects += size
            training_metrics.update(logits.detach().argmax(-1).cpu().numpy(), targets.cpu().numpy(), categories.cpu().numpy())
            if steps == 1 or steps % 25 == 0 or (args.max_steps and steps == args.max_steps):
                write_json(out / "status.json", {"state": "training", "epoch": epoch, "step": steps,
                           "updates": updates, "loss": float(loss.detach()), "at": timestamp()})
                append_json(out / "steps.jsonl", {"epoch": epoch, "step": steps, "updates": updates,
                            "loss": float(loss.detach()), "ce": float(ce.detach()), "reg": float(reg.detach()),
                            "gradient_norm": float(grad_norm), **diagnostics})
            if args.max_steps and steps >= args.max_steps:
                break
        if any(bn_counts(model)[name] - count != steps for name, count in before_bn.items()):
            raise AssertionError("BN must update exactly once per training batch")
        if any(not torch.isfinite(p).all() for p in model.parameters()):
            raise FloatingPointError("Nonfinite parameter after epoch")
        if any(not torch.isfinite(value).all() for state in optimizer.state.values()
               for value in state.values() if torch.is_tensor(value)):
            raise FloatingPointError("Nonfinite optimizer state")
        scheduler.step()
        validation = evaluate(model, val_loader, device)
        if validation["objects"] != len(val_set):
            raise AssertionError("Incomplete validation")
        improved = (validation["instance_miou"] >= best_score if args.protocol == "antao_testsel" else
                    validation["instance_miou"] > best_score or (
                    validation["instance_miou"] == best_score and validation["ce"] < best_ce))
        if improved:
            best_score, best_ce, best_epoch = validation["instance_miou"], validation["ce"], epoch
        record = {"epoch": epoch, "steps": steps, "updates": updates, "lr_used": actual_lr,
                  "next_lr": optimizer.param_groups[0]["lr"], "objects_seen": objects,
                  "train_loss": total_loss / objects, "train_ce": total_ce / objects,
                  "train_reg": total_reg / objects, "train": training_metrics.result(), selection_split: validation,
                  "selection_split": selection_split,
                  "best_epoch": best_epoch, "bn_counts": bn_counts(model),
                  "first_input_sha256": first_input_hash, "first_ids": first_ids,
                  "seconds": time.monotonic() - epoch_start,
                  "peak_memory_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else 0}
        append_json(out / "metrics.jsonl", record)
        save_checkpoint(out / "last.pth", snapshot(epoch))
        if improved:
            save_checkpoint(out / "best.pth", snapshot(epoch))
            write_json(out / "best_selection.json", {"selected_epoch": epoch, "metrics": validation,
                       "selection_split": selection_split})
        if epoch == 1 or epoch % 50 == 0:
            save_checkpoint(out / ("epoch_%03d.pth" % epoch), snapshot(epoch))
        write_json(out / "status.json", {"state": "epoch_complete", "epoch": epoch,
                   "updates": updates, "best_epoch": best_epoch, "selection_split": selection_split,
                   "selection_instance_miou": validation["instance_miou"],
                   "at": timestamp()})
        print(json.dumps({k: record[k] for k in ("epoch", "steps", "train_loss", "best_epoch", "seconds")}
                         | {selection_split + "_instance_miou": validation["instance_miou"]}), flush=True)
    best = torch.load(out / "best.pth", map_location=device, weights_only=False)
    model.load_state_dict(best["model"], strict=True)
    if args.protocol == "antao_testsel":
        selected = json.loads((out / "best_selection.json").read_text())
        selected.update(selection="test_each_epoch", checkpoint_sha256=sha256(out / "best.pth"))
        write_json(out / "final_test.json", selected)
    elif not args.no_final_test:
        if sha256(data_root / "test.h5") != data_manifest["splits"]["test"]["file_sha256"]:
            raise ValueError("Official test identity mismatch")
        test_set = ShapeNetPart(data_root, "test", args.num_points)
        if set(test_set.ids) & (set(train_set.ids) | set(val_set.ids)):
            raise ValueError("Official test ID overlap")
        test_loader = loader(test_set, args.eval_batch_size, args.workers,
                             torch.Generator().manual_seed(args.seed + 3001), False)
        test_result = evaluate(model, test_loader, device, out / "test_objects.json")
        test_result.update(selected_epoch=best["epoch"], selection="validation", checkpoint_sha256=sha256(out / "best.pth"))
        write_json(out / "final_test.json", test_result)
    write_json(out / "acceptance.json", {"complete": True, "epochs": args.epochs,
               "updates": updates, "best_epoch": best_epoch, "selection_split": selection_split,
               "best_" + selection_split + "_instance_miou": best_score,
               "best_sha256": sha256(out / "best.pth"), "last_sha256": sha256(out / "last.pth"),
               "final_test_evaluations": (args.epochs - start_epoch if args.protocol == "antao_testsel" else
                                          0 if args.no_final_test else 1),
               "seconds": time.monotonic() - started, "ended": timestamp()})
    write_json(out / "status.json", {"state": "complete", "epoch": args.epochs,
               "updates": updates, "best_epoch": best_epoch, "at": timestamp()})


def main():
    args = arguments()
    # Acquire ownership by creating a new directory before any exception handler.
    # A refused reuse never writes even a failure marker into another run.
    Path(args.out).mkdir(parents=True, exist_ok=False)
    try:
        train(args)
    except Exception as exc:
        write_json(Path(args.out) / "failure.json", {"error": repr(exc), "at": timestamp()})
        raise


if __name__ == "__main__":
    main()
