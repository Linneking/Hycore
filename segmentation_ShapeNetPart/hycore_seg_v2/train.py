"""Fresh, audited training for the three approved native HyCoRe-seg designs.

Data, IoU, loader and checkpoint helpers are reused from the frozen v1 package;
the architecture, original hyperbolic operators and feature regularizer are v2.
"""

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

import geoopt
import h5py
import numpy as np
import torch
from torch.nn import functional as F

from ..hycore_seg_v1.data import ShapeNetPart
from ..hycore_seg_v1.metrics import SegMetrics
from ..hycore_seg_v1.train import (bn_counts, ensure_idle, evaluate, loader,
                                  save_checkpoint)
from ..hycore_seg_v1.utils import (append_json, git_identity, restore_rng,
                                  rng_state, seed_all, sha256, state_hash,
                                  timestamp, write_json)
from .losses import HyCoReSegLoss
from .model import NativePartSeg
from .native import SOURCE_PROVENANCE, cal_loss


def arguments(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--config")
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--architecture", choices=("p0", "pglobal", "pseg3"), default="p0")
    p.add_argument("--protocol", choices=("validation", "antao_testsel"), default="antao_testsel")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--schedule-epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--eval-batch-size", type=int, default=16)
    p.add_argument("--num-points", type=int, default=2048)
    p.add_argument("--k", type=int, default=40)
    p.add_argument("--dropout", type=float, default=.5)
    p.add_argument("--lr", type=float, default=.01)
    p.add_argument("--min-lr", type=float, default=.0001)
    p.add_argument("--momentum", type=float, default=.9)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--clip-grad", type=float, default=1.)
    p.add_argument("--seed", type=int, default=4780)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--cpu-threads", type=int, default=4)
    p.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    p.add_argument("--physical-gpu", type=int)
    p.add_argument("--alpha", type=float, default=.01)
    p.add_argument("--beta", type=float, default=.01)
    p.add_argument("--gamma-base", type=float, default=1000.)
    p.add_argument("--delta", type=float, default=4.)
    p.add_argument("--part-min", type=int, default=200)
    p.add_argument("--part-max", type=int, default=600)
    p.add_argument("--max-steps", type=int, default=0)
    p.add_argument("--limit-train", type=int, default=0)
    p.add_argument("--limit-val", type=int, default=0)
    p.add_argument("--no-final-test", action="store_true")
    p.add_argument("--expected-initial-sha256")
    p.add_argument("--resume")
    preliminary, _ = p.parse_known_args(argv)
    if preliminary.config:
        defaults = json.loads(Path(preliminary.config).read_text())
        unknown = set(defaults) - {action.dest for action in p._actions}
        if unknown:
            raise ValueError(("Unknown configuration entries", unknown))
        p.set_defaults(**defaults)
    args = p.parse_args(argv)
    if not 1 <= args.epochs <= args.schedule_epochs:
        raise ValueError("Invalid training/schedule budget")
    if not 1 <= args.k <= args.num_points or not 1 <= args.part_min <= args.part_max <= args.num_points:
        raise ValueError("Invalid graph/part point budget")
    if args.batch_size < 2 or args.eval_batch_size < 1 or args.clip_grad <= 0:
        raise ValueError("Invalid batch size or clipping threshold")
    if min(args.alpha, args.beta) <= 0:
        raise ValueError("These three approved HC runs require both regularizers")
    if min(args.max_steps, args.limit_train, args.limit_val) < 0:
        raise ValueError("Negative bounded-run limit")
    if (args.max_steps or args.limit_train or args.limit_val) and not args.no_final_test:
        raise ValueError("Bounded smoke runs must disable official test")
    if args.protocol == "antao_testsel" and args.no_final_test:
        raise ValueError("No-test smoke must explicitly use the validation protocol")
    return args


def plain_ce(logits, targets):
    """Use the complete original CE function with smoothing explicitly disabled."""
    return cal_loss(logits.reshape(-1, 50), targets, smoothing=False)


def finite_tensor_state(value):
    if torch.is_tensor(value):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(finite_tensor_state(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return all(finite_tensor_state(item) for item in value)
    return True


def gradient_probe(model, ce, regularizer):
    """Extra derivative reads reuse the same forward; BN is not run twice."""
    shared = model.conv1[0].weight
    rg = torch.autograd.grad(regularizer, shared, retain_graph=True)[0]
    cg = torch.autograd.grad(ce, shared, retain_graph=True)[0]
    head_gradients = torch.autograd.grad(regularizer, tuple(model.head_parameters()),
                                         retain_graph=True, allow_unused=True)
    head_is_none = all(value is None for value in head_gradients)
    info = {"reg_shared_gradient_norm": float(rg.detach().norm()),
            "ce_shared_gradient_norm": float(cg.detach().norm()),
            "head_regularizer_gradient_is_none": head_is_none}
    if not torch.isfinite(rg).all() or not torch.isfinite(cg).all():
        raise FloatingPointError("Nonfinite shared-branch derivative")
    if info["reg_shared_gradient_norm"] <= 0 or info["ce_shared_gradient_norm"] <= 0:
        raise AssertionError("Both objective branches must reach the shared encoder")
    if not head_is_none:
        raise AssertionError("Feature regularizer unexpectedly reaches independent hyperbolic head")
    return info


def train(args):
    torch.set_num_threads(args.cpu_threads)
    out = Path(args.out)
    identity = git_identity()
    if identity["source_status"]:
        raise RuntimeError("Source directory is dirty; commit before running")
    write_json(out / "status.json", {"state": "initializing", "at": timestamp()})
    gpu = ensure_idle(args.physical_gpu) if args.device == "cuda" else None
    device = torch.device(args.device)
    seed_all(args.seed)
    data_root = Path(args.data)
    data_manifest = json.loads((data_root / "manifest.json").read_text())
    manifest_sha = sha256(data_root / "manifest.json")
    if not data_manifest.get("complete"):
        raise ValueError("Data preparation not complete")
    train_split, selection_split = (("train", "val") if args.protocol == "validation"
                                     else ("trainval", "test"))
    for split in (train_split, selection_split):
        if sha256(data_root / (split + ".h5")) != data_manifest["splits"][split]["file_sha256"]:
            raise ValueError("Prepared data hash mismatch: " + split)
    train_set = ShapeNetPart(data_root, train_split, args.num_points, args.limit_train or None)
    selection_set = ShapeNetPart(data_root, selection_split, args.num_points, args.limit_val or None)
    overlap = set(train_set.ids) & set(selection_set.ids)
    if args.protocol == "validation" and overlap:
        raise ValueError("Train/validation ID overlap")
    train_generator = torch.Generator().manual_seed(args.seed + 1009)
    eval_generator = torch.Generator().manual_seed(args.seed + 2003)
    train_loader = loader(train_set, args.batch_size, args.workers, train_generator, True)
    selection_loader = loader(selection_set, args.eval_batch_size, args.workers, eval_generator, False)
    if not len(train_loader):
        raise ValueError("No full training batch")
    model = NativePartSeg(args.architecture, k=args.k, dropout=args.dropout).to(device)
    initial_hash = state_hash(model.state_dict())
    if args.expected_initial_sha256 and initial_hash != args.expected_initial_sha256:
        raise AssertionError("Fresh production initialization differs from accepted smoke")
    head_ids = {id(parameter) for parameter in model.head_parameters()}
    backbone_hash = state_hash({name: parameter for name, parameter in model.named_parameters()
                                if id(parameter) not in head_ids})
    optimizer = geoopt.optim.RiemannianSGD(model.parameters(), lr=args.lr,
                                          momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.schedule_epochs, eta_min=args.min_lr)
    criterion = HyCoReSegLoss(args.alpha, args.beta, args.gamma_base, args.delta,
                             args.part_min, args.part_max, args.seed,
                             feature_dim=model.feature_dim).to(device)
    start_epoch, best_epoch, updates = 0, 0, 0
    best_score, best_ce = -1., float("inf")
    prior_best, prior_selection = None, None
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        previous, current = dict(checkpoint["config"]), vars(args).copy()
        for field in ("out", "resume", "config", "expected_initial_sha256"):
            previous.pop(field, None)
            current.pop(field, None)
        if previous != current or checkpoint["data_manifest_sha256"] != manifest_sha:
            raise ValueError("Resume configuration or data identity differs")
        if checkpoint["source"]["commit"] != identity["commit"]:
            raise ValueError("Resume source commit differs")
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        restore_rng(checkpoint["rng"])
        train_generator.set_state(checkpoint["train_generator"].cpu())
        eval_generator.set_state(checkpoint["eval_generator"].cpu())
        criterion.generator.set_state(checkpoint["part_generator"].cpu())
        start_epoch, best_epoch, updates = checkpoint["epoch"], checkpoint["best_epoch"], checkpoint["updates"]
        best_score, best_ce = checkpoint["best_score"], checkpoint["best_ce"]
        if start_epoch >= args.epochs:
            raise ValueError("Resume checkpoint already reached the requested epoch budget")
        prior_best = torch.load(Path(args.resume).parent / "best.pth", map_location="cpu", weights_only=False)
        if prior_best["epoch"] != best_epoch or prior_best["data_manifest_sha256"] != manifest_sha:
            raise ValueError("Missing matching prior best model")
        prior_selection = json.loads((Path(args.resume).parent / "best_selection.json").read_text())
        if prior_selection["selected_epoch"] != best_epoch:
            raise ValueError("Missing matching prior best selection record")
        save_checkpoint(out / "best.pth", prior_best)
    manifest = {
        "format": "hycore_seg_v2", "started": timestamp(), "source": identity,
        "source_provenance": SOURCE_PROVENANCE,
        "command": [sys.executable, *sys.argv], "config": vars(args), "seed": args.seed, "gpu": gpu,
        "initialization": "fresh" if not args.resume else {"checkpoint_sha256": sha256(args.resume)},
        "teacher": None, "initial_model_sha256": initial_hash, "initial_backbone_sha256": backbone_hash,
        "data_manifest_sha256": manifest_sha, "dataset_source": data_manifest["source"],
        "split_policy": data_manifest["split_policy"],
        "split_counts": {name: info["objects"] for name, info in data_manifest["splits"].items()},
        "train_objects": len(train_set), "selection_objects": len(selection_set),
        "train_selection_overlap_ids": len(overlap), "train_shuffle": True, "train_drop_last": True,
        "selection_shuffle": False, "selection_drop_last": False,
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "cuda": torch.version.cuda, "geoopt": geoopt.__version__,
                        "numpy": np.__version__, "h5py": h5py.__version__},
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "head_parameters": sum(p.numel() for p in model.head_parameters()),
        "feature_dimension": model.feature_dim, "network_category_input": args.architecture == "pseg3",
        "selection": ("val_unmasked_instance_miou_then_lower_ce_then_earlier_epoch" if args.protocol == "validation"
                      else "test_unmasked_instance_miou_greater_equal_later_tie_as_antao"),
        "selection_split": selection_split, "regularizer_constructed": True,
        "negative_policy": "original_batch_flip_no_category_filter",
        "ce": "original_hutil.cal_loss_smoothing_false",
        "precision": "fp32", "autocast": False, "early_stopping": False,
    }
    write_json(out / "manifest.json", manifest)

    def snapshot(epoch):
        return {"format": "hycore_seg_v2_complete", "epoch": epoch, "model": model.state_dict(),
                "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                "rng": rng_state(), "train_generator": train_generator.get_state(),
                "eval_generator": eval_generator.get_state(), "part_generator": criterion.generator.get_state(),
                "config": vars(args), "source": identity, "data_manifest_sha256": manifest_sha,
                "initial_model_sha256": initial_hash, "updates": updates,
                "best_epoch": best_epoch, "best_score": best_score, "best_ce": best_ce}

    save_checkpoint(out / "e0.pth", snapshot(start_epoch))
    if prior_best is not None:
        write_json(out / "best_selection.json", {**prior_selection, "copied_from_resume": True})
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
            logits, features = model(points.transpose(1, 2).contiguous(), one_hot, return_features=True)
            reg, diagnostics = criterion(features, categories)
            logits = logits.transpose(1, 2).contiguous()
            ce = plain_ce(logits, targets)
            loss = ce + reg
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training loss")
            if steps == 0 and epoch == start_epoch + 1:
                diagnostics.update(gradient_probe(model, ce, reg))
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad,
                                                       error_if_nonfinite=True)
            clipped_norm = torch.stack([p.grad.detach().square().sum() for p in model.parameters()
                                         if p.grad is not None]).sum().sqrt()
            if not torch.isfinite(clipped_norm):
                raise FloatingPointError("Nonfinite clipped gradient")
            optimizer.step()
            steps, updates = steps + 1, updates + 1
            size = len(categories)
            total_loss += float(loss.detach()) * size
            total_ce += float(ce.detach()) * size
            total_reg += float(reg.detach()) * size
            objects += size
            training_metrics.update(logits.detach().argmax(-1).cpu().numpy(), targets.cpu().numpy(),
                                     categories.cpu().numpy())
            if steps == 1 or steps % 25 == 0 or (args.max_steps and steps == args.max_steps):
                info = {"epoch": epoch, "step": steps, "updates": updates,
                        "loss": float(loss.detach()), "ce": float(ce.detach()), "reg": float(reg.detach()),
                        "gradient_norm": float(grad_norm), "gradient_norm_after_clip": float(clipped_norm),
                        "clipping_triggered": bool(grad_norm > args.clip_grad), **diagnostics}
                write_json(out / "status.json", {"state": "training", **info, "at": timestamp()})
                append_json(out / "steps.jsonl", info)
                print(json.dumps({"state": "training", **info}), flush=True)
            if args.max_steps and steps >= args.max_steps:
                break
        after_bn = bn_counts(model)
        if any(after_bn[name] - count != steps for name, count in before_bn.items()):
            raise AssertionError("BN must update exactly once per training batch")
        model_finite = finite_tensor_state(model.state_dict())
        optimizer_finite = finite_tensor_state(optimizer.state_dict())
        if not model_finite or not optimizer_finite:
            raise FloatingPointError("Nonfinite model or optimizer state after epoch")
        scheduler.step()
        write_json(out / "status.json", {"state": "evaluating", "epoch": epoch, "updates": updates,
                                          "selection_split": selection_split, "at": timestamp()})
        validation = evaluate(model, selection_loader, device)
        if validation["objects"] != len(selection_set):
            raise AssertionError("Incomplete selection evaluation")
        improved = (validation["instance_miou"] >= best_score if args.protocol == "antao_testsel" else
                    validation["instance_miou"] > best_score or (
                        validation["instance_miou"] == best_score and validation["ce"] < best_ce))
        if improved:
            best_score, best_ce, best_epoch = validation["instance_miou"], validation["ce"], epoch
        record = {"epoch": epoch, "steps": steps, "updates": updates, "lr_used": actual_lr,
                  "next_lr": optimizer.param_groups[0]["lr"], "objects_seen": objects,
                  "train_loss": total_loss / objects, "train_ce": total_ce / objects,
                  "train_reg": total_reg / objects, "train": training_metrics.result(),
                  selection_split: validation, "selection_split": selection_split, "best_epoch": best_epoch,
                  "bn_counts": after_bn, "model_finite": model_finite, "optimizer_finite": optimizer_finite,
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
        write_json(out / "status.json", {"state": "epoch_complete", "epoch": epoch, "updates": updates,
                                          "best_epoch": best_epoch, "selection_split": selection_split,
                                          "selection_instance_miou": validation["instance_miou"], "at": timestamp()})
        print(json.dumps({key: record[key] for key in ("epoch", "steps", "train_loss", "best_epoch", "seconds")}
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
        if set(test_set.ids) & (set(train_set.ids) | set(selection_set.ids)):
            raise ValueError("Official test ID overlap")
        test_loader = loader(test_set, args.eval_batch_size, args.workers,
                             torch.Generator().manual_seed(args.seed + 3001), False)
        final_test = evaluate(model, test_loader, device, out / "test_objects.json")
        final_test.update(selected_epoch=best["epoch"], selection="validation",
                           checkpoint_sha256=sha256(out / "best.pth"))
        write_json(out / "final_test.json", final_test)
    write_json(out / "acceptance.json", {
        "complete": True, "epochs": args.epochs, "updates": updates, "best_epoch": best_epoch,
        "selection_split": selection_split, "best_" + selection_split + "_instance_miou": best_score,
        "initial_model_sha256": initial_hash, "model_finite": model_finite, "optimizer_finite": optimizer_finite,
        "best_sha256": sha256(out / "best.pth"), "last_sha256": sha256(out / "last.pth"),
        "final_test_evaluations": (args.epochs - start_epoch if args.protocol == "antao_testsel" else
                                   0 if args.no_final_test else 1),
        "seconds": time.monotonic() - started, "ended": timestamp()})
    write_json(out / "status.json", {"state": "complete", "epoch": args.epochs, "updates": updates,
                                      "best_epoch": best_epoch, "at": timestamp()})


def main():
    args = arguments()
    # Refuse directory reuse before writing any status/failure into an old run.
    Path(args.out).mkdir(parents=True, exist_ok=False)
    try:
        train(args)
    except Exception as exc:
        write_json(Path(args.out) / "failure.json", {"error": repr(exc), "at": timestamp()})
        write_json(Path(args.out) / "status.json", {"state": "failed", "error": repr(exc), "at": timestamp()})
        raise


if __name__ == "__main__":
    main()
