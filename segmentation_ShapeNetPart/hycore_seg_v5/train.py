"""Fresh v5 paired D/E training: identical tail2 architecture, regularizers on/off.

Data, IoU, loader and checkpoint helpers are reused from the frozen v1 package;
the original hyperbolic operators and feature kernel remain unchanged.
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
from .model import NativeTail2
from ..hycore_seg_v3.monitoring import GeometryMonitor
from ..hycore_seg_v2.native import SOURCE_PROVENANCE, cal_loss


def arguments(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--config")
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--arm", choices=("d", "e"), required=True)
    p.add_argument("--protocol", choices=("validation", "antao_testsel"), default="antao_testsel")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--schedule-epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--eval-batch-size", type=int, default=16)
    p.add_argument("--num-points", type=int, default=2048)
    p.add_argument("--k", type=int, default=40)
    p.add_argument("--dropout", type=float, default=.5)
    p.add_argument("--lr", type=float, default=.1)
    p.add_argument("--min-lr", type=float, default=.001)
    p.add_argument("--momentum", type=float, default=.9)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--monitor-every", type=int, default=25)
    p.add_argument("--gradient-every", type=int, default=200)
    p.add_argument("--allow-dirty-cpu", action="store_true")
    p.add_argument("--seed", type=int, default=4780)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--cpu-threads", type=int, default=4)
    p.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    p.add_argument("--physical-gpu", type=int)
    p.add_argument("--alpha", type=float)
    p.add_argument("--beta", type=float)
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
    expected_coefficient = .01 if args.arm == "d" else 0.
    if args.alpha is None:
        args.alpha = expected_coefficient
    if args.beta is None:
        args.beta = expected_coefficient
    if not 1 <= args.epochs <= args.schedule_epochs:
        raise ValueError("Invalid training/schedule budget")
    if not 1 <= args.k <= args.num_points or not 1 <= args.part_min <= args.part_max <= args.num_points:
        raise ValueError("Invalid graph/part point budget")
    if args.batch_size < 2 or args.eval_batch_size < 1 or min(args.monitor_every, args.gradient_every) < 1:
        raise ValueError("Invalid batch size or monitoring interval")
    if (args.alpha, args.beta) != (expected_coefficient, expected_coefficient):
        raise ValueError("Approved D requires alpha=beta=.01; E requires alpha=beta=0")
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


def global_gradient_norm(parameters):
    """Read the total L2 norm without modifying any parameter gradient."""
    values = [p.grad.detach().square().sum() for p in parameters if p.grad is not None]
    total = torch.stack(values).sum().sqrt() if values else torch.tensor(0.)
    if not bool(torch.isfinite(total)):
        raise FloatingPointError("Nonfinite total parameter gradient")
    return total


def monitored_parameter_groups(model):
    groups = {"shared": {"conv1.weight": model.conv1[0].weight,
                         "conv8.weight": model.conv8[0].weight,
                         "bn8.weight": model.bn8.weight, "bn8.bias": model.bn8.bias}}
    groups["euclidean9"] = {"conv9.weight": model.conv9[0].weight,
                            "bn9.weight": model.bn9.weight, "bn9.bias": model.bn9.bias}
    for node, name in (("m1", "emb"), ("m2", "emb2"), ("m3", "mlr")):
        if hasattr(model, name):
            groups[node] = {key: value for key, value in getattr(model, name).named_parameters()
                            if value.requires_grad}
    return groups


def update_summary(groups, before):
    result = {}
    with torch.no_grad():
        for name, parameters in groups.items():
            squared = sum((p.detach() - before[(name, key)]).square().sum()
                          for key, p in parameters.items())
            parameter_squared = sum(before[(name, key)].square().sum() for key in parameters)
            result[name] = {"update_l2": float(squared.sqrt()),
                            "relative_update_l2": float(squared.sqrt() / parameter_squared.sqrt().clamp_min(1e-30))}
    return result


def train(args):
    torch.set_num_threads(args.cpu_threads)
    out = Path(args.out)
    identity = git_identity()
    if identity["source_status"] and not (args.device == "cpu" and args.allow_dirty_cpu):
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
    model = NativeTail2(args.arm, k=args.k, dropout=args.dropout).to(device)
    initial_hash = state_hash(model.state_dict())
    shared_head_hash = state_hash({name: tensor for name, tensor in model.state_dict().items()
                                   if name.startswith(("emb2.", "mlr."))})
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
    monitor = GeometryMonitor()
    model.monitor = criterion.monitor = monitor
    parameter_groups = monitored_parameter_groups(model)
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
        criterion.negative_generator.set_state(checkpoint["negative_generator"].cpu())
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
        "format": "hycore_seg_v5", "started": timestamp(), "source": identity,
        "source_provenance": SOURCE_PROVENANCE,
        "command": [sys.executable, *sys.argv], "config": vars(args), "seed": args.seed, "gpu": gpu,
        "initialization": "fresh" if not args.resume else {"checkpoint_sha256": sha256(args.resume)},
        "teacher": None, "initial_model_sha256": initial_hash, "initial_backbone_sha256": backbone_hash,
        "initial_shared_head_sha256": shared_head_hash, "arm": args.arm,
        "has_m1": model.has_m1, "dropout_active": True, "gradient_clipping": False,
        "architecture": model.architecture, "euclidean_conv9_retained": True,
        "feature_location": "after_conv9_bn9_leaky", "dp1_location": "before_conv9",
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
        "feature_dimension": model.feature_dim, "network_category_input": True,
        "selection": ("val_unmasked_instance_miou_then_lower_ce_then_earlier_epoch" if args.protocol == "validation"
                      else "test_unmasked_instance_miou_greater_equal_later_tie_as_antao"),
        "selection_split": selection_split, "regularizer_constructed": True,
        "regularizers_active": criterion.regularizers_active,
        "regularizer_diagnostics_only": not criterion.regularizers_active,
        "regularizer_diagnostics": "same_v3_kernel_on_detached_H_under_no_grad" if not criterion.regularizers_active
                                   else "active_v3_kernel",
        "negative_policy": "uniform_different_class_sample_with_reuse",
        "ce": "original_hutil.cal_loss_smoothing_false",
        "precision": "fp32", "autocast": False, "early_stopping": False,
    }
    write_json(out / "manifest.json", manifest)

    def snapshot(epoch):
        return {"format": "hycore_seg_v5_complete", "epoch": epoch, "model": model.state_dict(),
                "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                "rng": rng_state(), "train_generator": train_generator.get_state(),
                "eval_generator": eval_generator.get_state(), "part_generator": criterion.generator.get_state(),
                "negative_generator": criterion.negative_generator.get_state(),
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
        first_input_hash, first_ids, first_pairing = None, None, None
        gradient_norms, geometry_records = [], []
        epoch_start = time.monotonic()
        actual_lr = optimizer.param_groups[0]["lr"]
        for points, categories, targets, ids in train_loader:
            if first_input_hash is None:
                first_input_hash = state_hash({"points": points, "categories": categories, "targets": targets})
                first_ids = list(ids)
            points, categories, targets = points.to(device), categories.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            geometry_sample = steps == 0 or updates % args.monitor_every == 0
            gradient_sample = steps == 0 or updates % args.gradient_every == 0
            monitor.begin(geometry_sample or gradient_sample, epoch, steps + 1)
            one_hot = F.one_hot(categories, 16).float()
            logits, features = model(points.transpose(1, 2).contiguous(), one_hot, return_features=True)
            reg, diagnostics = criterion(features, categories)
            logits = logits.transpose(1, 2).contiguous()
            ce = plain_ce(logits, targets)
            # E uses the CE graph itself. Its detached observation-only kernel
            # must not contribute even a nominal zero regularizer graph.
            loss = ce + reg if criterion.regularizers_active else ce
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training loss")
            if first_pairing is None:
                first_pairing = {key: diagnostics[key] for key in
                                 ("negative_indices", "part_centers", "part_points", "part_indices_sha256")}
            observation = monitor.summarize() if monitor.enabled else None
            if gradient_sample:
                observation.update(monitor.gradient_summary({"ce": ce, **criterion.last_components}, parameter_groups))
            before_update = ({(name, key): p.detach().clone() for name, group in parameter_groups.items()
                              for key, p in group.items()} if gradient_sample else None)
            loss.backward()
            grad_norm = global_gradient_norm(model.parameters())
            gradient_norms.append(float(grad_norm))
            optimizer.step()
            if observation is not None:
                observation.update(updates=updates + 1, arm=args.arm,
                                   diagnostics=diagnostics, gradient_norm=float(grad_norm), clip_applied=False)
                if before_update is not None:
                    observation["parameter_updates"] = update_summary(parameter_groups, before_update)
                append_json(out / "geometry.jsonl", observation)
                geometry_records.append(observation["geometry"])
            monitor.clear()
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
                        "gradient_norm": float(grad_norm), "clip_applied": False,
                        "exceeds_old_clip_threshold": bool(grad_norm > 1.), **diagnostics}
                write_json(out / "status.json", {"state": "training", **info, "at": timestamp()})
                append_json(out / "steps.jsonl", info)
                print(json.dumps({"state": "training", **info}), flush=True)
            if args.max_steps and steps >= args.max_steps:
                break
        criterion.last_components = {}
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
        monitor.begin(False, epoch, 0)
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
                  "first_input_sha256": first_input_hash, "first_ids": first_ids, "first_pairing": first_pairing,
                  "gradient_observation": {"clip_applied": False, "mean": float(np.mean(gradient_norms)),
                                           "max": float(np.max(gradient_norms)),
                                           "p95": float(np.quantile(gradient_norms, .95)),
                                           "exceeds_one_fraction": float(np.mean(np.asarray(gradient_norms) > 1))},
                  "geometry_sample_count": len(geometry_records),
                  "seconds": time.monotonic() - epoch_start,
                  "peak_memory_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else 0}
        append_json(out / "metrics.jsonl", record)
        save_checkpoint(out / "last.pth", snapshot(epoch))
        if improved:
            save_checkpoint(out / "best.pth", snapshot(epoch))
            write_json(out / "best_selection.json", {"selected_epoch": epoch, "metrics": validation,
                                                       "selection_split": selection_split})
        if epoch in (1, 5) or epoch % 10 == 0:
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
