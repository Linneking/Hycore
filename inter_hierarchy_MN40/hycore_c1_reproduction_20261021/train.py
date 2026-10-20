"""From-scratch official c1 HyCoRe; original model/train/validate are reused unchanged."""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time

C = 1.0
NAME = "ORIG-B0-32-C1-CAPnone-GS0-M1000_4-TS-S4780-R2"


def now():
    return dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).isoformat()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def digest(path):
    result = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def state_digest(model):
    result = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        result.update(name.encode())
        result.update(str(value.dtype).encode())
        result.update(str(tuple(value.shape)).encode())
        result.update(value.numpy().tobytes())
    return result.hexdigest()


def argument_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--steps-per-epoch", type=int)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--source-commit", default=None)
    parser.add_argument("--source-branch", default=None)
    return parser


def validate_arguments(args):
    if args.epochs < 1:
        raise ValueError("epochs must be positive")
    if args.smoke:
        if args.epochs != 2 or args.steps_per_epoch != 2:
            raise ValueError("smoke is strictly two train-only epochs of two steps")
    elif args.epochs != 300 or args.steps_per_epoch is not None:
        raise ValueError("production is strictly 300 complete original epochs")
    if args.run_dir.exists():
        raise FileExistsError("Refusing to reuse any existing run directory")
    if not args.data_dir.is_dir():
        raise FileNotFoundError(args.data_dir)


def import_original(repo):
    """Import official functions while preserving the launcher's GPU selection."""
    source_dir = repo / "classification_ModelNet40"
    # The immutable snapshot carries the verified extension beside its Python package.
    sys.path.insert(0, str(repo / "pointnet2_ops_lib"))
    sys.path.insert(0, str(source_dir))
    previous = {key: os.environ.get(key) for key in ("CUDA_DEVICE_ORDER", "CUDA_VISIBLE_DEVICES")}
    try:
        spec = importlib.util.spec_from_file_location("hycore_original_c1_reproduction_source", source_dir / "main_pointmlp_hycore.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    return module


def install_data_root(original_data_module, data_dir):
    """Reuse original ModelNet40/augmentation and original unsorted glob order."""
    import h5py
    import numpy as np

    loaded = {}

    def load_data(partition):
        arrays, labels = [], []
        files = glob.glob(str(data_dir / ("ply_data_%s*.h5" % partition)))
        if not files:
            raise FileNotFoundError("No HDF5 files for partition " + partition)
        # Cross-host directory enumeration must not silently change the baseline rows.
        order = os.environ.get("HYCORE_TRAIN_SHARD_ORDER") if partition == "train" else None
        if order:
            names = json.loads(order)
            if len(names) != len(set(names)) or set(names) != {Path(p).name for p in files}:
                raise ValueError("Requested original training shard order does not match data")
            by_name = {Path(p).name: p for p in files}
            files = [by_name[name] for name in names]
        for filename in files:
            with h5py.File(filename, "r") as handle:
                arrays.append(handle["data"][:].astype("float32"))
                labels.append(handle["label"][:].astype("int64"))
        data, label = np.concatenate(arrays, axis=0), np.concatenate(labels, axis=0)
        loaded[partition] = {"size": len(data), "file_order": [Path(p).name for p in files],
                             "file_sha256": {Path(p).name: digest(p) for p in files},
                             "labels_sha256": hashlib.sha256(label.tobytes()).hexdigest()}
        return data, label

    original_data_module.load_data = load_data
    return loaded


def configure_model(model):
    """Verify original defaults; do not replace modules, register hooks or change weights."""
    import torch
    manifolds = (model.manifold, model.manifold2, model.emb.manifold, model.classifier.manifold)
    for manifold in manifolds:
        if abs(float(manifold.c) - 1.0) > 1e-6:
            raise AssertionError("All four original operations must keep c=1")
        if any(parameter.requires_grad for parameter in manifold.parameters()):
            raise AssertionError("Official curvature must already be fixed")
    for module in model.modules():
        if module._forward_hooks or module._backward_hooks:
            raise AssertionError("Native original model must have no added cap/GS hooks")
    if any(parameter.dtype != torch.float32 for parameter in model.parameters()):
        raise AssertionError("Official reproduction requires FP32 parameters")
    return model


class LimitedLoader:
    """Training-only smoke; never open the official test partition."""
    def __init__(self, loader, length):
        self.loader, self.length = loader, length

    def __len__(self):
        return self.length

    def __iter__(self):
        iterator = iter(self.loader)
        for _ in range(self.length):
            yield next(iterator)


def rng_state(torch, np):
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(), "torch_cuda": torch.cuda.get_rng_state_all()}


def checkpoint_state(net, epoch, optimizer, scheduler, metrics, best, torch, np, config):
    return {"net": net.state_dict(), "epoch": epoch, "completed_epochs": epoch + 1,
            "acc": None if metrics["test"] is None else metrics["test"]["acc"],
            "best_test_acc": best["test_acc"], "best_train_acc": best["train_acc"],
            "best_test_acc_avg": best["test_acc_avg"], "best_train_acc_avg": best["train_acc_avg"],
            "best_test_loss": best["test_loss"], "best_train_loss": best["train_loss"],
            "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
            "rng": rng_state(torch, np), "metrics": metrics, "training_config": config}


def source_identity(repo):
    paths = sorted((repo / "classification_ModelNet40").rglob("*.py"))
    paths += sorted(Path(__file__).parent.rglob("*.py"))
    return {str(path.relative_to(repo)): digest(path) for path in paths}


def training_config(args):
    return {"c": C, "clip_r": None,
              "gradient_hook": False, "radial_gamma": 1000, "contrastive_margin": 4,
              "seed": 4780, "batch_size": 32, "eval_batch_size": 16, "workers": 8,
              "num_points": 1024, "epochs": args.epochs, "smoke": args.smoke,
              "steps_per_epoch": 2 if args.smoke else 307,
              "optimizer": "RiemannianSGD", "learning_rate": .1, "min_lr": .005,
              "momentum": .9, "weight_decay": .0002, "clip_grad_norm": 1,
              "rcontr_coefficient": .01, "rhier_coefficient": .01,
              "selection": "none_train_only_smoke" if args.smoke else "best_rounded_test_OA_original",
              "original_alias_crops": True, "part_before_whole": True,
              "train_shard_order": json.loads(os.environ.get("HYCORE_TRAIN_SHARD_ORDER", "null")),
              "gradient_hook_scope": "none", "hier_loss": False, "proxy_parameter_count": 0,
              "precision": "FP32", "cosine_period": 300, "replicate_id": "R2",
              "data_loader_generator": "default_global_rng", "cudnn_benchmark": True,
              "cudnn_deterministic": True, "ce_label_smoothing": .2, "resume": None}


def main(argv=None):
    args = argument_parser().parse_args(argv)
    validate_arguments(args)
    repo = Path(__file__).resolve().parents[2]
    original = import_original(repo)
    import data as original_data
    import numpy as np
    import torch
    import geoopt
    from torch.utils.data import DataLoader
    from torch.optim.lr_scheduler import CosineAnnealingLR

    if not torch.cuda.is_available():
        raise RuntimeError("Original PointMLP FPS requires CUDA")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Exactly one completely idle GPU must be assigned by the launcher")
    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
    torch.manual_seed(4780)
    np.random.seed(4780)
    torch.cuda.manual_seed_all(4780)
    torch.cuda.manual_seed(4780)
    torch.set_printoptions(10)
    random.seed(4780)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ["PYTHONHASHSEED"] = "4780"

    config = training_config(args)
    args.run_dir.mkdir(parents=True, exist_ok=False)
    output = args.run_dir
    commit = args.source_commit or os.environ.get("HYCORE_C1_SOURCE_COMMIT")
    if commit is None:
        try:
            commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
        except (OSError, subprocess.SubprocessError):
            commit = "unknown_snapshot_commit_not_supplied"
    manifest = {"status": "initializing", "started_at": now(), "updated_at": now(),
                "completed_epochs": 0, "commit": commit, "source_branch": args.source_branch or os.environ.get("HYCORE_C1_SOURCE_BRANCH"),
                "canonical_name": NAME,
                "official_test_read": False, "train_size": 9840, "validation_size": 0,
                "test_size": 0 if args.smoke else 2468, "training_config": config,
                "source_sha256": source_identity(repo), "replicate_id": "R2",
                "comparison_reference": "same-environment ORIG c0.1 ablations; historical c1 had early resume", "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "physical_gpus": os.environ.get("HYCORE_PHYSICAL_GPUS"),
                "physical_gpu_uuids": os.environ.get("HYCORE_PHYSICAL_GPU_UUIDS"),
                "command_argv": [sys.executable, *sys.argv],
                "gpu_name": torch.cuda.get_device_name(0), "pid": os.getpid(),
                "teacher": None, "checkpoint_initialization": "from_scratch", "variant": "original_c1_fresh",
                "library_versions": {"torch": torch.__version__, "numpy": np.__version__, "geoopt": geoopt.__version__}}
    atomic_json(output / "manifest.json", manifest)
    atomic_json(output / "config.json", config)
    best = {"test_acc": 0., "test_acc_avg": 0., "train_acc": 0., "train_acc_avg": 0.,
            "test_loss": float("inf"), "train_loss": float("inf")}
    current = {"epoch": 0, "phase": "initializing", "steps": 0, "epoch_steps": 0}
    try:
        model = configure_model(original.Hype_pointMLP())
        manifest["initialized_model_sha256"] = state_digest(model)
        expected_init = os.environ.get("HYCORE_C1_EXPECTED_INIT")
        if expected_init and manifest["initialized_model_sha256"] != expected_init:
            raise AssertionError("Fresh original c1 initialization differs from the CPU gate")
        atomic_json(output / "manifest.json", manifest)
        torch.save({"net": model.state_dict(), "seed": 4780, "completed_epochs": 0,
                    "initialized_model_sha256": manifest["initialized_model_sha256"],
                    "canonical_name": NAME, "config": config}, output / "initialization_e0.pth")
        model = model.to("cuda")
        net = torch.nn.DataParallel(model)
        torch.backends.cudnn.benchmark = True  # Literal official main behavior after seeding.
        loaded = install_data_root(original_data, args.data_dir.resolve())
        train_set = original_data.ModelNet40(partition="train", num_points=1024)
        if len(train_set) != 9840:
            raise AssertionError("Official training partition must contain 9840 objects")
        reference_path = os.environ.get("HYCORE_C1_REFERENCE_MANIFEST")
        if reference_path:
            reference_train = json.loads(Path(reference_path).read_text())["dataset_identity"]["train"]
            if loaded["train"] != reference_train:
                raise AssertionError("Native c1 training rows differ from the ORIG reference")
        train_loader = DataLoader(train_set, num_workers=8, batch_size=32, shuffle=True, drop_last=True)
        test_loader = None
        if not args.smoke:
            manifest["official_test_read"] = True
            atomic_json(output / "manifest.json", manifest)
            test_set = original_data.ModelNet40(partition="test", num_points=1024)
            if len(test_set) != 2468:
                raise AssertionError("Official test partition must contain 2468 objects")
            test_loader = DataLoader(test_set, num_workers=8, batch_size=16, shuffle=False, drop_last=False)
        manifest["dataset_identity"] = loaded
        manifest["status"] = "running"
        atomic_json(output / "manifest.json", manifest)
        optimizer = geoopt.optim.RiemannianSGD(net.parameters(), lr=.1, momentum=.9, weight_decay=.0002)
        scheduler = CosineAnnealingLR(optimizer, 300, eta_min=.005, last_epoch=-1)

        # This observer runs after each original optimizer step; no RNG draw or new forward.
        def progress(batch_index, total, message=None):
            heartbeat = {"status": "running", "updated_at": now(), "epoch": current["epoch"],
                         "completed_epochs": current["epoch"], "phase": current["phase"],
                         "batch": batch_index + 1, "batches": total, "optimizer_steps": current["steps"]}
            if current["phase"] == "train":
                current["steps"] += 1
                current["epoch_steps"] += 1
                heartbeat["optimizer_steps"] = current["steps"]
                with (output / "steps.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps({**heartbeat, "original_progress": message}, ensure_ascii=False) + "\n")
            atomic_json(output / "heartbeat.json", heartbeat)
            if (batch_index + 1) % 25 == 0 or batch_index + 1 == total:
                print(json.dumps(heartbeat), flush=True)

        original.progress_bar = progress
        for epoch in range(args.epochs):
            started = time.monotonic()
            current.update(epoch=epoch, phase="train", epoch_steps=0)
            effective_lr = optimizer.param_groups[0]["lr"]
            loader = LimitedLoader(train_loader, 2) if args.smoke else train_loader
            train_output = original.train(net, loader, optimizer, original.cal_loss, original.hype_triplet_losses, "cuda")
            training = train_output[-1]
            if current["epoch_steps"] != (2 if args.smoke else 307):
                raise AssertionError("Unexpected number of original training updates")
            testing = None
            if not args.smoke:
                current["phase"] = "test"
                before_bn = {name: value.detach().clone() for name, value in net.named_buffers()}
                testing = original.validate(net, test_loader, original.cal_loss, "cuda")
                if any(not torch.equal(before_bn[name], value) for name, value in net.named_buffers()):
                    raise AssertionError("Official eval must not change running BN or model buffers")
            scheduler.step()
            selected = testing is not None and testing["acc"] > best["test_acc"]
            best["train_acc"] = max(best["train_acc"], training["acc"])
            best["train_acc_avg"] = max(best["train_acc_avg"], training["acc_avg"])
            best["train_loss"] = min(best["train_loss"], training["loss"])
            if testing is not None:
                best["test_acc"] = max(best["test_acc"], testing["acc"])
                best["test_acc_avg"] = max(best["test_acc_avg"], testing["acc_avg"])
                best["test_loss"] = min(best["test_loss"], testing["loss"])
            epoch_metrics = {"epoch": epoch, "completed_epochs": epoch + 1, "train": training,
                             "test": testing, "effective_lr": effective_lr,
                             "lr_after_scheduler": optimizer.param_groups[0]["lr"],
                             "optimizer_steps": current["steps"], "epoch_optimizer_steps": current["epoch_steps"],
                             "wall_seconds": time.monotonic() - started,
                             "last_batch": {key: float(value.detach().cpu()) if torch.is_tensor(value) else float(value)
                                            for key, value in zip(("ce", "rcontr", "rhier", "positive_distance", "negative_distance", "whole_radius", "part_radius"), train_output[1:-1])},
                             "selected_best": selected, "finished_at": now()}
            if not all(bool(torch.isfinite(value).all()) for value in net.state_dict().values()):
                raise FloatingPointError("Nonfinite model state")
            state = checkpoint_state(net, epoch, optimizer, scheduler, epoch_metrics, best, torch, np, config)
            temporary = output / "last_checkpoint.pth.tmp"
            torch.save(state, temporary)
            temporary.replace(output / "last_checkpoint.pth")
            if selected:
                shutil.copyfile(output / "last_checkpoint.pth", output / "best_checkpoint.pth")
                manifest["best_epoch"] = epoch
                manifest["best_completed_epochs"] = epoch + 1
                manifest["best_test_OA"] = testing["acc"]
            if epoch + 1 in (1, 5, 10, 50, 100, 150, 200, 250, 300) or (epoch + 1) % 20 == 0:
                shutil.copyfile(output / "last_checkpoint.pth", output / ("checkpoint_epoch_%03d.pth" % (epoch + 1)))
            atomic_json(output / ("metrics_epoch_%03d.json" % (epoch + 1)), epoch_metrics)
            with (output / "metrics.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(epoch_metrics, ensure_ascii=False) + "\n")
            manifest.update(completed_epochs=epoch + 1, optimizer_steps=current["steps"], updated_at=now())
            atomic_json(output / "manifest.json", manifest)
            print(json.dumps(epoch_metrics), flush=True)
        manifest.update(status="completed", finished_at=now(), updated_at=now())
        manifest["last_sha256"] = digest(output / "last_checkpoint.pth")
        if (output / "best_checkpoint.pth").exists():
            manifest["best_sha256"] = digest(output / "best_checkpoint.pth")
        atomic_json(output / "manifest.json", manifest)
        atomic_json(output / "heartbeat.json", {"status": "completed", "completed_epochs": args.epochs,
                                                "optimizer_steps": current["steps"], "updated_at": now()})
    except BaseException as error:
        manifest.update(status="failed", failure_type=type(error).__name__, failure=str(error), updated_at=now())
        atomic_json(output / "manifest.json", manifest)
        atomic_json(output / "heartbeat.json", {"status": "failed", **current, "updated_at": now()})
        raise


if __name__ == "__main__":
    main()
