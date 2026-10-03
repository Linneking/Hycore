"""Run the unchanged HyCoRe source defaults with passive run bookkeeping.

The production objective, data loader, random stream, two forwards, optimizer,
and 300-epoch cosine schedule are executed by classification_ModelNet40's
main_pointmlp_hycore.py. Only output paths, checkpoint I/O and observation are
wrapped. This explicitly authorized engineering baseline uses the source's
official-test evaluation every epoch and best-test selection; it is NOT the
matched global64 zero-HIER control.

Smoke limits two original-loader training/test batches and two epochs, while
retaining the 300-epoch schedule. Smoke results never initialize production.
Epoch-boundary resume restores model/optimizer/scheduler/RNG in a NEW directory;
the source's incomplete best-only resume is not used as a recovery protocol.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


REPO = Path(__file__).resolve().parents[2]
SOURCE_DIR = REPO / "classification_ModelNet40"
SOURCE = SOURCE_DIR / "main_pointmlp_hycore.py"
SOURCE_DATA = SOURCE_DIR / "data" / "modelnet40_ply_hdf5_2048"
PROTOCOL = "v6_original_hycore_b32_source_defaults_engineering_baseline"
DEFAULTS = {
    "model": "Hype_PointNet", "batch_size": 32, "epoch": 300,
    "num_points": 1024, "learning_rate": .1, "min_lr": .005,
    "weight_decay": 2e-4, "seed": 22, "workers": 8,
}


def timestamp():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def atomic_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_value(*arguments):
    return subprocess.check_output(["git", *arguments], cwd=REPO, text=True).strip()


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=SOURCE_DATA,
                        help="Must be the original source's data directory; no loader replacement")
    parser.add_argument("--seed", type=int, default=22,
                        help="Production is fixed to the original default22")
    parser.add_argument("--workers", type=int, default=8,
                        help="Production is fixed to the original default8")
    parser.add_argument("--resume", type=Path,
                        help="Complete wrapper checkpoint, restored into a NEW run directory")
    parser.add_argument("--smoke", action="store_true",
                        help="Two epochs, two original-loader train/test batches; no production initialization")
    args = parser.parse_args(argv)
    if args.seed != 22 or args.workers != 8:
        parser.error("Source-default production/smoke requires seed22 and workers8")
    args.run_dir = args.run_dir.resolve()
    if args.run_dir.exists():
        parser.error("--run-dir must be new, including resumed runs")
    args.data_dir = args.data_dir.resolve(strict=True)
    if args.data_dir != SOURCE_DATA.resolve(strict=True):
        parser.error("--data-dir must equal classification_ModelNet40/data/modelnet40_ply_hdf5_2048")
    if args.resume:
        args.resume = args.resume.resolve(strict=True)
    return args


class OutputNamespace(argparse.Namespace):
    """Redirect the source's unconditional checkpoint-path assignment only."""

    def __setattr__(self, name, value):
        if name == "checkpoint" and "checkpoint" in vars(self):
            return
        super().__setattr__(name, value)


class LimitedLoader:
    """Smoke-only truncation; production receives the untouched DataLoader."""

    def __init__(self, loader, count=2):
        self.loader, self.count = loader, count

    def __len__(self):
        return min(self.count, len(self.loader))

    def __iter__(self):
        iterator = iter(self.loader)
        for _ in range(len(self)):
            yield next(iterator)


def main():
    args = arguments()
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible or len(visible.split(",")) != 1:
        raise RuntimeError("Set CUDA_VISIBLE_DEVICES to exactly one idle physical GPU before launch")
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    args.run_dir.mkdir(parents=True, exist_ok=False)
    config = {**DEFAULTS, "steps_per_epoch": 307, "train_examples": 9840,
              "train_draws_per_epoch": 9824, "official_test_examples": 2468,
              "test_batch_size": 16, "validation_split": None,
              "smoke": args.smoke, "executed_epochs": 2 if args.smoke else 300,
              "executed_train_steps": 2 if args.smoke else 307,
              "executed_test_steps": 2 if args.smoke else 155,
              "cosine_period": 300, "test_selection": "strictly higher rounded source test OA",
              "data_dir": str(args.data_dir), "run_dir": str(args.run_dir)}
    source_hashes = {str(path.relative_to(REPO)).replace("\\", "/"): sha256(path)
                     for path in sorted(SOURCE_DIR.rglob("*.py"))}
    manifest = {"schema": 1, "protocol": PROTOCOL, "status": "preparing",
                "started_utc": timestamp(), "pid": os.getpid(),
                "commit": git_value("rev-parse", "HEAD"),
                "branch": git_value("branch", "--show-current"),
                "command": [sys.executable, *sys.argv], "config": config,
                "source_hashes": source_hashes, "completed_epochs": 0,
                "source_checkpoint_identity": None, "teacher_identity": None,
                "CUDA_VISIBLE_DEVICES": visible,
                "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                "physical_gpus": os.environ.get("HYCORE_PHYSICAL_GPUS", visible),
                "physical_gpu_uuids": os.environ.get("HYCORE_PHYSICAL_GPU_UUIDS"),
                "engineering_baseline": True, "matched_H20_zero_HIER_control": False,
                "passive_monitoring": "no extra forward, RNG draw/reseed, loss or gradient change",
                "resume_policy": "complete epoch only; source operators retained; no bitwise CUDA guarantee"}
    atomic_json(args.run_dir / "manifest.json", manifest)
    started = time.monotonic()
    installed = []
    stream = None
    try:
        # The unchanged source sets CUDA_VISIBLE_DEVICES=0 during import. Binding
        # CUDA to the launcher's single visible GPU first preserves placement.
        # The original main subsequently performs ALL source-default seeding.
        import torch
        import numpy as np
        import random
        import geoopt
        if not torch.cuda.is_available():
            raise RuntimeError("Original HyCoRe requires CUDA")
        torch.cuda.init()
        if torch.cuda.device_count() != 1:
            raise RuntimeError("The baseline requires one visible CUDA device")
        manifest["runtime"] = {"python": sys.version, "torch": torch.__version__,
            "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
            "numpy": np.__version__, "geoopt": geoopt.__version__}
        manifest["gpu"] = {"logical_index": 0, "name": torch.cuda.get_device_name(0),
            "total_memory_bytes": torch.cuda.get_device_properties(0).total_memory}
        sys.path.insert(0, str(REPO / "pointnet2_ops_lib"))
        sys.path.insert(0, str(SOURCE_DIR))
        spec = importlib.util.spec_from_file_location("_hycore_original_b32_source", SOURCE)
        original = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(original)
        finally:
            os.environ["CUDA_VISIBLE_DEVICES"] = visible
        # Record the original loader's unsorted glob order; do not sort or load
        # a second training dataset and do not replace original ModelNet40.
        data_module = sys.modules[original.ModelNet40.__module__]
        shards = {}
        for partition in ("train", "test"):
            files = data_module.glob.glob(str(args.data_dir / f"ply_data_{partition}*.h5"))
            if not files:
                raise FileNotFoundError(f"No original {partition} HDF5 shards")
            shards[partition] = [{"name": Path(path).name,
                                  "size_bytes": Path(path).stat().st_size,
                                  "sha256": sha256(Path(path))} for path in files]
        manifest["data_shards_original_glob_order"] = shards

        resume = None
        original_load = torch.load
        if args.resume:
            resume = original_load(args.resume, map_location="cpu", weights_only=False)
            if resume.get("wrapper_protocol") != PROTOCOL:
                raise ValueError("Resume requires a complete original-B32 wrapper checkpoint")
            if resume.get("source_hashes") != source_hashes:
                raise ValueError("Resume source-code identity differs")
            for key in ("seed", "workers", "epoch", "batch_size", "smoke"):
                if resume["config"][key] != config[key]:
                    raise ValueError(f"Resume config differs: {key}")
            if resume.get("data_shards") != shards:
                raise ValueError("Resume dataset shard identity/order differs")
            if not 0 < resume["completed_epochs"] < config["executed_epochs"]:
                raise ValueError("Resume must precede the requested final epoch")
            for key in ("net", "optimizer", "scheduler", "rng"):
                if key not in resume:
                    raise ValueError(f"Incomplete resume checkpoint: {key}")
            previous_best = args.resume.parent / "best_checkpoint.pth"
            previous_log = args.resume.parent / "log.txt"
            if not previous_best.is_file() or not previous_log.is_file():
                raise ValueError("Resume directory must retain original best checkpoint and log.txt")
            shutil.copyfile(previous_best, args.run_dir / "best_checkpoint.pth")
            shutil.copyfile(previous_log, args.run_dir / "log.txt")
            manifest["source_checkpoint_identity"] = {"path": str(args.resume),
                "sha256": sha256(args.resume), "completed_epochs": resume["completed_epochs"]}
            manifest["completed_epochs"] = resume["completed_epochs"]

        def patch(owner, name, replacement):
            installed.append((owner, name, getattr(owner, name)))
            setattr(owner, name, replacement)

        def capture_rng():
            return {"python": random.getstate(), "numpy": np.random.get_state(),
                    "torch_cpu": torch.get_rng_state(),
                    "torch_cuda": torch.cuda.get_rng_state_all()}

        def restore_rng(state):
            random.setstate(state["python"])
            np.random.set_state(state["numpy"])
            torch.set_rng_state(state["torch_cpu"])
            torch.cuda.set_rng_state_all(state["torch_cuda"])

        def scalar(value, name):
            result = float(value.detach()) if isinstance(value, torch.Tensor) else float(value)
            if not math.isfinite(result):
                raise FloatingPointError("Nonfinite " + name)
            return result

        epoch_index = resume["completed_epochs"] if resume else 0
        observation = {"phase": "setup", "last_intra": None}
        scheduler_holder = {}
        stream = (args.run_dir / "steps.jsonl").open("x", encoding="utf-8", buffering=1)

        original_parse = original.parse_args

        def redirected_args():
            saved_argv = sys.argv
            try:
                sys.argv = [str(SOURCE)]
                parsed = original_parse()
            finally:
                sys.argv = saved_argv
            for key, expected in DEFAULTS.items():
                if getattr(parsed, key) != expected:
                    raise ValueError(f"Original source default changed: {key}")
            parsed.checkpoint = str(args.run_dir)
            return OutputNamespace(**vars(parsed))

        patch(original, "parse_args", redirected_args)
        original_scheduler = original.CosineAnnealingLR

        def scheduler_factory(optimizer, *positional, **keywords):
            scheduler = original_scheduler(optimizer, *positional, **keywords)
            if resume:
                scheduler.load_state_dict(resume["scheduler"])
                # Constructor's initial step must not replace saved actual LR.
                optimizer.load_state_dict(resume["optimizer"])
            scheduler_holder["value"] = scheduler
            return scheduler

        patch(original, "CosineAnnealingLR", scheduler_factory)
        if resume:
            def resume_load(path, *positional, **keywords):
                if Path(path).resolve() == args.run_dir / "best_checkpoint.pth":
                    source_state = dict(resume)
                    source_state["epoch"] = resume["completed_epochs"]
                    return source_state
                return original_load(path, *positional, **keywords)
            patch(torch, "load", resume_load)

        original_ce, original_intra = original.cal_loss, original.hype_triplet_losses

        def ce_observed(*positional, **keywords):
            value = original_ce(*positional, **keywords)
            ce_value = scalar(value, "CE")
            if observation["phase"] == "training":
                observation["step_ce"] = ce_value
                t_loss, h_loss = observation["last_intra"]
                # Detached observation only; source performs its original sum.
                observation["step_loss"] = scalar(
                    value.detach() + .01 * t_loss.detach() + .01 * h_loss.detach(), "total loss")
            return value

        def intra_observed(*positional, **keywords):
            values = original_intra(*positional, **keywords)
            for index, value in enumerate(values):
                scalar(value, f"intra result{index}")
            observation["last_intra"] = (values[-2], values[-1])
            observation["step_contrastive"] = scalar(values[-2], "contrastive")
            observation["step_radial"] = scalar(values[-1], "radial")
            return values

        patch(original, "cal_loss", ce_observed)
        patch(original, "hype_triplet_losses", intra_observed)
        original_clip = torch.nn.utils.clip_grad_norm_

        def clip_observed(parameters, *positional, **keywords):
            # The exact source clipping operation performs the only gradient
            # norm reduction. Its return is the PRE-clipping norm.
            result = original_clip(parameters, *positional, **keywords)
            norm = scalar(result, "preclip gradient norm")
            if observation["phase"] == "training":
                observation["gradient_norms"].append(norm)
                observation["step_preclip_norm"] = norm
            return result

        patch(torch.nn.utils, "clip_grad_norm_", clip_observed)
        original_progress = original.progress_bar

        def progress_observed(index, total, message=None):
            result = original_progress(index, total, message)
            phase = observation["phase"]
            row = {"epoch": epoch_index + 1, "phase": phase, "step": index + 1,
                   "steps": total, "time_utc": timestamp()}
            if phase == "training":
                row.update({key: observation[key] for key in (
                    "step_ce", "step_loss", "step_contrastive", "step_radial", "step_preclip_norm")})
                row["clip_triggered"] = row["step_preclip_norm"] > 1
            stream.write(json.dumps(row, allow_nan=False) + "\n")
            atomic_json(args.run_dir / "heartbeat.json", {"status": phase,
                "epoch": epoch_index + 1, "step": index + 1, "steps": total,
                "time_utc": timestamp(), "pid": os.getpid()})
            return result

        patch(original, "progress_bar", progress_observed)
        original_train, original_validate = original.train, original.validate
        first_train = True

        def train_observed(net, loader, optimizer, *positional, **keywords):
            nonlocal first_train
            if len(loader.dataset) != 9840 or len(loader) != 307:
                raise ValueError("Original training loader must contain9840 rows/307 batches")
            if (loader.batch_size, loader.num_workers, loader.drop_last) != (32, 8, True):
                raise ValueError("Original B32/worker8/drop-last loader changed")
            if not isinstance(loader.sampler, torch.utils.data.RandomSampler) or loader.sampler.replacement:
                raise ValueError("Original training loader must shuffle without replacement")
            if first_train and resume:
                restore_rng(resume["rng"])
            first_train = False
            observation.update(phase="training", gradient_norms=[], last_intra=None,
                               epoch_start=time.monotonic(), lr_used=optimizer.param_groups[0]["lr"])
            torch.cuda.reset_peak_memory_stats()
            working_loader = LimitedLoader(loader) if args.smoke else loader
            result = original_train(net, working_loader, optimizer, *positional, **keywords)
            if len(observation["gradient_norms"]) != len(working_loader):
                raise RuntimeError("Original epoch did not perform its expected optimizer steps")
            observation["train_seconds"] = time.monotonic() - observation["epoch_start"]
            observation["train"] = result[-1]
            return result

        def validate_observed(net, loader, *positional, **keywords):
            if len(loader.dataset) != 2468 or len(loader) != 155:
                raise ValueError("Original test loader must contain2468 rows/155 batches")
            if (loader.batch_size, loader.num_workers, loader.drop_last) != (16, 8, False):
                raise ValueError("Original B16/worker8/full-test loader changed")
            observation["phase"] = "official_test"
            phase_start = time.monotonic()
            result = original_validate(net, LimitedLoader(loader) if args.smoke else loader,
                                       *positional, **keywords)
            observation["test_seconds"] = time.monotonic() - phase_start
            observation["test"] = result
            return result

        patch(original, "train", train_observed)
        patch(original, "validate", validate_observed)

        def checkpoint_observed(net, epoch, path, acc, is_best, **keywords):
            nonlocal epoch_index
            completed = epoch + 1
            state = {"net": net.state_dict(), "epoch": epoch, "acc": acc, **keywords,
                "wrapper_protocol": PROTOCOL, "completed_epochs": completed,
                "scheduler": scheduler_holder["value"].state_dict(), "rng": capture_rng(),
                "config": config, "source_hashes": source_hashes, "data_shards": shards,
                "commit": manifest["commit"], "source_run_dir": str(args.run_dir)}
            last = args.run_dir / "last_checkpoint.pth"
            temporary = last.with_name(last.name + ".tmp")
            torch.save(state, temporary)
            temporary.replace(last)
            written = [last]
            if is_best:
                best_path = args.run_dir / "best_checkpoint.pth"
                shutil.copyfile(last, best_path)
                written.append(best_path)
            if completed % 20 == 0 or args.smoke:
                archive = args.run_dir / f"checkpoint_epoch_{completed:03d}.pth"
                shutil.copyfile(last, archive)
                written.append(archive)
            norms = observation["gradient_norms"]
            report = {"epoch": completed, "protocol": PROTOCOL,
                "smoke_only": args.smoke, "training": observation["train"],
                "official_test": observation["test"],
                "lr_used": observation["lr_used"], "lr_next": keywords["optimizer"]["param_groups"][0]["lr"],
                "train_steps": len(norms), "train_draws": 32 * len(norms),
                "dataset_size": 9840, "permutation_without_replacement": True,
                "gradient": {"preclip_norm_mean": sum(norms) / len(norms),
                    "preclip_norm_max": max(norms),
                    "clip_trigger_count": sum(norm > 1 for norm in norms),
                    "clip_trigger_fraction": sum(norm > 1 for norm in norms) / len(norms)},
                "train_seconds": observation["train_seconds"],
                "test_seconds": observation["test_seconds"],
                "epoch_seconds": time.monotonic() - observation["epoch_start"],
                "peak_allocated_MiB": torch.cuda.max_memory_allocated() / 2**20,
                "peak_reserved_MiB": torch.cuda.max_memory_reserved() / 2**20,
                "cudnn_benchmark": torch.backends.cudnn.benchmark,
                "cudnn_deterministic": torch.backends.cudnn.deterministic,
                "best_test_oa": keywords["best_test_acc"],
                "is_best": is_best, "finite": True,
                "checkpoints": [{"path": str(file), "size_bytes": file.stat().st_size,
                                 "sha256": sha256(file)} for file in written]}
            atomic_json(args.run_dir / f"metrics_epoch_{completed:03d}.json", report)
            if is_best:
                manifest["best"] = {"epoch": completed, "test_oa": acc,
                    "test_aa": observation["test"]["acc_avg"], "test_ce": observation["test"]["loss"]}
            manifest.update(completed_epochs=completed, updated_utc=timestamp(),
                latest_epoch=report, wall_seconds=time.monotonic() - started)
            atomic_json(args.run_dir / "manifest.json", manifest)
            atomic_json(args.run_dir / "heartbeat.json", {"status": "epoch_completed",
                "epoch": completed, "step": len(norms), "time_utc": timestamp(), "pid": os.getpid()})
            epoch_index = completed
            observation["last_intra"] = None
            if args.smoke and completed >= 2:
                raise SmokeComplete()

        patch(original, "save_model", checkpoint_observed)
        manifest.update(status="running", updated_utc=timestamp())
        atomic_json(args.run_dir / "manifest.json", manifest)
        try:
            original.main()
        except SmokeComplete:
            if not args.smoke:
                raise
        expected = 2 if args.smoke else 300
        if manifest["completed_epochs"] != expected:
            raise RuntimeError("Original main ended before the expected final epoch")
        manifest.update(status="completed", finished_utc=timestamp(),
                        wall_seconds=time.monotonic() - started)
        atomic_json(args.run_dir / "manifest.json", manifest)
        atomic_json(args.run_dir / "heartbeat.json", {"status": "completed", "epoch": expected,
            "time_utc": timestamp(), "pid": os.getpid()})
    except BaseException as error:
        manifest.update(status="failed", failed_utc=timestamp(), error=repr(error),
                        wall_seconds=time.monotonic() - started)
        atomic_json(args.run_dir / "manifest.json", manifest)
        atomic_json(args.run_dir / "heartbeat.json", {"status": "failed", "error": repr(error),
            "time_utc": timestamp(), "pid": os.getpid()})
        raise
    finally:
        if stream is not None:
            stream.close()
        for owner, name, previous in reversed(installed):
            setattr(owner, name, previous)


class SmokeComplete(Exception):
    """Bounded smoke exit after a complete original-source epoch save."""


if __name__ == "__main__":
    main()
