"""Freeze reviewed V9 sources, verify them, and run one idle dual-GPU diagnostic."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from inter_hierarchy_MN40.hier_proxy_scratch_v8.queue import (
    now, write, query_gpus, idle_gpus, device_environment, run_logged,
)
from inter_hierarchy_MN40.hier_proxy_scratch_v9.queue import freeze_source

BASE_RELATIVE = Path("docs/research/diagnostics/v9_proxy_cap_cause_20261021")
RELATIVE = Path("docs/research/diagnostics/v9_lr_eps_20261022")


def freeze_diagnostics(repo, root, snapshot):
    identity = json.loads((root / "source_identity.json").read_text())
    for relative in (BASE_RELATIVE, RELATIVE):
        target = snapshot / relative
        target.mkdir(parents=True, exist_ok=False)
        for source in (repo / relative).glob("*.py"):
            destination = target / source.name
            shutil.copy2(source, destination)
            identity["files"][str(relative / source.name)] = hashlib.sha256(destination.read_bytes()).hexdigest()
    write(root / "source_identity.json", identity)


def smoke_accepted(manifest):
    config = manifest["training_config"]
    optimizer = config["proxy_optimizer"]
    return (manifest["status"] == "completed" and manifest["completed_epochs"] == 2
        and not manifest["official_test_read"] and not manifest["test_used_for_selection"]
        and manifest["selection_size"] == 0 and config["epochs"] == 300
        and optimizer["lr"] == .005 and optimizer["eps"] == 1e-4
        and optimizer["min_lr"] == .0005 and config["margin"] == .1
        and config["tau"] == .1 and config["lambda_hier_after_warmup"] == .5)


def safe_query(state, status_path):
    try:
        return query_gpus()
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, OSError) as exc:
        state.update(gpu_query_error=repr(exc), gpu_query_failed_utc=now(), updated_utc=now())
        write(status_path, state)
        return []  # Unknown resources are never considered idle.


def verify_saved_optimizer(run_dir, completed_epochs):
    import torch
    checkpoint = torch.load(run_dir / "last.pth", map_location="cpu", weights_only=False)
    groups = checkpoint["proxy_optimizer"]["param_groups"]
    scheduler = checkpoint["proxy_scheduler"]
    expected_lr = .0005 + (.005 - .0005) * (1 + math.cos(math.pi * completed_epochs / 300)) / 2
    if len(groups) != 1:
        raise RuntimeError("Unexpected number of proxy optimizer parameter groups")
    group = groups[0]
    if not (group["eps"] == 1e-4 and group["initial_lr"] == .005
        and math.isclose(group["lr"], expected_lr, rel_tol=1e-12, abs_tol=1e-15)
        and scheduler["T_max"] == 300 and scheduler["eta_min"] == .0005
        and scheduler["last_epoch"] == completed_epochs
        and checkpoint["completed_epochs"] == completed_epochs):
        raise RuntimeError("Saved actual optimizer/scheduler differs from the reviewed LR/eps configuration")
    return {"eps": group["eps"], "initial_lr": group["initial_lr"], "lr": group["lr"],
        "T_max": scheduler["T_max"], "eta_min": scheduler["eta_min"], "completed_epochs": completed_epochs}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--source-branch", required=True)
    parser.add_argument("--diagnostic-epochs", type=int, default=10, choices=[10])
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=False)
    status_path = args.root / "queue_status.json"
    state = {"status": "preparing", "queue_pid": os.getpid(), "created_utc": now(),
        "commit": args.expected_commit, "branch": args.source_branch,
        "diagnostic_epochs": 10, "steps_per_epoch": 200, "scheduler_epochs": 300,
        "seed": 22, "teacher": None, "source_checkpoint": None, "test_read": False,
        "selection": None, "proxy_initial_lr": .005, "proxy_eps": 1e-4,
        "display_name": "V9-HIER64 LR0.005 EPS1e-4 train-only 10-epoch diagnostic S22"}
    write(status_path, state)
    try:
        snapshot = freeze_source(args.repo, args.root, args.expected_commit)
        freeze_diagnostics(args.repo, args.root, snapshot)
        target = snapshot / RELATIVE
        env = dict(os.environ)
        env.update(OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2",
            PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1", HDF5_USE_FILE_LOCKING="FALSE",
            V9_SOURCE_COMMIT=args.expected_commit, V9_SOURCE_BRANCH=args.source_branch)
        env["PYTHONPATH"] = str(snapshot) + ":" + str(snapshot / "inter_hierarchy_MN40") + ":" + env.get("PYTHONPATH", "")
        env.pop("CUDA_VISIBLE_DEVICES", None)
        suites = (
            (snapshot / BASE_RELATIVE, ["test_runner_contracts", "test_optimizer_step_audit", "test_same_graph_observer"], "inherited_observer_contracts.log"),
            (target, ["test_runner_contracts"], "optimizer_variant_contracts.log"),
            (snapshot, ["inter_hierarchy_MN40.hier_proxy_scratch_v9.tests.test_geometry",
                "inter_hierarchy_MN40.hier_proxy_scratch_v9.tests.test_base_protocol"], "geometry_contracts.log"),
        )
        for cwd, modules, log in suites:
            if run_logged([sys.executable, "-m", "unittest", *modules, "-v"], args.root / log, env, cwd, timeout=300) != 0:
                raise RuntimeError(f"Diagnostic CPU contracts failed: {log}")
        state.update(status="waiting_for_two_idle_gpus", cpu_checks_passed=True, updated_utc=now())
        write(status_path, state)
        while True:
            rows = safe_query(state, status_path)
            candidates = idle_gpus(rows)
            state.update(gpu_observation=rows, updated_utc=now())
            write(status_path, state)
            if len(candidates) < 2:
                time.sleep(30)
                continue
            chosen, locks = candidates[:2], []
            try:
                for row in chosen:
                    lock = open(f"/tmp/hycore_{os.getuid()}_gpu_{row['index']}.lock", "a+")
                    locks.append(lock)
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                time.sleep(5)
                confirmed = {row["uuid"] for row in idle_gpus(safe_query(state, status_path))}
                if not all(row["uuid"] in confirmed for row in chosen):
                    continue
                env.update(device_environment(chosen))
                base = [sys.executable, "-m", "torch.distributed.run", "--standalone",
                    "--nproc_per_node=2", str(target / "runner.py"),
                    "--data-dir", str(args.data_dir), "--workers", "2"]
                state.update(status="smoke_running", gpus=chosen, snapshot=str(snapshot), updated_utc=now())
                write(status_path, state)
                smoke = args.root / "smoke"
                smoke_command = base + ["--run-dir", str(smoke), "--smoke", "--diagnostic-epochs", "2", "--steps-per-epoch", "2"]
                if run_logged(smoke_command, args.root / "smoke.log", env, snapshot, timeout=600) != 0:
                    raise RuntimeError("GPU 2x2 diagnostic smoke failed; main not launched")
                manifest = json.loads((smoke / "manifest.json").read_text())
                if not smoke_accepted(manifest):
                    raise RuntimeError("Diagnostic smoke manifest acceptance failed")
                optimizer_acceptance = verify_saved_optimizer(smoke, 2)
                state.update(smoke_passed=True, smoke_initialization=manifest["initialization"],
                    smoke_model_sha256=manifest["initialized_model_sha256"],
                    smoke_actual_optimizer=optimizer_acceptance, updated_utc=now())
                write(status_path, state)
                confirmed = {row["uuid"] for row in idle_gpus(safe_query(state, status_path))}
                if not all(row["uuid"] in confirmed for row in chosen):
                    raise RuntimeError("Chosen GPUs no longer idle after smoke; diagnostic not launched")
                main_run = args.root / "main"
                command = base + ["--run-dir", str(main_run), "--diagnostic-epochs", "10"]
                state.update(status="diagnostic_running", main_dir=str(main_run), main_command=command,
                    started_utc=now(), updated_utc=now())
                write(status_path, state)
                exit_code = run_logged(command, args.root / "diagnostic.log", env, snapshot)
                if exit_code == 0:
                    state["completed_actual_optimizer"] = verify_saved_optimizer(main_run, 10)
                state.update(status="completed" if exit_code == 0 else "diagnostic_failed",
                    exit_code=exit_code, finished_utc=now(), updated_utc=now())
                write(status_path, state)
                return
            except BlockingIOError:
                time.sleep(30)
            finally:
                for lock in locks:
                    lock.close()
    except BaseException as exc:
        state.update(status="failed", error=repr(exc), updated_utc=now())
        write(status_path, state)
        raise


if __name__ == "__main__":
    main()
