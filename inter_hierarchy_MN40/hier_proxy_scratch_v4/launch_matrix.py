"""Dispatch three matched 200-epoch runs without occupying busy GPUs.

B0 publishes its epoch-20 checkpoint and feature cache. H3 and H5 start as
detached waiting processes; train.py controls their shared-checkpoint barrier
and rechecks the assigned GPU before allocating CUDA memory. The launcher
records dispatch, then exits. Training manifests record subsequent progress.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
from pathlib import Path
import subprocess
import time
from typing import Callable


EXPERIMENT_CWD = Path(__file__).resolve().parent.parent
ARMS = ("B0", "H3", "H5")


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def gpu_inventory() -> dict[int, dict[str, object]]:
    """Inspect physical devices, independent of CUDA_VISIBLE_DEVICES.

    A small driver allocation is allowed, but every compute process makes a
    device unavailable even when its utilization is zero. Query failures are
    fatal: an unverified device is never treated as idle.
    """
    rows = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid,memory.used,utilization.gpu",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    ).stdout.splitlines()
    apps = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    ).stdout.splitlines()
    active = {line.strip() for line in apps if line.strip()}
    result: dict[int, dict[str, object]] = {}
    for line in rows:
        if not line.strip():
            continue
        index, uuid, used, utilization = (part.strip() for part in line.split(","))
        used_mb, utilization_pct = int(used), int(utilization)
        result[int(index)] = {
            "uuid": uuid,
            "used_mb": used_mb,
            "utilization": utilization_pct,
            "has_compute_process": uuid in active,
            "idle": used_mb <= 128 and utilization_pct == 0 and uuid not in active,
        }
    return result


def wait_for_idle_gpu(
    gpu: int,
    poll_seconds: float = 10.0,
    timeout_seconds: float = 0.0,
    on_wait: Callable[[dict], None] | None = None,
) -> dict[str, object]:
    """Recheck one assigned physical GPU before training allocates memory.

    Called by train.py after the shared warmup barrier. A zero timeout waits
    indefinitely without preempting a process that arrived during warmup.
    on_wait can update the training manifest while waiting.
    """
    if gpu < 0 or poll_seconds <= 0 or timeout_seconds < 0:
        raise ValueError("invalid GPU wait arguments")
    started = time.monotonic()
    while True:
        inventory = gpu_inventory()
        if gpu not in inventory:
            raise RuntimeError(f"physical GPU {gpu} not found in nvidia-smi")
        record = inventory[gpu]
        if record["idle"]:
            return {"gpu_physical": gpu, "checked_utc": utc_now(), **record}
        elapsed = time.monotonic() - started
        if on_wait is not None:
            on_wait({"gpu_physical": gpu, "checked_utc": utc_now(),
                     "wait_seconds": elapsed, **record})
        if timeout_seconds and elapsed >= timeout_seconds:
            raise TimeoutError(f"GPU {gpu} remained busy for {elapsed:.1f} seconds")
        delay = poll_seconds
        if timeout_seconds:
            delay = min(delay, max(0.0, timeout_seconds - elapsed))
        time.sleep(delay)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--candidate-gpus", type=int, nargs="+", required=True)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--warmup-epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=22)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--sample-t", type=int, required=True)
    parser.add_argument("--proxy-t", type=int, required=True)
    parser.add_argument("--gradient-mode", choices=("euclidean", "riemannian"),
                        required=True)
    parser.add_argument("--inter-weight", type=float, required=True)
    parser.add_argument("--proxy-weight", type=float, required=True)
    parser.add_argument("--startup-check-seconds", type=float, default=20.0)
    args = parser.parse_args()
    if args.epochs != 200 or args.warmup_epochs != 20:
        parser.error("this matrix requires 200 total epochs and 20 shared warmup epochs")
    if args.workers < 0 or args.sample_t < 1 or args.proxy_t < 1:
        parser.error("workers must be nonnegative and relation budgets positive")
    if (not math.isfinite(args.inter_weight) or args.inter_weight <= 0 or
            not math.isfinite(args.proxy_weight) or args.proxy_weight <= 0):
        parser.error("inter and proxy weights must be positive")
    if len(args.candidate_gpus) < 3 or len(set(args.candidate_gpus)) != len(args.candidate_gpus):
        parser.error("supply at least three distinct candidate GPU indices")
    if any(gpu < 0 for gpu in args.candidate_gpus):
        parser.error("GPU indices must be nonnegative")
    if not math.isfinite(args.startup_check_seconds) or not 0 <= args.startup_check_seconds <= 60:
        parser.error("startup check must be between zero and 60 seconds")
    args.python = args.python.resolve(strict=True)
    args.data_dir = args.data_dir.resolve(strict=True)
    args.root = args.root.resolve()
    if not args.python.is_file() or not args.data_dir.is_dir():
        parser.error("python must be an executable file and data-dir a directory")
    if args.root.exists():
        raise FileExistsError(f"run root already exists: {args.root}")
    args.root.parent.resolve(strict=True)
    return args


def training_command(args: argparse.Namespace, arm: str) -> list[str]:
    return [
        str(args.python), "-u", "-m", "hier_proxy_scratch_v4.train",
        "--arm", arm, "--data-dir", str(args.data_dir),
        "--run-dir", str(args.root / arm), "--device", "cuda:0",
        "--shared-warmup-dir", str(args.root / "B0"),
        "--epochs", str(args.epochs), "--warmup-epochs", str(args.warmup_epochs),
        "--seed", str(args.seed), "--workers", str(args.workers),
        "--sample-t", str(args.sample_t), "--proxy-t", str(args.proxy_t),
        "--gradient-mode", args.gradient_mode,
        "--inter-weight", str(args.inter_weight),
        "--proxy-weight", str(args.proxy_weight),
    ]


def launch_one(args: argparse.Namespace, state: dict, arm: str,
               gpu: int) -> subprocess.Popen:
    inventory = gpu_inventory()
    if not inventory.get(gpu, {}).get("idle", False):
        raise RuntimeError(f"GPU {gpu} became busy before dispatching {arm}")
    run_dir = args.root / arm
    console_path = args.root / f"{arm}_console.log"
    if run_dir.exists() or console_path.exists():
        raise FileExistsError(f"output for {arm} already exists")
    command = training_command(args, arm)
    environment = os.environ.copy()
    environment.update(
        CUDA_VISIBLE_DEVICES=str(gpu), HYCORE_PHYSICAL_GPU=str(gpu),
        HYCORE_PHYSICAL_GPU_UUID=str(inventory[gpu]["uuid"]),
        OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4",
        PYTHONUNBUFFERED="1",
    )
    started_utc = utc_now()
    with console_path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps({"started_utc": started_utc,
                                 "gpu_physical": gpu, "command": command}) + "\n")
        stream.flush()
        process = subprocess.Popen(
            command, cwd=EXPERIMENT_CWD, env=environment,
            stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    state["runs"][arm] = {
        "status": "dispatched", "pid": process.pid, "gpu_physical": gpu,
        "gpu_uuid": inventory[gpu]["uuid"], "gpu_inventory_at_dispatch": inventory[gpu],
        "run_dir": str(run_dir), "console_log": str(console_path),
        "manifest": str(run_dir / "manifest.json"), "command": command,
        "started_utc": started_utc,
        "shared_warmup_dir": str(args.root / "B0"),
    }
    atomic_json(args.root / "launcher_state.json", state)
    print(f"dispatched {arm}: PID {process.pid}, physical GPU {gpu}", flush=True)
    return process


def main() -> None:
    args = arguments()
    inventory = gpu_inventory()
    selected = [gpu for gpu in args.candidate_gpus
                if inventory.get(gpu, {}).get("idle", False)][:3]
    if len(selected) != 3:
        raise RuntimeError(f"need three idle GPUs before dispatch; available: {selected}")
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=EXPERIMENT_CWD,
                            capture_output=True, text=True, check=True).stdout.strip()
    branch = subprocess.run(["git", "branch", "--show-current"], cwd=EXPERIMENT_CWD,
                            capture_output=True, text=True, check=True).stdout.strip()
    args.root.mkdir(exist_ok=False)
    state = {
        "status": "starting", "started_utc": utc_now(),
        "git_commit": commit, "git_branch": branch,
        "scheduler": "scratch-v4-shared-20-epoch-warmup-three-detached-arms",
        "config": {key: str(value) if isinstance(value, Path) else value
                   for key, value in vars(args).items()},
        "gpu_inventory_at_start": inventory, "selected_gpus": selected, "runs": {},
        "progress_source": "each run manifest.json and metrics_epoch_NNN.json",
    }
    atomic_json(args.root / "launcher_state.json", state)
    try:
        processes = {arm: launch_one(args, state, arm, gpu)
                     for arm, gpu in zip(ARMS, selected)}
        deadline = time.monotonic() + args.startup_check_seconds
        while True:
            early = {arm: process.returncode for arm, process in processes.items()
                     if process.poll() is not None}
            if early:
                for arm, exit_code in early.items():
                    state["runs"][arm].update(status="early_exit", exit_code=exit_code,
                                              observed_exit_utc=utc_now())
                raise RuntimeError(f"training process exited during startup: {early}")
            if time.monotonic() >= deadline:
                break
            time.sleep(min(1.0, max(0.0, deadline - time.monotonic())))
        for arm, record in state["runs"].items():
            record["pid_alive_at_check"] = True
            record["checked_utc"] = utc_now()
            manifest_path = Path(record["manifest"])
            if manifest_path.is_file():
                try:
                    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                    record["manifest_status_at_check"] = manifest.get("status")
                except (OSError, json.JSONDecodeError) as error:
                    record["manifest_read_error"] = repr(error)
        state.update(status="dispatched", dispatched_utc=utc_now(),
                     startup_process_check="all three PIDs alive")
        atomic_json(args.root / "launcher_state.json", state)
        print("three 200-epoch processes dispatched; H3/H5 wait for shared warmup", flush=True)
    except Exception as error:
        # Running arms are preserved. This launcher never interrupts processes.
        state.update(status="failed", error=repr(error), failed_utc=utc_now())
        atomic_json(args.root / "launcher_state.json", state)
        raise


if __name__ == "__main__":
    main()
