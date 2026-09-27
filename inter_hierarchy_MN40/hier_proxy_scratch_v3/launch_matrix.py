"""Launch a scratch HyCoRe warmup, then matched B1/B2 branches on idle GPUs.

This server-side launcher deliberately does not preempt other users.  It leaves
each training process detached in its own session and writes an atomic state
file so the caller can verify dispatch without remaining connected over SSH.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import subprocess
import time


EXPERIMENT_CWD = Path(__file__).resolve().parent.parent


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def gpu_inventory() -> dict[int, dict[str, object]]:
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
        result[int(index)] = {
            "uuid": uuid,
            "used_mb": int(used),
            "utilization": int(utilization),
            "idle": int(used) <= 128 and int(utilization) == 0 and uuid not in active,
        }
    return result


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def check_paths(args: argparse.Namespace) -> None:
    args.python.resolve(strict=True)
    args.data_dir.resolve(strict=True)
    if args.root.exists():
        raise FileExistsError(f"run root already exists: {args.root}")
    args.root.parent.resolve(strict=True)
    if args.epochs != 300 or args.warmup_epochs != 20:
        raise ValueError("approved scratch protocol requires 300 total and 20 warmup epochs")
    if len(set(args.candidate_gpus)) != len(args.candidate_gpus):
        raise ValueError("candidate GPU indices must be distinct")


def launch(args: argparse.Namespace, state: dict, name: str,
           command: list[str], gpu: int) -> subprocess.Popen:
    run_dir = args.root / name
    console_path = args.root / f"{name}_console.log"
    if run_dir.exists() or console_path.exists():
        raise FileExistsError(f"output for {name} already exists")
    if not gpu_inventory().get(gpu, {}).get("idle", False):
        raise RuntimeError(f"GPU {gpu} became busy before launching {name}")
    environment = os.environ.copy()
    environment.update(CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="4",
                       MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4")
    with console_path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps({"started_utc": utc_now(), "gpu_physical": gpu,
                                 "command": command}) + "\n")
        stream.flush()
        process = subprocess.Popen(
            command, cwd=EXPERIMENT_CWD, env=environment,
            stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    state["runs"][name] = {
        "status": "running", "pid": process.pid, "gpu_physical": gpu,
        "run_dir": str(run_dir), "console_log": str(console_path),
        "command": command, "started_utc": utc_now(),
    }
    atomic_json(args.root / "launcher_state.json", state)
    print(f"started {name} PID {process.pid} on GPU {gpu}", flush=True)
    return process


def choose_idle(candidates: list[int], count: int) -> list[int]:
    current = gpu_inventory()
    return [gpu for gpu in candidates if current.get(gpu, {}).get("idle", False)][:count]


def wait_for_idle(candidates: list[int], count: int, state: dict,
                  root: Path) -> list[int]:
    last_report = 0.0
    while True:
        chosen = choose_idle(candidates, count)
        if len(chosen) == count:
            return chosen
        if time.monotonic() - last_report > 60:
            state["waiting_for_idle_gpus"] = {
                "need": count, "available": chosen, "checked_utc": utc_now(),
            }
            atomic_json(root / "launcher_state.json", state)
            print(f"waiting for {count} completely idle GPUs", flush=True)
            last_report = time.monotonic()
        time.sleep(10)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--candidate-gpus", type=int, nargs="+", required=True)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--warmup-epochs", type=int, default=20)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    check_paths(args)
    args.root.mkdir(exist_ok=False)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=EXPERIMENT_CWD,
                            capture_output=True, text=True, check=True).stdout.strip()
    state: dict = {
        "status": "starting", "started_utc": utc_now(), "git_commit": commit,
        "scheduler": "scratch-v3-warmup-then-matched-branches",
        "candidate_gpus": args.candidate_gpus, "runs": {},
    }
    atomic_json(args.root / "launcher_state.json", state)
    try:
        warm_gpu = wait_for_idle(args.candidate_gpus, 1, state, args.root)[0]
        warm_command = [
            str(args.python), "-u", "-m", "hier_proxy_scratch_v3.train",
            "--phase", "warmup", "--data-dir", str(args.data_dir),
            "--run-dir", str(args.root / "warmup"), "--device", "cuda:0",
            "--epochs", str(args.epochs), "--warmup-epochs", str(args.warmup_epochs),
            "--workers", str(args.workers),
        ]
        warm = launch(args, state, "warmup", warm_command, warm_gpu)
        warm_code = warm.wait()
        warm_record = state["runs"]["warmup"]
        warm_record.update(status="completed" if warm_code == 0 else "failed",
                           exit_code=warm_code, finished_utc=utc_now())
        atomic_json(args.root / "launcher_state.json", state)
        checkpoint = args.root / "warmup" / "last.pth"
        initial_mu = args.root / "warmup" / "initial_mu.pt"
        if warm_code != 0 or not checkpoint.is_file() or not initial_mu.is_file():
            raise RuntimeError("warmup failed or its checkpoint/initial features are missing")
        # The warmup process may take a few seconds to release CUDA allocations.
        first_gpu, second_gpu = wait_for_idle(args.candidate_gpus, 2, state, args.root)
        branch_processes: dict[str, subprocess.Popen] = {}
        for arm, gpu in (("B1", first_gpu), ("B2", second_gpu)):
            name = arm + "_scale3"
            command = [
                str(args.python), "-u", "-m", "hier_proxy_scratch_v3.train",
                "--phase", "branch", "--arm", arm,
                "--warmup-checkpoint", str(checkpoint),
                "--data-dir", str(args.data_dir),
                "--run-dir", str(args.root / name), "--device", "cuda:0",
                "--epochs", str(args.epochs), "--warmup-epochs", str(args.warmup_epochs),
                "--workers", str(args.workers),
            ]
            branch_processes[name] = launch(args, state, name, command, gpu)
        time.sleep(20)
        early = {name: process.returncode for name, process in branch_processes.items()
                 if process.poll() is not None}
        if early:
            state["status"] = "branch_early_failure"
            state["early_exit_codes"] = early
            raise RuntimeError(f"branch exited in first 20 seconds: {early}")
        state["status"] = "branches_launched"
        state["branches_launched_utc"] = utc_now()
        atomic_json(args.root / "launcher_state.json", state)
        print("both 300-epoch branches launched", flush=True)
    except Exception as error:
        state["status"] = "failed"
        state["error"] = repr(error)
        state["finished_utc"] = utc_now()
        atomic_json(args.root / "launcher_state.json", state)
        raise


if __name__ == "__main__":
    main()
