"""Detach the authorized V6 balanced64 B0 on two verified idle GPUs."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from inter_hierarchy_MN40.hier_proxy_scratch_v4.launch_matrix import gpu_inventory


def timestamp():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def atomic_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--gpus", type=int, nargs=2, required=True)
    parser.add_argument("--fork-smoke", action="store_true")
    parser.add_argument("--startup-check-seconds", type=float, default=10.)
    args = parser.parse_args()
    if len(set(args.gpus)) != 2 or min(args.gpus) < 0 or not 0 <= args.startup_check_seconds <= 60:
        parser.error("Two distinct GPUs and a bounded startup check are required")
    args.root = args.root.resolve()
    for key in ("python", "data_dir", "source_checkpoint"):
        setattr(args, key, getattr(args, key).resolve(strict=True))
    if args.root.exists():
        raise FileExistsError(args.root)
    digest = hashlib.sha256()
    with args.source_checkpoint.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != args.source_sha256:
        raise RuntimeError("Source checkpoint SHA256 differs from reviewed identity")
    inventory = gpu_inventory()
    if any(not inventory.get(gpu, {}).get("idle", False) for gpu in args.gpus):
        raise RuntimeError("Assigned GPUs are not completely idle")
    args.root.mkdir(parents=True, exist_ok=False)
    run_dir = args.root / "B0_balanced"
    command = [str(args.python), "-u", "-m", "torch.distributed.run", "--standalone",
               "--nnodes=1", "--nproc-per-node=2", "--module",
               "inter_hierarchy_MN40.v6_balanced_b0.train", "--data-dir", str(args.data_dir),
               "--run-dir", str(run_dir), "--source-checkpoint", str(args.source_checkpoint),
               "--source-sha256", args.source_sha256]
    if args.fork_smoke:
        command += ["--fork-smoke", "--smoke-steps", "2", "--smoke-epochs", "1",
                    "--smoke-eval-batches", "1", "--skip-final-test"]
    state = {"status": "dispatching", "started_utc": timestamp(),
             "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
             "branch": subprocess.check_output(["git", "branch", "--show-current"], cwd=REPO, text=True).strip(),
             "source_sha256": args.source_sha256, "gpu_inventory": inventory,
             "smoke_only": args.fork_smoke, "command": command,
             "protocol": "V6 balanced64 B0; complete historical e20 fork; validation selection"}
    atomic_json(args.root / "launcher_state.json", state)
    try:
        fresh = gpu_inventory()
        if any(not fresh.get(gpu, {}).get("idle", False) for gpu in args.gpus):
            raise RuntimeError("GPU became busy before dispatch")
        environment = os.environ.copy()
        environment.update(CUDA_VISIBLE_DEVICES=",".join(map(str, args.gpus)),
                           CUDA_DEVICE_ORDER="PCI_BUS_ID",
                           HYCORE_PHYSICAL_GPUS=",".join(map(str, args.gpus)),
                           HYCORE_PHYSICAL_GPU_UUIDS=",".join(fresh[gpu]["uuid"] for gpu in args.gpus),
                           OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2",
                           PYTHONUNBUFFERED="1", TORCH_NCCL_ASYNC_ERROR_HANDLING="1")
        log_path = args.root / "B0_console.log"
        with log_path.open("x", encoding="utf-8") as stream:
            process = subprocess.Popen(command, cwd=REPO, env=environment, stdin=subprocess.DEVNULL,
                                       stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        state.update(status="dispatched", pid=process.pid, run_dir=str(run_dir),
                     console_log=str(log_path), physical_gpus=args.gpus)
        atomic_json(args.root / "launcher_state.json", state)
        deadline = time.monotonic() + args.startup_check_seconds
        while time.monotonic() < deadline:
            if process.poll() is not None:
                if args.fork_smoke and process.returncode == 0:
                    break
                raise RuntimeError(f"Training exited during startup: {process.returncode}")
            time.sleep(min(1., max(0., deadline - time.monotonic())))
        state.update(status="smoke_completed" if process.poll() == 0 else "detached_process_alive",
                     checked_utc=timestamp(), alive_at_launcher_check=process.poll() is None)
        atomic_json(args.root / "launcher_state.json", state)
        print(json.dumps(state, allow_nan=False), flush=True)
    except BaseException as exc:
        state.update(status="dispatch_failed_or_partial", error=repr(exc), checked_utc=timestamp())
        atomic_json(args.root / "launcher_state.json", state)
        raise


if __name__ == "__main__":
    main()
