"""Detach one authorized dual-GPU HyCoRe B64 shuffle baseline."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from ..hier_proxy_scratch_v4.launch_matrix import gpu_inventory

REPO = Path(__file__).resolve().parents[2]


def timestamp():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def write_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def command_for(args):
    command = [str(args.python), "-u", "-m", "torch.distributed.run", "--standalone",
               "--nnodes=1", "--nproc-per-node=2", "--module",
               "inter_hierarchy_MN40.hycore_b64_v6.train",
               "--data-dir", str(args.data_dir), "--run-dir", str(args.root / "B64_shuffle"),
               "--epochs", "2" if args.smoke else "300", "--seed", "22",
               "--workers", str(args.workers), "--train-eval-every", "10",
               "--save-every", "20", "--eval-batch-size", "16"]
    if args.smoke:
        command.extend(["--smoke", "--smoke-steps", "2"])
    return command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--gpus", type=int, nargs=2, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--startup-check-seconds", type=float, default=20)
    args = parser.parse_args()
    if len(set(args.gpus)) != 2 or min(args.gpus) < 0 or args.workers < 0:
        parser.error("two distinct GPU indices and nonnegative workers required")
    if not 0 <= args.startup_check_seconds <= 60:
        parser.error("startup check must be between0 and60seconds")
    args.python = args.python.resolve(strict=True)
    args.data_dir = args.data_dir.resolve(strict=True)
    args.root = args.root.resolve()
    if not args.python.is_file() or not args.data_dir.is_dir():
        parser.error("invalid Python or dataset path")
    if args.root.exists():
        raise FileExistsError(args.root)
    inventory = gpu_inventory()
    if any(not inventory.get(g, {}).get("idle", False) for g in args.gpus):
        raise RuntimeError("assigned GPUs must be completely idle")
    args.root.mkdir(parents=True, exist_ok=False)
    state = {"status": "dispatching", "started_utc": timestamp(),
             "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
             "branch": subprocess.check_output(["git", "branch", "--show-current"], cwd=REPO, text=True).strip(),
             "smoke_only": args.smoke, "gpu_inventory": inventory,
             "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
             "protocol": "full9840 shuffle/drop_last; global64 on2ranks;300epochs;original best-test selection",
             "comparison": "B64 source-style sampling stability; differs from balanced V6 H20 sampling/data/step count."}
    write_json(args.root / "launcher_state.json", state)
    try:
        fresh = gpu_inventory()
        if any(not fresh.get(g, {}).get("idle", False) for g in args.gpus):
            raise RuntimeError("GPU became busy before dispatch")
        command = command_for(args)
        environment = os.environ.copy()
        environment.update(CUDA_VISIBLE_DEVICES=",".join(map(str, args.gpus)),
                           CUDA_DEVICE_ORDER="PCI_BUS_ID",
                           HYCORE_PHYSICAL_GPUS=",".join(map(str, args.gpus)),
                           HYCORE_PHYSICAL_GPU_UUIDS=",".join(str(fresh[g]["uuid"]) for g in args.gpus),
                           OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2",
                           PYTHONUNBUFFERED="1", TORCH_NCCL_ASYNC_ERROR_HANDLING="1")
        log_path = args.root / "B64_console.log"
        with log_path.open("x", encoding="utf-8") as stream:
            process = subprocess.Popen(command, cwd=REPO, env=environment,
                                       stdin=subprocess.DEVNULL, stdout=stream,
                                       stderr=subprocess.STDOUT, start_new_session=True)
        state["run"] = {"status": "dispatched", "pid": process.pid,
                        "physical_gpus": args.gpus, "command": command,
                        "run_dir": str(args.root / "B64_shuffle"), "console_log": str(log_path),
                        "started_utc": timestamp()}
        write_json(args.root / "launcher_state.json", state)
        deadline = time.monotonic() + args.startup_check_seconds
        while time.monotonic() < deadline:
            if process.poll() is not None:
                if args.smoke and process.returncode == 0:
                    state.update(status="smoke_process_completed", checked_utc=timestamp())
                    state["run"].update(exit_code=0, alive_at_launcher_check=False)
                    write_json(args.root / "launcher_state.json", state)
                    print(json.dumps({"status": state["status"], **state["run"]}), flush=True)
                    return
                raise RuntimeError(f"training exited during startup: {process.returncode}")
            time.sleep(min(1., max(0., deadline - time.monotonic())))
        state.update(status="detached_process_alive", checked_utc=timestamp())
        state["run"]["alive_at_launcher_check"] = True
        write_json(args.root / "launcher_state.json", state)
        print(json.dumps({"status": state["status"], **state["run"]}), flush=True)
    except BaseException as error:
        state.update(status="dispatch_failed_or_partial", error=repr(error), checked_utc=timestamp())
        write_json(args.root / "launcher_state.json", state)
        raise


if __name__ == "__main__":
    main()
