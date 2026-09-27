"""Launch the fixed B1/B2 × HIER-scale screen on independently idle GPUs.

The launcher is a server-side job scheduler, not a model-selection loop. It
starts three runs at once, then starts the fourth on the first run's GPU only
after that run exits successfully and the GPU is still idle. It never kills or
preempts another process. Each run has its own immutable output directory.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import subprocess


HERE = Path(__file__).resolve().parent
EXPERIMENT_CWD = HERE.parent


def args_from_cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("root", "pretrained", "data-dir", "initialization-cache",
                   "calibration-json", "python"):
        parser.add_argument("--" + option, type=Path, required=True)
    parser.add_argument("--gpus", type=int, nargs=3, metavar=("GPU1", "GPU2", "GPU3"),
                        required=True)
    parser.add_argument("--epochs", type=int, default=8)
    args = parser.parse_args()
    if len(set(args.gpus)) != 3 or any(gpu < 0 for gpu in args.gpus):
        parser.error("--gpus requires three distinct nonnegative GPU indices")
    if args.epochs < 1:
        parser.error("--epochs must be positive")
    for name in ("pretrained", "data_dir", "initialization_cache",
                 "calibration_json", "python"):
        getattr(args, name).resolve(strict=True)
    if not args.root.is_dir():
        parser.error("--root must be an existing run parent directory")
    return args


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def gpu_is_idle(index: int) -> bool:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid,memory.used,utilization.gpu",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    )
    rows = [tuple(part.strip() for part in line.split(","))
            for line in result.stdout.splitlines() if line.strip()]
    matches = [row for row in rows if int(row[0]) == index]
    if len(matches) != 1:
        raise RuntimeError(f"GPU {index} not found in nvidia-smi")
    _, uuid, used, utilization = matches[0]
    if int(used) > 128 or int(utilization) != 0:
        return False
    apps = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    )
    return uuid not in {line.strip() for line in apps.stdout.splitlines()}


def write_state(path: Path, state: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def launch(args: argparse.Namespace, state: dict, name: str, arm: str,
           scale: int, gpu: int) -> subprocess.Popen | None:
    record = {"arm": arm, "hier_scale": scale, "gpu_physical": gpu,
              "run_dir": str(args.root / name), "started_utc": utc_now()}
    if (args.root / name).exists() or (args.root / f"{name}_console.log").exists():
        record.update(status="skipped", reason="run directory or console log already exists")
        state["runs"][name] = record
        write_state(args.root / "launcher_state.json", state)
        return None
    if not gpu_is_idle(gpu):
        record.update(status="skipped", reason="GPU not completely idle at launch")
        state["runs"][name] = record
        write_state(args.root / "launcher_state.json", state)
        return None
    command = [
        str(args.python), "-u", "-m", "hier_proxy_v2_online.train",
        "--mode", "train", "--arm", arm,
        "--pretrained", str(args.pretrained),
        "--data-dir", str(args.data_dir),
        "--run-dir", str(args.root / name),
        "--initialization-cache", str(args.initialization_cache),
        "--calibration-json", str(args.calibration_json),
        "--hier-scale", str(scale), "--epochs", str(args.epochs),
        "--device", "cuda:0",
    ]
    environment = os.environ.copy()
    environment.update(CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="4",
                       MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4")
    console_path = args.root / f"{name}_console.log"
    with console_path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps({"gpu_physical": gpu, "command": command,
                                 "started_utc": record["started_utc"]}) + "\n")
        stream.flush()
        process = subprocess.Popen(
            command, cwd=EXPERIMENT_CWD, env=environment, stdin=subprocess.DEVNULL,
            stdout=stream, stderr=subprocess.STDOUT, start_new_session=True,
        )
    record.update(status="running", pid=process.pid, command=command)
    state["runs"][name] = record
    write_state(args.root / "launcher_state.json", state)
    print(f"started {name} on GPU {gpu} as PID {process.pid}", flush=True)
    return process


def complete(args: argparse.Namespace, state: dict, name: str,
             process: subprocess.Popen) -> None:
    code = process.wait()
    record = state["runs"][name]
    record.update(status="completed" if code == 0 else "failed",
                  exit_code=code, finished_utc=utc_now())
    write_state(args.root / "launcher_state.json", state)
    print(f"finished {name} with exit code {code}", flush=True)


def main() -> None:
    args = args_from_cli()
    state: dict = {"started_utc": utc_now(), "scheduler": "fixed-online-v2-screen",
                   "runs": {}}
    write_state(args.root / "launcher_state.json", state)
    g1, g2, g3 = args.gpus
    first = launch(args, state, "B1_scale1", "B1", 1, g1)
    second = launch(args, state, "B2_scale1", "B2", 1, g2)
    third = launch(args, state, "B1_scale2", "B1", 2, g3)
    if first is not None:
        complete(args, state, "B1_scale1", first)
        if first.returncode == 0:
            fourth = launch(args, state, "B2_scale2", "B2", 2, g1)
            if fourth is not None:
                complete(args, state, "B2_scale2", fourth)
        else:
            state["runs"]["B2_scale2"] = {
                "status": "skipped", "reason": "first GPU run failed; manual review required"
            }
            write_state(args.root / "launcher_state.json", state)
    else:
        state["runs"]["B2_scale2"] = {
            "status": "skipped", "reason": "first GPU unavailable; no safe queue target"
        }
        write_state(args.root / "launcher_state.json", state)
    for name, process in (("B2_scale1", second), ("B1_scale2", third)):
        if process is not None:
            complete(args, state, name, process)
    state["finished_utc"] = utc_now()
    write_state(args.root / "launcher_state.json", state)
    if any(item["status"] != "completed" for item in state["runs"].values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
