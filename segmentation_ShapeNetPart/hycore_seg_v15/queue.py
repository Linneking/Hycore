"""Launch one prefix experiment after a completely idle GPU is observed."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import time

from datetime import datetime, timezone


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def idle_gpus():
    query = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,uuid,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits"], text=True, timeout=20)
    processes = subprocess.check_output([
        "nvidia-smi", "--query-compute-apps=gpu_uuid,pid",
        "--format=csv,noheader,nounits"], text=True, timeout=20)
    result = []
    for row in query.splitlines():
        index, uuid, memory, utilization = (cell.strip() for cell in row.split(","))
        if uuid not in processes and int(memory) <= 32 and int(utilization) == 0:
            result.append(int(index))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--queue-out", type=Path, required=True)
    args = parser.parse_args()
    args.queue_out.mkdir(parents=True, exist_ok=False)
    status = args.queue_out / "queue_status.json"
    commit = subprocess.check_output(
        ["git", "-C", str(args.repo), "rev-parse", "HEAD"], text=True).strip()
    write_json(status, {"state": "cpu_preflight", "commit": commit, "at": timestamp()})
    preflight = subprocess.run(
        [str(args.python), "-m", "segmentation_ShapeNetPart.hycore_seg_v15.test_contracts"],
        cwd=args.repo, env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1"})
    if preflight.returncode:
        write_json(status, {"state": "cpu_preflight_failed", "commit": commit,
                            "returncode": preflight.returncode, "at": timestamp()})
        return
    while True:
        try:
            initial = idle_gpus()
            time.sleep(.5)
            available = sorted(set(initial) & set(idle_gpus()))
        except (subprocess.SubprocessError, OSError) as exc:
            write_json(status, {"state": "waiting_for_gpu_query", "error": str(exc), "at": timestamp()})
            time.sleep(30)
            continue
        if available:
            gpu = available[0]
            break
        write_json(status, {"state": "waiting_for_idle_gpu", "commit": commit, "at": timestamp()})
        time.sleep(30)
    run = args.queue_out / "main"
    config = args.repo / "segmentation_ShapeNetPart/hycore_seg_v15/configs/tan_r_prefix.json"
    command = [str(args.python), "-m", "segmentation_ShapeNetPart.hycore_seg_v15.train",
               "--config", str(config), "--arm", "h", "--data", str(args.data),
               "--out", str(run), "--physical-gpu", str(gpu)]
    with (args.queue_out / "train.log").open("xb") as log:
        process = subprocess.Popen(
            command, cwd=args.repo, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True, env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu),
                                         "PYTHONDONTWRITEBYTECODE": "1"})
    write_json(status, {"state": "dispatched", "gpu": gpu, "pid": process.pid,
                        "commit": commit, "command": command, "at": timestamp()})
    result = process.wait()
    write_json(status, {"state": "complete" if result == 0 else "failed", "gpu": gpu,
                        "pid": process.pid, "returncode": result, "commit": commit, "at": timestamp()})


if __name__ == "__main__":
    main()
