"""Verify three arms, then run at most one job on each of two authorized GPUs."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
from threading import Lock
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[3]))

from inter_hierarchy_MN40.hier_proxy_scratch_v8.queue import (
    device_environment, idle_gpus, now, query_gpus, write,
)


@contextmanager
def idle_device(index):
    with open(f"/tmp/hycore_{os.getuid()}_gpu_{index}.lock", "a+") as lock:
        while True:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                time.sleep(15)
                continue
            first = {row["index"]: row for row in idle_gpus(query_gpus())}
            if index in first:
                time.sleep(2)
                second = {row["index"]: row for row in idle_gpus(query_gpus())}
                if index in second and first[index]["uuid"] == second[index]["uuid"]:
                    yield second[index]
                    return
            fcntl.flock(lock, fcntl.LOCK_UN)
            time.sleep(15)


def run(root, data, name, pool, gpu, smoke):
    work = root / name / ("smoke" if smoke else "main")
    status = work.parent / ("smoke_queue.json" if smoke else "production_queue.json")
    work.parent.mkdir(exist_ok=True)
    record = {"arm": name, "gpu": gpu, "pid": None, "state": "waiting_for_idle_gpu", "updated": now()}
    write(status, record)
    with idle_device(gpu) as row, work.with_suffix(".log").open("x", encoding="utf-8") as log:
        env = {**os.environ, **device_environment([row]), "PYTHONHASHSEED": "4780",
               "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4"}
        command = [sys.executable, "-u", str(HERE / "runner.py"), "--pool", str(pool),
                   "--data-dir", str(data), "--output", str(work)]
        if smoke:
            command.append("--smoke")
        with subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT) as process:
            record.update(state="running", pid=process.pid, updated=now(), started=now())
            write(status, record)
            code = process.wait()
        record.update(state="passed" if code == 0 else "failed", exit_code=code, updated=now())
        write(status, record)
        if code:
            raise RuntimeError(f"{name} exited {code}; see {work.with_suffix('.log')}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--gpus", type=int, nargs=2, required=True)
    args = parser.parse_args()
    first_gpu, second_gpu = args.gpus
    if first_gpu == second_gpu:
        parser.error("Choose two distinct physical GPUs")
    root = args.root
    root.mkdir()
    status = root / "queue.json"
    state = {"pid": os.getpid(), "gpus": args.gpus, "state": "smoke", "updated": now()}
    write(status, state)
    arms = (("H0", 0), ("H32", 32), ("H64", 64))
    try:
        with ThreadPoolExecutor(max_workers=2) as jobs:
            smoke64 = jobs.submit(run, root, args.data_dir, *arms[2], second_gpu, True)
            for name, pool in arms[:2]:
                run(root, args.data_dir, name, pool, first_gpu, True)
            smoke64.result()

            inputs = {}
            for name, _ in arms:
                updates = [json.loads((root / name / "smoke" / f"update{step:02}.json").read_text())
                           for step in (1, 2)]
                inputs[name] = [update["input_sha256"] for update in updates]
            if any(fingerprints != inputs["H0"] for fingerprints in inputs.values()):
                raise RuntimeError(f"Augmented input or crop pairing differs: {inputs}")

            write(root / "H32" / "production_queue.json",
                  {"state": "waiting_for_first_free_gpu", "updated": now()})
            state.update(state="production", accepted_smoke_inputs=inputs["H0"], updated=now())
            write(status, state)

            # Whichever first arm finishes successfully gets the queued third
            # arm. Its GPU must pass the idle gate again before H32 starts.
            waiting = iter(arms[1:2])
            dispatch_lock = Lock()

            def lane(arm, gpu):
                run(root, args.data_dir, *arm, gpu, False)
                with dispatch_lock:
                    queued = next(waiting, None)
                if queued:
                    run(root, args.data_dir, *queued, gpu, False)

            h0 = jobs.submit(lane, arms[0], first_gpu)
            h64 = jobs.submit(lane, arms[2], second_gpu)
            h0.result()
            h64.result()
        state.update(state="complete", updated=now())
    except Exception as error:
        state.update(state="failed", error=repr(error), updated=now())
        raise
    finally:
        write(status, state)


if __name__ == "__main__":
    main()
