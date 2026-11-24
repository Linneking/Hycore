"""Verify and dispatch two independent, single-GPU author-launcher runs."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

from entry import ARM_NAMES, HERE, support
from inter_hierarchy_MN40.hier_proxy_scratch_v8.queue import (
    device_environment, idle_gpus, query_gpus,
)


@contextmanager
def reserve(index, extra_check=None):
    with Path("/tmp/hycore_%d_gpu_%d.lock" % (os.getuid(), index)).open("a+") as lock:
        while True:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                time.sleep(15)
                continue
            first = {row["index"]: row for row in idle_gpus(query_gpus())}
            if index in first and (extra_check is None or extra_check(first[index])):
                time.sleep(2)
                second = {row["index"]: row for row in idle_gpus(query_gpus())}
                if (index in second and second[index]["uuid"] == first[index]["uuid"]
                        and (extra_check is None or extra_check(second[index]))):
                    yield second[index]
                    return
            fcntl.flock(lock, fcntl.LOCK_UN)
            time.sleep(15)


def run_arm(root, arm, gpu, source_commit, smoke=False, state=None):
    command = [
        sys.executable, "-u", str(HERE / "entry.py"),
        "--arm", arm, "--output", str(root / "main"), "--source-commit", source_commit,
    ]
    if smoke:
        command.append("--smoke")
    env = dict(os.environ, **device_environment([gpu]), MPLBACKEND="Agg",
               OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2",
               PYTHONDONTWRITEBYTECODE="1")
    with (root / "main.log").open("x", encoding="utf-8") as log:
        child = subprocess.Popen(command, env=env, stdin=subprocess.DEVNULL,
                                 stdout=log, stderr=subprocess.STDOUT)
        if state is not None:
            state.update(state="running", pid=child.pid, started_at=support.now())
            support.write(root / "queue.json", state)
        return child.wait()


def check_smokes(root, source_commit):
    traces = [json.loads((root / arm / "main" / "smoke_trace.json").read_text())
              for arm in ARM_NAMES]
    no_hier, hier = traces
    keys = ("cloud", "labels", "whole", "part")
    first_losses = [row["batches"][0]["base_losses"] for row in traces]
    proxy = [json.loads(line) for line in
             (root / ARM_NAMES[1] / "main" / "hier" / "metrics.jsonl").read_text().splitlines()]
    checks = dict(
        initial_matches=no_hier["initial_model_sha256"] == hier["initial_model_sha256"],
        same_batches=len(no_hier["batches"]) == len(hier["batches"]) == 4 and
        all(a[key] == b[key] for a, b in zip(no_hier["batches"], hier["batches"]) for key in keys),
        same_first_base_losses=all(math.isclose(a, b, rel_tol=1e-5, abs_tol=1e-5)
                                   for a, b in zip(*first_losses)),
        same_main_rng=no_hier["rng"] == hier["rng"],
        proxy_updates=[row["optimizer_steps"] for row in proxy] == [2, 4],
        sample_relationships=all(row["stats"]["sample_triplets"] > 0 for row in proxy),
        proxy_relationships=all(row["stats"]["proxy_triplets"] > 0 for row in proxy),
    )
    result = dict(passed=all(checks.values()), source_commit=source_commit,
                  initial_model_sha256=no_hier["initial_model_sha256"],
                  first_base_losses=first_losses, checks=checks, observed_at=support.now())
    support.write(root / "acceptance.json", result)
    if not result["passed"]:
        raise RuntimeError("Author entry preflight failed: " + str(checks))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--gpus", type=int, nargs=2, required=True)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    if args.gpus[0] == args.gpus[1]:
        raise ValueError("Choose two separate physical GPUs")
    root = args.run_root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    preflight = root / "preflight"
    preflight.mkdir()
    lanes = []
    for index, arm in zip(args.gpus, ARM_NAMES):
        folder = root / arm
        folder.mkdir()
        state = dict(arm=arm, gpu=index, state="waiting_for_preflight", manager_pid=os.getpid(),
                     source_commit=args.source_commit, queued_at=support.now())
        support.write(folder / "queue.json", state)
        lanes.append((index, arm, folder, state))

    try:
        with reserve(args.gpus[0]) as gpu:
            for arm in ARM_NAMES:
                folder = preflight / arm
                folder.mkdir()
                code = run_arm(folder, arm, gpu, args.source_commit, smoke=True)
                if code:
                    raise RuntimeError("Smoke failed: %s, exit %d" % (arm, code))
        check_smokes(preflight, args.source_commit)
    except BaseException as error:
        for _, _, folder, state in lanes:
            state.update(state="preflight_failed", error=repr(error), updated_at=support.now())
            support.write(folder / "queue.json", state)
        raise

    def run_lane(lane):
        index, arm, folder, state = lane
        state.update(state="waiting_for_idle_gpu", updated_at=support.now())
        support.write(folder / "queue.json", state)
        try:
            with reserve(index) as gpu:
                code = run_arm(folder, arm, gpu, args.source_commit, state=state)
            state.update(state="completed" if code == 0 else "failed", exit_code=code)
        except BaseException as error:
            state.update(state="failed", error=repr(error))
        state["updated_at"] = support.now()
        support.write(folder / "queue.json", state)
        return state["state"]

    with ThreadPoolExecutor(max_workers=2) as workers:
        completed = list(workers.map(run_lane, lanes))
    if completed != ["completed", "completed"]:
        raise RuntimeError("At least one author launcher run failed")


if __name__ == "__main__":
    main()
