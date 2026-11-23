"""Preflight and dispatch the two seed-4780 ScanObjectNN author runs."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import math
import os
from pathlib import Path
import subprocess
import sys

from worker import ARM_NAMES, HERE, REFERENCE_COMMIT, SEED, author_args, support
from docs.research.diagnostics.scanobjectnn_author_s12.dispatch import reserve
from inter_hierarchy_MN40.hier_proxy_scratch_v8.queue import device_environment


def run_arm(root, arm, gpu, source_commit, smoke=False, state=None):
    command = [sys.executable, "-u", str(HERE / "worker.py"), "--arm", arm,
               "--output", str(root / "main"), "--source-commit", source_commit]
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
        if code := child.wait():
            raise RuntimeError(f"{arm} exited with status {code}")


def check_smokes(root, source_commit):
    runs = [root / arm / "main" for arm in ARM_NAMES]
    manifests = [json.loads((run / "manifest.json").read_text(encoding="utf-8")) for run in runs]
    traces = [json.loads((run / "smoke_trace.json").read_text(encoding="utf-8")) for run in runs]
    no_hier, hier = traces
    keys = ("cloud", "labels", "whole", "part")
    first_losses = [trace["batches"][0]["base_losses"] for trace in traces]
    proxy = [json.loads(line) for line in
             (runs[1] / "hier" / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    hierarchy = manifests[1]["hierarchy"]
    checks = dict(
        same_seed=all(m["seed"] == SEED and m["argv"] == author_args(True) for m in manifests),
        provenance=all(m["source_commit"] == source_commit and m["reference_commit"] == REFERENCE_COMMIT
                       and m["status"] == "completed" and m["smoke"] for m in manifests),
        same_verified_inputs=all(manifests[0][key] == manifests[1][key]
                                 for key in ("data_sha256", "extension_sha256", "gpu_uuid")),
        hierarchy_routing=manifests[0]["hierarchy"] is None and not (runs[0] / "hier").exists()
        and hierarchy["proxy_seed"] == hierarchy["first_relationship_seed"] == SEED
        and hierarchy["gumbel_seed"] == SEED + 300001,
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
    result = dict(passed=all(checks.values()), seed=SEED, source_commit=source_commit,
                  reference_commit=REFERENCE_COMMIT, initial_model_sha256=no_hier["initial_model_sha256"],
                  first_base_losses=first_losses, checks=checks, observed_at=support.now())
    support.write(root / "acceptance.json", result)
    if not result["passed"]:
        raise RuntimeError(f"Author entry preflight failed: {checks}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--gpus", type=int, nargs=2, required=True)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    if args.gpus[0] == args.gpus[1]:
        parser.error("Choose two distinct physical GPUs")

    root = args.run_root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    preflight = root / "preflight"
    preflight.mkdir()
    lanes = []
    for index, arm in zip(args.gpus, ARM_NAMES):
        folder = root / arm
        folder.mkdir()
        state = dict(arm=arm, gpu=index, seed=SEED, state="waiting_for_preflight", manager_pid=os.getpid(),
                     source_commit=args.source_commit, reference_commit=REFERENCE_COMMIT, queued_at=support.now())
        support.write(folder / "queue.json", state)
        lanes.append((index, arm, folder, state))

    try:
        with reserve(args.gpus[0]) as gpu:
            for arm in ARM_NAMES:
                folder = preflight / arm
                folder.mkdir()
                run_arm(folder, arm, gpu, args.source_commit, smoke=True)
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
                run_arm(folder, arm, gpu, args.source_commit, state=state)
            state.update(state="completed", exit_code=0)
        except Exception as error:
            state.update(state="failed", error=repr(error))
        state["updated_at"] = support.now()
        support.write(folder / "queue.json", state)
        return state["state"]

    with ThreadPoolExecutor(max_workers=2) as workers:
        completed = list(workers.map(run_lane, lanes))
    if completed != ["completed", "completed"]:
        raise RuntimeError("At least one author run failed")


if __name__ == "__main__":
    main()
