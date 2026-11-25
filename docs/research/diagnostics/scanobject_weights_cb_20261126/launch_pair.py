"""Verify and run one approved ScanObjectNN pair on two available GPUs."""

import argparse
from contextlib import ExitStack
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(REPO))
from inter_hierarchy_MN40.hier_proxy_scratch_v8.queue import device_environment, now, query_gpus, write

ARMS = {
    "pointmlp": ("SO-S12-HIER-L0.1", "SO-S12-HIER-L1.0"),
    "dgcnn": ("DG-S4780-CB32-H0", "DG-S4780-CB32-HIER"),
}


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def available(indices, display_mib):
    devices = {item["index"]: item for item in query_gpus()}
    used = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits",
    ], text=True, timeout=20)
    memory = {int(index): int(mib) for index, mib in (line.split(",") for line in used.splitlines())}
    for index in indices:
        device = devices[index]
        if device["compute_busy"] or device["utilization"] != 0 or memory[index] > display_mib:
            return None
        if device["free_MiB"] < 30000:
            return None
    return [devices[index] for index in indices]


def reserve(indices, display_mib):
    for _ in range(60):
        first = available(indices, display_mib)
        time.sleep(2 if first else 5)
        if first:
            second = available(indices, display_mib)
            if second and [item["uuid"] for item in first] == [item["uuid"] for item in second]:
                return second
    raise RuntimeError("Both assigned GPUs must be idle together")


def accept(stage, pair, commit):
    left, right = (stage / name for name in ARMS[pair])
    traces = [read(run / "smoke_trace.json") for run in (left, right)]
    manifests = [read(run / "manifest.json") for run in (left, right)]
    batches = [trace["batches"] for trace in traces]
    first_losses = [items[0]["base_losses"] for items in batches]
    keys = ("cloud", "labels", "whole", "part")
    checks = dict(
        provenance=all(m["source_commit"] == commit and m["status"] == "completed" and m["smoke"]
                       for m in manifests),
        inputs=all(manifests[0][key] == manifests[1][key]
                   for key in ("seed", "data_sha256", "extension_sha256", "gpu_uuid")),
        initial_model=traces[0]["initial_model_sha256"] == traces[1]["initial_model_sha256"],
        matching_batches=len(batches[0]) == len(batches[1]) == 4 and
        all(a[key] == b[key] for a, b in zip(*batches) for key in keys),
        first_losses=len(first_losses[0]) == len(first_losses[1]) > 0 and
        all(math.isclose(a, b, rel_tol=1e-5, abs_tol=1e-5) for a, b in zip(*first_losses)),
        main_rng=traces[0]["rng"] == traces[1]["rng"],
    )
    if pair == "pointmlp":
        checks["weights"] = all(m["hierarchy"]["sample_weight"] == m["hierarchy"]["proxy_weight"] == weight
                                for m, weight in zip(manifests, (0.1, 1.0)))
        enabled = (left, right)
    else:
        checks["sampler"] = rows(left / "cb_sampler.jsonl") == rows(right / "cb_sampler.jsonl")
        checks["h0"] = manifests[0]["hierarchy"] is None and not (left / "hier").exists()
        enabled = (right,)
    for run in enabled:
        metrics = rows(run / "hier" / "metrics.jsonl")
        checks[run.name] = [row["optimizer_steps"] for row in metrics] == [2, 4] and all(
            row["stats"]["sample_triplets"] > 0 and row["stats"]["proxy_triplets"] > 0 for row in metrics)
    passed = all(checks.values())
    write(stage / "acceptance.json", dict(passed=passed, checks=checks, first_base_losses=first_losses,
                                        observed_at=now()))
    if not passed:
        raise RuntimeError(f"Pair preflight rejected: {checks}")


def start(args, arm, device, stage, smoke):
    env = dict(os.environ, **device_environment([device]))
    threads = "2" if args.pair == "pointmlp" else "1"
    env.update(OMP_NUM_THREADS=threads, MKL_NUM_THREADS=threads, OPENBLAS_NUM_THREADS=threads,
               PYTHONDONTWRITEBYTECODE="1", MPLBACKEND="Agg")
    command = [sys.executable, "-u", str(HERE / f"{args.pair}.py"), "--arm", arm,
               "--output", str(stage / arm), "--source-commit", args.source_commit]
    if smoke:
        command.append("--smoke")
    with (stage / f"{arm}.log").open("x", encoding="utf-8") as log:
        return subprocess.Popen(command, cwd=REPO, env=env, stdin=subprocess.DEVNULL,
                                stdout=log, stderr=subprocess.STDOUT)


def run(args, root, status):
    arms = ARMS[args.pair]
    with ExitStack() as locks:
        for index in sorted(args.gpus):
            lock = locks.enter_context(open(f"/tmp/hycore_{os.getuid()}_gpu_{index}.lock", "a"))
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        devices = reserve(args.gpus, args.display_mib)
        stage = root / "preflight"
        stage.mkdir()
        status.update(status="preflight", updated_at=now())
        write(root / "status.json", status)
        for arm in arms:
            if start(args, arm, devices[0], stage, smoke=True).wait() != 0:
                raise RuntimeError(f"Preflight worker failed for {arm}")
        accept(stage, args.pair, args.source_commit)
        devices = reserve(args.gpus, args.display_mib)
        workers = {}
        try:
            for arm, device in zip(arms, devices):
                workers[arm] = start(args, arm, device, root, smoke=False)
            status.update(status="running", started_at=now(), pids={arm: p.pid for arm, p in workers.items()})
            write(root / "status.json", status)
        finally:
            codes = {arm: process.wait() for arm, process in workers.items()}
        status["exit_codes"] = codes
        if set(codes) != set(arms) or any(codes.values()):
            raise RuntimeError(f"Training worker failed: {codes}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair", choices=ARMS, required=True)
    parser.add_argument("--gpus", nargs=2, type=int, required=True)
    parser.add_argument("--display-mib", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    if len(set(args.gpus)) != 2 or not 0 <= args.display_mib <= 128:
        parser.error("Two distinct GPUs and a display allowance between 0 and 128 MiB are required")
    root = args.output.resolve()
    root.mkdir()
    status = dict(status="waiting", pair=args.pair, gpus=args.gpus, source_commit=args.source_commit,
                  controller_pid=os.getpid(), updated_at=now())
    write(root / "status.json", status)
    try:
        run(args, root, status)
        status["status"] = "completed"
    except BaseException as error:
        status.update(status="failed", error=repr(error))
        raise
    finally:
        status["updated_at"] = now()
        write(root / "status.json", status)


if __name__ == "__main__":
    main()
