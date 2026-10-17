"""Independently schedule full9840 V8 controls on idle GPUs only.

B64 needs two devices; B0-32 needs one. CPU checks and immutable source/data
preparation do not allocate CUDA. Each job passes its own training-only GPU
smoke before a fresh production launch. No requirement for three idle GPUs.
"""
from __future__ import annotations
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from ..hier_proxy_scratch_v8.queue import (
    now, write, query_gpus, idle_gpus, device_environment,
    freeze_source, run_logged, cache_data)

INITIAL_SHA = "f94e52202df5aaae21f21b8b9c6beaa3925424c4e04a6dcc7d2b84b89af04b73"


def requirements(control):
    if control not in ("B64", "B0-32"):
        raise ValueError("Unknown control")
    return 2 if control == "B64" else 1


def plan_allocations(pending, candidates):
    """Prefer B64 when a pair is free; permit B0 alone on one device."""
    remaining = list(candidates)
    plan = []
    for control in ("B64", "B0-32"):
        need = requirements(control)
        if control in pending and len(remaining) >= need:
            plan.append((control, remaining[:need]))
            remaining = remaining[need:]
    return plan


def command(control, data, directory, smoke=False):
    result = [sys.executable, "-m", "torch.distributed.run", "--standalone",
              "--nproc_per_node=" + str(requirements(control)), "--module",
              "inter_hierarchy_MN40.hier_proxy_scratch_v8_fulltrain.controls",
              "--control", control, "--data-dir", str(data), "--run-dir", str(directory)]
    if smoke:
        result += ["--smoke", "--epochs", "2", "--steps-per-epoch", "2",
                   "--skip-final-test", "--smoke-eval-batches", "1"]
    return result


def verify_smoke(directory, control, commit):
    manifest = json.loads((directory / "manifest.json").read_text())
    config = manifest["training_config"]
    if (manifest["status"] != "completed" or manifest["official_test_read"] or
            manifest["completed_epochs"] != 2 or manifest["commit"] != commit or
            manifest["train_size"] != 9840 or manifest["validation_size"] != 0 or
            manifest["selection_domain"] != "smoke_train" or
            manifest["initialized_model_sha256"] != INITIAL_SHA or
            config["control"] != control or config["P"] != 0 or
            config["lambda_hier_after_warmup"] != 0 or config["proxy_optimizer"] is not None or
            config["c"] != .1 or config["clip_r"] != 2.3 or
            config["radial_gamma"] != 500 or config["contrastive_margin"] != 2):
        raise RuntimeError("Control smoke identity/configuration failed")
    steps = [json.loads(line) for line in (directory / "steps.jsonl").read_text().splitlines()]
    if len(steps) != 4 or any(row["hier_calls"] != 0 or row["proxy_parameter_count"] != 0 or
                             row["lambda_hier"] != 0 or row["loss"] != row["base"] for row in steps):
        raise RuntimeError("Smoke is not a strict base-only run")
    if any(len(row["ids"]) != (64 if control == "B64" else 32) for row in steps):
        raise RuntimeError("Smoke batch size differs")
    return {"status": "passed", "control": control, "updates": 4,
            "initialized_model_sha256": manifest["initialized_model_sha256"],
            "split_sha256": manifest["split_sha256"], "official_test_read": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--source-branch", required=True)
    parser.add_argument("--poll-seconds", type=int, default=30)
    args = parser.parse_args()
    if args.poll_seconds < 5:
        parser.error("poll interval must be at least5 seconds")
    args.root.mkdir(parents=True, exist_ok=False)
    jobs = {c: {"status": "preparing", "world_size": requirements(c),
                "display_name": "V8-" + c + ("-C01" if c == "B0-32" else "") + "-FULL9840-TESTSEL",
                "epochs": 300, "steps_per_epoch": 307 if c == "B0-32" else 200,
                "production_started": False, "smoke_attempts": 0} for c in ("B64", "B0-32")}
    state = {"status": "preparing", "created_utc": now(), "queue_pid": os.getpid(),
             "commit": args.expected_commit, "branch": args.source_branch,
             "train_size": 9840, "validation_size": 0, "selection": "test2468 each epoch",
             "seed": 22, "source_checkpoint": None, "teacher": None, "jobs": jobs}
    status_path = args.root / "queue_status.json"
    children, locks, streams = {}, {}, {}

    def save():
        state["updated_utc"] = now()
        write(status_path, state)

    def release(control):
        for handle in locks.pop(control, []):
            handle.close()
        stream = streams.pop(control, None)
        if stream:
            stream.close()

    def spawn(control, directory, env, smoke):
        cmd = command(control, data, directory, smoke)
        log = args.root / (directory.name + ".log")
        stream = log.open("x", encoding="utf-8")
        process = subprocess.Popen(cmd, cwd=snapshot, env=env,
                                   stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT)
        children[control], streams[control] = process, stream
        jobs[control].update(pid=process.pid, command=cmd, directory=str(directory),
                             status="smoke_running" if smoke else "running",
                             phase_started_utc=now(), log=str(log))
        if not smoke:
            jobs[control].update(production_started=True, production_started_utc=now())

    save()
    try:
        snapshot = freeze_source(args.repo, args.root, args.expected_commit)
        env = dict(os.environ)
        env.update(PYTHONPATH=str(snapshot) + ":" + env.get("PYTHONPATH", ""),
                   V8_SOURCE_COMMIT=args.expected_commit, V8_SOURCE_BRANCH=args.source_branch,
                   OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2",
                   PYTHONUNBUFFERED="1", HDF5_USE_FILE_LOCKING="FALSE",
                   PYTHONPYCACHEPREFIX="/tmp/hycore_v8_fullcontrols_pycache_" + args.root.name)
        env.pop("CUDA_VISIBLE_DEVICES", None)
        cpu = [sys.executable, "-m", "unittest",
               "inter_hierarchy_MN40.hier_proxy_scratch_v8_fulltrain.tests.test_protocol",
               "inter_hierarchy_MN40.hier_proxy_scratch_v8_fulltrain.tests.test_controls",
               "inter_hierarchy_MN40.hier_proxy_scratch_v8.tests.test_controls", "-v"]
        if run_logged(cpu, args.root / "cpu_controls.log", env, snapshot, 1200) != 0:
            raise RuntimeError("CPU control/protocol contracts failed")
        data = cache_data(args.data_dir, args.root.name, args.root)
        state["cpu_checks_passed"] = True
        for job in jobs.values():
            job["status"] = "waiting_for_idle_gpus"
        save()
        while True:
            # Progress each active arm independently; one failed arm must not suppress the other.
            for control, child in list(children.items()):
                if child.poll() is None:
                    continue
                streams[control].close()
                del streams[control]
                del children[control]
                job = jobs[control]
                if child.returncode != 0:
                    job.update(status="failed", exit_code=child.returncode, finished_utc=now())
                    release(control)
                elif job["status"] == "smoke_running":
                    try:
                        accepted = verify_smoke(Path(job["directory"]), control, args.expected_commit)
                        write(args.root / (control + "_smoke_acceptance.json"), accepted)
                        job.update(status="smoke_passed_cooling", smoke_passed=True, smoke_finished=time.monotonic())
                    except Exception as exc:
                        job.update(status="failed", error=repr(exc), finished_utc=now())
                        release(control)
                else:
                    job.update(status="completed", exit_code=0, finished_utc=now())
                    release(control)
            rows = query_gpus()
            state["gpu_observation"] = rows
            free = {row["uuid"] for row in idle_gpus(rows)}
            for control, job in jobs.items():
                if job["status"] != "smoke_passed_cooling" or time.monotonic() - job["smoke_finished"] < 10:
                    continue
                if all(gpu["uuid"] in free for gpu in job["gpus"]):
                    spawn(control, args.root / control, dict(env, **device_environment(job["gpus"])), False)
                else:
                    release(control)
                    job["status"] = "waiting_for_idle_gpus"
            reserved = {gpu["uuid"] for c, j in jobs.items() if c in locks for gpu in j["gpus"]}
            candidates = [g for g in idle_gpus(rows) if g["uuid"] not in reserved]
            pending = [c for c, j in jobs.items() if j["status"] == "waiting_for_idle_gpus"]
            for control, chosen in plan_allocations(pending, candidates):
                acquired = []
                try:
                    for gpu in chosen:
                        handle = open(f"/tmp/hycore_{os.getuid()}_gpu_{gpu['index']}.lock", "a+")
                        acquired.append(handle)
                        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    time.sleep(10)
                    still_idle = {g["uuid"] for g in idle_gpus(query_gpus())}
                    if not all(g["uuid"] in still_idle for g in chosen):
                        continue
                    locks[control] = acquired
                    acquired = []
                    job = jobs[control]
                    job.update(gpus=chosen, smoke_attempts=job["smoke_attempts"] + 1)
                    directory = args.root / (control + "_smoke_" + str(job["smoke_attempts"]).zfill(2))
                    spawn(control, directory, dict(env, **device_environment(chosen)), True)
                except BlockingIOError:
                    pass
                finally:
                    for handle in acquired:
                        handle.close()
            statuses = [j["status"] for j in jobs.values()]
            if all(s in ("completed", "failed") for s in statuses):
                state.update(status="completed" if all(s == "completed" for s in statuses) else "finished_with_failure",
                             finished_utc=now())
                save()
                return
            state["status"] = "active" if children else "waiting_for_idle_gpus"
            save()
            time.sleep(args.poll_seconds)
    except BaseException as exc:
        state.update(status="failed", error=repr(exc))
        save()
        # Never kill unrelated processes. Any already launched child retains its own log/manifest.
        raise


if __name__ == "__main__":
    main()
