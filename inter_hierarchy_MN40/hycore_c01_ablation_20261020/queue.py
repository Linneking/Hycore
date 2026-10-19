"""Three authorized single-GPU HyCoRe ablations, each gated by a train-only smoke."""
from __future__ import annotations
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from ..hier_proxy_scratch_v8.queue import (
    now, write, query_gpus, idle_gpus, device_environment,
    freeze_source, run_logged, cache_data,
)

VARIANTS = ("v8_nohook", "orig_c01", "orig_c01_cap")
NAMES = {
    "v8_nohook": "V8-B0-32-C01-NOHOOK-FULL9840-TESTSEL",
    "orig_c01": "ORIG-B0-32-C01-S4780",
    "orig_c01_cap": "ORIG-B0-32-C01-CAP23-S4780",
}


def command(variant, data, directory, smoke=False):
    if variant not in VARIANTS:
        raise ValueError("unknown authorized arm")
    if variant == "v8_nohook":
        result = [sys.executable, "-m", "torch.distributed.run", "--standalone",
                  "--nproc_per_node=1", "--module",
                  "inter_hierarchy_MN40.hycore_c01_ablation_20261020.v8_nohook",
                  "--control", "B0-32", "--data-dir", str(data), "--run-dir", str(directory)]
        if smoke:
            result += ["--smoke", "--epochs", "2", "--steps-per-epoch", "2",
                       "--skip-final-test", "--smoke-eval-batches", "1"]
    else:
        result = [sys.executable, "-m",
                  "inter_hierarchy_MN40.hycore_c01_ablation_20261020.original_adapter",
                  "--variant", "c01" if variant == "orig_c01" else "c01-cap",
                  "--data-dir", str(data), "--run-dir", str(directory)]
        if smoke:
            result += ["--smoke", "--epochs", "2", "--steps-per-epoch", "2"]
    return result


def snapshot_source(repo, root, expected):
    snapshot = freeze_source(repo, root, expected)
    identity_path = root / "source_identity.json"
    identity = json.loads(identity_path.read_text())
    paths = subprocess.check_output(
        ["git", "-C", str(repo), "ls-files", "-z", "--", "classification_ModelNet40"]
    ).split(b"\0")
    for encoded in paths:
        if not encoded:
            continue
        relative = Path(os.fsdecode(encoded))
        if relative.suffix != ".py":
            continue
        destination = snapshot / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(repo / relative, destination)
        identity["files"][str(relative)] = hashlib.sha256(destination.read_bytes()).hexdigest()
    identity["original_source_included"] = True
    write(identity_path, identity)
    return snapshot


def verify_smoke(directory, variant, commit):
    manifest = json.loads((directory / "manifest.json").read_text())
    if (manifest["status"] != "completed" or manifest["completed_epochs"] != 2
            or manifest["official_test_read"] or manifest["commit"] != commit
            or manifest["train_size"] != 9840 or manifest["validation_size"] != 0
            or manifest["canonical_name"] != NAMES[variant]):
        raise RuntimeError("smoke run identity, data scope or completion mismatch")
    config = manifest["training_config"]
    if config["c"] != .1:
        raise RuntimeError("all arms require c=.1")
    if variant == "v8_nohook":
        if (config["radial_gamma"] != 500 or config["contrastive_margin"] != 2
                or config["clip_r"] != 2.3 or config["HIER_backward_hook"]
                or config["P"] != 0 or config["proxy_optimizer"] is not None):
            raise RuntimeError("V8 nohook smoke differs from the approved arm")
        if manifest["initialized_model_sha256"] != "f94e52202df5aaae21f21b8b9c6beaa3925424c4e04a6dcc7d2b84b89af04b73":
            raise RuntimeError("V8 initialization differs from existing B0 control")
        steps = [json.loads(line) for line in (directory / "steps.jsonl").read_text().splitlines()]
        if len(steps) != 4 or any(row["hier_calls"] or row["proxy_parameter_count"]
                                  or row["lambda_hier"] or row["loss"] != row["base"] for row in steps):
            raise RuntimeError("V8 smoke must have four base-only updates")
    else:
        if (config["radial_gamma"] != 1000 or config["contrastive_margin"] != 4
                or config["gradient_hook"] or config["seed"] != 4780
                or config["clip_r"] != (None if variant == "orig_c01" else 2.3)):
            raise RuntimeError("original arm configuration differs")
    return {
        "status": "passed", "variant": variant, "canonical_name": NAMES[variant],
        "commit": commit, "completed_epochs": 2, "updates": 4,
        "official_test_read": False,
        "initialized_model_sha256": manifest["initialized_model_sha256"],
        "train_identity": manifest.get("dataset_identity", {}).get("train"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--source-branch", required=True)
    parser.add_argument("--poll-seconds", type=int, default=15)
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    args = parser.parse_args()
    if len(set(args.variants)) != len(args.variants):
        parser.error("duplicate arms are forbidden")
    if args.poll_seconds < 5:
        parser.error("poll interval must be at least five seconds")
    args.root.mkdir(parents=True, exist_ok=False)
    jobs = {variant: {
        "status": "preparing", "canonical_name": NAMES[variant],
        "seed": 22 if variant == "v8_nohook" else 4780,
        "epochs": 300, "steps_per_epoch": 307, "world_size": 1,
        "production_started": False, "smoke_attempts": 0,
    } for variant in args.variants}
    state = {
        "status": "preparing", "created_utc": now(), "queue_pid": os.getpid(),
        "commit": args.expected_commit, "branch": args.source_branch, "jobs": jobs,
        "train_size": 9840, "validation_size": 0, "test_size": 2468,
        "selection": "each-epoch test OA; full 300 epochs; no early stopping",
        "source_checkpoint": None, "teacher": None,
    }
    status_path = args.root / "queue_status.json"
    children, locks, streams = {}, {}, {}

    def save():
        state["updated_utc"] = now()
        write(status_path, state)

    def release(variant):
        for handle in locks.pop(variant, []):
            handle.close()
        stream = streams.pop(variant, None)
        if stream:
            stream.close()

    def spawn(variant, directory, env, smoke):
        cmd = command(variant, data, directory, smoke)
        log = args.root / (directory.name + ".log")
        stream = log.open("x", encoding="utf-8")
        child = subprocess.Popen(cmd, cwd=snapshot, env=env, stdin=subprocess.DEVNULL,
                                 stdout=stream, stderr=subprocess.STDOUT)
        children[variant], streams[variant] = child, stream
        jobs[variant].update(
            status="smoke_running" if smoke else "running", pid=child.pid,
            command=cmd, directory=str(directory), log=str(log), phase_started_utc=now())
        if not smoke:
            jobs[variant].update(production_started=True, production_started_utc=now())

    save()
    try:
        snapshot = snapshot_source(args.repo, args.root, args.expected_commit)
        env = dict(os.environ)
        env.update(
            PYTHONPATH=str(snapshot) + ":" + env.get("PYTHONPATH", ""),
            V8_SOURCE_COMMIT=args.expected_commit, V8_SOURCE_BRANCH=args.source_branch,
            OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2",
            PYTHONUNBUFFERED="1", HDF5_USE_FILE_LOCKING="FALSE",
            PYTHONPYCACHEPREFIX="/tmp/hycore_c01_pycache_" + args.root.name,
        )
        env.pop("CUDA_VISIBLE_DEVICES", None)
        test_root = snapshot / "inter_hierarchy_MN40/hycore_c01_ablation_20261020/tests"
        tests = sorted(test_root.glob("test_*.py"))
        if len(tests) < 2:
            raise RuntimeError("missing independent ablation contracts")
        modules = ["inter_hierarchy_MN40.hier_proxy_scratch_v8.tests.test_contract"]
        modules += [".".join(path.relative_to(snapshot).with_suffix("").parts) for path in tests]
        for index, module in enumerate(modules):
            if run_logged([sys.executable, "-m", "unittest", module, "-v"],
                          args.root / ("cpu_contract_" + str(index) + ".log"),
                          env, snapshot, 900) != 0:
                raise RuntimeError("CPU contract failed: " + module)
        data = cache_data(args.data_dir, args.root.name, args.root)
        state.update(cpu_checks_passed=True, snapshot=str(snapshot), cached_data=str(data))
        for job in jobs.values():
            job["status"] = "waiting_for_idle_gpu"
        save()
        while True:
            for variant, child in list(children.items()):
                if child.poll() is None:
                    continue
                streams.pop(variant).close()
                del children[variant]
                job = jobs[variant]
                if child.returncode:
                    job.update(status="failed", exit_code=child.returncode, finished_utc=now())
                    release(variant)
                elif job["status"] == "smoke_running":
                    try:
                        accepted = verify_smoke(Path(job["directory"]), variant, args.expected_commit)
                        write(args.root / (variant + "_smoke_acceptance.json"), accepted)
                        job.update(status="smoke_passed_cooling", smoke_finished=time.monotonic())
                    except Exception as exc:
                        job.update(status="failed", error=repr(exc), finished_utc=now())
                        release(variant)
                else:
                    job.update(status="completed", exit_code=0, finished_utc=now())
                    release(variant)
            rows = query_gpus()
            state["gpu_observation"] = rows
            free = {row["uuid"] for row in idle_gpus(rows)}
            for variant, job in jobs.items():
                if job["status"] != "smoke_passed_cooling" or time.monotonic() - job["smoke_finished"] < 10:
                    continue
                if variant in ("orig_c01", "orig_c01_cap"):
                    pair_paths = [args.root / (arm + "_smoke_acceptance.json")
                                  for arm in ("orig_c01", "orig_c01_cap")]
                    if not all(path.exists() for path in pair_paths):
                        continue
                    left, right = [json.loads(path.read_text()) for path in pair_paths]
                    if (left["initialized_model_sha256"] != right["initialized_model_sha256"]
                            or left["train_identity"] is None
                            or left["train_identity"] != right["train_identity"]):
                        raise RuntimeError("Original cap pair initialization/data differ")
                    if "original_pair_identity" not in state:
                        state["original_pair_identity"] = {
                            "status": "passed", "cap_is_only_pair_difference": True,
                            "initialized_model_sha256": left["initialized_model_sha256"],
                            "train_identity": left["train_identity"],
                        }
                        write(args.root / "original_pair_acceptance.json",
                              state["original_pair_identity"])
                if all(gpu["uuid"] in free for gpu in job["gpus"]):
                    spawn(variant, args.root / variant, dict(env, **device_environment(job["gpus"])), False)
                else:
                    release(variant)
                    job["status"] = "waiting_for_idle_gpu"
            reserved = {gpu["uuid"] for variant, job in jobs.items()
                        if variant in locks for gpu in job["gpus"]}
            candidates = [gpu for gpu in idle_gpus(rows) if gpu["uuid"] not in reserved]
            pending = [variant for variant, job in jobs.items() if job["status"] == "waiting_for_idle_gpu"]
            for variant, gpu in zip(pending, candidates):
                acquired = []
                try:
                    handle = open("/tmp/hycore_" + str(os.getuid()) + "_gpu_" + str(gpu["index"]) + ".lock", "a+")
                    acquired.append(handle)
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    time.sleep(10)
                    if gpu["uuid"] not in {row["uuid"] for row in idle_gpus(query_gpus())}:
                        continue
                    locks[variant], acquired = acquired, []
                    jobs[variant].update(gpus=[gpu], smoke_attempts=jobs[variant]["smoke_attempts"] + 1)
                    smoke_dir = args.root / (variant + "_smoke_" + str(jobs[variant]["smoke_attempts"]).zfill(2))
                    spawn(variant, smoke_dir, dict(env, **device_environment([gpu])), True)
                except BlockingIOError:
                    pass
                finally:
                    for handle in acquired:
                        handle.close()
            statuses = [job["status"] for job in jobs.values()]
            if all(status in ("completed", "failed") for status in statuses):
                state.update(status="completed" if all(status == "completed" for status in statuses)
                             else "finished_with_failure", finished_utc=now())
                save()
                return
            state["status"] = "active" if children else "waiting_for_idle_gpu"
            save()
            time.sleep(args.poll_seconds)
    except BaseException as exc:
        state.update(status="failed", error=repr(exc))
        save()
        # Already launched children are never stopped on a manager-side error.
        raise


if __name__ == "__main__":
    main()
