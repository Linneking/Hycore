"""One fresh official c1 repeat: CPU gates, idle physical GPU2, train-only smoke, 300 epochs."""
from __future__ import annotations
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from ..hycore_c01_ablation_20261020.queue import snapshot_source
from ..hier_proxy_scratch_v8.queue import now, write, query_gpus, idle_gpus, device_environment, run_logged
from .train import NAME, digest


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--reference-manifest", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--source-branch", required=True)
    parser.add_argument("--gpu-index", type=int, choices=(2,), default=2)
    parser.add_argument("--poll-seconds", type=int, default=15)
    args = parser.parse_args(argv)
    if args.poll_seconds < 5:
        parser.error("GPU observation interval must be at least five seconds")
    return args


def command(data, directory, smoke=False):
    result = [sys.executable, "-m", "inter_hierarchy_MN40.hycore_c1_reproduction_20261021.train",
              "--data-dir", str(data), "--run-dir", str(directory)]
    if smoke:
        result += ["--smoke", "--epochs", "2", "--steps-per-epoch", "2"]
    return result


def observe_gpus():
    """An unavailable observation is never evidence that a GPU is free."""
    try:
        return query_gpus()
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as error:
        print(json.dumps({"event": "gpu_query_retry", "reason": type(error).__name__}), flush=True)
        return []


def candidate(rows, gpu_index):
    return next((row for row in idle_gpus(rows) if row["index"] == gpu_index), None)


def load_reference(path):
    reference = json.loads(path.read_text())
    config = reference["training_config"]
    if config["seed"] != 4780 or reference["train_size"] != 9840:
        raise ValueError("Reference must be an ORIG seed4780 full-training arm")
    train = reference["dataset_identity"]["train"]
    if train["size"] != 9840 or len(train["file_order"]) != 5:
        raise ValueError("Reference must identify all five official train shards")
    if set(train["file_order"]) != set(train["file_sha256"]):
        raise ValueError("Reference training identities are incomplete")
    return train


def verify_smoke(directory, commit, expected_init, expected_train):
    import torch
    manifest = json.loads((directory / "manifest.json").read_text())
    config = manifest["training_config"]
    if (manifest["status"] != "completed" or manifest["completed_epochs"] != 2
            or manifest["optimizer_steps"] != 4 or manifest["official_test_read"]
            or manifest["test_size"] != 0 or manifest["commit"] != commit
            or manifest["canonical_name"] != NAME or manifest["train_size"] != 9840
            or manifest["validation_size"] != 0 or config["c"] != 1
            or config["clip_r"] is not None or config["gradient_hook"]
            or config["radial_gamma"] != 1000 or config["contrastive_margin"] != 4
            or config["seed"] != 4780 or config["workers"] != 8
            or config["eval_batch_size"] != 16 or config["cosine_period"] != 300
            or manifest["initialized_model_sha256"] != expected_init
            or manifest["dataset_identity"]["train"] != expected_train):
        raise RuntimeError("Official c1 smoke identity, scientific config or data mismatch")
    state = torch.load(directory / "last_checkpoint.pth", map_location="cpu", weights_only=False)
    if not all(bool(torch.isfinite(value).all()) for value in state["net"].values()):
        raise FloatingPointError("Nonfinite native c1 smoke checkpoint")
    counters = [value for name, value in state["net"].items() if name.endswith("num_batches_tracked")]
    if not counters or any(int(value) != 8 for value in counters):
        raise RuntimeError("Four updates must give eight original part/whole BN forwards")
    steps = (directory / "steps.jsonl").read_text().splitlines()
    if len(steps) != 4:
        raise RuntimeError("Smoke update record is incomplete")
    return {"status": "passed", "canonical_name": NAME, "commit": commit,
            "train_only_updates": 4, "official_test_read": False,
            "initialized_model_sha256": expected_init, "training_identity_matches_ORIG_reference": True}


def main():
    args = arguments()
    args.root.mkdir(parents=True, exist_ok=False)
    status = args.root / "queue_status.json"
    state = {"status": "preparing", "created_utc": now(), "queue_pid": os.getpid(),
             "canonical_name": NAME, "commit": args.expected_commit, "branch": args.source_branch,
             "seed": 4780, "epochs": 300, "steps_per_epoch": 307, "gpu_index": args.gpu_index,
             "source_checkpoint": None, "teacher": None, "production_started": False,
             "train_size": 9840, "validation_size": 0, "test_size": 2468}

    def save():
        state["updated_utc"] = now()
        write(status, state)

    save()
    try:
        expected_train = load_reference(args.reference_manifest)
        snapshot = snapshot_source(args.repo, args.root, args.expected_commit)
        env = dict(os.environ)
        env.update(PYTHONPATH=str(snapshot) + ":" + env.get("PYTHONPATH", ""),
                   HYCORE_C1_SOURCE_COMMIT=args.expected_commit, HYCORE_C1_SOURCE_BRANCH=args.source_branch,
                   HYCORE_TRAIN_SHARD_ORDER=json.dumps(expected_train["file_order"]),
                   HYCORE_C1_CONTRACT_IDENTITY=str(args.root / "cpu_native_identity.json"),
                   OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2",
                   PYTHONUNBUFFERED="1", HDF5_USE_FILE_LOCKING="FALSE",
                   PYTHONPYCACHEPREFIX="/tmp/hycore_c1_pycache_" + args.root.name)
        env["CUDA_VISIBLE_DEVICES"] = ""  # CPU contracts may import extensions, never allocate CUDA.
        check = [sys.executable, "-m", "unittest",
                 "inter_hierarchy_MN40.hycore_c1_reproduction_20261021.tests.test_contract", "-v"]
        if run_logged(check, args.root / "cpu_contract.log", env, snapshot, 900):
            raise RuntimeError("Official c1 CPU contracts failed; production was not launched")
        identity = json.loads((args.root / "cpu_native_identity.json").read_text())
        expected_init = identity["initialized_model_sha256"]
        env["HYCORE_C1_EXPECTED_INIT"] = expected_init
        env["HYCORE_C1_REFERENCE_MANIFEST"] = str(args.reference_manifest.resolve())
        for name, sha in expected_train["file_sha256"].items():
            if digest(args.data_dir / name) != sha:
                raise ValueError("Training data differs from the ORIG reference")
        write(args.root / "data_identity.json", {"mode": "readonly_original_cache",
              "source": str(args.data_dir), "expected_train": expected_train, "test_content_read": False})
        state.update(cpu_checks_passed=True, native_identity=identity, snapshot=str(snapshot),
                     status="waiting_for_idle_gpu", cached_data=str(args.data_dir))
        save()
        attempts = 0
        while True:
            rows = observe_gpus()
            state["gpu_observation"] = rows
            save()
            gpu = candidate(rows, args.gpu_index)
            if gpu is None:
                time.sleep(args.poll_seconds)
                continue
            lock = open("/tmp/hycore_" + str(os.getuid()) + "_gpu_" + str(gpu["index"]) + ".lock", "a+")
            try:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    time.sleep(args.poll_seconds)
                    continue
                time.sleep(10)
                observed = candidate(observe_gpus(), args.gpu_index)
                if observed is None or observed["uuid"] != gpu["uuid"]:
                    continue
                launch_env = dict(env, **device_environment([gpu]))
                attempts += 1
                smoke = args.root / ("smoke_" + str(attempts).zfill(2))
                state.update(status="smoke_running", gpus=[gpu], smoke_dir=str(smoke))
                save()
                if run_logged(command(args.data_dir, smoke, True), args.root / (smoke.name + ".log"),
                              launch_env, snapshot, 3600):
                    raise RuntimeError("GPU c1 smoke failed; production was not launched")
                accepted = verify_smoke(smoke, args.expected_commit, expected_init, expected_train)
                write(args.root / "smoke_acceptance.json", accepted)
                state.update(smoke_passed=True, status="smoke_passed_cooling")
                save()
                time.sleep(10)
                observed = candidate(observe_gpus(), args.gpu_index)
                if observed is None or observed["uuid"] != gpu["uuid"]:
                    state["status"] = "waiting_for_idle_gpu"
                    save()
                    continue
                directory = args.root / "main"
                cmd = command(args.data_dir, directory)
                with (args.root / "main.log").open("x", encoding="utf-8") as log:
                    child = subprocess.Popen(cmd, cwd=snapshot, env=launch_env, stdin=subprocess.DEVNULL,
                                             stdout=log, stderr=subprocess.STDOUT)
                    state.update(status="running", production_started=True, production_started_utc=now(),
                                 production_pid=child.pid, directory=str(directory), command=cmd)
                    save()
                    code = child.wait()
                state.update(status="completed" if code == 0 else "failed", exit_code=code, finished_utc=now())
                save()
                if code:
                    raise RuntimeError("Official c1 training exited with failure")
                return
            finally:
                lock.close()
    except BaseException as error:
        state.update(status="failed", error=repr(error))
        save()
        raise


if __name__ == "__main__":
    main()
