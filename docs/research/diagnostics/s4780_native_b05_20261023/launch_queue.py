"""One approved seed4780 arm: immutable source, idle dual-GPU smoke, fresh300.

The native and B05 arms run as separate queues. GPU combinations are tried in
turn, so a second queue can acquire another idle pair while the first owns its
locks. Existing runs, environments and other users' processes are never changed.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib
import itertools
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
RELATIVE = Path("docs/research/diagnostics/s4780_native_b05_20261023")
for _path in (REPO, REPO / "inter_hierarchy_MN40"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from inter_hierarchy_MN40.hier_proxy_cycle_v9c.queue import (
    freeze_source as freeze_parent_source, TRAIN_ORDER, TRAIN_SHA256,
)
from inter_hierarchy_MN40.hier_proxy_scratch_v8.queue import (
    now, write, query_gpus, idle_gpus, device_environment, run_logged, cache_data,
)


def freeze_source(repo, root, expected, arm):
    """Freeze both historical dependency trees and this versioned successor."""
    snapshot = freeze_parent_source(repo, root, expected)
    source_directory = repo / RELATIVE
    sources = sorted(source_directory.glob("*.py"))
    if not all((source_directory / name).is_file()
               for name in ("runner.py", "launch_queue.py", "test_contracts.py")):
        raise RuntimeError("Published successor runner, queue and CPU contracts are required")
    tracked = set(subprocess.check_output(
        ["git", "-C", str(repo), "ls-files", "-z", "--", str(RELATIVE)],
    ).decode().split("\0"))
    target_directory = snapshot / RELATIVE
    target_directory.mkdir(parents=True, exist_ok=False)
    identity = json.loads((root / "source_identity.json").read_text(encoding="utf-8"))
    for source in sources:
        relative = RELATIVE / source.name
        if relative.as_posix() not in tracked:
            raise RuntimeError("Successor source must be committed before deployment: " + str(relative))
        destination = target_directory / source.name
        shutil.copy2(source, destination)
        identity["files"][relative.as_posix()] = hashlib.sha256(destination.read_bytes()).hexdigest()
    identity["successor"] = {
        "relative_path": RELATIVE.as_posix(), "arm": arm, "seed": 4780,
        "epochs": 300, "steps_per_epoch": 200, "hier_enabled": True,
        "proxy_initial_lr": .005, "proxy_eps": 1e-4, "proxy_min_lr": .0005,
        "source_checkpoint": None, "teacher": None,
    }
    write(root / "source_identity.json", identity)
    return snapshot


def safe_query(state, path):
    try:
        return query_gpus()
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, OSError) as exc:
        state.update(gpu_query_error=repr(exc), gpu_query_failed_utc=now())
        write(path, state)
        return []


def initial_identity(snapshot, env, root, arm):
    """Build a separate CPU-only seed4780 model using the arm's actual runner."""
    code = (
        "import sys; "
        "from pathlib import Path; "
        "sys.path.insert(0,str(Path(sys.argv[1]))); "
        "import runner; "
        "ns=runner.build_namespace(sys.argv[2]); "
        "ns['seed_all'](4780); "
        "m=ns['configure_backbone'](ns['Hype_pointMLP']()); "
        "print('INITIAL_HASH='+ns['state_hash'](m),flush=True)"
    )
    output = subprocess.check_output(
        [sys.executable, "-c", code, str(snapshot / RELATIVE), arm],
        env=env, cwd=snapshot, text=True, stderr=subprocess.STDOUT, timeout=300,
    )
    (root / "initial_identity.log").write_text(output, encoding="utf-8")
    values = [line.split("=", 1)[1].strip() for line in output.splitlines()
              if line.startswith("INITIAL_HASH=")]
    if len(values) != 1 or len(values[0]) != 64:
        raise RuntimeError("Independent seed4780 initialization identity is missing")
    int(values[0], 16)
    known = importlib.import_module("runner").INITIAL_HASHES.get(arm)
    if known is not None and values[0] != known:
        raise RuntimeError("Independent initialization differs from the pinned arm identity")
    return values[0]


def finite_tree(value, torch):
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(finite_tree(item, torch) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite_tree(item, torch) for item in value)
    return True


def verify_optimizer(saved, epochs, updates):
    optimizer = saved.get("proxy_optimizer")
    scheduler = saved.get("proxy_scheduler")
    if not optimizer or not scheduler:
        raise RuntimeError("HIER-on checkpoint lacks its optimizer or cosine scheduler")
    groups = optimizer.get("param_groups", [])
    if not groups or not optimizer.get("state"):
        raise RuntimeError("Proxy optimizer has no parameter groups or actual update state")
    for group in groups:
        if not (group.get("initial_lr") == .005 and group.get("eps") == 1e-4
                and tuple(group.get("betas", ())) == (.9, .999)
                and group.get("weight_decay") == .01
                and abs(float(group["lr"]) - .0005) < 1e-12):
            raise RuntimeError("Saved proxy AdamW differs from the approved A1 recipe")
    steps = {int(value["step"]) for value in optimizer["state"].values()}
    if steps != {updates}:
        raise RuntimeError("Saved proxy AdamW step count differs from actual training updates")
    if not (scheduler.get("T_max") == epochs and scheduler.get("eta_min") == .0005
            and scheduler.get("base_lrs") == [.005]
            and scheduler.get("last_epoch") == epochs):
        raise RuntimeError("Saved proxy cosine schedule differs from the fresh-run contract")
    importlib.import_module("runner").verify_optimizer_payload(
        saved, completed_epochs=epochs, schedule_epochs=epochs)
    return {"initial_lr": .005, "eps": 1e-4, "betas": [.9, .999],
            "weight_decay": .01, "min_lr": .0005, "steps": updates,
            "T_max": epochs, "last_epoch": epochs,
            "saved_lr": [float(group["lr"]) for group in groups]}


def accept_smoke(smoke, expected_hash, arm):
    import torch
    manifest = json.loads((smoke / "manifest.json").read_text(encoding="utf-8"))
    config = manifest["training_config"]
    expected_cap, expected_gs, expected_c = ((None, False, 1.) if arm == "native"
                                           else (2.3, True, .1))
    if not (manifest["status"] == "completed" and not manifest["official_test_read"]
            and manifest["train_size"] == 9840 and manifest["validation_size"] == 0
            and manifest["selection_domain"] == "smoke_train"
            and manifest["initialized_model_sha256"] == expected_hash
            and manifest["completed_epochs"] == 2 and manifest["resume_identity"] is None
            and config["seed"] == 4780 and config["epochs"] == 2
            and config["steps_per_epoch"] == 2 and config["base_c"] == expected_c
            and config["native_GS"] is expected_gs
            and config["native_extra_cap"] == expected_cap and config["hier_c"] == .1
            and config["global_batch"] == 64 and config["world_size"] == 2):
        raise RuntimeError("Train-only dual-GPU smoke manifest differs from the approved arm")
    if config.get("hier_enabled", True) is not True:
        raise RuntimeError("Both approved arms must enable HIER")
    actual_arm = manifest.get("args", {}).get("arm", config.get("arm"))
    if actual_arm != arm:
        raise RuntimeError("Smoke selected the wrong baseline arm")
    optimizer_config = config["proxy_optimizer"]
    if not (optimizer_config["lr"] == .005 and optimizer_config["eps"] == 1e-4
            and optimizer_config["min_lr"] == .0005):
        raise RuntimeError("Smoke declared the wrong proxy optimizer recipe")
    saved = torch.load(smoke / "last.pth", map_location="cpu", weights_only=False)
    if not (saved["completed_epochs"] == 2 and saved["completed_optimizer_steps"] == 4
            and len(saved["rank_states"]) == 2 and saved.get("proxy") is not None):
        raise RuntimeError("Smoke lacks four actual updates and complete dual-rank state")
    for rank, restored in enumerate(saved["rank_states"]):
        counters = [int(value) for name, value in restored["BN_buffers"].items()
                    if name.endswith("num_batches_tracked")]
        if restored["rank"] != rank or not counters or set(counters) != {8}:
            raise RuntimeError("Smoke BN counters do not match four part/whole update pairs")
        if "sampler" not in restored or "private_generators" not in restored:
            raise RuntimeError("Smoke lacks rank-local sampler or private RNG recovery state")
    if not all(saved.get(key) is not None and finite_tree(saved[key], torch)
               for key in ("net", "optimizer", "proxy", "proxy_optimizer")):
        raise RuntimeError("Smoke saved missing or nonfinite weights or optimizer states")
    for name in ("best.pth", "metrics_epoch_001.json", "metrics_epoch_002.json"):
        if not (smoke / name).is_file():
            raise RuntimeError("Smoke did not save its required artifacts: " + name)
    actual_optimizer = verify_optimizer(saved, epochs=2, updates=4)
    return {"accepted": True, "arm": arm, "seed": 4780,
            "initialized_model_sha256": expected_hash, "official_test_read": False,
            "train_size": 9840, "completed_epochs": 2, "optimizer_steps": 4,
            "BN_updates_per_rank": 8, "actual_proxy_optimizer": actual_optimizer}


def accept_completion(production, arm):
    import torch
    manifest = json.loads((production / "manifest.json").read_text(encoding="utf-8"))
    config = manifest["training_config"]
    if not (manifest["status"] == "completed" and manifest["completed_epochs"] == 300
            and manifest["official_test_read"] and manifest["train_size"] == 9840
            and manifest["selection_domain"] == "test" and manifest["selection_size"] == 2468
            and manifest["resume_identity"] is None and config["epochs"] == 300
            and config["steps_per_epoch"] == 200 and config["seed"] == 4780
            and manifest.get("args", {}).get("arm", config.get("arm")) == arm):
        raise RuntimeError("Completion manifest differs from approved fresh300x200 TS training")
    saved = torch.load(production / "last.pth", map_location="cpu", weights_only=False)
    if saved["completed_optimizer_steps"] != 60000:
        raise RuntimeError("Completed training did not record exactly 60000 updates")
    return verify_optimizer(saved, epochs=300, updates=60000)


def acquire_available_pair(candidates):
    """Try all combinations without blocking on another queue's first pair."""
    for chosen in itertools.combinations(candidates, 2):
        locks = []
        try:
            for row in chosen:
                lock = open(f"/tmp/hycore_{os.getuid()}_gpu_{row['index']}.lock", "a+")
                locks.append(lock)
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return list(chosen), locks
        except BlockingIOError:
            for lock in locks:
                lock.close()
        except BaseException:
            for lock in locks:
                lock.close()
            raise
    return None, []


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--source-branch", required=True)
    parser.add_argument("--arm", choices=("native", "b05"), required=True)
    parser.add_argument("--poll-seconds", type=int, default=30)
    args = parser.parse_args()
    if args.poll_seconds < 1:
        parser.error("poll-seconds must be positive")
    args.repo, args.data_dir = args.repo.resolve(), args.data_dir.resolve()
    args.root.mkdir(parents=True, exist_ok=False)
    state = {"status": "preparing", "created_utc": now(), "queue_pid": os.getpid(),
             "commit": args.expected_commit, "branch": args.source_branch, "arm": args.arm,
             "seed": 4780, "epochs": 300, "steps_per_epoch": 200,
             "global_batch": 64, "classes_per_rank": 16, "instances_per_class": 2,
             "sampler": "non-cyclic class-balanced; within-class replacement",
             "hier_enabled": True, "proxy_initial_lr": .005, "proxy_eps": 1e-4,
             "train_size": 9840, "validation_size": 0, "selection": "test2468 each epoch",
             "source_checkpoint": None, "teacher": None, "production_started": False}
    path = args.root / "queue_status.json"
    write(path, state)
    try:
        snapshot = freeze_source(args.repo, args.root, args.expected_commit, args.arm)
        env = dict(os.environ)
        env.update(OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2",
                   PYTHONUNBUFFERED="1", HDF5_USE_FILE_LOCKING="FALSE", MPLBACKEND="Agg",
                   V9C_SOURCE_COMMIT=args.expected_commit, V9C_SOURCE_BRANCH=args.source_branch,
                   V9_SOURCE_COMMIT=args.expected_commit, V9_SOURCE_BRANCH=args.source_branch,
                   HYCORE_TRAIN_SHARD_ORDER=json.dumps(TRAIN_ORDER))
        env.setdefault("PYTHONPYCACHEPREFIX", "/tmp/hycore_s4780_pycache_" + args.root.name)
        env["PYTHONPATH"] = ":".join((str(snapshot), str(snapshot / "inter_hierarchy_MN40"),
                                     env.get("PYTHONPATH", "")))
        env.pop("CUDA_VISIBLE_DEVICES", None)
        fixture = env.get("V8_OFFICIAL_HYPTORCH")
        if fixture and not Path(fixture).is_dir():
            env.pop("V8_OFFICIAL_HYPTORCH", None)
        ddp_module = ("inter_hierarchy_MN40.hier_proxy_scratch_v9.tests.ddp_contract"
                      if args.arm == "native" else
                      "inter_hierarchy_MN40.hier_proxy_cycle_v9c.tests.ddp_contract")
        commands = [
            [sys.executable, "-m", "unittest", "discover", "-s", str(snapshot / RELATIVE),
             "-p", "test_contracts.py", "-v"],
            [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=2",
             "--module", ddp_module],
        ]
        for index, command in enumerate(commands):
            if run_logged(command, args.root / f"cpu_check_{index}.log", env, snapshot, timeout=1200) != 0:
                raise RuntimeError(f"CPU contract {index} failed; production was not launched")
        expected_hash = initial_identity(snapshot, env, args.root, args.arm)
        state.update(status="preparing_training_data", cpu_checks_passed=True,
                     expected_initial_sha256=expected_hash, updated_utc=now())
        write(path, state)
        cached_data = cache_data(args.data_dir, args.root.name, args.root)
        identity = json.loads((args.root / "training_data_cache.json").read_text(encoding="utf-8"))
        hashes = {row["file"]: row["sha256"] for row in identity["train_files"]}
        if hashes != TRAIN_SHA256:
            raise RuntimeError("Training HDF5 content differs from the reviewed full9840 source")
        state.update(status="waiting_for_two_idle_gpus", training_files_match_reference=True,
                     updated_utc=now())
        write(path, state)
        attempts = 0
        while True:
            rows = safe_query(state, path)
            state.update(gpu_observation=rows, updated_utc=now())
            write(path, state)
            chosen, locks = acquire_available_pair(idle_gpus(rows))
            if chosen is None:
                time.sleep(args.poll_seconds)
                continue
            try:
                time.sleep(10)
                still_idle = {row["uuid"] for row in idle_gpus(safe_query(state, path))}
                if not all(row["uuid"] in still_idle for row in chosen):
                    continue
                env.update(device_environment(chosen))
                attempts += 1
                smoke = args.root / f"smoke_{attempts:02d}"
                base = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=2",
                        str(snapshot / RELATIVE / "runner.py"), "--arm", args.arm, "--hier", "on",
                        "--data-dir", str(cached_data), "--workers", "8", "--eval-batch-size", "16"]
                state.update(status="smoke_running", gpus=chosen, smoke_dir=str(smoke), updated_utc=now())
                write(path, state)
                command = base + ["--run-dir", str(smoke), "--epochs", "2", "--steps-per-epoch", "2",
                                  "--smoke", "--skip-final-test", "--smoke-eval-batches", "1",
                                  "--audit-epochs", "1"]
                if run_logged(command, args.root / f"smoke_{attempts:02d}.log", env, snapshot, timeout=3600) != 0:
                    raise RuntimeError("Dual-GPU train-only smoke failed; production was not launched")
                acceptance = accept_smoke(smoke, expected_hash, args.arm)
                write(args.root / "smoke_acceptance.json", acceptance)
                state.update(smoke_passed=True, smoke_initial_sha256=expected_hash,
                             smoke_actual_optimizer=acceptance["actual_proxy_optimizer"], updated_utc=now())
                write(path, state)
                time.sleep(5)
                idle_after = {row["uuid"] for row in idle_gpus(safe_query(state, path))}
                if not all(row["uuid"] in idle_after for row in chosen):
                    state.update(status="waiting_for_two_idle_gpus", updated_utc=now())
                    write(path, state)
                    continue
                production = args.root / "main"
                command = base + ["--run-dir", str(production)]
                state.update(status="production_running", production_started=True,
                             production_dir=str(production), production_command=command,
                             production_started_utc=now(), updated_utc=now())
                write(path, state)
                code = run_logged(command, args.root / "production.log", env, snapshot)
                if code == 0:
                    state["completed_actual_optimizer"] = accept_completion(production, args.arm)
                state.update(status="completed" if code == 0 else "production_failed",
                             exit_code=code, finished_utc=now(), updated_utc=now())
                write(path, state)
                return
            finally:
                for lock in locks:
                    lock.close()
    except BaseException as exc:
        state.update(status="failed", error=repr(exc), updated_utc=now())
        write(path, state)
        raise


if __name__ == "__main__":
    main()
