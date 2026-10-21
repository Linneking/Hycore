"""C01 cycle300 paired arm: CPU gates, idle dual-GPU smoke, fresh production."""
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
    freeze_source as freeze_legacy_source, run_logged, cache_data,
)

ORIGINAL_E0_DIGEST = "8926765a4842652b751873fb89aa37d24d9b0f4f2d831d853b9cc0c1cd46472d"
TRAIN_ORDER = [f"ply_data_train{i}.h5" for i in (1, 4, 3, 2, 0)]
TRAIN_SHA256 = {
    "ply_data_train1.h5": "0efaab3f3a7903f78037da933682267dfcdc548dc97aadda25bac218a1930276",
    "ply_data_train4.h5": "15143c2e796907767bb795d505a22d46d776e710ae55b9487e5a9684732bd1fa",
    "ply_data_train3.h5": "ee6fb75f9cb85bc36e589e7ebf23b3cd9c42ef00566c29e7a6679e83ed575495",
    "ply_data_train2.h5": "d705fc9b55b80e0a45103ec6a2938968f1e704c6bec8f2624bebbf4e435bb50a",
    "ply_data_train0.h5": "996e8f2741c5c2662314a79c4e7b461399e2835dff28925046e7e42339f9a74d",
}



def freeze_source(repo, root, expected):
    dirty = subprocess.check_output(
        ["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=no"], text=True)
    if dirty.strip():
        raise RuntimeError("Tracked server checkout must be clean before source freeze")
    snapshot = freeze_legacy_source(repo, root, expected)
    identity = json.loads((root / "source_identity.json").read_text())
    paths = subprocess.check_output(
        ["git", "-C", str(repo), "ls-files", "-z", "--", "classification_ModelNet40"]).split(b"\0")
    for encoded in paths:
        if not encoded:
            continue
        relative = Path(os.fsdecode(encoded))
        if relative.suffix != ".py":
            continue
        target = snapshot / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(repo / relative, target)
        identity["files"][str(relative)] = hashlib.sha256(target.read_bytes()).hexdigest()
    write(root / "source_identity.json", identity)
    return snapshot


def safe_query(state, path):
    try:
        return query_gpus()
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, OSError) as exc:
        state.update(gpu_query_error=repr(exc), gpu_query_failed_utc=now())
        write(path, state)
        return []


def initial_identity(snapshot, env, root):
    code = (
        "from inter_hierarchy_MN40.hier_proxy_cycle_v9c.train import "
        "Hype_pointMLP,seed_all,configure_backbone,state_hash; "
        "from inter_hierarchy_MN40.hycore_c01_gs_halfmargin_20261020.train import state_digest; "
        "seed_all(4780); m=configure_backbone(Hype_pointMLP()); "
        "print('INITIAL_HASH='+state_hash(m)); print('ORIGINAL_DIGEST='+state_digest(m))"
    )
    output = subprocess.check_output([sys.executable, "-c", code], env=env, cwd=snapshot,
                                     text=True, stderr=subprocess.STDOUT, timeout=300)
    (root / "initial_identity.log").write_text(output)
    fields = dict(line.split("=", 1) for line in output.splitlines()
                  if line.startswith(("INITIAL_HASH=", "ORIGINAL_DIGEST=")))
    if fields.get("ORIGINAL_DIGEST") != ORIGINAL_E0_DIGEST:
        raise RuntimeError("Fresh backbone differs from original c01/cap/GS seed4780 initialization")
    if len(fields.get("INITIAL_HASH", "")) != 64:
        raise RuntimeError("Missing initial model hash")
    return fields["INITIAL_HASH"]


def accept_smoke(smoke, expected_hash, hier):
    manifest = json.loads((smoke / "manifest.json").read_text())
    config = manifest["training_config"]
    if (manifest["status"] != "completed" or manifest["official_test_read"]
        or manifest["train_size"] != 9840 or manifest["validation_size"] != 0
        or manifest["selection_domain"] != "smoke_train"
        or manifest["initialized_model_sha256"] != expected_hash
        or manifest["completed_epochs"] != 2
        or config["base_c"] != .1 or not config["native_GS"]
        or config["native_extra_cap"] != 2.3 or config["hier_enabled"] != (hier == "on")):
        raise RuntimeError("Train-only smoke manifest acceptance failed")
    import torch
    row = json.loads((smoke / "metrics_epoch_002.json").read_text())
    saved = torch.load(smoke / "last.pth", map_location="cpu", weights_only=False)
    if (config["steps_per_epoch"] != 2 or saved["completed_epochs"] != 2
        or saved.get("completed_optimizer_steps") != 4 or len(saved["rank_states"]) != 2):
        raise RuntimeError("Smoke must contain four updates and both complete rank states")
    for rank, restored in enumerate(saved["rank_states"]):
        counters = [int(value) for name, value in restored["BN_buffers"].items()
                    if name.endswith("num_batches_tracked")]
        if restored["rank"] != rank or not counters or set(counters) != {8}:
            raise RuntimeError("Smoke BN counters/rank states differ from four two-forward updates")
        if "sampler" not in restored or "private_generators" not in restored:
            raise RuntimeError("Smoke lacks sampler/private RNG recovery state")
    def finite_tree(value):
        if isinstance(value, torch.Tensor):
            return bool(torch.isfinite(value).all())
        if isinstance(value, dict):
            return all(finite_tree(v) for v in value.values())
        if isinstance(value, (list, tuple)):
            return all(finite_tree(v) for v in value)
        return True
    if not all(finite_tree(saved.get(key)) for key in ("net", "optimizer", "proxy", "proxy_optimizer")):
        raise RuntimeError("Smoke saved nonfinite weights or optimizer state")
    if hier == "off":
        if any(saved.get(key) is not None for key in ("proxy", "proxy_optimizer", "proxy_scheduler", "best_proxy")):
            raise RuntimeError("No-HIER arm constructed hierarchy state")
    else:
        steps = [int(value["step"]) for value in saved["proxy_optimizer"]["state"].values()]
        if not steps or set(steps) != {4}:
            raise RuntimeError("HIER Adam must complete exactly four smoke updates")
    if not (smoke / "best.pth").is_file():
        raise RuntimeError("Smoke failed to save a selected checkpoint")
    return {"manifest": manifest, "last_epoch": row}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--source-branch", required=True)
    parser.add_argument("--hier", choices=("off", "on"), required=True)
    parser.add_argument("--poll-seconds", type=int, default=30)
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=False)
    state = {"status": "preparing", "created_utc": now(), "queue_pid": os.getpid(),
             "commit": args.expected_commit, "branch": args.source_branch,
             "hier": args.hier, "epochs": 300, "steps_per_epoch": 300, "seed": 4780,
             "base_c": .1, "native_cap": 2.3, "native_GS": True,
             "radial_gamma": 500, "contrastive_margin": 2,
             "train_size": 9840, "validation_size": 0, "selection": "test2468 each epoch",
             "classes_per_rank": 16, "instances_per_class": 2,
             "source_checkpoint": None, "teacher": None, "production_started": False}
    path = args.root / "queue_status.json"
    write(path, state)
    try:
        snapshot = freeze_source(args.repo, args.root, args.expected_commit)
        env = dict(os.environ)
        env.update(OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2",
                   PYTHONUNBUFFERED="1", HDF5_USE_FILE_LOCKING="FALSE", MPLBACKEND="Agg",
                   V9C_SOURCE_COMMIT=args.expected_commit, V9C_SOURCE_BRANCH=args.source_branch,
                   HYCORE_TRAIN_SHARD_ORDER=json.dumps(TRAIN_ORDER))
        env.setdefault("PYTHONPYCACHEPREFIX", "/tmp/hycore_v9c_pycache_" + args.root.name)
        env["PYTHONPATH"] = str(snapshot) + ":" + env.get("PYTHONPATH", "")
        env.pop("CUDA_VISIBLE_DEVICES", None)
        tests = ["test_sampler", "test_geometry", "test_base_protocol", "test_train_contract"]
        commands = [
            [sys.executable, "-m", "unittest",
             *["inter_hierarchy_MN40.hier_proxy_cycle_v9c.tests." + name for name in tests], "-v"],
            [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=2",
             "--module", "inter_hierarchy_MN40.hier_proxy_cycle_v9c.tests.ddp_contract"],
        ]
        for index, command in enumerate(commands):
            if run_logged(command, args.root / f"cpu_check_{index}.log", env, snapshot, timeout=1200) != 0:
                raise RuntimeError(f"CPU contract {index} failed; inspect its log")
        expected_hash = initial_identity(snapshot, env, args.root)
        state.update(status="preparing_training_data", cpu_checks_passed=True,
                     expected_initial_sha256=expected_hash, original_e0_digest=ORIGINAL_E0_DIGEST,
                     updated_utc=now())
        write(path, state)
        cached_data = cache_data(args.data_dir, args.root.name, args.root)
        cached_identity = json.loads((args.root / "training_data_cache.json").read_text())
        cached_hashes = {row["file"]: row["sha256"] for row in cached_identity["train_files"]}
        if cached_hashes != TRAIN_SHA256:
            raise RuntimeError("Training HDF5 hashes differ from original c01 baseline")
        state["training_files_match_original"] = True

        state.update(status="waiting_for_two_idle_gpus", updated_utc=now())
        write(path, state)
        attempts = 0
        while True:
            rows = safe_query(state, path)
            candidates = idle_gpus(rows)
            state.update(gpu_observation=rows, updated_utc=now())
            write(path, state)
            if len(candidates) < 2:
                time.sleep(args.poll_seconds)
                continue
            chosen, locks = candidates[:2], []
            try:
                for row in chosen:
                    lock = open(f"/tmp/hycore_{os.getuid()}_gpu_{row['index']}.lock", "a+")
                    locks.append(lock)
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                time.sleep(10)
                still_idle = {row["uuid"] for row in idle_gpus(safe_query(state, path))}
                if not all(row["uuid"] in still_idle for row in chosen):
                    continue
                env.update(device_environment(chosen))
                attempts += 1
                smoke = args.root / f"smoke_{attempts:02d}"
                base = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=2",
                        "--module", "inter_hierarchy_MN40.hier_proxy_cycle_v9c.train",
                        "--hier", args.hier, "--data-dir", str(cached_data),
                        "--workers", "8", "--eval-batch-size", "16"]
                state.update(status="smoke_running", gpus=chosen, smoke_dir=str(smoke), updated_utc=now())
                write(path, state)
                smoke_command = base + ["--run-dir", str(smoke), "--epochs", "2", "--steps-per-epoch", "2",
                                       "--smoke", "--skip-final-test", "--smoke-eval-batches", "1",
                                       "--audit-epochs", "1"]
                if run_logged(smoke_command, args.root / f"smoke_{attempts:02d}.log", env, snapshot, timeout=3600) != 0:
                    raise RuntimeError("GPU smoke failed; production was not launched")
                acceptance = accept_smoke(smoke, expected_hash, args.hier)
                state.update(smoke_passed=True, smoke_initial_sha256=expected_hash, updated_utc=now())
                write(args.root / "smoke_acceptance.json",
                      {"accepted": True, "expected_initial_sha256": expected_hash,
                       "hier": args.hier, "smoke_dir": str(smoke),
                       "official_test_read": acceptance["manifest"]["official_test_read"]})
                write(path, state)
                time.sleep(5)
                idle_after = {row["uuid"] for row in idle_gpus(safe_query(state, path))}
                if not all(row["uuid"] in idle_after for row in chosen):
                    state.update(status="waiting_for_two_idle_gpus")
                    continue
                command = base + ["--run-dir", str(args.root / "main")]
                state.update(status="production_running", production_started=True,
                             production_dir=str(args.root / "main"), production_command=command,
                             production_started_utc=now(), updated_utc=now())
                write(path, state)
                code = run_logged(command, args.root / "production.log", env, snapshot)
                state.update(status="completed" if code == 0 else "production_failed",
                             exit_code=code, finished_utc=now(), updated_utc=now())
                write(path, state)
                return
            except BlockingIOError:
                time.sleep(args.poll_seconds)
            finally:
                for lock in locks:
                    lock.close()
    except BaseException as exc:
        state.update(status="failed", error=repr(exc), updated_utc=now())
        write(path, state)
        raise


if __name__ == "__main__":
    main()