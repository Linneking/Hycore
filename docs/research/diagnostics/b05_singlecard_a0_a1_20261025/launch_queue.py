"""One approved B05 arm, bound to one physical GPU, with deferred smoke.

CPU preparation never allocates CUDA. The existing user/GPU lock and two idle
observations precede the independent train-only smoke and fresh production.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
RELATIVE = HERE.relative_to(REPO)
for item in (HERE, REPO, REPO / "inter_hierarchy_MN40"):
    sys.path.insert(0, str(item))

import runner
from inter_hierarchy_MN40.hier_proxy_cycle_v9c.queue import TRAIN_ORDER, TRAIN_SHA256
from inter_hierarchy_MN40.hier_proxy_scratch_v8.queue import (
    now, write, query_gpus, idle_gpus, device_environment, run_logged, cache_data,
)


def freeze_source(repo, root, expected, variant):
    parent = repo / "docs/research/diagnostics/s4780_native_b05_20261023/launch_queue.py"
    spec = importlib.util.spec_from_file_location("b05_dual_queue_dependency", parent)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    snapshot = module.freeze_source(repo, root, expected, "b05")
    tracked = set(subprocess.check_output(
        ["git", "-C", str(repo), "ls-files", "-z", "--", RELATIVE.as_posix()]
    ).decode().split("\0"))
    destination = snapshot / RELATIVE
    destination.mkdir(parents=True, exist_ok=False)
    identity = json.loads((root / "source_identity.json").read_text())
    for source in sorted((repo / RELATIVE).glob("*.py")):
        relative = RELATIVE / source.name
        if relative.as_posix() not in tracked:
            raise RuntimeError("Single-card sources must be committed: " + str(relative))
        target = destination / source.name
        shutil.copy2(source, target)
        identity["files"][relative.as_posix()] = hashlib.sha256(target.read_bytes()).hexdigest()
    identity["successor"] = dict(
        relative_path=RELATIVE.as_posix(), variant=variant, seed=4780,
        epochs=300, steps_per_epoch=200, world_size=1, global_batch=32,
        sample_K=10, proxy_K=20, proxy_optimizer=runner.recipe(variant),
        source_checkpoint=None, teacher=None,
    )
    write(root / "source_identity.json", identity)
    return snapshot


def safe_query(state, path):
    try:
        return query_gpus()
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, OSError) as exc:
        state.update(gpu_query_error=repr(exc), gpu_query_failed_utc=now())
        write(path, state)
        return []


def target_idle(rows, gpu):
    return next((row for row in idle_gpus(rows) if row["index"] == gpu), None)


def initial_identity(snapshot, env, root, variant):
    code = (
        "import sys;sys.path.insert(0,sys.argv[1]);import runner;"
        "n=runner.build_namespace(sys.argv[2]);n['seed_all'](4780);"
        "m=n['configure_backbone'](n['Hype_pointMLP']());"
        "assert n['state_digest'](m)==runner.INITIAL_DIGEST;"
        "print('INITIAL_HASH='+n['state_hash'](m),flush=True)"
    )
    output = subprocess.check_output(
        [sys.executable, "-c", code, str(snapshot / RELATIVE), variant],
        cwd=snapshot, env=env, text=True, stderr=subprocess.STDOUT, timeout=300)
    (root / "initial_identity.log").write_text(output)
    values = [line.split("=", 1)[1].strip() for line in output.splitlines()
              if line.startswith("INITIAL_HASH=")]
    if len(values) != 1 or len(values[0]) != 64:
        raise RuntimeError("Missing independently reconstructed initialization")
    return values[0]


def finite_tree(value, torch):
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(finite_tree(item, torch) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite_tree(item, torch) for item in value)
    return True


def accept_run(directory, variant, initial_hash, smoke):
    import torch
    manifest = json.loads((directory / "manifest.json").read_text())
    config = manifest["training_config"]
    epochs, steps = (2, 2) if smoke else (300, 200)
    expected = dict(world_size=1, global_batch=32, classes_per_rank=16,
                    instances_per_class=2, sample_K=10, proxy_K=20, T=50,
                    seed=4780, epochs=epochs, steps_per_epoch=steps,
                    base_c=.1, native_extra_cap=2.3, native_GS=True,
                    hier_c=.1, hier_view_depth=4.6, hier_enabled=True,
                    variant=variant, contrastive_margin=2., radial_gamma=500.)
    if any(config.get(key) != value for key, value in expected.items()):
        raise RuntimeError("Actual training configuration differs from approved arm")
    if not (manifest["status"] == "completed"
            and manifest["completed_epochs"] == epochs
            and manifest["train_size"] == 9840 and manifest["validation_size"] == 0
            and manifest["official_test_read"] is (not smoke)
            and manifest["selection_domain"] == ("smoke_train" if smoke else "test")
            and manifest["resume_identity"] is None
            and manifest["initialized_model_sha256"] == initial_hash):
        raise RuntimeError("Actual completion/selection/initialization identity differs")
    if not smoke and manifest["selection_size"] != 2468:
        raise RuntimeError("Production must evaluate the complete official test")
    saved = torch.load(directory / "last.pth", map_location="cpu", weights_only=False)
    if saved["completed_optimizer_steps"] != epochs * steps or len(saved["rank_states"]) != 1:
        raise RuntimeError("Incorrect optimizer count or single-rank recovery state")
    state, = saved["rank_states"]
    counters = [int(value) for name, value in state["BN_buffers"].items()
                if name.endswith("num_batches_tracked")]
    if state["rank"] != 0 or not counters or set(counters) != {2 * epochs * steps}:
        raise RuntimeError("BN must update exactly twice per optimizer step")
    if not {"sampler", "private_generators", "rng"}.issubset(state):
        raise RuntimeError("Missing complete sampler/BN/random recovery state")
    if not all(saved.get(key) is not None and finite_tree(saved[key], torch)
               for key in ("net", "optimizer", "proxy", "proxy_optimizer")):
        raise RuntimeError("Missing/nonfinite actual model, proxies or optimizer")
    actual = runner.verify_optimizer_payload(saved, epochs, epochs, variant)
    counts = {int(item["step"]) for item in saved["proxy_optimizer"]["state"].values()}
    if counts != {epochs * steps}:
        raise RuntimeError("Actual AdamW did not perform the approved update count")
    if smoke:
        snapshot = torch.load(directory / "actual_training_epoch_001.pth", map_location="cpu", weights_only=False)
        if (len(snapshot["ids"]) != 32 or snapshot["rank_order"] != [0]
                or snapshot["proxy_tangent"].shape != (512, 256)):
            raise RuntimeError("Real-forward sample/proxy shapes differ")
        audit = json.loads((directory / "parameter_audit_epoch_001.json").read_text())
        if (audit["sample_K"], audit["proxy_K"]) != (10, 20):
            raise RuntimeError("Parameter audit must use the same split K as production")
        metrics = json.loads((directory / "metrics_epoch_002.json").read_text())
        hierarchy = metrics["telemetry"]["hierarchy"]
        for name, k, population in (("sample_graph", 10, 32), ("proxy_graph", 20, 512)):
            graph = hierarchy[name]
            settings = graph["settings_observed"]
            if (settings["topk_including_self"] != [k]
                    or settings["t_per_anchor"] != [50]
                    or settings["exclude_self_negative"] != [True]
                    or graph["counts"]["batch_size"] != 2 * population):
                raise RuntimeError("Actual smoke mining differs from approved split K/T/population")
    if not (directory / "best.pth").is_file():
        raise RuntimeError("Missing selected checkpoint")
    return dict(accepted=True, variant=variant, epochs=epochs, updates=epochs * steps,
                BN_updates=2 * epochs * steps, actual_proxy_optimizer=actual,
                initial_sha256=initial_hash, official_test_read=not smoke)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--source-branch", required=True)
    parser.add_argument("--variant", choices=("a0", "a1"), required=True)
    parser.add_argument("--gpu", type=int, choices=(0, 1), required=True)
    parser.add_argument("--poll-seconds", type=int, default=60)
    args = parser.parse_args()
    if args.poll_seconds < 10 or (args.variant, args.gpu) not in (("a1", 0), ("a0", 1)):
        parser.error("Approved mapping: A1 GPU0, A0 GPU1; poll >=10 seconds")
    args.repo, args.data_dir = args.repo.resolve(), args.data_dir.resolve()
    args.root.mkdir(parents=True, exist_ok=False)
    path = args.root / "queue_status.json"
    state = dict(status="preparing_cpu_only", queue_pid=os.getpid(), created_utc=now(),
                 commit=args.expected_commit, branch=args.source_branch, variant=args.variant,
                 target_physical_gpu=args.gpu, seed=4780, epochs=300, steps_per_epoch=200,
                 world_size=1, global_batch=32, classes=16, instances_per_class=2,
                 sample_K=10, proxy_K=20, proxy_optimizer=runner.recipe(args.variant),
                 source_checkpoint=None, teacher=None, production_started=False)
    write(path, state)
    try:
        snapshot = freeze_source(args.repo, args.root, args.expected_commit, args.variant)
        env = dict(os.environ)
        env.update(OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2",
                   PYTHONUNBUFFERED="1", HDF5_USE_FILE_LOCKING="FALSE", MPLBACKEND="Agg",
                   CUDA_VISIBLE_DEVICES="", V9C_SOURCE_COMMIT=args.expected_commit,
                   V9C_SOURCE_BRANCH=args.source_branch,
                   HYCORE_TRAIN_SHARD_ORDER=json.dumps(TRAIN_ORDER),
                   PYTHONPYCACHEPREFIX="/tmp/hycore_cb32_pycache_" + args.root.name)
        env["PYTHONPATH"] = ":".join((str(snapshot), str(snapshot / "inter_hierarchy_MN40"),
                                      env.get("PYTHONPATH", "")))
        checks = [sys.executable, "-m", "unittest", "discover", "-s", str(snapshot / RELATIVE),
                  "-p", "test_contracts.py", "-v"]
        log = args.root / "cpu_contracts.log"
        if run_logged(checks, log, env, snapshot, timeout=1200) != 0 or "skipped=" in log.read_text():
            raise RuntimeError("CPU contracts failed or skipped; no GPU was allocated")
        initial_hash = initial_identity(snapshot, env, args.root, args.variant)
        cached = cache_data(args.data_dir, args.root.name, args.root)
        data = json.loads((args.root / "training_data_cache.json").read_text())
        if {item["file"]: item["sha256"] for item in data["train_files"]} != TRAIN_SHA256:
            raise RuntimeError("Training HDF5 differs from the reviewed source")
        state.update(status="waiting_for_target_gpu", cpu_checks_passed=True,
                     expected_initial_sha256=initial_hash, training_files_match=True, updated_utc=now())
        write(path, state)
        attempts = 0
        while True:
            rows = safe_query(state, path)
            state.update(gpu_observation=rows, updated_utc=now())
            write(path, state)
            chosen = target_idle(rows, args.gpu)
            if chosen is None:
                time.sleep(args.poll_seconds)
                continue
            lock = open(f"/tmp/hycore_{os.getuid()}_gpu_{args.gpu}.lock", "a+")
            try:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    time.sleep(args.poll_seconds)
                    continue
                time.sleep(10)
                check = target_idle(safe_query(state, path), args.gpu)
                if check is None or check["uuid"] != chosen["uuid"]:
                    continue
                env.update(device_environment([chosen]))
                attempts += 1
                smoke = args.root / f"smoke_{attempts:02d}"
                base = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=1",
                        str(snapshot / RELATIVE / "runner.py"), "--variant", args.variant, "--hier", "on",
                        "--data-dir", str(cached), "--workers", "8", "--eval-batch-size", "16"]
                state.update(status="smoke_running", gpus=[chosen], smoke_dir=str(smoke), updated_utc=now())
                write(path, state)
                smoke_command = base + ["--run-dir", str(smoke), "--epochs", "2", "--steps-per-epoch", "2",
                                        "--smoke", "--skip-final-test", "--smoke-eval-batches", "1", "--audit-epochs", "1"]
                if run_logged(smoke_command, args.root / f"smoke_{attempts:02d}.log", env, snapshot, timeout=3600) != 0:
                    raise RuntimeError("Deferred single-GPU smoke failed; production was not launched")
                accepted = accept_run(smoke, args.variant, initial_hash, True)
                write(args.root / "smoke_acceptance.json", accepted)
                state.update(smoke_passed=True, updated_utc=now())
                write(path, state)
                time.sleep(5)
                check = target_idle(safe_query(state, path), args.gpu)
                if check is None or check["uuid"] != chosen["uuid"]:
                    state.update(status="waiting_for_target_gpu", updated_utc=now())
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
                    state["completion_acceptance"] = accept_run(production, args.variant, initial_hash, False)
                state.update(status="completed" if code == 0 else "production_failed", exit_code=code,
                             finished_utc=now(), updated_utc=now())
                write(path, state)
                return
            finally:
                lock.close()
    except BaseException as exc:
        state.update(status="failed", error=repr(exc), updated_utc=now())
        write(path, state)
        raise


if __name__ == "__main__":
    main()
