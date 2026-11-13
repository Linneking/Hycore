"""TS35 GPU2 gate: verify TS06 parity, enable HIER, then start a fresh run."""
from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

from shuffle_runner import NAME, REPO, ts06
from inter_hierarchy_MN40.hier_proxy_scratch_v8.queue import (
    device_environment, idle_gpus, now, query_gpus, write,
)

GPU = 2
INITIAL = "8926765a4842652b751873fb89aa37d24d9b0f4f2d831d853b9cc0c1cd46472d"
REFERENCE_COMMIT = "e20cf8c1698ff3048c15ad66d01dfbb5b0a99731"


def json_file(path):
    return json.loads(path.read_text(encoding="utf-8"))


def target_idle():
    try:
        return next((row for row in idle_gpus(query_gpus()) if row["index"] == GPU), None)
    except (subprocess.SubprocessError, OSError):
        return None


def read_smoke(root, mode, expected_train, commit):
    import torch

    directory = root / mode
    manifest = json_file(directory / "manifest.json")
    assert manifest["commit"] == commit
    assert manifest["status"] == "completed"
    assert manifest["completed_epochs"] == 2 and manifest["optimizer_steps"] == 4
    assert manifest["dataset_identity"]["train"] == expected_train
    assert manifest["initialized_model_sha256"] == INITIAL
    assert manifest["official_test_read"] is False and manifest["test_size"] == 0
    assert manifest["training_config"]["hier_enabled"] is (mode == "enabled")
    state = torch.load(directory / "last_checkpoint.pth", map_location="cpu", weights_only=False)
    counters = [int(tensor) for name, tensor in state["net"].items() if name.endswith("num_batches_tracked")]
    assert counters and set(counters) == {8}
    assert all(bool(torch.isfinite(tensor).all()) for tensor in state["net"].values())
    assert state["scheduler"]["T_max"] == state["scheduler"]["last_epoch"] == 2
    trace = [json.loads(line) for line in (directory / "input_trace.jsonl").read_text().splitlines()]
    assert len(trace) == 4
    return trace, state


def accept(root, expected_train, commit):
    import numpy as np
    import torch
    from inter_hierarchy_MN40.hier_proxy_cycle_v9c.geometry import assert_hier_shell

    reference, original = read_smoke(root, "reference", expected_train, commit)
    off, overlay = read_smoke(root, "off", expected_train, commit)
    enabled, hierarchy = read_smoke(root, "enabled", expected_train, commit)
    for control, off_step, hier_step in zip(reference, off, enabled):
        for key in ("cloud", "labels", "whole", "part", "whole_count", "part_count"):
            assert control[key] == off_step[key] == hier_step[key], key
        np.testing.assert_allclose(off_step["base_losses"], control["base_losses"], rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(enabled[0]["base_losses"], reference[0]["base_losses"], rtol=1e-4, atol=1e-5)
    for key, value in original["net"].items():
        torch.testing.assert_close(overlay["net"][key], value, rtol=1e-4, atol=1e-5, msg=key)
    for control in (overlay, hierarchy):
        assert torch.equal(original["rng"]["torch_cpu"], control["rng"]["torch_cpu"])
        for first, second in zip(original["rng"]["torch_cuda"], control["rng"]["torch_cuda"]):
            assert torch.equal(first, second)
        np.testing.assert_equal(original["rng"]["numpy"], control["rng"]["numpy"])
    group, = hierarchy["proxy_optimizer"]["param_groups"]
    assert (group["initial_lr"], group["eps"], tuple(group["betas"]), group["weight_decay"]) == (
        .005, 1e-4, (.9, .999), .01)
    assert math.isclose(group["lr"], .0005, abs_tol=1e-12)
    assert {int(item["step"]) for item in hierarchy["proxy_optimizer"]["state"].values()} == {4}
    assert hierarchy["proxy_scheduler"]["T_max"] == hierarchy["proxy_scheduler"]["last_epoch"] == 2
    snapshot = torch.load(root / "enabled" / "actual_training_epoch_001.pth", map_location="cpu", weights_only=False)
    assert snapshot["hier_mu"].shape == (32, 256)
    assert snapshot["proxy_tangent"].shape == (512, 256)
    assert (snapshot["sample_K"], snapshot["proxy_K"], snapshot["lambda_hier"]) == (20, 20, .5)
    assert snapshot["whole_gradient_norm"] > 0
    assert_hier_shell(snapshot["hier_mu"])
    metrics = hierarchy["metrics"]["hierarchy"]
    assert metrics["sample_active_triplets"] > 0 and metrics["proxy_active_triplets"] > 0
    return dict(accepted=True, reference="TS06", reference_commit=REFERENCE_COMMIT,
                reference_initialization=INITIAL, smoke_updates_per_mode=4,
                augmentation_labels_crops_and_rng_match=True, h0_model_state_close=True,
                whole_gradient_norm=snapshot["whole_gradient_norm"],
                last_epoch_smoke_hierarchy=metrics, official_test_read=False)


def launch(command, env, log_path, timeout=None):
    with log_path.open("x") as stream:
        subprocess.run(command, cwd=REPO, env=env, stdin=subprocess.DEVNULL,
                       stdout=stream, stderr=subprocess.STDOUT, timeout=timeout, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--ts06-manifest", required=True, type=Path)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--source-branch", required=True)
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=False)
    path = args.root / "queue_status.json"
    state = dict(experiment_id="TS35", canonical_name=NAME, target_physical_gpu=GPU,
                 queue_pid=os.getpid(), commit=args.expected_commit, branch=args.source_branch,
                 status="checking_source_and_data", production_started=False)

    def status(value, **details):
        state.update(status=value, updated_utc=now(), **details)
        write(path, state)

    status("checking_source_and_data")
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO, text=True).strip()
        if revision != args.expected_commit or dirty:
            raise RuntimeError("TS35 isolated checkout changed")
        reference = json_file(args.ts06_manifest)
        if not (reference["commit"] == REFERENCE_COMMIT and reference["status"] == "completed"
                and reference["initialized_model_sha256"] == INITIAL):
            raise RuntimeError("TS06 reference identity changed")
        expected_train = reference["dataset_identity"]["train"]
        for filename, digest in expected_train["file_sha256"].items():
            if ts06.digest(args.data_dir / filename) != digest:
                raise RuntimeError(f"TS06 training shard differs: {filename}")
        command = [sys.executable, str(Path(__file__).with_name("shuffle_runner.py")),
                   "--variant", "c01-cap", "--data-dir", str(args.data_dir),
                   "--source-commit", args.expected_commit, "--source-branch", args.source_branch]
        env = dict(os.environ, HYCORE_TRAIN_SHARD_ORDER=json.dumps(expected_train["file_order"]),
                   PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1",
                   OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2")
        status("waiting_for_gpu")
        attempt = 0
        while True:
            chosen = target_idle()
            if chosen is None:
                time.sleep(15)
                continue
            with Path(f"/tmp/hycore_{os.getuid()}_gpu_{GPU}.lock").open("a+") as lock:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    time.sleep(15)
                    continue
                time.sleep(10)
                check = target_idle()
                if check is None or check["uuid"] != chosen["uuid"]:
                    continue
                attempt += 1
                attempt_root = args.root / f"smoke_{attempt:02d}"
                attempt_root.mkdir()
                env.update(device_environment([chosen]))
                for mode in ("reference", "off", "enabled"):
                    check = target_idle()
                    if check is None or check["uuid"] != chosen["uuid"]:
                        break
                    status("smoke_running", smoke_mode=mode, smoke_dir=str(attempt_root))
                    launch(command + ["--mode", mode, "--run-dir", str(attempt_root / mode),
                                      "--smoke", "--epochs", "2", "--steps-per-epoch", "2"],
                           env, attempt_root / f"{mode}.log", timeout=1800)
                    time.sleep(5)
                else:
                    accepted = accept(attempt_root, expected_train, args.expected_commit)
                    write(args.root / "smoke_acceptance.json", accepted)
                    check = target_idle()
                    if check is not None and check["uuid"] == chosen["uuid"]:
                        production = args.root / "main"
                        status("production_running", smoke_passed=True, production_started=True,
                               production_dir=str(production), production_started_utc=now())
                        launch(command + ["--mode", "enabled", "--run-dir", str(production)],
                               env, args.root / "production.log")
                        manifest = json_file(production / "manifest.json")
                        if (manifest["status"], manifest["completed_epochs"], manifest["optimizer_steps"]) != (
                                "completed", 300, 92100):
                            raise RuntimeError("TS35 production completion did not match the authorized budget")
                        status("completed")
                        return
            status("waiting_for_gpu")
    except BaseException as error:
        status("failed", error=repr(error))
        raise


if __name__ == "__main__":
    main()
