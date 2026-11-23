"""Check the paired seed 4781 entry on one idle GPU with train-only smokes."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

from runner import ARM_NAMES, HERE, REPO, SEED, TS44_COMMIT, ts06


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def dataset_identity(reference, data_dir):
    manifest = read_json(reference)
    if manifest["status"] != "completed" or not re.fullmatch(r"[0-9a-f]{40}", manifest["commit"]):
        raise ValueError("Expected a completed, commit-identified reference manifest")
    identity = {}
    for split, size in (("train", 9840), ("test", 2468)):
        record = manifest["dataset_identity"][split]
        names = record["file_order"]
        hashes = record["file_sha256"]
        actual_names = {p.name for p in data_dir.glob(f"ply_data_{split}*.h5")}
        if record["size"] != size or len(names) != len(set(names)) or set(names) != set(hashes) or set(names) != actual_names:
            raise ValueError(f"Unexpected official ModelNet40 {split} shards or size")
        if not re.fullmatch(r"[0-9a-f]{64}", record["labels_sha256"]):
            raise ValueError(f"Invalid {split} label digest")
        for name in names:
            if ts06.digest(data_dir / name) != hashes[name]:
                raise ValueError(f"ModelNet40 reference shard differs: {name}")
        identity[split] = {key: record[key] for key in ("size", "file_order", "file_sha256", "labels_sha256")}
    return manifest["commit"], identity


def check_checkout(commit):
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    if revision != commit:
        raise RuntimeError("Deployment HEAD differs from the requested source commit")
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO, text=True).strip()
    if dirty:
        raise RuntimeError("Deployment checkout has uncommitted changes")


@contextmanager
def idle_device(index):
    import fcntl

    from inter_hierarchy_MN40.hier_proxy_scratch_v8.queue import idle_gpus, query_gpus

    def match():
        return next((gpu for gpu in idle_gpus(query_gpus()) if gpu["index"] == index), None)

    with Path(f"/tmp/hycore_{os.getuid()}_gpu_{index}.lock").open("a+") as lock:
        while True:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                time.sleep(15)
                continue
            first = match()
            if first is not None:
                time.sleep(2)
                second = match()
                if second is not None and first["uuid"] == second["uuid"]:
                    yield second
                    return
            fcntl.flock(lock, fcntl.LOCK_UN)
            time.sleep(15)


def environment(gpu):
    from inter_hierarchy_MN40.hier_proxy_scratch_v8.queue import device_environment

    return {**os.environ, **device_environment([gpu]), "PYTHONHASHSEED": str(SEED),
            "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1", "MPLBACKEND": "Agg",
            "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2"}


def command(args, arm, output, identity, *, smoke=False, expected_initial_model=None):
    argv = [sys.executable, "-B", "-u", str(HERE / "runner.py"), "--arm", arm,
            "--data-dir", str(args.data_dir.resolve()), "--expected-data", str(identity.resolve()),
            "--run-dir", str(output.resolve()), "--source-commit", args.source_commit,
            "--source-branch", args.source_branch]
    if smoke:
        argv.append("--smoke")
    else:
        argv += ["--expected-initial-model", expected_initial_model]
    return argv


def accept(root, expected, commit):
    import numpy as np
    import torch

    from inter_hierarchy_MN40.hier_proxy_cycle_v9c.geometry import assert_hier_shell

    traces, states, manifests = {}, {}, {}
    for arm in ("reference", *ARM_NAMES):
        run = root / arm
        m = manifests[arm] = read_json(run / "manifest.json")
        assert (m["status"], m["commit"], m["completed_epochs"], m["optimizer_steps"]) == ("completed", commit, 2, 4)
        assert m["ts44_source_commit"] == TS44_COMMIT and m["training_config"]["seed"] == SEED
        assert m["training_config"]["hier_enabled"] is (arm == "hier")
        assert m["dataset_identity"] == {"train": expected["train"]} and not m["official_test_read"]
        state = states[arm] = torch.load(run / "last_checkpoint.pth", map_location="cpu", weights_only=False)
        counters = {int(x) for key, x in state["net"].items() if key.endswith("num_batches_tracked")}
        assert counters == {8} and all(bool(torch.isfinite(x).all()) for x in state["net"].values())
        assert state["scheduler"]["T_max"] == state["scheduler"]["last_epoch"] == 2
        assert ("proxy" in state) is (arm == "hier")
        traces[arm] = [json.loads(line) for line in (run / "input_trace.jsonl").read_text().splitlines()]
        assert len(traces[arm]) == 4
        for step, trace in enumerate(traces[arm]):
            for key in ("cloud", "labels", "whole", "part", "whole_count", "part_count"):
                assert trace[key] == traces["reference"][step][key], (arm, step, key)
        np.testing.assert_allclose(traces[arm][0]["base_losses"], traces["reference"][0]["base_losses"],
                                   rtol=1e-4, atol=1e-5)
        rng = states["reference"]["rng"]
        assert state["rng"]["python"] == rng["python"]
        np.testing.assert_equal(state["rng"]["numpy"], rng["numpy"])
        assert torch.equal(state["rng"]["torch_cpu"], rng["torch_cpu"])
        assert len(state["rng"]["torch_cuda"]) == len(rng["torch_cuda"]) == 1
        assert torch.equal(state["rng"]["torch_cuda"][0], rng["torch_cuda"][0])
    assert len({m["initialized_model_sha256"] for m in manifests.values()}) == 1

    proxy_state = states["hier"]["proxy_optimizer"]
    group, = proxy_state["param_groups"]
    assert (group["initial_lr"], group["eps"], tuple(group["betas"]), group["weight_decay"]) == (.005, 1e-4, (.9, .999), .01)
    assert abs(group["lr"] - .0005) < 1e-12
    assert {int(item["step"]) for item in proxy_state["state"].values()} == {4}
    assert states["hier"]["proxy_scheduler"]["T_max"] == states["hier"]["proxy_scheduler"]["last_epoch"] == 2
    initial_proxy = torch.load(root / "hier" / "initial_proxies.pth", map_location="cpu", weights_only=False)
    from inter_hierarchy_MN40.hier_proxy_scratch_v8.hier_loss import HIERLoss

    assert torch.equal(initial_proxy["tangent_proxies"], HIERLoss(seed=SEED).tangent_proxies)
    trained_proxy = states["hier"]["proxy"]["tangent_proxies"]
    assert bool(torch.isfinite(trained_proxy).all()) and not torch.equal(trained_proxy, initial_proxy["tangent_proxies"])
    snapshot = torch.load(root / "hier" / "actual_training_epoch_001.pth", map_location="cpu", weights_only=False)
    assert snapshot["hier_mu"].shape == (32, 256) and snapshot["proxy_tangent"].shape == (512, 256)
    assert (snapshot["lambda_hier"], snapshot["sample_K"], snapshot["proxy_K"]) == (.5, 20, 20)
    assert snapshot["whole_gradient_norm"] > 0
    assert_hier_shell(snapshot["hier_mu"])
    hierarchy = []
    for epoch in (1, 2):
        metrics = read_json(root / "hier" / f"metrics_epoch_{epoch:03d}.json")["hierarchy"]
        assert (metrics["cached_pool_updates"], metrics["max_sample_pool"], metrics["proxy_updates"]) == (1, 64, epoch * 2)
        assert 0 < metrics["cached_sample_triplets"] <= 50 * 32
        hierarchy.append(metrics)
    return dict(passed=True, source_commit=commit, ts44_source_commit=TS44_COMMIT, seed=SEED,
                initialized_model_sha256=manifests["h0"]["initialized_model_sha256"],
                first_step_base_losses={arm: steps[0]["base_losses"] for arm, steps in traces.items()},
                augmented_inputs_crops_and_main_rng_match=True, four_main_updates_per_smoke=True,
                whole_gradient_norm=snapshot["whole_gradient_norm"], hierarchy_metrics=hierarchy)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--reference-manifest", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--source-branch", required=True)
    args = parser.parse_args()
    from inter_hierarchy_MN40.hier_proxy_scratch_v8.queue import now, write

    args.root.mkdir(parents=True, exist_ok=False)
    status = args.root / "status.json"
    destination = args.root / "acceptance.json"
    try:
        check_checkout(args.source_commit)
        reference_commit, expected = dataset_identity(args.reference_manifest, args.data_dir)
        identity = args.root / "dataset_identity.json"
        write(identity, expected)
        for arm in ("reference", *ARM_NAMES):
            write(status, dict(state="waiting_for_idle_gpu", arm=arm, gpu=args.gpu, updated_at=now()))
            with idle_device(args.gpu) as gpu:
                write(status, dict(state="running_smoke", arm=arm, gpu=gpu, updated_at=now()))
                with (args.root / f"{arm}.log").open("x", encoding="utf-8") as log:
                    subprocess.run(command(args, arm, args.root / arm, identity, smoke=True), cwd=REPO,
                                   env=environment(gpu), stdin=subprocess.DEVNULL, stdout=log,
                                   stderr=subprocess.STDOUT, timeout=1800, check=True)
        result = accept(args.root, expected, args.source_commit)
        result.update(reference_commit=reference_commit, reference_manifest_sha256=ts06.digest(args.reference_manifest),
                      dataset_identity_sha256=ts06.digest(identity), updated_at=now())
        write(destination, result)
        write(status, dict(state="passed", **result))
    except BaseException as error:
        result = dict(passed=False, source_commit=args.source_commit, error=repr(error), updated_at=now())
        write(destination, result)
        write(status, dict(state="failed", **result))
        raise


if __name__ == "__main__":
    main()
