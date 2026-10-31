"""Queue Experiment I on an idle GPU, validate a real-data smoke, then train fresh.

The launcher never modifies an existing run.  It rechecks GPU idleness before
both the smoke and production phases, and refuses production after a failed gate.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from segmentation_ShapeNetPart.hycore_seg_v1.train import ensure_idle
from segmentation_ShapeNetPart.hycore_seg_v1.utils import git_identity, timestamp, write_json


TRAIN_MODULE = "segmentation_ShapeNetPart.hycore_seg_v8.train"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def wait_idle(gpu, root):
    """Three spaced checks prevent a transient idle reading from taking a GPU."""
    while True:
        observations = []
        for attempt in range(3):
            try:
                observations.append(ensure_idle(gpu))
            except (RuntimeError, subprocess.SubprocessError) as error:
                write_json(root / "queue.json", {"state": "waiting_for_idle_gpu",
                                                 "reason": str(error), "at": timestamp()})
                break
            if attempt < 2:
                time.sleep(10)
        else:
            write_json(root / "queue.json", {"state": "idle_gpu_acquired",
                                             "checks": observations, "at": timestamp()})
            return
        time.sleep(30)


def command(args, out, data, smoke=False, initial_hash=None):
    cmd = [sys.executable, "-m", TRAIN_MODULE, "--config", str(args.config),
           "--arm", "i",
           "--data", str(data), "--out", str(out), "--physical-gpu", str(args.gpu)]
    if smoke:
        cmd.extend(("--protocol", "validation", "--epochs", "2", "--max-steps", "2",
                    "--limit-train", "64", "--limit-val", "16", "--no-final-test",
                    "--part-min", "600", "--part-max", "600"))
    if initial_hash is not None:
        cmd.extend(("--expected-initial-sha256", initial_hash))
    return cmd


def invoke(cmd, root, phase):
    with (root / (phase + ".log")).open("w", encoding="utf-8") as stream:
        result = subprocess.run(cmd, stdout=stream, stderr=subprocess.STDOUT,
                                env=os.environ.copy(), check=False)
    if result.returncode:
        raise RuntimeError(f"{phase} exited {result.returncode}; see {phase}.log")


def validate_smoke(smoke):
    import torch

    manifest = read_json(smoke / "manifest.json")
    accepted = read_json(smoke / "acceptance.json")
    metrics = [json.loads(line) for line in (smoke / "metrics.jsonl").read_text().splitlines()
               if line.strip()]
    geometry = [json.loads(line) for line in (smoke / "geometry.jsonl").read_text().splitlines()
                if line.strip()]
    e0 = torch.load(smoke / "e0.pth", map_location="cpu", weights_only=False)
    last = torch.load(smoke / "last.pth", map_location="cpu", weights_only=False)
    checks = {
        "complete_two_epochs_four_updates": accepted.get("complete") is True
            and accepted.get("epochs") == 2 and accepted.get("updates") == 4
            and len(metrics) == 2 and [row.get("steps") for row in metrics] == [2, 2],
        "full_size_real_data_no_test": manifest.get("config", {}).get("batch_size") == 32
            and manifest.get("config", {}).get("num_points") == 2048
            and manifest.get("config", {}).get("part_min") == 600
            and manifest.get("config", {}).get("part_max") == 600
            and manifest.get("selection_split") == "val"
            and accepted.get("final_test_evaluations") == 0
            and not (smoke / "final_test.json").exists(),
        "bn_two_updates_per_batch": bool(metrics)
            and all(row.get("bn_counts") for row in metrics)
            and all(value == 8 for value in metrics[-1]["bn_counts"].values()),
        "finite_model_and_optimizer": all(row.get("model_finite") and row.get("optimizer_finite")
            for row in metrics),
        "fresh_checkpoint": e0.get("epoch") == 0 and e0.get("updates") == 0
            and last.get("epoch") == 2 and last.get("updates") == 4,
        "part_reencoding_recorded": manifest.get("part_input") ==
            "gathered_raw_xyz_reencoded_through_full_trunk_and_conv9",
    }
    def gradient_groups(item, loss):
        payload = item.get("gradients", {})
        return payload.get(loss, {}).get("parameter_groups", {})
    sampled = [row for row in geometry if row.get("gradients")]
    checks["regularizer_reaches_first_m_only"] = bool(sampled) and all(
        gradient_groups(row, loss).get("m2", {}).get("status") in {"zero", "nonzero"}
        and gradient_groups(row, loss).get("m3", {}).get("status") == "none"
        for row in sampled for loss in ("contr", "hier"))
    checks["all_checks_pass"] = all(checks.values())
    return checks, manifest["initial_model_sha256"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--smoke-data", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--gpu", type=int, required=True)
    args = p.parse_args()
    if args.gpu < 0 or not args.config.is_file():
        raise ValueError("Invalid GPU or configuration")
    config = read_json(args.config)
    if config.get("arm") != "i" or (config.get("alpha"), config.get("beta")) != (.01, .01):
        raise ValueError("Only the approved active Experiment I configuration may launch")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    root = args.out.resolve()
    root.mkdir(parents=True, exist_ok=False)
    identity = git_identity()
    if identity["source_status"]:
        raise RuntimeError("Production source is dirty")
    write_json(root / "submitted.json", {"source": identity, "config": config,
                                         "gpu": args.gpu, "started": timestamp(),
                                         "smoke_data": str(args.smoke_data),
                                         "production_data": str(args.data)})
    try:
        wait_idle(args.gpu, root)
        smoke = root / "smoke"
        write_json(root / "queue.json", {"state": "smoke_running", "at": timestamp()})
        invoke(command(args, smoke, args.smoke_data, smoke=True), root, "smoke")
        checks, initial_hash = validate_smoke(smoke)
        write_json(root / "smoke_validation.json", checks)
        if not checks["all_checks_pass"]:
            raise AssertionError("Real-data smoke acceptance failed")
        wait_idle(args.gpu, root)
        production = root / "main"
        write_json(root / "queue.json", {"state": "production_running", "at": timestamp(),
                                         "initial_model_sha256": initial_hash})
        invoke(command(args, production, args.data, initial_hash=initial_hash), root,
               "production")
        accepted = read_json(production / "acceptance.json")
        if accepted.get("complete") is not True or accepted.get("epochs") != 200:
            raise AssertionError("Production did not complete its 200-epoch budget")
        write_json(root / "queue.json", {"state": "complete", "at": timestamp(),
                                         "best_epoch": accepted["best_epoch"]})
    except Exception as error:
        write_json(root / "queue.json", {"state": "failed", "error": repr(error),
                                         "at": timestamp()})
        raise


if __name__ == "__main__":
    main()
