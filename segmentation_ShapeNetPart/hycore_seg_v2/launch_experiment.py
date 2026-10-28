"""One authorized architecture: real-data smoke, then a fresh 200-epoch run."""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch

from segmentation_ShapeNetPart.hycore_seg_v1.utils import git_identity, timestamp, write_json
from .train import ensure_idle


def wait_for_idle(gpu, root):
    while True:
        try:
            checked = ensure_idle(gpu)
            write_json(root / "queue.json", {"state": "idle_gpu_acquired", "check": checked})
            return
        except RuntimeError as exc:
            write_json(root / "queue.json", {"state": "waiting_for_idle_gpu", "reason": str(exc), "at": timestamp()})
            time.sleep(30)


def validate_smoke(smoke):
    manifest = json.loads((smoke / "manifest.json").read_text())
    accepted = json.loads((smoke / "acceptance.json").read_text())
    metrics = [json.loads(line) for line in (smoke / "metrics.jsonl").read_text().splitlines()]
    steps = [json.loads(line) for line in (smoke / "steps.jsonl").read_text().splitlines()]
    state = torch.load(smoke / "last.pth", map_location="cpu", weights_only=False)
    checks = {
        "two_epochs_two_steps": len(metrics) == 2 and all(row["steps"] == 2 for row in metrics),
        "four_updates": accepted["updates"] == state["updates"] == 4,
        "no_test": accepted["final_test_evaluations"] == 0 and not (smoke / "final_test.json").exists(),
        "full_point_count_batch": manifest["config"]["num_points"] == 2048 and manifest["config"]["batch_size"] == 32,
        "seed4780": manifest["seed"] == 4780,
        "bn_once_per_step": all(n == 4 for n in metrics[-1]["bn_counts"].values()),
        "finite_parameters": all(torch.isfinite(v).all().item() for v in state["model"].values()),
        "finite_optimizer": all(torch.isfinite(v).all().item() for x in state["optimizer"]["state"].values()
                                for v in x.values() if torch.is_tensor(v)),
        "scheduler_keeps_200": state["scheduler"]["T_max"] == 200 and state["scheduler"]["last_epoch"] == 2,
        "full_state": all(k in state for k in ("rng", "train_generator", "eval_generator", "part_generator", "config", "source")),
        "regularizer_reaches_backbone": steps[0]["reg_shared_gradient_norm"] > 0,
        "ce_reaches_backbone": steps[0]["ce_shared_gradient_norm"] > 0,
        "head_has_no_regularizer_gradient": steps[0]["head_regularizer_gradient_is_none"],
        "margin1000_over_n": all(abs(row["gamma"] * row["part_points"] - 1000) < 1e-7 for row in steps),
        "delta4": all(row["delta"] == 4 for row in steps),
        "flip_no_self": all(row["self_pair_fraction"] == 0 for row in steps),
    }
    result = {"passed": all(checks.values()), "checks": checks,
              "initial_model_sha256": manifest["initial_model_sha256"],
              "first_input_sha256": metrics[0]["first_input_sha256"]}
    write_json(smoke.parent / "smoke_validation.json", result)
    if not result["passed"]:
        raise RuntimeError("Smoke acceptance failed: " + str(checks))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-data", required=True)
    parser.add_argument("--production-data", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--architecture", choices=("p0", "pglobal", "pseg3"), required=True)
    args = parser.parse_args()
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=False)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["OMP_NUM_THREADS"] = "4"
    os.environ["MKL_NUM_THREADS"] = "4"
    identity = git_identity()
    if identity["source_status"]:
        raise RuntimeError("Commit segmentation source before dispatch")
    write_json(root / "dispatch.json", {"source": identity, "started": timestamp(), "gpu": args.gpu,
               "architecture": args.architecture, "protocol": "antao_testsel", "seed": 4780,
               "pid": os.getpid(), "python": sys.executable, "teacher": None, "source_checkpoint": None,
               "scope": "native_v2_fresh_smoke_then_fresh200"})
    config = Path(__file__).parent / "configs" / "native_seg.json"
    command = [sys.executable, "-m", "segmentation_ShapeNetPart.hycore_seg_v2.train", "--config", str(config),
               "--architecture", args.architecture, "--physical-gpu", str(args.gpu)]
    try:
        wait_for_idle(args.gpu, root)
        with (root / "smoke.log").open("w") as log:
            subprocess.run(command + ["--data", args.smoke_data, "--out", str(root / "smoke"),
                           "--protocol", "validation", "--epochs", "2", "--max-steps", "2",
                           "--limit-val", "32", "--no-final-test"],
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        checked = validate_smoke(root / "smoke")
        wait_for_idle(args.gpu, root)
        write_json(root / "queue.json", {"state": "production_starting", "at": timestamp(),
                   "expected_initial_model_sha256": checked["initial_model_sha256"]})
        with (root / "production.log").open("w") as log:
            subprocess.run(command + ["--data", args.production_data, "--out", str(root / "main"),
                           "--protocol", "antao_testsel", "--expected-initial-sha256", checked["initial_model_sha256"]],
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        manifest = json.loads((root / "main" / "manifest.json").read_text())
        if manifest["initial_model_sha256"] != checked["initial_model_sha256"]:
            raise AssertionError("Production did not initialize freshly with the checked seed")
        write_json(root / "queue.json", {"state": "complete", "at": timestamp()})
    except Exception as exc:
        write_json(root / "queue.json", {"state": "failed", "error": repr(exc), "at": timestamp()})
        raise


if __name__ == "__main__":
    main()
