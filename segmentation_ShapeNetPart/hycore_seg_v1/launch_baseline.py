"""Detached baseline-only orchestration: idle GPU -> fresh smoke -> fresh200."""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch

from .train import ensure_idle
from .utils import git_identity, timestamp, write_json


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
    state = torch.load(smoke / "last.pth", map_location="cpu", weights_only=False)
    checks = {
        "two_epochs_two_steps": len(metrics) == 2 and all(row["steps"] == 2 for row in metrics),
        "four_updates": accepted["updates"] == state["updates"] == 4,
        "no_test": accepted["final_test_evaluations"] == 0 and not (smoke / "final_test.json").exists(),
        "full_point_count_batch": manifest["config"]["num_points"] == 2048 and manifest["config"]["batch_size"] == 32,
        "bn_once_per_step": all(n == 4 for n in metrics[-1]["bn_counts"].values()),
        "finite_parameters": all(torch.isfinite(v).all().item() for v in state["model"].values()),
        "finite_optimizer": all(torch.isfinite(v).all().item() for x in state["optimizer"]["state"].values()
                                for v in x.values() if torch.is_tensor(v)),
        "scheduler_keeps_200": state["scheduler"]["T_max"] == 200 and state["scheduler"]["last_epoch"] == 2,
        "full_state": all(k in state for k in ("rng", "train_generator", "eval_generator", "config", "source")),
        "baseline_has_no_regularizer": manifest["regularizer_constructed"] is False,
    }
    write_json(smoke / "smoke_validation.json", {"checks": checks, "passed": all(checks.values())})
    if not all(checks.values()):
        raise RuntimeError("Smoke acceptance failed: " + str(checks))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--gpu", required=True, type=int)
    args = parser.parse_args()
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=False)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["OMP_NUM_THREADS"] = "4"
    os.environ["MKL_NUM_THREADS"] = "4"
    identity = git_identity()
    write_json(root / "dispatch.json", {"started": timestamp(), "source": identity,
               "gpu": args.gpu, "pid": os.getpid(), "scope": "baseline_only_fresh_smoke_then_fresh200",
               "data": args.data, "python": sys.executable})
    config = Path(__file__).parent / "configs" / "dgcnn_baseline.json"
    command = [sys.executable, "-m", "segmentation_ShapeNetPart.hycore_seg_v1.train",
               "--config", str(config), "--data", args.data, "--physical-gpu", str(args.gpu)]
    try:
        wait_for_idle(args.gpu, root)
        smoke = root / "smoke"
        with (root / "smoke.log").open("w") as log:
            subprocess.run(command + ["--out", str(smoke), "--epochs", "2", "--max-steps", "2",
                           "--limit-val", "32", "--no-final-test"],
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        validate_smoke(smoke)
        wait_for_idle(args.gpu, root)
        write_json(root / "queue.json", {"state": "production_starting", "at": timestamp()})
        with (root / "production.log").open("w") as log:
            subprocess.run(command + ["--out", str(root / "main")],
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        write_json(root / "queue.json", {"state": "complete", "at": timestamp()})
    except Exception as exc:
        write_json(root / "queue.json", {"state": "failed", "error": repr(exc), "at": timestamp()})
        raise


if __name__ == "__main__":
    main()
