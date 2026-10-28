"""Explicitly selected protocol: clean-data smoke, then fresh benchmark run."""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch

from .launch_baseline import wait_for_idle
from .utils import git_identity, timestamp, write_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke-data", required=True)
    p.add_argument("--production-data", required=True)
    p.add_argument("--root", required=True)
    p.add_argument("--gpu", type=int, required=True)
    p.add_argument("--mode", choices=("baseline", "head", "hycore"), required=True)
    args = p.parse_args()
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=False)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["OMP_NUM_THREADS"] = "4"
    os.environ["MKL_NUM_THREADS"] = "4"
    config = Path(__file__).parent / "configs" / ("dgcnn_baseline.json" if args.mode == "baseline" else "hycore_seg.json")
    command = [sys.executable, "-m", "segmentation_ShapeNetPart.hycore_seg_v1.train", "--config", str(config),
               "--mode", args.mode, "--physical-gpu", str(args.gpu), "--workers", "8"]
    write_json(root / "dispatch.json", {"source": git_identity(), "started": timestamp(),
               "mode": args.mode, "protocol": "antao_testsel", "seed": 4780, "pid": os.getpid()})
    try:
        while not (Path(args.production_data) / "manifest.json").exists():
            write_json(root / "queue.json", {"state": "waiting_for_prepared_data", "at": timestamp()})
            time.sleep(30)
        wait_for_idle(args.gpu, root)
        with (root / "smoke.log").open("w") as log:
            subprocess.run(command + ["--data", args.smoke_data, "--out", str(root / "smoke"),
                           "--epochs", "2", "--max-steps", "2", "--limit-val", "32", "--no-final-test"],
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        state = torch.load(root / "smoke/last.pth", map_location="cpu", weights_only=False)
        ack = json.loads((root / "smoke/acceptance.json").read_text())
        records = [json.loads(x) for x in (root / "smoke/metrics.jsonl").read_text().splitlines()]
        steps = [json.loads(x) for x in (root / "smoke/steps.jsonl").read_text().splitlines()]
        checks = {"four_updates": ack["updates"] == 4, "no_test": ack["final_test_evaluations"] == 0,
                  "bn4": all(n == 4 for n in records[-1]["bn_counts"].values()),
                  "scheduler200": state["scheduler"]["T_max"] == 200,
                  "finite_state": all(torch.isfinite(v).all().item() for v in state["model"].values())}
        if args.mode == "hycore":
            checks["regularizer_reaches_backbone"] = steps[0]["reg_shared_gradient_norm"] > 0
            checks["margin1000_over_n"] = all(abs(x["gamma"] * x["part_points"] - 1000) < 1e-8 for x in steps)
            checks["valid_negatives"] = all(x["valid_negative"] == 32 for x in steps)
        write_json(root / "smoke_validation.json", {"passed": all(checks.values()), "checks": checks})
        if not all(checks.values()):
            raise RuntimeError("Smoke gate failed")
        wait_for_idle(args.gpu, root)
        write_json(root / "queue.json", {"state": "production_starting", "at": timestamp()})
        with (root / "production.log").open("w") as log:
            subprocess.run(command + ["--data", args.production_data, "--out", str(root / "main"),
                           "--protocol", "antao_testsel"], stdout=log, stderr=subprocess.STDOUT, check=True)
        write_json(root / "queue.json", {"state": "complete", "at": timestamp()})
    except Exception as exc:
        write_json(root / "queue.json", {"state": "failed", "error": repr(exc), "at": timestamp()})
        raise


if __name__ == "__main__":
    main()
