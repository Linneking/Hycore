"""Exercise all three complete train entry points on synthetic CPU-only data.

This creates new temporary artifacts only. It never opens official data or uses
a GPU. Invoke after committing the v2 package so the source guard stays active.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import h5py
import numpy as np
import torch

from ..hycore_seg_v1.metrics import PART_COUNTS, PART_STARTS
from ..hycore_seg_v1.train import bn_counts
from ..hycore_seg_v1.utils import sha256, state_hash, write_json
from .model import NativePartSeg
from .train import finite_tensor_state


def prepare_data(root):
    root.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(4780)
    manifest = {"complete": True, "source": "synthetic CPU entrypoint contract, not ShapeNetPart",
                "split_policy": "unique synthetic IDs, zero split overlap", "splits": {}}
    prepared = {}
    for split, count in (("train", 12), ("val", 8), ("test", 8)):
        categories = np.arange(count, dtype=np.int64) % 16
        xyz = rng.normal(0., .1, (count, 32, 3)).astype(np.float32)
        parts = np.stack([rng.integers(PART_STARTS[c], PART_STARTS[c] + PART_COUNTS[c], 32)
                          for c in categories]).astype(np.int64)
        ids = [f"synthetic-{split}-{index:03d}" for index in range(count)]
        prepared[split] = xyz, categories, parts, ids
    prepared["trainval"] = tuple(np.concatenate([prepared["train"][i], prepared["val"][i]], axis=0)
                                  if i < 3 else prepared["train"][i] + prepared["val"][i]
                                  for i in range(4))
    for split, (xyz, categories, parts, ids) in prepared.items():
        path = root / (split + ".h5")
        with h5py.File(path, "w") as handle:
            handle.create_dataset("data", data=xyz)
            handle.create_dataset("label", data=categories[:, None])
            handle.create_dataset("pid", data=parts)
            handle.create_dataset("ids", data=ids, dtype=h5py.string_dtype("utf-8"))
        manifest["splits"][split] = {"objects": len(categories), "file_sha256": sha256(path)}
    write_json(root / "manifest.json", manifest)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    root = Path(args.out).resolve()
    root.mkdir(parents=True, exist_ok=False)
    data = root / "data"
    prepare_data(data)
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
    module = "segmentation_ShapeNetPart.hycore_seg_v2.train"
    common = ["--data", str(data), "--device", "cpu", "--protocol", "validation",
              "--batch-size", "4", "--eval-batch-size", "4", "--workers", "0", "--cpu-threads", "2",
              "--num-points", "32", "--k", "4", "--part-min", "4", "--part-max", "10",
              "--epochs", "2", "--schedule-epochs", "200", "--max-steps", "1", "--no-final-test"]
    results = {}
    for architecture, width in (("p0", 512), ("pglobal", 512), ("pseg3", 256)):
        out = root / architecture
        command = [sys.executable, "-m", module, "--architecture", architecture, "--out", str(out), *common]
        with (root / (architecture + ".log")).open("w", encoding="utf-8") as stream:
            subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True, env=environment)
        acceptance = json.loads((out / "acceptance.json").read_text())
        manifest = json.loads((out / "manifest.json").read_text())
        metrics = [json.loads(line) for line in (out / "metrics.jsonl").read_text().splitlines()]
        steps = [json.loads(line) for line in (out / "steps.jsonl").read_text().splitlines()]
        last = torch.load(out / "last.pth", map_location="cpu", weights_only=False)
        e0 = torch.load(out / "e0.pth", map_location="cpu", weights_only=False)
        torch.set_num_threads(2)
        model = NativePartSeg(architecture, k=4)
        model.load_state_dict(last["model"], strict=True)
        checks = {
            "two_updates": acceptance["updates"] == 2 and len(metrics) == 2,
            "no_test": acceptance["final_test_evaluations"] == 0 and not (out / "final_test.json").exists(),
            "bn_two": all(value == 2 for value in bn_counts(model).values()),
            "width": manifest["feature_dimension"] == width,
            "original_provenance": bool(manifest["source_provenance"]["files"]),
            "correct_category_input": manifest["network_category_input"] == (architecture == "pseg3"),
            "finite_model_optimizer": finite_tensor_state(last["model"]) and finite_tensor_state(last["optimizer"]),
            "scheduler_200": last["scheduler"]["T_max"] == 200,
            "complete_state": all(name in last for name in ("rng", "train_generator", "eval_generator",
                                                            "part_generator", "optimizer", "scheduler")),
            "initial_hash": state_hash(e0["model"]) == manifest["initial_model_sha256"],
            "head_no_regularizer_gradient": steps[0]["head_regularizer_gradient_is_none"],
            "both_branches_reach_encoder": min(steps[0]["reg_shared_gradient_norm"],
                                               steps[0]["ce_shared_gradient_norm"]) > 0,
            "margin": all(abs(row["gamma"] * row["part_points"] - 1000.) < 1e-8
                          and row["delta"] == 4. for row in steps),
            "clipping": all(row["gradient_norm_after_clip"] <= 1.00001 for row in steps),
            "no_self_pair_even_batch": all(row["self_pair_fraction"] == 0. for row in steps),
            "selection_complete": all(row["val"]["objects"] == 8 for row in metrics),
            "fresh_identity": manifest["initialization"] == "fresh" and manifest["teacher"] is None,
        }
        results[architecture] = {"passed": all(checks.values()), "checks": checks,
                                  "initial_sha256": manifest["initial_model_sha256"],
                                  "trainable_parameters": manifest["trainable_parameters"]}
        write_json(root / "checks.json", results)
        if not all(checks.values()):
            raise AssertionError((architecture, checks))
    print(json.dumps({"passed": True, "results": results}), flush=True)


if __name__ == "__main__":
    main()
