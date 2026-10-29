"""Independent synthetic CPU acceptance of the installed Pseg3 v3 trainer.

This is an entry-point/serialization/observation check, not a dataset experiment
or acceptance of a committed production source identity. It creates a new run
directory and never reads real point clouds, connects to a server, or uses CUDA.
"""

import ast
import contextlib
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback

# Establish a CPU-only process before importing Torch or any training helpers.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.dont_write_bytecode = True

import h5py
import numpy as np
import torch


def find_root():
    for parent in Path(__file__).resolve().parents:
        if (parent / "classification_ModelNet40").is_dir():
            return parent
    raise RuntimeError("Cannot locate authoritative HyCoRe working copy")


ROOT = find_root()
sys.path.insert(0, str(ROOT))

from segmentation_ShapeNetPart.hycore_seg_v1.metrics import PART_COUNTS, PART_STARTS
from segmentation_ShapeNetPart.hycore_seg_v1.utils import sha256, state_hash, write_json
from segmentation_ShapeNetPart.hycore_seg_v3 import train as trainer
from segmentation_ShapeNetPart.hycore_seg_v3.losses import HyCoReSegLoss


STAGING_IDENTITY = {
    "commit": "SYNTHETIC_CPU_STAGING_UNCOMMITTED_NOT_PRODUCTION",
    "branch": "synthetic_cpu_entrypoint_acceptance_only",
    "source_status": "uncommitted_v3_under_review",
    "synthetic_identity_override": True,
    "scope": "Local synthetic CPU test only; not a committed training source identity",
}


def fresh_output():
    directory = ROOT / "docs/research/tmp"
    for suffix in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        candidate = directory / ("pseg3_v3_cpu_20261029_" + suffix)
        try:
            candidate.mkdir(parents=True, exist_ok=False)
            return candidate
        except FileExistsError:
            continue
    raise FileExistsError("All reserved fresh CPU acceptance suffixes already exist")


def write_synthetic_data(directory):
    directory.mkdir(exist_ok=False)
    manifest = {
        "complete": True,
        "source": {"kind": "synthetic_cpu_contract_fixture", "real_dataset": False,
                   "seed": 91817, "point_cloud_source": "independent NumPy Gaussian fixture"},
        "split_policy": "synthetic_train8_val8_disjoint_IDs_no_test_created",
        "splits": {},
    }
    random = np.random.RandomState(91817)
    for split in ("train", "val"):
        categories = np.tile(np.arange(4, dtype=np.int64), 2)
        points = random.normal(0., .3, (8, 32, 3)).astype(np.float32)
        points -= points.mean(axis=1, keepdims=True)
        points /= np.linalg.norm(points, axis=-1).max(axis=1)[:, None, None]
        labels = np.empty((8, 32), dtype=np.int64)
        for row, category in enumerate(categories):
            labels[row] = PART_STARTS[int(category)] + np.arange(32) % PART_COUNTS[int(category)]
        ids = [f"synthetic_cpu_only/{split}/{row:05d}" for row in range(8)]
        path = directory / (split + ".h5")
        with h5py.File(path, "w") as stream:
            stream.create_dataset("data", data=points)
            stream.create_dataset("label", data=categories[:, None])
            stream.create_dataset("pid", data=labels)
            stream.create_dataset("ids", data=np.asarray(ids, dtype=object),
                                  dtype=h5py.string_dtype(encoding="utf-8"))
        manifest["splits"][split] = {"objects": 8, "points_per_object": 32,
                                             "file_sha256": sha256(path)}
    write_json(directory / "manifest.json", manifest)
    assert not (directory / "test.h5").exists()
    return manifest


def source_hashes():
    package = ROOT / "segmentation_ShapeNetPart/hycore_seg_v3"
    return {name: sha256(package / name)
            for name in ("model.py", "losses.py", "monitoring.py", "train.py", "test_contracts.py")}


def json_lines(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def verify_arm(out, arm):
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    acceptance = json.loads((out / "acceptance.json").read_text(encoding="utf-8"))
    status = json.loads((out / "status.json").read_text(encoding="utf-8"))
    metrics = json_lines(out / "metrics.jsonl")
    geometry = json_lines(out / "geometry.jsonl")
    steps = json_lines(out / "steps.jsonl")
    checkpoints = {name: torch.load(out / (name + ".pth"), map_location="cpu", weights_only=False)
                   for name in ("e0", "last", "best")}
    initial, last, best = (checkpoints[name] for name in ("e0", "last", "best"))
    config = manifest["config"]
    require(status["state"] == "complete" and acceptance["complete"], "Entry point did not complete")
    require(config["device"] == "cpu" and config["allow_dirty_cpu"], "CPU staging flag missing")
    require(config["protocol"] == "validation" and config["no_final_test"], "Synthetic check must not use test")
    require(config["lr"] == .1 and config["min_lr"] == .001, "Approved learning rates changed")
    require(config["schedule_epochs"] == 200 and config["epochs"] == 2, "Short check changed 200-epoch schedule")
    require(config["num_points"] == 32 and config["k"] == 4 and config["batch_size"] == 4,
            "Unexpected CPU fixture dimensions")
    require(manifest["source"] == STAGING_IDENTITY, "Synthetic identity must be explicit")
    require(manifest["gpu"] is None and manifest["gradient_clipping"] is False, "Unexpected GPU/clip configuration")
    require(manifest["train_objects"] == 8 and manifest["selection_objects"] == 8,
            "Synthetic split not fully consumed")
    require(manifest["train_selection_overlap_ids"] == 0, "Synthetic IDs overlap")
    require(acceptance["epochs"] == 2 and acceptance["updates"] == 4, "Not exactly two epochs by two updates")
    require(acceptance["final_test_evaluations"] == 0, "Test was evaluated")
    require(not (out / "final_test.json").exists() and not (out / "test_objects.json").exists(), "Test artifact exists")
    require(initial["epoch"] == 0 and initial["updates"] == 0, "Missing true e0")
    require(last["epoch"] == 2 and last["updates"] == 4, "Last checkpoint incomplete")
    require(best["epoch"] in (1, 2), "Best checkpoint missing")
    require(last["scheduler"]["T_max"] == 200 and last["scheduler"]["eta_min"] == .001,
            "Scheduler state lost approved horizon")
    require(last["scheduler"]["last_epoch"] == 2, "Scheduler did not advance exactly twice")
    require(initial["optimizer"]["param_groups"][0]["lr"] == .1, "e0 optimizer did not start at .1")
    require(last["optimizer"]["param_groups"][0]["lr"] > .099,
            "Two-epoch smoke accidentally annealed to the production minimum")
    require(trainer.finite_tensor_state(last["model"]) and trainer.finite_tensor_state(last["optimizer"]),
            "Nonfinite saved state")
    counters = {name: int(value) for name, value in last["model"].items()
                if name.endswith("num_batches_tracked")}
    require(counters and set(counters.values()) == {4}, "BN did not update exactly four times")
    require(set(metrics_row["epoch"] for metrics_row in metrics) == {1, 2}, "Missing epoch metrics")
    require(all(row["steps"] == 2 and row["objects_seen"] == 8 and row["val"]["objects"] == 8
                for row in metrics), "Incomplete train/validation traversal")
    require(all(not row["gradient_observation"]["clip_applied"] for row in metrics), "Reported applied clipping")
    require(all(row["selection_split"] == "val" for row in metrics), "Wrong selection split")
    require(len(geometry) == 4 and len(steps) == 4, "All four smoke updates must be logged")
    expected_nodes = {"h", "point_exp", "m2", "m3", "whole_tangent", "part_tangent", "whole_exp", "part_exp"}
    if arm != "b":
        expected_nodes.add("m1")
    for observation in geometry:
        require(set(observation["geometry"]) == expected_nodes, "Geometry node missing or wrong M1 topology")
        require(set(observation["gradients"]) == {"ce", "contr", "hier"}, "Loss derivatives not separately logged")
        require(not observation["clip_applied"], "Geometry log reports clipping")
        require(np.isfinite(observation["gradient_norm"]), "Nonfinite gradient norm")
        diagnostics = observation["diagnostics"]
        require(diagnostics["valid_negative"] == 4 and diagnostics["negative_legal_rate"] == 1.,
                "Strict negative legality failed")
        require(diagnostics["same_class_pair_fraction"] == 0. and diagnostics["self_pair_fraction"] == 0.,
                "Same-class or same-record negative selected")
        require(4 <= diagnostics["part_points"] <= 8 and len(diagnostics["part_centers"]) == 4,
                "Part RNG identity missing")
        for loss_name in ("contr", "hier"):
            for node in expected_nodes & {"point_exp", "m1", "m2", "m3"}:
                gradient = observation["gradients"][loss_name]["nodes"][node]
                require(gradient["status"] == "none" and gradient["l2"] is None,
                        "Unused head derivative must be logged as None, not zero")
        for node in ("point_exp", "whole_exp", "part_exp"):
            require("analytic_projection_rate" in observation["geometry"][node],
                    "EXP analytic/preproject evidence missing")
            require("near_boundary_fraction" in observation["geometry"][node], "Postproject boundary statistic missing")
        require("point_norm" in observation["geometry"]["h"] and
                "object_mean_norm" in observation["geometry"]["h"], "Point vs pooled feature norms missing")
        require(set(observation["parameter_updates"]) == ({"shared", "m1", "m2", "m3"}
                                                         if arm != "b" else {"shared", "m2", "m3"}),
                "Actual update observations missing")
    expected_loss = HyCoReSegLoss(seed=4780)
    require(torch.equal(initial["negative_generator"], expected_loss.negative_generator.get_state()),
            "Saved e0 negative RNG has wrong seed")
    require(torch.equal(initial["part_generator"], expected_loss.generator.get_state()),
            "Saved e0 part RNG has wrong seed")
    require(not torch.equal(initial["negative_generator"], last["negative_generator"]),
            "Negative RNG did not advance")
    require(not torch.equal(initial["part_generator"], last["part_generator"]), "Part RNG did not advance")
    for name in ("rng", "train_generator", "eval_generator", "part_generator", "negative_generator"):
        require(name in last, "Checkpoint RNG missing: " + name)
    fresh = torch.Generator(device="cpu")
    fresh.set_state(last["negative_generator"])
    require(torch.equal(fresh.get_state(), last["negative_generator"]), "Negative RNG cannot be restored")
    return {
        "arm": arm, "complete": True, "epochs": 2, "updates": 4, "bn_counter_values": sorted(set(counters.values())),
        "scheduler_T_max": last["scheduler"]["T_max"], "scheduler_last_epoch": last["scheduler"]["last_epoch"],
        "initial_lr": initial["optimizer"]["param_groups"][0]["lr"],
        "last_lr": last["optimizer"]["param_groups"][0]["lr"], "no_gradient_clipping": True,
        "max_observed_gradient_norm": max(row["gradient_norm"] for row in geometry),
        "old_clip_threshold_exceeded_updates": sum(row["gradient_norm"] > 1 for row in geometry),
        "geometry_and_gradient_observations": len(geometry), "geometry_nodes": sorted(expected_nodes),
        "final_test_evaluations": 0, "negative_rng_saved_and_restorable": True,
        "initial_backbone_sha256": manifest["initial_backbone_sha256"],
        "initial_shared_head_sha256": manifest["initial_shared_head_sha256"],
        "initial_model_sha256": manifest["initial_model_sha256"],
        "initial_m1_sha256": (state_hash({name: value for name, value in initial["model"].items()
                                         if name.startswith("emb.")}) if arm != "b" else None),
        "first_input_sha256_by_epoch": [row["first_input_sha256"] for row in metrics],
        "first_pairing_by_epoch": [row["first_pairing"] for row in metrics],
        "pairing_by_update": [{key: row["diagnostics"][key] for key in
                               ("negative_indices", "part_centers", "part_points", "part_indices_sha256")}
                              for row in geometry],
        "negative_rng_e0_sha256": hashlib.sha256(initial["negative_generator"].numpy().tobytes()).hexdigest(),
        "negative_rng_last_sha256": hashlib.sha256(last["negative_generator"].numpy().tobytes()).hexdigest(),
        "part_rng_last_sha256": hashlib.sha256(last["part_generator"].numpy().tobytes()).hexdigest(),
        "e0_sha256": sha256(out / "e0.pth"), "last_sha256": sha256(out / "last.pth"),
        "best_sha256": sha256(out / "best.pth"), "trainable_parameters": manifest["trainable_parameters"],
    }


def main():
    output = fresh_output()
    try:
        source_before = source_hashes()
        manifest = write_synthetic_data(output / "data")
        trainer.git_identity = lambda: dict(STAGING_IDENTITY)
        results = []
        for arm in ("a", "b", "c"):
            out = output / arm
            out.mkdir(exist_ok=False)
            args = trainer.arguments([
                "--data", str(output / "data"), "--out", str(out), "--arm", arm,
                "--protocol", "validation", "--epochs", "2", "--schedule-epochs", "200",
                "--batch-size", "4", "--eval-batch-size", "4", "--num-points", "32", "--k", "4",
                "--lr", ".1", "--min-lr", ".001", "--seed", "4780", "--workers", "0", "--cpu-threads", "2",
                "--device", "cpu", "--part-min", "4", "--part-max", "8", "--max-steps", "2",
                "--monitor-every", "1", "--gradient-every", "1", "--no-final-test", "--allow-dirty-cpu",
            ])
            with (output / (arm + ".log")).open("w", encoding="utf-8") as log:
                with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                    trainer.train(args)
            results.append(verify_arm(out, arm))
            print(json.dumps({"arm": arm, "cpu_entrypoint_passed": True, "output": str(out)}), flush=True)
        require(source_hashes() == source_before, "Installed source changed during acceptance")
        a, b, c = results
        for field in ("initial_backbone_sha256", "initial_shared_head_sha256", "negative_rng_e0_sha256",
                      "negative_rng_last_sha256", "part_rng_last_sha256", "first_input_sha256_by_epoch"):
            require(a[field] == b[field] == c[field], "Three-arm initialization/input/RNG mismatch: " + field)
        require(a["initial_model_sha256"] == c["initial_model_sha256"], "A/C complete e0 state differs")
        require(a["initial_m1_sha256"] == c["initial_m1_sha256"], "A/C M1 differs")
        require(a["trainable_parameters"] - b["trainable_parameters"] == 65792,
                "B did not remove exactly M1's 65792 parameters")
        for index in range(4):
            for field in ("negative_indices", "part_centers", "part_points"):
                require(a["pairing_by_update"][index][field] == b["pairing_by_update"][index][field] ==
                        c["pairing_by_update"][index][field], "Three-arm pairing RNG mismatch: " + field)
        require(not torch.cuda.is_initialized(), "CPU acceptance initialized CUDA")
        tree = ast.parse((ROOT / "segmentation_ShapeNetPart/hycore_seg_v3/train.py").read_text(encoding="utf-8"))
        forbidden = {"clip_grad_norm_", "clip_grad_value_", "clip_grad_norm", "clip_grad_value"}
        called = {node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
                  for node in ast.walk(tree) if isinstance(node, ast.Call) and
                  isinstance(node.func, (ast.Attribute, ast.Name))}
        require(not forbidden & called, "Trainer calls gradient clipping")
        report = {
            "complete": True, "scope": "synthetic_CPU_full_entrypoint_only",
            "production_commit_acceptance": False, "synthetic_git_identity": STAGING_IDENTITY,
            "cuda_initialized": False, "synthetic_data_manifest": manifest,
            "source_sha256": source_before, "helper_sha256": sha256(Path(__file__)),
            "three_arm_initialization_inputs_and_rng_matched": True,
            "feature_neighbor_membership_equal_claimed": False,
            "feature_neighbor_note": "Part centers and N match; feature-kNN membership may differ naturally across arms.",
            "arms": results,
        }
        write_json(output / "checks.json", report)
        print(json.dumps({"complete": True, "output": str(output), "cuda_initialized": False,
                          "arms": [{key: row[key] for key in ("arm", "updates", "scheduler_T_max", "max_observed_gradient_norm")}
                                   for row in results]}), flush=True)
    except Exception as exc:
        write_json(output / "failure.json", {"error": repr(exc), "traceback": traceback.format_exc(),
                                             "synthetic_git_identity": STAGING_IDENTITY})
        print(json.dumps({"complete": False, "output": str(output), "error": repr(exc)}), flush=True)
        raise


if __name__ == "__main__":
    main()
