"""Synthetic CPU acceptance of the installed paired D/E v5 training entry points.

No real data, checkpoint, test split, CUDA context, or production source identity
is used.  Every invocation creates a new local fixture/run directory.  This
checks the training loop and saved observations, not real-data GPU acceptance.
"""

import ast
import contextlib
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import traceback

os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.dont_write_bytecode = True

import h5py
import numpy as np
import torch


def find_root():
    for parent in Path(__file__).resolve().parents:
        if (parent / "classification_ModelNet40").is_dir():
            return parent
    raise RuntimeError("Cannot locate the authoritative HyCoRe working copy")


ROOT = find_root()
sys.path.insert(0, str(ROOT))

from segmentation_ShapeNetPart.hycore_seg_v1.metrics import PART_COUNTS, PART_STARTS
from segmentation_ShapeNetPart.hycore_seg_v1.utils import (
    restore_rng, rng_state, sha256, state_hash, write_json,
)
from segmentation_ShapeNetPart.hycore_seg_v5.losses import HyCoReSegLoss
from segmentation_ShapeNetPart.hycore_seg_v5 import train as trainer


STAGING_IDENTITY = {
    "commit": "SYNTHETIC_CPU_STAGING_UNCOMMITTED_NOT_PRODUCTION",
    "branch": "synthetic_cpu_entrypoint_acceptance_only",
    "source_status": "v5_under_cpu_acceptance_not_production_identity",
    "synthetic_identity_override": True,
    "scope": "Local synthetic CPU test only; not a committed training source identity",
}


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def fresh_output():
    parent = ROOT / "docs/research/tmp"
    for suffix in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        candidate = parent / ("pseg3_v5_de_cpu_20261029_" + suffix)
        try:
            candidate.mkdir(parents=True, exist_ok=False)
            return candidate
        except FileExistsError:
            continue
    raise FileExistsError("All reserved fresh v5 CPU acceptance suffixes already exist")


def write_fixture(directory):
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
    require(not (directory / "test.h5").exists(), "Fixture must have no test split")
    return manifest


def source_hashes():
    source = ROOT / "segmentation_ShapeNetPart"
    paths = ["hycore_seg_v5/" + name for name in
             ("model.py", "losses.py", "train.py", "launch_experiment.py", "test_contracts.py")]
    paths += ["hycore_seg_v3/losses.py", "hycore_seg_v3/monitoring.py"]
    return {name: sha256(source / name) for name in paths}


def json_lines(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def verify_run(out, arm):
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    accepted = json.loads((out / "acceptance.json").read_text(encoding="utf-8"))
    status = json.loads((out / "status.json").read_text(encoding="utf-8"))
    metrics, observations, steps = (json_lines(out / name) for name in
                                    ("metrics.jsonl", "geometry.jsonl", "steps.jsonl"))
    initial, last, best = (torch.load(out / (name + ".pth"), map_location="cpu", weights_only=False)
                           for name in ("e0", "last", "best"))
    config = manifest["config"]
    require(status["state"] == "complete" and accepted["complete"], "Entry point did not complete")
    require(manifest["format"] == "hycore_seg_v5" and last["format"] == "hycore_seg_v5_complete",
            "Saved identity is not the versioned v5 successor")
    require(config["arm"] == arm and manifest["arm"] == arm, "Wrong experiment arm")
    require(manifest["dropout_active"] is True and manifest["has_m1"] is False,
            "Both arms must retain dp1 and omit the equal-width M")
    require(manifest["euclidean_conv9_retained"] is True and
            manifest["feature_location"] == "after_conv9_bn9_leaky" and
            manifest["dp1_location"] == "before_conv9", "Wrong Euclidean block or regularizer location")
    require(config["device"] == "cpu" and config["allow_dirty_cpu"], "CPU staging flag missing")
    require(config["protocol"] == "validation" and config["no_final_test"], "Fixture must not use test")
    require(config["lr"] == .1 and config["min_lr"] == .001, "Approved learning rates changed")
    require((config["alpha"], config["beta"]) == ((.01, .01) if arm == "d" else (0., 0.)),
            "Arm regularizer coefficients differ from approved paired control")
    require(manifest["regularizers_active"] is (arm == "d") and
            manifest["regularizer_diagnostics_only"] is (arm == "e"), "Ambiguous active versus observational loss identity")
    require(config["epochs"] == 2 and config["schedule_epochs"] == 200,
            "Bounded check changed the production scheduler horizon")
    require(config["num_points"] == 32 and config["k"] == 4 and config["batch_size"] == 4,
            "Unexpected CPU fixture dimensions")
    require(manifest["source"] == STAGING_IDENTITY, "Fixture source identity is not explicit")
    require(manifest["gpu"] is None and manifest["gradient_clipping"] is False,
            "Unexpected GPU or clipping configuration")
    require(manifest["train_objects"] == manifest["selection_objects"] == 8 and
            manifest["train_selection_overlap_ids"] == 0, "Synthetic split traversal or identity is wrong")
    require(accepted["epochs"] == 2 and accepted["updates"] == 4, "Not exactly four updates")
    require(accepted["final_test_evaluations"] == 0 and not (out / "final_test.json").exists() and
            not (out / "test_objects.json").exists(), "Official test was accessed")
    require(initial["epoch"] == initial["updates"] == 0 and last["epoch"] == 2 and last["updates"] == 4,
            "Incomplete initial/final checkpoint")
    require(best["epoch"] in (1, 2), "Best checkpoint missing")
    require(last["scheduler"]["T_max"] == 200 and last["scheduler"]["eta_min"] == .001 and
            last["scheduler"]["last_epoch"] == 2, "Scheduler checkpoint state is wrong")
    require(initial["optimizer"]["param_groups"][0]["lr"] == .1 and
            last["optimizer"]["param_groups"][0]["lr"] > .099, "Smoke accidentally annealed to minimum")
    require(trainer.finite_tensor_state(last["model"]) and trainer.finite_tensor_state(last["optimizer"]),
            "Nonfinite saved state")
    require(state_hash(initial["model"]) == manifest["initial_model_sha256"] ==
            initial["initial_model_sha256"] == last["initial_model_sha256"], "e0 provenance changed")
    counters = {name: int(value) for name, value in last["model"].items()
                if name.endswith("num_batches_tracked")}
    require(counters and set(counters.values()) == {4} and "bn9.num_batches_tracked" in counters,
            "All BN, including restored BN9, must update exactly four times")
    require([row["epoch"] for row in metrics] == [1, 2] and all(
        row["steps"] == 2 and row["objects_seen"] == row["val"]["objects"] == 8 and
        row["selection_split"] == "val" and not row["gradient_observation"]["clip_applied"]
        for row in metrics), "Epoch traversal, selection, or clipping record is wrong")
    require(len(observations) == len(steps) == 4, "All four fixture updates must be observed")
    expected_nodes = {"pre_conv9", "h", "point_exp", "m2", "m3", "whole_tangent",
                      "part_tangent", "whole_exp", "part_exp"}
    expected_groups = {"shared", "euclidean9", "m2", "m3"}
    for row in observations:
        require(set(row["geometry"]) == expected_nodes, "Missing new H or pre-Conv9 observation")
        require(set(row["gradients"]) == {"ce", "contr", "hier"}, "Missing separate loss gradients")
        require(set(row["parameter_updates"]) == expected_groups, "Missing Euclidean9 parameter updates")
        require(not row["clip_applied"] and np.isfinite(row["gradient_norm"]), "Invalid unclipped gradient")
        diagnostics = row["diagnostics"]
        require(diagnostics["regularizers_active"] is (arm == "d") and
                diagnostics["regularizer_diagnostics_only"] is (arm == "e"),
                "Per-step active versus observational diagnostics identity missing")
        if arm == "e":
            require(diagnostics["active_weighted_contr"] == diagnostics["active_weighted_hier"] == 0.,
                    "E has a nonzero active regularizer contribution")
        require(diagnostics["valid_negative"] == 4 and diagnostics["negative_legal_rate"] == 1. and
                diagnostics["same_class_pair_fraction"] == diagnostics["self_pair_fraction"] == 0.,
                "Strict batch-negative legality failed")
        require(4 <= diagnostics["part_points"] <= 8 and len(diagnostics["part_centers"]) == 4,
                "Missing bounded part sampling identity")
        for loss_name in ("ce", "contr", "hier"):
            info = row["gradients"][loss_name]
            require(set(info["parameter_groups"]) == expected_groups, "Gradient groups differ from updates")
            active = loss_name == "ce" or arm == "d"
            for node in ("pre_conv9", "h"):
                require((info["nodes"][node]["status"] != "none") is active,
                        "Active versus disabled gradient differs at Euclidean feature path")
            require((info["parameter_groups"]["euclidean9"]["status"] != "none") is active,
                    "Active versus disabled gradient differs at restored Conv9/BN9")
            if not active:
                require(info["loss"] == 0. and info["loss_requires_grad"] is False,
                        "Disabled E regularizer acquired an active graph or nonzero loss")
                require(all(node["status"] == "none" for node in info["nodes"].values()),
                        "Disabled E regularizer produces a feature gradient")
                require(all(group["status"] == "none" for group in info["parameter_groups"].values()),
                        "Disabled E regularizer produces a parameter gradient")
        for loss_name in ("contr", "hier"):
            for node in ("point_exp", "m2", "m3"):
                require(row["gradients"][loss_name]["nodes"][node]["status"] == "none",
                        "Feature regularizer unexpectedly uses the M head")
            for group in ("m2", "m3"):
                info = row["gradients"][loss_name]["parameter_groups"][group]
                require(info["status"] == "none" and info["none_count"] == info["parameter_tensors"],
                        "Regularizer unexpectedly updates M parameters")
        # Rhier can be saturated; graph connectivity is checked above, not nonzero magnitude.
        for node in ("point_exp", "whole_exp", "part_exp"):
            require("analytic_projection_rate" in row["geometry"][node] and
                    "near_boundary_fraction" in row["geometry"][node], "EXP projection diagnostics missing")
        require("point_norm" in row["geometry"]["h"] and "object_mean_norm" in row["geometry"]["h"],
                "Point and pooled feature summaries missing")
    if arm == "e":
        require(all(row["loss"] == row["ce"] and row["reg"] == 0. for row in steps),
                "E actual training loss must equal CE exactly at every recorded update")
        require(all(row["train_loss"] == row["train_ce"] and row["train_reg"] == 0. for row in metrics),
                "E epoch loss must remain CE-only")
    criterion = HyCoReSegLoss(.01, .01, seed=4780)
    for key, generator in (("part_generator", criterion.generator),
                           ("negative_generator", criterion.negative_generator)):
        require(torch.equal(initial[key], generator.get_state()), "Wrong e0 independent RNG: " + key)
        require(not torch.equal(initial[key], last[key]), "Independent RNG did not advance: " + key)
    for name in ("rng", "train_generator", "eval_generator", "part_generator", "negative_generator"):
        require(name in last and name in initial, "Missing checkpoint RNG: " + name)
    for name in ("train_generator", "eval_generator", "part_generator", "negative_generator"):
        restored = torch.Generator(device="cpu")
        restored.set_state(last[name])
        require(torch.equal(restored.get_state(), last[name]), "Saved generator cannot be restored: " + name)
    require(not initial["rng"]["cuda"] and not last["rng"]["cuda"], "CPU checkpoint contains CUDA RNG")
    current_rng = rng_state()
    try:
        restore_rng(last["rng"])
        first_draws = (random.random(), np.random.random(), torch.rand(4))
        restore_rng(last["rng"])
        second_draws = (random.random(), np.random.random(), torch.rand(4))
        require(first_draws[:2] == second_draws[:2] and torch.equal(first_draws[2], second_draws[2]),
                "Python/NumPy/Torch global RNG snapshot cannot be restored")
        restored_model = trainer.NativeTail2(arm, k=config["k"], dropout=config["dropout"])
        restored_model.load_state_dict(last["model"], strict=True)
        require(state_hash(restored_model.state_dict()) == state_hash(last["model"]),
                "Saved model state cannot be loaded into the paired architecture")
        restored_optimizer = trainer.geoopt.optim.RiemannianSGD(
            restored_model.parameters(), lr=config["lr"], momentum=config["momentum"],
            weight_decay=config["weight_decay"])
        restored_optimizer.load_state_dict(last["optimizer"])
        require(trainer.finite_tensor_state(restored_optimizer.state_dict()) and
                len(restored_optimizer.state) == len(last["optimizer"]["state"]),
                "Saved optimizer state cannot be restored")
    finally:
        restore_rng(current_rng)
    return {
        "arm": arm, "complete": True, "epochs": 2, "updates": 4,
        "bn_counter_values": sorted(set(counters.values())), "restored_bn9_observed": True,
        "scheduler_T_max": last["scheduler"]["T_max"], "scheduler_last_epoch": last["scheduler"]["last_epoch"],
        "initial_lr": initial["optimizer"]["param_groups"][0]["lr"],
        "last_lr": last["optimizer"]["param_groups"][0]["lr"], "no_gradient_clipping": True,
        "max_observed_gradient_norm": max(row["gradient_norm"] for row in observations),
        "geometry_and_gradient_observations": len(observations), "geometry_nodes": sorted(expected_nodes),
        "parameter_groups": sorted(expected_groups), "regularizers_reach_euclidean9_not_M_head": arm == "d",
        "regularizers_active": arm == "d", "regularizer_diagnostics_only": arm == "e",
        "active_regularizer_zero_ce_only": arm == "e",
        "final_test_evaluations": 0, "independent_RNGs_saved_and_restorable": True,
        "global_RNG_saved_and_restorable": True, "model_and_optimizer_state_restorable": True,
        "initial_backbone_sha256": manifest["initial_backbone_sha256"],
        "initial_shared_head_sha256": manifest["initial_shared_head_sha256"],
        "initial_model_sha256": manifest["initial_model_sha256"],
        "first_input_sha256_by_epoch": [row["first_input_sha256"] for row in metrics],
        "e0_sha256": sha256(out / "e0.pth"), "last_sha256": sha256(out / "last.pth"),
        "best_sha256": sha256(out / "best.pth"), "trainable_parameters": manifest["trainable_parameters"],
        "negative_rng_e0_sha256": hashlib.sha256(initial["negative_generator"].numpy().tobytes()).hexdigest(),
    }


def verify_pair(output, results):
    initial = {arm: torch.load(output / arm / "e0.pth", map_location="cpu", weights_only=False)
               for arm in ("d", "e")}
    final = {arm: torch.load(output / arm / "last.pth", map_location="cpu", weights_only=False)
             for arm in ("d", "e")}
    metrics = {arm: json_lines(output / arm / "metrics.jsonl") for arm in ("d", "e")}
    require(results["d"]["initial_model_sha256"] == results["e"]["initial_model_sha256"],
            "D/E initial complete state including BN differs")
    require(results["d"]["initial_backbone_sha256"] == results["e"]["initial_backbone_sha256"] and
            results["d"]["initial_shared_head_sha256"] == results["e"]["initial_shared_head_sha256"],
            "D/E initial backbone or head differs")
    require(results["d"]["first_input_sha256_by_epoch"] == results["e"]["first_input_sha256_by_epoch"],
            "D/E first shuffled augmented input differs between paired epochs")
    for checkpoint in (initial, final):
        for key in ("train_generator", "eval_generator", "part_generator", "negative_generator"):
            require(torch.equal(checkpoint["d"][key], checkpoint["e"][key]),
                    "D/E independent RNG progression differs: " + key)
        require(torch.equal(checkpoint["d"]["rng"]["torch"], checkpoint["e"]["rng"]["torch"]),
                "D/E global Torch dropout RNG progression differs")
        require(checkpoint["d"]["rng"]["python"] == checkpoint["e"]["rng"]["python"],
                "D/E Python global RNG progression differs")
        dn, en = checkpoint["d"]["rng"]["numpy"], checkpoint["e"]["rng"]["numpy"]
        require(dn[0] == en[0] and np.array_equal(dn[1], en[1]) and dn[2:] == en[2:],
                "D/E NumPy augmentation RNG progression differs")
    for d, e in zip(metrics["d"], metrics["e"]):
        require(d["first_ids"] == e["first_ids"], "D/E first batch identities differ")
        for key in ("negative_indices", "part_centers", "part_points"):
            require(d["first_pairing"][key] == e["first_pairing"][key],
                    "D/E paired independent draw differs: " + key)
    require(metrics["d"][0]["first_pairing"] == metrics["e"][0]["first_pairing"],
            "D/E initial feature-part indices differ before first update")
    return {"exact_initial_state_including_bn": True, "paired_first_input_each_epoch": True,
            "paired_initial_part_indices": True, "paired_independent_RNG_progression": True,
            "paired_global_RNG_progression": True, "only_active_loss_coefficients_differ": True}


def main():
    output = fresh_output()
    original_identity = trainer.git_identity
    try:
        source_before = source_hashes()
        fixture = write_fixture(output / "data")
        trainer.git_identity = lambda: dict(STAGING_IDENTITY)
        results = {}
        for arm in ("d", "e"):
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
            results[arm] = verify_run(out, arm)
        pair = verify_pair(output, results)
        require(source_hashes() == source_before, "Source changed during CPU acceptance")
        tree = ast.parse((ROOT / "segmentation_ShapeNetPart/hycore_seg_v5/train.py").read_text(encoding="utf-8"))
        forbidden = {"clip_grad_norm_", "clip_grad_value_", "clip_grad_norm", "clip_grad_value"}
        calls = {n.func.attr if isinstance(n.func, ast.Attribute) else n.func.id for n in ast.walk(tree)
                 if isinstance(n, ast.Call) and isinstance(n.func, (ast.Attribute, ast.Name))}
        require(not forbidden & calls, "Trainer calls gradient clipping")
        require(not torch.cuda.is_initialized(), "CPU acceptance initialized CUDA")
        write_json(output / "checks.json", {
            "complete": True, "scope": "synthetic_CPU_full_entrypoint_only",
            "production_commit_acceptance": False, "synthetic_git_identity": STAGING_IDENTITY,
            "cuda_initialized": False, "synthetic_data_manifest": fixture,
            "source_sha256": source_before, "helper_sha256": sha256(Path(__file__)), "arms": results, "pair": pair,
        })
        print(json.dumps({"complete": True, "output": str(output), "cuda_initialized": False,
                          "arms": list(results), "updates_each": 4, "pair": pair,
                          "max_observed_gradient_norm": {arm: row["max_observed_gradient_norm"]
                                                         for arm, row in results.items()}}), flush=True)
    except Exception as exc:
        write_json(output / "failure.json", {"error": repr(exc), "traceback": traceback.format_exc(),
                                             "synthetic_git_identity": STAGING_IDENTITY})
        print(json.dumps({"complete": False, "output": str(output), "error": repr(exc)}), flush=True)
        raise
    finally:
        trainer.git_identity = original_identity


if __name__ == "__main__":
    main()
