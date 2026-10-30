"""Matched v6 F/G/H arms: predecessor gate, idle GPU, smoke, then fresh 200 epochs.

Data and run locations are supplied by CLI; no server identity is embedded.
The launcher never resumes smoke weights or modifies an existing run directory.
"""

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

from segmentation_ShapeNetPart.hycore_seg_v1.train import ensure_idle
from segmentation_ShapeNetPart.hycore_seg_v1.utils import git_identity, state_hash, timestamp, write_json


TRAIN_MODULE = "segmentation_ShapeNetPart.hycore_seg_v6.train"


def approved_coefficients(arm):
    """All three approved arms use both regularizers."""
    if arm not in {"f", "g", "h"}:
        raise ValueError("The approved v6 arms are f, g and h")
    return (.01, .01)


def _optional_json(path):
    """An absent or partially replaced status file is not completion evidence."""
    try:
        return _json(path)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def predecessor_state(parent):
    """Read an existing run parent; never modify predecessor artifacts."""
    parent = Path(parent).resolve(strict=True)
    if not parent.is_dir() or parent.name == "main":
        raise ValueError("Predecessor must be a run parent containing dispatch.json or main/")
    queue = _optional_json(parent / "queue.json")
    if not (parent / "main").is_dir():
        # H may be submitted while F is still waiting for an occupied GPU.
        # Only a genuine same-version dispatch can precede its main directory.
        dispatch = _optional_json(parent / "dispatch.json")
        pending_states = {None, "waiting_for_predecessor", "predecessor_complete",
                          "waiting_for_idle_gpu", "idle_gpu_acquired", "production_starting"}
        if not (dispatch.get("arm") in {"f", "g", "h"}
                and dispatch.get("gpu") in {0, 1}
                and dispatch.get("scope") == "pseg3_fgh_v6_fresh_smoke_then_fresh200"
                and queue.get("state") in pending_states | {"failed"}):
            raise ValueError("Predecessor must be a run parent containing dispatch.json or main/")
    status = _optional_json(parent / "main" / "status.json")
    acceptance = _optional_json(parent / "main" / "acceptance.json")
    if queue.get("state") == "failed" or status.get("state") == "failed" or (
            parent / "main" / "failure.json").exists():
        raise RuntimeError("Predecessor failed; refuse automatic replacement")
    completed = (queue.get("state") == "complete" and status.get("state") == "complete"
                 and acceptance.get("complete") is True)
    return {"complete": completed, "queue_state": queue.get("state"),
            "state": status.get("state"), "epoch": status.get("epoch"),
            "updates": status.get("updates"), "checked_at": timestamp()}


def wait_for_predecessor(parent, root):
    """Queue behind successful completion; idle-GPU checks still follow."""
    while True:
        evidence = predecessor_state(parent)
        write_json(root / "queue.json", {
            "state": "predecessor_complete" if evidence["complete"] else "waiting_for_predecessor",
            "predecessor": str(Path(parent).resolve()), "predecessor_status": evidence,
            "at": timestamp()})
        if evidence["complete"]:
            return evidence
        time.sleep(30)


def wait_for_idle(gpu, root):
    """Use the existing nvidia-smi process/utilization/memory preflight."""
    while True:
        try:
            checked = ensure_idle(gpu)
            write_json(root / "queue.json", {"state": "idle_gpu_acquired", "check": checked,
                                               "at": timestamp()})
            return checked
        except (RuntimeError, subprocess.SubprocessError) as exc:
            write_json(root / "queue.json", {"state": "waiting_for_idle_gpu", "reason": str(exc),
                                               "at": timestamp()})
            time.sleep(30)


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _rows(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _finite(value):
    import torch
    if torch.is_tensor(value):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite(item) for item in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def _diagnostics(row):
    # The trainer may keep scalar diagnostics nested or flatten them into steps.
    return {**row, **row.get("diagnostics", {})}


def _geometry(row):
    payload = row.get("monitor", row)
    geometry = payload.get("geometry", {})
    return geometry.get("geometry", geometry)


def _gradients(row):
    payload = row.get("monitor", row)
    gradients = payload.get("gradients", row.get("gradients", {}))
    return gradients.get("gradients", gradients)


def _head_gradients_unused(gradient_rows):
    """Both weighted R losses must leave all retained head parameters unused."""
    checked = False
    for row in gradient_rows:
        gradients = _gradients(row)
        for loss in ("contr", "hier"):
            if loss not in gradients:
                return False
            groups = gradients[loss].get("parameter_groups", {})
            head = {name: value for name, value in groups.items()
                    if name == "head" or name.lower() in {"m1", "m2", "m3"}}
            if not head:
                return False
            for value in head.values():
                checked = True
                if value.get("status") != "none":
                    return False
                if value.get("none_count") != value.get("parameter_tensors"):
                    return False
    return checked


def _regularizer_routing(gradient_rows, arm):
    """Check graph use, allowing saturation to give connected zero gradients."""
    if arm not in {"f", "g", "h"} or not gradient_rows:
        return False
    post9 = arm == "h"
    for row in gradient_rows:
        for loss in ("contr", "hier"):
            item = _gradients(row).get(loss, {})
            if item.get("loss_requires_grad") is not True:
                return False
            nodes, groups = item.get("nodes", {}), item.get("parameter_groups", {})
            for name in ("pre_conv9", "regularizer_source"):
                value = nodes.get(name, {})
                if value.get("status") not in {"zero", "nonzero"} or value.get("finite") is not True:
                    return False
            for name in ("post_conv9", "h"):
                value = nodes.get(name, {})
                if post9:
                    if value.get("status") not in {"zero", "nonzero"} or value.get("finite") is not True:
                        return False
                elif value.get("status") != "none":
                    return False
            shared = groups.get("shared", {})
            if (shared.get("status") not in {"zero", "nonzero"}
                    or shared.get("finite") is not True
                    or shared.get("numel_with_gradient", 0) <= 0):
                return False
            group = groups.get("euclidean9", {})
            if post9:
                if (group.get("status") not in {"zero", "nonzero"}
                        or group.get("finite") is not True
                        or group.get("numel_with_gradient", 0) <= 0):
                    return False
            elif (group.get("status") != "none" or group.get("numel_with_gradient") != 0
                  or group.get("none_count") != group.get("parameter_tensors")):
                return False
    return True


def _ce_reaches_euclidean9_and_head(gradient_rows):
    """CE must still use the preserved Euclidean block and the two-map head."""
    if not gradient_rows:
        return False
    for row in gradient_rows:
        ce = _gradients(row).get("ce", {})
        if ce.get("loss_requires_grad") is not True:
            return False
        nodes = ce.get("nodes", {})
        for name in ("pre_conv9", "post_conv9", "h", "regularizer_source"):
            node = nodes.get(name, {})
            if node.get("status") not in {"zero", "nonzero"} or node.get("finite") is not True:
                return False
        groups = ce.get("parameter_groups", {})
        for name in ("euclidean9", "m2", "m3"):
            group = groups.get(name, {})
            if group.get("status") not in {"zero", "nonzero"} or group.get("finite") is not True:
                return False
            if group.get("numel_with_gradient", 0) <= 0:
                return False
    return True


def validate_smoke(smoke, arm=None):
    """Validate saved real-data smoke artifacts on CPU, without another forward."""
    import torch
    smoke = Path(smoke)
    manifest = _json(smoke / "manifest.json")
    accepted = _json(smoke / "acceptance.json")
    metrics = _rows(smoke / "metrics.jsonl")
    steps = _rows(smoke / "steps.jsonl")
    geometry_rows = _rows(smoke / "geometry.jsonl")
    e0 = torch.load(smoke / "e0.pth", map_location="cpu", weights_only=False)
    state = torch.load(smoke / "last.pth", map_location="cpu", weights_only=False)
    config = manifest["config"]
    selected_arm = config.get("arm")
    active = True
    diagnostics = [_diagnostics(row) for row in steps]
    gradient_rows = [row for row in geometry_rows if _gradients(row)]
    required_geometry = {"pre_conv9", "post_conv9", "regularizer_source", "h", "point_exp", "m2", "m3",
                         "whole_tangent", "part_tangent", "whole_exp", "part_exp"}
    final_bn = metrics[-1].get("bn_counts", {}) if metrics else {}
    expected_lr = .001 + (.1 - .001) * (1 + math.cos(math.pi * 2 / 200)) / 2
    rng_keys = {"rng", "train_generator", "eval_generator", "part_generator", "negative_generator"}
    state_keys = rng_keys | {"config", "source", "model", "optimizer", "scheduler",
                             "initial_model_sha256"}
    checks = {
        "two_epochs_two_steps": len(metrics) == 2 and all(row.get("steps") == 2 for row in metrics),
        "four_updates": accepted.get("updates") == state.get("updates") == 4,
        "accepted_complete": accepted.get("complete") is True,
        "no_test": accepted.get("final_test_evaluations") == 0 and not (smoke / "final_test.json").exists(),
        "validation_only": config.get("protocol") == "validation"
                           and all(row.get("selection_split") == "val" for row in metrics),
        "full_point_count_batch": config.get("num_points") == 2048 and config.get("batch_size") == 32,
        "real_data": "synthetic" not in json.dumps(manifest.get("dataset_source", "synthetic")).lower()
                     and manifest.get("train_objects", 0) >= 64,
        "seed4780": manifest.get("seed") == config.get("seed") == 4780,
        "arm_identity": config.get("arm") in {"f", "g", "h"}
                        and (arm is None or config.get("arm") == arm),
        "approved_regularizer_coefficients": (config.get("alpha"), config.get("beta"))
                                             == approved_coefficients(selected_arm),
        "active_regularizer_identity": manifest.get("regularizers_active") is active
                                       and manifest.get("regularizer_diagnostics_only") is (not active)
                                       and all(row.get("regularizers_active") is active
                                               and row.get("regularizer_diagnostics_only") is (not active)
                                               for row in diagnostics),
        "dp1_matches_arm": manifest.get("dropout_active") is (selected_arm == "g")
                           and config.get("dropout") == .5
                           and manifest.get("dp1_location") == ("before_conv9" if selected_arm == "g" else "removed"),
        "euclidean9_retained": manifest.get("euclidean_conv9_retained") is True
                               and tuple(e0["model"].get("conv9.0.weight", torch.empty(0)).shape) == (256, 256, 1)
                               and "bn9.running_var" in e0["model"],
        "two_mobius_head_no_equal_width_m1": manifest.get("has_m1") is False
                                             and not any(name.startswith("emb.") for name in e0["model"])
                                             and tuple(e0["model"].get("emb2._weight", torch.empty(0)).shape) == (128, 256)
                                             and tuple(e0["model"].get("mlr._weight", torch.empty(0)).shape) == (50, 128),
        "main_h_after_euclidean9": manifest.get("feature_location") == "after_conv9_bn9_leaky",
        "regularizer_source_matches_arm": manifest.get("regularizer_feature_location")
                                          == ("after_conv9_bn9_leaky" if selected_arm == "h" else "before_conv9"),
        "versioned_source": manifest.get("format") == "hycore_seg_v6"
                            and state.get("format") == e0.get("format") == "hycore_seg_v6_complete",
        "parameter_count": manifest.get("trainable_parameters") == 1460283,
        "old_tail_absent": not any(name.startswith(("conv10.", "conv11.", "bn10."))
                                   for name in e0["model"]),
        "bn_once_per_step": bool(final_bn) and all(value == 4 for value in final_bn.values()),
        "finite_parameters": _finite(e0["model"]) and _finite(state["model"]),
        "finite_optimizer": _finite(e0["optimizer"]) and _finite(state["optimizer"]),
        "recorded_finite": all(row.get("model_finite") is True and row.get("optimizer_finite") is True
                               for row in metrics),
        "initial_lr_point1": config.get("lr") == .1
                             and all(group.get("lr") == .1 for group in e0["optimizer"]["param_groups"]),
        "minimum_lr_point001": config.get("min_lr") == .001 and state["scheduler"].get("eta_min") == .001,
        "scheduler_keeps_200": config.get("schedule_epochs") == 200
                               and state["scheduler"].get("T_max") == 200
                               and state["scheduler"].get("last_epoch") == 2,
        "scheduler_actual_lr": all(abs(group["lr"] - expected_lr) < 1e-12
                                   for group in state["optimizer"]["param_groups"]),
        "full_state": state_keys.issubset(state) and state_keys.issubset(e0),
        "independent_rngs_saved": all(torch.is_tensor(state.get(key)) and state[key].numel() > 0
                                      for key in rng_keys - {"rng"}),
        "fresh_e0": e0.get("epoch") == 0 and e0.get("updates") == 0
                    and manifest.get("initialization") == "fresh",
        "e0_hash": state_hash(e0["model"]) == manifest.get("initial_model_sha256")
                   == e0.get("initial_model_sha256") == state.get("initial_model_sha256"),
        "no_gradient_clipping": bool(steps) and all(row.get("clip_applied") is False for row in steps),
        "finite_unclipped_gradient_norms": bool(steps) and all(
            isinstance(row.get("gradient_norm"), (int, float))
            and math.isfinite(row["gradient_norm"]) for row in steps),
        "old_threshold_observed_only": bool(steps) and all(
            row.get("exceeds_old_clip_threshold") == (row["gradient_norm"] > 1.) for row in steps),
        "all_negatives_legal": bool(diagnostics) and all(row.get("negative_legal_rate") == 1.
                                                         for row in diagnostics),
        "no_self_negatives": bool(diagnostics) and all(row.get("self_pair_fraction") == 0.
                                                       for row in diagnostics),
        "no_same_class_negatives": bool(diagnostics) and all(row.get("same_class_pair_fraction") == 0.
                                                             for row in diagnostics),
        "full_negative_denominator": bool(diagnostics) and all(
            row.get("contrast_denominator") == row.get("valid_negative") == 32 for row in diagnostics),
        "margin1000_over_n": bool(diagnostics) and all(
            abs(row["gamma"] * row["part_points"] - 1000.) < 1e-7 for row in diagnostics),
        "delta4": bool(diagnostics) and all(row.get("delta") == 4 for row in diagnostics),
        "geometry_logged": bool(geometry_rows) and all(required_geometry.issubset(_geometry(row))
                                                       for row in geometry_rows),
        "geometry_finite": bool(geometry_rows) and all(
            _geometry(row)[name].get("finite") is True
            for row in geometry_rows for name in required_geometry if name in _geometry(row)),
        "gradients_logged": bool(gradient_rows) and all({"ce", "contr", "hier"}.issubset(_gradients(row))
                                                       for row in gradient_rows),
        "regularizers_do_not_use_head": _head_gradients_unused(gradient_rows),
        "ce_uses_euclidean9_and_head": _ce_reaches_euclidean9_and_head(gradient_rows),
        # A real batch may be radially saturated. Do not require nonzero Rhier.
        "monitor_intervals": config.get("monitor_every") == 25 and config.get("gradient_every") == 200,
    }
    checks["regularizer_gradient_route_matches_arm"] = _regularizer_routing(gradient_rows, selected_arm)
    result = {"passed": all(checks.values()), "checks": checks,
              "arm": config.get("arm"), "initial_model_sha256": manifest.get("initial_model_sha256"),
              "regularizers_active": active, "regularizer_diagnostics_only": not active,
              "first_input_sha256": metrics[0].get("first_input_sha256") if metrics else None,
              "geometry_records": len(geometry_rows), "gradient_records": len(gradient_rows),
              "cuda_initialized_in_validator": torch.cuda.is_initialized()}
    write_json(smoke.parent / "smoke_validation.json", result)
    if not result["passed"]:
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError("Smoke acceptance failed: " + ", ".join(failed))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-data", required=True)
    parser.add_argument("--production-data", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--arm", choices=("f", "g", "h"), required=True)
    parser.add_argument("--wait-for-run", help="Predecessor run parent; successful completion and idle GPU are both required")
    parser.add_argument("--config", help="Defaults to the approved config for the selected arm")
    args = parser.parse_args(argv)
    if args.gpu not in {0, 1}:
        raise ValueError("Only physical GPU0/1 are authorized; GPU2 must remain available to others")
    config = Path(args.config or Path(__file__).parent / "configs" / f"fgh_{args.arm}.json").resolve(strict=True)
    resolved = _json(config)
    if resolved.get("arm") != args.arm or (resolved.get("alpha"), resolved.get("beta")) != approved_coefficients(args.arm):
        raise ValueError("Configuration does not match the approved F/G/H arm and active regularizers")
    root = Path(args.root).resolve()
    root.mkdir(parents=True, exist_ok=False)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["OMP_NUM_THREADS"] = "4"
    os.environ["MKL_NUM_THREADS"] = "4"
    try:
        identity = git_identity()
        if identity["source_status"]:
            raise RuntimeError("Commit segmentation source before dispatch")
        write_json(root / "dispatch.json", {
            "source": identity, "started": timestamp(), "gpu": args.gpu, "arm": args.arm,
            "protocol": "antao_testsel", "seed": 4780, "pid": os.getpid(),
            "python": sys.executable, "teacher": None, "source_checkpoint": None,
            "wait_for_run": str(Path(args.wait_for_run).resolve()) if args.wait_for_run else None,
            "scope": "pseg3_fgh_v6_fresh_smoke_then_fresh200",
            "regularizers_active": True, "regularizer_diagnostics_only": False,
        })
        command = [sys.executable, "-m", TRAIN_MODULE, "--config", str(config), "--arm", args.arm,
                   "--physical-gpu", str(args.gpu), "--monitor-every", "25", "--gradient-every", "200"]
        if args.wait_for_run:
            wait_for_predecessor(args.wait_for_run, root)
        wait_for_idle(args.gpu, root)
        with (root / "smoke.log").open("w", encoding="utf-8") as log:
            subprocess.run(command + ["--data", args.smoke_data, "--out", str(root / "smoke"),
                                      "--protocol", "validation", "--epochs", "2", "--max-steps", "2",
                                      "--limit-val", "32", "--no-final-test"],
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        checked = validate_smoke(root / "smoke", args.arm)
        wait_for_idle(args.gpu, root)
        write_json(root / "queue.json", {"state": "production_starting", "at": timestamp(),
                                         "expected_initial_model_sha256": checked["initial_model_sha256"]})
        with (root / "production.log").open("w", encoding="utf-8") as log:
            subprocess.run(command + ["--data", args.production_data, "--out", str(root / "main"),
                                      "--protocol", "antao_testsel", "--epochs", "200",
                                      "--expected-initial-sha256", checked["initial_model_sha256"]],
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        production = _json(root / "main" / "manifest.json")
        if production["initial_model_sha256"] != checked["initial_model_sha256"]:
            raise AssertionError("Production initial state differs from accepted fresh smoke")
        write_json(root / "queue.json", {"state": "complete", "at": timestamp()})
    except Exception as exc:
        write_json(root / "queue.json", {"state": "failed", "error": repr(exc), "at": timestamp()})
        raise


if __name__ == "__main__":
    main()
