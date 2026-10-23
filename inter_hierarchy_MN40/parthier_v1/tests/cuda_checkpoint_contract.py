"""Real PointMLP CUDA checkpoint and one-RSGD-step equivalence contract.

Run only after an external idle-GPU guard assigns exactly one visible device:
python -m inter_hierarchy_MN40.parthier_v1.tests.cuda_checkpoint_contract \
    --result-json /new/private/run/cuda_checkpoint_contract.json

This does not choose a GPU, load data, use a fallback operator or train a run.
"""
from __future__ import annotations

import argparse
import datetime as dt
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time


LIMITS = {
    "output_rtol": 1e-5, "output_atol": 1e-6,
    "bn_rtol": 1e-6, "bn_atol": 1e-7,
    "gradient_global_relative_l2": 1e-4,
    "gradient_parameter_relative_l2": 1e-3,
    "gradient_max_abs_floor": 1e-5, "gradient_max_abs_peak_factor": 1e-3,
    "update_global_relative_l2": 2e-4,
    "update_parameter_relative_l2": 2e-3,
    "update_max_abs_floor": 1e-6, "update_max_abs_peak_factor": 2e-3,
    "relative_metric_denominator_floor": 1e-8,
}


def tensor_sha(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def state_sha(state):
    h = hashlib.sha256()
    for name, value in sorted(state.items()):
        h.update(name.encode("utf-8") + b"\0")
        h.update(str(value.dtype).encode("ascii") + b"\0")
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def capture_rng():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "cpu": torch.get_rng_state().clone(),
            "cuda": [value.clone() for value in torch.cuda.get_rng_state_all()]}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["cpu"])
    torch.cuda.set_rng_state_all(state["cuda"])


def rng_equal(a, b):
    numpy_equal = (a["numpy"][0] == b["numpy"][0]
                   and np.array_equal(a["numpy"][1], b["numpy"][1])
                   and a["numpy"][2:] == b["numpy"][2:])
    return {"python": a["python"] == b["python"], "numpy": bool(numpy_equal),
            "torch_cpu": torch.equal(a["cpu"], b["cpu"]),
            "torch_cuda": len(a["cuda"]) == len(b["cuda"])
                          and all(torch.equal(x, y) for x, y in zip(a["cuda"], b["cuda"]))}


def cpu_buffers(model):
    return {name: value.detach().cpu().clone() for name, value in model.named_buffers()}


def run_case(initial, points, start_rng, activation_checkpoint):
    gc.collect()
    torch.cuda.empty_cache()
    model = configure_backbone(Hype_pointMLP()).to("cuda:0").train()
    model.load_state_dict(initial, strict=True)
    wrapper = HyCoReTrainingForward(model, activation_checkpoint=activation_checkpoint).train()
    named = {name: value for name, value in model.named_parameters() if value.requires_grad}
    optimizer = geoopt.optim.RiemannianSGD(list(named.values()), lr=.1, momentum=.9, weight_decay=2e-4)
    initial_parameters = {name: value.detach().cpu().clone() for name, value in named.items()}
    initial_buffers = cpu_buffers(model)
    tensor = points.to("cuda:0")
    if tensor.requires_grad:
        raise AssertionError("the contract requires ordinary point inputs without gradients")
    optimizer.zero_grad(set_to_none=True)
    restore_rng(start_rng)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    outputs = wrapper(tensor, 1024, whole_centers=[0, 137, 509, 1023], centers=[17, 211, 509, 1023])
    torch.cuda.synchronize()
    forward_peak = torch.cuda.max_memory_allocated() / 2**20
    forward_buffers = cpu_buffers(model)
    output_cpu = {key: outputs[key].detach().cpu().clone() for key in ("mu", "parts", "logits")}
    if not all(bool(torch.isfinite(value).all()) for value in output_cpu.values()):
        raise FloatingPointError("nonfinite model outputs")
    loss = sum(outputs[key].square().sum() for key in ("mu", "parts", "logits"))
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError("nonfinite all-output square loss")
    loss.backward()
    missing = [name for name, value in named.items() if value.grad is None]
    if missing:
        raise AssertionError("trainable parameters without gradients: " + ", ".join(missing))
    raw_gradients = {name: value.grad.detach().cpu().clone() for name, value in named.items()}
    if not all(bool(torch.isfinite(value).all()) for value in raw_gradients.values()):
        raise FloatingPointError("nonfinite parameter gradients")
    norm_before = float(torch.nn.utils.clip_grad_norm_(list(named.values()), 1.))
    if not math.isfinite(norm_before) or norm_before <= 0:
        raise FloatingPointError("expected a finite, nonzero gradient norm before clipping")
    clipped_gradients = {name: value.grad.detach().cpu().clone() for name, value in named.items()}
    norm_after = float(sum(value.double().square().sum() for value in clipped_gradients.values()).sqrt())
    if not math.isfinite(norm_after) or norm_after <= 0:
        raise FloatingPointError("expected a finite, nonzero gradient norm after clipping")
    optimizer.step()
    torch.cuda.synchronize()
    final_parameters = {name: value.detach().cpu().clone() for name, value in named.items()}
    if not all(bool(torch.isfinite(value).all()) for value in final_parameters.values()):
        raise FloatingPointError("nonfinite RSGD update")
    final_buffers = cpu_buffers(model)
    counts = {}
    for name, value in forward_buffers.items():
        if name.endswith("num_batches_tracked"):
            delta = int(value) - int(initial_buffers[name])
            counts[name] = delta
            if delta != 4 or not torch.equal(value, final_buffers[name]):
                raise AssertionError("BN effective updates must remain four: " + name)
        if not torch.equal(value, final_buffers[name]):
            raise AssertionError("backward/RSGD changed BN or other model buffer: " + name)
    if not norm_after <= 1.00001:
        raise AssertionError("production norm clipping did not apply")
    final_rng = capture_rng()
    momentum = {name: optimizer.state[value]["momentum_buffer"].detach().cpu().clone()
                for name, value in named.items()}
    summary = {
        "activation_checkpoint": activation_checkpoint, "loss": float(loss.detach()),
        "trainable_parameter_tensors": len(named), "all_trainable_parameters_have_gradient": True,
        "input_requires_grad": False, "gradient_norm_before_clip": norm_before,
        "gradient_norm_after_clip": norm_after, "bn_update_deltas": counts,
        "BN_unchanged_by_backward_and_step": True, "optimizer_group_steps": [g["step"] for g in optimizer.param_groups],
        "forward_peak_allocated_MiB": forward_peak,
        "total_peak_allocated_MiB": torch.cuda.max_memory_allocated() / 2**20,
        "total_peak_reserved_MiB": torch.cuda.max_memory_reserved() / 2**20,
        "wall_seconds": time.perf_counter() - started,
        "crop_indices_sha256": tensor_sha(outputs["input_stats"]["part_source_indices"]),
        "rng_cpu_sha256": tensor_sha(final_rng["cpu"]),
        "rng_cuda_sha256": [tensor_sha(value) for value in final_rng["cuda"]],
    }
    return {"summary": summary, "outputs": output_cpu, "raw_gradients": raw_gradients,
            "clipped_gradients": clipped_gradients, "momentum": momentum,
            "updates": {name: value - initial_parameters[name] for name, value in final_parameters.items()},
            "buffers": final_buffers, "rng": final_rng}


def compare_tensors(reference, actual, *, category):
    if reference.keys() != actual.keys():
        raise AssertionError("comparison tensor names differ: " + category)
    update = category == "updates"
    prefix = "update" if update else "gradient"
    floor = LIMITS[prefix + "_max_abs_floor"]
    factor = LIMITS[prefix + "_max_abs_peak_factor"]
    per_parameter_rtol = LIMITS[prefix + "_parameter_relative_l2"]
    sum_square_error, sum_square_ref = 0., 0.
    max_abs, max_relative, exact = 0., 0., True
    violations, details = [], {}
    for name, reference_tensor in reference.items():
        expected, value = reference_tensor.double(), actual[name].double()
        if not bool(torch.isfinite(value).all()):
            raise FloatingPointError(category + " nonfinite: " + name)
        diff = value - expected
        error_l2, ref_l2 = float(diff.norm()), float(expected.norm())
        abs_error = float(diff.abs().max())
        relative = float((diff.abs() / expected.abs().clamp_min(LIMITS["relative_metric_denominator_floor"])).max())
        peak = max(float(expected.abs().max()), float(value.abs().max()))
        relative_l2 = error_l2 / max(ref_l2, 1e-30)
        # Relative error near a zero/cancelled coordinate is recorded but is
        # not a standalone gate. Use whole-vector and peak-scaled gates.
        if abs_error > floor + factor * peak or (error_l2 > floor and relative_l2 > per_parameter_rtol):
            violations.append(name)
        details[name] = {"reference_l2": ref_l2, "relative_l2": relative_l2,
                         "max_absolute_error": abs_error, "max_relative_error": relative,
                         "maximum_allowed_absolute_error": floor + factor * peak}
        sum_square_error += error_l2**2
        sum_square_ref += ref_l2**2
        max_abs, max_relative = max(max_abs, abs_error), max(max_relative, relative)
        exact &= torch.equal(reference_tensor, actual[name])
    global_relative_l2 = (sum_square_error / max(sum_square_ref, 1e-60)) ** .5
    passed = not violations and global_relative_l2 <= LIMITS[prefix + "_global_relative_l2"]
    return {"passed": passed, "bitwise_equal": exact, "global_relative_l2": global_relative_l2,
            "max_absolute_error": max_abs, "max_relative_error": max_relative,
            "violating_parameter_names": violations, "per_parameter": details}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-json", type=Path, required=True)
    args = parser.parse_args()
    target = args.result_json.expanduser().resolve()
    if target.exists():
        raise FileExistsError("refusing to replace an existing contract result")
    result = {"status": "starting", "thresholds": LIMITS,
              "threshold_explanation": "CUDA pointnet2 scatter/atomic backward may differ in summation order. "
                  "Raw/clipped gradients and momentum use global relL2<=1e-4, per-parameter relL2<=1e-3 "
                  "outside an absolute floor, plus max-error<=1e-5+1e-3*parameter-gradient peak. "
                  "RSGD deltas use global relL2<=2e-4 and parameter relL2<=2e-3 with "
                  "max-error<=1e-6+2e-3*delta peak to allow FP32 parameter-subtraction quantization. "
                  "Maximum coordinate-relative error uses denominator floor1e-8 and is descriptive near zeros.",
              "fixture": {"seed": 4780, "local_batch": 4, "input_points": 1024, "whole_count": 1024,
                          "part_counts": [200, 400, 600], "precision": "FP32", "input_requires_grad": False},
              "optimizer": {"name": "geoopt.RiemannianSGD", "lr": .1, "momentum": .9,
                            "weight_decay": 2e-4, "norm_clip": 1., "steps": 1}}
    try:
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if visible is None or not visible.strip() or len(visible.split(",")) != 1 or visible.strip() == "-1":
            raise RuntimeError("an external idle-GPU guard must assign one explicit CUDA_VISIBLE_DEVICES entry")
        global torch, np, geoopt, Hype_pointMLP, HyCoReTrainingForward, configure_backbone
        import torch
        import numpy as np
        import geoopt
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("the externally guarded contract requires exactly one visible CUDA device")
        torch.cuda.set_device(0)
        repo = Path(__file__).resolve().parents[3]
        result.update(source_commit=os.environ.get("PARTHIER_SOURCE_COMMIT") or subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
            started_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
            python=sys.version, physical_gpu_uuids=os.environ.get("HYCORE_PHYSICAL_GPU_UUIDS"))
        sys.path[:0] = [str(repo), str(repo / "inter_hierarchy_MN40"), str(repo / "pointnet2_ops_lib")]
        from models.pointmlp import Hype_pointMLP
        from inter_hierarchy_MN40.parthier_v1.base_protocol import HyCoReTrainingForward, configure_backbone
        # Import the real extension explicitly. An absent compiled operator is
        # a failed contract; there is deliberately no Python/CPU substitute.
        from pointnet2_ops import pointnet2_utils
        if not callable(pointnet2_utils.furthest_point_sample):
            raise RuntimeError("real pointnet2 FPS operator unavailable")
        torch.set_num_threads(2)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        random.seed(4780)
        np.random.seed(4780)
        torch.manual_seed(4780)
        torch.cuda.manual_seed_all(4780)
        original = configure_backbone(Hype_pointMLP())
        initial = {name: value.detach().cpu().clone() for name, value in original.state_dict().items()}
        del original
        points = torch.randn(4, 1024, 3, generator=torch.Generator().manual_seed(4780)) * .2
        start_rng = capture_rng()
        result.update(torch=torch.__version__, geoopt=geoopt.__version__, gpu_name=torch.cuda.get_device_name(0),
                      cuda_visible_devices=visible, initial_state_sha256=state_sha(initial),
                      input_sha256=tensor_sha(points), real_pointnet2_operator=True,
                      cudnn_benchmark=False, cudnn_deterministic=True, tf32=False)
        reference = run_case(initial, points, start_rng, False)
        gc.collect()
        torch.cuda.empty_cache()
        actual = run_case(initial, points, start_rng, True)
        result["ordinary"], result["checkpoint"] = reference["summary"], actual["summary"]
        result["comparisons"] = {}
        for category in ("raw_gradients", "clipped_gradients", "momentum", "updates"):
            result["comparisons"][category] = compare_tensors(reference[category], actual[category], category=category)
        result["outputs"] = {}
        for name, expected in reference["outputs"].items():
            value = actual["outputs"][name]
            torch.testing.assert_close(value, expected, rtol=LIMITS["output_rtol"], atol=LIMITS["output_atol"])
            result["outputs"][name] = {"bitwise_equal": torch.equal(expected, value),
                                      "max_absolute_error": float((value - expected).abs().max())}
        buffer_max = 0.
        for name, expected in reference["buffers"].items():
            value = actual["buffers"][name]
            if expected.is_floating_point():
                torch.testing.assert_close(value, expected, rtol=LIMITS["bn_rtol"], atol=LIMITS["bn_atol"])
                buffer_max = max(buffer_max, float((value - expected).abs().max()))
            elif not torch.equal(expected, value):
                raise AssertionError("BN counter or integer buffer differs: " + name)
        result["BN_buffers"] = {"passed": True, "max_absolute_error": buffer_max,
                                "bitwise_equal": all(torch.equal(value, actual["buffers"][name])
                                                     for name, value in reference["buffers"].items())}
        result["RNG_equal"] = rng_equal(reference["rng"], actual["rng"])
        if not all(result["RNG_equal"].values()):
            raise AssertionError("checkpoint changed final RNG state")
        if reference["summary"]["crop_indices_sha256"] != actual["summary"]["crop_indices_sha256"]:
            raise AssertionError("fixed crop identity differs")
        failed = [name for name, comparison in result["comparisons"].items() if not comparison["passed"]]
        if failed:
            raise AssertionError("numerical equivalence thresholds exceeded: " + ", ".join(failed))
        result["status"] = "passed"
    except BaseException as exc:
        result.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
        raise
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps({"status": result["status"],
                      "raw_gradient_global_relative_l2": result["comparisons"]["raw_gradients"]["global_relative_l2"],
                      "update_global_relative_l2": result["comparisons"]["updates"]["global_relative_l2"],
                      "ordinary_peak_MiB": result["ordinary"]["total_peak_allocated_MiB"],
                      "checkpoint_peak_MiB": result["checkpoint"]["total_peak_allocated_MiB"]}))


if __name__ == "__main__":
    main()
