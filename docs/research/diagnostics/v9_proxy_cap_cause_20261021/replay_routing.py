"""Frozen, same-GPU replay separating GHHC routing from direct distance gradients.

Only saved training-forward snapshots are read. No encoder, data loader,
optimizer, parameter update, evaluation, or test data are used. CUDA visibility
and an immediately preceding idle-GPU check are the launcher's responsibility.
The JSON output contains anonymous aggregate statistics, never IDs or tensors.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from analyze import CAP, DEFAULT_SNAPSHOTS, describe, fixed_choice_loss, gradient_report
from inter_hierarchy_MN40.hier_proxy_scratch_v8.geometry import official_distance, to_ball
from inter_hierarchy_MN40.hier_proxy_scratch_v6.hier_loss import ghhc_loss

EXPECTED_SOURCE = {
    "v9_training": ("inter_hierarchy_MN40/hier_proxy_scratch_v9/train.py",
        "b7d06467e4f6a8650fb4317655ea5c3fcfef567126198b325e885a0b2f28a040"),
    "v8_geometry": ("inter_hierarchy_MN40/hier_proxy_scratch_v8/geometry.py",
        "16d5a7879d8d23924c6fe542bb1d47973448044604602f4ab44c9c72f8c871dc"),
    "v8_loss": ("inter_hierarchy_MN40/hier_proxy_scratch_v8/hier_loss.py",
        "e77ff498d1cd056a30c7932dfcf367379755711a4aefc7f70b1d8a88dbed651d"),
    "v6_ghhc": ("inter_hierarchy_MN40/hier_proxy_scratch_v6/hier_loss.py",
        "e2cd382dfc3be20605c3f50b01f8be3fed11e11f7c8b0f3b4f869997e521dcfa"),
}


def source_guard():
    result = {}
    for key, (relative, expected) in EXPECTED_SOURCE.items():
        source = (REPO / relative).read_text(encoding="utf-8").replace("\r\n", "\n")
        actual = hashlib.sha256(source.encode("utf-8")).hexdigest()
        if actual != expected:
            raise RuntimeError(f"Reviewed source contract differs: {key}")
        result[key] = actual
    return result


def device_tree(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: device_tree(item, device) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(device_tree(item, device) for item in value)
    return value


def comparison(replayed, recorded):
    replayed, recorded = replayed.detach().double(), recorded.detach().to(replayed.device).double()
    difference = replayed - recorded
    denominator = recorded.norm()
    return {"max_abs_error": float(difference.abs().max()) if difference.numel() else 0.,
            "mean_abs_error": float(difference.abs().mean()) if difference.numel() else 0.,
            "l2_error": float(difference.norm()),
            "recorded_l2_norm": float(denominator),
            "relative_l2_error": float(difference.norm() / denominator) if float(denominator) else None,
            "bitwise_equal": bool(torch.equal(replayed, recorded)),
            "elements": difference.numel()}


def gradient_alignment(full, direct):
    full, direct = full.detach().double().flatten(), direct.detach().double().flatten()
    denominator = full.norm() * direct.norm()
    residual = full - direct
    return {"fixed_vs_full_cosine": float(torch.dot(full, direct) / denominator) if float(denominator) else None,
            "fixed_vs_full_norm_ratio": float(direct.norm() / full.norm()) if float(full.norm()) else None,
            "routing_vs_full_norm_ratio": float(residual.norm() / full.norm()) if float(full.norm()) else None}


def replay_component(p, whole, recorded, observed_gradient, weight, device,
                     gradient_atol=1e-6, gradient_rtol=1e-3):
    """Rebuild both gradients on the SAME distance graph and CUDA noise stream."""
    proxies = to_ball(p)
    distances = official_distance(whole if whole is not None else proxies, proxies)
    cuda_recorded = device_tree(recorded, device)
    state = recorded.get("generator_before")
    if not isinstance(state, torch.Tensor) or state.device.type != "cpu" or state.dtype != torch.uint8:
        raise ValueError("A saved CPU ByteTensor CUDA generator state is required")
    generator = torch.Generator(device=device)
    generator.set_state(state)
    full, _stats, details = ghhc_loss(distances, cuda_recorded["triplets"],
        tau=float(recorded["tau"]), margin=float(recorded["margin"]),
        generator=generator, return_details=True)
    g_full, = torch.autograd.grad(weight * full, p, retain_graph=True)
    fixed, relu_scalar, fixed_hinge_disagreements = fixed_choice_loss(distances, cuda_recorded)
    g_fixed, = torch.autograd.grad(weight * fixed, p)
    g_routing = g_full.detach() - g_fixed.detach()
    checks = {}
    for key in ("pair_proxy_idx", "triple_proxy_idx"):
        saved, actual = cuda_recorded["details"][key], details[key]
        checks[key + "_mismatches"] = int((actual != saved).sum())
    checks["masked_hinge_activity_mismatches"] = int((
        (details["masked_hinges"] > 0) != (cuda_recorded["details"]["masked_hinges"] > 0)).sum())
    checks["fixed_hinge_activity_mismatches"] = fixed_hinge_disagreements
    distance_error = comparison(distances, cuda_recorded["distances"])
    gradient_error = comparison(g_full, observed_gradient)
    hinge_error = comparison(details["masked_hinges"], cuda_recorded["details"]["masked_hinges"])
    loss_error = abs(float(full.detach()) - float(recorded["loss"]))
    failures = [key for key, value in checks.items() if value]
    # Exact hard choices/activity plus independently bounded full-gradient replay
    # are prerequisites for attributing full-minus-fixed to the ST routing path.
    if gradient_error["max_abs_error"] > gradient_atol:
        failures.append("full_gradient_absolute_error")
    if gradient_error["relative_l2_error"] is not None and gradient_error["relative_l2_error"] > gradient_rtol:
        failures.append("full_gradient_relative_error")
    if not torch.allclose(distances.detach(), cuda_recorded["distances"], atol=2e-5, rtol=2e-6):
        failures.append("distance_replay_error")
    if loss_error > 1e-6:
        failures.append("loss_replay_error")
    result = {"validation_passed": not failures, "failures": failures,
        "exact_choice_checks": checks, "distances": distance_error,
        "masked_hinges": hinge_error, "full_gradient_vs_observed": gradient_error,
        "unweighted_recorded_loss": float(recorded["loss"]),
        "unweighted_replayed_full_st_loss": float(full.detach()),
        "unweighted_fixed_active_scalar": float(fixed.detach()),
        "unweighted_fixed_relu_scalar": float(relu_scalar.detach()),
        "loss_abs_error": loss_error,
        **gradient_alignment(g_full, g_fixed),
        "decomposition_max_abs_error": float((g_full.detach() - g_fixed.detach() - g_routing).abs().max())}
    return result, {"full_st": g_full.detach().cpu(), "fixed_choice": g_fixed.detach().cpu(),
                    "routing": g_routing.cpu(), "observed_full_st": observed_gradient.detach().cpu()}


def analyze_snapshot(path, device, gradient_tol=1e-10, step_tol=1e-12,
                     gradient_atol=1e-6, gradient_rtol=1e-3):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    same, actual = payload["same_forward"], payload["per_proxy"]
    raw = same["proxy_before"].detach()
    if not torch.equal(raw, payload["actual_optimizer_step"]["parameter"]):
        raise ValueError("Saved same-forward proxies differ from optimizer pre-state")
    if payload["source"]["original_normalized_sha256"] != EXPECTED_SOURCE["v9_training"][1]:
        raise ValueError("Snapshot original training source differs from reviewed V9")
    p = raw.to(device).clone().requires_grad_(True)
    whole = same["hier_mu"].detach().to(device)
    radii = raw.double().norm(dim=-1)
    masks = {"all": torch.ones(len(raw), dtype=torch.bool), "inside": radii <= CAP, "outside": radii > CAP}
    output = {"epoch": int(payload["epoch"]), "step": int(payload["step"]),
        "snapshot_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "proxy_count": len(raw), "parameter_norm": describe(radii.numpy()),
        "hier_weight": float(same["hier_weight"]),
        "recorded_component_sum_vs_actual_synced_gradient_max_abs": float(
            same["weighted_component_sum_vs_actual_synced_gradient_max_abs"]), "components": {}}
    for name in ("sample", "proxy"):
        result, gradients = replay_component(p, whole if name == "sample" else None,
            same[name + "_ghhc"], same["weighted_component_gradients"][name + "_loss"],
            float(same["hier_weight"]), device, gradient_atol, gradient_rtol)
        result["gradients"] = {kind: {cohort: gradient_report(raw, gradient, actual, mask,
            gradient_tol, step_tol) for cohort, mask in masks.items()} for kind, gradient in gradients.items()}
        output["components"][name] = result
    output["validation_passed"] = all(row["validation_passed"] for row in output["components"].values())
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="New anonymous aggregate JSON file")
    parser.add_argument("--snapshots", default=DEFAULT_SNAPSHOTS)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--gradient-tolerance", type=float, default=1e-10)
    parser.add_argument("--step-tolerance", type=float, default=1e-12)
    parser.add_argument("--gradient-atol", type=float, default=1e-6)
    parser.add_argument("--gradient-rtol", type=float, default=1e-3)
    args = parser.parse_args(argv)
    if not 1 <= args.threads <= 2:
        parser.error("Frozen replay permits one or two CPU threads")
    labels = args.snapshots.split(",")
    if not labels or len(labels) != len(set(labels)) or any(not re.fullmatch(r"e\d{3}_s\d{3}", x) for x in labels):
        parser.error("Use unique comma-separated e001_s001 snapshot labels")
    if args.out.exists():
        raise FileExistsError("Output already exists; use a new JSON file")
    before = source_guard()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Expose exactly one previously checked idle CUDA device")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    rows = [analyze_snapshot(args.run_dir / f"proxy_cause_{label}.pth", device,
        args.gradient_tolerance, args.step_tolerance, args.gradient_atol, args.gradient_rtol) for label in labels]
    torch.cuda.synchronize(device)
    after = source_guard()
    if before != after:
        raise RuntimeError("Production source changed during frozen replay")
    passed = all(row["validation_passed"] for row in rows)
    output = {"status": "passed" if passed else "failed", "scope": "frozen saved training forward only",
        "encoder_forwards": 0, "optimizer_steps": 0, "training_updates": 0, "test_reads": 0,
        "semantics": "routing = same-GPU replayed full ST gradient minus recorded-choice recorded-active-hinge direct gradient",
        "source_guard": {"before": before, "after": after, "unchanged": before == after},
        "runtime": {"torch": torch.__version__, "cuda": torch.version.cuda,
                    "device_name": torch.cuda.get_device_name(device),
                    "allow_tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
                    "float32_matmul_precision": torch.get_float32_matmul_precision()},
        "validation_thresholds": {"gradient_atol": args.gradient_atol, "gradient_relative_l2": args.gradient_rtol,
            "exact_choice_and_hinge_activity_mismatches": 0,
            "distance_atol": 2e-5, "distance_rtol": 2e-6, "loss_atol": 1e-6},
        "snapshots": rows}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(output, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": output["status"], "snapshot_count": len(rows),
        "component_count": 2 * len(rows), "production_source_unchanged": before == after}))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
