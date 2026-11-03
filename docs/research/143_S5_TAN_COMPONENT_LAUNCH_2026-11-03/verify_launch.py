"""CPU-only acceptance of a TAN single-component production run.

Read existing checkpoints/JSON only. No forward, optimizer step, CUDA context,
run mutation, raw IDs, or private paths are emitted. Invoke after epoch 1 exists.
"""
import argparse
import json
import math
import os
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def finite(value):
    if torch.is_tensor(value):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(finite(v) for v in value.values())
    if isinstance(value, (tuple, list)):
        return all(finite(v) for v in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def first_json_line(path):
    with path.open(encoding="utf-8") as stream:
        return json.loads(next(line for line in stream if line.strip()))


def matching_tensors(first, second):
    return first.keys() == second.keys() and all(
        torch.equal(tensor, second[name]) for name, tensor in first.items())


def close(first, second):
    return math.isclose(first, second, rel_tol=1e-5, abs_tol=1e-7)


def verify(run, reference_run, component):
    torch.set_num_threads(2)
    if torch.cuda.is_initialized():
        raise AssertionError("Acceptance must not initialize CUDA")
    manifest = read(run / "manifest.json")
    reference = read(reference_run / "manifest.json")
    first = first_json_line(run / "metrics.jsonl")
    ref_first = first_json_line(reference_run / "metrics.jsonl")
    geometry = first_json_line(run / "geometry.jsonl")
    e0 = torch.load(run / "e0.pth", map_location="cpu", weights_only=False)
    ref0 = torch.load(reference_run / "e0.pth", map_location="cpu", weights_only=False)
    e1 = torch.load(run / "epoch_001.pth", map_location="cpu", weights_only=False)

    expected = (.01, 0.) if component == "contr" else (0., .01)
    inactive = "hier" if component == "contr" else "contr"
    display = "S5-TAN-C" if component == "contr" else "S5-TAN-HIER"
    config = manifest["config"]
    ignored = {"component", "alpha", "beta", "config", "out", "physical_gpu"}
    if not reference["config"]["regularizers"]:
        ignored.add("regularizers")

    gradients = geometry["gradients"]
    active_gradient = gradients[component]
    inactive_gradient = gradients[inactive]
    diag = geometry["diagnostics"]
    q = geometry["geometry"]["final_tangent"]
    pairs = geometry["gradient_pairs_at_u"]["pairs"]
    active_pair = pairs["ce__" + component]
    inactive_pair = pairs["ce__" + inactive]
    active_pair_valid = (
        active_pair["status"] == "nonzero"
        and isinstance(active_pair["cosine"], (int, float))
        and math.isfinite(active_pair["cosine"])
        and -1. <= active_pair["cosine"] <= 1.
        and finite(active_pair)
    ) or (active_pair["status"] == "zero_norm"
          and active_pair["cosine"] is None and finite(active_pair))

    checks = {
        "display_and_component_match": manifest["display_name"] == display
            and manifest["active_regularizer_component"] == component
            and config["component"] == component,
        "original_single_component_coefficients": (config["alpha"], config["beta"]) == expected
            and config["regularizers"],
        "fresh_initial_hash_matches_reference": manifest["initial_model_sha256"]
            == reference["initial_model_sha256"] == e0["initial_model_sha256"],
        "all_e0_model_tensors_exactly_equal": matching_tensors(e0["model"], ref0["model"]),
        "initial_cpu_rng_exactly_equal": torch.equal(e0["rng"]["torch"], ref0["rng"]["torch"]),
        "data_manifest_exactly_equal": manifest["data_manifest_sha256"]
            == reference["data_manifest_sha256"] == e0["data_manifest_sha256"],
        "reference_config_equal_except_allowed_fields":
            {k: v for k, v in config.items() if k not in ignored}
            == {k: v for k, v in reference["config"].items() if k not in ignored},
        "first_input_exactly_equal": first["first_input_sha256"] == ref_first["first_input_sha256"],
        "first_pairing_exactly_equal": first["first_pairing"] == ref_first["first_pairing"],
        "first_full_epoch437": first["epoch"] == 1 and first["steps"] == 437
            and first["updates"] == 437 and e1["updates"] == 437,
        "bn437": bool(first["bn_counts"]) and all(v == 437 for v in first["bn_counts"].values()),
        "test2874": first["test"]["objects"] == 2874,
        "all50parts": len(first["test"]["parts"]) == 50,
        "train_regularizer_positive": first["train_reg"] > 0,
        "manifest_regularizer_on": manifest["regularizers_active"]
            and not manifest["regularizer_diagnostics_only"],
        "scale5": manifest["supervision_scale"] == config["supervision_scale"] == 5.,
        "real_output_metadata": manifest["final_output_space"] == "real_tangent_scores",
        "post_conv9_regularizer_source": manifest["regularizer_feature_location"]
            == "after_conv9_bn9_leaky",
        "no_dropout_or_gradient_clip": not manifest["dropout_active"]
            and not manifest["gradient_clipping"] and not geometry["clip_applied"],
        "model_and_optimizer_finite": finite(e1["model"]) and finite(e1["optimizer"])
            and first["model_finite"] and first["optimizer_finite"],
        "no_failure_file": not (run / "failure.json").exists(),
        "q_finite_without_ball_fields": q["finite"] and not any(
            key in q for key in ("near_boundary_fraction", "radius", "depth", "projection_radius")),
        "q_probability_recorded": "scaled_correct_class_probability" in q
            and finite(q["scaled_correct_class_probability"]),
        "no_m3_ball_node": "m3" not in geometry["geometry"],
        "all_gradient_terms_present": set(gradients) == {"ce", "contr", "hier"},
        "active_r_graph_reaches_u": active_gradient["loss_requires_grad"]
            and active_gradient["nodes"]["h"]["status"] != "none"
            and active_gradient["nodes"]["h"]["finite"],
        "active_r_reaches_frontend": all(
            active_gradient["parameter_groups"][group]["status"] != "none"
            and active_gradient["parameter_groups"][group]["finite"]
            for group in ("shared", "euclidean9")),
        "active_r_bypasses_both_maps": all(
            active_gradient["parameter_groups"][group]["status"] == "none"
            for group in ("m2", "m3")),
        "active_r_bypasses_real_scores": active_gradient["nodes"]["final_tangent"]["status"] == "none",
        "inactive_component_has_no_training_graph": not inactive_gradient["loss_requires_grad"]
            and inactive_gradient["loss"] == 0.
            and all(node["status"] == "none" for node in inactive_gradient["nodes"].values())
            and all(group["status"] == "none" for group in inactive_gradient["parameter_groups"].values()),
        "single_active_diagnostic_flags": diag["active_regularizer_component"] == component
            and diag[component + "_training_active"] and not diag[inactive + "_training_active"]
            and (diag["alpha"], diag["beta"]) == expected,
        "active_weight_matches_original_kernel": close(diag["active_weighted_" + component],
                                                        .01 * diag[component])
            and close(active_gradient["loss"], diag["active_weighted_" + component]),
        "inactive_weight_exactly_zero": diag["active_weighted_" + inactive] == 0.,
        "active_gradient_pair_valid": active_pair_valid,
        "inactive_gradient_pairs_unavailable": inactive_pair["status"] == "inactive_or_unused"
            and inactive_pair["cosine"] is None
            and pairs["contr__hier"]["status"] == "inactive_or_unused",
        "gradient_pair_interpretation_explicit": geometry["gradient_pairs_at_u"]["coordinate_space"]
            == "ambient_euclidean_U_post_conv9"
            and geometry["gradient_pairs_at_u"]["loss_units"]
            == "CE5_over5_and_active_weighted_regularizers",
        "cuda_not_initialized": not torch.cuda.is_initialized(),
    }
    status = read(run / "status.json")
    return {
        "checks": checks, "all_passed": all(checks.values()),
        "commit": manifest["source"]["commit"], "display_name": display,
        "component": component, "alpha": expected[0], "beta": expected[1],
        "first_epoch_instance_miou": first["test"]["instance_miou"],
        "first_epoch_class_miou": first["test"]["class_miou"],
        "first_epoch_seconds": first["seconds"],
        "first_input_sha256": first["first_input_sha256"],
        "initial_model_sha256": manifest["initial_model_sha256"],
        "geometry_nodes": list(geometry["geometry"]),
        "first_active_gradient_pair_at_u": active_pair,
        "current_status": {key: value for key, value in status.items()
                           if key in ("state", "epoch", "step", "updates", "at")},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--component", choices=("contr", "hier"), required=True)
    args = parser.parse_args()
    out = verify(args.run, args.reference, args.component)
    print(json.dumps(out, allow_nan=False))
    if not out["all_passed"]:
        raise AssertionError([name for name, passed in out["checks"].items() if not passed])


if __name__ == "__main__":
    main()
