"""CPU-only read-only acceptance for an existing v13 first complete epoch.

No forward, optimizer step, CUDA context, run mutation, raw IDs, or private paths
are emitted. Invoke only after epoch_001.pth and the first metric row exist.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def first_json_line(path):
    with path.open(encoding="utf-8") as stream:
        return json.loads(next(line for line in stream if line.strip()))


def finite(value):
    if torch.is_tensor(value):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(finite(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return all(finite(item) for item in value)
    return math.isfinite(value) if isinstance(value, float) else True


def matching_tensors(first, second):
    return first.keys() == second.keys() and all(
        torch.equal(tensor, second[name]) for name, tensor in first.items())


def state_hash(state):
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def active_finite(group):
    return group["status"] == "nonzero" and group["finite"]


def verify(run, reference_run, variant):
    torch.set_num_threads(2)
    if torch.cuda.is_initialized():
        raise AssertionError("Acceptance must not initialize CUDA")
    manifest, reference = read(run / "manifest.json"), read(reference_run / "manifest.json")
    first = first_json_line(run / "metrics.jsonl")
    ref_first = first_json_line(reference_run / "metrics.jsonl")
    geometry = first_json_line(run / "geometry.jsonl")
    e0 = torch.load(run / "e0.pth", map_location="cpu", weights_only=False)
    ref0 = torch.load(reference_run / "e0.pth", map_location="cpu", weights_only=False)
    e1 = torch.load(run / "epoch_001.pth", map_location="cpu", weights_only=False)
    shared = {name: tensor for name, tensor in e0["model"].items()
              if not name.startswith("tail_bn.") and name != "score_bias"}
    config = manifest["config"]
    ignored = {"variant", "expected_initial_sha256",
               "expected_shared_initial_sha256", "config", "out", "physical_gpu"}
    display = {"bnleaky": "TAN-BNLEAKY-R0", "bias": "TAN-BIAS-R0",
               "bnleaky_bias": "TAN-BNLEAKY-BIAS-R0"}[variant]
    has_bn = variant in {"bnleaky", "bnleaky_bias"}
    has_bias = variant in {"bias", "bnleaky_bias"}
    added_parameters = (256 if has_bn else 0) + (50 if has_bias else 0)
    added_state = ({"tail_bn.weight", "tail_bn.bias", "tail_bn.running_mean",
                    "tail_bn.running_var", "tail_bn.num_batches_tracked"} if has_bn else set())
    if has_bias:
        added_state.add("score_bias")
    groups = geometry["gradients"]["ce"]["parameter_groups"]
    gradients = geometry["gradients"]
    nodes = geometry["geometry"]
    q = nodes["final_tangent"]
    diagnostics = geometry["diagnostics"]
    source_commit = manifest["source"]["commit"]
    fixed_curvature = {"manifold.isp_c", "manifold2.isp_c", "manifold3.isp_c"}

    checks = {
        "v13_variant_and_display": manifest["format"] == "hycore_seg_v13"
            and manifest["tail_variant"] == config["variant"] == variant
            and manifest["display_name"] == display,
        "fresh_without_teacher_or_resume": manifest["initialization"] == "fresh"
            and manifest["teacher"] is None and not config.get("resume"),
        "all_shared_e0_tensors_exactly_equal": matching_tensors(shared, ref0["model"]),
        "shared_e0_hash_matches_reference": state_hash(shared)
            == manifest["initial_shared_model_sha256"] == reference["initial_model_sha256"],
        "full_e0_hash_matches_manifest": state_hash(e0["model"])
            == manifest["initial_model_sha256"] == e0["initial_model_sha256"],
        "shared_head_hash_matches_reference": manifest["initial_shared_head_sha256"]
            == reference["initial_shared_head_sha256"],
        "initial_cpu_rng_exactly_equal": torch.equal(e0["rng"]["torch"], ref0["rng"]["torch"]),
        "data_manifest_exactly_equal": manifest["data_manifest_sha256"]
            == reference["data_manifest_sha256"] == e0["data_manifest_sha256"]
            == e1["data_manifest_sha256"],
        "reference_config_equal_except_declared_tail_and_guards":
            {key: value for key, value in config.items() if key not in ignored}
            == {key: value for key, value in reference["config"].items() if key not in ignored},
        "shared_initial_guard_present": config["expected_shared_initial_sha256"]
            == reference["initial_model_sha256"],
        "first_input_exactly_equal": first["first_input_sha256"] == ref_first["first_input_sha256"],
        "first_pairing_exactly_equal": first["first_pairing"] == ref_first["first_pairing"],
        "first_complete_epoch437": first["epoch"] == e1["epoch"] == 1
            and first["steps"] == first["updates"] == e1["updates"] == 437,
        "all_bn_once_per_batch": bool(first["bn_counts"])
            and all(value == 437 for value in first["bn_counts"].values())
            and all(int(value) == 437 for name, value in e1["model"].items()
                    if name.endswith("num_batches_tracked")),
        "test2874_and_50parts": first["test"]["objects"] == 2874
            and len(first["test"]["parts"]) == 50,
        "model_and_optimizer_finite": finite(e1["model"]) and finite(e1["optimizer"])
            and first["model_finite"] and first["optimizer_finite"],
        "no_failure_file": not (run / "failure.json").exists(),
        "training_regularizer_zero": first["train_reg"] == 0.
            and first["train_loss"] == first["train_ce"]
            and not config["regularizers"] and config["alpha"] == config["beta"] == 0.,
        "regularizers_observation_only": not manifest["regularizers_active"]
            and manifest["regularizer_diagnostics_only"]
            and not diagnostics["regularizers_active"]
            and diagnostics["regularizer_diagnostics_only"],
        "both_r_without_training_graph": all(
            not gradients[name]["loss_requires_grad"] and gradients[name]["loss"] == 0.
            and all(node["status"] == "none" for node in gradients[name]["nodes"].values())
            and all(group["status"] == "none" for group in gradients[name]["parameter_groups"].values())
            for name in ("contr", "hier")),
        "both_active_weighted_r_exactly_zero": diagnostics["active_weighted_contr"]
            == diagnostics["active_weighted_hier"] == 0.,
        "scale5_and_plain_ce": manifest["supervision_scale"] == config["supervision_scale"] == 5.
            and manifest["ce"] == "CE(scale*z)/scale_smoothing_false",
        "post_conv9_regularizer_source": manifest["regularizer_feature_location"]
            == "after_conv9_bn9_leaky",
        "no_dropout_or_gradient_clip": not manifest["dropout_active"]
            and not manifest["gradient_clipping"] and not geometry["clip_applied"],
        "real_q_without_ball_interpretation": manifest["final_output_space"] == "real_tangent_scores"
            and q["finite"] and "m3" not in nodes
            and not any(key in q for key in ("radius", "depth", "projection_radius", "near_boundary_fraction")),
        "q_correct_probability_recorded": "scaled_correct_class_probability" in q
            and finite(q["scaled_correct_class_probability"]),
        "ce_graph_reaches_u_frontend_and_both_maps": gradients["ce"]["loss_requires_grad"]
            and active_finite(gradients["ce"]["nodes"]["h"])
            and all(active_finite(groups[name]) for name in ("shared", "euclidean9", "m2", "m3")),
        "ce_reaches_both_mapping_weights": all(
            active_finite(groups[name]["parameters"]["_weight"]) for name in ("m2", "m3")),
        "source_commit_frozen_across_manifest_and_checkpoints": len(source_commit) == 40
            and source_commit == e0["source"]["commit"] == e1["source"]["commit"]
            and not manifest["source"]["source_status"]
            and not e0["source"]["source_status"] and not e1["source"]["source_status"],
        "checkpoint_config_and_schedule_match": e0["config"] == e1["config"] == config
            and e1["scheduler"]["T_max"] == 200 and e1["config"]["epochs"] == 200,
        "cuda_not_initialized": not torch.cuda.is_initialized(),
    }

    if has_bn:
        order = list(nodes)
        sequence = ("m2", "tail_tangent", "tail_bn", "tail_activation", "final_tangent")
        checks.update({
            "bnleaky_only_declared_new_state": set(e0["model"]) - set(ref0["model"])
                == added_state,
            "bnleaky_declared_trainable_parameter_increase": manifest["trainable_parameters"]
                == reference["trainable_parameters"] + added_parameters
                and manifest["head_parameters"] == reference["head_parameters"] + added_parameters,
            "bnleaky_frozen_parameters_unchanged": set(manifest["frozen_parameter_names"]) == fixed_curvature,
            "bn128_initial_state_and_counter": e0["model"]["tail_bn.weight"].numel() == 128
                and torch.equal(e0["model"]["tail_bn.weight"], torch.ones(128))
                and torch.equal(e0["model"]["tail_bn.bias"], torch.zeros(128))
                and torch.equal(e0["model"]["tail_bn.running_mean"], torch.zeros(128))
                and torch.equal(e0["model"]["tail_bn.running_var"], torch.ones(128))
                and int(e0["model"]["tail_bn.num_batches_tracked"]) == 0
                and first["bn_counts"].get("tail_bn") == 437,
            "bnleaky_tangent_nodes_have_expected_position_and_width": all(name in nodes for name in sequence)
                and [order.index(name) for name in sequence] == sorted(order.index(name) for name in sequence)
                and all(nodes[name]["shape"][-1] == 128 and nodes[name]["finite"]
                        for name in ("tail_tangent", "tail_bn", "tail_activation")),
            "bnleaky_new_affine_has_ce_gradients_and_updates": active_finite(groups["tail_bn"])
                and all(active_finite(groups["tail_bn"]["parameters"][name]) for name in ("weight", "bias"))
                and any(not torch.equal(e0["model"]["tail_bn." + name],
                                        e1["model"]["tail_bn." + name]) for name in ("weight", "bias")),
            "bnleaky_retains_last_transport_gradient": active_finite(groups["m3"]["parameters"]["_bias"]),
            "bnleaky_bias_presence_matches_declaration": ("score_bias" in e0["model"]) == has_bias
                and ("readout_bias" in groups) == has_bias,
        })
    if has_bias:
        checks.update({
            "bias_only_declared_new_state": set(e0["model"]) - set(ref0["model"])
                == added_state,
            "bias_declared_trainable_parameter_increase": manifest["trainable_parameters"]
                == reference["trainable_parameters"] + added_parameters
                and manifest["head_parameters"] == reference["head_parameters"] + added_parameters,
            "bias_does_not_freeze_original_parameters": set(manifest["frozen_parameter_names"])
                == fixed_curvature,
            "bias_zero_initial_intercept": e0["model"]["score_bias"].shape == (50,)
                and torch.equal(e0["model"]["score_bias"], torch.zeros(50)),
            "bias_new_intercept_has_ce_gradient": active_finite(groups["readout_bias"])
                and active_finite(groups["readout_bias"]["parameters"]["score_bias"]),
            "bias_updates_and_stays_finite": finite(e1["model"]["score_bias"])
                and not torch.equal(e0["model"]["score_bias"], e1["model"]["score_bias"]),
            "bias_retains_last_transport_gradient": active_finite(groups["m3"]["parameters"]["_bias"]),
            "bias_bn_presence_matches_declaration": ("tail_bn" in nodes) == has_bn
                and ("tail_bn" in first["bn_counts"]) == has_bn
                and "tail_tangent" in nodes and nodes["tail_tangent"]["shape"][-1] == 128,
        })
    status = read(run / "status.json")
    return {
        "all_passed": all(checks.values()), "checks": checks,
        "display_name": display, "variant": variant, "commit": source_commit,
        "initial_model_sha256": manifest["initial_model_sha256"],
        "initial_shared_model_sha256": manifest["initial_shared_model_sha256"],
        "first_input_sha256": first["first_input_sha256"],
        "first_epoch_instance_miou": first["test"]["instance_miou"],
        "first_epoch_class_miou": first["test"]["class_miou"],
        "first_epoch_seconds": first["seconds"],
        "trainable_parameters": manifest["trainable_parameters"],
        "first_q_range": [q["logit_min"], q["logit_max"]],
        "geometry_nodes": list(nodes),
        "current_status": {key: value for key, value in status.items()
                           if key in ("state", "epoch", "step", "updates", "at")},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--variant", choices=("bnleaky", "bias", "bnleaky_bias"), required=True)
    args = parser.parse_args()
    result = verify(args.run, args.reference, args.variant)
    print(json.dumps(result, allow_nan=False))
    if not result["all_passed"]:
        raise AssertionError([name for name, passed in result["checks"].items() if not passed])


if __name__ == "__main__":
    main()
