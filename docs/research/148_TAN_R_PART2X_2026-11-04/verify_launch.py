"""Read-only, CPU first-epoch acceptance for the TAN-R part-size experiment."""

import argparse
import json
import math
from pathlib import Path

import torch


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def records(path):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def finite(value):
    if torch.is_tensor(value):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite(item) for item in value)
    return not isinstance(value, float) or math.isfinite(value)


def sampling_matches(diagnostics):
    points = diagnostics["part_points"]
    return (
        400 <= points <= 1200
        and diagnostics["part_indices_shape"][1] == points
        and math.isclose(diagnostics["gamma"], 2000.0 / points, rel_tol=1e-6)
        and diagnostics["delta"] == 4.0
        and diagnostics["regularizers_active"]
        and not diagnostics["regularizer_diagnostics_only"]
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--reference", required=True, type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)

    manifest = read(args.run / "manifest.json")
    reference = read(args.reference / "manifest.json")
    first = next(records(args.run / "metrics.jsonl"))
    ref_first = next(records(args.reference / "metrics.jsonl"))
    initial = torch.load(args.run / "e0.pth", map_location="cpu", weights_only=False)
    ref_initial = torch.load(args.reference / "e0.pth", map_location="cpu", weights_only=False)
    epoch_one = torch.load(args.run / "epoch_001.pth", map_location="cpu", weights_only=False)

    config = manifest["config"]
    ref_config = reference["config"]
    ignored = {"config", "out", "physical_gpu", "part_min", "part_max", "gamma_base"}
    pair = first["first_pairing"]
    ref_pair = ref_first["first_pairing"]
    # Both logs may already contain later epochs by the time acceptance runs.
    geometry_records = [row for row in records(args.run / "geometry.jsonl") if row["epoch"] == 1]
    step_records = [row for row in records(args.run / "steps.jsonl") if row["epoch"] == 1]
    geometry = geometry_records[0]
    diagnostics = geometry["diagnostics"]
    gradients = geometry["gradients"]
    q = geometry["geometry"]["final_tangent"]

    checks = {
        "reference_is_original_tan_r": reference["display_name"] == "S5-TAN-R"
        and ref_config["regularizers"]
        and (ref_config["part_min"], ref_config["part_max"], ref_config["gamma_base"]) == (200, 600, 1000.0),
        "target_part_and_margins": (config["num_points"], config["part_min"], config["part_max"],
                                    config["gamma_base"], config["delta"]) == (2048, 400, 1200, 2000.0, 4.0),
        "reference_config_equal_except_part_scale_and_paths": {
            key: value for key, value in config.items() if key not in ignored
        } == {
            key: value for key, value in ref_config.items() if key not in ignored
        },
        "initial_hash_matches_reference": manifest["initial_model_sha256"] == reference["initial_model_sha256"],
        "all_initial_model_tensors_equal": initial["model"].keys() == ref_initial["model"].keys()
        and all(torch.equal(tensor, ref_initial["model"][key]) for key, tensor in initial["model"].items()),
        "initial_cpu_rng_equal": torch.equal(initial["rng"]["torch"], ref_initial["rng"]["torch"]),
        "initial_sampling_generators_equal": all(torch.equal(initial[key], ref_initial[key]) for key in (
            "train_generator", "eval_generator", "part_generator", "negative_generator"
        )),
        "data_manifest_equal": manifest["data_manifest_sha256"] == reference["data_manifest_sha256"],
        "first_input_equal": first["first_input_sha256"] == ref_first["first_input_sha256"],
        "first_centers_and_negatives_equal": all(pair[key] == ref_pair[key] for key in (
            "part_centers", "negative_indices"
        )),
        "first_geometry_and_pairing_agree": all(pair[key] == diagnostics[key] for key in (
            "part_centers", "negative_indices", "part_points", "part_indices_sha256"
        )),
        "first_epoch_sampling_and_margins": bool(step_records)
        and all(sampling_matches(row["diagnostics"]) for row in geometry_records)
        and all(sampling_matches(row) for row in step_records),
        "first_epoch437": first["epoch"] == 1 and first["steps"] == 437 and first["updates"] == 437
        and epoch_one["epoch"] == 1 and epoch_one["updates"] == 437,
        "bn437": bool(first["bn_counts"]) and all(value == 437 for value in first["bn_counts"].values()),
        "test2874_and_50_parts": first["test"]["objects"] == 2874 and len(first["test"]["parts"]) == 50,
        "both_regularizers_enabled": manifest["regularizers_active"] and not manifest["regularizer_diagnostics_only"]
        and config["regularizers"] and config["alpha"] == config["beta"] == 0.01,
        "finite_ce_and_regularizer": all(math.isfinite(first[key]) and first[key] > 0
                                        for key in ("train_ce", "train_reg", "train_loss"))
        and math.isclose(first["train_loss"], first["train_ce"] + first["train_reg"], rel_tol=1e-6),
        "tangent_scores_and_scale5": manifest["final_output_space"] == "real_tangent_scores"
        and manifest["supervision_scale"] == 5.0 and q["finite"],
        "ce_and_both_regularizer_graphs": set(gradients) == {"ce", "contr", "hier"}
        and all(gradients[key]["loss_requires_grad"] and gradients[key]["nodes"]["h"]["finite"]
                for key in ("ce", "contr", "hier")),
        "regularizers_bypass_last_maps": all(gradients[key]["parameter_groups"][group]["status"] == "none"
                                             for key in ("contr", "hier") for group in ("m2", "m3")),
        "model_and_optimizer_finite": first["model_finite"] and first["optimizer_finite"]
        and finite(epoch_one["model"]) and finite(epoch_one["optimizer"]),
        "no_failure_file": not (args.run / "failure.json").exists(),
    }
    status = read(args.run / "status.json")
    result = {
        "checks": checks,
        "all_passed": all(checks.values()),
        "commit": manifest["source"]["commit"],
        "first_epoch_seconds": first["seconds"],
        "first_epoch_instance_miou": first["test"]["instance_miou"],
        "first_epoch_class_miou": first["test"]["class_miou"],
        "first_sample": {key: diagnostics[key] for key in ("part_points", "gamma", "delta")},
        "current_status": {key: status[key] for key in ("state", "epoch", "step", "updates", "at") if key in status},
    }
    print(json.dumps(result, allow_nan=False))
    return 0 if result["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
