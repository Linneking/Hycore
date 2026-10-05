"""Fixed-triplet Gumbel noise controls from saved whole-mechanism probes.

No backbone, optimizer, GPU-sized activation graphs or training update.
Every condition uses exactly the reference ordered triplet rows. Within a
condition, independent repeated seeds measure stochastic choice variability;
across conditions, paired seeds hold Gumbel noise fixed. Natural mined graphs
are reported separately and never substituted into these controlled draws.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
from datetime import datetime, timezone
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

from hier_whole_mechanism_v1 import (
    ROOT, NAMES, array_hash, digest, fixed_ancestor_loss, radial_partials, save_json,
)
from inter_hierarchy_MN40.hier_proxy_scratch_v5.relations import poincare_distance
from inter_hierarchy_MN40.hier_proxy_scratch_v6.hier_loss import HIERLoss, ghhc_loss
from hier_proxy_objective_probe_v1 import (
    CHECKPOINT_FORMAT, SOURCE_PROTOCOL, validate_checkpoint_protocol,
)


COMPONENTS = ("sample_st", "sample_fixed", "selection_path")


def state_map_hash(state):
    """Same semantic state_dict hash as the source whole mechanism probe."""
    result = hashlib.sha256()
    for name, value in state.items():
        result.update(name.encode())
        result.update(array_hash(value).encode())
    return result.hexdigest()


def checkpoint_identity(saved, checkpoint_sha):
    validate_checkpoint_protocol(saved)
    required = {"net", "proxy", "optimizer", "proxy_optimizer", "scheduler", "proxy_scheduler",
                "rank_states", "train_ids", "validation_ids", "training_config", "split_sha256",
                "epoch", "completed_epochs"}
    if not required.issubset(saved) or saved.get("diagnostic_only", False):
        raise RuntimeError("Require a complete production V6-H20 checkpoint, including optimizers/rank states")
    if type(saved["completed_epochs"]) is not int or saved["completed_epochs"] != saved["epoch"]:
        raise RuntimeError("Invalid complete-checkpoint epoch identity")
    if len(saved["rank_states"]) != 2 or [row.get("rank") for row in saved["rank_states"]] != [0, 1]:
        raise RuntimeError("Require both ordered V6 rank states")
    if saved["training_config"].get("lambda_hier_after_warmup") != .5:
        raise RuntimeError("Expected source HIER weight .5")
    train_ids = np.asarray(saved["train_ids"], dtype=np.int64)
    if train_ids.shape != (8856,) or len(np.unique(train_ids)) != 8856:
        raise RuntimeError("Expected original distinct V6 train8856 IDs")
    return {"checkpoint_sha256": checkpoint_sha, "checkpoint_format": CHECKPOINT_FORMAT,
            "checkpoint_epoch": saved["epoch"], "checkpoint_source_commit": saved.get("commit"),
            "split_sha256": saved["split_sha256"], "train_ids_sha256": array_hash(train_ids),
            "backbone_initial_sha256": state_map_hash(saved["net"]),
            "proxy_initial_sha256": state_map_hash(saved["proxy"])}


def validate_probe_identity(manifest, source_identity):
    if manifest.get("status") != "completed" or manifest.get("optimizer_updates") != 0:
        raise RuntimeError("Source whole probe must be completed and contain no optimizer updates")
    if manifest.get("diagnostic_only") is not True or manifest.get("global_batch") != 64 or manifest.get("local_forward_batch") != 32:
        raise RuntimeError("Require the saved read-only global64/local32 whole probe")
    for field, expected in source_identity.items():
        if manifest.get(field) != expected:
            raise RuntimeError("Source probe/checkpoint identity mismatch: " + field)


def cosine(a, b):
    a, b = np.asarray(a, dtype=np.float64).ravel(), np.asarray(b, dtype=np.float64).ravel()
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denominator) if denominator else None


def finite(value, name):
    if not np.isfinite(value).all():
        raise RuntimeError("Nonfinite " + name)


def one_draw(mu, proxies, triplets, seed, tau=.1, margin=.1):
    point = mu.detach().clone().requires_grad_(True)
    distance = poincare_distance(point, proxies.detach())
    generator = torch.Generator(device=point.device).manual_seed(int(seed))
    actual, stats, details = ghhc_loss(distance, triplets, tau=tau, margin=margin,
                                     generator=generator, return_details=True)
    direct = fixed_ancestor_loss(distance, triplets, details["pair_proxy_idx"].detach(),
                                details["triple_proxy_idx"].detach(), margin)
    if not torch.allclose(actual.detach(), direct.detach(), atol=2e-6, rtol=2e-5):
        raise RuntimeError("Straight-through/fixed-ancestor scalar mismatch")
    st = torch.autograd.grad(.5 * actual, point, retain_graph=True)[0].detach()
    fixed = torch.autograd.grad(.5 * direct, point)[0].detach()
    result = {"seed": int(seed), "loss": .5 * float(actual.detach()), "stats": stats,
              "pair": details["pair_proxy_idx"].detach().cpu().numpy(),
              "triple": details["triple_proxy_idx"].detach().cpu().numpy(),
              "active": (details["per_draw_loss"].detach() > 0).cpu().numpy(),
              "gradients": {"sample_st": st.cpu().numpy(), "sample_fixed": fixed.cpu().numpy(),
                            "selection_path": (st - fixed).cpu().numpy()}, "partials": {}}
    for name, gradient in (("sample_st", st), ("sample_fixed", fixed), ("selection_path", st - fixed)):
        result["partials"][name] = {key: value.detach().cpu().numpy() for key, value in radial_partials(point, gradient).items()}
        finite(result["gradients"][name], name + " gradient")
    return result


def agreement(first, second):
    if len(first["pair"]) != len(second["pair"]):
        raise RuntimeError("Agreement requires identical ordered draw count")
    domains = {"all_draws": np.ones(len(first["pair"]), dtype=bool),
               "both_noncollision": (first["pair"] != first["triple"]) & (second["pair"] != second["triple"]),
               "reference_noncollision": first["pair"] != first["triple"]}
    result = {}
    for name, mask in domains.items():
        count = int(mask.sum())
        result[name] = {"draws": count,
                        "pair": float((first["pair"][mask] == second["pair"][mask]).mean()) if count else None,
                        "triple": float((first["triple"][mask] == second["triple"][mask]).mean()) if count else None,
                        "both_roles": float(((first["pair"][mask] == second["pair"][mask]) &
                                               (first["triple"][mask] == second["triple"][mask])).mean()) if count else None}
    return result


def pooled_agreements(comparisons):
    result = {}
    for domain in ("all_draws", "both_noncollision", "reference_noncollision"):
        total = sum(item[domain]["draws"] for item in comparisons)
        row = {"comparisons": len(comparisons), "comparison_draw_slots": total}
        for role in ("pair", "triple", "both_roles"):
            values = [item[domain][role] for item in comparisons if item[domain]["draws"]]
            row[role + "_pooled_agreement"] = sum(item[domain][role] * item[domain]["draws"] for item in comparisons if item[domain]["draws"]) / total if total else None
            row[role + "_comparison_mean"] = float(np.mean(values)) if values else None
            row[role + "_comparison_sd"] = float(np.std(values)) if values else None
            row[role + "_comparison_range"] = [float(np.min(values)), float(np.max(values))] if values else None
        result[domain] = row
    return result


def noise_summary(draws, labels, ids, hotspot_ids):
    groups = {"all": np.ones(len(labels), dtype=bool)}
    for label in np.unique(labels):
        groups["class/" + NAMES[int(label)]] = labels == label
    if hotspot_ids:
        groups["hotspots"] = np.isin(ids, hotspot_ids)
    result = {}
    for component in COMPONENTS:
        gradients = np.stack([row["gradients"][component] for row in draws]).astype(np.float64)
        radial = np.stack([row["partials"][component]["radial"] for row in draws]).astype(np.float64)
        angular_norm = np.stack([row["partials"][component]["angular_norm"] for row in draws]).astype(np.float64)
        # Compute vector noise with the condition's fixed radial decomposition.
        # The Euclidean coordinate covariance is not a transported tangent covariance.
        norm = np.linalg.norm(gradients, axis=-1)
        mean_gradient = gradients.mean(0)
        gradient_noise = np.sqrt(np.mean(np.sum((gradients - mean_gradient[None]) ** 2, axis=-1), axis=0))
        # For a fixed condition each point has one radial direction, so the
        # coordinate variance trace splits exactly into radial and angular.
        angular_vector_noise = np.sqrt(np.maximum(gradient_noise ** 2 - radial.var(0), 0))
        result[component] = {}
        for name, mask in groups.items():
            count = int(mask.sum())
            result[component][name] = {"positions": count,
                "mean_radial_partial": float(radial[:, mask].mean()) if count else None,
                "radial_noise_sd_mean": float(radial[:, mask].std(0).mean()) if count else None,
                "positive_radial_fraction_over_noise": float((radial[:, mask] > 0).mean()) if count else None,
                "negative_radial_fraction_over_noise": float((radial[:, mask] < 0).mean()) if count else None,
                "all_noise_radial_sign_same_fraction": float(((radial[:, mask] > 0).all(0) | (radial[:, mask] < 0).all(0)).mean()) if count else None,
                "all_noise_zero_radial_fraction": float((radial[:, mask] == 0).all(0).mean()) if count else None,
                "angular_partial_norm_mean": float(angular_norm[:, mask].mean()) if count else None,
                "angular_norm_noise_sd_mean": float(angular_norm[:, mask].std(0).mean()) if count else None,
                "angular_vector_noise_rms_mean": float(angular_vector_noise[mask].mean()) if count else None,
                "gradient_mean_vector_norm_mean": float(np.linalg.norm(mean_gradient[mask], axis=-1).mean()) if count else None,
                "gradient_vector_noise_rms_mean": float(gradient_noise[mask].mean()) if count else None,
                "nonzero_partial_fraction_over_noise": float((norm[:, mask] > 0).mean()) if count else None}
        result[component]["within_condition_coordinate_gradient_cosine"] = [cosine(draws[i]["gradients"][component], draws[j]["gradients"][component])
                                                                          for i, j in itertools.combinations(range(len(draws)), 2)]
    return result


def paired_gradient_summary(reference, current):
    result = {}
    for component in COMPONENTS:
        values = []
        for left, right in zip(reference, current):
            old, new = left["partials"][component]["radial"], right["partials"][component]["radial"]
            values.append({"seed": left["seed"],
                "coordinate_gradient_cosine": cosine(left["gradients"][component], right["gradients"][component]),
                "radial_difference_mean": float((new - old).mean()),
                "radial_absolute_difference_mean": float(np.abs(new - old).mean()),
                "radial_sign_agreement": float((np.sign(new) == np.sign(old)).mean()),
                "angular_norm_absolute_difference_mean": float(np.abs(right["partials"][component]["angular_norm"] -
                                                                        left["partials"][component]["angular_norm"]).mean())})
        result[component] = values
    return result


def load_probe(directory, batch, requested_conditions, source_identity):
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_probe_identity(manifest, source_identity)
    report_path = directory / f"batch_{batch:03d}.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    available = list(report["conditions"])
    conditions = requested_conditions or available
    if any(condition not in available for condition in conditions):
        raise RuntimeError("Requested condition absent: " + str(directory))
    files = {str(manifest_path): digest(manifest_path), str(report_path): digest(report_path)}
    rows = {}
    for condition in conditions:
        path = directory / f"batch_{batch:03d}_{condition}.npz"
        with np.load(path, allow_pickle=False) as arrays:
            row = {key: arrays[key].copy() for key in ("mu", "ids", "labels", "triplets", "mutual")}
            for key in ("physical_whole_anchor", "physical_child_anchor", "whole_preoverwrite_original_ids",
                        "whole_postoverwrite_original_ids", "child_original_ids", "original_point_permutation"):
                if key in arrays:
                    row[key] = arrays[key].copy()
        if not np.array_equal(row["ids"], np.asarray(report["ids"])) or not np.array_equal(row["labels"], np.asarray(report["labels"])):
            raise RuntimeError("NPZ/report ID or label mismatch")
        if row["mu"].shape != (64, 256) or row["mutual"].shape != (64, 64):
            raise RuntimeError("Expected saved global64/D256 probe")
        finite(row["mu"], "saved mu")
        if np.any(np.sum(row["mu"].astype(np.float64) ** 2, axis=-1) >= 1):
            raise RuntimeError("Saved whole outside c1 ball")
        row["source_condition"] = report["conditions"][condition]
        row["source_path"] = str(path)
        files[str(path)] = digest(path)
        rows[condition] = row
    return manifest, report, rows, files


def validate_pairing(first_manifest, first_report, other_manifest, other_report, allow_augmentation):
    for name in ("checkpoint_sha256", "backbone_initial_sha256", "proxy_initial_sha256", "train_ids_sha256",
                 "input_shard_sha256", "labels_sha256", "input_mode", "seed", "plan_epoch", "hotspot_ids",
                 "augmentation_family", "crop_policy"):
        if first_manifest.get(name) != other_manifest.get(name):
            raise RuntimeError("Cross-probe control mismatch: " + name)
    for name in ("ids", "labels", "global_flip_negative_ids", "whole_points", "part_points", "whole_centers", "part_centers"):
        if first_report.get(name) != other_report.get(name):
            raise RuntimeError("Cross-probe panel mismatch: " + name)
    changed_input = first_report["input_cloud_sha256"] != other_report["input_cloud_sha256"]
    if changed_input and not allow_augmentation:
        raise RuntimeError("Input cloud changed; explicit --allow-augmentation required")
    if changed_input and first_manifest.get("input_mode") != "augmented":
        raise RuntimeError("Augmentation comparison requires augmented input mode")
    return {"input_cloud_changed": changed_input,
            "comparison": "different augmented input views with fixed IDs/numeric center indices/counts/negative mapping; physical anchors/members reported separately" if changed_input else "identical input cloud",
            "input_view_metadata": [first_manifest.get("input_view"), other_manifest.get("input_view")],
            "augmentation_view_field_present": "input_view" in first_manifest and "input_view" in other_manifest}


def source_graph_summary(reference, current):
    old, new = reference["mutual"].astype(bool), current["mutual"].astype(bool)
    union = np.count_nonzero(old | new)
    result = {"saved_natural_mutual_graph_jaccard": int(np.count_nonzero(old & new)) / int(union) if union else None,
            "saved_natural_triplets_sha256": array_hash(current["triplets"]),
            "saved_natural_triplet_draws": len(current["triplets"]),
            "saved_natural_mining": current["source_condition"]["mining"],
            "semantics": "Saved natural graph/draws reported separately; controlled Gumbel gradients use only the common reference ordered triplets."}
    physical = {}
    for key in ("physical_whole_anchor", "physical_child_anchor"):
        if key in reference and key in current:
            physical[key + "_agreement"] = float((reference[key] == current[key]).mean())
    for key in ("whole_preoverwrite_original_ids", "whole_postoverwrite_original_ids", "child_original_ids"):
        if key in reference and key in current:
            values = []
            for left, right in zip(reference[key], current[key]):
                a, b = set(left.tolist()), set(right.tolist())
                values.append(len(a & b) / len(a | b))
            physical[key + "_jaccard_mean"] = float(np.mean(values))
    if physical:
        result["physical_crop_controls"] = physical
    return result


def write_gradient_rows(writer, batch, name, draws, ids, labels, hotspot_ids):
    for repetition, draw in enumerate(draws):
        for component, partial in draw["partials"].items():
            for position in range(64):
                writer.writerow({"batch": batch, "condition": name, "repetition": repetition, "noise_seed": draw["seed"],
                                 "position": position, "id": int(ids[position]), "class": NAMES[int(labels[position])],
                                 "hotspot": int(ids[position] in hotspot_ids), "component": component,
                                 "radial_partial": float(partial["radial"][position]),
                                 "angular_partial_norm": float(partial["angular_norm"][position]),
                                 "partial_wrt_depth": float(partial["partial_wrt_depth"][position]),
                                 "partial_norm": float(partial["norm"][position])})


def self_test():
    torch.set_num_threads(2)
    generator = torch.Generator().manual_seed(7)
    mu = torch.randn(64, 4, generator=generator) * .07
    proxy = torch.randn(7, 4, generator=generator) * .08
    triplets = torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8]] * 24)
    first = one_draw(mu, proxy, triplets, 17, tau=.6)
    repeated = one_draw(mu, proxy, triplets, 17, tau=.6)
    assert np.array_equal(first["pair"], repeated["pair"]) and np.array_equal(first["triple"], repeated["triple"])
    for component in COMPONENTS:
        np.testing.assert_allclose(first["gradients"][component], repeated["gradients"][component], atol=1e-7, rtol=1e-6)
    independent = [one_draw(mu, proxy, triplets, 17 + 1009 * repetition, tau=.6) for repetition in range(4)]
    control = pooled_agreements([agreement(first, repeated)])
    assert control["all_draws"]["pair_pooled_agreement"] == 1
    assert control["all_draws"]["triple_pooled_agreement"] == 1
    noise = pooled_agreements([agreement(independent[i], independent[j]) for i, j in itertools.combinations(range(4), 2)])
    assert noise["all_draws"]["pair_pooled_agreement"] < 1 and noise["all_draws"]["triple_pooled_agreement"] < 1
    label = np.arange(64) // 2
    summary = noise_summary(independent, label, np.arange(64), [])
    assert summary["sample_st"]["all"]["gradient_vector_noise_rms_mean"] > 0
    changed = one_draw(mu * 1.3, proxy, triplets, 17, tau=.6)
    agreement(first, changed)
    valid = {"format": CHECKPOINT_FORMAT, "model_selection_only": False, "epoch": 300, "completed_epochs": 300,
             "training_config": {**SOURCE_PROTOCOL, "self_negative": False, "extra_HIER_tangent_cap": False,
                                 "HIER_backward_hook": False, "smoke": False, "lambda_hier_after_warmup": .5},
             "net": {"weight": torch.ones(2)}, "proxy": {"tangent_proxies": torch.ones(512, 256)},
             "optimizer": {}, "proxy_optimizer": {}, "scheduler": {}, "proxy_scheduler": {},
             "rank_states": [{"rank": 0}, {"rank": 1}], "train_ids": np.arange(8856),
             "validation_ids": np.arange(8856, 9840), "split_sha256": "synthetic", "commit": "synthetic"}
    identity = checkpoint_identity(valid, "synthetic-sha")
    source_probe = {**identity, "status": "completed", "optimizer_updates": 0, "diagnostic_only": True,
                    "global_batch": 64, "local_forward_batch": 32}
    validate_probe_identity(source_probe, identity)
    for field, bad in (("format", "legacy"), ("completed_epochs", 299), ("model_selection_only", True)):
        wrong = copy.deepcopy(valid)
        wrong[field] = bad
        try:
            checkpoint_identity(wrong, "synthetic-sha")
        except (RuntimeError, ValueError):
            pass
        else:
            raise AssertionError("Checkpoint identity gate failed to reject " + field)
    for field, bad in (("c", .1), ("D", 128), ("sample_K", 10), ("self_negative", True), ("HIER_backward_hook", True), ("smoke", True)):
        wrong = copy.deepcopy(valid)
        wrong["training_config"][field] = bad
        try:
            checkpoint_identity(wrong, "synthetic-sha")
        except (RuntimeError, ValueError):
            pass
        else:
            raise AssertionError("Protocol gate failed to reject " + field)
    for field, bad in (("checkpoint_epoch", 299), ("checkpoint_format", "legacy"), ("status", "running"), ("optimizer_updates", 1)):
        wrong = {**source_probe, field: bad}
        try:
            validate_probe_identity(wrong, identity)
        except RuntimeError:
            pass
        else:
            raise AssertionError("Probe identity gate failed to reject " + field)
    print(json.dumps({"self_test": "passed", "checks": ["identical seed/input selects identical ancestors and gradients",
        "independent seed noise baseline", "fixed/direct scalar equivalence", "finite gradient variance", "paired changed geometry",
        "complete V6 format/protocol gates", "matching source probe epoch/format/hash gates"],
        "noise_baseline": noise["all_draws"]}))


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--probe-dir", action="append", type=Path)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--gpu", type=int, default=0, help="Logical index in CUDA_VISIBLE_DEVICES; caller checks physical idle GPU")
    parser.add_argument("--batches", type=int, default=2)
    parser.add_argument("--batch-start", type=int, default=0)
    parser.add_argument("--conditions", default="", help="Comma separated; empty selects saved available conditions")
    parser.add_argument("--reference-condition", default="eval")
    parser.add_argument("--repeats", type=int, choices=(8, 16), default=8)
    parser.add_argument("--noise-seed", type=int, help="Defaults to each reference panel's source relation_seed")
    parser.add_argument("--tau", type=float, default=.1)
    parser.add_argument("--margin", type=float, default=.1)
    parser.add_argument("--allow-augmentation", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if not args.self_test:
        if not all((args.checkpoint, args.probe_dir, args.run_dir)):
            parser.error("checkpoint, probe-dir and run-dir required")
        if not 1 <= args.batches <= 32 or args.batch_start < 0 or args.tau <= 0 or args.margin < 0:
            parser.error("Invalid batch or loss settings")
        args.conditions = [value.strip() for value in args.conditions.split(",") if value.strip()]
    return args


def main():
    args = arguments()
    if args.self_test:
        self_test()
        return
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    if torch.distributed.is_initialized():
        raise RuntimeError("Single-process diagnostic only")
    args.run_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    checkpoint_sha = digest(args.checkpoint)
    manifest = {"status": "starting", "diagnostic_only": True, "optimizer_updates": 0, "backbone_forwards": 0,
                "checkpoint": str(args.checkpoint), "checkpoint_sha256": checkpoint_sha,
                "script_sha256": digest(__file__), "utility_script_sha256": digest(Path(__file__).with_name("hier_whole_mechanism_v1.py")),
                "probe_directories": [str(value) for value in args.probe_dir], "command": [sys.executable, *sys.argv],
                "started_utc": datetime.now(timezone.utc).isoformat(), "device": args.device, "logical_gpu": args.gpu,
                "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"), "torch_version": torch.__version__,
                "repeats": args.repeats, "tau": args.tau, "margin": args.margin, "lambda_sample": .5,
                "conditions": args.conditions, "reference_condition": args.reference_condition,
                "allow_augmentation": args.allow_augmentation,
                "semantics": ["Pair and triple selections use separate sequential Exp(1) noise draws from each dedicated seeded Generator.",
                              "All conditions use common ordered triplet rows, before collision masking, including the same repeated draws.",
                              "Within-input different-seed comparisons are the stochastic baseline; across-condition same-seed comparisons hold noise fixed.",
                              "Eight independent noise draws yield 28 overlapping pairwise comparisons; those 28 comparisons are dependent summaries, not 28 independent repetitions or a confidence interval.",
                              "Natural saved mining graphs are reported separately, and changing eligible anchors never changes controlled draw rows.",
                              "Gradient variances are Euclidean mu-coordinate partials, before classifier/backbone VJP, optimizer geometry, clipping or momentum.",
                              "BN and noise instability percentages must not be subtracted as additive causal effects."]}
    try:
        manifest["commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        manifest["commit"] = None
    save_json(args.run_dir / "manifest.json", manifest)
    try:
        if args.device == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA unavailable; use --device cpu")
            torch.cuda.set_device(args.gpu)
            device = torch.device("cuda", args.gpu)
            manifest["gpu_name"] = torch.cuda.get_device_name(device)
            torch.cuda.reset_peak_memory_stats(device)
        else:
            device = torch.device("cpu")
        saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        source_identity = checkpoint_identity(saved, checkpoint_sha)
        proxy = HIERLoss().to(device)
        proxy.load_state_dict(saved["proxy"], strict=True)
        with torch.no_grad():
            proxy_points = proxy.proxies().detach()
        manifest.update(status="running", checkpoint_epoch=int(saved.get("completed_epochs", saved.get("epoch", -1))),
                        proxy_points_sha256=array_hash(proxy_points), checkpoint_format=saved.get("format"),
                        checkpoint_source_commit=saved.get("commit"), source_training_config=saved["training_config"],
                        loss_uses_source_tau_margin=args.tau == saved["training_config"]["tau"] and args.margin == saved["training_config"]["margin"])
        del saved
        save_json(args.run_dir / "manifest.json", manifest)
        source_hashes, all_reports = {}, []
        fieldnames = ["batch", "condition", "repetition", "noise_seed", "position", "id", "class", "hotspot",
                      "component", "radial_partial", "angular_partial_norm", "partial_wrt_depth", "partial_norm"]
        with (args.run_dir / "gradient_noise_partials.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            for batch in range(args.batch_start, args.batch_start + args.batches):
                probe_rows = [load_probe(directory, batch, args.conditions, source_identity) for directory in args.probe_dir]
                reference_manifest, reference_report, reference_rows, _ = probe_rows[0]
                if args.reference_condition not in reference_rows:
                    raise RuntimeError("Reference condition absent from first probe")
                ref = reference_rows[args.reference_condition]
                fixed_triplets_np = ref["triplets"].astype(np.int64)
                if len(fixed_triplets_np) and (np.any(fixed_triplets_np < 0) or np.any(fixed_triplets_np >= 64) or
                                                np.any(fixed_triplets_np[:, 0] == fixed_triplets_np[:, 2])):
                    raise RuntimeError("Invalid or self-k reference triplets")
                fixed_triplets = torch.from_numpy(fixed_triplets_np).to(device)
                noise_seed = reference_report["relation_seed"] if args.noise_seed is None else args.noise_seed
                seeds = [int(noise_seed) + repetition * 1000003 for repetition in range(args.repeats)]
                batch_report = {"batch": batch, "fixed_ordered_triplets_sha256": array_hash(fixed_triplets_np),
                                "controlled_draws": len(fixed_triplets_np), "noise_seeds": seeds,
                                "reference_condition": "probe0/" + args.reference_condition,
                                "ids_sha256": array_hash(ref["ids"]), "ids": ref["ids"].tolist(), "labels": ref["labels"].tolist(),
                                "reference_input_cloud_sha256": reference_report["input_cloud_sha256"],
                                "conditions": {}, "cross_condition": {}}
                all_draws = {}
                for probe_number, (probe_manifest, probe_report, rows, files) in enumerate(probe_rows):
                    source_hashes.update(files)
                    pairing = validate_pairing(reference_manifest, reference_report, probe_manifest, probe_report, args.allow_augmentation)
                    hotspot_ids = probe_manifest.get("hotspot_ids", [])
                    for condition, row in rows.items():
                        if not np.array_equal(row["ids"], ref["ids"]) or not np.array_equal(row["labels"], ref["labels"]):
                            raise RuntimeError("Condition global order mismatch")
                        name = f"probe{probe_number}/{condition}"
                        mu = torch.from_numpy(row["mu"]).to(device)
                        draws = [one_draw(mu, proxy_points, fixed_triplets, seed, args.tau, args.margin) for seed in seeds]
                        repeated = one_draw(mu, proxy_points, fixed_triplets, seeds[0], args.tau, args.margin)
                        if not np.array_equal(draws[0]["pair"], repeated["pair"]) or not np.array_equal(draws[0]["triple"], repeated["triple"]):
                            raise RuntimeError("Same input/seed ancestor repeat mismatch")
                        for component in COMPONENTS:
                            np.testing.assert_allclose(draws[0]["gradients"][component], repeated["gradients"][component], atol=1e-4, rtol=2e-4)
                        baseline = [agreement(draws[i], draws[j]) for i, j in itertools.combinations(range(args.repeats), 2)]
                        condition_report = {"pairing_controls": pairing, "source_npz": row["source_path"],
                            "mu_sha256": array_hash(row["mu"]), "natural_graph": source_graph_summary(ref, row),
                            "same_input_same_seed_repeat": {"ancestors_identical": True, "gradients_close": True},
                            "same_input_independent_noise_baseline": pooled_agreements(baseline),
                            "noise_repetitions": args.repeats, "dependent_pairwise_comparisons": len(baseline),
                            "pairwise_uncertainty_semantics": "Pairwise comparisons reuse the same noise repetitions; SD/range describe comparisons, not independent-sample standard errors.",
                            "loss_mean": float(np.mean([draw["loss"] for draw in draws])),
                            "loss_noise_sd": float(np.std([draw["loss"] for draw in draws])),
                            "per_repetition": [{"seed": draw["seed"], "loss": draw["loss"], "sample": draw["stats"],
                                "coordinate_gradient_norms": {component: float(np.linalg.norm(draw["gradients"][component])) for component in COMPONENTS},
                                "st_vs_fixed_norm_ratio": float(np.linalg.norm(draw["gradients"]["sample_st"]) / np.linalg.norm(draw["gradients"]["sample_fixed"])) if np.linalg.norm(draw["gradients"]["sample_fixed"]) else None,
                                "st_vs_fixed_coordinate_cosine": cosine(draw["gradients"]["sample_st"], draw["gradients"]["sample_fixed"])} for draw in draws],
                            "gradient_noise": noise_summary(draws, row["labels"], row["ids"], hotspot_ids)}
                        write_gradient_rows(writer, batch, name, draws, row["ids"], row["labels"], hotspot_ids)
                        arrays = {"mu": row["mu"], "ids": row["ids"], "labels": row["labels"], "ordered_triplets": fixed_triplets_np,
                                  "noise_seeds": np.asarray(seeds, dtype=np.int64),
                                  "pair_proxy_idx": np.stack([draw["pair"] for draw in draws]),
                                  "triple_proxy_idx": np.stack([draw["triple"] for draw in draws])}
                        for component in COMPONENTS:
                            arrays[component + "_mu_partial"] = np.stack([draw["gradients"][component] for draw in draws])
                        np.savez_compressed(args.run_dir / f"batch_{batch:03d}_probe{probe_number}_{condition}.npz", **arrays)
                        batch_report["conditions"][name] = condition_report
                        all_draws[name] = draws
                        print(json.dumps({"batch": batch, "condition": name, "status": "passed",
                            "noise_pair_agreement": condition_report["same_input_independent_noise_baseline"]["all_draws"]["pair_pooled_agreement"],
                            "noise_triple_agreement": condition_report["same_input_independent_noise_baseline"]["all_draws"]["triple_pooled_agreement"]}), flush=True)
                reference_name = batch_report["reference_condition"]
                for name, draws in all_draws.items():
                    if name == reference_name:
                        continue
                    batch_report["cross_condition"][name] = {
                        "same_seed_paired_ancestor_agreement": pooled_agreements([agreement(left, right) for left, right in zip(all_draws[reference_name], draws)]),
                        "paired_coordinate_gradient_changes": paired_gradient_summary(all_draws[reference_name], draws),
                        "noise_control": "Same dedicated noise seed and fixed ordered draw rows; pair and triple noises remain independent."}
                save_json(args.run_dir / f"batch_{batch:03d}.json", batch_report)
                all_reports.append(batch_report)
                stream.flush()
                manifest["completed_batches"] = len(all_reports)
                save_json(args.run_dir / "manifest.json", manifest)
        if digest(args.checkpoint) != checkpoint_sha:
            raise RuntimeError("Checkpoint changed during analysis")
        for path, expected in source_hashes.items():
            if digest(path) != expected:
                raise RuntimeError("Source probe changed during analysis: " + path)
        if proxy.tangent_proxies.grad is not None:
            raise RuntimeError("Proxy gradient accumulated")
        manifest.update(status="completed", finished_utc=datetime.now(timezone.utc).isoformat(),
                        wall_seconds=time.perf_counter() - started, input_file_sha256=source_hashes,
                        peak_allocated_MiB=torch.cuda.max_memory_allocated(device) / 1024 ** 2 if device.type == "cuda" else None)
        save_json(args.run_dir / "summary.json", {"manifest": manifest, "batches": all_reports,
            "state_checks": {"checkpoint_unchanged": True, "source_probes_unchanged": True, "proxy_grad_buffer_empty": True},
            "interpretation": "Paired same-noise condition changes measure sensitivity to changed geometry; independent-noise agreement quantifies stochastic variability. Percentages are not additive. Whole partials alone cannot explain parameter or optimizer movements."})
        save_json(args.run_dir / "manifest.json", manifest)
    except BaseException as error:
        manifest.update(status="failed", error=repr(error), wall_seconds=time.perf_counter() - started)
        save_json(args.run_dir / "manifest.json", manifest)
        raise


if __name__ == "__main__":
    main()
