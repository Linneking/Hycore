"""CPU-only identity/geometry replay validation with anonymous JSON output.

CLI: --inputs-dir PRIVATE_INPUTS --gpu-run-dir PRIVATE_GPU_RESULT [repeatable]
     --output PUBLIC_VALIDATION.json [--reference-map PRIVATE_MAP.json]

Reference map schema: {"entries": [{"run_label": "model label in manifest",
"reference_cache": "local earlier whole-cache.npz", "epoch": 300}, ...]}.
Optional --pair-config PRIVATE.json schema: {"native_sources": [{"run_label":
"ORIG_best", "cache": "private original_best_eval.npz"}, ...]}. With
--panel-output-dir NEW_PRIVATE_DIR, full native caches are subset to the exact
BN_current_batch_only IDs/order. Native V8 sources are discovered within the
run; the pair config supplies original native sources reused from prior caches.
Use --existing-panel-dir to revalidate already created panels without writes.
--historical-index-bridge accepts the successful anonymous CPU bridge JSON;
each run's nested-index digest must equal the canonical8856 bridge digest.
Optional --whole-points-npz accepts sample_ids and whole_points/clouds [N,1024,3]
for an exact recomputation of the exported nested part indices.

No model, Torch, GPU, network, training, source writes or raw IDs in output.
Historical full9840 part-index hashes cannot be equated to subset8856 hashes;
without raw whole coordinates or the independent CPU bridge, that stronger
check is explicitly unverified.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

from setup_local_inputs import legacy_canonical_ids


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha(value):
    value = np.ascontiguousarray(value)
    return hashlib.sha256(str(value.dtype).encode() + str(value.shape).encode() + value.tobytes()).hexdigest()


def load_npz(path):
    with np.load(path, allow_pickle=False) as source:
        return {key: source[key].copy() for key in source.files}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs-dir", required=True)
    parser.add_argument("--gpu-run-dir", required=True, action="append")
    parser.add_argument("--reference-map")
    parser.add_argument("--pair-config")
    panel_options = parser.add_mutually_exclusive_group()
    panel_options.add_argument("--panel-output-dir")
    panel_options.add_argument("--existing-panel-dir")
    parser.add_argument("--historical-index-bridge")
    parser.add_argument("--whole-points-npz")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    inputs_dir = Path(args.inputs_dir)
    paths = {"input_metadata": inputs_dir / "metadata.json",
             "common_ids": inputs_dir / "common_ids.npz", "centers": inputs_dir / "centers.npz"}
    metadata = json.loads(paths["input_metadata"].read_text(encoding="utf-8-sig"))
    common, centres = load_npz(paths["common_ids"]), load_npz(paths["centers"])
    ids, labels = common["sample_ids"], common["labels"]
    if len(ids) != 8856 or len(np.unique(ids)) != 8856 or not np.all(ids[1:] > ids[:-1]):
        raise ValueError("Common identities must be the canonical ascending8856 set")
    np.testing.assert_array_equal(ids, centres["sample_ids"])
    for key in ("sample_ids", "labels"):
        if array_sha(common[key]) != metadata[key + "_sha256"]:
            raise ValueError("Input identity digest does not match package metadata")
    source_digests = {key: file_sha(path) for key, path in paths.items()}
    result = {"complete": False, "created_utc": datetime.now(timezone.utc).isoformat(),
              "count": len(ids), "canonical_training_ids_sha256": array_sha(ids),
              "canonical_labels_sha256": array_sha(labels),
              "center_indices_sha256": array_sha(centres["center_indices"]),
              "input_package_identity_checks_passed": True, "original_source_checks": {},
              "gpu_run_checks": [], "earlier_whole_cache_replays": [],
              "paired_native_panels": [],
              "no_model_forward": True, "no_GPU": True, "optimizer_updates": 0,
              "test_forwards": 0, "anonymous_output": True, "runner_sha256": file_sha(__file__)}
    # Re-establish the legacy source -> canonical subset bridge directly from
    # the two original caches, independently of generated package row ordering.
    for stage in ("best", "last"):
        source = metadata["sources"][stage]
        cache_path, summary_path = Path(source["source_cache"]), Path(source["source_summary"])
        if (file_sha(cache_path) != source["source_cache_sha256"]
                or file_sha(summary_path) != source["source_summary_sha256"]):
            raise ValueError("Original source cache or summary digest changed")
        original = load_npz(cache_path)
        summary = json.loads(summary_path.read_text(encoding="utf-8-sig"))
        canonical = legacy_canonical_ids(np.asarray(original["ids"], dtype=np.int64),
                                         summary["data_identity"]["source_shards_in_glob_order"])
        lookup = {int(identity): row for row, identity in enumerate(canonical)}
        rows = np.asarray([lookup[int(identity)] for identity in ids])
        np.testing.assert_array_equal(original["labels"][rows], labels)
        np.testing.assert_array_equal(original["part_center_index"][rows], centres["center_indices"])
        package_path = inputs_dir / f"original_{stage}_eval.npz"
        package = load_npz(package_path)
        if file_sha(package_path) != metadata["outputs"][package_path.name]["sha256"]:
            raise ValueError("Prepared original geometry package changed")
        np.testing.assert_array_equal(package["sample_ids"], ids)
        np.testing.assert_array_equal(package["labels"], labels)
        np.testing.assert_array_equal(package["whole_mu"], original["whole_mu"][rows])
        for N in (200, 400, 600):
            np.testing.assert_array_equal(package[f"part_mu_{N}"], original[f"part_{N}_mu"][rows])
        result["original_source_checks"][stage] = {
            "original_cache_sha256": source["source_cache_sha256"],
            "source_checkpoint_sha256": source["source_checkpoint_sha256"],
            "source_epoch_unmodified": source["source_checkpoint_epoch_unmodified"],
            "canonical_labels_elementwise_equal": True, "canonical_centres_elementwise_equal": True,
            "cached_whole_and_all_three_parts_elementwise_equal": True,
            "historic_full9840_part_index_sha256_by_size": source["source_part_indices_sha256_by_size"],
            "historic_index_hash_scope": "full9840 legacy order; differs from the new canonical8856 subset scope"}
    reference_entries = []
    if args.reference_map:
        private_map = json.loads(Path(args.reference_map).read_text(encoding="utf-8-sig"))
        reference_entries = private_map["entries"]
        if len({entry["run_label"] for entry in reference_entries}) != len(reference_entries):
            raise ValueError("Earlier-cache map contains duplicate run labels")
    reference_lookup = {entry["run_label"]: entry for entry in reference_entries}
    historical_bridge = None
    if args.historical_index_bridge:
        bridge_path = Path(args.historical_index_bridge)
        historical_bridge = json.loads(bridge_path.read_text(encoding="utf-8-sig"))
        if (not historical_bridge.get("complete")
                or historical_bridge.get("common_pool_objects") != 8856
                or historical_bridge.get("historical_pool_objects") != 9840
                or not historical_bridge.get("historic_full9840_part_indices_all_three_sizes_exact")
                or not historical_bridge.get("historical_whole_input_bytes_exact")
                or not historical_bridge.get("canonical8856_nested_parts_elementwise_equal")
                or not historical_bridge.get("canonical8856_centres_elementwise_equal")
                or not historical_bridge.get("canonical8856_labels_elementwise_equal")):
            raise ValueError("Historical index bridge is incomplete or did not pass its full contract")
        for source in metadata["sources"].values():
            if (historical_bridge["historic_full9840_part_indices_sha256_by_size"]
                    != source["source_part_indices_sha256_by_size"]):
                raise ValueError("Bridge historical part digests differ from original package provenance")
        result["historical_index_bridge"] = {
            "validation_json_sha256": file_sha(bridge_path),
            "complete": True, "common_pool_objects": 8856, "historical_pool_objects": 9840,
            "historic_full9840_part_indices_all_three_sizes_exact": True,
            "canonical8856_nested_parts_elementwise_equal": True,
            "canonical8856_nested_part_indices_sha256": historical_bridge["canonical8856_nested_part_indices_sha256"],
            "proof_scope": "independent CPU recomputation of full historical coordinates/indices, followed by canonical8856 elementwise GPU-index comparison"}
    native_sources = {}
    if args.pair_config:
        pair_config = json.loads(Path(args.pair_config).read_text(encoding="utf-8-sig"))
        for entry in pair_config["native_sources"]:
            if entry["run_label"] in native_sources:
                raise ValueError("Pair config repeats a model label")
            native_sources[entry["run_label"]] = Path(entry["cache"])
    panel_dir = Path(args.panel_output_dir or args.existing_panel_dir) if (args.panel_output_dir or args.existing_panel_dir) else None
    if args.panel_output_dir:
        panel_dir.mkdir(parents=True, exist_ok=False)
    elif args.existing_panel_dir and not panel_dir.is_dir():
        raise FileNotFoundError("Existing panel directory does not exist")
    raw_whole = None
    if args.whole_points_npz:
        whole_source = load_npz(args.whole_points_npz)
        key = "whole_points" if "whole_points" in whole_source else "clouds"
        whole_ids = np.asarray(whole_source["sample_ids"], dtype=np.int64)
        lookup = {int(identity): row for row, identity in enumerate(whole_ids)}
        raw_whole = np.asarray(whole_source[key])[[lookup[int(identity)] for identity in ids]]
        if raw_whole.shape != (8856, 1024, 3) or not np.isfinite(raw_whole).all():
            raise ValueError("Raw whole point input has unexpected shape/values")
    all_vector_replays_exact = True
    seen_references = set()
    expected_nested_indices, expected_whole_input_sha = None, None
    restored_models = set()
    for run_number, run_value in enumerate(args.gpu_run_dir, 1):
        run_dir = Path(run_value)
        manifest_path = run_dir / "manifest.json"
        identity_path = run_dir / "private_input_identity.npz"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
        summary_path = run_dir / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8-sig"))
        if (not summary.get("complete") or not summary.get("source_files_unchanged")
                or not summary.get("all_state_BN_RNG_restored")
                or any(summary.get(key) != 0 for key in
                       ("backward_calls", "optimizer_updates", "test_forward_calls"))):
            raise ValueError("Frozen run did not certify source/state preservation or zero updates/test")
        if any(not record.get("state_restored") for record in summary["records"]):
            raise ValueError("A recorded model/condition did not restore source state")
        identity = load_npz(identity_path)
        np.testing.assert_array_equal(identity["sample_ids"], ids)
        np.testing.assert_array_equal(identity["labels"], labels)
        np.testing.assert_array_equal(identity["centre_indices"], centres["center_indices"])
        indices = np.asarray(identity["nested_part_indices"], dtype=np.int64)
        if indices.shape != (8856, 600) or np.any(indices < 0) or np.any(indices >= 1024):
            raise ValueError("Nested part index dimensions/range differ from the contract")
        if np.any(np.diff(np.sort(indices, axis=1), axis=1) == 0):
            raise ValueError("A nested part repeats a point index")
        if historical_bridge is not None and (
                array_sha(indices) != historical_bridge["canonical8856_nested_part_indices_sha256"]):
            raise ValueError("A GPU run's nested-index digest differs from the proven historical bridge")
        if expected_nested_indices is None:
            expected_nested_indices = indices.copy()
            expected_whole_input_sha = manifest["canonical_whole_points_sha256"]
        else:
            np.testing.assert_array_equal(indices, expected_nested_indices)
            if manifest["canonical_whole_points_sha256"] != expected_whole_input_sha:
                raise ValueError("The GPU runs did not receive identical raw whole input bytes")
        for field, actual in (("canonical_training_ids_sha256", ids),
                              ("nested_part_indices_sha256", indices),
                              ("center_indices_sha256", identity["centre_indices"]),
                              ("BN_panel_ids_sha256", identity["BN_panel_ids"])):
            if manifest[field] != array_sha(actual):
                raise ValueError("GPU identity arrays and their manifest digests differ")
        positions = np.searchsorted(ids, identity["BN_panel_ids"])
        if np.any(positions >= len(ids)) or not np.array_equal(ids[positions], identity["BN_panel_ids"]):
            raise ValueError("BN subset contains an object outside the canonical training pool")
        index_recomputed = False
        if raw_whole is not None:
            for row, centre in enumerate(centres["center_indices"]):
                xyz = np.asarray(raw_whole[row], dtype=np.float64)
                offsets = xyz - xyz[int(centre)]
                distances = np.einsum("ij,ij->i", offsets, offsets)
                expected = np.lexsort((np.arange(1024), distances))[:600]
                np.testing.assert_array_equal(indices[row], expected)
            index_recomputed = True
        run_check = {"run_number": run_number, "manifest_sha256": file_sha(manifest_path),
                     "summary_sha256": file_sha(summary_path),
                     "identity_npz_sha256": file_sha(identity_path), "objects": len(ids),
                     "sample_ids_elementwise_equal": True, "labels_elementwise_equal": True,
                     "centres_elementwise_equal": True, "nested_index_shape": list(indices.shape),
                     "nested_indices_within_first1024_and_unique_per_row": True,
                     "nested_part_indices_sha256": array_sha(indices),
                     "nested_indices_recomputed_from_raw_whole_exact": index_recomputed,
                     "historical_part_index_identity_established_by_CPU_bridge": historical_bridge is not None,
                     "historical_part_index_identity_status": "exact raw-coordinate recomputation passed"
                     if index_recomputed else "established_by_historical_index_bridge"
                     if historical_bridge is not None else "not established: raw whole coordinates unavailable; full9840 historical hash is not a subset8856 hash",
                     "historical_index_proof_note": "This validator did not load raw whole coordinates; the separate CPU bridge established exact historical9840 hashes and canonical8856 index identity"
                     if historical_bridge is not None and not index_recomputed else None,
                     "BN_panel_objects": len(positions), "BN_panel_membership_verified": True,
                     "source_files_unchanged_certified_by_frozen_runner": True,
                     "all_state_BN_RNG_restored": True, "backward_calls": 0,
                     "optimizer_updates": 0, "test_forward_calls": 0,
                     "models": [],
                     "model_condition_files": []}
        for entry in manifest["spec"]["models"]:
            label = entry["label"]
            records = [record for record in summary["records"] if record["model"] == label]
            if not records:
                raise ValueError("A manifest model has no completed frozen conditions")
            parameter_hashes = {record["source_parameter_sha256"] for record in records}
            if len(parameter_hashes) != 1:
                raise ValueError("A model's parameter digest differs between conditions")
            run_check["models"].append({"model": label, "kind": entry["kind"],
                "diagnostic_checkpoint_sha256": manifest["source_file_sha256"]["checkpoint_" + label],
                "source_parameter_sha256": next(iter(parameter_hashes)),
                "source_epoch": records[0]["source_epoch"],
                "source_completed_epochs": records[0]["source_completed_epochs"],
                "conditions": [record["condition"] for record in records],
                "all_recorded_states_restored": True})
            restored_models.add(label)
            safe_label = "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in label)
            for condition in ("native_eval", "BN_current_batch_only"):
                measured_path = run_dir / f"private_{safe_label}__{condition}.npz"
                if not measured_path.exists():
                    continue
                measured = load_npz(measured_path)
                expected_rows = np.arange(len(ids)) if condition == "native_eval" else positions
                np.testing.assert_array_equal(measured["sample_ids"], ids[expected_rows])
                np.testing.assert_array_equal(measured["labels"], labels[expected_rows])
                run_check["model_condition_files"].append({"model": label, "condition": condition,
                    "objects": len(expected_rows), "file_sha256": file_sha(measured_path),
                    "sample_ids_and_labels_match_identity": True})
                if label not in reference_lookup or condition != "native_eval":
                    continue
                mapping = reference_lookup[label]
                reference_path = Path(mapping["reference_cache"])
                reference = load_npz(reference_path)
                ref_ids = np.asarray(reference["sample_ids"], dtype=np.int64)
                lookup = {int(identity): row for row, identity in enumerate(ref_ids)}
                ref_rows = np.asarray([lookup[int(identity)] for identity in ids])
                np.testing.assert_array_equal(reference["labels"][ref_rows], labels)
                reference_mu = np.asarray(reference["mu"])[ref_rows]
                replays = {}
                for N in (200, 400, 600):
                    mu = measured[f"whole_mu_{N}"]
                    difference = np.abs(np.asarray(mu, dtype=np.float64) - reference_mu)
                    exact = bool(np.array_equal(mu, reference_mu))
                    all_vector_replays_exact = all_vector_replays_exact and exact
                    replays[str(N)] = {"elementwise_identical": exact,
                                        "max_abs_difference": float(difference.max()),
                                        "differing_scalar_count": int(np.count_nonzero(difference))}
                result["earlier_whole_cache_replays"].append({"model": label,
                    "epoch": mapping.get("epoch"), "objects": len(ids),
                    "reference_cache_sha256": file_sha(reference_path),
                    "source_checkpoint_sha256": mapping.get("source_checkpoint_sha256"),
                    "whole_vectors_by_part_size": replays})
                seen_references.add(label)
            if panel_dir is not None:
                batch_path = run_dir / f"private_{safe_label}__BN_current_batch_only.npz"
                native_path = run_dir / f"private_{safe_label}__native_eval.npz"
                if not batch_path.exists():
                    continue
                if not native_path.exists():
                    if label not in native_sources:
                        raise ValueError("A model with a BN panel has no native source in run or pair config")
                    native_path = native_sources[label]
                native_hash_before = file_sha(native_path)
                native = load_npz(native_path)
                batch = load_npz(batch_path)
                native_ids = np.asarray(native["sample_ids"], dtype=np.int64)
                if len(native_ids) != 8856 or len(np.unique(native_ids)) != 8856:
                    raise ValueError("Native source must cover the same unique8856 objects")
                native_lookup = {int(value): row for row, value in enumerate(native_ids)}
                selected = np.asarray([native_lookup[int(value)] for value in batch["sample_ids"]])
                native_panel = {key: value[selected].copy() if value.ndim > 0 and value.shape[0] == len(native_ids)
                                else value.copy() for key, value in native.items()}
                native_panel["condition"] = np.asarray("native_eval_panel")
                np.testing.assert_array_equal(native_panel["sample_ids"], batch["sample_ids"])
                np.testing.assert_array_equal(native_panel["labels"], batch["labels"])
                target = panel_dir / f"private_{safe_label}__native_eval_panel.npz"
                if args.existing_panel_dir:
                    if not target.is_file():
                        raise FileNotFoundError("An existing native panel file is missing")
                else:
                    if target.exists():
                        raise FileExistsError("Panel output model label repeats across run directories")
                    np.savez_compressed(target, **native_panel)
                reloaded = load_npz(target)
                if set(reloaded) != set(native_panel):
                    raise AssertionError("Panel NPZ field round-trip differs")
                for key, value in native_panel.items():
                    np.testing.assert_array_equal(reloaded[key], value)
                if file_sha(native_path) != native_hash_before:
                    raise RuntimeError("A native source changed during subset preparation")
                result["paired_native_panels"].append({"model": label,
                    "condition": "native_eval_panel", "objects": len(selected),
                    "native_source_sha256": native_hash_before, "BN_source_sha256": file_sha(batch_path),
                    "panel_npz_sha256": file_sha(target), "same_ids_and_labels_as_BN_panel": True,
                    "all_fields_preserved": True, "source_row_subset_exact": True,
                    "schema": {key: {"shape": list(value.shape), "dtype": str(value.dtype)}
                               for key, value in native_panel.items()}})
        result["gpu_run_checks"].append(run_check)
    for key, path in paths.items():
        if file_sha(path) != source_digests[key]:
            raise RuntimeError("A local input package source changed during validation")
    result.update(complete=True, source_files_unchanged=True,
                  earlier_whole_vector_replays_all_exact=all_vector_replays_exact,
                  earlier_cache_reference_count=len(reference_entries),
                  earlier_cache_reference_count_checked=len(seen_references),
                  unmatched_reference_count=len(set(reference_lookup) - seen_references),
                  identical_whole_input_bytes_between_GPU_runs=True,
                  identical_nested_part_indices_between_GPU_runs=True,
                  restored_model_count=len(restored_models),
                  all_identity_checks_passed=True)
    write_json(args.output, result)
    print(json.dumps({key: result[key] for key in ("complete", "count", "all_identity_checks_passed",
          "earlier_whole_vector_replays_all_exact", "earlier_cache_reference_count_checked", "unmatched_reference_count")}, ensure_ascii=False))
    if not all_vector_replays_exact:
        raise SystemExit("Whole replay mismatch: inspect anonymous validation JSON")


if __name__ == "__main__":
    main()
