"""Build a PRIVATE canonical-ID whole/part input package from existing caches.

Usage: python setup_local_inputs.py --config PRIVATE_DENSITY_CONFIG.json
                                    --out-dir NEW_PRIVATE_INPUT_DIRECTORY

The config is the existing density audit config: reference_cache plus one model
whose source_kind is legacy and stages contain best/last cache and summary.
No model forward, source modification or checkpoint loading occurs. Outputs
include object IDs and feature vectors and must remain outside version control.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np


PART_SIZES = (200, 400, 600)


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha(value):
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(str(value.shape).encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def legacy_canonical_ids(ids, shards):
    """Reuse density_figures.legacy_canonical_ids' shard-name/row mapping.

    Kept NumPy-only here so preparing input packages needs no plotting runtime.
    Canonical order is sorted HDF5 filename, followed by the original row.
    """
    counts = {row["name"]: int(row["count"]) for row in shards}
    if len(counts) != len(shards):
        raise ValueError("Legacy source repeats a shard name")
    offsets = {}
    cursor = 0
    for name in sorted(counts):
        offsets[name] = cursor
        cursor += counts[name]
    mapping = np.concatenate(
        [offsets[row["name"]] + np.arange(int(row["count"])) for row in shards]
    )
    if ids.min() < 0 or ids.max() >= len(mapping):
        raise ValueError("Legacy object ID is outside the documented shard order")
    return mapping[ids]


def origin_depth(vectors, curvature):
    vectors = np.asarray(vectors, dtype=np.float64)
    radius = np.linalg.norm(vectors, axis=1)
    scaled = np.sqrt(curvature) * radius
    if not np.isfinite(vectors).all() or np.any(scaled >= 1):
        raise ValueError("Invalid ball vector in source cache")
    return 2 / np.sqrt(curvature) * np.arctanh(scaled)


def schema(arrays):
    return {key: {"shape": list(value.shape), "dtype": str(value.dtype)}
            for key, value in arrays.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    out_dir = Path(args.out_dir).resolve()
    if out_dir.exists():
        raise FileExistsError("Private input output directory must be new")
    inputs = {"config": config_path}
    hashes = {"config": file_sha(config_path)}
    config = json.loads(config_path.read_text(encoding="utf-8-sig"))
    reference_path = Path(config["reference_cache"]).resolve()
    inputs["reference_cache"] = reference_path
    hashes["reference_cache"] = file_sha(reference_path)
    with np.load(reference_path, allow_pickle=False) as source:
        ref_ids = np.asarray(source["sample_ids"], dtype=np.int64).reshape(-1)
        ref_labels = np.asarray(source["labels"], dtype=np.int64).reshape(-1)
        order = np.argsort(ref_ids, kind="stable")
        common_ids, labels = ref_ids[order].copy(), ref_labels[order].copy()
    if (len(common_ids) != 8856 or len(np.unique(common_ids)) != 8856
            or common_ids.min() < 0 or common_ids.max() >= 9840
            or labels.shape != common_ids.shape):
        raise ValueError("Expected the established 8856 unique canonical training IDs")
    legacy_models = [m for m in config["models"] if m["source_kind"] == "legacy"]
    if len(legacy_models) != 1:
        raise ValueError("Config must identify exactly one original HyCoRe model")
    model = legacy_models[0]
    curvature = float(model["curvature"])
    if curvature != 1:
        raise ValueError("Original cache requires curvature 1")
    proof = None
    if config.get("legacy_input_byte_proof"):
        proof_path = Path(config["legacy_input_byte_proof"]).resolve()
        inputs["legacy_input_byte_proof"] = proof_path
        hashes["legacy_input_byte_proof"] = file_sha(proof_path)
        proof = json.loads(proof_path.read_text(encoding="utf-8-sig"))
        if not proof["matches"]:
            raise ValueError("Existing legacy input-byte identity proof did not pass")

    packages, records, centres = {}, {}, {}
    for stage in ("best", "last"):
        entry = model["stages"][stage]
        cache_path = Path(entry["cache"]).resolve()
        summary_path = Path(entry["summary"]).resolve()
        for suffix, path in (("cache", cache_path), ("summary", summary_path)):
            key = "original_" + stage + "_" + suffix
            inputs[key], hashes[key] = path, file_sha(path)
        summary = json.loads(summary_path.read_text(encoding="utf-8-sig"))
        if not summary["patch_identity"]["clean_whole_input_unchanged"]:
            raise ValueError("Original source did not preserve the clean whole input")
        if proof is not None and summary["data_identity"]["points1024_sha256"] != proof["legacy_order_input_sha256"]:
            raise ValueError("Original source input bytes differ from the established proof")
        with np.load(cache_path, allow_pickle=False) as source:
            ids = legacy_canonical_ids(np.asarray(source["ids"], dtype=np.int64),
                                       summary["data_identity"]["source_shards_in_glob_order"])
            if len(np.unique(ids)) != len(ids):
                raise ValueError("Legacy cache has duplicate canonical identities")
            lookup = {int(value): i for i, value in enumerate(ids)}
            rows = np.asarray([lookup[int(value)] for value in common_ids], dtype=np.int64)
            np.testing.assert_array_equal(source["labels"][rows], labels)
            centre = np.asarray(source["part_center_index"][rows], dtype=np.int64)
            if np.any(centre < 0) or np.any(centre >= 1024):
                raise ValueError("Part centre does not index the original first1024 input")
            centres[stage] = centre.copy()
            whole = np.asarray(source["whole_mu"][rows], dtype=np.float32)
            whole_depth = np.asarray(source["whole_d0"][rows], dtype=np.float64)
            np.testing.assert_allclose(origin_depth(whole, curvature), whole_depth, rtol=1e-6, atol=1e-6)
            arrays = {"sample_ids": common_ids.copy(), "labels": labels.copy(),
                      "whole_mu": whole.copy(), "whole_depth": whole_depth.copy(),
                      "c": np.asarray(curvature, dtype=np.float64),
                      "gamma": np.asarray(1000., dtype=np.float64),
                      "delta": np.asarray(4., dtype=np.float64)}
            for size in PART_SIZES:
                part = np.asarray(source[f"part_{size}_mu"][rows], dtype=np.float32)
                part_depth = np.asarray(source[f"part_{size}_d0"][rows], dtype=np.float64)
                if part.shape != whole.shape or whole.shape != (8856, 256):
                    raise ValueError("Unexpected source whole/part feature shape")
                np.testing.assert_allclose(origin_depth(part, curvature), part_depth, rtol=1e-6, atol=1e-6)
                np.testing.assert_allclose(whole_depth - part_depth,
                                           source[f"part_{size}_depth_gap"][rows], rtol=1e-6, atol=1e-6)
                arrays[f"part_mu_{size}"] = part.copy()
                arrays[f"part_depth_{size}"] = part_depth.copy()
            packages[f"original_{stage}_eval.npz"] = arrays
        records[stage] = {
            "source_cache": str(cache_path), "source_cache_sha256": hashes["original_" + stage + "_cache"],
            "source_summary": str(summary_path), "source_summary_sha256": hashes["original_" + stage + "_summary"],
            "source_checkpoint_epoch_unmodified": summary["checkpoint_fields_unmodified"]["epoch"],
            "source_checkpoint_epoch_convention": "original zero-based checkpoint metadata",
            "source_checkpoint_sha256": summary["checkpoint_sha256"],
            "source_shards_in_glob_order": summary["data_identity"]["source_shards_in_glob_order"],
            "source_part_centers_sha256": summary["patch_identity"]["part_centers_sha256"],
            "source_part_indices_sha256_by_size": summary["patch_identity"]["part_indices_sha256_by_size"],
            "common_centres_sha256": array_sha(centres[stage]),
            "canonical_labels_match_reference": True, "depths_recomputed_from_vectors_match": True,
        }
    np.testing.assert_array_equal(centres["best"], centres["last"])
    packages["common_ids.npz"] = {"sample_ids": common_ids, "labels": labels}
    packages["centers.npz"] = {"sample_ids": common_ids.copy(), "center_indices": centres["best"]}
    if any(file_sha(path) != hashes[key] for key, path in inputs.items()):
        raise RuntimeError("A source file changed during read-only package preparation")
    out_dir.mkdir(parents=True, exist_ok=False)
    output_records = {}
    for name, arrays in packages.items():
        destination = out_dir / name
        np.savez_compressed(destination, **arrays)
        with np.load(destination, allow_pickle=False) as check:
            if set(check.files) != set(arrays):
                raise AssertionError("Private NPZ round-trip fields differ")
            for key, value in arrays.items():
                np.testing.assert_array_equal(check[key], value)
        output_records[name] = {"sha256": file_sha(destination), "bytes": destination.stat().st_size,
                                "schema": schema(arrays)}
    metadata = {"created_utc": datetime.now(timezone.utc).isoformat(), "complete": True,
                "private_contains_object_ids_and_features": True,
                "runner_sha256": file_sha(__file__),
                "source_paths": {key: str(path) for key, path in inputs.items()},
                "source_file_sha256": hashes, "source_files_unchanged": True,
                "canonical_order": "ascending official sorted-HDF5-name + row index; same8856 V8 training objects",
                "count": len(common_ids), "part_sizes": list(PART_SIZES),
                "sample_ids_sha256": array_sha(common_ids), "labels_sha256": array_sha(labels),
                "centres_identical_between_original_best_and_last": True,
                "inference_condition": "cached clean first1024 / source eval BN; nested kNN200/400/600; no alias overwrite",
                "legacy_input_byte_proof": proof,
                "sources": records, "outputs": output_records,
                "model_forwards": 0, "optimizer_updates": 0, "test_forwards": 0}
    (out_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"complete": True, "count": len(common_ids), "part_sizes": list(PART_SIZES),
                      "centres_match": True, "output_files": list(output_records),
                      "source_files_unchanged": True, "out_dir": str(out_dir)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
