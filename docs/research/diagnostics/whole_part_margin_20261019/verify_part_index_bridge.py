"""CPU-only exact bridge from historic full9840 parts to current8856 parts.

CLI: --spec PRIVATE_JSON --out-dir NEW_PRIVATE_DIRECTORY
Spec: data_dir, legacy_centers_npz, gpu_input_identity_npz, legacy_shards,
historical_points1024_sha256, historical_part_indices_sha256_by_size,
historical_centres_sha256. legacy_shards is the original glob-order list of
{name,count}; legacy_centers_npz contains center_indices, legacy_to_canonical,
and optional labels. All data are official training HDF5; never test files.

The script needs only NumPy/h5py. It loads no network or checkpoint, invokes no
CUDA, and writes only a new private output directory. Anonymous validation.json
may be copied to public evidence; manifest.json retains private input paths.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import h5py
import numpy as np


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha(array):
    array = np.ascontiguousarray(array)
    return hashlib.sha256(str(array.dtype).encode() + str(array.shape).encode() + array.tobytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    started = time.monotonic()
    spec_path, output = Path(args.spec), Path(args.out_dir)
    spec = json.loads(spec_path.read_text(encoding="utf-8-sig"))
    if output.exists():
        raise FileExistsError("Bridge output must be a new private directory")
    source_paths = {"spec": spec_path, "legacy_centres": Path(spec["legacy_centers_npz"]),
                    "gpu_identity": Path(spec["gpu_input_identity_npz"])}
    source_hashes = {name: file_sha(path) for name, path in source_paths.items()}
    with np.load(source_paths["legacy_centres"], allow_pickle=False) as source:
        centres = np.asarray(source["center_indices"], dtype=np.int64)
        legacy_to_canonical = np.asarray(source["legacy_to_canonical"], dtype=np.int64)
        source_labels = source["labels"].copy() if "labels" in source else None
    if (centres.shape != (9840,) or legacy_to_canonical.shape != (9840,)
            or not np.array_equal(np.sort(legacy_to_canonical), np.arange(9840))
            or np.any(centres < 0) or np.any(centres >= 1024)):
        raise ValueError("Legacy centres/mapping do not cover the official9840 pool")
    if array_sha(centres) != spec["historical_centres_sha256"]:
        raise ValueError("Full legacy centre bytes differ from the historic source")
    shards = spec["legacy_shards"]
    if (len(shards) != 5 or len({entry["name"] for entry in shards}) != 5
            or any(not entry["name"].startswith("ply_data_train")
                   or Path(entry["name"]).name != entry["name"]
                   or not entry["name"].endswith(".h5") for entry in shards)):
        raise ValueError("Require exactly five original training shard basenames")
    clouds, labels, shard_hashes = [], [], {}
    expected_mapping, cursor = {}, 0
    for entry in sorted(shards, key=lambda value: value["name"]):
        expected_mapping[entry["name"]] = cursor
        cursor += int(entry["count"])
    mapping_rows = []
    for entry in shards:
        path = Path(spec["data_dir"]) / entry["name"]
        with h5py.File(path, "r") as handle:
            cloud = np.asarray(handle["data"][:, :1024, :], dtype=np.float32)
            label = np.asarray(handle["label"], dtype=np.int64).reshape(-1)
        if cloud.shape != (int(entry["count"]), 1024, 3) or len(label) != len(cloud):
            raise ValueError("An original training shard has unexpected size/shape")
        clouds.append(cloud)
        labels.append(label)
        mapping_rows.append(expected_mapping[entry["name"]] + np.arange(len(cloud)))
        shard_hashes[entry["name"]] = file_sha(path)
    np.testing.assert_array_equal(legacy_to_canonical, np.concatenate(mapping_rows))
    clouds, labels = np.concatenate(clouds), np.concatenate(labels)
    if source_labels is not None:
        np.testing.assert_array_equal(labels, source_labels)
    points_hash = array_sha(clouds)
    if points_hash != spec["historical_points1024_sha256"]:
        raise ValueError("Full legacy-order whole coordinate bytes differ from the historical cache")
    indices = np.empty((9840, 600), dtype=np.int64)
    tie_order = np.arange(1024)
    for row, centre in enumerate(centres):
        xyz = np.asarray(clouds[row], dtype=np.float64)
        offset = xyz - xyz[int(centre)]
        distance = np.einsum("ij,ij->i", offset, offset)
        indices[row] = np.lexsort((tie_order, distance))[:600]
    part_hashes = {str(N): array_sha(indices[:, :N]) for N in (200, 400, 600)}
    for N, digest in part_hashes.items():
        if digest != spec["historical_part_indices_sha256_by_size"][N]:
            raise ValueError("Recomputed complete legacy part indices differ from historical digest")
    with np.load(source_paths["gpu_identity"], allow_pickle=False) as source:
        common_ids = np.asarray(source["sample_ids"], dtype=np.int64)
        gpu_labels = np.asarray(source["labels"], dtype=np.int64)
        gpu_centres = np.asarray(source["centre_indices"], dtype=np.int64)
        gpu_indices = np.asarray(source["nested_part_indices"], dtype=np.int64)
    if len(common_ids) != 8856 or len(np.unique(common_ids)) != 8856:
        raise ValueError("Current GPU identity does not contain exactly8856 unique objects")
    canonical_to_legacy = np.argsort(legacy_to_canonical)
    rows = canonical_to_legacy[common_ids]
    np.testing.assert_array_equal(labels[rows], gpu_labels)
    np.testing.assert_array_equal(centres[rows], gpu_centres)
    np.testing.assert_array_equal(indices[rows], gpu_indices)
    if any(file_sha(path) != source_hashes[name] for name, path in source_paths.items()):
        raise RuntimeError("A bridge source input changed during its read-only computation")
    for entry in shards:
        if file_sha(Path(spec["data_dir"]) / entry["name"]) != shard_hashes[entry["name"]]:
            raise RuntimeError("A training HDF5 shard changed during read-only verification")
    output.mkdir(parents=True, exist_ok=False)
    validation = {"complete": True, "CPU_only": True, "model_forwards": 0,
        "optimizer_updates": 0, "test_forwards": 0, "source_files_unchanged": True,
        "historical_pool_objects": 9840, "common_pool_objects": 8856,
        "historical_whole_input_bytes_exact": True, "historical_centres_exact": True,
        "legacy_to_canonical_shard_row_mapping_exact": True,
        "historical_points1024_sha256": points_hash,
        "historic_full9840_part_indices_sha256_by_size": part_hashes,
        "historic_full9840_part_indices_all_three_sizes_exact": True,
        "canonical8856_labels_elementwise_equal": True,
        "canonical8856_centres_elementwise_equal": True,
        "canonical8856_nested_parts_elementwise_equal": True,
        "canonical8856_nested_part_indices_sha256": array_sha(gpu_indices),
        "hash_convention": "SHA256(dtype string + shape string + C-order contiguous bytes)",
        "kNN_convention": "float64 squared Euclidean distance; original first1024 point-index tie break",
        "part_sizes": [200, 400, 600], "runner_sha256": file_sha(__file__),
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.monotonic() - started}
    write_json(output / "validation.json", validation)
    write_json(output / "manifest.json", {"spec": spec,
        "source_paths": {name: str(path) for name, path in source_paths.items()},
        "source_file_sha256": source_hashes, "training_shard_sha256": shard_hashes,
        "validation_sha256": file_sha(output / "validation.json")})
    print(json.dumps(validation, ensure_ascii=False))


if __name__ == "__main__":
    main()
