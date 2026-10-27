"""Prepare the public raw ShapeNetPart archive without changing official splits.

Normalization is a documented adapter choice for the raw normal-bearing release:
center all XYZ points and divide by maximum radius, discard normals, retain first
2048 points. This is not asserted byte-identical to the unavailable upstream H5.
"""

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

import h5py
import numpy as np

from .metrics import NAMES, PART_COUNTS, PART_STARTS
from .utils import sha256, timestamp, write_json

SOURCE = {
    "url": "https://huggingface.co/datasets/wangps/shapenet_segmentation/resolve/dbde146b974e1fc8628b47b1b1c4e50d8bc1a2ef/shapenetcore_partanno_segmentation_benchmark_v0_normal.zip",
    "dataset_revision": "dbde146b974e1fc8628b47b1b1c4e50d8bc1a2ef",
}


def safe_extract(archive, destination):
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(archive) as z:
        for member in z.infolist():
            resolved = (destination / member.filename).resolve()
            if destination not in resolved.parents and resolved != destination:
                raise ValueError("Archive entry outside extraction root")
            if (member.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError("Archive symbolic link rejected")
        z.extractall(destination)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-archive-sha256", required=True)
    args = parser.parse_args()
    archive, output = Path(args.archive), Path(args.output)
    actual_sha = sha256(archive)
    if actual_sha != args.expected_archive_sha256.lower():
        raise ValueError("Archive identity mismatch")
    output.mkdir(parents=True, exist_ok=False)
    raw_parent = output / "raw"
    safe_extract(archive, raw_parent)
    roots = list(raw_parent.rglob("synsetoffset2category.txt"))
    if len(roots) != 1:
        raise ValueError("Ambiguous raw data root")
    raw = roots[0].parent
    mapping = {}
    for line in roots[0].read_text().splitlines():
        name, category = line.split()
        mapping[name.lower()] = category
    synset_to_class = {mapping[name]: i for i, name in enumerate(NAMES)}
    split_ids = {}
    for split in ("train", "val", "test"):
        ids = json.loads((raw / "train_test_split" / ("shuffled_" + split + "_file_list.json")).read_text())
        split_ids[split] = ["/".join(item.split("/")[-2:]) for item in ids]
        if len(set(split_ids[split])) != len(ids):
            raise ValueError("Duplicate IDs within split")
    original_counts = {s: len(ids) for s, ids in split_ids.items()}
    overlaps = {a + "_" + b: sorted(set(split_ids[a]) & set(split_ids[b]))
                for a, b in (("train", "val"), ("train", "test"), ("val", "test"))}
    # The public release really repeats 35 IDs across its official lists.
    # Preserve test, then val, and remove higher-priority IDs from training.
    test_ids = set(split_ids["test"])
    original_val = set(split_ids["val"])
    removed = {"train": [x for x in split_ids["train"] if x in test_ids | original_val],
               "val": [x for x in split_ids["val"] if x in test_ids], "test": []}
    split_ids["train"] = [x for x in split_ids["train"] if x not in test_ids | original_val]
    split_ids["val"] = [x for x in split_ids["val"] if x not in test_ids]
    manifest = {"format": "hycore_seg_raw_to_h5_v1", "source": SOURCE,
                "archive_sha256": actual_sha, "created": timestamp(),
                "preprocessing": "center_full_xyz_unit_max_radius_first2048_no_normals_no_extra_augmentation",
                "official_split_files": {}, "splits": {},
                "original_split_counts": original_counts, "original_cross_split_overlaps": overlaps,
                "split_policy": "test_priority_then_val_then_train_remove_cross_split_duplicate_IDs",
                "removed_ids": removed}
    for split, ids in split_ids.items():
        print("PREPARE", split, len(ids), flush=True)
        h5_path = output / (split + ".h5")
        counts = np.zeros(16, dtype=np.int64)
        numeric_digest = hashlib.sha256()
        with h5py.File(h5_path, "w") as f:
            data = f.create_dataset("data", (len(ids), 2048, 3), dtype="f4", compression="lzf")
            pid = f.create_dataset("pid", (len(ids), 2048), dtype="i2", compression="lzf")
            labels = f.create_dataset("label", (len(ids),), dtype="i8")
            f.create_dataset("ids", data=np.asarray(ids, dtype=object), dtype=h5py.string_dtype("utf-8"))
            for row, sample_id in enumerate(ids):
                synset, stem = sample_id.split("/")
                category = synset_to_class[synset]
                values = np.loadtxt(raw / synset / (stem + ".txt"), dtype=np.float32)
                if len(values) < 1 or values.shape[1] != 7 or not np.isfinite(values).all():
                    raise ValueError(("Unexpected input", sample_id, values.shape))
                xyz = values[:, :3].copy()
                xyz -= xyz.mean(axis=0, keepdims=True)
                radius = np.linalg.norm(xyz, axis=1).max()
                if radius <= 0:
                    raise ValueError("Degenerate shape")
                source_indices = np.arange(2048) % len(values)
                xyz = (xyz / radius)[source_indices].astype(np.float32)
                parts = values[source_indices, -1].astype(np.int16)
                start = PART_STARTS[category]
                if not np.all((parts >= start) & (parts < start + PART_COUNTS[category])):
                    raise ValueError(("Illegal part label", sample_id))
                data[row], pid[row], labels[row] = xyz, parts, category
                counts[category] += 1
                numeric_digest.update(xyz.tobytes())
                numeric_digest.update(parts.tobytes())
                if row and row % 2000 == 0:
                    print(split, row, flush=True)
        manifest["official_split_files"][split] = sha256(raw / "train_test_split" / ("shuffled_" + split + "_file_list.json"))
        manifest["splits"][split] = {"objects": len(ids), "category_counts": counts.tolist(),
                                     "id_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest(),
                                     "numeric_sha256": numeric_digest.hexdigest(),
                                     "file_sha256": sha256(h5_path)}
        write_json(output / "manifest.partial.json", manifest)
    if original_counts != {"train": 12137, "val": 1870, "test": 2874}:
        raise ValueError("Unexpected official split counts")
    manifest["total_unique_objects"] = sum(x["objects"] for x in manifest["splits"].values())
    manifest["complete"] = True
    write_json(output / "manifest.json", manifest)
    print(json.dumps({s: x["objects"] for s, x in manifest["splits"].items()}), flush=True)


if __name__ == "__main__":
    main()
