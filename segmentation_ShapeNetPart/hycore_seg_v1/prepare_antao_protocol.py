"""Restore release train+val membership for the explicitly chosen test-selection protocol.

Reuse already hashed XYZ preprocessing, including repeated official IDs; never
rewrite the earlier disjoint validation dataset. This remains a raw-release
adapter, not an assertion of byte equality with unavailable Stanford HDF5.
"""
import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

from .utils import git_identity, sha256, timestamp, write_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    source, out = Path(args.source), Path(args.output)
    original = json.loads((source / "manifest.json").read_text())
    out.mkdir(parents=True, exist_ok=False)
    files = {s: h5py.File(source / (s + ".h5"), "r") for s in ("train", "val", "test")}
    lookup = {sample: (s, i) for s, f in files.items() for i, sample in enumerate(f["ids"].asstr()[:])}
    split_root = next((source / "raw").rglob("shuffled_train_file_list.json")).parent
    ids = {s: ["/".join(x.split("/")[-2:]) for x in json.loads(
        (split_root / ("shuffled_" + s + "_file_list.json")).read_text())] for s in ("train", "val", "test")}
    manifest = {"complete": False, "created": timestamp(), "source": original["source"],
                "preparation_source": git_identity(), "parent_manifest_sha256": sha256(source / "manifest.json"),
                "preprocessing": original["preprocessing"], "archive_sha256": original["archive_sha256"],
                "split_policy": "original_release_train_plus_val_and_test_no_ID_deduplication",
                "overlap_note": "Preserve provided benchmark memberships, including duplicate IDs; user-selected protocol",
                "splits": {}}
    memberships = {"trainval": ids["train"] + ids["val"], "test": ids["test"]}
    manifest["trainval_duplicate_entries"] = len(memberships["trainval"]) - len(set(memberships["trainval"]))
    manifest["trainval_test_shared_ids"] = len(set(memberships["trainval"]) & set(ids["test"]))
    for split, samples in memberships.items():
        print("BUILD", split, len(samples), flush=True)
        path = out / (split + ".h5")
        with h5py.File(path, "w") as f:
            xyz = f.create_dataset("data", (len(samples), 2048, 3), dtype="f4", compression="lzf")
            parts = f.create_dataset("pid", (len(samples), 2048), dtype="i2", compression="lzf")
            categories = f.create_dataset("label", (len(samples),), dtype="i8")
            f.create_dataset("ids", data=np.asarray(samples, dtype=object), dtype=h5py.string_dtype())
            for row, sample in enumerate(samples):
                old_split, index = lookup[sample]
                parent = files[old_split]
                xyz[row], parts[row], categories[row] = parent["data"][index], parent["pid"][index], parent["label"][index]
        manifest["splits"][split] = {"objects": len(samples), "id_sha256": hashlib.sha256("\n".join(samples).encode()).hexdigest(),
                                      "file_sha256": sha256(path)}
    for f in files.values():
        f.close()
    manifest["complete"] = True
    write_json(out / "manifest.json", manifest)
    print(json.dumps({s: v["objects"] for s, v in manifest["splits"].items()}), flush=True)


if __name__ == "__main__":
    main()
