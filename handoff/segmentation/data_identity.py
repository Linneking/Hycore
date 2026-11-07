"""Compare logical HDF5 contents without binding to file metadata or GPU type."""
import argparse
import hashlib
import json
from pathlib import Path
import struct

import h5py
import numpy as np

FORMAT = "hycore_seg_ordered_arrays_v1"
SPECS = {"data": ("<f4", (2048, 3)), "pid": ("<i2", (2048,)), "label": ("<i8", ())}


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def content_identity(path):
    fields = {}
    with h5py.File(path, "r") as data:
        if set(data.keys()) != {*SPECS, "ids"}:
            raise ValueError("Unexpected prepared fields")
        count = len(data["ids"])
        for name, (dtype, suffix) in SPECS.items():
            field = data[name]
            expected = np.dtype(dtype)
            if field.shape != (count, *suffix):
                raise ValueError("Unexpected shape: " + name)
            if field.dtype.kind != expected.kind or field.dtype.itemsize != expected.itemsize:
                raise ValueError("Unexpected dtype: " + name)
            if name == "data" and not field.dtype.isnative:
                raise ValueError("Coordinates must use native float32 for the existing loader")
            digest = hashlib.sha256()
            for start in range(0, count, 128):
                values = np.ascontiguousarray(field[start:start + 128], dtype=expected)
                if name == "data" and not np.isfinite(values).all():
                    raise ValueError("Nonfinite coordinates")
                digest.update(values.tobytes(order="C"))
            fields[name] = {"shape": list(field.shape), "dtype": dtype,
                            "sha256": digest.hexdigest()}
        if data["ids"].shape != (count,):
            raise ValueError("Unexpected ID shape")
        digest = hashlib.sha256()
        for start in range(0, count, 128):
            for value in data["ids"].asstr()[start:start + 128]:
                encoded = value.encode("utf-8")
                digest.update(struct.pack("<Q", len(encoded)))
                digest.update(encoded)
        fields["ids"] = {"shape": [count], "encoding": "utf8_length_prefixed_u64le",
                         "sha256": digest.hexdigest()}
    identity = {"format": FORMAT, "objects": count, "fields": fields}
    identity["content_sha256"] = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return identity


def verify_prepared(root, profile, splits):
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if not manifest.get("complete"):
        raise ValueError("Prepared manifest is incomplete")
    verified = {}
    for split in splits:
        path = root / (split + ".h5")
        declared = manifest["splits"][split]
        if file_sha256(path) != declared["file_sha256"]:
            raise ValueError("Prepared file differs from its local manifest: " + split)
        actual = content_identity(path)
        if actual["objects"] != declared["objects"] or actual != profile["splits"][split]:
            raise ValueError("Prepared arrays differ from the historical reference: " + split)
        verified[split] = actual["content_sha256"]
    return {"mode": "content", "original_manifest_sha256": profile["original_manifest_sha256"],
            "local_manifest_sha256": file_sha256(root / "manifest.json"), "splits": verified}


def reference_for(original_manifest_sha256):
    profiles = json.loads(Path(__file__).with_name("data_reference.json").read_text(encoding="utf-8"))
    matches = [p for p in profiles["profiles"] if p["original_manifest_sha256"] == original_manifest_sha256]
    if len(matches) != 1:
        raise ValueError("No trusted content reference for this checkpoint's original data manifest")
    return matches[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", choices=("trainval", "test"), default=["trainval", "test"])
    args = parser.parse_args()
    profiles = json.loads(Path(__file__).with_name("data_reference.json").read_text(encoding="utf-8"))
    if len(profiles["profiles"]) != 1:
        parser.error("Choose an unambiguous reference")
    print(json.dumps(verify_prepared(args.data, profiles["profiles"][0], args.splits), indent=2))


if __name__ == "__main__":
    main()
