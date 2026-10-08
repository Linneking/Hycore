"""Join a reordered HIER Cars MAT to downloaded original image identities.

Filename equality is never an identity criterion: the handoff renamed images.
Only the original class plus all four bounding-box coordinates form the join.
Outputs contain complete identities and absolute paths; keep them outside Git.
No image decoder, model forward, GPU or network request is used by this tool.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath

import numpy as np
from scipy.io import loadmat


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def safe_relative(value):
    text = str(value).replace("\\", "/")
    while text.startswith("./"):
        text = text[2:]
    path = PurePosixPath(text)
    if path.is_absolute() or ".." in path.parts or ":" in text:
        raise ValueError(f"Unsafe image-relative path: {text}")
    if not text or not path.parts:
        raise ValueError("Empty image path")
    return str(path)


def mat_rows(path):
    """Read consolidated relative_im_path or original split fname annotations."""
    content = loadmat(path, squeeze_me=True, struct_as_record=False)
    if "annotations" not in content:
        raise ValueError("MAT lacks annotations")
    annos = np.asarray(content["annotations"], dtype=object).ravel()
    rows = []
    for row in annos:
        fields = getattr(row, "_fieldnames", [])
        pathfield = next((k for k in ("relative_im_path", "relative_impath", "fname") if k in fields), None)
        required = ("class", "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2")
        if pathfield is None or any(k not in fields for k in required):
            raise ValueError(f"Unexpected annotation fields: {fields}")
        numbers = tuple(int(np.asarray(getattr(row, k)).item()) for k in required)
        if numbers[0] < 1:
            raise ValueError("Cars MAT classes must be one based")
        rows.append({"relative_path": safe_relative(getattr(row, pathfield)),
                     "class_original_1based": numbers[0], "class_zero_based": numbers[0] - 1,
                     "bbox_x1": numbers[1], "bbox_y1": numbers[2],
                     "bbox_x2": numbers[3], "bbox_y2": numbers[4], "key": numbers})
    paths = [r["relative_path"] for r in rows]
    if len(paths) != len(set(paths)):
        raise ValueError("Duplicate annotation image paths")
    return rows


def consolidated_source(rows, images_root):
    root = Path(images_root).resolve()
    out = []
    for row in rows:
        relative = PurePosixPath(row["relative_path"])
        # The caller may supply either the dataset parent or its car_ims itself.
        if root.name == relative.parts[0]:
            relative = PurePosixPath(*relative.parts[1:])
        out.append(dict(row, source_path=str(root.joinpath(*relative.parts))))
    return out


def split_source(rows, root, split):
    root = Path(root).resolve()
    candidates = [root / f"cars_{split}", root / "stanford-cars" / f"cars_{split}"]
    if root.name == f"cars_{split}":
        candidates.insert(0, root)
    existing = [p for p in candidates if p.is_dir()]
    if len(existing) != 1:
        raise ValueError(f"Expected exactly one cars_{split} image directory, found {existing}")
    image_dir = existing[0]
    return [dict(row, source_path=str(image_dir / Path(row["relative_path"]).name), source_split=split) for row in rows]


def auto_mat(root, filename):
    root = Path(root)
    paths = [root / filename, root / "devkit" / filename,
             root / "stanford-cars" / filename, root / "stanford-cars" / "devkit" / filename]
    existing = [p for p in paths if p.is_file()]
    if len(existing) != 1:
        raise ValueError(f"Provide --train-mat/--test-mat: expected one {filename}, found {existing}")
    return existing[0]


def build_join(provided, source, allow_missing=False):
    buckets = defaultdict(list)
    for row in source:
        buckets[row["key"]].append(row)
    hash_cache, output, mapping, unresolved = {}, [], {}, []
    for target in provided:
        candidates = sorted(buckets.get(target["key"], []), key=lambda r: r["source_path"])
        files = [Path(r["source_path"]) for r in candidates]
        chosen, hashes, equivalent = "", {}, False
        if not candidates:
            status = "no_annotation_match"
        elif not all(p.is_file() for p in files):
            status = "missing_file"
            if allow_missing and len(candidates) == 1:
                chosen = candidates[0]["source_path"]
                status = "annotation_only_missing_file"
        elif len(candidates) == 1:
            chosen = candidates[0]["source_path"]
            status = "unique"
        else:
            for candidate in files:
                key = str(candidate)
                if key not in hash_cache:
                    hash_cache[key] = sha256(candidate)
                hashes[key] = hash_cache[key]
            equivalent = len(set(hashes.values())) == 1
            if equivalent:
                chosen = candidates[0]["source_path"]
                status = "byte_equivalent"
            else:
                status = "ambiguous_forward_match"
        row = {"target_path": target["relative_path"], "source_path": chosen,
               "class_original_1based": target["class_original_1based"],
               "class_zero_based": target["class_zero_based"],
               **{k: target[k] for k in ("bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2")},
               "status": status, "candidate_count": len(candidates),
               "candidate_paths_json": json.dumps([r["source_path"] for r in candidates], ensure_ascii=False),
               "candidate_sha256_json": json.dumps(hashes, ensure_ascii=False),
               "byte_equivalent": equivalent}
        output.append(row)
        if status in ("unique", "byte_equivalent"):
            mapping[target["relative_path"]] = chosen
        else:
            unresolved.append(row)
    return output, mapping, unresolved


def run(args):
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"Fresh output required: {output}")
    annotation_inputs = [Path(args.provided_mat)]
    if args.canonical_mat:
        if not args.images_root:
            raise ValueError("--canonical-mat requires --images-root")
        annotation_inputs.append(Path(args.canonical_mat))
        source = consolidated_source(mat_rows(args.canonical_mat), args.images_root)
        mode = "original_car_ims"
    else:
        if not args.fastai_root:
            raise ValueError("Provide --canonical-mat or --fastai-root")
        train_mat = Path(args.train_mat) if args.train_mat else auto_mat(args.fastai_root, "cars_train_annos.mat")
        test_mat = Path(args.test_mat) if args.test_mat else auto_mat(args.fastai_root, "cars_test_annos_withlabels.mat")
        annotation_inputs.extend([train_mat, test_mat])
        source = split_source(mat_rows(train_mat), args.fastai_root, "train") + split_source(mat_rows(test_mat), args.fastai_root, "test")
        mode = "original_train_test_annotations_for_fastai_images"
    identities = {str(path.resolve()): sha256(path) for path in annotation_inputs}
    provided = mat_rows(args.provided_mat)
    target_pattern = all(Path(r["relative_path"]).name == f"{i+1:06d}.jpg" for i, r in enumerate(provided))
    rows, mapping, unresolved = build_join(provided, source, args.allow_missing_images)
    counts = Counter(r["status"] for r in rows)
    matched = [r for r in rows if r["candidate_count"] > 0]
    missing_keys = [r for r in rows if r["candidate_count"] == 0]
    unique_annotation = sum(r["candidate_count"] == 1 for r in rows)
    summary = {"format": "hier-original-cars-identity-v1", "mode": mode,
               "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "script_sha256": sha256(__file__), "annotation_sha256": identities,
               "provided_rows": len(provided), "source_rows": len(source), "annotation_matched_rows": len(matched),
               "unique_annotation_rows": unique_annotation,
               "ambiguous_annotation_rows": len(provided) - unique_annotation - len(missing_keys),
               "resolved_existing_image_rows": len(mapping), "unresolved_rows": len(unresolved),
               "status_counts": dict(counts), "target_sequential_1based_six_digit": target_pattern,
               "join_key": ["class", "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"],
               "class_label_definition": "MAT original one-based class and zero-based class; HIER split-local labels must still be joined against its sample map",
               "mapping_definition": "only unique or byte-identical equivalent existing image paths are included; unresolved rows are never silently chosen",
               "privacy": "full identities and absolute image paths are private, not for Git publication",
               "training_updates": 0, "gpu_use": False, "network_use": False}
    for path in annotation_inputs:
        if sha256(path) != identities[str(path.resolve())]:
            raise AssertionError("Input annotation changed during mapping")
    output.mkdir(parents=True)
    with (output / "mapping.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    payload = {"mapping": mapping, "unresolved": unresolved, "summary": summary}
    (output / "identity_map.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("annotation_sha256",)}, ensure_ascii=False))
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--provided-mat", required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--canonical-mat")
    group.add_argument("--fastai-root")
    parser.add_argument("--images-root")
    parser.add_argument("--train-mat")
    parser.add_argument("--test-mat")
    parser.add_argument("--allow-missing-images", action="store_true", help="Prepare annotated paths while downloads are pending; missing files remain excluded from runnable mapping")
    parser.add_argument("--output", required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
