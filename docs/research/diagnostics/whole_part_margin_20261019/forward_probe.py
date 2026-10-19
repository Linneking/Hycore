"""Read-only paired whole/part geometry from existing ModelNet40 checkpoints.

CLI: --spec PRIVATE.json --out-dir NEW_PRIVATE_DIR --gpu ID --max-seconds 1800.
Spec: source_root, data_dir, source_train_ids_npz, models=[{label, checkpoint,
kind: v8|original, c, gamma, delta}], seed=22, include_batch_bn=false,
subset_count=1280, threads=2, centers_npz={sample_ids,center_indices}. Models
may set eval_enabled=false to reuse an existing full eval cache. Eval otherwise
uses all 8856 canonical training identities.
BN-only current-batch diagnostics use a fixed balanced subset, with every other
module in eval. Parts are deterministic nested 200/400/600 nearest neighbours
around supplied centres, or the original ID-stable seed22 centre procedure.
No augmentation or aliasing.
No optimizer, backward, test inputs, or writes to source checkpoints/data.
All output is private; manifests include paths, and NPZ files include sample IDs.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

sys.dont_write_bytecode = True


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            result.update(chunk)
    return result.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                   allow_nan=False, default=str), encoding="utf-8")
    temporary.replace(path)


def check_processes(uuid, allow_self=True):
    result = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid",
                                      "--format=csv,noheader,nounits"], text=True)
    for line in result.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) >= 2 and fields[0] == uuid:
            if not allow_self or int(fields[1]) != os.getpid():
                raise RuntimeError("Another compute process is present; leaving GPU untouched")


def idle_gpu(index):
    result = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,memory.used,utilization.gpu",
                                      "--format=csv,noheader,nounits"], text=True)
    rows = [[field.strip() for field in line.split(",")] for line in result.splitlines()]
    selected = next((row for row in rows if int(row[0]) == index), None)
    if selected is None or float(selected[2]) > 64 or float(selected[3]) != 0:
        raise RuntimeError("Requested GPU is absent or not idle")
    check_processes(selected[1], allow_self=False)
    os.environ["CUDA_VISIBLE_DEVICES"] = selected[1]
    return {"physical_index": index, "uuid": selected[1], "preflight_memory_MiB": float(selected[2]),
            "preflight_utilization_percent": float(selected[3])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--max-seconds", type=float, default=1800)
    args = parser.parse_args()
    if args.gpu < 0 or args.max_seconds <= 0:
        parser.error("GPU must be nonnegative and time budget positive")
    started = time.monotonic()
    spec = json.loads(Path(args.spec).read_text(encoding="utf-8-sig"))
    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=False)
    gpu = idle_gpu(args.gpu)
    root = Path(spec["source_root"]).resolve()
    for directory in (root, root / "classification_ModelNet40", root / "pointnet2_ops_lib",
                      root / "inter_hierarchy_MN40"):
        sys.path.insert(0, str(directory))
    import h5py
    import numpy as np
    import torch
    from inter_hierarchy_MN40.models.pointmlp import Hype_pointMLP
    from inter_hierarchy_MN40.hier_proxy_scratch_v8.geometry import configure_backbone

    torch.set_num_threads(int(spec.get("threads", 2)))
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; no CPU fallback")
    device = torch.device("cuda:0")
    seed = int(spec.get("seed", 22))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    def budget():
        if time.monotonic() - started > args.max_seconds:
            raise TimeoutError("Frozen forward time budget reached")

    def array_sha(array):
        array = np.ascontiguousarray(array)
        return hashlib.sha256(str(array.dtype).encode() + str(array.shape).encode() + array.tobytes()).hexdigest()

    def state_sha(model, parameters_only=False):
        result = hashlib.sha256()
        values = model.named_parameters() if parameters_only else model.state_dict().items()
        for name, value in sorted(values):
            array = value.detach().cpu().contiguous().numpy()
            result.update(name.encode())
            result.update(array_sha(array).encode())
        return result.hexdigest()

    def capture_rng():
        return (random.getstate(), np.random.get_state(), torch.get_rng_state().clone(),
                torch.cuda.get_rng_state(device).clone())

    def restore_rng(saved):
        random.setstate(saved[0])
        np.random.set_state(saved[1])
        torch.set_rng_state(saved[2])
        torch.cuda.set_rng_state(saved[3], device)

    def same_rng(left, right):
        return (left[0] == right[0] and left[1][0] == right[1][0]
                and np.array_equal(left[1][1], right[1][1]) and left[1][2:] == right[1][2:]
                and torch.equal(left[2], right[2]) and torch.equal(left[3], right[3]))

    def describe(values):
        values = np.asarray(values, dtype=np.float64).reshape(-1)
        if not values.size or not np.isfinite(values).all():
            raise ValueError("Nonfinite or empty output")
        return {"count": int(values.size), "mean": float(values.mean()), "std": float(values.std()),
                **dict(zip(("min", "p05", "median", "p95", "max"),
                           map(float, np.quantile(values, [0, .05, .5, .95, 1]))))}

    models = spec["models"]
    if not models or len({entry["label"] for entry in models}) != len(models):
        raise ValueError("Models require unique nonempty labels")
    source_files = {"spec": args.spec, "source_train_ids": spec["source_train_ids_npz"]}
    if spec.get("centers_npz"):
        source_files["part_centers"] = spec["centers_npz"]
    for entry in models:
        if entry["kind"] not in ("original", "v8"):
            raise ValueError("Model kind must be original or v8")
        source_files["checkpoint_" + entry["label"]] = entry["checkpoint"]
    source_hashes = {name: sha(path) for name, path in source_files.items()}
    with np.load(spec["source_train_ids_npz"], allow_pickle=False) as source:
        key = next((key for key in ("sample_ids", "ids", "train_ids") if key in source), None)
        if key is None:
            raise ValueError("Training identity file requires sample_ids/ids/train_ids")
        allowed = np.sort(np.asarray(source[key], dtype=np.int64).reshape(-1))
    if len(allowed) != 8856 or len(np.unique(allowed)) != 8856 or allowed[0] < 0 or allowed[-1] >= 9840:
        raise ValueError("Expected exactly 8856 unique canonical official training IDs")
    shards = sorted(Path(spec["data_dir"]).glob("ply_data_train*.h5"))
    if len(shards) != 5:
        raise ValueError("Expected five official training shards; no test inputs allowed")
    points, labels, shard_hashes = [], [], {}
    for path in shards:
        budget()
        with h5py.File(path, "r") as handle:
            points.append(np.asarray(handle["data"], dtype=np.float32)[:, :1024, :3])
            labels.append(np.asarray(handle["label"], dtype=np.int64).reshape(-1))
        shard_hashes[path.name] = sha(path)
    points, labels = np.concatenate(points), np.concatenate(labels)
    if points.shape != (9840, 1024, 3) or sorted(np.unique(labels).tolist()) != list(range(40)):
        raise ValueError("Unexpected official training shape or labels")
    subset_count = int(spec.get("subset_count", 1280))
    if subset_count <= 0 or subset_count % 40 or subset_count % 32:
        raise ValueError("subset_count must be positive and divisible by both 40 and 32")
    chooser = np.random.RandomState(seed + 20261019)
    panel_ids = np.concatenate([chooser.choice(allowed[labels[allowed] == cls], subset_count // 40,
                                              replace=False) for cls in range(40)])
    chooser.shuffle(panel_ids)
    whole_points = points[allowed].copy()
    if spec.get("centers_npz"):
        with np.load(spec["centers_npz"], allow_pickle=False) as source:
            centre_ids = np.asarray(source["sample_ids"], dtype=np.int64).reshape(-1)
            incoming_centres = np.asarray(source["center_indices"], dtype=np.int64).reshape(-1)
        if len(centre_ids) != len(incoming_centres) or len(np.unique(centre_ids)) != len(centre_ids):
            raise ValueError("Centre identities must be unique and match centre array length")
        lookup = dict(zip(centre_ids.tolist(), incoming_centres.tolist()))
        if any(int(identity) not in lookup for identity in allowed):
            raise ValueError("Explicit centres do not cover every canonical training ID")
        centre_indices = np.asarray([lookup[int(identity)] for identity in allowed], dtype=np.int64)
        if np.any((centre_indices < 0) | (centre_indices >= 1024)):
            raise ValueError("Part centre outside first1024 points")
        nested_indices = np.empty((len(allowed), 600), dtype=np.int64)
        for row, centre in enumerate(centre_indices):
            xyz = np.asarray(whole_points[row], dtype=np.float64)
            offset = xyz - xyz[centre]
            distances = np.einsum("ij,ij->i", offset, offset)
            nested_indices[row] = np.lexsort((np.arange(1024), distances))[:600]
        centre_semantics = "explicit canonical-ID-matched centre_indices; float64 distance and index tie order"
    else:
        from inter_hierarchy_MN40.compare_hycore_class_geometry import deterministic_part_indices
        centre_indices, index_map = deterministic_part_indices(whole_points, allowed, "train", seed, [200, 400, 600])
        nested_indices = index_map[600]
        centre_semantics = "original deterministic_part_indices ID-stable train seed procedure"
    whole_sha = array_sha(whole_points)
    panel_positions = np.searchsorted(allowed, panel_ids)
    np.savez_compressed(output / "private_input_identity.npz", sample_ids=allowed, labels=labels[allowed],
                        BN_panel_ids=panel_ids, centre_indices=centre_indices, nested_part_indices=nested_indices)
    manifest = {"started_utc": datetime.now(timezone.utc).isoformat(), "gpu": gpu, "spec": spec,
                "runner_sha256": sha(__file__), "source_file_sha256": source_hashes,
                "training_shard_sha256": shard_hashes, "canonical_training_ids_sha256": array_sha(allowed),
                "canonical_whole_points_sha256": whole_sha, "nested_part_indices_sha256": array_sha(nested_indices),
                "part_center_semantics": centre_semantics, "center_indices_sha256": array_sha(centre_indices),
                "BN_panel_ids_sha256": array_sha(panel_ids), "eval_objects": len(allowed),
                "BN_panel_objects": len(panel_ids), "N": [200, 400, 600], "local_batch": 32}
    write_json(output / "manifest.json", manifest)
    records = []
    try:
        for entry in models:
            budget()
            check_processes(gpu["uuid"])
            checkpoint = torch.load(entry["checkpoint"], map_location="cpu", weights_only=False)
            net = {key.removeprefix("module."): value for key, value in checkpoint["net"].items()}
            model = Hype_pointMLP()
            if entry["kind"] == "v8":
                model = configure_backbone(model)
            model.load_state_dict(net, strict=True)
            model.to(device).eval().requires_grad_(False)
            actual_c = float(model.emb.manifold.c)
            if not np.isclose(actual_c, float(entry["c"]), rtol=1e-5):
                raise ValueError("Checkpoint/manifold curvature differs from spec")
            layers = {name: module for name, module in model.named_modules()
                      if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)}
            source_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
            source_state_sha, source_parameter_sha = state_sha(model), state_sha(model, True)
            rng = capture_rng()

            def restore():
                model.load_state_dict(source_state, strict=True)
                model.eval()
                restore_rng(rng)

            conditions = [("native_eval", np.arange(len(allowed)))] if entry.get("eval_enabled", True) else []
            if spec.get("include_batch_bn", False):
                conditions.append(("BN_current_batch_only", panel_positions))
            try:
                for condition, positions in conditions:
                    arrays, N_records = {}, []
                    for N in (200, 400, 600):
                        budget()
                        check_processes(gpu["uuid"])
                        restore()
                        if condition == "BN_current_batch_only":
                            for module in layers.values():
                                module.train(True)
                        whole_mu, part_mu, whole_depth, part_depth, positive_distance = [], [], [], [], []
                        for start in range(0, len(positions), 32):
                            budget()
                            selected = positions[start:start + 32]
                            batch_whole = whole_points[selected].copy()
                            batch_part = np.take_along_axis(batch_whole, nested_indices[selected, :N, None], axis=1).copy()
                            whole = torch.from_numpy(batch_whole).to(device).transpose(1, 2).contiguous()
                            part = torch.from_numpy(batch_part).to(device).transpose(1, 2).contiguous()
                            with torch.no_grad():
                                nu, _ = model(part, emb=True)
                                mu, _ = model(whole, emb=True)
                                distance = model.emb.manifold.dist(mu, nu)
                                wd = model.emb.manifold.dist0(mu)
                                pd = model.emb.manifold.dist0(nu)
                            for collection, value in ((whole_mu, mu), (part_mu, nu), (whole_depth, wd),
                                                      (part_depth, pd), (positive_distance, distance)):
                                collection.append(value.detach().float().cpu().numpy())
                        measured = {"whole_mu": np.concatenate(whole_mu), "part_mu": np.concatenate(part_mu),
                                    "whole_depth": np.concatenate(whole_depth), "part_depth": np.concatenate(part_depth),
                                    "positive_distance": np.concatenate(positive_distance)}
                        if any(not np.isfinite(value).all() for value in measured.values()):
                            raise ValueError("Nonfinite forward output")
                        gap = measured["whole_depth"] - measured["part_depth"]
                        measured["depth_gap"] = gap
                        for key, value in measured.items():
                            arrays[key + "_" + str(N)] = value
                        if N == 200:
                            arrays["whole_mu"] = measured["whole_mu"]
                            arrays["whole_depth"] = measured["whole_depth"]
                        margin = float(entry["gamma"]) / N
                        N_records.append({"N": N, "target_gamma_over_N": margin, "delta": float(entry["delta"]),
                                          "whole_depth": describe(measured["whole_depth"]),
                                          "part_depth": describe(measured["part_depth"]), "depth_gap": describe(gap),
                                          "positive_distance": describe(measured["positive_distance"]),
                                          "radial_order_fraction": float(np.mean(gap >= 0)),
                                          "radial_margin_met_fraction": float(np.mean(gap >= margin)),
                                          "whole_repeat_max_abs_difference_to_N200": float(np.max(np.abs(measured["whole_mu"] - arrays["whole_mu"])))})
                        if state_sha(model, True) != source_parameter_sha:
                            raise AssertionError("A learned parameter changed")
                        if condition == "native_eval" and state_sha(model) != source_state_sha:
                            raise AssertionError("Eval modified BN/model state")
                        if any(parameter.grad is not None for parameter in model.parameters()):
                            raise AssertionError("Unexpected gradient")
                        restore()
                        if state_sha(model) != source_state_sha or not same_rng(capture_rng(), rng):
                            raise AssertionError("Per-N state/BN/RNG restoration failed")
                        write_json(output / "progress.json", {"phase": "N_complete", "model": entry["label"],
                                   "condition": condition, "N": N, "objects": len(positions)})
                    if array_sha(whole_points) != whole_sha:
                        raise AssertionError("Part construction/forward mutated whole inputs")
                    safe_label = "".join(c if c.isalnum() or c in "_-" else "_" for c in entry["label"])
                    np.savez_compressed(output / ("private_" + safe_label + "__" + condition + ".npz"),
                                        sample_ids=allowed[positions], labels=labels[allowed[positions]],
                                        condition=np.asarray(condition), **arrays)
                    record = {"model": entry["label"], "kind": entry["kind"], "condition": condition,
                              "c": actual_c, "gamma": float(entry["gamma"]), "delta": float(entry["delta"]),
                              "source_epoch": checkpoint.get("epoch"), "source_completed_epochs": checkpoint.get("completed_epochs"),
                              "objects": len(positions), "N_measurements": N_records,
                              "source_parameter_sha256": source_parameter_sha, "state_restored": True}
                    records.append(record)
                    print(json.dumps({"model": entry["label"], "condition": condition, "objects": len(positions),
                                      "complete": True}), flush=True)
            finally:
                restore()
            if state_sha(model) != source_state_sha or not same_rng(capture_rng(), rng):
                raise AssertionError("Final model/BN/RNG restoration failed")
            del model, source_state, checkpoint, net
            torch.cuda.empty_cache()
        if any(sha(path) != source_hashes[name] for name, path in source_files.items()):
            raise AssertionError("A source file changed")
        if any(sha(path) != shard_hashes[path.name] for path in shards):
            raise AssertionError("Training HDF5 changed")
        write_json(output / "summary.json", {"complete": True, "records": records, "input_condition":
                   "clean first1024; deterministic explicit or ID-stable nested NN parts; canonical sorted8856 train IDs",
                   "part_center_semantics": centre_semantics,
                   "batch_order": "fixed local32; part before whole; each N restores original BN/RNG",
                   "BN_current_batch_limit": "BN-only train, all other modules eval; diagnostic, not inference score",
                   "distance_semantics": "native model.emb.manifold.dist/dist0 at checkpoint curvature; mu retained for independent recomputation",
                   "source_files_unchanged": True, "all_state_BN_RNG_restored": True, "backward_calls": 0,
                   "optimizer_updates": 0, "test_forward_calls": 0, "elapsed_seconds": time.monotonic() - started,
                   "finished_utc": datetime.now(timezone.utc).isoformat(), "runner_sha256": sha(__file__)})
    except BaseException as error:
        write_json(output / "failure.json", {"complete": False, "exception": type(error).__name__,
                   "message": str(error), "completed_conditions": len(records), "elapsed_seconds": time.monotonic() - started})
        raise


if __name__ == "__main__":
    main()
