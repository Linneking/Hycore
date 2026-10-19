"""Frozen BN-bank comparison on a disjoint, stratified ModelNet40 train panel.

CLI: python bn_reference_probe.py --spec PRIVATE_SPEC.json --out-dir NEW_DIR
     --gpu PHYSICAL_IDLE_INDEX --max-seconds 1200

Spec keys: source_root, data_dir, reference_bn_npz, source_train_ids_npz,
models=[{label, checkpoint}, ...], seed (22), include_batch_bn (true).
NPZ IDs may use ids/sample_ids/train_ids. Reference NPZ must contain exactly
the original checkpoint's BN running_mean/running_var/num_batches_tracked,
with their state_dict names (module. prefixes are accepted). No affine transfer.

No optimizer, gradients, test inputs, checkpoint writes, or production changes.
The output directory is PRIVATE: private_*.npz and manifest.json contain IDs or
paths. Only stats.csv and the anonymous conditions in summary.json are intended
as aggregate evidence. A clean statistical bank is model-dependent; importing
another model's bank is an intervention, not a recommended deployment change.
"""
from __future__ import annotations

import argparse
import csv
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
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            value.update(chunk)
    return value.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False, default=str), encoding="utf-8")
    temporary.replace(path)


def checked_idle_uuid(index):
    """Run before importing torch, then pin the selected physical GPU by UUID."""
    text = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,uuid,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits"], text=True)
    selected = None
    for line in text.splitlines():
        fields = [p.strip() for p in line.split(",")]
        if int(fields[0]) == index:
            selected = fields
    if selected is None:
        raise RuntimeError("Requested physical GPU does not exist")
    _, uuid, memory, utilization = selected
    if float(memory) > 64 or float(utilization) != 0:
        raise RuntimeError("GPU is not idle (memory/utilization guard)")
    check_other_processes(uuid, allow_self=False)
    os.environ["CUDA_VISIBLE_DEVICES"] = uuid
    return {"physical_index": index, "uuid": uuid, "preflight_memory_MiB": float(memory),
            "preflight_utilization_percent": float(utilization)}


def check_other_processes(uuid, allow_self=True):
    text = subprocess.check_output([
        "nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"],
        text=True)
    for line in text.splitlines():
        fields = [p.strip() for p in line.split(",")]
        if len(fields) >= 2 and fields[0] == uuid:
            if not allow_self or int(fields[1]) != os.getpid():
                raise RuntimeError("Another compute process is present; leaving GPU untouched")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--gpu", type=int, required=True, help="Verified idle physical GPU index")
    parser.add_argument("--max-seconds", type=float, default=1200)
    args = parser.parse_args()
    if args.gpu < 0 or args.max_seconds <= 0:
        parser.error("GPU index must be nonnegative and time budget positive")
    started = time.monotonic()
    spec = json.loads(Path(args.spec).read_text(encoding="utf-8-sig"))
    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=False)
    gpu = checked_idle_uuid(args.gpu)
    root = Path(spec["source_root"]).resolve()
    for directory in (root, root / "classification_ModelNet40", root / "pointnet2_ops_lib",
                      root / "inter_hierarchy_MN40"):
        sys.path.insert(0, str(directory))
    import h5py
    import numpy as np
    import torch
    from inter_hierarchy_MN40.models.pointmlp import Hype_pointMLP
    from inter_hierarchy_MN40.hier_proxy_scratch_v8.geometry import configure_backbone, depth

    torch.set_num_threads(int(spec.get("threads", 2)))
    device = torch.device("cuda:0")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; no CPU fallback for this probe")
    seed = int(spec.get("seed", 22))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    def budget():
        if time.monotonic() - started > args.max_seconds:
            raise TimeoutError("Frozen diagnostic time budget reached")

    def array_hash(a):
        a = np.ascontiguousarray(a)
        return hashlib.sha256(str(a.dtype).encode() + str(a.shape).encode() + a.tobytes()).hexdigest()

    def state_hash(model, parameters_only=False):
        digest = hashlib.sha256()
        values = model.named_parameters() if parameters_only else model.state_dict().items()
        for name, tensor in sorted(values):
            a = tensor.detach().cpu().contiguous().numpy()
            digest.update(name.encode())
            digest.update(str(a.dtype).encode())
            digest.update(str(a.shape).encode())
            digest.update(a.tobytes())
        return digest.hexdigest()

    def capture_rng():
        return (random.getstate(), np.random.get_state(), torch.get_rng_state().clone(),
                torch.cuda.get_rng_state(device).clone())

    def restore_rng(saved):
        random.setstate(saved[0])
        np.random.set_state(saved[1])
        torch.set_rng_state(saved[2])
        torch.cuda.set_rng_state(saved[3], device)

    def same_rng(a, b):
        return (a[0] == b[0] and a[1][0] == b[1][0]
                and np.array_equal(a[1][1], b[1][1]) and a[1][2:] == b[1][2:]
                and torch.equal(a[2], b[2]) and torch.equal(a[3], b[3]))

    def describe(a):
        a = np.asarray(a, dtype=np.float64).reshape(-1)
        if a.size == 0 or not np.isfinite(a).all():
            raise ValueError("Empty or nonfinite measured array")
        names = ("min", "p01", "p05", "p25", "median", "p75", "p95", "p99", "max")
        result = dict(zip(names, map(float, np.quantile(a, [0, .01, .05, .25, .5, .75, .95, .99, 1]))))
        return dict(result, count=int(a.size), mean=float(a.mean()), std=float(a.std()),
                    variance=float(a.var()))

    def load_ids(path):
        with np.load(path, allow_pickle=False) as source:
            for key in ("sample_ids", "ids", "train_ids"):
                if key in source:
                    a = np.asarray(source[key], dtype=np.int64).reshape(-1)
                    if len(a) != 8856 or len(np.unique(a)) != 8856:
                        raise ValueError("The source pool must contain exactly8856 unique train IDs")
                    return a
        raise ValueError("ID NPZ has no sample_ids/ids/train_ids key")

    source_paths = {"spec": str(Path(args.spec)), "reference_bn": spec["reference_bn_npz"],
                    "source_train_ids": spec["source_train_ids_npz"]}
    model_specs = spec["models"]
    if len(model_specs) != 3 or len({m["label"] for m in model_specs}) != 3:
        raise ValueError("Require three uniquely labelled completed V8 models")
    for entry in model_specs:
        source_paths["checkpoint_" + entry["label"]] = entry["checkpoint"]
    source_hashes = {name: sha(path) for name, path in source_paths.items()}
    shards = sorted(Path(spec["data_dir"]).glob("ply_data_train*.h5"))
    if len(shards) != 5:
        raise ValueError("Expected five official training shards; test input is prohibited")
    data, labels = [], []
    shard_hashes = {}
    for path in shards:
        budget()
        with h5py.File(path, "r") as handle:
            data.append(np.asarray(handle["data"], dtype=np.float32)[:, :1024, :3])
            labels.append(np.asarray(handle["label"], dtype=np.int64).reshape(-1))
        shard_hashes[path.name] = sha(path)
    data, labels = np.concatenate(data), np.concatenate(labels)
    if data.shape != (9840, 1024, 3) or sorted(np.unique(labels).tolist()) != list(range(40)):
        raise ValueError("Unexpected official ModelNet40 training geometry/labels")
    allowed = load_ids(spec["source_train_ids_npz"])
    if allowed.min() < 0 or allowed.max() >= 9840:
        raise ValueError("Source IDs outside official training pool")
    chooser = np.random.RandomState(seed + 20261019)
    measure_ids, calibration_ids = [], []
    for cls in range(40):
        candidates = np.sort(allowed[labels[allowed] == cls])
        chosen = chooser.choice(candidates, 24, replace=False)
        measure_ids.extend(chosen[:8].tolist())
        calibration_ids.extend(chosen[8:].tolist())
    measure_ids = np.asarray(measure_ids, dtype=np.int64)
    calibration_ids = np.asarray(calibration_ids, dtype=np.int64)
    chooser.shuffle(measure_ids)
    chooser.shuffle(calibration_ids)
    if np.intersect1d(measure_ids, calibration_ids).size:
        raise AssertionError("Calibration and measurement panels intersect")
    whole_panel = data[measure_ids].copy()
    # A fixed centre is the sampled point closest to each whole's centroid.
    # Stable full sort preserves deterministic order; constructing parts never
    # aliases or overwrites the original whole tensor.
    centroids = whole_panel.mean(axis=1, keepdims=True)
    centre_indices = np.square(whole_panel - centroids).sum(-1).argmin(-1)
    centres = whole_panel[np.arange(len(whole_panel)), centre_indices]
    distances = np.square(whole_panel - centres[:, None, :]).sum(-1)
    part_indices = np.argsort(distances, axis=1, kind="stable")[:, :400]
    part_panel = np.take_along_axis(whole_panel, part_indices[..., None], axis=1).copy()
    whole_hash = array_hash(whole_panel)
    with np.load(spec["reference_bn_npz"], allow_pickle=False) as source:
        reference = {(key[7:] if key.startswith("module.") else key): source[key].copy()
                     for key in source.files}
    if any(not key.endswith(("running_mean", "running_var", "num_batches_tracked"))
           for key in reference):
        raise ValueError("Reference NPZ must contain running buffers only; no affine weights or metadata")
    manifest = {"started_utc": datetime.now(timezone.utc).isoformat(), "gpu": gpu,
                "spec": spec, "input_file_sha256": source_hashes, "training_shard_sha256": shard_hashes,
                "source_train_ids_sha256": array_hash(np.sort(allowed)),
                "measurement_ids_sha256": array_hash(measure_ids),
                "calibration_ids_sha256": array_hash(calibration_ids),
                "measurement_whole_sha256": whole_hash, "measurement_part_sha256": array_hash(part_panel),
                "measurement_part_indices_sha256": array_hash(part_indices),
                "measurement_objects": 320, "calibration_objects": 640, "classes": 40,
                "measurement_per_class": 8, "calibration_per_class": 16,
                "disjoint_panels": True, "runner_sha256": sha(__file__)}
    write_json(output / "manifest.json", manifest)
    np.savez_compressed(output / "private_panel_identity.npz", measurement_ids=measure_ids,
                        calibration_ids=calibration_ids, measurement_labels=labels[measure_ids],
                        part_indices=part_indices)
    rows, scalar_rows, model_records = [], [], []
    try:
        for model_spec in model_specs:
            budget()
            check_other_processes(gpu["uuid"])
            checkpoint = torch.load(model_spec["checkpoint"], map_location="cpu", weights_only=False)
            if "train_ids" in checkpoint and not np.array_equal(
                    np.sort(np.asarray(checkpoint["train_ids"], dtype=np.int64)), np.sort(allowed)):
                raise ValueError("Checkpoint does not use the same8856 training identities")
            if int(checkpoint.get("completed_epochs", checkpoint.get("epoch", -1))) != 300:
                raise ValueError("This probe requires completed e300 checkpoints")
            net = {key[7:] if key.startswith("module.") else key: value
                   for key, value in checkpoint["net"].items()}
            model = configure_backbone(Hype_pointMLP())
            model.load_state_dict(net, strict=True)
            model.to(device).eval().requires_grad_(False)
            layers = {name: module for name, module in model.named_modules()
                      if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)}
            if not layers or any(not module.track_running_stats for module in layers.values()):
                raise ValueError("All BN layers must track running statistics")
            expected = {name + "." + suffix for name in layers for suffix in
                        ("running_mean", "running_var", "num_batches_tracked")}
            if set(reference) != expected:
                raise ValueError("Reference BN keys mismatch: missing=%s extra=%s" %
                                 (sorted(expected - set(reference)), sorted(set(reference) - expected)))
            source_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
            modes = {name: module.training for name, module in model.named_modules()}
            momenta = {name: module.momentum for name, module in layers.items()}
            source_state_hash, source_parameters_hash = state_hash(model), state_hash(model, True)
            rng = capture_rng()

            def restore():
                model.load_state_dict(source_state, strict=True)
                for name, module in model.named_modules():
                    module.training = modes[name]
                for name, module in layers.items():
                    module.momentum = momenta[name]
                restore_rng(rng)

            def measure():
                raw_whole, raw_part, whole_mu, part_mu, predictions = [], [], [], [], []
                active_raw = [None]

                def raw_hook(module, _inputs, ball):
                    active_raw[0].append(module.manifold.logmap0(ball).detach().float().cpu())

                handle = model.emb.register_forward_hook(raw_hook, prepend=True)
                before = array_hash(whole_panel)
                try:
                    with torch.no_grad():
                        for start in range(0, 320, 32):
                            budget()
                            whole = torch.from_numpy(whole_panel[start:start + 32].copy()).to(device).transpose(1, 2).contiguous()
                            part = torch.from_numpy(part_panel[start:start + 32].copy()).to(device).transpose(1, 2).contiguous()
                            active_raw[0] = raw_part
                            nu, _ = model(part, emb=True)
                            active_raw[0] = raw_whole
                            mu, logits = model(whole)
                            part_mu.append(nu.detach().float().cpu())
                            whole_mu.append(mu.detach().float().cpu())
                            predictions.append(logits.argmax(-1).cpu())
                finally:
                    handle.remove()
                if before != whole_hash or array_hash(whole_panel) != whole_hash:
                    raise AssertionError("Part construction/forward overwrote whole inputs")
                return {"whole_mu": torch.cat(whole_mu).numpy(), "part_mu": torch.cat(part_mu).numpy(),
                        "whole_raw_norm": torch.cat(raw_whole).norm(dim=-1).numpy(),
                        "part_raw_norm": torch.cat(raw_part).norm(dim=-1).numpy(),
                        "predictions": torch.cat(predictions).numpy()}

            conditions = ["native_saved_bank", "reference_original_last_running_buffers",
                          "self_whole_only_bank"]
            if spec.get("include_batch_bn", True):
                conditions.append("BN_batch_statistics")
            baseline = None
            try:
                for condition in conditions:
                    budget()
                    check_other_processes(gpu["uuid"])
                    restore()
                    model.eval()
                    calibration_calls = 0
                    if condition == "reference_original_last_running_buffers":
                        current = dict(model.named_buffers())
                        with torch.no_grad():
                            for name, value in reference.items():
                                target = current[name]
                                incoming = torch.from_numpy(np.asarray(value).copy()).to(target.device)
                                if incoming.shape != target.shape or incoming.dtype != target.dtype:
                                    raise ValueError("Reference BN shape/dtype mismatch: " + name)
                                if not bool(torch.isfinite(incoming).all()):
                                    raise ValueError("Nonfinite reference BN buffer: " + name)
                                if name.endswith("running_var") and bool((incoming < 0).any()):
                                    raise ValueError("Reference running variance is negative")
                                target.copy_(incoming)
                    elif condition == "self_whole_only_bank":
                        for module in layers.values():
                            module.reset_running_stats()
                            module.momentum = None
                            module.train(True)
                        with torch.no_grad():
                            for start in range(0, 640, 32):
                                budget()
                                clouds = torch.from_numpy(data[calibration_ids[start:start + 32]].copy())
                                model(clouds.to(device).transpose(1, 2).contiguous(), emb=True)
                                calibration_calls += 1
                        if any(int(module.num_batches_tracked) != 20 for module in layers.values()):
                            raise AssertionError("Unexpected number of BN calibration calls")
                        model.eval()
                    elif condition == "BN_batch_statistics":
                        # Diagnostic-only: use each current role-specific batch.
                        # Running buffers are allowed to change here, then restored.
                        for module in layers.values():
                            module.train(True)
                    pre_measure_state_hash = state_hash(model)
                    arrays = measure()
                    if condition != "BN_batch_statistics" and state_hash(model) != pre_measure_state_hash:
                        raise AssertionError("Eval measurement modified BN/model state")
                    if state_hash(model, True) != source_parameters_hash:
                        raise AssertionError("A learned parameter, including BN affine, changed")
                    if any(parameter.grad is not None for parameter in model.parameters()):
                        raise AssertionError("Unexpected parameter gradient")
                    for role in ("whole", "part"):
                        arrays[role + "_depth"] = depth(torch.from_numpy(arrays[role + "_mu"])).numpy()
                    if baseline is None:
                        baseline = {key: value.copy() for key, value in arrays.items()}
                    metrics = {}
                    for role in ("whole", "part"):
                        cosine = np.sum(arrays[role + "_mu"] * baseline[role + "_mu"], axis=-1)
                        denominator = np.linalg.norm(arrays[role + "_mu"], axis=-1) * np.linalg.norm(baseline[role + "_mu"], axis=-1)
                        if np.any(denominator <= 0):
                            raise AssertionError("Undefined output direction from zero vector")
                        cosine = np.clip(cosine / denominator, -1, 1)
                        metrics[role] = {"depth": describe(arrays[role + "_depth"]),
                                         "raw_precap_tangent_norm": describe(arrays[role + "_raw_norm"]),
                                         "cap_fraction": float(np.mean(arrays[role + "_depth"] >= 4.5999)),
                                         "precap_over_2_3_fraction": float(np.mean(arrays[role + "_raw_norm"] > 2.3)),
                                         "direction_cosine_to_native": describe(cosine)}
                    gap = arrays["whole_depth"] - arrays["part_depth"]
                    row = {"model": model_spec["label"], "condition": condition,
                           "whole": metrics["whole"], "part": metrics["part"],
                           "whole_minus_part_depth": describe(gap),
                           "whole_deeper_than_part_fraction": float(np.mean(gap > 0)),
                           "prediction_changed_from_native_fraction": float(np.mean(arrays["predictions"] != baseline["predictions"])),
                           "training_panel_accuracy": float(np.mean(arrays["predictions"] == labels[measure_ids])),
                           "measurement_objects": 320, "calibration_objects": 640 if calibration_calls else 0,
                           "calibration_local32_forward_calls": calibration_calls,
                           "BN_affine_and_other_parameters_unchanged": True,
                           "backward_calls": 0, "optimizer_updates": 0, "test_forward_calls": 0}
                    rows.append(row)
                    for role in ("whole", "part"):
                        for key, value in metrics[role].items():
                            if isinstance(value, dict):
                                for stat, number in value.items():
                                    scalar_rows.append({"model": row["model"], "condition": condition,
                                                        "role": role, "metric": key, "statistic": stat, "value": number})
                            else:
                                scalar_rows.append({"model": row["model"], "condition": condition,
                                                    "role": role, "metric": key, "statistic": "value", "value": value})
                    safe_label = "".join(c if c.isalnum() or c in "_-" else "_" for c in model_spec["label"])
                    np.savez_compressed(output / ("private_" + safe_label + "__" + condition + ".npz"),
                                        sample_ids=measure_ids, labels=labels[measure_ids], **arrays)
                    restore()
                    if state_hash(model) != source_state_hash or not same_rng(capture_rng(), rng):
                        raise AssertionError("Per-condition full state/BN/RNG restoration failed")
                    write_json(output / "progress.json", {"phase": "condition_complete", "completed": len(rows),
                                                           "model": row["model"], "condition": condition,
                                                           "whole_depth_mean": row["whole"]["depth"]["mean"]})
                    print(json.dumps({"model": row["model"], "condition": condition,
                                      "whole_depth_mean": row["whole"]["depth"]["mean"],
                                      "whole_cap_fraction": row["whole"]["cap_fraction"]}), flush=True)
            finally:
                restore()
            if state_hash(model) != source_state_hash or not same_rng(capture_rng(), rng):
                raise AssertionError("Final state/RNG restoration failed")
            model_records.append({"model": model_spec["label"], "BN_layers": len(layers),
                                  "source_epoch": checkpoint.get("epoch"), "state_restored": True,
                                  "learned_parameter_fingerprint": source_parameters_hash,
                                  "native_state_fingerprint": source_state_hash})
            del model, source_state, net, checkpoint
            torch.cuda.empty_cache()
        for name, path in source_paths.items():
            if sha(path) != source_hashes[name]:
                raise AssertionError("Input file modified: " + name)
        for path in shards:
            if sha(path) != shard_hashes[path.name]:
                raise AssertionError("Training HDF5 modified")
        with (output / "stats.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=("model", "condition", "role", "metric", "statistic", "value"))
            writer.writeheader()
            writer.writerows(scalar_rows)
        summary = {"complete": True, "conditions": rows, "models": model_records,
                   "measurement_objects": 320, "calibration_objects": 640,
                   "stratification": "measurement40classes x8; calibration40classes x16; disjoint training objects",
                   "input_condition": "clean first1024 whole; fixed centroid-nearest-point centred400NN part; no alias mutation; no augmentation",
                   "batch_order": "fixed shuffled local32; part before whole; same objects and order for every model",
                   "reference_transfer": "running_mean/running_var/num_batches_tracked only; own gamma/beta and all learned weights retained",
                   "calibration": "reset own running stats, momentum=None,20local32whole forwards; BN-only train and all other modules eval; no parameter updates",
                   "BN_batch_statistics_limit": "diagnostic input-batch statistics; not a replacement inference score; temporary buffers restored",
                   "source_files_unchanged": True, "all_state_BN_RNG_restored": True,
                   "optimizer_updates": 0, "backward_calls": 0, "test_forward_calls": 0,
                   "elapsed_seconds": time.monotonic() - started,
                   "finished_utc": datetime.now(timezone.utc).isoformat(),
                   "runner_sha256": sha(__file__),
                   "limits": "Frozen training panel only. Training-panel accuracy is diagnostic, not validation/test, and no condition is selected by it. Transferring running buffers across independently trained networks is generally mismatched; shell recovery alone is not classification or hierarchy benefit."}
        write_json(output / "summary.json", summary)
        write_json(output / "progress.json", {"phase": "complete", "completed": len(rows)})
    except BaseException as error:
        write_json(output / "failure.json", {"complete": False, "exception": type(error).__name__,
                                             "message": str(error), "elapsed_seconds": time.monotonic() - started,
                                             "completed_conditions": len(rows), "optimizer_updates": 0,
                                             "backward_calls": 0, "test_forward_calls": 0})
        raise


if __name__ == "__main__":
    main()
