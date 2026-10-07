"""Read-only clean whole export on one freshly checked idle GPU."""
from __future__ import annotations
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[2]
OWN_PROXY_FORMATS = {
    "hycore-hier-v5-h20-1", "hycore-hier-v6-h20-selfk300-1",
    "hycore-hier-v7-w0-projected-proxy-1",
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_run(path):
    from .inventory import resolve_run_dir
    return resolve_run_dir(path)


def load_cpu_checkpoint(path):
    """Only the explicitly supplied user's own run checkpoints are trusted here."""
    import torch
    try:
        return torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except RuntimeError as exc:
        if "mmap" not in str(exc):
            raise
        return torch.load(path, map_location="cpu", weights_only=False)


def checkpoint_epoch(saved, alias=None):
    """Resolve the saved model's epoch; embedded best is not a last-model epoch."""
    values = [int(saved[key]) for key in ("epoch", "completed_epochs")
              if saved.get(key) is not None]
    if len(set(values)) > 1:
        raise ValueError("Checkpoint epoch and completed_epochs disagree")
    if values:
        return values[0]
    if saved.get("initialization_only"):
        return 0
    if alias == "best" and saved.get("model_selection_only") and (saved.get("best") or {}).get("epoch") is not None:
        return int(saved["best"]["epoch"])
    raise ValueError("Saved checkpoint has no unambiguous actual model epoch")


def resolve_proxy_mapping(saved, config=None):
    """Use inspected live V5/V6/V7 mapping, never the shadow clip_r default."""
    proxy_state = saved.get("proxy") or {}
    if not any(str(key).split(".")[-1] == "tangent_proxies" for key in proxy_state):
        return None
    if saved.get("format") not in OWN_PROXY_FORMATS:
        raise ValueError("Unknown proxy checkpoint format; an explicit version adapter is required")
    config = saved.get("training_config", {}) if config is None else config
    if not math.isclose(float(config.get("c", float("nan"))), 1., rel_tol=0., abs_tol=1e-7):
        raise ValueError("Inspected own proxy formats require c=1")
    cap_enabled = config.get("extra_HIER_tangent_cap")
    if cap_enabled is False:
        declared = config.get("proxy_mapping")
        if declared and declared != {"numeric_radius_fraction": .999}:
            raise ValueError("Declared proxy mapping contradicts the inspected live mapping")
        return {"numeric_radius_fraction": .999}
    if cap_enabled is True:
        # None of the current own trainers enable this. A future explicitly
        # described adaptation must give both cap and denominator convention.
        declared = config.get("proxy_mapping")
        allowed = {"numeric_radius_fraction", "source_tangent_cap", "source_cap_epsilon"}
        if not isinstance(declared, dict) or set(declared) != allowed:
            raise ValueError("Enabled source tangent cap requires explicit proxy_mapping cap and epsilon")
        if (float(declared["numeric_radius_fraction"]) != .999 or
                not math.isfinite(float(declared["source_tangent_cap"])) or float(declared["source_tangent_cap"]) <= 0 or
                not math.isfinite(float(declared["source_cap_epsilon"])) or float(declared["source_cap_epsilon"]) < 0):
            raise ValueError("Invalid declared source tangent cap mapping")
        return dict(declared)
    raise ValueError("extra_HIER_tangent_cap is missing; do not infer active clipping from clip_r")


def validate_training_ids(saved, reference_ids):
    """When present, every checkpoint must have the exact saved split ordering."""
    import numpy as np
    if "train_ids" in saved and not np.array_equal(np.asarray(saved["train_ids"]).reshape(-1), reference_ids):
        raise ValueError("Checkpoint training IDs differ from the fixed source split")


def config_digest(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":"), allow_nan=False)
                          .encode("utf-8")).hexdigest()


def reset_inference_rng(torch_module, numpy_module, seed):
    """Apply after constructing/loading each model, immediately before forward."""
    torch_module.manual_seed(seed)
    torch_module.cuda.manual_seed_all(seed)
    numpy_module.random.seed(seed)


def choose_checkpoints(run_dir, policy="standard", maximum=12, checkpoint_loader=None, epochs=None):
    run = Path(run_dir)
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8-sig"))
    config = manifest.get("training_config", manifest.get("config", {}))
    if policy not in ("standard", "all") or maximum < 1:
        raise ValueError("checkpoint policy must be standard/all and budget positive")
    loader = checkpoint_loader or load_cpu_checkpoint
    files = {}
    for path in run.glob("checkpoint_epoch_*.pth"):
        match = re.search(r"checkpoint_epoch_(\d+)\.pth$", path.name)
        if match:
            files[int(match.group(1))] = path
    actual_aliases = {}
    for filename, alias in (("initialization_e0.pth", "initialization"), ("last.pth", "last"), ("best.pth", "best")):
        path = run / filename
        if path.exists():
            saved = loader(path)
            epoch = checkpoint_epoch(saved, alias)
            if alias == "initialization" and epoch != 0:
                raise ValueError("initialization_e0 checkpoint is not the initial model")
            files[epoch] = path
            actual_aliases[alias] = epoch
            del saved
    last_epoch = actual_aliases.get("last", max(files, default=0))
    best_epoch = actual_aliases.get("best")
    if epochs is not None:
        tokens = [value.strip() for value in (epochs.split(",") if isinstance(epochs, str) else epochs)]
        requested = {actual_aliases.get(value) if value in ("last", "best", "initialization") else int(value) for value in tokens}
        missing_requested = sorted(value for value in requested if value is not None and value not in files)
        files = {epoch: path for epoch, path in files.items() if epoch in requested}
        manifest["requested_epochs_missing"] = missing_requested
    elif policy == "standard":
        warmup = int(config.get("warmup_epochs", 0))
        requested = {0, 1, 5, 20, 40, 99, 100, 160, 200, last_epoch, warmup, warmup + 1}
        if best_epoch is not None:
            requested.add(int(best_epoch))
        files = {epoch: path for epoch, path in files.items() if epoch in requested}
    if len(files) > maximum:
        raise ValueError("Available selected checkpoints exceed the explicit checkpoint budget")
    if not files:
        raise ValueError("No compatible saved checkpoints; cannot reconstruct missing epochs")
    manifest = dict(manifest)
    manifest["resolved_checkpoint_aliases"] = actual_aliases
    manifest["manifest_epoch_aliases_agree"] = (
        manifest.get("completed_epochs") in (None, actual_aliases.get("last")) and
        (manifest.get("best") or {}).get("epoch") in (None, actual_aliases.get("best")))
    return sorted(files.items()), manifest


def idle_gpu(gpu):
    query = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid,memory.used,utilization.gpu",
         "--format=csv,noheader,nounits"], text=True)
    active = set(subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid",
         "--format=csv,noheader,nounits"], text=True).split())
    for line in query.splitlines():
        index, uuid, used, utilization = [part.strip() for part in line.split(",")]
        if int(index) == gpu:
            if uuid in active or int(used) > 128 or int(utilization):
                raise RuntimeError("Requested GPU is busy; no process will be interrupted")
            return uuid
    raise ValueError("Requested physical GPU does not exist")


def load_training_shards(data_dir):
    import h5py
    import numpy as np
    shards = sorted(Path(data_dir).glob("ply_data_train*.h5"))
    if not shards:
        raise FileNotFoundError("No ModelNet40 training shards")
    points, labels = [], []
    for path in shards:
        with h5py.File(path, "r") as stream:
            points.append(stream["data"][:].astype(np.float32))
            labels.append(stream["label"][:].astype(np.int64).reshape(-1))
    return np.concatenate(points), np.concatenate(labels), [path.name for path in shards]


def state_digest(state):
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        digest.update(name.encode())
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def extract_features(run_dir, output_dir, data_dir, gpu, policy="standard", maximum=12,
                     population="panel", per_class=32, seed=22, batch_size=32,
                     max_seconds=3600, with_pointclouds=False, epochs=None):
    started = time.monotonic()
    started_utc = dt.datetime.now(dt.timezone.utc).isoformat()
    import numpy as np
    from .inventory import resolve_run_dir, audit_run_key
    run = resolve_run_dir(run_dir)
    out = Path(output_dir).resolve()
    if out == run or run in out.parents:
        raise ValueError("Feature export output must be outside its source run directory")
    if out.exists():
        raise FileExistsError("Feature export requires a fresh output directory")
    if batch_size < 1 or per_class < 1 or maximum < 1 or max_seconds <= 0:
        raise ValueError("Invalid export budget")
    selected, source_manifest = choose_checkpoints(run, policy, maximum, epochs=epochs)
    if source_manifest.get("status") != "completed":
        raise ValueError("GPU export requires a completed source run")
    uuid = idle_gpu(gpu)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    import torch
    if torch.cuda.is_initialized():
        raise RuntimeError("CUDA was already initialized; use a fresh export process")
    sys.path.insert(0, str(REPO / "pointnet2_ops_lib"))
    sys.path.insert(0, str(REPO / "inter_hierarchy_MN40"))
    from models.pointmlp import Hype_pointMLP
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    np.random.seed(seed)
    points, labels, shards = load_training_shards(data_dir)
    recorded_shards = source_manifest.get("training_shards")
    if recorded_shards is not None:
        recorded_names = [str(name).replace("\\", "/").split("/")[-1] for name in recorded_shards]
        if recorded_names != shards:
            raise ValueError("Loaded training shard basename/order differs from the source manifest")
    recorded_label_sha = source_manifest.get("train_labels_sha256")
    if recorded_label_sha is not None and hashlib.sha256(labels.astype(np.int64).tobytes()).hexdigest() != recorded_label_sha:
        raise ValueError("Loaded training labels differ from the source manifest identity")
    # These are the user's own explicit completed training artifacts.
    full_path = run / "last.pth"
    if not full_path.exists():
        full_path = selected[-1][1]
    full = load_cpu_checkpoint(full_path)
    training_ids = np.asarray(full.get("train_ids", []), dtype=np.int64)
    if not len(training_ids):
        raise ValueError("Checkpoint lacks saved train IDs; do not guess its split")
    if len(np.unique(training_ids)) != len(training_ids) or training_ids.min() < 0 or training_ids.max() >= len(points):
        raise ValueError("Invalid source training population IDs")
    if population == "panel":
        rng = np.random.default_rng(seed)
        chosen = []
        for label in sorted(np.unique(labels[training_ids])):
            candidates = training_ids[labels[training_ids] == label]
            chosen.extend(rng.choice(candidates, min(per_class, len(candidates)), replace=False).tolist())
        ids = np.sort(np.asarray(chosen, dtype=np.int64))
    elif population == "full":
        ids = np.sort(training_ids)
    else:
        raise ValueError("population must be panel or full")
    clouds = points[ids, :1024].copy()
    gold = labels[ids].copy()
    pool_sha = hashlib.sha256(ids.astype("<i8").tobytes()).hexdigest()
    input_sha = hashlib.sha256(clouds.tobytes()).hexdigest()
    source_config = full.get("training_config", source_manifest.get("training_config", source_manifest.get("config", {})))
    source_commit = full.get("commit", source_manifest.get("commit"))
    source_split_sha = full.get("split_sha256", source_manifest.get("split_sha256"))
    if source_split_sha is not None and "validation_ids" in full:
        observed_split = hashlib.sha256(training_ids.astype(np.int64).tobytes() +
            np.asarray(full["validation_ids"], dtype=np.int64).tobytes()).hexdigest()
        if observed_split != source_split_sha:
            raise ValueError("Fixed source split IDs do not match its saved split SHA256")
    source_checkpoint_sha = sha256(full_path)
    del points, full
    fps_mode = "active pointnet2 CUDA FPS; inspected kernel starts at index0; binary realization not independently audited"
    input_mode = "clean first1024/eval BN; seed=%d; batch_size=%d; FPS=pointnet2_index0_source" % (seed, batch_size)
    inference_condition = {"rng_seed": seed, "batch_size": batch_size, "object_order": "sorted saved training IDs",
                           "FPS_mode": fps_mode,
                           "pointmlp_source_sha256": sha256(REPO / "inter_hierarchy_MN40" / "models" / "pointmlp.py"),
                           "FPS_kernel_source_sha256": sha256(REPO / "pointnet2_ops_lib" / "pointnet2_ops" / "_ext-src" / "src" / "sampling_gpu.cu")}
    out.mkdir(parents=True, exist_ok=False)
    manifest = {"format": "hier-postrun-clean-export-v1", "status": "running",
                "source_run_id": run.name, "physical_gpu": gpu, "seed": seed,
                "population": population, "sample_count": len(ids),
                "pool_ids_sha256": pool_sha, "input_sha256": input_sha,
                "fixed_train_ids_sha256": hashlib.sha256(training_ids.astype("<i8").tobytes()).hexdigest(),
                "training_shards": shards, "input_mode": input_mode,
                "inference_rng_seed": seed, "inference_batch_size": batch_size,
                "inference_condition": inference_condition,
                "training_shard_names_match_source": recorded_shards is not None,
                "training_labels_match_source_sha256": recorded_label_sha is not None,
                "full_training_shard_content_hash_verified": False,
                "source_commit": source_commit, "source_training_config": source_config,
                "source_training_config_sha256": config_digest(source_config),
                "source_split_sha256": source_split_sha,
                "source_checkpoint_file": full_path.name, "source_checkpoint_sha256": source_checkpoint_sha,
                "source_manifest_sha256": sha256(run / "manifest.json"),
                "resolved_checkpoint_aliases": source_manifest["resolved_checkpoint_aliases"],
                "manifest_epoch_aliases_agree": source_manifest["manifest_epoch_aliases_agree"],
                "optimizer_updates": 0, "validation_forwarded": False,
                "test_forwarded": False, "maximum_checkpoints": maximum,
                "max_seconds": max_seconds, "started_utc": started_utc}
    try:
        manifest["extractor_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        manifest["extractor_commit"] = None
    def write_manifest():
        (out / "export_manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8")
    write_manifest()
    specs = []
    manifest["completed_checkpoint_identities"] = []
    def write_specs():
        temporary = out / "snapshot_spec.json.tmp"
        temporary.write_text(json.dumps({"snapshots": specs}, indent=2, allow_nan=False), encoding="utf-8")
        temporary.replace(out / "snapshot_spec.json")
    def finish(status):
        if hashlib.sha256(clouds.tobytes()).hexdigest() != input_sha:
            raise RuntimeError("Input clouds changed during extraction")
        if with_pointclouds:
            np.savez_compressed(out / "candidate_clouds.npz", sample_ids=ids, labels=gold, clouds=clouds)
        write_specs()
        manifest.update(status=status, completed_snapshots=len(specs), finished_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                        wall_seconds=time.monotonic() - started, source_files_unchanged=True,
                        model_BN_unchanged=True, gradients_absent=True)
        write_manifest()
        return {"snapshots": specs, "manifest": manifest, "pointcloud_pool": str(out / "candidate_clouds.npz") if with_pointclouds else None}
    try:
        device = torch.device("cuda:0")
        first_cuda_allocation = True
        for epoch, checkpoint in selected:
            if time.monotonic() - started > max_seconds:
                manifest["status"] = "partial_budget"
                break
            before_sha = sha256(checkpoint)
            saved = load_cpu_checkpoint(checkpoint)
            actual_epoch = checkpoint_epoch(saved, "best" if checkpoint.name == "best.pth" else None)
            if actual_epoch != epoch:
                raise ValueError("Saved checkpoint epoch differs from its declared snapshot")
            validate_training_ids(saved, training_ids)
            if saved.get("split_sha256") is not None and source_split_sha is not None and saved["split_sha256"] != source_split_sha:
                raise ValueError("Checkpoint split identity differs from the fixed source split")
            config = saved.get("training_config", source_config)
            mapping = resolve_proxy_mapping(saved, config)
            model = Hype_pointMLP()
            net = {key.removeprefix("module."): value for key, value in saved["net"].items()}
            model.load_state_dict(net, strict=True)
            model.requires_grad_(False)
            model.eval()
            c = float(model.manifold.c.detach().cpu())
            if not math.isfinite(c) or c <= 0:
                raise ValueError("Invalid saved whole curvature")
            if mapping and not math.isclose(c, float(config["c"]), rel_tol=1e-5, abs_tol=1e-5):
                raise ValueError("Whole and proxy curvatures differ; explicit adapter required")
            if time.monotonic() - started > max_seconds:
                raise TimeoutError("Export wall-time budget reached before GPU allocation")
            if first_cuda_allocation:
                # Recheck after potentially lengthy CPU shard/checkpoint loading,
                # immediately before the first allocation; never reject our own
                # CUDA context on subsequent snapshots.
                if idle_gpu(gpu) != uuid:
                    raise RuntimeError("Requested physical GPU identity changed")
                first_cuda_allocation = False
            model = model.to(device)
            before_state = state_digest(model.state_dict())
            reset_inference_rng(torch, np, seed)
            mus, logits = [], []
            with torch.no_grad():
                for start in range(0, len(ids), batch_size):
                    if time.monotonic() - started > max_seconds:
                        raise TimeoutError("Export wall-time budget reached within checkpoint")
                    cloud = torch.from_numpy(clouds[start:start + batch_size].copy()).to(device)
                    mu, scores = model(cloud.transpose(1, 2).contiguous())
                    if not bool(torch.isfinite(mu).all() and torch.isfinite(scores).all()):
                        raise ValueError("Nonfinite frozen clean forward")
                    mus.append(mu.cpu().numpy())
                    logits.append(scores.cpu().numpy())
            if state_digest(model.state_dict()) != before_state or any(parameter.grad is not None for parameter in model.parameters()):
                raise RuntimeError("Read-only export changed model/BN or created gradients")
            cache = {"sample_ids": ids, "labels": gold, "mu": np.concatenate(mus),
                     "logits": np.concatenate(logits), "curvature": np.asarray(c), "c": np.asarray(c),
                     "pool_ids_sha256": np.asarray(pool_sha), "input_sha256": np.asarray(input_sha),
                     "input_mode": np.asarray(input_mode),
                     "metadata_json": np.asarray(json.dumps({"inference_condition": inference_condition}))}
            proxy_state = saved.get("proxy", {})
            tangent_candidates = [value for key, value in proxy_state.items() if key.split(".")[-1] == "tangent_proxies"]
            if len(tangent_candidates) > 1:
                raise ValueError("Ambiguous saved proxy parameter tensors")
            tangent = tangent_candidates[0] if tangent_candidates else None
            if tangent is not None:
                cache["proxy_tangent"] = tangent.detach().cpu().numpy()
                cache["proxy_ids"] = np.arange(len(tangent), dtype=np.int64)
            path = out / ("e%03d_whole_cache.npz" % epoch)
            np.savez_compressed(path, **cache)
            if sha256(checkpoint) != before_sha:
                raise RuntimeError("Source checkpoint changed during read-only extraction")
            spec = {"epoch": epoch, "cache": str(path), "c": c, "run_key": audit_run_key(run),
                    "storage_run_id": run.name, "source_commit": source_commit,
                    "version": "v" + saved["format"].split("-v")[-1].split("-")[0] if "-v" in str(saved.get("format")) else None,
                    "display_name": source_manifest.get("canonical_experiment_name", run.name),
                    "tau": config.get("tau", .1), "margin": config.get("margin", .1),
                    "lambda_hier": config.get("lambda_hier_after_warmup", 0.) if epoch > config.get("warmup_epochs", 0) else 0.,
                    "exclude_self_negative": not config.get("self_negative", False),
                    "sample_K": config.get("sample_K", 20), "proxy_K": config.get("proxy_K", 20),
                    "training_config": config,
                    "label": "e%d" % epoch, "sample_pool_sha256": pool_sha,
                    "input_sha256": input_sha, "input_mode": input_mode,
                    "inference_condition": inference_condition,
                    "proxy_mapping": mapping,
                    "proxy_id_policy": "stable saved parameter rows within this source run",
                    "checkpoint_sha256": before_sha, "checkpoint_file": checkpoint.name,
                    "optimizer_checkpoint": str(checkpoint), "trusted_checkpoint": True}
            specs.append(spec)
            manifest["completed_checkpoint_identities"].append({"epoch": epoch, "file": checkpoint.name,
                "sha256": before_sha, "format": saved.get("format"), "source_commit": saved.get("commit", source_commit),
                "training_config_sha256": config_digest(config), "cache_sha256": sha256(path), "proxy_mapping": mapping})
            manifest["completed_snapshots"] = len(specs)
            write_specs()
            write_manifest()
            del model, saved, net, tangent, mus, logits, cache
            torch.cuda.empty_cache()
        else:
            manifest["status"] = "completed"
        return finish(manifest["status"])
    except TimeoutError:
        manifest["budget_stop_reason"] = "Wall-time budget reached; only fully completed snapshots are retained"
        return finish("partial_budget")
    except BaseException as exc:
        write_specs()
        manifest.update(status="failed", error=repr(exc), completed_snapshots=len(specs),
                        finished_utc=dt.datetime.now(dt.timezone.utc).isoformat())
        write_manifest()
        raise
