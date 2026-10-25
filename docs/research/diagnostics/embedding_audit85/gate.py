"""Bounded clean/eval regression against a previously frozen full cache.

Reads only ModelNet40 training shards. The actual CUDA run is explicit and
requires an idle GPU. No optimizer, backward, calibration or test data is used.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np

sys.dont_write_bytecode = True


def compare_embeddings(actual, reference, actual_prediction, reference_prediction, c=.1):
    """Return aggregate coordinate/direction/depth and prediction gate results."""
    actual, reference = np.asarray(actual, np.float64), np.asarray(reference, np.float64)
    if actual.shape != reference.shape or actual.ndim != 2 or not len(actual):
        raise ValueError("Actual/reference embeddings must have matching nonempty [N,D] shape")
    if not np.isfinite(actual).all() or not np.isfinite(reference).all():
        raise ValueError("Nonfinite actual/reference embeddings")
    a_norm, r_norm = np.linalg.norm(actual, axis=1), np.linalg.norm(reference, axis=1)
    if np.any(a_norm <= 1e-12) or np.any(r_norm <= 1e-12):
        raise ValueError("Zero/near-zero embedding direction")
    if np.any(math.sqrt(c)*a_norm >= 1) or np.any(math.sqrt(c)*r_norm >= 1):
        raise ValueError("Embedding outside its declared Poincare ball")
    difference = np.abs(actual-reference)
    cosine = np.sum((actual/a_norm[:, None])*(reference/r_norm[:, None]), axis=1)
    angle = np.degrees(np.arccos(np.clip(cosine, -1., 1.)))
    depth = 2/math.sqrt(c)*np.arctanh(math.sqrt(c)*a_norm)
    ref_depth = 2/math.sqrt(c)*np.arctanh(math.sqrt(c)*r_norm)
    depth_error = np.abs(depth-ref_depth)
    actual_prediction, reference_prediction = np.asarray(actual_prediction), np.asarray(reference_prediction)
    if actual_prediction.shape != (len(actual),) or reference_prediction.shape != (len(actual),):
        raise ValueError("Prediction vectors do not match the measured population")
    thresholds = dict(max_abs_coordinate=2e-4, mean_abs_coordinate=2e-6,
                      angle_p99_degrees=.05, depth_p99=.005, prediction_disagreements=0)
    measured = dict(max_abs_coordinate=float(difference.max()),
                    mean_abs_coordinate=float(difference.mean()),
                    angle_p99_degrees=float(np.quantile(angle, .99)),
                    angle_max_degrees=float(angle.max()),
                    depth_p99=float(np.quantile(depth_error, .99)),
                    depth_max=float(depth_error.max()),
                    prediction_disagreements=int(np.count_nonzero(actual_prediction != reference_prediction)))
    checks = {name: measured[name] <= value for name, value in thresholds.items()}
    return dict(passed=all(checks.values()), count=len(actual), dimension=actual.shape[1],
                c=c, finite=True, strictly_inside_ball=True, thresholds=thresholds,
                measured=measured, checks=checks)


def import_collector(repo):
    path = repo / "docs/research/diagnostics/v9_embedding_review_20261021/collect.py"
    spec = importlib.util.spec_from_file_location("embedding85_legacy_collector", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def canonical_contexts(selected, total, batch_size=32):
    """Keep each selected row in its historical full-export batch context.

    Returns canonical row arrays and offsets to measure within each array;
    extra context rows are forwarded but never added to the80-row statistics.
    """
    selected = np.asarray(selected)
    if (selected.ndim != 1 or not len(selected)
            or not np.issubdtype(selected.dtype, np.integer)
            or np.any(selected < 0) or np.any(selected >= total)
            or np.any(np.diff(selected) <= 0) or batch_size < 1):
        raise ValueError("Selected canonical IDs must be sorted, unique integers in range")
    contexts = []
    for start in np.unique((selected // batch_size)*batch_size):
        rows = np.arange(start, min(int(start)+batch_size, total), dtype=np.int64)
        offsets = selected[(selected >= start) & (selected < start+batch_size)]-start
        contexts.append((rows, offsets))
    return contexts


def run(args):
    started = time.monotonic()
    legacy = import_collector(args.repo)
    uuid = legacy.idle_uuid(args.gpu)
    os.environ["CUDA_VISIBLE_DEVICES"] = uuid
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    import h5py
    import torch
    if torch.cuda.is_initialized():
        raise RuntimeError("CUDA initialized before idle-resource check")
    torch.set_num_threads(2)
    legacy.make_context(args.repo)
    from inter_hierarchy_MN40.models.pointmlp import Hype_pointMLP
    from inter_hierarchy_MN40.hier_proxy_scratch_v8.geometry import configure_backbone
    checkpoint_before = legacy.sha(args.checkpoint)
    if checkpoint_before != args.expected_checkpoint_sha:
        raise ValueError("Actual checkpoint file SHA differs from the locked identity")
    reference_before = legacy.sha(args.reference_cache)
    with np.load(args.reference_cache, allow_pickle=False) as source:
        expected_mu = np.array(source["native_mu"], copy=True)
        expected_labels = np.array(source["labels"], copy=True).reshape(-1)
        expected_ids = np.array(source["canonical_ids"], copy=True).reshape(-1)
        prediction_key = next((key for key in ("native_prediction", "prediction", "whole_prediction") if key in source), None)
        expected_predictions = np.array(source[prediction_key], copy=True).reshape(-1) if prediction_key else None
    if expected_mu.shape != (9840, 256) or not np.array_equal(expected_ids, np.arange(9840)):
        raise ValueError("Golden cache must contain all9840 canonical sorted-shard rows")
    if expected_labels.shape != (9840,) or not np.issubdtype(expected_ids.dtype, np.integer):
        raise ValueError("Invalid golden identity arrays")
    shards = sorted(args.data_dir.glob("ply_data_train*.h5"))
    clouds, labels, shard_proof = [], [], []
    for shard in shards:
        with h5py.File(shard, "r") as handle:
            clouds.append(handle["data"][:, :1024].astype(np.float32))
            labels.append(handle["label"][:].reshape(-1).astype(np.int64))
        shard_proof.append(dict(name=shard.name, sha256=legacy.sha(shard), count=len(labels[-1])))
    clouds, labels = np.concatenate(clouds), np.concatenate(labels)
    if clouds.shape != (9840, 1024, 3) or not np.array_equal(labels, expected_labels):
        raise ValueError("Golden cache label/input population mismatch")
    input_sha = hashlib.sha256(clouds.tobytes()).hexdigest()
    if input_sha != args.expected_input_sha:
        raise ValueError("Clean first1024 input byte identity differs")
    if not np.array_equal(np.unique(labels), np.arange(40)):
        raise ValueError("Expected40 ModelNet40 classes")
    selected = np.sort(np.concatenate([np.flatnonzero(labels == c)[:2] for c in range(40)]))
    if selected.shape != (80,):
        raise ValueError("Expected exactly two fixed objects per class")
    contexts = canonical_contexts(selected, len(labels), 32)
    saved = legacy.load(args.checkpoint)
    if legacy.epoch_of(saved) != args.expected_epoch:
        raise ValueError("Golden checkpoint epoch differs")
    torch.manual_seed(22)
    np.random.seed(22)
    model = configure_backbone(Hype_pointMLP())
    model.load_state_dict({key.removeprefix("module."): value for key, value in saved["net"].items()}, strict=True)
    model.requires_grad_(False)
    model.eval()
    if not math.isclose(float(model.manifold.c.detach()), .1, rel_tol=1e-5):
        raise ValueError("TS13 reference model must have native c=.1")
    if legacy.idle_uuid(args.gpu) != uuid:
        raise RuntimeError("GPU identity changed before first allocation")
    device = torch.device("cuda:0")
    model.to(device)
    state_before = legacy.state_hash(model)
    torch.manual_seed(22)
    torch.cuda.manual_seed_all(22)
    np.random.seed(22)
    mus, predictions, ref_predictions = [], [], []
    with torch.no_grad():
        for ids, offsets in contexts:
            batch = torch.from_numpy(clouds[ids].copy()).to(device)
            mu, logits = model(batch.transpose(1, 2).contiguous())
            mus.append(mu.cpu().numpy()[offsets])
            predictions.append(logits.argmax(-1).cpu().numpy()[offsets])
            if expected_predictions is None:
                reference_mu = torch.from_numpy(expected_mu[ids].copy()).to(device)
                ref_predictions.append(model.classifier(reference_mu).argmax(-1).cpu().numpy()[offsets])
    actual_mu = np.concatenate(mus)
    actual_prediction = np.concatenate(predictions)
    reference_prediction = (expected_predictions[selected] if expected_predictions is not None
                            else np.concatenate(ref_predictions))
    report = compare_embeddings(actual_mu, expected_mu[selected], actual_prediction, reference_prediction)
    state_after = legacy.state_hash(model)
    unchanged = state_after == state_before and legacy.sha(args.checkpoint) == checkpoint_before
    reference_unchanged = legacy.sha(args.reference_cache) == reference_before
    no_grad = all(parameter.grad is None for parameter in model.parameters())
    input_unchanged = hashlib.sha256(clouds.tobytes()).hexdigest() == input_sha
    report.update(format="embedding85-frozen-server-regression-2", status="passed" if report["passed"] else "failed",
                  completed_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                  mode="clean first1024; eval saved BN; fixed first2 canonical objects/class in historical canonical batch32 contexts; no optimizer/backward/test",
                  selected_statistic_object_count=len(selected), historical_context_batch_size=32,
                  forwarded_context_object_count=sum(len(rows) for rows, _ in contexts),
                  forwarded_context_batch_count=len(contexts),
                  selected_canonical_id_sha256=hashlib.sha256(selected.astype(np.int64).tobytes()).hexdigest(),
                  expected_epoch=args.expected_epoch, checkpoint_sha256=checkpoint_before,
                  reference_cache_sha256=reference_before, input_sha256=input_sha,
                  label_sha256=hashlib.sha256(labels.tobytes()).hexdigest(), training_shards=shard_proof,
                  GPU_uuid=uuid, torch_version=torch.__version__, cuda_version=torch.version.cuda,
                  cudnn_benchmark=torch.backends.cudnn.benchmark,
                  cudnn_deterministic=torch.backends.cudnn.deterministic,
                  cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
                  matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                  float32_matmul_precision=torch.get_float32_matmul_precision(),
                  model_BN_hash_before=state_before, model_BN_hash_after=state_after,
                  model_BN_checkpoint_unchanged=unchanged, reference_cache_unchanged=reference_unchanged,
                  input_unchanged=input_unchanged, no_gradients=no_grad,
                  prediction_reference=("cache field " + prediction_key if prediction_key else
                                        "same frozen classifier applied to golden cached embeddings; coordinate gate is the independent feature reference"),
                  peak_allocated_MiB=torch.cuda.max_memory_allocated(device)/1024**2,
                  wall_seconds=time.monotonic()-started, script_sha256=legacy.sha(__file__))
    report["passed"] = bool(report["passed"] and unchanged and reference_unchanged and no_grad and input_unchanged)
    report["status"] = "passed" if report["passed"] else "failed"
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--reference-cache", type=Path, required=True)
    parser.add_argument("--expected-checkpoint-sha", required=True)
    parser.add_argument("--expected-input-sha", required=True)
    parser.add_argument("--expected-epoch", type=int, default=223)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("Gate output must be a new file")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    try:
        report = run(args)
    except Exception as exc:
        args.out.write_text(json.dumps(dict(passed=False, status="failed", error=repr(exc)), indent=2), encoding="utf-8")
        raise
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"status": report["status"], "count": report["count"], "measured": report["measured"]}), flush=True)
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
