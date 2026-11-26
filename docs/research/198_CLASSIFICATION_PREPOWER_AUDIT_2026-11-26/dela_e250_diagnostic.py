"""Read-only fixed-e250 paired predictions; writes only a fresh diagnostic directory."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
from types import SimpleNamespace


sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
EXPECTED_CHECKPOINTS = {
    "h0": "e1b369e8f0bb073fad3d0b29db21c6fbe420b0baf73f459b28644f36a8bd31b8",
    "hier": "6aff1622d1dfcd3f586059218e5f6f1d5a47628b2e2585eebf063b848feee517",
}


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def idle_device(index):
    query = subprocess.check_output([
        "nvidia-smi", "-i", str(index), "--query-gpu=index,uuid,name,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ], text=True, timeout=20).strip().split(",")
    physical, uuid, name, memory, utilization = [x.strip() for x in query]
    apps = subprocess.check_output([
        "nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits",
    ], text=True, timeout=20)
    if int(memory) != 0 or int(utilization) != 0 or uuid in apps:
        raise RuntimeError("Assigned GPU is not completely idle")
    return dict(index=int(physical), uuid=uuid, name=name)


def load_frozen(args, recipe):
    folder = args.frozen_root / "docs/research/diagnostics/dela_h0_20261125"
    for filename, expected in recipe["source_sha256"].items():
        if sha(folder / filename) != expected:
            raise RuntimeError("Frozen adapter hash mismatch: " + filename)
    for filename, expected in recipe["dela_source_sha256"].items():
        if sha(args.dela_root / "ScanObjectNN" / filename) != expected:
            raise RuntimeError("Frozen author hash mismatch: " + filename)
    sys.path[:0] = [str(folder), str(args.frozen_root), str(args.frozen_root / "pointnet2_ops_lib")]
    spec = importlib.util.spec_from_file_location("frozen_dela_training", folder / "run.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def restore_rng(state, torch, np):
    random.setstate(state["random_rng"])
    np.random.set_state(state["numpy_rng"])
    torch.set_rng_state(state["torch_rng"].cpu())
    torch.cuda.set_rng_state(state["cuda_rng"].cpu())


def evaluate_arm(args, frozen, config, dataset, make_model, mode, deadline):
    import numpy as np
    import torch
    from torch.utils.data import DataLoader

    folder = args.formal_root / mode / "work"
    checkpoint = folder / "last.pt"
    checkpoint_sha = sha(checkpoint)
    if checkpoint_sha != EXPECTED_CHECKPOINTS[mode]:
        raise RuntimeError("Locked e250 checkpoint identity changed: " + mode)
    original = read(folder / "test_e250.json")
    manifest = read(folder / "manifest.json")
    if manifest["selection"] != "fixed e250; evaluate official test once":
        raise RuntimeError("Unexpected original endpoint protocol")
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if state["epoch"] != 250:
        raise RuntimeError("Checkpoint is not e250")
    model = make_model(config.dela_args, manifest["seed"] + 100_001,
                       exp_input_scale=manifest["input_scale"]).cuda()
    model.load_state_dict(state["model"], strict=True)
    model.eval()
    before = frozen.state_hash(model.state_dict())
    # The original checkpoint was saved immediately before the terminal evaluate().
    restore_rng(state, torch, np)
    eval_data = dataset(partition="test")
    if len(eval_data) != 2882:
        raise RuntimeError("Unexpected official test length")
    loader = DataLoader(frozen.WithRows(eval_data), batch_size=config.batch_size,
                        pin_memory=True, num_workers=args.workers)
    confusion = torch.zeros((15, 15), dtype=torch.int64, device="cuda")
    input_hash = hashlib.sha256()
    rows = []
    with torch.no_grad():
        for xyz, labels, ids in loader:
            if time.monotonic() > deadline:
                raise TimeoutError("Diagnostic time limit exceeded")
            input_hash.update(xyz.contiguous().numpy().tobytes())
            input_hash.update(labels.contiguous().numpy().tobytes())
            with torch.autocast("cuda", dtype=torch.float16):
                scores = model(xyz.cuda(non_blocking=True))
            y = labels.cuda(non_blocking=True).view(-1).long()
            pred = scores.argmax(1)
            if not bool(torch.isfinite(scores).all()):
                raise RuntimeError("Nonfinite diagnostic output")
            confusion += torch.bincount(y * 15 + pred, minlength=225).view(15, 15)
            probability = scores.float().softmax(dim=1)
            for row_id, gt, prediction, logits, probabilities in zip(
                ids.tolist(), y.cpu().tolist(), pred.cpu().tolist(),
                scores.float().cpu().tolist(), probability.cpu().tolist(),
            ):
                rows.append(dict(row_id=row_id, label=gt, prediction=prediction,
                                 confidence=probabilities[prediction], scores=logits,
                                 probabilities=probabilities))
    after = frozen.state_hash(model.state_dict())
    if before != after:
        raise RuntimeError("Parameters or BN buffers changed during evaluation")
    if [x["row_id"] for x in rows] != list(range(2882)):
        raise RuntimeError("Test row identity or coverage mismatch")
    correct, counts = confusion.diag(), confusion.sum(1)
    result = dict(
        mode=mode, checkpoint_epoch=state["epoch"], checkpoint_sha256=checkpoint_sha,
        checkpoint_bytes=checkpoint.stat().st_size, updates=state["updates"], skipped=state["skipped"],
        input_sha256=input_hash.hexdigest(), parameter_bn_sha256_before=before,
        parameter_bn_sha256_after=after, rows=len(rows),
        oa=100 * float(correct.sum() / counts.sum()),
        aa=100 * float((correct / counts).mean()), confusion=confusion.cpu().tolist(),
        original_test_sha256=sha(folder / "test_e250.json"),
        original_oa=original["oa"], original_aa=original["aa"],
        original_confusion_exact_match=confusion.cpu().tolist() == original["confusion"],
        manifest_sha256=sha(folder / "manifest.json"), seed=manifest["seed"],
        source_revision=manifest["revision"], amp="float16, hyperbolic FP32 as frozen model",
    )
    write(args.output / (mode + "_predictions.json"), rows)
    write(args.output / (mode + "_summary.json"), result)
    del state, model, confusion
    gc.collect()
    torch.cuda.empty_cache()
    return result, rows


def paired_results(left, right):
    if [(x["row_id"], x["label"]) for x in left] != [(x["row_id"], x["label"]) for x in right]:
        raise RuntimeError("Paired test row order or labels differ")
    by_class = []
    for label in range(15):
        pairs = [(a, b) for a, b in zip(left, right) if a["label"] == label]
        by_class.append(dict(
            label=label, rows=len(pairs), h0_correct=sum(a["prediction"] == label for a, _ in pairs),
            hier_correct=sum(b["prediction"] == label for _, b in pairs),
            hier_only_correct=sum(a["prediction"] != label and b["prediction"] == label for a, b in pairs),
            h0_only_correct=sum(a["prediction"] == label and b["prediction"] != label for a, b in pairs),
        ))
    both = sum(a["prediction"] == a["label"] and b["prediction"] == b["label"] for a, b in zip(left, right))
    only_h0 = sum(x["h0_only_correct"] for x in by_class)
    only_hier = sum(x["hier_only_correct"] for x in by_class)
    return dict(both_correct=both, h0_only_correct=only_h0, hier_only_correct=only_hier,
                both_wrong=len(left) - both - only_h0 - only_hier,
                net_additional_correct=only_hier - only_h0, classes=by_class)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal-root", type=Path, required=True)
    parser.add_argument("--frozen-root", type=Path, required=True)
    parser.add_argument("--dela-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--diagnostic-commit", required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--time-limit-seconds", type=int, default=600)
    args = parser.parse_args()
    if not 0 < args.time_limit_seconds <= 1200:
        parser.error("Diagnostic time limit must be between 1 and 1200 seconds")
    args.output.mkdir(parents=True, exist_ok=False)
    status = dict(status="starting", started_at=now(), diagnostic_commit=args.diagnostic_commit,
                  script_sha256=sha(Path(__file__)), training_updates=0,
                  protocol="independent read-only diagnostic of locked e250; no selection")
    write(args.output / "status.json", status)
    deadline = time.monotonic() + args.time_limit_seconds
    try:
        with open(f"/tmp/hycore_{os.getuid()}_gpu_{args.gpu}.lock", "a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            device = idle_device(args.gpu)
            time.sleep(2)
            if idle_device(args.gpu) != device:
                raise RuntimeError("Assigned GPU identity changed")
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = device["uuid"]
            import torch
            torch.set_num_threads(2)
            torch.set_float32_matmul_precision("high")
            recipe = read(args.formal_root / "recipe.json")
            frozen = load_frozen(args, recipe)
            config, dataset, make_model, _, _ = frozen.source_imports(
                SimpleNamespace(dela_root=args.dela_root, data_root=args.data_root))
            test_path = args.data_root / "main_split/test_objectdataset_augmentedrot_scale75.h5"
            if sha(test_path) != frozen.TEST_SHA:
                raise RuntimeError("Official test data hash mismatch")
            status.update(device=device, original_source_commit=recipe["source_commit"],
                          recipe_sha256=sha(args.formal_root / "recipe.json"),
                          test_sha256=frozen.TEST_SHA, torch=str(torch.__version__),
                          cuda=torch.version.cuda)
            write(args.output / "status.json", status)
            summaries, predictions = [], []
            for mode in ("h0", "hier"):
                summary, rows = evaluate_arm(args, frozen, config, dataset, make_model, mode, deadline)
                summaries.append(summary)
                predictions.append(rows)
            same_inputs = summaries[0]["input_sha256"] == summaries[1]["input_sha256"]
            pair = paired_results(*predictions) if same_inputs else dict(error="Input fingerprints differ")
            pair["same_input_fingerprint"] = same_inputs
            pair["both_original_confusions_reproduced"] = all(x["original_confusion_exact_match"] for x in summaries)
            write(args.output / "paired_summary.json", pair)
            status.update(status="completed" if same_inputs and pair["both_original_confusions_reproduced"]
                          else "diagnostic_mismatch", finished_at=now(), summaries=summaries)
            write(args.output / "status.json", status)
    except Exception as ex:
        status.update(status="failed", finished_at=now(), error=type(ex).__name__ + ": " + str(ex))
        write(args.output / "status.json", status)
        raise


if __name__ == "__main__":
    main()
