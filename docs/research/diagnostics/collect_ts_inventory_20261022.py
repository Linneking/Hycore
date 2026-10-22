"""Read saved experiment JSON only; never import Torch or run inference.

Supply private ID/path plans separately through ``__plans__`` when executing.
The output deliberately excludes paths, commands, devices, sample IDs and arrays.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import statistics
from pathlib import Path


def load(path):
    raw = path.read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def accuracy(record, prefix):
    record = record or {}
    return {"oa": record.get(prefix + "oa", record.get("acc")),
            "aa": record.get(prefix + "aa", record.get("acc_avg")),
            "ce_or_logged_loss": record.get(prefix + "ce", record.get("loss")),
            "count": record.get(prefix + "count")}


def compact_epoch(record, filename):
    training = (record.get("telemetry") or {}).get("training") or {}
    if training:
        online = {"oa": 100 * training["oa"],
                  "aa": 100 * training["aa_observed_classes"],
                  "ce_or_logged_loss": None,
                  "count": sum(training["per_class_positions"])}
    else:
        online = accuracy(record.get("train"), "")
    return {"completed_epoch": record.get("completed_epochs", record["epoch"]),
            "raw_epoch": record["epoch"], "source_file": filename,
            "online_train": online,
            "clean_train": accuracy(record.get("clean_train"), "train_eval_"),
            "test": accuracy(record.get("test"), "test_")}


def collect_one(identifier, directory):
    root = Path(directory)
    manifest, digest = load(root / "manifest.json")
    files = sorted(root.glob("metrics_epoch_*.json"))
    completed = manifest.get("completed_epochs")
    if completed is not None:
        files = [path for path in files if int(path.stem.rsplit("_", 1)[1]) <= completed]
    if not files:
        raise ValueError("no saved epoch metrics")
    selected = manifest.get("best") or {}
    best_epoch = selected.get("epoch", manifest.get("best_completed_epochs"))
    if best_epoch is None and manifest.get("best_epoch") is not None:
        best_epoch = manifest["best_epoch"] + 1
    best_file = root / ("metrics_epoch_%03d.json" % best_epoch)
    best_record, best_sha = load(best_file)
    last_record, last_sha = load(files[-1])
    tail = [compact_epoch(load(path)[0], path.name) for path in files[-10:]]
    clean_records = []
    for path in reversed(files):
        record = load(path)[0]
        if record.get("clean_train"):
            clean_records.append(compact_epoch(record, path.name))
            break
    numbers = [int(path.stem.rsplit("_", 1)[1]) for path in files]
    values = [row["test"]["oa"] for row in tail]
    config = manifest.get("training_config") or {}
    config_keys = ("proxy_optimizer", "hier_view", "hier_view_depth", "proxy_tangent_cap",
                   "geometry_domains", "sampler", "steps_per_epoch", "epochs", "seed",
                   "batch", "global_batch", "local_batch", "native_extra_cap",
                   "native_inverse_metric_hook", "hier_inverse_metric_hook")
    return {"id": identifier, "status": manifest.get("status"),
            "production_commit": manifest.get("commit"),
            "completed_epochs": manifest.get("completed_epochs"),
            "train_count": manifest.get("train_size"),
            "validation_count": manifest.get("validation_size"),
            "test_count": manifest.get("selection_size", manifest.get("test_size")),
            "started": manifest.get("started_utc", manifest.get("started_at")),
            "finished": manifest.get("finished_utc", manifest.get("finished_at")),
            "manifest_sha256_at_capture": digest,
            "config_excerpt": {key: config[key] for key in config_keys if key in config},
            "best_epoch": best_epoch,
            "manifest_best_test_oa": selected.get("test_oa", manifest.get("best_test_OA")),
            "best": compact_epoch(best_record, best_file.name),
            "best_metric_sha256": best_sha,
            "latest": compact_epoch(last_record, files[-1].name),
            "latest_metric_sha256": last_sha,
            "latest_clean_train_record": clean_records[0] if clean_records else None,
            "metric_files": len(files),
            "metric_files_contiguous_from_one": numbers == list(range(1, numbers[-1] + 1)),
            "tail_test_oa_mean": statistics.mean(values),
            "tail_test_oa_population_std": statistics.pstdev(values),
            "tail": tail}


def collect(plans):
    rows = []
    for identifier, directory in plans.items():
        try:
            rows.append(collect_one(identifier, directory))
        except Exception as error:
            rows.append({"id": identifier, "error_type": type(error).__name__})
    return {"schema_version": 1,
            "observed_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "scope": "Saved JSON metadata/scalars only; no training, inference or GPU allocation",
            "accuracy_unit": "percent", "rows": rows}


if "__plans__" in globals():
    print(json.dumps(collect(__plans__), ensure_ascii=False, allow_nan=False))
