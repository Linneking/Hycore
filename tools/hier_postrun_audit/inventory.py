"""Read-only, standard-library inventory of versioned HyCoRe run artifacts.

No checkpoint is deserialized and steps.jsonl is never scanned implicitly.
Public inventory values use relative names; private server paths stay local.
"""
from __future__ import annotations

import json
import hashlib
import math
from pathlib import Path
import re
from typing import Iterator


_EPOCH_FILE = re.compile(r"metrics_epoch_(\d+)\.json$")
_CHECKPOINT_EPOCH = re.compile(r"(?:checkpoint|epoch)[_-](?:epoch[_-])?(\d+)")


def _valid_epoch(value):
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and value >= 0 and int(value) == value)


def resolve_run(path: str | Path) -> Path:
    """Resolve a run or a root containing exactly one arm; reject ambiguity."""
    candidate = Path(path).expanduser().resolve()
    if not candidate.is_dir():
        raise ValueError("Run input must be an existing directory")
    if _is_run(candidate):
        return candidate
    # Launch roots sometimes contain an arms/ or runs/ directory. Deliberately
    # bounded discovery avoids wandering through checkpoints/datasets.
    found = [child for child in candidate.iterdir() if child.is_dir() and _is_run(child)]
    for container in ("arms", "runs"):
        directory = candidate / container
        if directory.is_dir():
            found.extend(child for child in directory.iterdir() if child.is_dir() and _is_run(child))
    found = sorted(set(found))
    if len(found) != 1:
        raise ValueError("Expected one run arm; found %d. Pass an explicit arm directory." % len(found))
    return found[0]


def _is_run(path: Path) -> bool:
    return (path / "manifest.json").is_file() and (
        (path / "metrics.jsonl").is_file() or any(path.glob("metrics_epoch_*.json"))
        or (path / "heartbeat.json").is_file())


# Explicit alias used by checkpoint extraction without changing path rules.
resolve_run_dir = resolve_run


def audit_run_key(path):
    """Public stable identity: an arm basename plus hashed parent namespace.

    H20 is shared by multiple versions. Never align runs by that basename alone.
    The private absolute path is not part of the exported identifier.
    """
    run = Path(path).resolve()
    namespace = run.parent.name + "/" + run.name
    return run.name + "_" + hashlib.sha256(namespace.encode("utf-8")).hexdigest()[:12]

def read_json(path: Path, warnings: list[str] | None = None) -> dict:
    """Read a small JSON object, preserving malformed/missing as unavailable."""
    if not path.is_file():
        return {}
    try:
        with path.open("r", encoding="utf-8-sig") as stream:
            value = json.load(stream)
        if not isinstance(value, dict):
            raise ValueError("Expected JSON object")
        return value
    except (OSError, ValueError, json.JSONDecodeError) as error:
        if warnings is not None:
            warnings.append("Cannot read %s: %s" % (path.name, type(error).__name__))
        return {}


def iter_epoch_rows(run_dir: Path, warnings: list[str] | None = None) -> Iterator[tuple[dict, str]]:
    """Stream epoch rows, preferring the append-only log over duplicate files."""
    log = run_dir / "metrics.jsonl"
    if log.is_file():
        with log.open("r", encoding="utf-8-sig") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    if not isinstance(row, dict) or not _valid_epoch(row.get("epoch")):
                        raise ValueError("Epoch row must contain numeric epoch")
                    yield row, "metrics.jsonl:%d" % line_number
                except (ValueError, json.JSONDecodeError):
                    if warnings is not None:
                        warnings.append("Malformed metrics.jsonl line %d skipped" % line_number)
        return
    files = sorted(run_dir.glob("metrics_epoch_*.json"),
                   key=lambda item: int(_EPOCH_FILE.search(item.name).group(1)) if _EPOCH_FILE.search(item.name) else -1)
    for file in files:
        row = read_json(file, warnings)
        if row:
            if "epoch" not in row:
                matched = _EPOCH_FILE.search(file.name)
                if matched:
                    row["epoch"] = int(matched.group(1))
            if not _valid_epoch(row.get("epoch")):
                if warnings is not None:
                    warnings.append("Invalid epoch in %s skipped" % file.name)
                continue
            yield row, file.name


def _files(run_dir: Path, patterns: tuple[str, ...]) -> list[dict]:
    files = set()
    for pattern in patterns:
        files.update(run_dir.glob(pattern))
    result = []
    for file in sorted(files):
        if not file.is_file():
            continue
        match = _CHECKPOINT_EPOCH.search(file.stem)
        if match is None:
            match = re.search(r"(?:initialization|initial)[_-]e(\d+)$", file.stem)
        result.append({"filename": file.relative_to(run_dir).as_posix(),
                       "size_bytes": file.stat().st_size,
                       "epoch_from_filename": int(match.group(1)) if match else None})
    return result


def inventory_run(path: str | Path) -> dict:
    """Return cheap capabilities and recorded epochs without loading weights."""
    run_dir = resolve_run(path)
    warnings: list[str] = []
    manifest = read_json(run_dir / "manifest.json", warnings)
    epochs, duplicated, seen = [], [], set()
    capabilities = {"epoch_scalar_logs": False, "training_geometry": False,
                    "actual_training_ancestor_counts": False, "clean_evaluation": False,
                    "per_sample_training_draw_counts": False, "structure_depth_gaps": False}
    for row, _source in iter_epoch_rows(run_dir, warnings):
        epoch = int(row["epoch"])
        if epoch in seen:
            duplicated.append(epoch)
        else:
            epochs.append(epoch)
            seen.add(epoch)
        telemetry = row.get("telemetry") or {}
        if not telemetry and isinstance(row.get("training"), dict) and "batches" in row["training"]:
            telemetry = row["training"]
        capabilities["epoch_scalar_logs"] = True
        capabilities["training_geometry"] |= bool(telemetry.get("geometry"))
        capabilities["clean_evaluation"] |= bool(row.get("validation") or row.get("clean_train") or row.get("train_clean"))
        capabilities["per_sample_training_draw_counts"] |= bool((telemetry.get("sample_coverage") or {}).get("id_draw_counts"))
        structure = row.get("structure") or {}
        capabilities["actual_training_ancestor_counts"] |= any(
            bool((value or {}).get("selected_ancestor_usage")) for value in structure.values() if isinstance(value, dict))
        capabilities["structure_depth_gaps"] |= any(
            bool((value or {}).get("depth_distributions")) for value in structure.values() if isinstance(value, dict))
    epochs.sort()
    if duplicated:
        warnings.append("Duplicate epoch records: %s; normalization retains the last logged record." % sorted(set(duplicated)))
    if epochs:
        gaps = sorted(set(range(epochs[0], epochs[-1] + 1)) - set(epochs))
        if gaps:
            warnings.append("Missing logged epochs within observed span: %s" % gaps)
        if epochs[0] > 1:
            warnings.append("Logs start at epoch %d; earlier history is not reconstructed as observations." % epochs[0])
    else:
        gaps = []
        warnings.append("No parseable epoch records found")
    completed = manifest.get("completed_epochs")
    if epochs and _valid_epoch(completed) and epochs[-1] != int(completed):
        warnings.append("Latest logged epoch differs from manifest.completed_epochs; investigate incomplete or stale artifacts.")
    if manifest.get("status") not in (None, "completed"):
        warnings.append("Manifest status is %s; report is a partial snapshot, not a completed-run certificate." % manifest["status"])
    checkpoints = _files(run_dir, ("*.pth", "*.pt", "checkpoints/*.pth", "checkpoints/*.pt"))
    snapshots = _files(run_dir, ("*features*.npz", "*embedding*.npz", "*snapshot*.npz", "features/*.npz", "snapshots/*.npz"))
    capabilities.update(checkpoint_files_present=bool(checkpoints),
                        fixed_embedding_snapshot_files_present=bool(snapshots),
                        step_log_available=(run_dir / "steps.jsonl").is_file(),
                        step_log_scanned=False,
                        checkpoint_contents_verified=False,
                        exact_sample_proxy_neighbor_stability=False,
                        fixed_snapshot_candidate_files_present=bool(snapshots),
                        class_or_morphology_validation=False)
    steps = run_dir / "steps.jsonl"
    return {"run_id": run_dir.name, "status": manifest.get("status"),
            "recorded_epoch_count": len(epochs), "recorded_epochs": epochs,
            "first_recorded_epoch": epochs[0] if epochs else None,
            "last_recorded_epoch": epochs[-1] if epochs else None,
            "missing_epochs_within_logged_span": gaps,
            "duplicate_epochs": sorted(set(duplicated)),
            "manifest_completed_epochs": manifest.get("completed_epochs"),
            "metrics_source": "metrics.jsonl" if (run_dir / "metrics.jsonl").is_file() else "metrics_epoch_*.json",
            "checkpoint_files": checkpoints, "embedding_snapshot_files": snapshots,
            "steps_file_bytes": steps.stat().st_size if steps.is_file() else None,
            "capabilities": capabilities, "warnings": warnings}
