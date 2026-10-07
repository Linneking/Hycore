"""Fresh public delivery of completed read-only audits and real backbone probes.

An allowlist excludes weights, NPZ, raw logs, working specs, normalized_runs and
private/raw manifests. Only the new delivery is written. Every copied source is
hashed before/after. Cross-run bars compare within-run scalar ratios, never
parameter rows. Input matching gates any joint interpretation.
"""
from __future__ import annotations
import csv
import datetime as dt
import hashlib
import html
import io
import json
import math
from pathlib import Path
import re
import shutil
import subprocess

ROOT_PUBLIC = {
    "report_data.json", "comparison_summary.json", "longitudinal_summary.json",
    "comparison_benefit_ledger.csv", "comparison_aligned_observations.csv",
    "epochs.csv", "proxy_lifetimes.csv", "proxy_lifetime_epochs.csv",
}
MODULE_PUBLIC = ("mechanisms", "structure", "redundancy")
MARKER = "backbone-probe-overview"


def public_text(value):
    from .report import _public_text
    text = _public_text(value)
    # Connection identifiers may be adjacent to human labels ("host10.0.0.1").
    text = re.sub(r"(?<![0-9.])(?:[0-9]{1,3}\.){3}[0-9]{1,3}(?![0-9.])", "<private-host>", text)
    text = re.sub(r"[A-Za-z0-9_.-]+@<private-host>", "<private-connection>", text)
    return text


def sanitize(value):
    if isinstance(value, dict):
        return {public_text(key): sanitize(child) for key, child in value.items()}
    if isinstance(value, list):
        return [sanitize(child) for child in value]
    if isinstance(value, tuple):
        return [sanitize(child) for child in value]
    if isinstance(value, str):
        return public_text(value)
    return value


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1048576), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _write(path, value):
    Path(path).write_text(json.dumps(sanitize(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _copy_public(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.suffix.lower() == ".json":
        _write(target, _read(source))
    elif source.suffix.lower() == ".csv":
        stream = io.StringIO(source.read_text(encoding="utf-8-sig"))
        rows = [[public_text(cell) for cell in row] for row in csv.reader(stream)]
        with target.open("x", encoding="utf-8", newline="") as handle:
            csv.writer(handle).writerows(rows)
    elif source.suffix.lower() in (".svg", ".html"):
        text = public_text(source.read_text(encoding="utf-8-sig"))
        for name in ("path", "host", "id", "connection"):
            text = text.replace("<private-" + name + ">", "&lt;private-" + name + "&gt;")
        target.write_text(text, encoding="utf-8")
    else:
        shutil.copyfile(source, target)


def _gradient_method(record):
    return record.get("gradient_method") or record.get("input_identity", {}).get("gradient_method")


def _replay_check(record):
    """Check producer replay evidence and exact object coverage for the VJP path."""
    method = _gradient_method(record)
    if method in ("ordinary_autograd", "single_pass_autograd", "single_graph_autograd"):
        return None
    if method != "two_pass_feature_adjoint_vjp":
        return "Explicit supported gradient_method is missing or unknown"
    identity = record.get("input_identity", {})
    if identity.get("gradient_method") != method or record.get("input_condition", {}).get("gradient_method") != method:
        return "Top-level, input and condition gradient_method identities disagree"
    replay = record.get("replay_validation", {})
    if (replay.get("available") is not True or replay.get("passed") is not True or
            replay.get("same_actual_forward_RNG") is not True):
        return "Two-pass replay validation or exact forward RNG verification failed/missing"
    microbatch = identity.get("microbatch_size")
    if not isinstance(microbatch, int) or isinstance(microbatch, bool) or microbatch < 1:
        return "Two-pass replay microbatch_size is invalid"
    expected = 2 * ((64 + microbatch - 1) // microbatch)
    records = replay.get("records")
    if (replay.get("forward_replays") != expected or replay.get("expected_forward_replays") != expected or
            not isinstance(records, list) or len(records) != expected):
        return "Two-pass replay count does not cover both roles and64 objects"
    for key in ("rtol", "atol", "max_absolute_feature_difference"):
        value = replay.get(key)
        if not isinstance(value, (float, int)) or not math.isfinite(value) or value < 0:
            return "Two-pass replay numerical validation values are invalid: " + key
    coverage = {"whole": set(), "child": set()}
    for item in records:
        role, start, end = item.get("role"), item.get("start"), item.get("end")
        difference = item.get("max_absolute_difference")
        if (role not in coverage or not isinstance(start, int) or not isinstance(end, int) or
                not 0 <= start < end <= 64 or
                not isinstance(difference, (float, int)) or not math.isfinite(difference) or difference < 0):
            return "Two-pass replay role/object range/numerical evidence is invalid"
        positions = set(range(start, end))
        if positions & coverage[role]:
            return "Two-pass replay has overlapping object coverage"
        coverage[role].update(positions)
    if any(values != set(range(64)) for values in coverage.values()):
        return "Two-pass replay object coverage is incomplete"
    measured = max(item["max_absolute_difference"] for item in records)
    if not math.isclose(float(replay["max_absolute_feature_difference"]), measured, rel_tol=1e-7, abs_tol=1e-12):
        return "Two-pass maximum replay difference disagrees with its records"
    return None


def _probe(directory):
    directory = Path(directory).resolve()
    manifest_path = directory / "backbone_probe_manifest.json"
    summary_path = directory / "backbone_probe_summary.json"
    status = {"probe_directory": directory.name, "available": False,
              "status": "missing", "source_files": []}
    if not manifest_path.is_file():
        status["reason"] = "Probe manifest is missing; no success is inferred"
        return status, {}, []
    manifest = _read(manifest_path)
    paths = [manifest_path]
    status.update(status=manifest.get("status", "unknown"),
                  epoch=manifest.get("epoch"), run_key=manifest.get("run_key"),
                  budget_seconds=manifest.get("max_seconds"), elapsed_seconds=manifest.get("elapsed_seconds"),
                  source_checkpoint_sha256=manifest.get("source_checkpoint_sha256"),
                  audit_code_commit=manifest.get("audit_code_commit"),
                  source_files=[{"file": manifest_path.name, "sha256": _sha(manifest_path)}])
    if manifest.get("status") != "completed" or not summary_path.is_file():
        status["reason"] = public_text(manifest.get("error") or "No completed probe summary; failed/running attempts are not evidence")
        return status, {}, paths
    paths.append(summary_path)
    result = _read(summary_path)
    status["source_files"].append({"file": summary_path.name, "sha256": _sha(summary_path)})
    read_only = result.get("read_only", {})
    invariants = (
        manifest.get("source_unchanged") is True,
        read_only.get("optimizer_updates") == 0,
        read_only.get("new_validation_forwards") == 0,
        read_only.get("new_test_forwards") == 0,
        read_only.get("BN_parameters_grad_buffers_unchanged") is True,
        read_only.get("RNG_restored") is True,
        read_only.get("source_clouds_unchanged") is True,
    )
    if not all(invariants):
        status["reason"] = "Completed summary lacks required read-only/source invariants"
        return status, {}, paths
    identity, condition = result.get("input_identity", {}), result.get("input_condition", {})
    ids, labels = identity.get("sample_ids"), identity.get("labels")
    required_identity = ("input_sha256", "whole_count", "child_count", "microbatch_size",
                         "whole_centers", "child_centers", "crop_seed", "mode")
    if (not isinstance(ids, list) or len(ids) != 64 or len(set(ids)) != 64 or
            not isinstance(labels, list) or len(labels) != 64 or
            any(identity.get(key) is None for key in required_identity) or
            "eval" not in str(identity.get("mode", "")).lower() or
            "bn" not in str(identity.get("mode", "")).lower() or
            not condition):
        status["reason"] = "Input identity lacks64 unique ordered objects, labels, explicit crop/eval conditions"
        return status, {}, paths
    for key in ("whole_count", "child_count", "microbatch_size"):
        if condition.get(key) != identity[key]:
            status["reason"] = "Input identity and inference conditions disagree: " + key
            return status, {}, paths
    if condition.get("seed") != identity["crop_seed"]:
        status["reason"] = "Actual crop seed and input condition seed disagree"
        return status, {}, paths
    if manifest.get("source_checkpoint_sha256") != result.get("checkpoint_sha256"):
        status["reason"] = "Manifest and measured source checkpoint hashes disagree"
        return status, {}, paths
    replay_error = _replay_check(result)
    if replay_error:
        status["reason"] = replay_error
        return status, {}, paths
    encoder = result.get("parameter_groups", {}).get("shared_encoder", {})
    if not isinstance(encoder.get("hier_vs_base"), dict):
        status["reason"] = "Shared-encoder HIER/base comparison was not measured"
        return status, {}, paths
    public = {key: result.get(key) for key in (
        "format", "scope", "run_key", "display_name", "epoch", "checkpoint_sha256",
        "source_checkpoint_file", "source_commit", "model_source_sha256", "train_labels_sha256",
        "recorded_train_labels_identity_verified", "lambda_hier", "objective_values",
        "parameter_groups", "input_identity", "input_condition", "microbatches", "read_only",
        "gradient_method", "replay_validation", "gradient_identity")}
    public["source_probe_directory"] = directory.name
    public["audit_code_commit"] = manifest.get("audit_code_commit")
    public["proxy_objective_shared_encoder_norm"] = encoder.get("norms", {}).get("proxy")
    public["proxy_objective_encoder_note"] = "Proxy-proxy HIER has no direct shared-encoder dependency; norm0 is expected under this explicit objective graph. It can still change later proxy parameters and future sample relations."
    status.update(available=True, status="completed", display_name=result.get("display_name"),
                  run_key=result.get("run_key"), epoch=result.get("epoch"), reason=None)
    return status, public, paths


def joint_input_identity(records):
    """Require identical real inputs/crops/eval conditions before joint claims."""
    if len(records) < 2:
        return {"available": False, "reason": "At least two successful real probes are required",
                "matched_fields": []}
    reference = records[0]
    identity_keys = ("sample_ids", "labels", "input_sha256", "whole_count", "child_count",
                     "microbatch_size", "whole_centers", "child_centers", "crop_seed", "mode",
                     "classifier_on_concatenated_mu", "activation_checkpoint", "checkpoint_early_stop", "gradient_method")
    for record in records:
        replay_error = _replay_check(record)
        if replay_error:
            return {"available": False, "reason": replay_error, "matched_fields": []}
    if reference["input_identity"].get("activation_checkpoint"):
        identity_keys = identity_keys + ("checkpoint_use_reentrant", "checkpoint_preserve_rng_state")
    for candidate in records[1:]:
        if _gradient_method(candidate) != _gradient_method(reference):
            return {"available": False, "reason": "Real probe gradient_method differs", "matched_fields": []}
        for key in identity_keys:
            if reference["input_identity"].get(key) is None or candidate["input_identity"].get(key) != reference["input_identity"].get(key):
                return {"available": False, "reason": "Real probe input conditions are missing or differ: " + key,
                        "matched_fields": []}
        if _gradient_method(reference) == "two_pass_feature_adjoint_vjp":
            for field in ("rtol", "atol", "same_actual_forward_RNG"):
                if candidate["replay_validation"].get(field) != reference["replay_validation"].get(field):
                    return {"available": False, "reason": "Two-pass replay validation conditions differ: " + field, "matched_fields": []}
            shape = lambda record: [(item["role"], item["start"], item["end"]) for item in record["replay_validation"]["records"]]
            if shape(candidate) != shape(reference):
                return {"available": False, "reason": "Two-pass replay object-role partitions differ", "matched_fields": []}
        if candidate["input_condition"] != reference["input_condition"]:
            return {"available": False, "reason": "Real probe input_condition objects differ", "matched_fields": []}
        if candidate.get("model_source_sha256") != reference.get("model_source_sha256") or not reference.get("model_source_sha256"):
            return {"available": False, "reason": "Measured model source identity differs or is missing", "matched_fields": []}
    return {"available": True, "object_count": 64, "matched_fields": list(identity_keys) + ["input_condition", "model_source_sha256", "gradient_method", "replay_validation"],
            "reason": None, "scope": "Same ordered64 objects/labels/crops/eval BN and model source. Each HIER/base comparison uses that run's own shared parameters; cross-run parameter rows are never aligned. Changed training protocol/hyperparameters remain confounders."}


def _backbone_figure(summary, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    records = summary["successful_probes"]
    if not records:
        return None
    names = [str(row.get("display_name") or row["run_key"]) + "\ne" + str(row["epoch"]) for row in records]
    values = [row["parameter_groups"]["shared_encoder"]["hier_vs_base"] for row in records]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4))
    for index, value in enumerate(values):
        ratio, cosine = value.get("norm_ratio"), value.get("cosine")
        if isinstance(ratio, (float, int)):
            axes[0].bar(index, ratio, color="#2563eb")
        else:
            axes[0].text(index, 0., "N/A", ha="center", va="bottom", fontsize=8)
        if isinstance(cosine, (float, int)):
            axes[1].bar(index, cosine, color="#d97706")
        else:
            axes[1].text(index, 0., "N/A", ha="center", va="bottom", fontsize=8)
    for axis in axes:
        axis.set_xticks(range(len(names)))
        axis.set_xticklabels(names, rotation=18, ha="right", fontsize=8)
        axis.grid(axis="y", alpha=.2)
    axes[0].set(title="Actual shared-encoder gradient ratio", ylabel="||lambda * g_HIER|| / ||g_CE+intra||")
    axes[1].set(title="Actual shared-encoder gradient alignment", ylabel="Cosine(g_HIER_weighted, g_base)", ylim=(-1.05, 1.05))
    axes[1].axhline(0, color="#777777", linewidth=.7)
    matched = summary["joint_input_comparison"]["available"]
    label = "Matched ordered64-object clean/eval crop probes" if matched else "Separate probes: joint input equivalence not established"
    fig.suptitle(label + "\nReal model-parameter gradients before clipping; no optimizer step or augmented train-BN replay", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, .89))
    folder = Path(output) / "figures"
    folder.mkdir(parents=True, exist_ok=True)
    png, svg = folder / "real_shared_encoder_backbone.png", folder / "real_shared_encoder_backbone.svg"
    fig.savefig(png, dpi=150, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)
    return {"run_id": "all_runs", "kind": "real_backbone_gradient", "title": "Real shared-encoder HIER/base gradient comparison",
            "png": "figures/" + png.name, "svg": "figures/" + svg.name,
            "note": "Ratios and cosines come from actual shared-encoder parameter gradients at each run's checkpoint. Same64 ordered input/crop/eval conditions gate joint interpretation. No cross-run parameter-row alignment, clipping/optimizer displacement, causal training attribution or augmented train-BN replay."}


def _overview(summary, figure):
    escape = lambda value: html.escape(public_text(value), quote=True)
    rows = []
    for item in summary["probe_availability"]:
        if item["available"]:
            record = next(row for row in summary["successful_probes"]
                          if row["run_key"] == item["run_key"] and row["epoch"] == item["epoch"]
                          and row["source_probe_directory"] == item["probe_directory"])
            value = record["parameter_groups"]["shared_encoder"]["hier_vs_base"]
            cells = [record.get("display_name") or record["run_key"], record["epoch"],
                     value.get("norm_ratio"), value.get("cosine"),
                     record["proxy_objective_shared_encoder_norm"]]
        else:
            cells = [item.get("display_name") or item.get("run_key") or item["probe_directory"],
                     item.get("epoch"), "Unavailable", item.get("reason"), None]
        rows.append("<tr>" + "".join("<td>" + escape("N/A" if cell is None else cell) + "</td>" for cell in cells) + "</tr>")
    joint = summary["joint_input_comparison"]
    text = "Matched input equivalence verified." if joint["available"] else "Joint comparison unavailable: " + str(joint["reason"])
    content = (f'<section id="{MARKER}" class="run-section" data-run="all_runs"><h2>Real shared-encoder gradient probes</h2>'
               '<p>' + escape(text) + ' <a href="backbone.json">Measured backbone JSON</a>; <a href="delivery_manifest.json">Public delivery/source verification</a>.</p>'
               '<div class="table-scroll"><table><thead><tr><th>Run</th><th>Epoch</th><th>Weighted HIER/base gradient norm ratio</th><th>Gradient cosine</th><th>Proxy-only encoder norm</th></tr></thead><tbody>' +
               "".join(rows) + '</tbody></table></div><p>Actual parameter gradients under fixed clean crops and eval BN, before clipping or optimizer updates. Cached-feature partials elsewhere have a different definition. Failed/OOM/incompatible attempts are not successful evidence. Single-seed historical parameter/protocol changes do not isolate HIER causality.</p>')
    if figure:
        content += '<figure><a href="' + escape(figure["png"]) + '"><img src="' + escape(figure["png"]) + '" alt="Real shared encoder gradient comparison"></a><figcaption>' + escape(figure["note"]) + ' <a href="' + escape(figure["svg"]) + '">Vector SVG</a></figcaption></figure>'
    return content + "</section>"


_LINK = re.compile(r'(?P<name>href|src)\s*=\s*(?P<quote>["\x27])(?P<value>.*?)(?P=quote)', re.I)
_FORBIDDEN_FILES = {"normalized_runs.json", "combined_snapshot_spec.json", "snapshot_spec.json",
                    "system_manifest.json", "audit_manifest.json"}
_FORBIDDEN_SUFFIXES = {".npz", ".npy", ".pt", ".pth", ".ckpt", ".log", ".jsonl", ".h5", ".hdf5", ".bin"}
_EXPORT_PUBLIC_FIELDS = (
    "format", "status", "source_run_id", "seed", "population", "sample_count",
    "pool_ids_sha256", "input_sha256", "fixed_train_ids_sha256", "input_mode",
    "inference_rng_seed", "inference_batch_size", "inference_condition",
    "source_commit", "source_training_config_sha256", "source_split_sha256",
    "source_checkpoint_file", "source_checkpoint_sha256", "source_manifest_sha256",
    "completed_checkpoint_identities", "resolved_checkpoint_aliases",
    "manifest_epoch_aliases_agree", "optimizer_updates", "validation_forwarded",
    "test_forwarded", "maximum_checkpoints", "max_seconds", "started_utc", "finished_utc",
    "completed_snapshots", "wall_seconds", "source_files_unchanged", "model_BN_unchanged",
    "gradients_absent",
)


def _relative(value):
    from urllib.parse import unquote, urlsplit
    value = html.unescape(value).strip()
    parsed = urlsplit(value)
    if parsed.scheme or parsed.netloc:
        return None
    if not parsed.path:
        return ""
    path = Path(unquote(parsed.path).replace("\\", "/"))
    if path.is_absolute() or any(part in ("..", "") for part in path.parts):
        return None
    return path.as_posix()


def _referenced_extras(system, document):
    """Copy safe referenced snapshot summary or derive a compact export summary."""
    extras, derived = [], {}
    for match in _LINK.finditer(document):
        relative = _relative(match["value"])
        if not relative:
            continue
        source = system / relative
        if not source.is_file() or source.is_symlink() or system not in source.resolve().parents:
            continue
        if source.name == "snapshot_summary.json" and source.suffix == ".json":
            extras.append(source)
        elif source.name == "export_manifest.json":
            raw = _read(source)
            value = {key: raw.get(key) for key in _EXPORT_PUBLIC_FIELDS if key in raw}
            value["public_summary"] = True
            value["raw_config_paths_and_private_arguments_omitted"] = True
            value["raw_manifest_sha256"] = _sha(source)
            derived[source] = value
    return list(dict.fromkeys(extras)), derived


def _repair_links(document, output):
    """Keep every relative HTML link functional; private artifacts stay unavailable."""
    from urllib.parse import urlsplit
    excluded, changed = [], 0
    placeholder = "figures/server_local_artifact_placeholder.svg"
    def repair(match):
        nonlocal changed
        kind, value = match["name"].lower(), html.unescape(match["value"])
        if not value or value.startswith("#") or value.startswith("data:"):
            return match.group(0)
        parsed = urlsplit(value)
        if parsed.scheme in ("http", "https") and kind == "href":
            return match.group(0)  # Optional documentation link; no network asset.
        relative = _relative(value)
        if relative == "system_manifest.json":
            changed += 1
            return kind + '="delivery_manifest.json"'
        target = output / relative if relative else None
        forbidden = relative and (Path(relative).name in _FORBIDDEN_FILES or Path(relative).suffix.lower() in _FORBIDDEN_SUFFIXES)
        if target is not None and target.is_file() and not forbidden:
            return match.group(0)
        changed += 1
        excluded.append(public_text(value))
        if kind == "href":
            return 'href="#public-delivery-exclusions"'
        (output / "figures").mkdir(exist_ok=True)
        (output / placeholder).write_text(
            '<svg xmlns="http://www.w3.org/2000/svg" width="600" height="100">'
            '<rect width="600" height="100" fill="#f8fafc"/>'
            '<text x="24" y="55" font-family="sans-serif" font-size="18">'
            'Artifact unavailable in public delivery; retained server-local</text></svg>', encoding="utf-8")
        return 'src="' + placeholder + '"'
    document = _LINK.sub(repair, document)
    if excluded and 'id="public-delivery-exclusions"' not in document:
        section = ('<section id="public-delivery-exclusions" class="run-section" data-run="all_runs">'
                   '<h2>Server-local or unavailable artifacts</h2>'
                   '<p>Weights, caches, raw logs, private working specifications and unapproved files are not included. '
                   'The public report contains scientific summaries and figures; these links were replaced, without changing source artifacts.</p>'
                   '<ul>' + "".join("<li>" + html.escape(value) + "</li>" for value in sorted(set(excluded))) + "</ul></section>")
        document = document.replace("</main>", section + "</main>", 1)
    return document, {"rewritten_links": changed, "excluded_links": sorted(set(excluded))}


def validate_offline_links(document, output):
    """Check every local href/src resolves inside the new delivery tree."""
    missing, checked = [], 0
    for match in _LINK.finditer(document):
        value = html.unescape(match["value"])
        if not value or value.startswith(("#", "data:", "http:", "https:")):
            continue
        relative = _relative(value)
        if relative is None or not (output / relative).is_file():
            missing.append(public_text(value))
        else:
            checked += 1
    return {"relative_file_links_checked": checked, "missing": sorted(set(missing)),
            "available": not missing}


def build_delivery(system_dir, backbone_dirs, output_dir, make_plots=True):
    """Copy only reviewed public artifacts into a NEW, external delivery folder."""
    system, output = Path(system_dir).resolve(), Path(output_dir).resolve()
    sources = [system] + [Path(path).resolve() for path in backbone_dirs]
    if output.exists():
        raise FileExistsError("Public delivery must be a new directory")
    if any(output == source or source in output.parents or output in source.parents for source in sources):
        raise ValueError("Delivery/source trees must not contain each other")
    manifest_path, report_path = system / "system_manifest.json", system / "report.html"
    if not manifest_path.is_file() or not report_path.is_file():
        raise ValueError("Completed CPU system manifest and offline report are required")
    system_manifest = _read(manifest_path)
    if system_manifest.get("status") not in ("completed", "partial", "partial_budget") or system_manifest.get("source_files_unchanged") is not True:
        raise ValueError("CPU system artifact lacks completed/partial immutable-source verification")
    selected = [report_path]
    selected.extend(system / name for name in sorted(ROOT_PUBLIC) if (system / name).is_file())
    for subtree in ("figures", "tables", *MODULE_PUBLIC):
        folder = system / subtree
        if folder.is_dir():
            extensions = (".png", ".svg") if subtree == "figures" else (".csv", ".json") if subtree == "tables" else (".json",)
            for path in sorted(folder.rglob("*")):
                if path.is_file() and path.suffix.lower() in extensions and not path.is_symlink():
                    selected.append(path)
    extras, derived = _referenced_extras(system, report_path.read_text(encoding="utf-8-sig"))
    selected = list(dict.fromkeys([*selected, *extras]))
    if any(path.is_symlink() or system not in path.resolve().parents for path in selected):
        raise ValueError("Public source links must stay within the immutable CPU artifact tree")
    display_names = {}
    for filename, field in (("report_data.json", "run_id"), ("comparison_summary.json", "run_key")):
        candidate = system / filename
        if candidate.is_file():
            for record in _read(candidate).get("runs", []):
                if record.get(field) and record.get("display_name"):
                    display_names[record[field]] = record["display_name"]
    statuses, records, observed_sources = [], [], [manifest_path, *selected, *derived]
    for directory in sources[1:]:
        status, record, paths = _probe(directory)
        statuses.append(status)
        observed_sources.extend(paths)
        if status["available"]:
            canonical = display_names.get(record.get("run_key"))
            if canonical:
                record["display_name"] = canonical
                status["display_name"] = canonical
            records.append(record)
    before = {path: _sha(path) for path in observed_sources}
    output.mkdir(parents=True)
    public_manifest = {"format": "hier-public-delivery-v1", "status": "running",
                       "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                       "source_system": {"directory_name": system.name,
                           **{key: system_manifest.get(key) for key in ("status", "audit_code_commit", "budgets",
                                                                       "optimizer_updates", "GPU_forwards", "new_test_forwards",
                                                                       "source_files_unchanged", "completed_utc")}},
                       "probe_availability": statuses, "sources_unchanged": None,
                       "copy_policy": "Allowlisted public report/JSON/CSV/PNG/SVG only; noNPZ/weights/logs/specs/normalized runs/raw manifests",
                       "make_plots": bool(make_plots), "files": []}
    try:
        public_manifest["delivery_code_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[2], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        public_manifest["delivery_code_commit"] = None
    _write(output / "delivery_manifest.json", public_manifest)
    try:
        for path in selected:
            relative = path.relative_to(system)
            target = output / relative
            _copy_public(path, target)
            public_manifest["files"].append({"file": relative.as_posix(), "source_sha256": before[path], "public_sha256": _sha(target)})
        for path, value in derived.items():
            relative = path.relative_to(system)
            target = output / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            _write(target, value)
            public_manifest["files"].append({"file": relative.as_posix(), "source_sha256": before[path],
                                              "public_sha256": _sha(target), "transform": "scalar/hash public manifest summary"})
        backbone = {"format": "hier-public-real-backbone-v1", "successful_probes": records,
                    "probe_availability": statuses, "joint_input_comparison": joint_input_identity(records),
                    "scope": "Actual shared-encoder gradients at saved checkpoints; no source updates, test forward or cross-run parameter-row alignment"}
        _write(output / "backbone.json", backbone)
        figure = _backbone_figure(backbone, output) if make_plots else None
        document = (output / "report.html").read_text(encoding="utf-8")
        if f'id="{MARKER}"' not in document:
            section = _overview(backbone, figure)
            if "</h1>" not in document:
                raise ValueError("Public source report has no insertion anchor")
            document = document.replace("</h1>", "</h1>" + section, 1)
            (output / "report.html").write_text(document, encoding="utf-8")
            public_manifest["backbone_overview_inserted"] = True
        else:
            public_manifest["backbone_overview_inserted"] = False
        document, repaired = _repair_links(document, output)
        (output / "report.html").write_text(document, encoding="utf-8")
        public_manifest["link_repair"] = repaired
        public_manifest["offline_link_integrity"] = validate_offline_links(document, output)
        if not public_manifest["offline_link_integrity"]["available"]:
            raise RuntimeError("Public offline report contains unresolved relative links")
        for item in public_manifest["files"]:
            item["public_sha256"] = _sha(output / item["file"])
        public_manifest["sources_unchanged"] = all(_sha(path) == digest for path, digest in before.items())
        if not public_manifest["sources_unchanged"]:
            raise RuntimeError("A primary CPU/probe artifact changed while building delivery")
        public_manifest.update(status="completed" if all(item["available"] for item in statuses) and system_manifest["status"] == "completed" else "partial",
                               completed_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                               backbone_success_count=len(records), joint_input_comparison=backbone["joint_input_comparison"],
                               backbone_figure=figure,
                               generated_files=[{"file": "backbone.json", "sha256": _sha(output / "backbone.json")}] +
                               ([] if figure is None else [{"file": figure[key], "sha256": _sha(output / figure[key])} for key in ("png", "svg")]))
        _write(output / "delivery_manifest.json", public_manifest)
        return sanitize({**public_manifest, "backbone": backbone})
    except Exception as exc:
        public_manifest.update(status="failed", error=type(exc).__name__ + ": " + public_text(str(exc)),
                               sources_unchanged=all(_sha(path) == digest for path, digest in before.items()))
        _write(output / "delivery_manifest.json", public_manifest)
        raise
