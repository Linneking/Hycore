"""Version-tolerant adapters for V5/V6/V7 HIER and HyCoRe baselines.

Only numeric summaries are retained. Missing observations remain None. Training
sampled positions, clean evaluation, monitored warmup losses, and optimization
updates remain separate. This module has no Torch/NumPy dependency.
"""
from __future__ import annotations

from collections import defaultdict
import json
import math
from pathlib import Path, PureWindowsPath
import re

from .inventory import inventory_run, iter_epoch_rows, read_json, resolve_run, audit_run_key


def _get(value, path):
    for part in path.split("/"):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def _number(value):
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
        return float(value)
    return None


def _first(row, paths):
    for path in paths:
        value = _number(_get(row, path))
        if value is not None:
            return value, path
    return None, None


def _basename(value):
    text = str(value)
    return PureWindowsPath(text).name if "\\" in text else Path(text).name


def public_config(value):
    """Strip private locations/launch metadata while retaining scientific config."""
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            lower = str(key).lower()
            if any(token in lower for token in ("host", "username", "user_name", "uuid", "password", "credential", "private_key", "command", "argv", "data_dir", "run_dir", "physical_gpu", "visible_gpu")):
                continue
            result[str(key)] = public_config(item)
        return result
    if isinstance(value, (list, tuple)):
        return [public_config(item) for item in value]
    if isinstance(value, str) and (re.match(r"^[A-Za-z]:[\\/]", value) or value.startswith("/")):
        return _basename(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


_SCALARS = {
    "loss": "loss", "base_loss": "base", "ce_loss": "ce",
    "hier_loss": "hier_monitored_loss", "sample_hier_loss": "hier_sample_loss_evaluated",
    "proxy_hier_loss": "hier_proxy_loss_evaluated", "intra_contrastive_loss": "intra_contrastive",
    "intra_radial_loss": "intra_radial", "model_preclip_norm": "model_preclip_norm",
    "model_postclip_norm": "model_postclip_norm", "model_clip_fraction": "model_clip_applied",
    "model_clip_scale": "model_clip_scale", "proxy_gradient_norm": "proxy_gradient_norm",
    "proxy_replica_max_difference": "proxy_replica_max_difference",
    "proxy_update_norm": "proxy_update_norm", "proxy_radial_update_norm": "proxy_radial_update_norm",
    "proxy_angular_update_norm": "proxy_angular_update_norm",
    "proxy_signed_radial_update_mean": "proxy_signed_radial_update_mean",
    "proxy_outward_update_fraction": "proxy_outward_update_fraction",
    "proxy_inward_update_fraction": "proxy_inward_update_fraction",
    "proxy_tangent_norm_change_mean": "proxy_tangent_norm_change_mean",
    "proxy_cap_hit_fraction": "proxy_cap_hit_fraction",
    "proxy_cap_hit_count_mean": "proxy_cap_hit_count",
    "proxy_parameter_max_tangent_norm_mean": "proxy_cap_post_max_tangent_norm",
    "proxy_parameter_median_tangent_norm_mean": "proxy_cap_post_median_tangent_norm",
    "proxy_parameter_max_depth_mean": "proxy_cap_post_max_depth",
    "step_seconds": "step_seconds_max_rank", "peak_allocated_mib": "peak_allocated_MiB_max_rank",
}
_DISTRIBUTION_STATS = ("mean", "std", "min", "p10", "median", "p90", "max")
_REQUIRED_METRICS = (
    "train_aug_oa_pct", "train_aug_aa_pct", "val_oa_pct", "val_aa_pct", "val_ce_loss",
    "clean_train_oa_pct", "clean_train_aa_pct", "loss", "base_loss", "ce_loss", "hier_loss",
    "sample_hier_loss", "proxy_hier_loss", "whole_depth_mean", "whole_depth_median",
    "whole_radius_mean", "whole_radius_median", "whole_q_mean", "whole_q_median",
    "proxy_depth_mean", "proxy_depth_median", "proxy_radius_median_mean",
    "sample_used_proxy_count", "sample_effective_proxy_count", "proxy_used_proxy_count",
    "sample_collision_fraction", "proxy_collision_fraction", "sample_anchor_coverage",
    "sample_candidate_triplet_coverage", "proxy_numerical_saturation_fraction",
    "proxy_cap_hit_fraction", "proxy_parameter_max_tangent_norm", "model_clip_fraction",
    "model_lr", "proxy_lr")


def _set(metrics, sources, key, value, source):
    number = _number(value)
    metrics[key] = number
    if number is not None:
        sources[key] = source


def _accuracy(row, metrics, sources):
    fraction_fields = {
        "train_aug_oa_pct": ("telemetry/training/oa",),
        "train_aug_aa_pct": ("telemetry/training/aa_observed_classes",),
    }
    percent_fields = {
        "val_oa_pct": ("validation/val_oa", "validation/oa"),
        "val_aa_pct": ("validation/val_aa", "validation/aa"),
        "clean_train_oa_pct": ("clean_train/train_eval_oa", "clean_train/oa"),
        "clean_train_aa_pct": ("clean_train/train_eval_aa", "clean_train/aa"),
        "test_oa_pct": ("test/test_oa", "official_test/acc", "test/acc"),
        "test_aa_pct": ("test/test_aa", "official_test/acc_avg", "test/acc_avg"),
    }
    for key, paths in fraction_fields.items():
        value, source = _first(row, paths)
        if value is None:
            legacy = "training/acc" if key.endswith("oa_pct") else "training/acc_avg"
            value, source = _first(row, (legacy,))
            _set(metrics, sources, key, value, source)
        else:
            _set(metrics, sources, key, 100 * value, source + " [fraction ×100]")
    for key, paths in percent_fields.items():
        value, source = _first(row, paths)
        _set(metrics, sources, key, value, source)
    for key, paths in {
        "val_ce_loss": ("validation/val_ce", "validation/ce"),
        "clean_train_ce_loss": ("clean_train/train_eval_ce", "clean_train/ce"),
        "val_count": ("validation/val_count", "validation/examples"), "clean_train_count": ("clean_train/train_eval_count", "clean_train/examples"),
        "test_count": ("test/test_count", "official_test/count"),
    }.items():
        value, source = _first(row, paths)
        _set(metrics, sources, key, value, source)


def _geometry(row, metrics, sources, curvature):
    root = "telemetry/geometry"
    for part in ("whole", "part"):
        for quantity in ("radius", "depth", "inverse_metric_factor"):
            for statistic in _DISTRIBUTION_STATS:
                key = "%s_%s_%s" % (part, quantity, statistic)
                path = "%s/%s/%s/%s" % (root, part, quantity, statistic)
                _set(metrics, sources, key, _get(row, path), path)
        path = "%s/%s/near_native_numerical_boundary_fraction" % (root, part)
        _set(metrics, sources, part + "_near_boundary_fraction", _get(row, path), path)
        if curvature is not None and curvature > 0:
            for statistic in _DISTRIBUTION_STATS:
                radius = metrics.get("%s_radius_%s" % (part, statistic))
                if radius is not None:
                    _set(metrics, sources, "%s_q_%s" % (part, statistic), math.sqrt(curvature) * radius,
                         "%s/%s/radius/%s [q=sqrt(c)*r]" % (root, part, statistic))
                # This is the inverse-map norm, not the hidden preprojection
                # encoder vector norm. The conversion is linear in d0.
                measured_depth = metrics.get("%s_depth_%s" % (part, statistic))
                if measured_depth is not None:
                    _set(metrics, sources, "%s_logmap_norm_%s" % (part, statistic), measured_depth / 2,
                         "%s/%s/depth/%s [logmap norm=d0/2]" % (root, part, statistic))
    batch_geometry = _get(row, root + "/hier_reported_per_batch") or {}
    names = {"proxy_radius_min_mean": "proxy.ball_radius_min",
             "proxy_radius_median_mean": "proxy.ball_radius_median",
             "proxy_radius_max_mean": "proxy.ball_radius_max",
             "proxy_logmap_norm_max_mean": "proxy.tangent_radius_max",
             "proxy_numerical_saturation_fraction": "proxy_numerical_ball_project_fraction",
             "proxy_numerical_saturation_count_mean": "proxy_numerical_ball_project_count",
             "proxy_shadow_tangent_cap_fraction": "proxy_tangent_shadow_clip_fraction"}
    for name, source_name in names.items():
        value = batch_geometry.get(source_name)
        _set(metrics, sources, name, value.get("mean") if isinstance(value, dict) else value,
             root + "/hier_reported_per_batch/" + source_name + "/mean")
    # A maximum over batch maxima is the true maximum of observed outputs.
    # Means/medians of nonlinear transforms must NOT be reconstructed this way.
    radius_summary = batch_geometry.get("proxy.ball_radius_max") or {}
    observed_max = _number(radius_summary.get("max")) if isinstance(radius_summary, dict) else None
    if curvature and observed_max is not None and 0 <= math.sqrt(curvature) * observed_max < 1:
        _set(metrics, sources, "proxy_depth_max", 2 / math.sqrt(curvature) * math.atanh(math.sqrt(curvature) * observed_max),
             root + "/hier_reported_per_batch/proxy.ball_radius_max/max [exact monotonic d0 transform]")
    if curvature and metrics.get("proxy_radius_median_mean") is not None:
        _set(metrics, sources, "proxy_q_median_mean", math.sqrt(curvature) * metrics["proxy_radius_median_mean"],
             root + "/hier_reported_per_batch/proxy.ball_radius_median/mean [linear q transform]")


def _usage_record(epoch, updates, component, domain, role, counts, source):
    if not isinstance(counts, list) or not counts or any(_number(value) is None or value < 0 for value in counts):
        return None
    total = sum(counts)
    entropy = -sum((value / total) * math.log(value / total) for value in counts if value) if total else None
    return {"epoch": epoch, "model_updates": updates, "component": component,
            "domain": domain, "role": role, "proxy_ids": list(range(len(counts))),
            "counts": counts, "definition": "actual_training_hard_gumbel",
            "source": source, "selection_count": total,
            "used_proxy_count": sum(value > 0 for value in counts),
            "used_proxy_fraction": sum(value > 0 for value in counts) / len(counts),
            "entropy": entropy, "effective_proxy_count": math.exp(entropy) if entropy is not None else None}


def _structure(row, epoch, updates, metrics, sources):
    usage_rows = []
    for component in ("sample", "proxy"):
        report = _get(row, "structure/" + component) or {}
        depth_distributions = report.get("depth_distributions") or {}
        for name, distribution in depth_distributions.items():
            if not isinstance(distribution, dict):
                continue
            safe = name.replace("/", "_")
            for statistic in ("mean", "std", "min", "max", "positive_fraction", "negative_fraction"):
                key = component + "_" + safe + "_" + statistic
                path = "structure/%s/depth_distributions/%s/%s" % (component, name, statistic)
                _set(metrics, sources, key, distribution.get(statistic), path)
        reports = report.get("selected_ancestor_usage") or {}
        by_domain = defaultdict(dict)
        for domain_role, value in reports.items():
            if not isinstance(value, dict) or "/" not in domain_role:
                continue
            domain, role = domain_role.split("/", 1)
            item = _usage_record(epoch, updates, component, domain, role, value.get("counts"),
                                 "structure/%s/selected_ancestor_usage/%s/counts" % (component, domain_role))
            if item is not None:
                usage_rows.append(item)
                by_domain[domain][role] = item
        for domain, roles in by_domain.items():
            if "pair" in roles and "triple" in roles and len(roles["pair"]["counts"]) == len(roles["triple"]["counts"]):
                combined = [left + right for left, right in zip(roles["pair"]["counts"], roles["triple"]["counts"])]
                item = _usage_record(epoch, updates, component, domain, "combined", combined,
                                     "structure/%s/selected_ancestor_usage/%s [pair+triple counts]" % (component, domain))
                usage_rows.append(item)
                for measure in ("used_proxy_count", "used_proxy_fraction", "effective_proxy_count", "entropy"):
                    _set(metrics, sources, component + "_" + domain + "_" + measure, item[measure], item["source"])
                    if domain == "all_draws":
                        _set(metrics, sources, component + "_" + measure, item[measure], item["source"])
        graph_root = "telemetry/hierarchy/" + component + "_graph"
        for target, field in {"collision_fraction": "collision_rate", "active_fraction": "active_fraction_all_draws",
                              "active_noncollision_fraction": "active_fraction_noncollision",
                              "anchor_coverage": "anchor_coverage", "candidate_triplet_coverage": "candidate_triplet_coverage",
                              "self_k_fraction": "self_k_fraction", "repeat_fraction": "repeat_fraction",
                              "cross_class_j_fraction": "cross_class_j_fraction", "any_role_position_coverage": "any_role_position_coverage"}.items():
            path = graph_root + "/rates/" + field
            _set(metrics, sources, component + "_" + target, _get(row, path), path)
        for count in ("triplets", "collisions", "noncollision_triplets", "active_triplets", "self_k_triplets", "equal_data_id_ik_triplets"):
            path = graph_root + "/counts/" + count
            _set(metrics, sources, component + "_" + count, _get(row, path), path)
        if report.get("draw_count") is not None:
            for target, field in (("structure_draw_count", "draw_count"), ("structure_noncollision_count", "noncollision_count"),
                                  ("structure_active_count", "active_noncollision_count")):
                _set(metrics, sources, component + "_" + target, report.get(field), "structure/%s/%s" % (component, field))
    return usage_rows


def _version(run_id, manifest):
    text = " ".join(str(value) for value in (run_id, manifest.get("protocol", ""), manifest.get("canonical_experiment_name", "")))
    matched = re.search(r"(?:^|[^a-z])v([5-7])(?:[^0-9]|$)", text, flags=re.I)
    return "V" + matched.group(1) if matched else None


def _display_name(run_dir, manifest, config, explicit_name=None):
    """Resolve registry aliases only within their confirmed storage namespace.

    Bare H20/B0/B32 strings are intentionally insufficient: the same alias has
    different meanings across versions and method lines.
    """
    if explicit_name:
        return explicit_name, "explicit --name"
    if manifest.get("canonical_experiment_name"):
        return manifest["canonical_experiment_name"], "manifest.canonical_experiment_name"
    root, arm = run_dir.parent.name, run_dir.name
    scopes = (
        ("hier_proxy_v5_", "H20", "V5", "shared_whole_proxy_v5", "v5_hier64_main"),
        ("hier_proxy_v5_", "B32", "V5", "shared_whole_proxy_v5", "v5_shuffle_stability_main"),
        ("hier_proxy_v6_", "H20", "V6", "shared_whole_proxy_v6", "v6_hier64_main"),
        ("hier_proxy_v6_", "B0_original_B32", "V6", "source_hycore_wrapper", "v6_original_source_shuffle32_main"),
        ("hycore_b64_v6_", "B64_shuffle", "V6", "distributed_shuffle_hycore", "v6_shuffle64_main"),
        ("v6_balanced_b0_", "B0_balanced", "V6", "shared_whole_proxy_v6_matched_off", "v6_matched_balanced64_main"),
    )
    selected = [rule for rule in scopes if root.startswith(rule[0]) and arm == rule[1]]
    if len(selected) == 1:
        _prefix, storage_id, version, method_line, scope = selected[0]
        registry_path = Path(__file__).resolve().parents[2] / "docs" / "research" / "experiment_registry.json"
        registry = read_json(registry_path)
        matches = []
        for entry in registry.get("runs", []):
            if entry.get("version") != version or entry.get("method_line") != method_line:
                continue
            aliases = entry.get("aliases") or []
            scoped_alias = any(alias.get("scope") == scope and storage_id in alias.get("values", []) for alias in aliases)
            if entry.get("source_run_id") == storage_id and scoped_alias:
                expected_batch = _number((entry.get("batch") or {}).get("global"))
                observed_batch = _number(config.get("global_batch", config.get("batch_size")))
                if observed_batch is not None and expected_batch is not None and observed_batch != expected_batch:
                    continue
                matches.append(entry)
        if len(matches) == 1:
            return matches[0]["display_name"], "experiment_registry.json:%s" % matches[0]["canonical_id"]
    # V7 has a structured storage name carrying its full public protocol. Verify
    # its version/batch/K/warmup against the manifest before adopting it.
    typed = re.fullmatch(r"(V[5-7])_HIER(32|64)_K(\d+)_W(\d+)", arm)
    if typed:
        version, batch, sample_k, warmup = typed.groups()
        observed = (_version(root + "/" + arm, manifest),
                    _number(config.get("global_batch")),
                    _number(config.get("sample_K", config.get("sample_k"))),
                    _number(config.get("warmup_epochs")))
        expected = (version, float(batch), float(sample_k), float(warmup))
        if observed == expected and config.get("proxy_optimizer") is not None:
            return arm.replace("_", "-"), "structured storage name verified against training_config"
    return arm, "unresolved storage identifier; no global alias guessing"


def normalize_run(path: str | Path, name: str | None = None, inventory: dict | None = None) -> dict:
    """Normalize a run into small public scalar/usage summaries for reports."""
    run_dir = resolve_run(path)
    if inventory is None:
        inventory = inventory_run(run_dir)
    warnings = list(inventory["warnings"])
    manifest = read_json(run_dir / "manifest.json", warnings)
    config = {**(manifest.get("fixed") or {}), **(manifest.get("training_config") or manifest.get("config") or {})}
    safe_config = public_config(config)
    curvature = _number(config.get("c", config.get("curvature")))
    n_proxy = _number(config.get("P", config.get("num_proxies", config.get("n_proxies", config.get("proxy_count")))))
    budget = _number(config.get("steps_per_epoch", config.get("executed_train_steps")))
    warmup = _number(config.get("warmup_epochs"))
    prefix = _number(manifest.get("inherited_prefix_epochs", config.get("shared_historical_prefix_epochs")))
    metadata = {"curvature": curvature, "n_proxy": int(n_proxy) if n_proxy is not None else None,
                "steps_per_epoch": int(budget) if budget is not None else None,
                "warmup_epochs": int(warmup) if warmup is not None else None,
                "inherited_prefix_epochs": int(prefix) if prefix is not None else None,
                "train_size": manifest.get("train_size", config.get("train_examples")),
                "validation_size": manifest.get("validation_size"),
                "geometry_domain": "actual training-forward sampled positions; includes augmentation, sampling repeats and BN effects",
                "proxy_geometry_domain": "per-training-batch scalar observations; medians averaged over batches are labeled explicitly",
                "sample_selection_stability_available": False,
                "model_updates_coordinate": "original epoch × configured steps_per_epoch; nominal axis, not evidence for unlogged prefix updates"}
    identity = {"version": _version(run_dir.parent.name + "/" + run_dir.name, manifest), "seed": config.get("seed"),
                "commit": manifest.get("commit"), "split_sha256": manifest.get("split_sha256"),
                "status": manifest.get("status"), "config": safe_config,
                "started_utc": manifest.get("started_utc"), "finished_utc": manifest.get("finished_utc"),
                "initialized_model_sha256": manifest.get("initialized_model_sha256", manifest.get("initial_model_sha256")),
                "initialization": public_config(manifest.get("initialization") or {}),
                "resume_identity": public_config(manifest.get("resume_identity") or manifest.get("source_checkpoint_identity") or {}),
                "best": public_config(manifest.get("best")), "final_test": public_config(manifest.get("final_test"))}
    display_name, name_source = _display_name(run_dir, manifest, config, name)
    identity["display_name_source"] = name_source
    if not identity.get("final_test"):
        for filename in ("final_test.json", "final_selected_test.json", "official_test.json"):
            saved_test = read_json(run_dir / filename)
            if saved_test:
                identity["saved_final_test"] = public_config(saved_test)
                identity["saved_final_test_source"] = filename
                break
    normalized = {}
    for raw, source in iter_epoch_rows(run_dir, warnings):
        raw = dict(raw)
        legacy_telemetry = not raw.get("telemetry") and isinstance(raw.get("training"), dict) and "batches" in raw["training"]
        legacy_clean = not raw.get("clean_train") and bool(raw.get("train_clean"))
        if legacy_telemetry:
            raw["telemetry"] = raw["training"]
        if legacy_clean:
            raw["clean_train"] = raw["train_clean"]
        epoch = int(raw["epoch"])
        metrics, sources = {key: None for key in _REQUIRED_METRICS}, {}
        _accuracy(raw, metrics, sources)
        for target, field in _SCALARS.items():
            root = "telemetry/scalars/" + field
            value, found = _first(raw, (root + "/mean", root + "/batch_size_weighted_mean", field))
            _set(metrics, sources, target, value, found)
        for target, field in {"proxy_parameter_max_tangent_norm": "proxy_cap_post_max_tangent_norm",
                              "proxy_parameter_max_depth": "proxy_cap_post_max_depth",
                              "model_preclip_norm_max": "model_preclip_norm",
                              "proxy_gradient_norm_max": "proxy_gradient_norm"}.items():
            source_path = "telemetry/scalars/" + field + "/max"
            _set(metrics, sources, target, _get(raw, source_path), source_path)
        if metrics.get("loss") is None:
            value, found = _first(raw, ("training/loss",))
            _set(metrics, sources, "loss", value, found)
        _set(metrics, sources, "lambda_hier", raw.get("lambda_hier"), "lambda_hier")
        if metrics.get("lambda_hier") is None:
            value, found = _first(raw, ("telemetry/scalars/lambda_hier/mean",))
            _set(metrics, sources, "lambda_hier", value, found)
        if metrics.get("hier_loss") is not None and metrics.get("lambda_hier") is not None:
            _set(metrics, sources, "weighted_hier_loss", metrics["hier_loss"] * metrics["lambda_hier"],
                 "lambda_hier × telemetry/scalars/hier_monitored_loss/mean")
        for optimizer in ("model", "proxy"):
            value, found = _first(raw, ("lr_used/" + optimizer,))
            if optimizer == "model" and value is None:
                value, found = _first(raw, ("lr_used",))
            _set(metrics, sources, optimizer + "_lr", value, found)
        if metrics.get("model_clip_fraction") is None:
            root = "telemetry/scalars/model_clip_triggered/mean"
            _set(metrics, sources, "model_clip_fraction", _get(raw, root), root)
        _geometry(raw, metrics, sources, curvature)
        for target, field in {"training_unique_ids": "unique_ids", "training_split_fraction": "fraction",
                              "training_draws": "draws", "training_draw_repetition_fraction": "repeat_fraction"}.items():
            root = "telemetry/sample_coverage/" + field
            _set(metrics, sources, target, _get(raw, root), root)
        _set(metrics, sources, "sample_eligible_unique_ids", _get(raw, "telemetry/hierarchy/eligible_anchors/unique_ids"),
             "telemetry/hierarchy/eligible_anchors/unique_ids")
        _set(metrics, sources, "sample_eligible_training_fraction", _get(raw, "telemetry/hierarchy/eligible_anchors/fraction"),
             "telemetry/hierarchy/eligible_anchors/fraction")
        for key, paths in {"model_preclip_norm": ("gradient/preclip_norm_mean",),
                           "model_clip_fraction": ("gradient/clip_trigger_fraction",)}.items():
            if metrics.get(key) is None:
                value, found = _first(raw, paths)
                _set(metrics, sources, key, value, found)
        model_steps, steps_source = _first(raw, ("actual_optimizer_steps", "telemetry/batches", "train_steps"))
        model_updates = epoch * budget if budget is not None else None
        if budget is None and model_steps is not None:
            model_updates = None  # One epoch's count cannot establish prior budgets.
        proxy_steps, proxy_steps_source = _first(raw, ("telemetry/hierarchy/objective_active_batches",))
        phase = raw.get("phase") or raw.get("protocol")
        proxy_disabled = config.get("proxy_optimizer") in (False, None) and (
            manifest.get("objective_delta") is not None or "base" in str(phase).lower()
            or "b0" in str(phase).lower() or "original_hycore" in str(phase).lower()
            or "hycore_b32" in str(phase).lower())
        if proxy_steps is None and proxy_disabled and model_steps is not None:
            proxy_steps, proxy_steps_source = 0., "no-proxy optimizer protocol; recorded model step count"
        usage = _structure(raw, epoch, model_updates, metrics, sources)
        if metadata["n_proxy"] is None and usage:
            metadata["n_proxy"] = len(usage[0]["counts"])
        actual_projection = _get(raw, "telemetry/scalars/proxy_cap_hit_count/sum")
        _set(metrics, sources, "proxy_cap_hit_events", actual_projection, "telemetry/scalars/proxy_cap_hit_count/sum")
        if legacy_telemetry:
            sources = {key: value.replace("telemetry/", "training/") if isinstance(value, str) else value for key, value in sources.items()}
            steps_source = steps_source.replace("telemetry/", "training/") if steps_source else None
        if legacy_clean:
            sources = {key: value.replace("clean_train/", "train_clean/") if isinstance(value, str) else value for key, value in sources.items()}
        normalized[epoch] = {"row": {"epoch": epoch, "phase": phase, "model_updates": model_updates,
            "model_updates_kind": "nominal_epoch_budget" if model_updates is not None else "unavailable",
            "model_steps": int(model_steps) if model_steps is not None else None,
            "proxy_steps": int(proxy_steps) if proxy_steps is not None else None,
            "proxy_steps_source": proxy_steps_source, "model_steps_source": steps_source,
            "metrics": metrics, "availability": sources, "source": source,
            "epoch_wall_seconds": raw.get("epoch_wall_seconds", raw.get("epoch_seconds")),
            "telemetry_finite": _get(raw, "telemetry/finite") if _get(raw, "telemetry/finite") is not None else raw.get("finite")},
            "usage": usage}
    epochs, proxy_usage = [], []
    logged_model, logged_proxy = 0, 0
    model_known, proxy_known = True, True
    for epoch in sorted(normalized):
        row = normalized[epoch]["row"]
        if row["model_steps"] is None:
            model_known = False
        else:
            logged_model += row["model_steps"]
        if row["proxy_steps"] is None:
            proxy_known = False
        else:
            logged_proxy += row["proxy_steps"]
        row["logged_model_updates"] = logged_model if model_known else None
        row["logged_proxy_updates"] = logged_proxy if proxy_known else None
        # Proxy update axis is exactly logged history; unlogged inherited states
        # are not fabricated from epoch numbers/warmup configuration.
        row["proxy_updates"] = row["logged_proxy_updates"]
        epochs.append(row)
        proxy_usage.extend(normalized[epoch]["usage"])
    availability = []
    keys = sorted(set(_REQUIRED_METRICS) | {key for row in epochs for key in row["metrics"]})
    for metric in keys:
        available = [row for row in epochs if row["metrics"].get(metric) is not None]
        status = "available" if epochs and len(available) == len(epochs) else "partial" if available else "missing"
        availability.append({"metric": metric, "status": status, "available_epoch_count": len(available),
                             "source": sorted(set(row["availability"].get(metric, "") for row in available)),
                             "reason": None if status == "available" else "Not saved for every observed epoch; no zero substitution."})
    if curvature is None:
        warnings.append("Curvature unavailable in manifest; no normalized q or derived depths supplied.")
    if not proxy_usage:
        warnings.append("Per-proxy hard-selection counts are unavailable; activation heatmap and identity stability cannot be reconstructed from scalar counts.")
    warnings.append("Epoch activation-set overlap is training usage turnover, not retention of each proxy's nearest sample IDs; fixed-ID snapshots are required for that question.")
    if inventory["capabilities"]["training_geometry"]:
        warnings.append("Training-forward geometry is not clean fixed-panel trajectory; changing augmentation/batches/BN can change these observations.")
    return {"schema_version": 1, "run_id": run_dir.name, "storage_run_id": run_dir.name, "audit_run_key": audit_run_key(run_dir),
            "display_name": display_name,
            "identity": identity, "metadata": metadata, "inventory": inventory,
            "epochs": epochs, "proxy_usage": proxy_usage, "availability": availability,
            "warnings": list(dict.fromkeys(warnings))}


def summarize_steps(path: str | Path, max_rows: int | None = None) -> dict:
    """Optional streaming integrity counters. Default normalize_run never calls it."""
    run_dir = resolve_run(path)
    file = run_dir / "steps.jsonl"
    if not file.is_file():
        return {"available": False, "rows": None, "epochs": [], "warnings": ["steps.jsonl absent"]}
    per_epoch = defaultdict(lambda: {"rows": 0, "first_step": None, "last_step": None, "finite_scalar_rows": 0})
    warnings, rows = [], 0
    with file.open("r", encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, 1):
            if max_rows is not None and rows >= max_rows:
                break
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                epoch, step = int(row["epoch"]), int(row["step"])
            except (ValueError, KeyError, TypeError, json.JSONDecodeError):
                warnings.append("Malformed steps.jsonl line %d skipped" % line_number)
                continue
            rows += 1
            report = per_epoch[epoch]
            report["rows"] += 1
            report["first_step"] = step if report["first_step"] is None else min(step, report["first_step"])
            report["last_step"] = step if report["last_step"] is None else max(step, report["last_step"])
            numeric = [value for value in row.values() if isinstance(value, (int, float)) and not isinstance(value, bool)]
            report["finite_scalar_rows"] += int(all(math.isfinite(value) for value in numeric))
    return {"available": True, "rows": rows, "partial_scan": max_rows is not None and rows >= max_rows,
            "epochs": [{"epoch": epoch, **per_epoch[epoch]} for epoch in sorted(per_epoch)], "warnings": warnings}
