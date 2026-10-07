"""Evidence ledger for saved V5/V6/V7 runs, without test-driven reselection.

Correctness checks, training protocols and hierarchy observations remain distinct.
Historical, single-seed, multi-factor differences are descriptive comparisons.
"""
from __future__ import annotations
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import re

from .longitudinal import run_key

LOG_METRICS = (
    "val_oa_pct", "val_aa_pct", "val_ce_loss", "clean_train_oa_pct",
    "whole_depth_mean", "whole_depth_median", "whole_near_boundary_fraction",
    "proxy_numerical_saturation_fraction", "proxy_cap_hit_fraction",
    "proxy_replica_max_difference", "sample_used_proxy_count", "sample_effective_proxy_count",
    "sample_collision_fraction", "sample_active_noncollision_fraction", "sample_self_k_fraction",
    "proxy_collision_fraction", "proxy_active_noncollision_fraction", "proxy_self_k_fraction",
    "sample_hier_loss", "proxy_hier_loss",
)


def _finite(value):
    return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value)


def _mean(values):
    values = [float(v) for v in values if _finite(v)]
    return sum(values) / len(values) if values else None


def _first(record, names):
    if not isinstance(record, dict):
        return None
    for name in names:
        if _finite(record.get(name)):
            return record[name]
    return None


def _saved_selection(analysis, registry):
    identity = analysis.get("identity", {})
    best = identity.get("best") or {}
    test = identity.get("saved_final_test") or identity.get("final_test") or {}
    selection = identity.get("selection") or registry.get("selection") or {}
    if not isinstance(selection, dict):
        selection = {"metric": str(selection)}
    metric = str(selection.get("metric", ""))
    if not metric:
        # A recorded validation-best identity is sufficient, test rows are not.
        if _first(best, ("val_oa", "val_oa_pct", "validation_oa")) is not None:
            metric = "validation_OA (recorded best identity)"
        elif _first(best, ("test_oa", "test_oa_raw", "test_oa_source")) is not None:
            metric = "historical official_test_OA (saved identity; never reselected)"
        else:
            metric = "unknown; no automatic best selection"
    best_epoch = _first(best, ("epoch", "completed_epochs", "best_epoch"))
    if best_epoch is None:
        best_epoch = _first(test, ("selected_epoch", "best_epoch", "checkpoint_epoch", "model_epoch"))
    if best_epoch is None and isinstance(test.get("selection"), dict):
        best_epoch = _first(test["selection"], ("epoch", "best_epoch"))
    is_validation = "val" in metric.lower() and "test" not in metric.lower()
    val_oa = _first(best, ("val_oa", "val_oa_pct", "validation_oa", "oa"))
    val_aa = _first(best, ("val_aa", "val_aa_pct", "validation_aa"))
    if best_epoch is not None and is_validation:
        observed = next((r for r in analysis.get("epochs", []) if r.get("epoch") == int(best_epoch)), None)
        if observed:
            if val_oa is None:
                val_oa = observed.get("metrics", {}).get("val_oa_pct")
            if val_aa is None:
                val_aa = observed.get("metrics", {}).get("val_aa_pct")
    return {"selection_metric": metric, "selection_source": "saved manifest identity / registry; no model reselection",
            "validation_selected": is_validation, "selected_epoch": best_epoch,
            "selected_validation_oa_pct": val_oa if is_validation else None,
            "selected_validation_aa_pct": val_aa if is_validation else None,
            "saved_final_test_oa_pct": _first(test, ("test_oa", "test_oa_pct", "acc", "oa")),
            "saved_final_test_aa_pct": _first(test, ("test_aa", "test_aa_pct", "acc_avg", "aa")),
            "saved_final_test_count": _first(test, ("test_count", "count", "examples")),
            "test_evaluation_policy": selection.get("official_test_access",
                "read saved final result only; audit runs no official-test forward"),
            "final_test_available": bool(test),
            "saved_final_test_source": identity.get("saved_final_test_source", "manifest.final_test" if test else None)}


def _flatten(value, prefix=""):
    result = {}
    if isinstance(value, dict):
        for key, child in value.items():
            name = prefix + "." + str(key) if prefix else str(key)
            if isinstance(child, dict):
                result.update(_flatten(child, name))
            elif isinstance(child, (str, int, float, bool, type(None))):
                result[name] = child
            elif isinstance(child, list) and all(isinstance(x, (str, int, float, bool)) for x in child):
                result[name] = child
    return result


def _protocol(analysis, registry):
    identity, metadata = analysis.get("identity", {}), analysis.get("metadata", {})
    config = identity.get("config", {})
    protocol = {}
    for section in ("batch", "sampler", "data", "budget", "operators", "hier_schedule", "hier_config", "objective", "selection"):
        protocol.update(_flatten(registry.get(section, {}), section))
    scalar_names = ("global_batch", "batch_size", "local_batch", "world_size", "steps_per_epoch",
                    "epochs", "warmup_epochs", "lambda_hier", "c", "sample_K", "sample_k",
                    "proxy_K", "proxy_k", "P", "triples_per_anchor", "self_k_excluded",
                    "extra_HIER_tangent_cap", "proxy_parameter_depth_cap", "proxy_init_depth",
                    "proxy_depth_cap", "proxy_tangent_cap", "proxy_lr", "weight_decay", "seed")
    for name in scalar_names:
        if name in config:
            protocol["recorded_config." + name] = config[name]
    for name in ("curvature", "steps_per_epoch", "warmup_epochs", "inherited_prefix_epochs"):
        if name in metadata:
            protocol["metadata." + name] = metadata[name]
    protocol["identity.seed"] = identity.get("seed")
    protocol["identity.split_sha256"] = identity.get("split_sha256")
    return protocol


def _metric_summary(analysis):
    rows = sorted(analysis.get("epochs", []), key=lambda row: row["epoch"])
    result = {}
    for key in LOG_METRICS:
        points = [(r["epoch"], r.get("metrics", {}).get(key)) for r in rows]
        points = [(e, float(v)) for e, v in points if _finite(v)]
        result[key] = {"available_epoch_count": len(points),
                       "first": {"epoch": points[0][0], "value": points[0][1]} if points else None,
                       "last": {"epoch": points[-1][0], "value": points[-1][1]} if points else None,
                       "last40_recorded_mean": _mean(v for _, v in points[-40:]),
                       "aggregation": "Arithmetic average of saved epoch-level statistics; not a pooled draw-count estimate.",
                       "last40_recorded_window": [points[-40:][0][0], points[-1][0]] if points else None,
                       "maximum_observed": max((v for _, v in points), default=None),
                       "source_fields": sorted({r.get("availability", {}).get(key, "normalized saved metric")
                                                for r in rows if _finite(r.get("metrics", {}).get(key))}),
                       "population": "training-forward sampled positions" if any(x in key for x in ("whole_", "proxy_", "sample_")) and not key.startswith("val") else "saved evaluation / training scalar"}
    return result


def _align_logs(left, right, axis):
    left_rows = {row.get(axis): row for row in left.get("epochs", []) if _finite(row.get(axis))}
    right_rows = {row.get(axis): row for row in right.get("epochs", []) if _finite(row.get(axis))}
    common = sorted(set(left_rows) & set(right_rows))
    metrics, rows = {}, []
    for key in LOG_METRICS:
        pairs = []
        for x in common:
            a, b = left_rows[x].get("metrics", {}).get(key), right_rows[x].get("metrics", {}).get(key)
            if _finite(a) and _finite(b):
                pairs.append({"coordinate": x, "left_epoch": left_rows[x]["epoch"],
                              "right_epoch": right_rows[x]["epoch"], "left": a, "right": b, "delta": b - a})
        metrics[key] = {"paired_observation_count": len(pairs),
                        "left_mean": _mean(p["left"] for p in pairs),
                        "right_mean": _mean(p["right"] for p in pairs),
                        "mean_delta": _mean(p["delta"] for p in pairs),
                        "last": pairs[-1] if pairs else None}
        rows.extend({"metric": key, "axis": axis, **p} for p in pairs)
    return {"axis": axis, "common_coordinates": common, "metrics": metrics, "observations": rows,
            "coordinate_kind": "actual saved epoch" if axis == "epoch" else "nominal epoch × configured steps; incomplete logged prefixes are not reconstructed",
            "interpretation": "Coordinates align observations, not stochastic batches or proxy identities; missing observations are not interpolated."}


def _conditions(left, right, arrays_base=None):
    source_a, source_b = left.get("source", {}), right.get("source", {})
    for field in ("sample_pool_sha256", "input_sha256"):
        if not source_a.get(field) or source_a.get(field) != source_b.get(field):
            return False, field + " is missing or differs", None
    if source_a.get("inputmode") in (None, "unknown") or source_a.get("inputmode") != source_b.get("inputmode"):
        return False, "Input/evaluation mode is missing or differs", None
    if not source_a.get("inference_condition") or source_a.get("inference_condition") != source_b.get("inference_condition"):
        return False, "Explicit inference conditions are missing or differ", None
    for field in ("c", "dimension"):
        if left.get("whole", {}).get(field) is None or left["whole"][field] != right.get("whole", {}).get(field):
            return False, "Curvature or embedding dimension differs/missing", None
    if arrays_base and left.get("arrays_file") and right.get("arrays_file"):
        import numpy as np
        with np.load(Path(arrays_base) / left["arrays_file"], allow_pickle=False) as archive:
            a = {key: archive[key].copy() for key in ("sample_ids", "labels", "whole_depth", "whole_normalized_radius")}
        with np.load(Path(arrays_base) / right["arrays_file"], allow_pickle=False) as archive:
            b = {key: archive[key].copy() for key in ("sample_ids", "labels", "whole_depth", "whole_normalized_radius")}
        if not np.array_equal(a["sample_ids"], b["sample_ids"]) or not np.array_equal(a["labels"], b["labels"]):
            return False, "Actual sorted IDs or labels differ", None
        delta = b["whole_depth"] - a["whole_depth"]
        numeric = {"object_count": len(delta), "mean_depth_delta": float(np.mean(delta)),
                   "median_depth_delta": float(np.median(delta)), "p10_depth_delta": float(np.quantile(delta, .1)),
                   "p90_depth_delta": float(np.quantile(delta, .9)),
                   "outward_depth_fraction": float(np.mean(delta > 0)),
                   "left_fixed_q_ge_0_99599_fraction": float(np.mean(a["whole_normalized_radius"] >= .99599)),
                   "right_fixed_q_ge_0_99599_fraction": float(np.mean(b["whole_normalized_radius"] >= .99599)),
                   "threshold_definition": "Explicit common q >= 0.99599; descriptive threshold, not a universal numerical projection rule"}
        return True, None, numeric
    label_sha = source_a.get("labels_sha256")
    if not label_sha or label_sha != source_b.get("labels_sha256"):
        return False, "Label consistency unverified: supply audited arrays or equal labels SHA256", None
    return True, None, None


def _snapshot_metrics(snapshot):
    whole, proxy = snapshot.get("whole", {}), snapshot.get("proxy", {})
    retrieval = snapshot.get("retrieval", {})
    result = {"whole_depth_mean": whole.get("depth_quantiles", {}).get("mean"),
              "whole_depth_median": whole.get("depth_quantiles", {}).get("p50", whole.get("depth_quantiles", {}).get("median")),
              "proxy_depth_median": proxy.get("depth_quantiles", {}).get("p50", proxy.get("depth_quantiles", {}).get("median")),
              "proxy_parameter_norm_median": proxy.get("parameter_tangent_norm_quantiles", {}).get("p50", proxy.get("parameter_tangent_norm_quantiles", {}).get("median")),
              "proxy_numeric_projection_count": proxy.get("numeric_projection_count")}
    for method in ("hyperbolic", "direction", "equal_radius_hyperbolic"):
        value = retrieval.get(method, {})
        if not value.get("available", retrieval.get("available", False)) or not value:
            continue
        prefix = method + "_"
        for section, fields in {"coverage": ("unique_samples", "unique_fraction"),
                               "slot_concentration": ("gini", "top4_sample_slot_share", "effective_sample_count"),
                               "repeated_tuples": ("largest_identical_topk_fraction", "unique_topk_sets")}.items():
            for field in fields:
                result[prefix + field] = value.get(section, {}).get(field)
        result[prefix + "selected_radius_percentile_mean"] = value.get("radius_bias", {}).get("selected_radius_percentile_mean")
    return {key: value for key, value in result.items() if _finite(value)}


def _snapshot_pairs(left_key, right_key, snapshots):
    grouped = {}
    for snapshot in (snapshots or {}).get("snapshots", []):
        grouped.setdefault(snapshot.get("run_key"), {})[snapshot.get("epoch")] = snapshot
    left, right = grouped.get(left_key, {}), grouped.get(right_key, {})
    common = sorted(set(left) & set(right))
    result = []
    for epoch in common:
        a, b = left[epoch], right[epoch]
        try:
            valid, reason, paired_objects = _conditions(a, b, (snapshots or {}).get("arrays_base_dir"))
        except (ValueError, KeyError, FileNotFoundError) as exc:
            valid, reason, paired_objects = False, "Audited snapshot arrays could not verify label/object identities", None
        record = {"epoch": epoch, "paired_pool_available": valid, "reason": reason,
                  "proxy_identity_paired": False,
                  "interpretation": "Aggregate proxy distributions can be compared in a verified object pool. Proxy parameter row IDs are never paired across runs."}
        if valid:
            aa, bb = _snapshot_metrics(a), _snapshot_metrics(b)
            record["metrics"] = {key: {"left": aa[key], "right": bb[key], "delta": bb[key] - aa[key]}
                                 for key in sorted(set(aa) & set(bb))}
            record["paired_object_radial_changes"] = paired_objects
            record["input_sha256"] = a["source"]["input_sha256"]
            record["sample_pool_sha256"] = a["source"]["sample_pool_sha256"]
        result.append(record)
    return result


def _changes(a, b):
    rows = []
    for field in sorted(set(a) | set(b)):
        if field not in a or field not in b:
            continue
        if a[field] == b[field]:
            continue
        category = "training_protocol"
        if "self_k_excluded" in field or any(token in field for token in ("synchron", "replica")):
            category = "correctness"
        elif any(token in field for token in ("objective.hier", "hier_config.sample_k", "hier_config.proxy_k", "hier_terms")):
            category = "hierarchy_objective"
        rows.append({"field": field, "left": a[field], "right": b[field], "effect_family": category,
                     "interpretation": "Recorded/configured change; observed metric differences do not isolate its causal effect."})
    return rows


def _ledger_pair(left, right, snapshots, mechanisms=None, structure=None):
    a, b = left["analysis"], right["analysis"]
    epoch_alignment = _align_logs(a, b, "epoch")
    update_alignment = _align_logs(a, b, "model_updates")
    values = epoch_alignment["metrics"]
    def evidence(metric, expectation):
        record = values[metric]
        return {"metric": metric, "expected_direction": expectation,
                "paired_epoch_observation_count": record["paired_observation_count"],
                "mean_delta": record["mean_delta"], "last_same_epoch": record["last"],
                "status": "observed" if record["paired_observation_count"] else "unavailable"}
    saved_a, saved_b = left["selection"], right["selection"]
    saved_delta = {}
    for key in ("selected_validation_oa_pct", "selected_validation_aa_pct", "saved_final_test_oa_pct", "saved_final_test_aa_pct"):
        aa, bb = saved_a.get(key), saved_b.get(key)
        saved_delta[key] = {"left": aa, "right": bb, "delta": bb - aa if _finite(aa) and _finite(bb) else None}
    split_a, split_b = a.get("identity", {}).get("split_sha256"), b.get("identity", {}).get("split_sha256")
    return {"left_run_key": left["run_key"], "right_run_key": right["run_key"],
            "left_display_name": left["display_name"], "right_display_name": right["display_name"],
            "same_recorded_split": bool(split_a and split_a == split_b),
            "same_recorded_seed": a.get("identity", {}).get("seed") is not None and a.get("identity", {}).get("seed") == b.get("identity", {}).get("seed"),
            "changes": _changes(left["protocol"], right["protocol"]),
            "epoch_alignment": epoch_alignment, "model_update_alignment": update_alignment,
            "paired_fixed_pool_snapshots": _snapshot_pairs(left["run_key"], right["run_key"], snapshots),
            "paired_independent_structure": _structure_pairs(left["run_key"], right["run_key"], structure),
            "controlled_mechanism_alignment": _mechanism_pairs(left["run_key"], right["run_key"], mechanisms),
            "benefit_ledger": {
                "correctness": [evidence("proxy_replica_max_difference", "zero replica discrepancy"),
                                evidence("sample_self_k_fraction", "remove trivial self-k mining if that was the intended fix"),
                                evidence("proxy_self_k_fraction", "remove trivial self-k mining if that was the intended fix")],
                "training_protocol": [evidence("proxy_numerical_saturation_fraction", "avoid numerical projection saturation"),
                                      evidence("whole_near_boundary_fraction", "descriptive; inward movement alone is not better"),
                                      evidence("proxy_cap_hit_fraction", "descriptive post-step constraint events, not distinct dead proxies")],
                "hierarchy_objective": [evidence("sample_effective_proxy_count", "descriptive; more proxies is not necessarily better"),
                                       evidence("sample_active_noncollision_fraction", "descriptive hinge rate; zero does not prove useful structure"),
                                       evidence("sample_collision_fraction", "fewer collisions leave more usable constraints")],
                "joint_classification_outcome": saved_delta},
            "interpretation": ["All differences are single-seed historical observations; concurrent changes prevent isolated causal attribution.",
                               "Saved final test metrics are read for the already selected model; this audit neither selects a model on test nor runs test forwards.",
                               "Different original/source selection policies or training splits are reported, never converted into a matched HIER effect.",
                               "Training scalar geometry and fixed clean-pool geometry have different populations and must remain separate."]}



def _public_value(value):
    if isinstance(value, dict):
        return {k: _public_value(v) for k, v in value.items()
                if not any(token in str(k).lower() for token in ("host", "username", "user_name", "password", "credential", "uuid", "private_path", "physical_gpu"))}
    if isinstance(value, list):
        return [_public_value(v) for v in value]
    if isinstance(value, str):
        value = re.sub(r"[A-Za-z]:[\\/][^\s<>\"']+", "<private-path>", value)
        value = re.sub(r"/(?:mnt|home|tmp|root|Users|media|srv|opt)/[^\s<>\"']+", "<private-path>", value)
        value = re.sub(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", "<private-host>", value)
        value = re.sub(r"\b(?:GPU-)?[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b", "<private-id>", value)
        return value
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _shape_metrics(snapshot):
    result = {}
    for normalization, report in snapshot.get("shape", {}).items():
        for distance, measures in report.get("whole_relations", {}).items():
            for name, value in measures.items():
                if isinstance(value, dict) and _finite(value.get("mean")):
                    result["whole." + normalization + "." + distance + "." + name] = {
                        "mean": value["mean"], "class_balanced_mean": value.get("class_balanced_mean"),
                        "object_bootstrap_95ci": value.get("object_bootstrap_95ci"),
                        "class_bootstrap_95ci": value.get("class_bootstrap_95ci"),
                        "object_count": value.get("object_count"), "class_count": value.get("class_count")}
        proxies = report.get("proxy_relations", {})
        if proxies.get("available"):
            for distance, measures in proxies.get("methods", {}).items():
                if _finite(measures.get("mean_chamfer_enrichment")):
                    result["proxy." + normalization + "." + distance + ".chamfer_enrichment"] = {
                        "mean": measures["mean_chamfer_enrichment"], "proxy_count": measures.get("proxy_count"),
                        "confidence_interval": None, "interval_reason": "Shared objects prevent treating proxies as IID."}
    return result


def _mechanism_metrics(snapshot):
    result = {}
    for component, report in snapshot.get("components", {}).items():
        noise = report.get("noise_summary", {})
        for name, value in noise.get("scalar_variation", {}).items():
            if _finite(value.get("mean")):
                result[component + ".noise." + name] = {"mean": value["mean"], "std": value.get("std"),
                                                        "draws": noise.get("draws")}
        for name, value in noise.get("independent_noise_pair_agreement", {}).items():
            if _finite(value.get("mean")):
                result[component + ".noise_agreement." + name] = {"mean": value["mean"], "pairs": value.get("pairs")}
    for component, repeats in snapshot.get("gradient_summaries", {}).items():
        for kind in ("proxy_tangent", "whole_ball", "unweighted_proxy_tangent", "unweighted_whole_ball"):
            for metric in ("gradient_norm", "radial_norm", "angular_norm", "signed_radial_mean",
                           "above_tolerance_row_count", "exact_nonzero_row_count"):
                values = [row.get(kind, {}).get(metric) for row in repeats if isinstance(row.get(kind), dict)]
                value = _mean(values)
                if value is not None:
                    result[component + ".gradient." + kind + "." + metric] = {"mean": value, "repeats": len(values),
                        "interpretation": "Gradient components alone do not establish optimizer displacement; weighted and unweighted measurements remain separate."}
    return result


def _external_for_run(key, summary, kind):
    if not summary:
        return []
    if not isinstance(summary, dict):
        raise ValueError("External evidence must be a mapping")
    results = []
    for row in summary.get("snapshots", []):
        if row.get("run_key") != key:
            continue
        if kind == "shape":
            identity = {k: row.get(k) for k in ("sample_pool_sha256", "input_sha256", "c", "candidate_count")}
            identity["audit_ids_sha256"] = summary.get("audit_ids_sha256")
            metrics = _shape_metrics(row)
            note = "Independent geometry signal with object/class bootstrap; class purity is auxiliary. Proxy coherence has no proxy-IID confidence interval."
        else:
            identity = {**row.get("identity", {}), "query_plan_sha256": row.get("query_plan_sha256"),
                        "production_operator": row.get("production_operator"),
                        "margin_distance_units": row.get("margin_distance_units"),
                        "gumbel_distance_scale_tau": row.get("gumbel_distance_scale_tau"),
                        "lambda_hier": row.get("lambda_hier"), "noise_plan": row.get("noise_plan")}
            metrics = _mechanism_metrics(row)
            note = "Fixed-query noise repetitions isolate same-checkpoint Gumbel variation; other versions may have different reference-mined query identities."
        results.append({"epoch": row.get("epoch"), "identity": _public_value(identity), "metrics": metrics,
                        "classification": row.get("classification") if kind == "shape" else None,
                        "interpretation": note})
    return results


def _structure_pairs(a_key, b_key, summary):
    a = {r["epoch"]: r for r in _external_for_run(a_key, summary, "shape")}
    b = {r["epoch"]: r for r in _external_for_run(b_key, summary, "shape")}
    results = []
    for epoch in sorted(set(a) & set(b)):
        aa, bb = a[epoch], b[epoch]
        fields = ("sample_pool_sha256", "input_sha256", "audit_ids_sha256", "c")
        matched = all(aa["identity"].get(f) is not None and aa["identity"].get(f) == bb["identity"].get(f) for f in fields)
        result = {"epoch": epoch, "available": matched,
                  "reason": None if matched else "Independent audit object/input/pool/curvature identity is missing or differs",
                  "proxy_identity_paired": False}
        if matched:
            result["metrics"] = {key: {"left": aa["metrics"][key], "right": bb["metrics"][key],
                                      "mean_delta": bb["metrics"][key]["mean"] - aa["metrics"][key]["mean"],
                                      "delta_confidence_interval": None,
                                      "interpretation": "Individual bootstrap intervals are retained; no paired-difference significance is inferred."}
                                 for key in sorted(set(aa["metrics"]) & set(bb["metrics"]))}
        results.append(result)
    return results


def _mechanism_pairs(a_key, b_key, summary):
    a = {r["epoch"]: r for r in _external_for_run(a_key, summary, "mechanism")}
    b = {r["epoch"]: r for r in _external_for_run(b_key, summary, "mechanism")}
    results = []
    fields = ("sample_pool_sha256", "input_sha256", "query_plan_sha256", "production_operator",
              "margin_distance_units", "gumbel_distance_scale_tau", "noise_plan")
    for epoch in sorted(set(a) & set(b)):
        aa, bb = a[epoch], b[epoch]
        matched = all(aa["identity"].get(f) is not None and aa["identity"].get(f) == bb["identity"].get(f) for f in fields)
        # Equal query IDs allow scalar comparison, never semantic row-ID matching.
        result = {"epoch": epoch, "paired_query_available": matched, "proxy_identity_paired": False,
                  "reason": None if matched else "Query plans, inputs or distance/noise identities differ/missing; compare each run descriptively, not as paired ancestors"}
        if matched:
            result["metrics"] = {key: {"left": aa["metrics"][key]["mean"], "right": bb["metrics"][key]["mean"],
                                      "delta": bb["metrics"][key]["mean"] - aa["metrics"][key]["mean"]}
                                 for key in sorted(set(aa["metrics"]) & set(bb["metrics"]))
                                 if ".noise." in key}
        results.append(result)
    return results


def _external_inventory(value, kind):
    if not value:
        return {"available": False, "kind": kind, "reason": "No supplied independent/control audit summary"}
    if not isinstance(value, dict):
        raise ValueError(kind + " summary must be an explicit mapping")
    return {"available": True, "kind": kind, "schema_version": value.get("schema_version"),
            "source_identity": _public_value(value.get("source_identity", value.get("provenance"))),
            "supplied_keys": sorted(k for k in value if k not in ("arrays_base_dir", "private_source_path")),
            "interpretation": "Attached evidence retains its own run, input/query and operator identity. It is not automatically assigned as a causal benefit to a version change."}


def _write_csv(path, rows):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def _figures(runs, pairs, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    directory = Path(output) / "figures"
    directory.mkdir(parents=True, exist_ok=True)
    figures = []
    def save(fig, stem, title, note):
        fig.tight_layout()
        for extension in ("png", "svg"):
            fig.savefig(directory / (stem + "." + extension), dpi=150, bbox_inches="tight")
        plt.close(fig)
        figures.append({"run_id": "all_runs", "run_ids": [r["run_key"] for r in runs],
                        "kind": "version_comparison", "title": title, "note": note,
                        "png": "figures/" + stem + ".png", "svg": "figures/" + stem + ".svg"})
    def plot_series(ax, run, field, axis="epoch"):
        rows = [r for r in run["analysis"].get("epochs", []) if _finite(r.get(axis)) and _finite(r.get("metrics", {}).get(field))]
        if not rows:
            return False
        rows.sort(key=lambda r: r[axis])
        xx, yy = [], []
        previous = None
        for row in rows:
            if previous is not None and row["epoch"] != previous + 1:
                xx.append(np.nan); yy.append(np.nan)
            xx.append(row[axis]); yy.append(row["metrics"][field]); previous = row["epoch"]
        ax.plot(xx, yy, label=run["display_name"], linewidth=1.1)
        return True
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for run in runs:
        for ax, axis in zip(axes, ("epoch", "model_updates")):
            plot_series(ax, run, "val_oa_pct", axis)
    for ax, label in zip(axes, ("Saved epoch", "Nominal model updates (epoch × configured budget)")):
        ax.set(xlabel=label, ylabel="Validation OA (%)"); ax.grid(alpha=.2); ax.legend(fontsize=7)
    save(fig, "comparison_validation_axes", "Validation trajectories by actual epoch and nominal update coordinate",
         "Observed logs only. Missing epochs are not connected. Update alignment does not imply the same sampled inputs, HIER exposure or complete inherited prefix.")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for run in runs:
        plot_series(axes[0], run, "whole_near_boundary_fraction")
        plot_series(axes[1], run, "proxy_numerical_saturation_fraction")
    for ax, title in zip(axes, ("Training whole near native boundary", "Proxy actual numerical projection saturation")):
        ax.set(xlabel="Saved epoch", ylabel="Observed fraction", title=title); ax.grid(alpha=.2)
        ax.set_ylim(-.025, 1.025)
        if ax.lines: ax.legend(fontsize=7)
        else: ax.text(.5, .5, "Not recorded", transform=ax.transAxes, ha="center")
    save(fig, "comparison_boundary_and_saturation", "Whole boundary concentration and proxy numerical saturation",
         "Training-forward populations only. Eliminating proxy numeric saturation does not establish that whole uses its radial range or that the learned structure is meaningful.")
    selected = [r for r in runs if r["selection"]["validation_selected"] and
                _finite(r["selection"]["selected_validation_oa_pct"])]
    if selected:
        fig, ax = plt.subplots(figsize=(max(9, 1.8 * len(selected)), 4.5))
        x = np.arange(len(selected))
        ax.scatter(x - .12, [r["selection"]["selected_validation_oa_pct"] for r in selected], label="Saved validation-selected OA", marker="o")
        present = [(i, r["selection"]["saved_final_test_oa_pct"]) for i, r in enumerate(selected)
                   if _finite(r["selection"]["saved_final_test_oa_pct"])]
        if present: ax.scatter([i + .12 for i, _ in present], [v for _, v in present], label="Saved selected-model final test OA", marker="s")
        ax.set_xticks(x)
        ax.set_xticklabels([r["display_name"] for r in selected], rotation=15, ha="right")
        ax.set_ylabel("OA (%)"); ax.grid(axis="y", alpha=.2); ax.legend(fontsize=8)
        save(fig, "comparison_saved_selected_results", "Saved validation-selected model results",
             "No model was reselected and no test forward was added. Validation and test use different populations; single-seed differences with several configuration changes are not causal HIER benefits.")
    paired = [pair for pair in pairs if any(s["paired_pool_available"] for s in pair["paired_fixed_pool_snapshots"])]
    if paired:
        fig, ax = plt.subplots(figsize=(12, 4.8))
        for pair in paired:
            valid = [s for s in pair["paired_fixed_pool_snapshots"] if s["paired_pool_available"]]
            points = [(s["epoch"], s.get("metrics", {}).get("hyperbolic_unique_fraction", {}).get("delta")) for s in valid]
            points = [(x, y) for x, y in points if _finite(y)]
            if points:
                ax.plot([x for x, _ in points], [y for _, y in points], marker="o",
                        label=pair["right_display_name"] + " minus " + pair["left_display_name"])
        ax.axhline(0, color="black", linewidth=.7); ax.set(xlabel="Common actual checkpoint epoch", ylabel="Raw top4 candidate coverage fraction difference")
        ax.grid(alpha=.2)
        if ax.lines: ax.legend(fontsize=7, loc="best")
        save(fig, "comparison_paired_pool_coverage", "Top4 coverage changes in identity-verified fixed input pools",
             "Only equal sample pools, actual input SHA, labels, inference conditions and curvature are compared. Proxy rows are not paired across versions. Coverage is bounded by proxy_count × topk.")
    return figures


def build_comparison(analyses, snapshots=None, mechanisms=None, structure=None,
                     output_dir=None, make_plots=True, registry=None):
    """Create a joint descriptive benefit ledger and optional JSON/CSV/figures.

    Existing comparison output files are rejected. No GPU, checkpoint loading,
    optimizer updates, test forwarding or test-driven model selection occurs.
    """
    if registry is None:
        source = Path(__file__).resolve().parents[2] / "docs/research/experiment_registry.json"
        registry = json.loads(source.read_text(encoding="utf-8-sig")) if source.is_file() else {}
    entries = {r["display_name"]: r for r in registry.get("runs", [])}
    runs = []
    for analysis in analyses:
        name = analysis.get("display_name", run_key(analysis))
        entry = entries.get(name, {})
        runs.append({"run_key": run_key(analysis), "display_name": name, "analysis": analysis,
                     "protocol": _protocol(analysis, entry), "selection": _saved_selection(analysis, entry),
                     "observations": _metric_summary(analysis),
                     "independent_structure": _external_for_run(run_key(analysis), structure, "shape"),
                     "controlled_mechanisms": _external_for_run(run_key(analysis), mechanisms, "mechanism")})
    if len({r["run_key"] for r in runs}) != len(runs):
        raise ValueError("Cross-version audit requires unique qualified run keys")
    pairs = [_ledger_pair(a, b, snapshots, mechanisms, structure) for a, b in itertools.combinations(runs, 2)]
    result = {"schema_version": 1, "runs": [{k: v for k, v in r.items() if k != "analysis"} for r in runs],
              "pairs": pairs, "external_evidence": [_external_inventory(mechanisms, "controlled_mechanisms"),
                                                   _external_inventory(structure, "independent_structure")],
              "warnings": ["No isolated causal ranking is produced for multi-factor, single-seed historical runs.",
                           "Proxy IDs are stable only inside a run; cross-run comparisons use distributions and verified candidate pools."],
              "figures": []}
    if output_dir is not None:
        output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
        for name in ("comparison_summary.json", "comparison_benefit_ledger.csv", "comparison_aligned_observations.csv"):
            if (output / name).exists():
                raise FileExistsError("Existing comparison audit outputs cannot be overwritten")
        if make_plots:
            result["figures"] = _figures(runs, pairs, output)
        (output / "comparison_summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        ledger_rows, aligned_rows = [], []
        for pair in pairs:
            context = {k: pair[k] for k in ("left_run_key", "right_run_key", "same_recorded_split", "same_recorded_seed")}
            for family, records in pair["benefit_ledger"].items():
                if family == "joint_classification_outcome":
                    ledger_rows.extend({**context, "effect_family": family, "metric": key, **value} for key, value in records.items())
                else:
                    ledger_rows.extend({**context, "effect_family": family, **{k: v for k, v in value.items() if k != "last_same_epoch"}}
                                       for value in records)
            for snapshot in pair["paired_independent_structure"]:
                if snapshot.get("available"):
                    for metric, value in snapshot.get("metrics", {}).items():
                        ledger_rows.append({**context, "effect_family": "hierarchy_objective",
                                            "evidence_kind": "independent_shape",
                                            "checkpoint_epoch": snapshot["epoch"], "metric": metric,
                                            "left": value["left"]["mean"], "right": value["right"]["mean"],
                                            "delta": value["mean_delta"], "delta_confidence_interval": None,
                                            "interpretation": value["interpretation"]})
            for snapshot in pair["controlled_mechanism_alignment"]:
                if snapshot.get("paired_query_available"):
                    for metric, value in snapshot.get("metrics", {}).items():
                        ledger_rows.append({**context, "effect_family": "hierarchy_objective",
                                            "evidence_kind": "fixed_query_noise",
                                            "checkpoint_epoch": snapshot["epoch"], "metric": metric, **value})
                else:
                    ledger_rows.append({**context, "effect_family": "hierarchy_objective",
                                        "evidence_kind": "fixed_query_noise",
                                        "checkpoint_epoch": snapshot["epoch"], "status": "unpaired",
                                        "interpretation": snapshot.get("reason")})
            for alignment in ("epoch_alignment", "model_update_alignment"):
                aligned_rows.extend({**context, **value} for value in pair[alignment]["observations"])
        _write_csv(output / "comparison_benefit_ledger.csv", ledger_rows)
        _write_csv(output / "comparison_aligned_observations.csv", aligned_rows)
    return result
