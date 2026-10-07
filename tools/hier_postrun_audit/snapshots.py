"""Version-independent, read-only CPU audits of explicit embedding snapshots.

A nearest-neighbour relationship is never called a training-selected ancestor.
Missing cache, mapping, identity or activation data are reported as unavailable.
Saved reports contain basenames/hashes, never private absolute source paths.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import pickle
from pathlib import Path
import numpy as np

from .geometry import (align_rows, ball_geometry, displacement, expmap0, matrix,
                       pool_hash, quantiles, topk_neighbours, topk_stability, unique_ids)


def _file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_npz(path):
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    metadata = {}
    for key in ("c", "checkpoint_epoch", "input_mode", "inputmode", "input_sha256",
                "sample_pool_sha256", "sample_pool_sha", "pool_ids_sha256"):
        if key in arrays and arrays[key].shape == ():
            metadata[key] = arrays[key].item()
    for key in ("metadata_json", "metadata"):
        if key in arrays and arrays[key].shape == () and arrays[key].dtype.kind in "US":
            value = json.loads(str(arrays[key].item()))
            if not isinstance(value, dict):
                raise ValueError("Cache metadata must be a JSON object")
            metadata.update(value)
    return arrays, metadata


def _checkpoint(path, trusted=False):
    # Torch is optional unless the user explicitly requests checkpoint extraction.
    import torch
    try:
        saved = torch.load(path, map_location="cpu", weights_only=not bool(trusted))
    except pickle.UnpicklingError as exc:
        raise ValueError("Checkpoint needs explicit trusted_checkpoint=true to load legacy Python/RNG objects; use only your own trusted checkpoint") from exc
    if not isinstance(saved, dict):
        raise ValueError("Checkpoint must be a mapping")
    result = {"epoch": saved.get("completed_epochs", saved.get("epoch")),
              "format": saved.get("format"), "proxy_tangent": None}
    state = saved.get("proxy", {})
    if isinstance(state, dict):
        candidates = [value for key, value in state.items() if str(key).split(".")[-1] == "tangent_proxies"]
        if len(candidates) > 1:
            raise ValueError("Checkpoint contains ambiguous proxy parameter tensors")
        if candidates:
            result["proxy_tangent"] = candidates[0].detach().cpu().numpy().copy()
    return result


def _summary_geometry(geometry, mask=None):
    mask = np.ones(len(geometry["radius"]), dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    return {"count": int(mask.sum()), "radius_quantiles": quantiles(geometry["radius"][mask]),
            "normalized_radius_quantiles": quantiles(geometry["normalized_radius"][mask]),
            "depth_quantiles": quantiles(geometry["depth"][mask]),
            "mapped_tangent_norm_quantiles": quantiles(geometry["mapped_tangent_norm"][mask])}


def _retrieval_summary(rows, samples, labels, geometry, distances):
    ids = samples[rows]
    count = len(samples)
    slot_counts = np.bincount(rows.reshape(-1), minlength=count)
    total_slots = int(slot_counts.sum())
    probabilities = slot_counts.astype(float) / max(total_slots, 1)
    descending = np.sort(probabilities)[::-1]
    active_probabilities = probabilities[probabilities > 0]
    entropy = -float(np.sum(active_probabilities * np.log(active_probabilities)))
    ordered = np.sort(slot_counts.astype(float))
    gini = ((2. * np.dot(np.arange(1, count + 1), ordered) / (count * total_slots))
            - (count + 1.) / count) if total_slots and count else 0.
    tuple_counts = Counter(tuple(sorted(row.tolist())) for row in ids)
    purity = np.array([max(Counter(labels[row].tolist()).values()) / len(row) for row in rows])
    sorted_radius = np.sort(geometry["radius"])
    # Midrank CDF handles large boundary ties; no artificial jitter is introduced.
    radius = geometry["radius"]
    radius_percentile = (np.searchsorted(sorted_radius, radius, side="left") +
                         np.searchsorted(sorted_radius, radius, side="right")) / (2. * count)
    selected_percentiles = radius_percentile[rows]
    hot = np.argsort(-slot_counts, kind="stable")[:min(20, count)]
    source_geometry = {"candidate_radius_quantiles": quantiles(radius),
                       "selected_radius_quantiles": quantiles(radius[rows]),
                       "candidate_depth_quantiles": quantiles(geometry["depth"]),
                       "selected_depth_quantiles": quantiles(geometry["depth"][rows]),
                       "selected_radius_percentile_quantiles": quantiles(selected_percentiles),
                       "selected_radius_percentile_mean": float(np.mean(selected_percentiles)) if selected_percentiles.size else None}
    return {"topk_sample_ids": ids.tolist(), "proxy_topk_purity": purity.tolist(),
            "nearest_distance_quantiles": quantiles(distances),
            "coverage": {"unique_samples": int(np.count_nonzero(slot_counts)),
                         "total_candidates": count,
                         "unique_fraction": float(np.count_nonzero(slot_counts) / count)},
            "slot_concentration": {"max_sample_slot_share": float(descending[0]) if len(descending) else None,
                                   "top4_sample_slot_share": float(descending[:4].sum()),
                                   "top10_sample_slot_share": float(descending[:10].sum()),
                                   "effective_sample_count": float(np.exp(entropy)) if total_slots else 0.,
                                   "slot_hhi": float(np.sum(probabilities ** 2)), "gini": float(gini)},
            "repeated_tuples": {"unique_topk_sets": len(tuple_counts),
                                "largest_identical_topk_count": max(tuple_counts.values(), default=0),
                                "largest_identical_topk_fraction": max(tuple_counts.values(), default=0) / max(len(rows), 1)},
            "purity": {"mean": float(purity.mean()) if len(purity) else None,
                       "all_same_class_count": int(np.sum(purity == 1)),
                       "all_same_class_fraction": float(np.mean(purity == 1)) if len(purity) else None},
            "radius_bias": source_geometry,
            "hotspots": [{"sample_id": samples[i].item(), "label": labels[i].item(),
                          "slots": int(slot_counts[i]), "slot_share": float(probabilities[i]),
                          "depth": float(geometry["depth"][i]), "radius_percentile": float(radius_percentile[i])}
                         for i in hot if slot_counts[i] > 0]}, slot_counts, purity


def _activation(spec, arrays, proxy_ids):
    used_ids = spec.get("used_proxy_ids")
    if used_ids is None:
        for key in ("training_used_proxy_ids", "used_proxy_ids"):
            if key in arrays:
                used_ids = arrays[key]
                break
    counts = spec.get("proxy_activation_counts", arrays.get("proxy_activation_counts"))
    if counts is not None:
        counts = np.asarray(counts).reshape(-1)
        if len(counts) != len(proxy_ids) or np.any(counts < 0) or not np.isfinite(counts).all():
            raise ValueError("Training activation counts do not match proxy axis")
        count_ids = proxy_ids[counts > 0]
        if used_ids is not None and set(np.asarray(used_ids).tolist()) != set(count_ids.tolist()):
            raise ValueError("Training used IDs and activation counts disagree")
        used_ids = count_ids
    if used_ids is None:
        return None, {"available": False, "reason": "No explicit training-selected proxy identities/counts"}
    used_ids = unique_ids(used_ids, "used_proxy_ids")
    if not set(used_ids.tolist()).issubset(set(proxy_ids.tolist())):
        raise ValueError("Used proxy identity is absent from the snapshot")
    mask = np.isin(proxy_ids, used_ids)
    summary = {"available": True, "used_count": int(mask.sum()), "inactive_count": int((~mask).sum()),
               "scope": spec.get("activation_scope", "provided explicit selection window"),
               "used_proxy_ids": used_ids.tolist()}
    if counts is not None:
        total = float(counts.sum())
        probabilities = counts / total if total else counts.astype(float)
        positive = probabilities[probabilities > 0]
        summary.update(total_selection_slots=total, effective_proxy_count=float(np.exp(-np.sum(positive * np.log(positive)))) if total else 0.,
                       max_proxy_selection_share=float(probabilities.max()) if len(probabilities) else None)
    return mask, summary


def _analyze_one(spec, index, output):
    cache_path = Path(spec["cache"])
    cache, cache_metadata = _load_npz(cache_path)
    checkpoint_info = None
    if spec.get("checkpoint"):
        checkpoint_info = _checkpoint(spec["checkpoint"], spec.get("trusted_checkpoint", False))
    proxy_cache, proxy_metadata = {}, {}
    if spec.get("proxy_cache"):
        proxy_cache, proxy_metadata = _load_npz(spec["proxy_cache"])
    arrays = {**cache, **proxy_cache}
    metadata = {**cache_metadata, **proxy_metadata, **spec.get("metadata", {})}
    c = spec.get("c", metadata.get("c"))
    if c is None:
        raise ValueError("Curvature c is missing; numerical geometry is not inferred from version/name")
    c = float(c)
    whole = matrix(cache.get("mu", cache.get("whole_mu")), "whole ball coordinates")
    samples = unique_ids(cache["sample_ids"])
    labels = np.asarray(cache["labels"]).reshape(-1)
    if len(whole) != len(samples) or len(labels) != len(samples) or not len(samples):
        raise ValueError("Whole coordinates, identities and labels have inconsistent lengths")
    if labels.dtype.kind not in "iuUS":
        raise ValueError("Labels must be integers or strings")
    order = np.argsort(samples, kind="stable")
    whole, samples, labels = whole[order], samples[order], labels[order]
    geometry = ball_geometry(whole, c)
    identity_hash = pool_hash(samples)
    declared_hash = metadata.get("sample_pool_sha256", metadata.get("sample_pool_sha", metadata.get("pool_ids_sha256")))
    epoch = spec.get("epoch", metadata.get("checkpoint_epoch"))
    if epoch is None and checkpoint_info:
        epoch = checkpoint_info["epoch"]
    if epoch is None:
        raise ValueError("Snapshot epoch is missing; filename order is not a training identity")
    if checkpoint_info and checkpoint_info["epoch"] is not None and int(epoch) != int(checkpoint_info["epoch"]):
        raise ValueError("Declared snapshot epoch does not match checkpoint epoch")
    inputmode = spec.get("inputmode", spec.get("input_mode", metadata.get("inputmode", metadata.get("input_mode", "unknown"))))
    input_sha = spec.get("input_sha256", metadata.get("input_sha256"))
    inference_condition = spec.get("inference_condition", metadata.get("inference_condition"))
    if inference_condition is not None and not isinstance(inference_condition, dict):
        raise ValueError("inference_condition must be an explicit JSON object")
    whole_summary = {**_summary_geometry(geometry), "dimension": whole.shape[1], "c": c,
                     "class_depth": [{"label": label.item(), "count": int(np.sum(labels == label)),
                                      "depth_quantiles": quantiles(geometry["depth"][labels == label])}
                                     for label in np.unique(labels)]}
    source = {"cache_name": cache_path.name, "cache_sha256": _file_hash(cache_path),
              "sample_pool_sha256": identity_hash, "declared_sample_pool_sha": declared_hash,
              "inputmode": inputmode, "input_sha256": input_sha,
              "inference_condition": inference_condition, "epoch_identity_available": True}
    if spec.get("checkpoint"):
        source.update(checkpoint_name=Path(spec["checkpoint"]).name,
                      checkpoint_sha256=_file_hash(spec["checkpoint"]),
                      checkpoint_format=checkpoint_info["format"])
    result = {"run_key": spec.get("run_key"), "label": spec.get("label", spec.get("run_key", "unnamed snapshot")),
              "epoch": int(epoch), "source": source, "whole": whole_summary,
              "proxy": {"available": False, "reason": "No proxy tensors in explicit inputs"},
              "retrieval": {"available": False, "reason": "No mapped proxies"}}
    saved_arrays = {"sample_ids": samples, "labels": labels, "whole_radius": geometry["radius"],
                    "whole_normalized_radius": geometry["normalized_radius"], "whole_depth": geometry["depth"]}
    tangent = arrays.get("proxy_tangent", arrays.get("tangent_proxies"))
    if tangent is None and checkpoint_info:
        tangent = checkpoint_info["proxy_tangent"]
    proxy = arrays.get("proxy_ball")
    mapping = spec.get("proxy_mapping", metadata.get("proxy_mapping"))
    if tangent is not None:
        tangent = matrix(tangent, "proxy tangent parameters")
        if mapping is not None:
            expected = expmap0(tangent, c, **mapping)
            if proxy is None:
                proxy = expected
            elif not np.allclose(proxy, expected, atol=float(spec.get("mapping_atol", 2e-6)), rtol=2e-6):
                raise ValueError("Explicit proxy mapping does not match cached proxy coordinates")
        elif proxy is None:
            result["proxy"] = {"available": True, "mapped_available": False,
                               "count": len(tangent), "parameter_tangent_norm_quantiles": quantiles(np.linalg.norm(tangent, axis=1)),
                               "reason": "Proxy forward mapping is missing; saved tangent parameters are not ball positions"}
            saved_arrays["proxy_tangent_norm"] = np.linalg.norm(tangent, axis=1)
    private = {"whole": whole, "samples": samples, "labels": labels, "c": c,
               "pool_hash": identity_hash, "inputmode": inputmode, "input_sha256": input_sha,
               "inference_condition": inference_condition, "proxy": None,
               "tangent": tangent, "proxy_ids": None, "proxy_ids_stable": False, "topk": {},
               "used_mask": None, "activation_scope": None}
    if proxy is not None:
        proxy = matrix(proxy, "proxy ball coordinates")
        if proxy.shape[1] != whole.shape[1] or tangent is not None and tangent.shape != proxy.shape:
            raise ValueError("Whole/proxy/tangent dimensions are inconsistent")
        stable_ids = "proxy_ids" in arrays or "proxy_ids" in spec
        proxy_ids = unique_ids(spec.get("proxy_ids", arrays.get("proxy_ids", np.arange(len(proxy)))), "proxy_ids")
        if len(proxy_ids) != len(proxy):
            raise ValueError("Proxy ID axis does not match proxy coordinates")
        pgeom = ball_geometry(proxy, c)
        used, activation = _activation(spec, arrays, proxy_ids)
        result["proxy"] = {"available": True, "mapped_available": True, "count": len(proxy),
                           "depth_quantiles": quantiles(pgeom["depth"]),
                           "mapped_radius_quantiles": quantiles(pgeom["radius"]),
                           "normalized_radius_quantiles": quantiles(pgeom["normalized_radius"]),
                           "all": _summary_geometry(pgeom), "training_activation": activation,
                           "used_proxy_ids_available": used is not None, "proxy_ids_stable": stable_ids,
                           "mapping": mapping if mapping is not None else {"status": "unknown; cached ball coordinates measured directly"}}
        result["proxy"]["relation_to_whole_depth"] = {
            "fraction_below_whole_median": float(np.mean(pgeom["depth"] < np.median(geometry["depth"]))),
            "fraction_above_whole_p95": float(np.mean(pgeom["depth"] > np.quantile(geometry["depth"], .95))),
            "fraction_above_whole_max": float(np.mean(pgeom["depth"] > np.max(geometry["depth"]))),
            "note": "Population depth comparison is not a per-proxy ancestor constraint"}
        if used is not None:
            result["proxy"].update(used=_summary_geometry(pgeom, used), inactive=_summary_geometry(pgeom, ~used))
        if tangent is not None:
            norms = np.linalg.norm(tangent, axis=1)
            result["proxy"]["parameter_tangent_norm_quantiles"] = quantiles(norms)
            saved_arrays["proxy_tangent_norm"] = norms
            if mapping and mapping.get("numeric_radius_fraction") is not None:
                threshold = np.arctanh(float(mapping["numeric_radius_fraction"])) / np.sqrt(c)
                effective = norms.copy()
                if mapping.get("source_tangent_cap") is not None:
                    effective *= np.minimum(1., float(mapping["source_tangent_cap"]) /
                                            np.maximum(norms + float(mapping.get("source_cap_epsilon", 0.)), np.finfo(float).tiny))
                result["proxy"]["numeric_projection_count"] = int(np.sum(effective > threshold))
            if mapping and mapping.get("source_tangent_cap") is not None:
                result["proxy"]["source_tangent_cap_exceeded_count"] = int(np.sum(norms > float(mapping["source_tangent_cap"])))
        saved_arrays.update(proxy_ids=proxy_ids, proxy_depth=pgeom["depth"], proxy_radius=pgeom["radius"],
                            proxy_normalized_radius=pgeom["normalized_radius"])
        if used is not None:
            saved_arrays["proxy_used_mask"] = used
        k = int(spec.get("topk", 4))
        result["retrieval"] = {"available": len(samples) >= k, "topk": k,
                               "relation_kind": "full-candidate-pool nearest neighbours; not training-selected ancestors",
                               "candidate_count": len(samples), "proxy_ids": proxy_ids.tolist()}
        if len(samples) >= k:
            metrics = ["hyperbolic", "direction"]
            if spec.get("equal_radius_fraction") is not None:
                metrics.append("equal_radius_hyperbolic")
            for metric in metrics:
                valid_proxy = pgeom["direction_defined"] if metric in ("direction", "equal_radius_hyperbolic") else np.ones(len(proxy), dtype=bool)
                if metric in ("direction", "equal_radius_hyperbolic") and not geometry["direction_defined"].all():
                    result["retrieval"][metric] = {"available": False, "reason": "Zero-radius candidate has undefined direction"}
                    continue
                if not valid_proxy.all():
                    result["retrieval"][metric] = {"available": False, "reason": "Zero-radius proxy has undefined direction"}
                    continue
                rows, distances = topk_neighbours(proxy, whole, samples, c, k=k, metric=metric,
                                                   equal_radius_fraction=spec.get("equal_radius_fraction"))
                summary, slot_counts, purity = _retrieval_summary(rows, samples, labels, geometry, distances)
                summary["available"] = True
                neighbour_depth = geometry["depth"][rows]
                summary["proxy_neighbour_depth"] = {
                    "proxy_shallower_than_all_topk_fraction": float(np.mean(np.all(pgeom["depth"][:, None] < neighbour_depth, axis=1))),
                    "proxy_shallower_than_any_topk_fraction": float(np.mean(np.any(pgeom["depth"][:, None] < neighbour_depth, axis=1))),
                    "neighbour_mean_depth_minus_proxy_depth_quantiles": quantiles(neighbour_depth.mean(axis=1) - pgeom["depth"]),
                    "note": "Nearest neighbours are not training-selected descendants"}
                if used is not None:
                    summary["groups"] = {}
                    for group_name, group_mask in (("used", used), ("inactive", ~used)):
                        group_summary, _, _ = _retrieval_summary(rows[group_mask], samples, labels, geometry, distances[group_mask])
                        group_summary["proxy_count"] = int(group_mask.sum())
                        summary["groups"][group_name] = group_summary
                result["retrieval"][metric] = summary
                saved_arrays[metric + "_topk_ids"] = samples[rows]
                saved_arrays[metric + "_slot_counts"] = slot_counts
                saved_arrays[metric + "_proxy_topk_purity"] = purity
                private["topk"][metric] = samples[rows]
            if "hyperbolic" in private["topk"] and "direction" in private["topk"]:
                result["retrieval"]["raw_direction_agreement"] = topk_stability(private["topk"]["hyperbolic"], private["topk"]["direction"])
        else:
            result["retrieval"]["reason"] = "Candidate pool is smaller than topk; no silent change to k"
        private.update(proxy=proxy, proxy_ids=proxy_ids, proxy_ids_stable=stable_ids,
                       used_mask=used, activation_scope=activation.get("scope"))
    arrays_name = "snapshot_%03d_arrays.npz" % index
    np.savez_compressed(output / arrays_name, **saved_arrays)
    result["arrays_file"] = arrays_name
    return result, private


def _transition(previous, current, old_private, new_private):
    result = {"run_key": current["run_key"], "from_epoch": previous["epoch"],
              "to_epoch": current["epoch"], "available": False}
    if old_private["pool_hash"] != new_private["pool_hash"]:
        result["reason"] = "Sample ID pools differ; this is not a turnover comparison"
        return result
    if old_private["inputmode"] == "unknown" or old_private["inputmode"] != new_private["inputmode"]:
        result["reason"] = "Input/evaluation mode is missing or differs"
        return result
    if not old_private["input_sha256"] or not new_private["input_sha256"]:
        result["reason"] = "Actual input SHA256 is missing; equal IDs/modes do not establish equal point-cloud inputs"
        return result
    if old_private["input_sha256"] != new_private["input_sha256"]:
        result["reason"] = "Actual point-cloud input SHA256 differs"
        return result
    old_condition, new_condition = old_private["inference_condition"], new_private["inference_condition"]
    if old_condition is not None or new_condition is not None:
        if old_condition != new_condition:
            result["reason"] = "Inference conditions differ or are missing for one snapshot"
            return result
    if old_private["c"] != new_private["c"] or old_private["whole"].shape[1] != new_private["whole"].shape[1]:
        result["reason"] = "Curvature or embedding dimension differs"
        return result
    rows = align_rows(old_private["samples"], new_private["samples"])
    if not np.array_equal(old_private["labels"], new_private["labels"][rows]):
        result["reason"] = "Aligned sample IDs have inconsistent labels"
        return result
    result.update(available=True, whole=displacement(old_private["whole"], new_private["whole"][rows], new_private["c"]),
                  inference_condition_matched=old_condition is not None,
                  coordinate_frame_warning="Angular coordinate motion includes any global frame rotation; neighbour turnover is invariant to shared rotations")
    if old_private["proxy"] is None or new_private["proxy"] is None:
        result["proxy"] = {"available": False, "reason": "Mapped proxies missing"}
        return result
    if not old_private["proxy_ids_stable"] or not new_private["proxy_ids_stable"]:
        result["proxy"] = {"available": False, "reason": "Explicit stable proxy IDs missing"}
        return result
    try:
        rows = align_rows(old_private["proxy_ids"], new_private["proxy_ids"])
    except ValueError:
        result["proxy"] = {"available": False, "reason": "Proxy identity pools differ"}
        return result
    result["proxy"] = {"available": True, **displacement(old_private["proxy"], new_private["proxy"][rows], new_private["c"])}
    if old_private["used_mask"] is not None and new_private["used_mask"] is not None:
        if old_private["activation_scope"] == new_private["activation_scope"]:
            old_used, new_used = old_private["used_mask"], new_private["used_mask"][rows]
            union = int(np.sum(old_used | new_used))
            result["proxy"]["training_active_set_stability"] = {
                "available": True, "used_set_jaccard": float(np.sum(old_used & new_used) / union) if union else 1.,
                "newly_used_count": int(np.sum(~old_used & new_used)),
                "became_inactive_count": int(np.sum(old_used & ~new_used)),
                "scope": old_private["activation_scope"]}
        else:
            result["proxy"]["training_active_set_stability"] = {"available": False, "reason": "Activation windows differ"}
    if old_private["tangent"] is not None and new_private["tangent"] is not None:
        old_u, new_u = old_private["tangent"], new_private["tangent"][rows]
        result["proxy"]["parameter_tangent_norm_change"] = quantiles(np.linalg.norm(new_u, axis=1) - np.linalg.norm(old_u, axis=1))
        result["proxy"]["parameter_tangent_displacement"] = quantiles(np.linalg.norm(new_u - old_u, axis=1))
    result["retrieval"] = {}
    for metric in sorted(set(old_private["topk"]) & set(new_private["topk"])):
        try:
            result["retrieval"][metric] = topk_stability(old_private["topk"][metric], new_private["topk"][metric][rows])
        except ValueError as exc:
            result["retrieval"][metric] = {"available": False, "reason": str(exc)}
    return result


def analyze_snapshots(specs, output_dir):
    """Audit explicit snapshots into a NEW directory and return JSON-safe results.

    Returned arrays_base_dir is a local rendering hint. The persisted summary
    deliberately omits it. Never reuse a source cache/result directory here.
    """
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    results, private, availability = [], [], []
    for index, spec in enumerate(specs):
        source = Path(spec.get("cache", ""))
        if not spec.get("cache") or not source.is_file():
            availability.append({"label": spec.get("label", spec.get("run_key")), "epoch": spec.get("epoch"),
                                 "available": False, "reason": "Explicit clean feature cache is missing"})
            continue
        try:
            summary, internal = _analyze_one(spec, index, output)
        except (KeyError, ValueError, FileNotFoundError, ImportError) as exc:
            # Do not expose exceptions containing private source paths.
            reason = str(exc)
            for field in ("cache", "proxy_cache", "checkpoint"):
                if spec.get(field):
                    reason = reason.replace(str(spec[field]), Path(spec[field]).name)
            availability.append({"label": spec.get("label", spec.get("run_key")), "epoch": spec.get("epoch"),
                                 "available": False, "reason": "Snapshot rejected: " + reason})
            continue
        results.append(summary)
        private.append(internal)
        availability.append({"label": summary["label"], "epoch": summary["epoch"], "available": True})
    transitions = []
    groups = {}
    for index, summary in enumerate(results):
        if summary["run_key"]:
            groups.setdefault(summary["run_key"], []).append(index)
    for run_indices in groups.values():
        run_indices.sort(key=lambda i: results[i]["epoch"])
        for old_index, new_index in zip(run_indices, run_indices[1:]):
            transitions.append(_transition(results[old_index], results[new_index], private[old_index], private[new_index]))
    report = {"schema_version": "1", "snapshots": results, "transitions": transitions, "availability": availability,
              "interpretation": ["Nearest-neighbour coverage and training ancestor activation are separate measurements.",
                                  "Coordinates are measured in their declared curvature and actual forward mapping.",
                                  "Temporal stability requires the same run, sample pool, proxy IDs, input mode and actual input SHA256.",
                                  "Purity uses class labels and is not independent morphology validation."]}
    (output / "snapshot_summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return {**report, "arrays_base_dir": str(output.resolve())}
