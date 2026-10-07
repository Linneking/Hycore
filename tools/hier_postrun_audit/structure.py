"""Bounded CPU evidence from actual point clouds, independent of network/teacher.

Chamfer is a morphology proxy, not semantic hierarchy ground truth. Dataset
rotations are retained. Labels only stratify queries and match random controls.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from .geometry import ball_geometry, direction_distance, matrix, poincare_distance, pool_hash, quantiles, unique_ids


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1048576), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_npz(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def scalar(value):
    return value.item() if isinstance(value, np.generic) else value


def rank(values):
    order = np.argsort(values, kind="stable")
    answer = np.empty(len(values), float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[start]] == values[order[end]]:
            end += 1
        answer[order[start:end]] = (start + end - 1) / 2.
        start = end
    return answer


def spearman(first, second):
    first, second = np.asarray(first, float), np.asarray(second, float)
    keep = np.isfinite(first) & np.isfinite(second)
    if keep.sum() < 3:
        return None
    a, b = rank(first[keep]), rank(second[keep])
    a, b = a - a.mean(), b - b.mean()
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / denominator) if denominator else None


def fixed_points(clouds, points=64, mode="center_unit_rms", deadline=None):
    """Deterministic FPS starting farthest from full-cloud centroid; tie by index."""
    clouds = np.asarray(clouds)
    if clouds.ndim != 3 or clouds.shape[2] != 3 or not np.isfinite(clouds).all():
        raise ValueError("Clouds must be finite objects x points x 3")
    if not 3 <= int(points) <= 128 or int(points) > clouds.shape[1]:
        raise ValueError("Shape points must be 3–128 within the supplied point budget")
    if mode not in ("native", "center_unit_rms"):
        raise ValueError("Unknown shape normalization")
    answer, scales = np.empty((len(clouds), int(points), 3), float), np.empty(len(clouds))
    for index, original in enumerate(clouds):
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError("Shape FPS wall-time budget reached")
        cloud = np.asarray(original, float)
        centered = cloud - cloud.mean(axis=0)
        scales[index] = np.sqrt(np.mean(np.sum(centered ** 2, axis=1)))
        if scales[index] <= 1e-12:
            raise ValueError("Degenerate cloud has no measurable shape")
        processed = centered / scales[index] if mode == "center_unit_rms" else cloud
        current = int(np.argmax(np.sum(centered ** 2, axis=1)))
        distance, visited = np.full(len(cloud), np.inf), np.zeros(len(cloud), bool)
        for slot in range(int(points)):
            answer[index, slot] = processed[current]
            visited[current] = True
            distance = np.minimum(distance, np.sum((processed - processed[current]) ** 2, axis=1))
            distance[visited] = -1.
            current = int(np.argmax(distance))
    return answer, scales


def chamfer(first, second):
    """Symmetric mean squared Euclidean nearest-point Chamfer."""
    a, b = np.asarray(first, float), np.asarray(second, float)
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != 3 or b.shape[1] != 3:
        raise ValueError("Chamfer requires two P x 3 clouds")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Nonfinite Chamfer cloud")
    distance = np.sum((a[:, None] - b[None]) ** 2, axis=2)
    return float((distance.min(axis=0).mean() + distance.min(axis=1).mean()) / 2.)


class ShapeDistances:
    def __init__(self, clouds, deadline):
        self.clouds, self.deadline, self.values = clouds, deadline, {}

    def __call__(self, a, b):
        if a == b:
            return 0.
        key = tuple(sorted((int(a), int(b))))
        if key not in self.values:
            if time.monotonic() >= self.deadline:
                raise TimeoutError("Shape Chamfer wall-time budget reached")
            self.values[key] = chamfer(self.clouds[key[0]], self.clouds[key[1]])
        return self.values[key]

    def square(self, rows):
        result = np.zeros((len(rows), len(rows)))
        for i, a in enumerate(rows):
            for j in range(i):
                result[i, j] = result[j, i] = self(a, rows[j])
        return result


def clustered_summary(values, labels, bootstrap=200, seed=22):
    """Conditional fixed-graph uncertainty: query objects/classes, never pairs."""
    values, labels = np.asarray(values, float), np.asarray(labels)
    keep = np.isfinite(values)
    values, labels = values[keep], labels[keep]
    if not len(values):
        return {"mean": None, "class_balanced_mean": None, "object_count": 0,
                "class_count": 0, "object_bootstrap_95ci": None, "class_bootstrap_95ci": None}
    classes = np.unique(labels)
    means = np.array([values[labels == label].mean() for label in classes])
    rng, object_draws, class_draws = np.random.default_rng(seed), [], []
    for _ in range(int(bootstrap)):
        object_draws.append(float(values[rng.integers(len(values), size=len(values))].mean()))
        class_draws.append(float(means[rng.integers(len(means), size=len(means))].mean()))
    ci = lambda rows: list(map(float, np.quantile(rows, [.025, .975]))) if rows else None
    return {"mean": float(values.mean()), "class_balanced_mean": float(means.mean()),
            "object_count": len(values), "class_count": len(classes),
            "object_bootstrap_95ci": ci(object_draws), "class_bootstrap_95ci": ci(class_draws),
            "bootstrap_repeats": int(bootstrap),
            "bootstrap_note": "Fixed graph query-object or class resampling, not iid neighbour pairs. Conditional descriptive intervals do not establish causality."}


def nearest(distance, ids, allowed, k):
    rows = np.flatnonzero(allowed & np.isfinite(distance))
    rows = rows[np.argsort(ids[rows], kind="stable")]
    return rows[np.argsort(distance[rows], kind="stable")[:int(k)]]


def classification(arrays, ids, labels, geometry):
    if "logits" not in arrays:
        return {"available": False, "reason": "No cached clean logits"}, None
    logits = matrix(arrays["logits"], "cached logits")
    if len(logits) != len(ids):
        raise ValueError("Logit and identity lengths differ")
    if labels.dtype.kind not in "iu" or np.any(labels < 0) or np.any(labels >= logits.shape[1]):
        return {"available": False, "reason": "Labels are not declared logits-column indices"}, None
    predicted, correct = logits.argmax(axis=1), logits.argmax(axis=1) == labels
    confusion = np.zeros((logits.shape[1], logits.shape[1]), int)
    np.add.at(confusion, (labels.astype(int), predicted), 1)
    classes = []
    for label in np.unique(labels):
        mask = labels == label
        classes.append({"label": scalar(label), "count": int(mask.sum()),
                        "clean_accuracy": float(correct[mask].mean()), "error_count": int((mask & ~correct).sum()),
                        "depth_quantiles": quantiles(geometry["depth"][mask]),
                        "correct_depth": quantiles(geometry["depth"][mask & correct]),
                        "error_depth": quantiles(geometry["depth"][mask & ~correct])})
    errors = np.flatnonzero(~correct)
    errors = errors[np.argsort(geometry["depth"][errors], kind="stable")]
    examples = np.unique(np.r_[errors[:10], errors[-10:]])
    return {"available": True, "population": "fixed clean training cache; not validation/test",
            "object_count": len(ids), "clean_accuracy": float(correct.mean()), "error_count": len(errors),
            "confusion_matrix": confusion.tolist(), "classes": classes,
            "correct_depth": quantiles(geometry["depth"][correct]),
            "error_depth": quantiles(geometry["depth"][~correct]),
            "error_examples": [{"sample_id": scalar(ids[i]), "true_label": scalar(labels[i]),
                                "predicted_label": int(predicted[i]), "depth": float(geometry["depth"][i])}
                               for i in examples]}, correct


def whole_relations(whole, c, ids, labels, queries, pool_rows, shape_matrix, distance,
                    lookup, k, bootstrap, seed, random_draws):
    result = {}
    matrices = {"hyperbolic": poincare_distance(whole[queries], whole, c),
                "direction": direction_distance(whole[queries], whole, c)}
    for method, embedding_distance in matrices.items():
        objects = []
        for qi, row in enumerate(queries):
            small_allowed = labels[queries] == labels[row]
            small_allowed[qi] = False
            shape_near = nearest(shape_matrix[qi], ids[queries], small_allowed, k)
            embed_near = nearest(embedding_distance[qi, queries], ids[queries], small_allowed, k)
            overlap = len(set(shape_near) & set(embed_near)) / min(len(shape_near), len(embed_near)) if len(shape_near) and len(embed_near) else None
            valid_small = np.flatnonzero(small_allowed & np.isfinite(embedding_distance[qi, queries]))
            denominator = min(len(shape_near), len(embed_near))
            expected = (len(set(shape_near) & set(valid_small)) * len(embed_near) / (len(valid_small) * denominator)) if len(valid_small) and denominator else None
            adjusted = (overlap - expected) / (1. - expected) if overlap is not None and expected is not None and expected < 1. else None
            rho = spearman(shape_matrix[qi, small_allowed], embedding_distance[qi, queries[small_allowed]])
            full_allowed = labels == labels[row]
            full_allowed[row] = False
            full_near = nearest(embedding_distance[qi], ids, full_allowed, k)
            eligible = np.flatnonzero(full_allowed)
            rng = np.random.default_rng(int(seed) + 100003 * qi)
            random_rows = rng.choice(eligible, size=min(random_draws, len(eligible)), replace=False) if len(eligible) else []
            observed = [distance(pool_rows[qi], lookup[ids[j]]) for j in full_near]
            control = [distance(pool_rows[qi], lookup[ids[j]]) for j in random_rows]
            observed_mean, control_mean = float(np.mean(observed)) if observed else None, float(np.mean(control)) if control else None
            gain = 1. - observed_mean / control_mean if observed_mean is not None and control_mean and control_mean > 0 else None
            objects.append({"sample_id": scalar(ids[row]), "label": scalar(labels[row]),
                            "audit_class_size": int(small_allowed.sum() + 1), "within_class_spearman": rho,
                            "shape_topk_overlap": overlap, "shape_topk_overlap_expected_random": expected, "chance_adjusted_overlap": adjusted, "full_pool_topk_chamfer": observed_mean,
                            "same_class_random_chamfer": control_mean, "chamfer_enrichment": gain,
                            "full_pool_topk_ids": ids[full_near].tolist(),
                            "shape_panel_topk_ids": ids[queries[shape_near]].tolist()})
        fields = {"within_class_spearman": "within_class_spearman", "shape_topk_overlap": "shape_topk_overlap",
                  "full_pool_chamfer_enrichment": "chamfer_enrichment", "shape_topk_overlap_expected_random": "shape_topk_overlap_expected_random", "chance_adjusted_overlap": "chance_adjusted_overlap"}
        qlabels = labels[queries]
        result[method] = {field: clustered_summary([np.nan if row[key] is None else row[key] for row in objects], qlabels, bootstrap, seed)
                          for field, key in fields.items()}
        result[method]["objects"] = objects
        result[method]["classes"] = [{"label": scalar(label),
            **{field: clustered_summary([np.nan if row[key] is None else row[key] for row in objects if row["label"] == scalar(label)],
                                         qlabels[qlabels == label], 0, seed) for field, key in fields.items()}}
            for label in np.unique(qlabels)]
    return result


def proxy_relations(snapshot, ids, labels, lookup, distance, maximum, seed):
    retrieval = (snapshot or {}).get("retrieval", {})
    methods = {name: retrieval.get(name, {}).get("topk_sample_ids", []) for name in ("hyperbolic", "direction")}
    total = max(map(len, methods.values()))
    if not total:
        return {"available": False, "reason": "No explicit mapped proxy top-k identities"}
    selected = np.unique(np.linspace(0, total - 1, min(maximum, total), dtype=int))
    label_by_id = dict(zip(ids.tolist(), labels.tolist()))
    candidates = {label: ids[labels == label].tolist() for label in np.unique(labels).tolist()}
    proxy_ids, result = retrieval.get("proxy_ids", []), {}
    for method, rows in methods.items():
        values = []
        for row in selected:
            neighbours = rows[row][:4] if row < len(rows) else []
            if len(neighbours) < 2 or any(value not in lookup or value not in label_by_id for value in neighbours):
                continue
            observed = [distance(lookup[a], lookup[b]) for i, a in enumerate(neighbours) for b in neighbours[:i]]
            rng, reference = np.random.default_rng(seed + 7879 * int(row)), []
            for _ in range(5):
                sampled, used = [], set()
                for value in neighbours:
                    eligible = [candidate for candidate in candidates[label_by_id[value]] if candidate not in used]
                    if not eligible:
                        sampled = []
                        break
                    chosen = eligible[int(rng.integers(len(eligible)))]
                    sampled.append(chosen)
                    used.add(chosen)
                reference.extend(distance(lookup[a], lookup[b]) for i, a in enumerate(sampled) for b in sampled[:i])
            control = float(np.mean(reference)) if reference else None
            measured = float(np.mean(observed))
            values.append({"proxy_row": int(row), "proxy_id": scalar(proxy_ids[row]) if row < len(proxy_ids) else None,
                           "topk_sample_ids": neighbours, "topk_pair_chamfer": measured,
                           "class_composition_matched_random_chamfer": control,
                           "enrichment": 1. - measured / control if control and control > 0 else None,
                           "labels": [label_by_id[value] for value in neighbours]})
        finite = [row["enrichment"] for row in values if row["enrichment"] is not None]
        result[method] = {"proxy_count": len(values), "mean_chamfer_enrichment": float(np.mean(finite)) if finite else None,
                          "confidence_interval": None, "proxy_rows": values}
    return {"available": True, "total_proxy_rows": total, "audited_proxy_rows": selected.tolist(), "methods": result,
            "sampling": "Evenly spaced stable rows; shared objects across proxies. No proxy-iid confidence interval."}


def analyze_structure(specs, pointcloud_pool, output_dir, *, snapshots=None,
                      shape_points=64, per_class=8, normalization="both", k=4,
                      bootstrap=200, seed=22, max_seconds=900, max_objects=400,
                      max_proxies=64, random_draws=20):
    """Fresh report from explicit snapshot specs and {sample_ids, labels, clouds} NPZ."""
    start = time.monotonic()
    if not specs or not 1 <= per_class <= 32 or not 1 <= max_objects <= 512:
        raise ValueError("Missing snapshots or invalid bounded object budget")
    if not 1 <= k <= 16 or not 0 <= bootstrap <= 2000 or not 1 <= max_proxies <= 128:
        raise ValueError("Invalid topk/bootstrap/proxy budget")
    if not 1 <= random_draws <= 100 or not 1 <= max_seconds <= 14400:
        raise ValueError("Invalid random-control or wall-time budget")
    modes = ("native", "center_unit_rms") if normalization == "both" else (normalization,)
    if any(mode not in ("native", "center_unit_rms") for mode in modes):
        raise ValueError("Unknown normalization mode")
    source, out = Path(pointcloud_pool).resolve(), Path(output_dir).resolve()
    if out.exists():
        raise FileExistsError("Shape output must be a new directory")
    paths = [source] + [Path(spec["cache"]).resolve() for spec in specs]
    if any(path.parent == out or path.parent in out.parents for path in paths):
        raise ValueError("Shape output must be outside source cache directories")
    hashes = {path: file_sha256(path) for path in paths}
    pool = load_npz(source)
    ids, labels, clouds = unique_ids(pool["sample_ids"]), np.asarray(pool["labels"]).reshape(-1), pool["clouds"]
    if len(ids) != len(labels) or len(ids) != len(clouds):
        raise ValueError("Cloud pool array lengths differ")
    pool_input_sha = hashlib.sha256(np.asarray(clouds).tobytes()).hexdigest()
    lookup, loaded = {value: i for i, value in enumerate(ids.tolist())}, []
    for spec in specs:
        cache = load_npz(spec["cache"])
        cids, clabels = unique_ids(cache["sample_ids"]), np.asarray(cache["labels"]).reshape(-1)
        whole = matrix(cache.get("mu", cache.get("whole_mu")))
        if len(cids) != len(clabels) or len(cids) != len(whole):
            raise ValueError("Snapshot array lengths differ")
        if (cids.dtype.kind in "iu") != (ids.dtype.kind in "iu"):
            raise ValueError("Cache/cloud identity types differ")
        if set(cids.tolist()) - set(lookup):
            raise ValueError("Pointcloud pool is missing cached object IDs")
        if not np.array_equal(labels[[lookup[value] for value in cids.tolist()]], clabels):
            raise ValueError("Cloud/cache labels disagree")
        c = spec.get("c", cache.get("c", cache.get("curvature")))
        if c is None:
            raise ValueError("Missing explicit curvature")
        ball_geometry(whole, float(c))
        declared_sha = spec.get("input_sha256")
        if declared_sha is None and "input_sha256" in cache:
            declared_sha = scalar(np.asarray(cache["input_sha256"]).reshape(()))
        if declared_sha is not None:
            input_rows = np.array([lookup[value] for value in cids.tolist()])
            expected_sha = hashlib.sha256(np.asarray(clouds[input_rows]).tobytes()).hexdigest()
            if str(declared_sha) != expected_sha:
                raise ValueError("Point-cloud bytes do not match cached input SHA256")
        loaded.append((dict(spec), cache, cids, clabels, whole, float(c)))
    common = set(loaded[0][2].tolist())
    for row in loaded[1:]:
        common &= set(row[2].tolist())
    if not common:
        raise ValueError("Snapshots have no common training objects")
    reference_ids, reference_labels = loaded[0][2:4]
    rng, chosen = np.random.default_rng(seed), []
    for label in np.unique(reference_labels):
        group = np.sort(reference_ids[(reference_labels == label) & np.isin(reference_ids, list(common))])
        if len(group):
            chosen.extend(rng.choice(group, size=min(per_class, len(group)), replace=False).tolist())
    chosen = np.asarray(sorted(chosen), dtype=reference_ids.dtype)
    if len(chosen) > max_objects:
        raise ValueError("Stratified queries exceed max_objects; lower per_class or raise explicit budget")
    pool_rows = np.array([lookup[value] for value in chosen.tolist()])
    out.mkdir(parents=True)
    result = {"format": "hier-independent-shape-audit-v1", "status": "running",
              "source": {"pointcloud_file": source.name, "pointcloud_sha256": hashes[source],
                         "pool_ids_sha256": pool_hash(ids), "input_sha256": pool_input_sha, "supplied_cloud_count": len(ids)},
              "config": {"shape_points": shape_points, "per_class": per_class, "k": k, "normalization": list(modes),
                         "bootstrap": bootstrap, "seed": seed, "max_seconds": max_seconds,
                         "max_objects": max_objects, "max_proxies": max_proxies, "random_draws": random_draws},
              "audit_ids": chosen.tolist(), "audit_ids_sha256": pool_hash(chosen), "audit_object_count": len(chosen),
              "signal": {"definition": "Symmetric mean squared Euclidean nearest-point Chamfer on deterministic FPS; independent of network/proxies/teacher/labels",
                         "native": "Dataset coordinates and physical scales preserved; no translation/rotation alignment",
                         "center_unit_rms": "Full-cloud centroid and RMS radius normalization; saved rotations retained",
                         "limitations": ["Geometric proxy, not semantic hierarchy truth", "No rotation/topology invariance",
                                         "Bounded query/point subsampling", "Neighbour pairs are not iid",
                                         "Historical multi-factor comparisons are not causal"]},
              "snapshots": [], "warnings": []}
    candidate_hashes = {pool_hash(row[2]) for row in loaded}
    result["candidate_pools_match"] = len(candidate_hashes) == 1
    if not result["candidate_pools_match"]:
        result["warnings"].append("Candidate pools differ: shared shape queries are paired, but full-pool retrieval differences also include candidate-pool changes.")
    if any(not row[0].get("input_sha256") and "input_sha256" not in row[1] for row in loaded):
        result["warnings"].append("Some snapshots lack input SHA256; IDs/labels agree but cache input-byte equivalence cannot be verified.")
    def save():
        (out / "structure_summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    save()
    try:
        deadline, processors = start + max_seconds, {}
        for mode in modes:
            sampled, scales = fixed_points(clouds, shape_points, mode, deadline)
            distances = ShapeDistances(sampled, deadline)
            shape_matrix = distances.square(pool_rows)
            processors[mode] = (distances, shape_matrix)
            np.savez_compressed(out / ("shape_" + mode + ".npz"), sample_ids=chosen, labels=labels[pool_rows],
                                symmetric_chamfer=shape_matrix, sampled_clouds=sampled[pool_rows], original_rms_radius=scales[pool_rows])
        summaries = (snapshots or {}).get("snapshots", [])
        for spec, cache, cids, clabels, whole, c in loaded:
            if time.monotonic() >= deadline:
                raise TimeoutError("Shape snapshot wall-time budget reached")
            by_id = {value: i for i, value in enumerate(cids.tolist())}
            queries = np.array([by_id[value] for value in chosen.tolist()])
            geometry = ball_geometry(whole, c)
            failures, correct = classification(cache, cids, clabels, geometry)
            row = {"run_key": spec.get("run_key"), "display_name": spec.get("display_name", spec.get("label")),
                   "version": spec.get("version"), "storage_run_id": spec.get("storage_run_id"),
                   "epoch": int(spec["epoch"]), "candidate_count": len(cids), "c": c,
                   "cache_file": Path(spec["cache"]).name, "cache_sha256": hashes[Path(spec["cache"]).resolve()],
                   "sample_pool_sha256": pool_hash(cids), "input_sha256": spec.get("input_sha256"),
                   "classification": failures, "shape": {},
                   "class_radial": [{"label": scalar(label), "count": int((clabels == label).sum()),
                                     "depth": quantiles(geometry["depth"][clabels == label]),
                                     "normalized_radius": quantiles(geometry["normalized_radius"][clabels == label])}
                                    for label in np.unique(clabels)],
                   "training_usage_by_object": {"available": False, "reason": "No explicit training query-to-ancestor object IDs; nearest sets are not ancestors"}}
            snapshot = next((item for item in summaries if item.get("run_key") == spec.get("run_key") and item.get("epoch") == int(spec["epoch"])), None)
            for mode, (distances, shape_matrix) in processors.items():
                relations = whole_relations(whole, c, cids, clabels, queries, pool_rows, shape_matrix, distances,
                                            lookup, k, bootstrap, seed, random_draws)
                row["shape"][mode] = {"whole_relations": relations,
                                     "proxy_relations": proxy_relations(snapshot, cids, clabels, lookup, distances, max_proxies, seed)}
                if correct is not None:
                    for method in relations.values():
                        for name, mask in (("correct", correct[queries]), ("error", ~correct[queries])):
                            gains = [item["chamfer_enrichment"] for item, yes in zip(method["objects"], mask) if yes and item["chamfer_enrichment"] is not None]
                            method[name + "_objects"] = {"count": int(mask.sum()), "mean_chamfer_enrichment": float(np.mean(gains)) if gains else None}
            result["snapshots"].append(row)
            save()
        result["shape_pair_evaluations"] = {mode: len(distances.values) for mode, (distances, _) in processors.items()}
        result["status"] = "completed"
    except TimeoutError as exc:
        result["status"] = "partial_budget"
        result["warnings"].append(str(exc))
    except Exception as exc:
        result["status"], result["error"] = "failed", type(exc).__name__ + ": " + str(exc)
        save()
        raise
    finally:
        result["elapsed_seconds"] = time.monotonic() - start
        result["sources_unchanged"] = all(file_sha256(path) == expected for path, expected in hashes.items())
        if not result["sources_unchanged"]:
            result["status"], result["error"] = "failed", "A source cache/cloud changed during audit"
        save()
    if not result["sources_unchanged"]:
        raise RuntimeError(result["error"])
    return {**result, "arrays_base_dir": str(out)}


def save_structure_figures(structure, output_dir):
    from .structure_figures import save_structure_figures as implementation
    return implementation(structure, output_dir)
