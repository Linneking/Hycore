"""Frozen V7 proxy-to-whole hotspots and same-object HyCoRe baseline audit.

NumPy CPU only; no model import, forward, optimizer, test split or source writes.
All per-object / proxy outputs contain private IDs and belong outside Git.
Canonical object identity is sorted HDF5 shard name + row. Baseline glob-order
IDs MUST be explicitly remapped; label agreement alone is not an ID mapping.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import datetime as dt
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np


NAMES = "airplane bathtub bed bench bookshelf bottle bowl car chair cone cup curtain desk door dresser flower_pot glass_box guitar keyboard lamp laptop mantel monitor night_stand person piano plant radio range_hood sink sofa stairs stool table tent toilet tv_stand vase wardrobe xbox".split()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def stats(values):
    a = np.asarray(values, dtype=float)
    if not a.size:
        return {"n": 0}
    if not np.isfinite(a).all():
        raise ValueError("Nonfinite diagnostic value")
    q = np.quantile(a, [.05, .25, .5, .75, .95])
    return dict(n=int(a.size), min=float(a.min()), p05=float(q[0]),
                q25=float(q[1]), median=float(q[2]), q75=float(q[3]),
                p95=float(q[4]), max=float(a.max()), mean=float(a.mean()),
                std_population=float(a.std()), variance_population=float(a.var()),
                iqr=float(q[3] - q[1]))


def scalar(z, key, fallback=None):
    if key not in z.files:
        if fallback is None:
            raise KeyError(key)
        return fallback
    a = np.asarray(z[key])
    if a.size != 1:
        raise ValueError("Expected scalar " + key)
    return a.reshape(()).item()


def ball_geometry(points, c):
    x = np.asarray(points, dtype=np.float64)
    if x.ndim != 2 or not np.isfinite(x).all() or not (c > 0):
        raise ValueError("Expected finite 2D vectors and positive curvature")
    r = np.linalg.norm(x, axis=1)
    q = np.sqrt(c) * r
    if np.any(q >= 1) or np.any(r <= 0):
        raise ValueError("Direction analysis requires nonzero open-ball vectors")
    return x, r, x / r[:, None], 2 * np.arctanh(q) / np.sqrt(c)


def map_proxy(tangent, c):
    u = np.asarray(tangent, dtype=np.float64)
    norm = np.linalg.norm(u, axis=1)
    if not np.isfinite(u).all() or np.any(norm <= 0):
        raise ValueError("Nonzero finite saved proxy parameters required")
    # Reproduce the actual V7 forward numeric projection, without applying any
    # post-Adam parameter constraint to the saved parameters during analysis.
    radius = np.minimum(np.tanh(np.sqrt(c) * norm), .999) / np.sqrt(c)
    return u / norm[:, None] * radius[:, None]


def remap_ids(ids, shards):
    by_name = {str(s["name"]): int(s["count"]) for s in shards}
    if len(by_name) != len(shards) or any(v <= 0 for v in by_name.values()):
        raise ValueError("Invalid or repeated shard identity")
    offsets, offset = {}, 0
    for name in sorted(by_name):
        offsets[name] = offset
        offset += by_name[name]
    mapping = np.concatenate([offsets[str(s["name"])] + np.arange(int(s["count"])) for s in shards])
    ids = np.asarray(ids, dtype=np.int64)
    if np.any(ids < 0) or np.any(ids >= len(mapping)):
        raise ValueError("Glob-order ID outside declared shard population")
    return mapping[ids]


def ordinal_ranks(values, ids):
    """One-based increasing-depth rank, ties broken by canonical object ID."""
    order = np.lexsort((ids, values))
    ranks = np.empty(len(values), dtype=np.int64)
    ranks[order] = np.arange(1, len(values) + 1)
    return ranks


def class_metrics(ids, labels, depth, direction, correct):
    center_angle = np.empty(len(ids), dtype=float)
    class_rank = np.empty(len(ids), dtype=np.int64)
    rows = []
    for label, name in enumerate(NAMES):
        at = np.flatnonzero(labels == label)
        if not len(at):
            rows.append(dict(class_label=label, class_name=name, n=0))
            continue
        mean_direction = direction[at].mean(0)
        resultant = float(np.linalg.norm(mean_direction))
        if resultant <= 1e-12:
            raise ValueError("Class mean direction undefined")
        angle = np.degrees(np.arccos(np.clip(direction[at] @ (mean_direction / resultant), -1, 1)))
        center_angle[at] = angle
        class_rank[at] = ordinal_ranks(depth[at], ids[at])
        rows.append(dict(class_label=label, class_name=name, n=int(len(at)),
                         depth=stats(depth[at]), mean_direction_resultant=resultant,
                         angle_to_class_center_deg=stats(angle),
                         clean_train_accuracy_percent=None if correct is None else float(100 * correct[at].mean())))
    return rows, center_angle, class_rank


def load_model(item, kind, common_ids=None, reference_labels=None):
    path = Path(item["cache"])
    source = dict(cache_file=path.name, cache_sha256=sha(path), kind=kind,
                  epoch=int(item["epoch"]), name=item.get("name", "V7_e" + str(item["epoch"])))
    for key in ["checkpoint_sha256", "checkpoint_file", "source_commit", "run_key", "input_mode"]:
        if key in item:
            source[key] = item[key]
    summary = None
    if item.get("summary"):
        summary_path = Path(item["summary"])
        source.update(summary_file=summary_path.name, summary_sha256=sha(summary_path))
        summary = json.loads(summary_path.read_text(encoding="utf-8-sig"))
    with np.load(path, allow_pickle=False) as z:
        if kind == "V7":
            ids = np.asarray(z["sample_ids"], dtype=np.int64).reshape(-1)
            mu = z["mu"].copy()
            c = float(scalar(z, "c", scalar(z, "curvature", item.get("c", 1.))))
            source["input_sha256"] = str(scalar(z, "input_sha256"))
            pids = np.asarray(z["proxy_ids"], dtype=np.int64).reshape(-1)
            proxy_order = np.argsort(pids, kind="stable")
            pids = pids[proxy_order]
            if len(np.unique(pids)) != len(pids):
                raise ValueError("Duplicate proxy parameter ID")
            proxy = map_proxy(z["proxy_tangent"][proxy_order], c)
            source["identity_mapping"] = "already canonical sample_ids; V7 sorted HDF5 shard + row"
        else:
            ids = np.asarray(z["ids"], dtype=np.int64).reshape(-1)
            mu = z["whole_mu"].copy()
            c = float(item.get("c", scalar(z, "c", 1.)))
            source["curvature_source"] = "spec/cache c, otherwise original HyCoRe fixed c=1"
            if item.get("canonical_ids") is True:
                source["identity_mapping"] = "explicit spec declaration: IDs already canonical"
            elif summary is not None and "source_shards_in_glob_order" in summary.get("data_identity", {}):
                shards = summary["data_identity"]["source_shards_in_glob_order"]
                ids = remap_ids(ids, shards)
                source["canonical_shards"] = sorted(str(s["name"]) for s in shards)
                source["identity_mapping"] = "glob-order IDs remapped by sorted HDF5 shard name + row"
            else:
                raise ValueError("Baseline ID mapping absent: require summary shards or explicit canonical_ids=true")
            if summary is not None:
                source["checkpoint_sha256"] = summary.get("checkpoint_sha256")
        labels = np.asarray(z["labels"], dtype=np.int64).reshape(-1)
        logits = None
        if "logits" in z.files:
            logits = np.asarray(z["logits"], dtype=float)
            predictions = np.argmax(logits, axis=1)
            prediction_source = "cached logits argmax"
        elif "whole_logits" in z.files:
            logits = np.asarray(z["whole_logits"], dtype=float)
            predictions = np.argmax(logits, axis=1)
            prediction_source = "cached whole_logits argmax"
        elif "whole_prediction" in z.files:
            predictions = np.asarray(z["whole_prediction"], dtype=np.int64).reshape(-1)
            prediction_source = "cached whole_prediction"
        else:
            predictions, prediction_source = None, "missing; no new forward"
        ce = np.asarray(z["whole_ce"], dtype=float).reshape(-1) if "whole_ce" in z.files else None
        true_softmax, max_softmax, true_logit_margin = None, None, None
        if logits is not None:
            if logits.shape != (len(labels), 40) or not np.isfinite(logits).all():
                raise ValueError("Invalid cached classification outputs")
            ex = np.exp(logits - logits.max(1, keepdims=True))
            probs = ex / ex.sum(1, keepdims=True)
            true_softmax = probs[np.arange(len(labels)), labels]
            max_softmax = probs.max(1)
            competitor = logits.copy()
            competitor[np.arange(len(labels)), labels] = -np.inf
            true_logit_margin = logits[np.arange(len(labels)), labels] - competitor.max(1)
    if len(ids) != len(labels) or len(ids) != len(mu) or len(np.unique(ids)) != len(ids):
        raise ValueError("Invalid candidate dimensions / duplicate canonical IDs")
    if np.any((labels < 0) | (labels >= len(NAMES))):
        raise ValueError("ModelNet40 class index outside [0,40)")
    order = np.argsort(ids, kind="stable")
    full_ids, full_labels, full_mu = ids[order], labels[order], mu[order]
    full_x, full_radius, full_direction, full_depth = ball_geometry(full_mu, c)
    full_rank = ordinal_ranks(full_depth, full_ids)
    full_class_rank = np.empty(len(full_ids), dtype=np.int64)
    full_class_n = np.bincount(full_labels, minlength=40)
    for label in range(40):
        at = np.flatnonzero(full_labels == label)
        full_class_rank[at] = ordinal_ranks(full_depth[at], full_ids[at])
    if common_ids is None:
        common_ids = full_ids.copy()
    positions = np.searchsorted(full_ids, common_ids)
    if np.any(positions >= len(full_ids)) or not np.array_equal(full_ids[positions], common_ids):
        raise ValueError("Baseline / snapshot does not contain every common canonical object")
    labels = full_labels[positions]
    if reference_labels is not None:
        np.testing.assert_array_equal(labels, reference_labels, err_msg="All common labels must match after explicit ID mapping")
    correct = None if predictions is None else predictions[order][positions] == labels
    source["prediction_source"] = prediction_source
    source["full_pool_count"] = len(full_ids)
    source["curvature"] = c
    rows, center_angle, class_rank = class_metrics(common_ids, labels, full_depth[positions], full_direction[positions], correct)
    result = dict(name=source["name"], kind=kind, epoch=source["epoch"], source=source, path=path,
                  ids=common_ids, labels=labels, x=full_x[positions], r=full_radius[positions],
                  u=full_direction[positions], depth=full_depth[positions], c=c, correct=correct,
                  cached_smoothed_ce=None if ce is None else ce[order][positions],
                  predictions=None if predictions is None else predictions[order][positions],
                  true_class_softmax=None if true_softmax is None else true_softmax[order][positions],
                  max_softmax=None if max_softmax is None else max_softmax[order][positions],
                  true_class_output_margin=None if true_logit_margin is None else true_logit_margin[order][positions],
                  rank=ordinal_ranks(full_depth[positions], common_ids), full_rank=full_rank[positions],
                  full_class_rank=full_class_rank[positions], full_class_n=full_class_n,
                  class_rank=class_rank, center_angle=center_angle, classes=rows,
                  common_class_n=np.bincount(labels, minlength=40),
                  full_depth_stats=stats(full_depth), missing=[])
    if correct is None:
        result["missing"].append("cached classification predictions / logits")
    if kind == "V7":
        p, pr, pu, pd = ball_geometry(proxy, c)
        result.update(proxy_ids=pids, p=p, pr=pr, pu=pu, proxy_depth=pd)
    return result


def retrieve(model, topks):
    p, x, c = model["p"], model["x"], model["c"]
    kmax = max(topks)
    raw, direction = [], []
    wn = np.sum(x * x, axis=1)
    for start in range(0, len(p), 64):
        query = p[start:start + 64]
        sq = np.sum(query * query, axis=1)[:, None] + wn[None, :] - 2 * query @ x.T
        if np.min(sq) < -1e-11:
            raise ValueError("Squared distance cancellation outside tolerance")
        # Positive proxy-only factor omitted; ranking equals geodesic distance.
        score = np.maximum(sq, 0) / (1 - c * wn[None, :])
        raw.append(np.argsort(score, axis=1, kind="stable")[:, :kmax])
        direction.append(np.argsort(-(model["pu"][start:start + 64] @ model["u"].T), axis=1, kind="stable")[:, :kmax])
    raw, direction = np.concatenate(raw), np.concatenate(direction)
    for i in range(min(4, len(p))):
        squared = np.sum((x - p[i]) ** 2, axis=1)
        denominator = (1 - c * np.sum(p[i] ** 2)) * (1 - c * wn)
        distance = 2 * np.arcsinh(np.sqrt(c * squared / denominator)) / np.sqrt(c)
        np.testing.assert_array_equal(np.argsort(distance, kind="stable")[:kmax], raw[i])
    cases = {}
    for mode, rows in [("raw", raw), ("direction", direction)]:
        for k in topks:
            nn = rows[:, :k]
            counts = np.bincount(nn.reshape(-1), minlength=len(x))
            groups = Counter(tuple(sorted(v)) for v in model["ids"][nn].tolist())
            class_slots = np.bincount(model["labels"][nn].reshape(-1), minlength=40)
            covered = counts > 0
            class_covered = np.bincount(model["labels"][covered], minlength=40)
            class_pool = np.bincount(model["labels"], minlength=40)
            low = model["rank"] <= int(np.ceil(.01 * len(x)))
            hottest = np.lexsort((model["ids"], -counts))
            hottest = hottest[counts[hottest] > 0][:20]
            q = counts[covered] / nn.size
            cases[mode + "_top" + str(k)] = dict(proxy_count=len(p), topk=k, slot_count=int(nn.size),
                covered_object_count=int(covered.sum()), unique_neighbor_sets=len(groups),
                max_repeated_neighbor_set=int(max(groups.values())),
                top4_object_slot_share=float(np.sort(counts)[-4:].sum() / nn.size),
                low_depth_one_percent_slot_share=float(counts[low].sum() / nn.size),
                entropy_effective_covered_objects=float(np.exp(-np.sum(q * np.log(q)))),
                classes=[dict(class_label=i, class_name=NAMES[i], pool_count=int(class_pool[i]),
                              slots=int(class_slots[i]), slot_share=float(class_slots[i] / nn.size),
                              slot_enrichment_vs_pool=float((class_slots[i] / nn.size) / (class_pool[i] / len(x))) if class_pool[i] else None,
                              covered_objects=int(class_covered[i])) for i in range(40)],
                hottest=[dict(canonical_id=int(model["ids"][i]), class_label=int(model["labels"][i]),
                              class_name=NAMES[int(model["labels"][i])], slots=int(counts[i]),
                              depth=float(model["depth"][i]), common_pool_depth_rank=int(model["rank"][i]),
                              common_pool_depth_percentile=float(model["rank"][i] / len(x)),
                              within_class_depth_rank=int(model["class_rank"][i]),
                              angle_to_own_class_center_deg=float(model["center_angle"][i]),
                              cached_prediction=None if model["predictions"] is None else int(model["predictions"][i]),
                              true_class_softmax=None if model["true_class_softmax"] is None else float(model["true_class_softmax"][i]),
                              true_class_output_margin=None if model["true_class_output_margin"] is None else float(model["true_class_output_margin"][i]),
                              cached_correct=None if model["correct"] is None else bool(model["correct"][i])) for i in hottest])
    model.update(raw_rows=raw, direction_rows=direction,
                 raw_counts4=np.bincount(raw[:, :4].reshape(-1), minlength=len(x)),
                 direction_counts4=np.bincount(direction[:, :4].reshape(-1), minlength=len(x)))
    return cases


def write_csv(path, fields, rows):
    with Path(path).open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def object_row(model, i):
    label = int(model["labels"][i])
    class_n = int(model["common_class_n"][label])
    return dict(model=model["name"], kind=model["kind"], epoch=model["epoch"],
                canonical_id=int(model["ids"][i]), class_label=label, class_name=NAMES[label],
                depth=float(model["depth"][i]), euclidean_radius=float(model["r"][i]),
                common_pool_n=len(model["ids"]), common_depth_rank=int(model["rank"][i]),
                common_depth_percentile=float(model["rank"][i] / len(model["ids"])),
                full_pool_n=model["source"]["full_pool_count"], full_depth_rank=int(model["full_rank"][i]),
                full_depth_percentile=float(model["full_rank"][i] / model["source"]["full_pool_count"]),
                class_common_n=class_n, class_common_depth_rank=int(model["class_rank"][i]),
                class_common_depth_percentile=float(model["class_rank"][i] / class_n),
                class_full_n=int(model["full_class_n"][label]), class_full_depth_rank=int(model["full_class_rank"][i]),
                angle_to_own_class_center_deg=float(model["center_angle"][i]),
                cached_correct=None if model["correct"] is None else bool(model["correct"][i]),
                cached_prediction=None if model["predictions"] is None else int(model["predictions"][i]),
                true_class_softmax=None if model["true_class_softmax"] is None else float(model["true_class_softmax"][i]),
                max_softmax=None if model["max_softmax"] is None else float(model["max_softmax"][i]),
                true_class_output_margin=None if model["true_class_output_margin"] is None else float(model["true_class_output_margin"][i]),
                cached_label_smoothed_ce=None if model["cached_smoothed_ce"] is None else float(model["cached_smoothed_ce"][i]),
                raw_top4_slots=None if model["kind"] != "V7" else int(model["raw_counts4"][i]),
                direction_top4_slots=None if model["kind"] != "V7" else int(model["direction_counts4"][i]))


def main(spec_path, output_path):
    started = time.monotonic()
    spec_path, output = Path(spec_path), Path(output_path)
    spec = json.loads(spec_path.read_text(encoding="utf-8-sig"))
    snapshots = sorted(spec["snapshots"], key=lambda v: int(v["epoch"]))
    if not snapshots or len({int(v["epoch"]) for v in snapshots}) != len(snapshots):
        raise ValueError("Distinct V7 snapshot epochs required")
    topks = sorted(set([4] + list(map(int, spec.get("topks", [4, 16])))))
    if min(topks) <= 0:
        raise ValueError("Positive topk required")
    output.mkdir(parents=True, exist_ok=False)
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[3], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    manifest = dict(status="running", started_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                    spec_sha256=sha(spec_path), script_sha256=sha(__file__), code_commit=commit,
                    numpy=np.__version__, GPU=None, encoder_forwards=0, optimizer_updates=0, test_forwards=0)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    ref = load_model(snapshots[0], "V7")
    ids, labels, input_sha = ref["ids"], ref["labels"], ref["source"]["input_sha256"]
    if max(topks) > len(ids):
        raise ValueError("Topk exceeds common candidate pool")
    models = [ref] + [load_model(item, "V7", ids, labels) for item in snapshots[1:]]
    for m in models:
        if m["source"]["input_sha256"] != input_sha:
            raise ValueError("V7 snapshots use different frozen input identities")
        if not np.array_equal(m["ids"], ids) or not np.array_equal(m["proxy_ids"], ref["proxy_ids"]):
            raise ValueError("V7 object / proxy identity differs across snapshots")
    models += [load_model(item, "baseline", ids, labels) for item in spec.get("baselines", [])]
    if len({m["name"] for m in models}) != len(models):
        raise ValueError("Unique model display names required")
    results = []
    class_rows, retrieval_class_rows, proxy_rows = [], [], []
    for m in models:
        r = dict(name=m["name"], kind=m["kind"], epoch=m["epoch"], source=m["source"],
                 common_pool_count=len(ids), whole_depth=stats(m["depth"]), full_pool_whole_depth=m["full_depth_stats"],
                 clean_train_accuracy_percent=None if m["correct"] is None else float(100 * m["correct"].mean()),
                 classes=m["classes"], missing=m["missing"])
        for row in m["classes"]:
            if row["n"]:
                class_rows.append(dict(model=m["name"], kind=m["kind"], epoch=m["epoch"],
                    class_label=row["class_label"], class_name=row["class_name"], n=row["n"],
                    mean_direction_resultant=row["mean_direction_resultant"],
                    clean_train_accuracy_percent=row["clean_train_accuracy_percent"],
                    **{"depth_" + key: value for key, value in row["depth"].items() if key != "n"},
                    **{"center_angle_deg_" + key: value for key, value in row["angle_to_class_center_deg"].items() if key != "n"}))
        if m["kind"] == "V7":
            r["proxy_depth"] = stats(m["proxy_depth"])
            r["retrieval"] = retrieve(m, topks)
            for case, value in r["retrieval"].items():
                for row in value["classes"]:
                    retrieval_class_rows.append(dict(model=m["name"], epoch=m["epoch"], retrieval=case, **row))
            for i, pid in enumerate(m["proxy_ids"]):
                proxy_rows.append(dict(model=m["name"], epoch=m["epoch"], proxy_id=int(pid),
                    proxy_origin_depth=float(m["proxy_depth"][i]),
                    **{mode + "_top" + str(k) + "_ids": json.dumps(m["ids"][m[mode + "_rows"][i, :k]].tolist())
                       for mode in ["raw", "direction"] for k in topks}))
        flower = m["classes"][15]
        others = [row for row in m["classes"] if row["n"] and row["class_label"] != 15]
        r["flower_pot_vs_other_classes_descriptive"] = dict(
            flower_pot_depth_median=flower.get("depth", {}).get("median"),
            other_classes_macro_mean_depth_median=float(np.mean([v["depth"]["median"] for v in others])),
            flower_pot_direction_resultant=flower.get("mean_direction_resultant"),
            other_classes_macro_mean_direction_resultant=float(np.mean([v["mean_direction_resultant"] for v in others])),
            flower_pot_mean_center_angle_deg=flower.get("angle_to_class_center_deg", {}).get("mean"),
            other_classes_macro_mean_center_angle_deg=float(np.mean([v["angle_to_class_center_deg"]["mean"] for v in others])),
            flower_pot_cached_train_accuracy_percent=flower.get("clean_train_accuracy_percent"),
            other_classes_macro_mean_cached_train_accuracy_percent=None if m["correct"] is None else float(np.mean([v["clean_train_accuracy_percent"] for v in others])))
        results.append(r)
        print(m["name"], "n", len(ids), "whole median", round(r["whole_depth"]["median"], 6),
              "raw/direction top4 coverage", None if m["kind"] != "V7" else
              [r["retrieval"][v + "_top4"]["covered_object_count"] for v in ["raw", "direction"]], flush=True)
    final = [m for m in models if m["kind"] == "V7"][-1]
    tracking = []
    tracked_rows = set()
    for mode in ["raw", "direction"]:
        counts = final[mode + "_counts4"]
        order = np.lexsort((ids, -counts))
        order = order[counts[order] > 0][:20]
        tracked_rows.update(order.tolist())
        for rank, i in enumerate(order, 1):
            for m in models:
                tracking.append(dict(final_hotspot_definition=mode + "_top4_slots", final_hotspot_rank=rank,
                                     final_hotspot_slots=int(counts[i]), **object_row(m, i)))
    obj_fields = list(object_row(models[0], 0))
    write_csv(output / "object_metrics.csv", obj_fields,
              (object_row(m, i) for m in models for i in range(len(ids))))
    write_csv(output / "hotspot_trajectory.csv", ["final_hotspot_definition", "final_hotspot_rank", "final_hotspot_slots"] + obj_fields, tracking)
    write_csv(output / "class_metrics.csv", list(class_rows[0]), class_rows)
    write_csv(output / "retrieval_class_metrics.csv", list(retrieval_class_rows[0]), retrieval_class_rows)
    write_csv(output / "proxy_retrieval.csv", list(proxy_rows[0]), proxy_rows)
    snapshot_models = [m for m in models if m["kind"] == "V7"]
    baseline_models = [m for m in models if m["kind"] == "baseline"]
    arrays = dict(canonical_ids=ids, labels=labels, v7_epochs=np.array([m["epoch"] for m in snapshot_models]),
                  v7_depth=np.stack([m["depth"] for m in snapshot_models]),
                  v7_depth_rank=np.stack([m["rank"] for m in snapshot_models]),
                  v7_center_angle_deg=np.stack([m["center_angle"] for m in snapshot_models]),
                  v7_raw_counts4=np.stack([m["raw_counts4"] for m in snapshot_models]),
                  v7_direction_counts4=np.stack([m["direction_counts4"] for m in snapshot_models]),
                  v7_proxy_depth=np.stack([m["proxy_depth"] for m in snapshot_models]),
                  tracked_hotspot_rows=np.array(sorted(tracked_rows), dtype=np.int64),
                  baseline_names=np.array([m["name"] for m in baseline_models]))
    if baseline_models:
        arrays.update(baseline_depth=np.stack([m["depth"] for m in baseline_models]),
                      baseline_depth_rank=np.stack([m["rank"] for m in baseline_models]),
                      baseline_full_depth_rank=np.stack([m["full_rank"] for m in baseline_models]),
                      baseline_center_angle_deg=np.stack([m["center_angle"] for m in baseline_models]))
    np.savez_compressed(output / "plot_arrays.npz", **arrays)
    for m in models:
        if sha(m["path"]) != m["source"]["cache_sha256"]:
            raise RuntimeError("Source cache changed during read-only audit")
    for item in snapshots + spec.get("baselines", []):
        if item.get("summary"):
            m = next(v for v in models if v["epoch"] == int(item["epoch"]) and v["name"] == item.get("name", "V7_e" + str(item["epoch"])))
            if sha(item["summary"]) != m["source"]["summary_sha256"]:
                raise RuntimeError("Source summary changed")
    summary = dict(schema="hier_v7_hotspot_baseline_v1", identity="sorted HDF5 shard name + row; every shared label asserted",
                   common_objects=len(ids), classes=NAMES, rank_definition="one-based increasing depth, exact ties by canonical ID; percentile=rank/pool_n",
                   retrieval_definition="512 full mapped V7 proxy positions to all common whole objects; label-free geodesic / direction ranking",
                   proxy_mapping="saved tangent expmap0 then numeric ball radius fraction .999; no added tangent cap",
                   class_direction_definition="unit whole vectors; class center normalized mean; resultant norm and original-dimensional angles",
                   cached_ce_definition="whole_ce is original label-smoothed CE, not unmodified class confidence",
                   cached_softmax_definition="softmax applied to cached raw classifier output; not calibrated probability; absent for baseline caches with predictions only",
                   topks=topks, final_hotspot_epoch=final["epoch"], models=results,
                   tracked_hotspot_ids=ids[np.array(sorted(tracked_rows), dtype=int)].tolist(),
                   limits=["All objects are existing clean TRAIN caches. No new train/validation/test model forwards.",
                           "Proxy retrieval is different from actual hard-Gumbel training ancestor selection.",
                           "Class geometry and cross-model same-ID ranks are descriptive: baseline protocols / model-selection epochs differ.",
                           "V7 proxies are never queried against independently trained, unaligned baseline directions.",
                           "Whole equal-depth retrieval equals direction ranking mathematically; it is not an independent metric.",
                           "Hotspot class association alone is not evidence of independent shape or morphology hierarchy.",
                           "A deepest or shallowest rank says relative radial position, not semantic genericity."],
                   read_only=dict(all_source_caches_unchanged=True, all_source_summaries_unchanged=True,
                                  mapped_ids_and_all_labels_verified=True, optimizer_updates=0, encoder_forwards=0, test_forwards=0, GPU=None))
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    manifest.update(status="completed", finished_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                    wall_seconds=time.monotonic() - started, source_caches_unchanged=True,
                    source_summaries_unchanged=True, shared_objects=len(ids), source_models=len(models))
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("completed", len(models), "models in", round(manifest["wall_seconds"], 2), "seconds", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    main(args.spec, args.output)
