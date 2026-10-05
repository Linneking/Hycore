"""CPU-only, rotation-invariant diagnosis of existing HyCoRe/HIER caches.

Input config and full instance-level output remain private. No model forward,
optimizer update, new mining rule or change to existing result directories.
Canonical object identity is (HDF5 shard name, row), not an unqualified ID.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np


NAMES = "airplane bathtub bed bench bookshelf bottle bowl car chair cone cup curtain desk door dresser flower_pot glass_box guitar keyboard lamp laptop mantel monitor night_stand person piano plant radio range_hood sink sofa stairs stool table tent toilet tv_stand vase wardrobe xbox".split()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def quantiles(x):
    return np.quantile(x, [.01, .1, .5, .9, .99]).tolist()


def rank(x):
    _, inv, counts = np.unique(np.asarray(x), return_inverse=True, return_counts=True)
    ends = np.cumsum(counts)
    return ((ends - counts + ends - 1) / 2)[inv]


def spearman(x, y):
    a, b = rank(x), rank(y)
    if np.std(a) == 0 or np.std(b) == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def geom(x):
    x = np.asarray(x, dtype=np.float64)
    r = np.linalg.norm(x, axis=1)
    if not np.isfinite(x).all() or np.any(r <= 0) or np.any(r >= 1):
        raise ValueError("Expected finite nonzero c1 ball vectors")
    return x, r, x / r[:, None], 2 * np.arctanh(r)


def ball_dist(x, y):
    xx = np.sum(x*x, 1)
    yy = np.sum(y*y, 1)
    diff = np.maximum(xx[:, None] + yy[None, :] - 2*x@y.T, 0)
    return 2*np.arcsinh(np.sqrt(diff / ((1-xx)[:, None]*(1-yy)[None, :])))


def topk(d, k=4):
    # Cache rows are sorted by canonical ID: stable sorting breaks ties by ID.
    return np.argsort(d, axis=1, kind="stable")[:, :k]


def remap_ids(ids, shards):
    by_name = {s["name"]: int(s["count"]) for s in shards}
    if len(by_name) != len(shards):
        raise ValueError("Repeated shard name")
    offsets, offset = {}, 0
    for name in sorted(by_name):
        offsets[name] = offset
        offset += by_name[name]
    mapping = np.concatenate([offsets[s["name"]] + np.arange(int(s["count"])) for s in shards])
    return mapping[np.asarray(ids, dtype=np.int64)]


def load_model(item, common_ids=None, reference_labels=None):
    path = Path(item["cache"])
    summary = json.loads(path.with_name("summary.json").read_text())
    z = np.load(path, allow_pickle=False)
    is_hier = "mu" in z.files
    if is_hier:
        ids = z["sample_ids"]
        mu = z["mu"]
        meta = summary["checkpoint"]
        info = {"checkpoint_sha256": meta["sha256"], "epoch": meta["epoch"]}
        shard_names = summary["dataset"]["training_shards_sorted"]
    else:
        ids = remap_ids(z["ids"], summary["data_identity"]["source_shards_in_glob_order"])
        mu = z["whole_mu"]
        info = {"checkpoint_sha256": summary["checkpoint_sha256"], "epoch": item["epoch"]}
        shard_names = sorted(s["name"] for s in summary["data_identity"]["source_shards_in_glob_order"])
    labels = z["labels"]
    if len(np.unique(ids)) != len(ids):
        raise ValueError("Duplicate canonical IDs")
    lookup = {int(v): i for i, v in enumerate(ids)}
    common_ids = np.sort(ids) if common_ids is None else common_ids
    rows = np.array([lookup[int(v)] for v in common_ids])
    labels = labels[rows]
    if reference_labels is not None:
        np.testing.assert_array_equal(labels, reference_labels)
    x, r, u, d0 = geom(mu[rows])
    m = {"name": item["name"], "kind": "HIER" if is_hier else "HyCoRe",
         "ids": common_ids, "labels": labels, "x": x, "r": r, "u": u,
         "depth": d0, "source": {**info, "cache_sha256": digest(path)}, "cache": z,
         "canonical_shards": shard_names}
    if is_hier:
        m["p"], m["pr"], m["pu"], _ = geom(z["proxy_ball"])
        m["eligible"] = z["proxy_eligible"].astype(bool)
    return m


def summarize_model(m):
    n = len(m["ids"])
    hyp_nn = np.empty((n, 5), dtype=np.int64)
    angle_nn = np.empty_like(hyp_nn)
    center_angle = np.empty(n)
    hyp_in = np.zeros(n, dtype=np.int64)
    angle_in = np.zeros(n, dtype=np.int64)
    classes, angle_pairs, hyp_pairs = [], [], []
    for c, name in enumerate(NAMES):
        at = np.flatnonzero(m["labels"] == c)
        u, x = m["u"][at], m["x"][at]
        cosine = np.clip(u@u.T, -1, 1)
        angles = np.degrees(np.arccos(cosine))
        distances = ball_dist(x, x)
        tri = np.triu_indices(len(at), 1)
        aa, dd = angles[tri], distances[tri]
        angle_pairs.append(aa)
        hyp_pairs.append(dd)
        center = u.mean(0)
        resultant = np.linalg.norm(center)
        center_angle[at] = np.degrees(np.arccos(np.clip(u@(center/resultant), -1, 1)))
        np.fill_diagonal(angles, np.inf)
        np.fill_diagonal(distances, np.inf)
        hyp_nn[at] = at[topk(distances, 5)]
        angle_nn[at] = at[topk(angles, 5)]
        hyp_in[at] = np.bincount(topk(distances, 5).ravel(), minlength=len(at))
        angle_in[at] = np.bincount(topk(angles, 5).ravel(), minlength=len(at))
        classes.append({"class": name, "n": int(len(at)), "depth": quantiles(m["depth"][at]),
                        "radius": quantiles(m["r"][at]), "angle": quantiles(aa),
                        "distance": quantiles(dd), "R": float(resultant),
                        "hyp_hub_max": int(hyp_in[at].max()), "angle_hub_max": int(angle_in[at].max())})
    m.update(hyp_nn=hyp_nn, angle_nn=angle_nn, center_angle=center_angle,
             hyp_in=hyp_in, angle_in=angle_in)
    return {"name": m["name"], "kind": m["kind"], "source": m["source"],
            "n": n, "depth": quantiles(m["depth"]), "radius": quantiles(m["r"]),
            "angle": quantiles(np.concatenate(angle_pairs)), "distance": quantiles(np.concatenate(hyp_pairs)),
            "class_R_mean": float(np.mean([c["R"] for c in classes])), "classes": classes,
            "depth_cdf": np.quantile(m["depth"], np.linspace(0, 1, 101)).tolist()}


def hub_report(indices, m, mask=None, omitted=None):
    indices = indices if mask is None else indices[mask]
    counts = np.bincount(indices.ravel(), minlength=len(m["ids"]))
    total = int(counts.sum())
    q = counts[counts > 0] / total
    lowest = np.argsort(m["r"], kind="stable")[:int(np.ceil(.01*len(counts))) if omitted is None else 0]
    groups = Counter(tuple(sorted(row)) for row in indices.tolist())
    best_set, repeats = groups.most_common(1)[0]
    return {"proxies": int(len(indices)), "distinct_ids": int(np.sum(counts > 0)),
            "effective_ids": float(np.exp(-np.sum(q*np.log(q)))), "top_id_share": float(counts.max()/total),
            "distinct_top4_sets": len(groups), "largest_set_proxies": int(repeats),
            "largest_set_ids": m["ids"][list(best_set)].tolist(),
            "lowest_one_percent_slot_share": float(counts[lowest].sum()/total) if len(lowest) else None,
            "radius_hub_spearman": spearman(m["r"], counts),
            "top_ids": [{"id": int(m["ids"][i]), "class": NAMES[int(m["labels"][i])],
                         "slots": int(counts[i]), "depth": float(m["depth"][i]),
                         "radius": float(m["r"][i])} for i in np.argsort(-counts, kind="stable")[:12]],
            "hub_counts": counts}


def proxy_diagnosis(m, baseline):
    raw_d = ball_dist(m["p"], m["x"])
    raw = topk(raw_d)
    # Reproduce prior saved raw nearest IDs before interpreting controls.
    np.testing.assert_array_equal(m["ids"][raw], m["cache"]["all_proxy_nearest_ids"])
    cases = {"raw": raw,
             "whole_equal_radius": topk(ball_dist(m["p"], .98*m["u"])),
             "proxy_equal_radius": topk(ball_dist(.98*m["pu"], m["x"])),
             "both_equal_radius": topk(ball_dist(.98*m["pu"], .98*m["u"])),
             "direction": topk(-m["pu"]@m["u"].T),
             "borrow_b32_radii": topk(ball_dist(m["p"], baseline["r"][:, None]*m["u"]))}
    np.testing.assert_array_equal(cases["whole_equal_radius"], cases["direction"])
    np.testing.assert_array_equal(cases["both_equal_radius"], cases["direction"])
    excluded = np.argsort(m["r"], kind="stable")[:int(np.ceil(.01*len(m["ids"])))]
    raw_removed = raw_d.copy()
    raw_removed[:, excluded] = np.inf
    cases["remove_lowest_one_percent"] = topk(raw_removed)
    report = {}
    for case, nn in cases.items():
        report[case] = {}
        for group, mask in [("all", None), ("eligible", m["eligible"])]:
            r = hub_report(nn, m, mask, excluded if case.startswith("remove_") else None)
            r.pop("hub_counts")
            report[case][group] = r
    m["proxy_hubs"] = hub_report(raw, m, m["eligible"])["hub_counts"]
    m["proxy_hubs_angle"] = hub_report(cases["direction"], m, m["eligible"])["hub_counts"]
    report["eligible_proxy_radii"] = quantiles(m["pr"][m["eligible"]])
    # Same-top4 proxies can be far apart: repeated retrieval is not coincidence.
    fixed = raw[m["eligible"]]
    selected = m["p"][m["eligible"]]
    selected_u = m["pu"][m["eligible"]]
    tri = np.triu_indices(len(selected), 1)
    keys = [tuple(sorted(row)) for row in fixed.tolist()]
    same = np.array([keys[i] == keys[j] for i, j in zip(*tri)])
    dp = ball_dist(selected, selected)[tri]
    ap = np.degrees(np.arccos(np.clip(selected_u@selected_u.T, -1, 1)))[tri]
    report["same_set_proxy_pairs"] = {"n": int(same.sum()), "distance": quantiles(dp[same]),
                                       "angle": quantiles(ap[same]), "coincident_pairs_le_1e_6": int(np.sum(dp <= 1e-6))}
    return report


def neighbor_overlap(a, b):
    shared = np.array([len(set(x).intersection(y)) for x, y in zip(a, b)])
    return float(np.mean(shared / (10-shared)))


def self_check():
    np.testing.assert_array_equal(remap_ids(np.arange(5), [{"name": "b", "count": 2}, {"name": "a", "count": 3}]), [3,4,0,1,2])
    np.testing.assert_allclose(rank([1,1,5,9]), [.5,.5,2,3])
    x = np.array([[.2,0], [.6,0], [0,.2]])
    np.testing.assert_allclose(ball_dist(x[:1], x[1:2])[0,0], 2*(np.arctanh(.6)-np.arctanh(.2)))
    p = np.array([[.5,.1],[-.2,.3]])
    _, _, u, _ = geom(x)
    np.testing.assert_array_equal(topk(ball_dist(p,.7*u),3), topk(-p@u.T,3))
    rotation = np.array([[0,-1],[1,0]])
    np.testing.assert_allclose(ball_dist(x@rotation,p@rotation),ball_dist(x,p))
    print("identity remapping, tied ranks, c1 radial distance, equal-radius ranking and rotation checks passed")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        self_check()
        return
    started = time.monotonic()
    config = json.loads(args.config.read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = {"status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
                "GPU": "none; NumPy CPU only", "seed": 22, "config": config,
                "numpy": np.__version__, "script_sha256": digest(__file__), "no_parameter_updates": True,
                "code_commit": subprocess.check_output(["git","rev-parse","HEAD"], text=True).strip()}
    (args.output/"manifest.json").write_text(json.dumps(manifest, indent=2))
    models, results = {}, {}
    ref = load_model(config["models"][0])
    ids, labels = ref["ids"], ref["labels"]
    for item in config["models"]:
        m = load_model(item, ids, labels)
        assert m["canonical_shards"] == ref["canonical_shards"]
        models[m["name"]] = m
        results[m["name"]] = summarize_model(m)
        print("geometry",m["name"],"matched",len(ids),"depth",results[m["name"]]["depth"][2],flush=True)
    baseline = models[config["radius_baseline"]]
    for name, m in models.items():
        if m["kind"] == "HIER":
            results[name]["proxy"] = proxy_diagnosis(m, baseline)
            print("proxy",name,"raw/direction distinct",results[name]["proxy"]["raw"]["eligible"]["distinct_ids"],results[name]["proxy"]["direction"]["eligible"]["distinct_ids"],flush=True)
    comparisons = []
    for a_name, a in models.items():
        if a["kind"] != "HIER":
            continue
        for b_name, b in models.items():
            if a_name == b_name:
                continue
            comparisons.append({"a": a_name,"b": b_name,"depth_rank_spearman":spearman(a["depth"],b["depth"]),
                                "within_class_hyp_knn5_jaccard":neighbor_overlap(a["hyp_nn"],b["hyp_nn"]),
                                "within_class_angle_knn5_jaccard":neighbor_overlap(a["angle_nn"],b["angle_nn"])})
    rng = np.random.default_rng(22)
    selected = set()
    for c in range(40):
        at = np.flatnonzero(labels == c)
        selected.update(rng.choice(at, min(24,len(at)), replace=False).tolist())
    for m in models.values():
        if m["kind"] == "HIER":
            selected.update(np.argsort(-m["proxy_hubs"], kind="stable")[:16].tolist())
    selected = np.array(sorted(selected))
    for name,m in models.items():
        hub_raw = m.get("proxy_hubs",np.zeros(len(ids),dtype=int))
        hub_angle = m.get("proxy_hubs_angle",np.zeros(len(ids),dtype=int))
        results[name]["points"] = [[int(ids[i]),int(labels[i]),round(float(m["depth"][i]),5),
                                    round(float(m["center_angle"][i]),5),int(m["hyp_in"][i]),int(m["angle_in"][i]),
                                    int(hub_raw[i]),int(hub_angle[i])] for i in selected]
    output = {"status":"completed","canonical_identity":"sorted HDF5 shard name + row; label alignment asserted for every model",
              "shared_training_objects":len(ids),"seed":22,"classes":NAMES,"quantile_levels":[.01,.1,.5,.9,.99],
              "point_fields":["canonical_id","class_index","depth","angle_to_own_class_mean_deg","hyp_knn5_incoming","angle_knn5_incoming","eligible_proxy_raw_top4_slots","eligible_proxy_direction_top4_slots"],
              "models":results,"comparisons":comparisons,
              "limits":["Existing full clean eval caches; training protocols and checkpoint epochs differ, so descriptive comparisons are not isolated HIER effects.",
                        "Equal-radius copies and borrowed radii are frozen retrieval controls, not training proposals or evidence of morphology.",
                        "Whole equal-radius and direction top4 rankings are mathematically equivalent; reported as a validation check, not independent evidence.",
                        "Proxy-to-sample retrieval differs from actual hard-Gumbel ancestor use and sample-to-proxy assignment.",
                        "All original 256D calculations use FP64 conversion of cached FP32 vectors; no 2D projection is used for distances."]}
    (args.output/"diagnostic.json").write_text(json.dumps(output,separators=(",",":"),allow_nan=False))
    manifest.update(status="completed",finished_utc=datetime.now(timezone.utc).isoformat(),wall_seconds=time.monotonic()-started,
                    matched_labels_all=True,shared_objects=len(ids),all_source_caches_unchanged=True)
    # The source NPZs are never opened for writing; verify their identities again.
    for item in config["models"]:
        assert digest(item["cache"]) == results[item["name"]]["source"]["cache_sha256"]
    (args.output/"manifest.json").write_text(json.dumps(manifest,indent=2))
    print("completed",round(manifest["wall_seconds"],2),"seconds",flush=True)


if __name__ == "__main__":
    main()
