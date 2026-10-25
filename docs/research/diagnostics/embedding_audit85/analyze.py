"""CPU-only analysis of immutable, identity-aligned clean/eval embedding caches.

No model, checkpoint loader, CUDA, training, calibration or test-set access.
Public outputs contain aggregated geometry and class names, never coordinates,
object IDs or private source paths. Frozen minimax diagnostics are not replays
of the stochastic Gumbel choices made during training.
"""
from __future__ import annotations

import os
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "4"

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np

CLASS_NAMES = ("airplane bathtub bed bench bookshelf bottle bowl car chair cone cup curtain desk door dresser "
               "flower_pot glass_box guitar keyboard lamp laptop mantel monitor night_stand person piano plant "
               "radio range_hood sink sofa stairs stool table tent toilet tv_stand vase wardrobe xbox").split()
QUANTILES = {"min": 0., "p01": .01, "p05": .05, "p10": .1, "p25": .25,
             "median": .5, "p75": .75, "p90": .9, "p95": .95, "p99": .99, "max": 1.}
ANGLE_EDGES = np.arange(0., 180.0001, .25)
FIXED_PAIRS = (("chair", "stool"), ("table", "desk"), ("dresser", "night_stand"),
               ("bottle", "vase"), ("cup", "vase"), ("airplane", "car"))
TOLERANCES = (1e-4, 1e-3, 1e-2)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def array_digest(array):
    x = np.ascontiguousarray(array)
    h = hashlib.sha256()
    h.update(str(x.dtype).encode())
    h.update(str(x.shape).encode())
    h.update(x.tobytes())
    return h.hexdigest()


def integer_vector(values, name):
    values=np.asarray(values)
    if not np.issubdtype(values.dtype,np.integer):
        raise ValueError(name+" must be stored as integers; do not silently truncate IDs/labels")
    flat=values.reshape(-1)
    if np.any(flat<0) or (len(flat) and int(flat.max())>np.iinfo(np.int64).max):
        raise ValueError("Invalid nonnegative integer "+name)
    return flat.astype(np.int64)


def describe(values, *, overwrite=False):
    x = np.asarray(values).reshape(-1)
    if not len(x):
        return None
    if not np.isfinite(x).all():
        raise ValueError("Nonfinite distribution")
    result = {"count": int(len(x)), "mean": float(np.mean(x, dtype=np.float64)),
              "std_population": float(np.std(x, dtype=np.float64))}
    q = np.quantile(x, list(QUANTILES.values()), overwrite_input=overwrite)
    result.update({name: float(value) for name, value in zip(QUANTILES, q)})
    return result


def unit_directions(points):
    x = np.asarray(points, dtype=np.float64)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError("Expected finite [N,D] coordinates")
    radius = np.linalg.norm(x, axis=1)
    if np.any(radius <= 1e-12):
        raise ValueError("A zero/near-zero vector has no defined direction")
    return x / radius[:, None]


def origin_depth(points, c):
    if not np.isfinite(c) or c <= 0:
        raise ValueError("Positive curvature magnitude required")
    x = np.asarray(points, dtype=np.float64)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError("Expected finite [N,D] coordinates")
    q = math.sqrt(c) * np.linalg.norm(x, axis=1)
    if np.any(q >= 1):
        raise ValueError("Coordinates are outside their declared Poincare ball")
    return 2 / math.sqrt(c) * np.arctanh(q)


def angle_degrees(left, right):
    return np.degrees(np.arccos(np.clip(left @ right.T, -1., 1.)))


def class_centers(unit, labels):
    classes = np.unique(labels)
    means = np.stack([unit[labels == k].mean(axis=0) for k in classes])
    resultant = np.linalg.norm(means, axis=1)
    if np.any(resultant <= 1e-12):
        raise ValueError("An antipodally balanced class has no defined direction center")
    return classes, means / resultant[:, None], resultant


def class_cones(unit, labels, classes=None, centers=None):
    """Pooled sample-to-own-center angles, independent of individual radii.

    A percentile cone is a description of the observed class directions,
    not a classifier decision region or a guarantee about unseen samples.
    """
    unit = np.asarray(unit, dtype=np.float64)
    labels = np.asarray(labels).reshape(-1)
    if classes is None or centers is None:
        classes, centers, _ = class_centers(unit, labels)
    return {
        int(k): describe(np.degrees(np.arccos(np.clip(
            unit[labels == k] @ centers[pos], -1., 1.))))
        for pos, k in enumerate(classes)
    }


def poincare_distances(left, right, c, *, same=False):
    x, y = np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)
    xs, ys = (x*x).sum(1), (y*y).sum(1)
    if np.any(c*xs >= 1) or np.any(c*ys >= 1):
        raise ValueError("Invalid ball input")
    squared = np.maximum(xs[:, None] + ys[None, :] - 2*x @ y.T, 0.)
    denominator = np.sqrt((1-c*xs)[:, None] * (1-c*ys)[None, :])
    result = 2 / math.sqrt(c) * np.arcsinh(math.sqrt(c) * np.sqrt(squared) / denominator)
    if same:
        np.fill_diagonal(result, 0.)
    return result


def usage_summary(counts):
    if counts is None:
        return {"available": False}
    counts = np.asarray(counts, dtype=np.float64)
    if counts.ndim != 1 or np.any(counts < 0) or not np.isfinite(counts).all():
        raise ValueError("Invalid proxy usage vector")
    total = float(counts.sum())
    p = counts[counts > 0] / total if total else np.empty(0)
    entropy = float(-(p*np.log(p)).sum())
    return {"available": True, "selection_count": total, "used_proxy_count": int(np.count_nonzero(counts)),
            "entropy_nats": entropy if total else None, "effective_proxy_count": math.exp(entropy) if total else 0.,
            "maximum_selection_share": float(counts.max()/total) if total else None}


def histogram_rows(values, edges, meta):
    counts, edges = np.histogram(values, bins=edges)
    return [{**meta, "bin_left": float(a), "bin_right": float(b), "bin_center": float((a+b)/2),
             "count": int(n), "percent": float(100*n/len(values))}
            for a,b,n in zip(edges[:-1],edges[1:],counts)]


def depth_outputs(values, c, meta):
    # Bins centered on multiples of .1 keep both 4.59999 and 4.60000 in the
    # same physical shell bin. First bin is clipped at the physical origin.
    top = max(8., math.ceil(float(np.max(values))/.1)*.1)
    edges = np.arange(-.05, top+.10001, .1)
    edges[0] = 0.
    histogram = histogram_rows(values, edges, {**meta, "c": float(c)})
    grid = np.arange(0., top+.02501, .025)
    ordered = np.sort(values)
    cdf = [{**meta, "c": float(c), "depth": float(t),
            "cumulative_percent": float(100*np.searchsorted(ordered,t,side="right")/len(ordered))}
           for t in grid]
    return histogram, cdf


def csv_write(path, rows):
    path = Path(path)
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w",encoding="utf-8-sig",newline="") as stream:
        writer = csv.DictWriter(stream,fieldnames=keys)
        writer.writeheader(); writer.writerows(rows)


def resolve_cache(manifest_path, spec):
    path = Path(spec["cache"])
    return path if path.is_absolute() else manifest_path.parent/path


def observation_meta(spec):
    key = str(spec.get("key",spec.get("run_id","")))
    stage = str(spec["stage"])
    run_id=str(spec.get("run_id",key))
    if not key or not all(x.isalnum() or x in "_-" for x in key+stage+run_id):
        raise ValueError("Observation key/stage must use safe labels")
    return {"observation_id": key+"__"+stage, "run_id": run_id,
            "key": key, "stage": stage, "epoch": int(spec["epoch"]), "primary": bool(spec.get("primary",False))}


def reference_and_pairs(manifest, manifest_path, names):
    observations = manifest["observations"]
    explicit = manifest.get("representative_reference")
    candidate = [s for s in observations if str(s.get("run_id",s.get("key"))) == "TS13" and s["stage"] == "best"]
    if explicit:
        candidate = [s for s in observations if observation_meta(s)["observation_id"] == explicit or s.get("key") == explicit]
    if not candidate:
        candidate = [s for s in observations if s["stage"] == "best" and (s.get("kind") == "cycle_off" or "cycle_off" in s.get("key",""))]
    if len(candidate) != 1:
        raise ValueError("A unique TS13-best representative reference must be declared")
    ref = candidate[0]
    with np.load(resolve_cache(manifest_path,ref),allow_pickle=False) as cache:
        unit = unit_directions(cache["native_mu"])
        labels = np.asarray(cache["labels"]).reshape(-1)
    classes, centers, _ = class_centers(unit,labels)
    angles = angle_degrees(centers,centers)
    mapping = {name:i for i,name in enumerate(names)}
    selected = {}
    for a,b in FIXED_PAIRS:
        if a in mapping and b in mapping:
            pair = tuple(sorted((mapping[a],mapping[b])))
            selected[pair] = "fixed_named_comparison"
    options = sorted((float(angles[i,j]),int(a),int(classes[j]))
                     for i,a in enumerate(classes) for j in range(i+1,len(classes)))
    added = 0
    for _,a,b in options:
        pair = (a,b)
        if pair not in selected:
            selected[pair] = "five_closest_extra_pairs_in_TS13_best"
            added += 1
            if added == 5:
                break
    return selected, {"reference": observation_meta(ref), "fixed_names": [list(p) for p in FIXED_PAIRS],
                      "selection": "six predeclared named pairs plus five closest additional centroid pairs in TS13-best; same pairs for every observation",
                      "pairs": [{"class_a":a,"class_b":b,"name_a":names[a],"name_b":names[b],"reason":why}
                                for (a,b),why in selected.items()]}


def analyze_angles(unit, labels, names, meta, representative):
    classes, centers, resultant = class_centers(unit,labels)
    cones = class_cones(unit, labels, classes, centers)
    center_angles = angle_degrees(centers,centers)
    groups = {int(k):unit[labels==k] for k in classes}
    intra_rows, pair_rows, center_rows, angle_hist, rep_rows, rep_hist = [],[],[],[],[],[]
    intra_values = []
    intra_by_class = {}
    inter_count = (len(labels)**2 - sum(len(v)**2 for v in groups.values()))//2
    inter_pooled = np.empty(inter_count,dtype=np.float32)
    offset = 0
    for pos,k in enumerate(classes):
        k=int(k); u=groups[k]
        a=angle_degrees(u,u); values=a[np.triu_indices(len(u),k=1)]
        stat=describe(values)
        intra_by_class[k]=stat
        intra_values.append(values.astype(np.float32))
        intra_rows.append({**meta,"class_id":k,"class_name":names[k],"objects":len(u),
                           "direction_resultant_R":float(resultant[pos]),
                           "cone_radius_p90":cones[k]["p90"],
                           "cone_radius_p95":cones[k]["p95"],
                           "sample_to_center_mean":cones[k]["mean"],
                           "sample_to_center_median":cones[k]["median"],
                           **(stat or {})})
    intra_pooled=np.concatenate(intra_values)
    angle_hist.extend(histogram_rows(intra_pooled,ANGLE_EDGES,{**meta,"domain":"intra_all_pairs"}))
    intra_summary=describe(intra_pooled,overwrite=True)
    for i,a in enumerate(classes):
        a=int(a)
        for j in range(i+1,len(classes)):
            b=int(classes[j]); values=angle_degrees(groups[a],groups[b]).reshape(-1)
            stat=describe(values)
            inter_pooled[offset:offset+len(values)]=values
            offset+=len(values)
            row={**meta,"class_a":a,"class_b":b,"name_a":names[a],"name_b":names[b],
                 "objects_a":len(groups[a]),"objects_b":len(groups[b]),
                 "centroid_angle":float(center_angles[i,j]),**stat,
                 "cone_radius_a_p90":cones[a]["p90"],"cone_radius_b_p90":cones[b]["p90"],
                 "cone_radius_a_p95":cones[a]["p95"],"cone_radius_b_p95":cones[b]["p95"],
                 "cone_gap_p90":float(center_angles[i,j]-cones[a]["p90"]-cones[b]["p90"]),
                 "cone_gap_p95":float(center_angles[i,j]-cones[a]["p95"]-cones[b]["p95"]),
                 "separation_p10_minus_max_intra_p90":float(stat["p10"]-max(intra_by_class[a]["p90"],intra_by_class[b]["p90"]))}
            pair_rows.append(row)
            center_rows.append({**meta,"class_a":a,"class_b":b,"name_a":names[a],"name_b":names[b],"angle_degrees":float(center_angles[i,j])})
            if (a,b) in representative:
                rep_rows.append({**row,"selection_reason":representative[(a,b)]})
                rep_hist.extend(histogram_rows(values,ANGLE_EDGES,{**meta,"class_a":a,"class_b":b,"name_a":names[a],"name_b":names[b]}))
    if offset != inter_count:
        raise RuntimeError("Inter-class pair denominator mismatch")
    angle_hist.extend(histogram_rows(inter_pooled,ANGLE_EDGES,{**meta,"domain":"inter_all_pairs"}))
    inter_summary=describe(inter_pooled,overwrite=True)
    class_balanced={name:float(np.mean([row[name] for row in intra_rows])) for name in ("mean","median","p10","p90")}
    pair_balanced={name:float(np.mean([row[name] for row in pair_rows])) for name in ("mean","median","p10","p90","centroid_angle")}
    summary={"angle_unit":"degrees", "intra_pair_weighted":intra_summary,"inter_pair_weighted":inter_summary,
             "class_balanced_intra_average_of_class_statistics":class_balanced,
             "class_pair_balanced_inter_average_of_pair_statistics":pair_balanced,
             "class_direction_resultant_R":describe(resultant),
             "class_cone_radius_p90":describe([cones[int(k)]["p90"] for k in classes]),
             "class_cone_radius_p95":describe([cones[int(k)]["p95"] for k in classes]),
             "class_pair_cone_gap_p90":describe([row["cone_gap_p90"] for row in pair_rows]),
             "class_pair_cone_gap_p95":describe([row["cone_gap_p95"] for row in pair_rows]),
             "class_pairs_with_negative_cone_gap_p90":sum(row["cone_gap_p90"]<0 for row in pair_rows),
             "class_pairs_with_negative_cone_gap_p95":sum(row["cone_gap_p95"]<0 for row in pair_rows),
             "class_cone_limit":"P90/P95 observed directional cones; center separation minus radii is a geometric description, not a classifier boundary or a proof of accuracy changes",
             "centroid_separation_upper_triangle":describe(center_angles[np.triu_indices(len(classes),1)]),
             "cross_model_comparison":"within-model angles and relations only; raw axes/proxy IDs are not semantically aligned across models"}
    return summary,(intra_rows,pair_rows,center_rows,angle_hist,rep_rows,rep_hist),centers


def mine_triplets(similarity, *, seed, topk=20, per_anchor=50):
    n=len(similarity)
    # Stable index tie resolution is explicit; not a claim of CUDA topk replay.
    nearest=np.argsort(-similarity,axis=1,kind="stable")[:,:min(topk,n)]
    directed=np.zeros((n,n),dtype=bool)
    np.put_along_axis(directed,nearest,True,axis=1)
    mutual=directed & directed.T
    np.fill_diagonal(mutual,False)
    negative=~mutual; np.fill_diagonal(negative,False)
    eligible=(mutual.sum(1)>=2)&negative.any(1)
    rng=np.random.RandomState(seed)
    rows=[]
    for i in np.flatnonzero(eligible):
        js=rng.choice(np.flatnonzero(mutual[i]),per_anchor,replace=True)
        ks=rng.choice(np.flatnonzero(negative[i]),per_anchor,replace=True)
        rows.extend(zip(np.repeat(i,per_anchor),js,ks))
    return np.asarray(rows,dtype=np.int64).reshape(-1,3),{"eligible_anchors":int(eligible.sum()),"anchors":n,"mutual_degree_mean":float(mutual.sum(1).mean())}


def minimax_selection(distances, triplets, chunk=512):
    pair,triple=[],[]
    for begin in range(0,len(triplets),chunk):
        i,j,k=triplets[begin:begin+chunk].T
        pair_cost=np.maximum(distances[i],distances[j])
        pair.append(pair_cost.argmin(1))
        triple.append(np.maximum(pair_cost,distances[k]).argmin(1))
    return (np.concatenate(pair),np.concatenate(triple)) if pair else (np.empty(0,dtype=int),np.empty(0,dtype=int))


def order_statistics(endpoint_depth, proxy_depth, triplets, pair, triple, *, proxy_endpoints=False):
    i,j,k=triplets.T
    pair_gap=np.minimum(endpoint_depth[i],endpoint_depth[j])-proxy_depth[pair]
    triple_gap=np.minimum(np.minimum(endpoint_depth[i],endpoint_depth[j]),endpoint_depth[k])-proxy_depth[triple]
    nested_gap=proxy_depth[pair]-proxy_depth[triple]
    result={"draws":len(triplets),"collision_fraction":float(np.mean(pair==triple)) if len(pair) else None,"domains":{}}
    nonendpoint=np.ones(len(pair),dtype=bool)
    if proxy_endpoints:
        pair_endpoint=(pair==i)|(pair==j)
        triple_endpoint=(triple==i)|(triple==j)|(triple==k)
        result["pair_equals_an_endpoint_fraction"]=float(np.mean(pair_endpoint)) if len(pair) else None
        result["triple_equals_an_endpoint_fraction"]=float(np.mean(triple_endpoint)) if len(pair) else None
        result["pair_equals_an_endpoint_count"]=int(pair_endpoint.sum())
        result["triple_equals_an_endpoint_count"]=int(triple_endpoint.sum())
        nonendpoint=~(pair_endpoint|triple_endpoint)
    result["nonendpoint_domain_semantics"]=(
        "pair ancestor differs from i/j; triple ancestor differs from i/j/k; proxy endpoint pool shares ancestor identities"
        if proxy_endpoints else
        "sample endpoints and proxy ancestors are different identity domains; nonendpoint filtering removes no additional draws")
    for name,mask in (("all",np.ones(len(pair),dtype=bool)),("noncollision",pair!=triple),
                      ("noncollision_nonendpoint",(pair!=triple)&nonendpoint)):
        details={"draws":int(mask.sum()),"gaps":{k:describe(v[mask]) for k,v in (("pair_min_endpoint_gap",pair_gap),("triple_min_endpoint_gap",triple_gap),("pair_minus_triple_depth",nested_gap))},"tolerance":{}}
        for tol in TOLERANCES:
            def frac(condition): return float(np.mean(condition[mask])) if mask.any() else None
            details["tolerance"][str(tol)]={
                "pair_shallower_than_both_endpoints":frac(pair_gap>tol),
                "pair_deeper_than_at_least_one_endpoint":frac(pair_gap < -tol),
                "pair_endpoint_gap_tied":frac(np.abs(pair_gap)<=tol),
                "triple_shallower_than_all_endpoints":frac(triple_gap>tol),
                "triple_shallower_than_pair":frac(nested_gap>tol),
                "pair_triple_depth_tied":frac(np.abs(nested_gap)<=tol),
                "all_three_orders_strict":frac((pair_gap>tol)&(triple_gap>tol)&(nested_gap>tol))}
        result["domains"][name]=details
    return result


def frozen_orders(hier_mu, proxies, labels, *, sample_batches=8):
    c=.1; dp=poincare_distances(proxies,proxies,c,same=True)
    proxy_d=origin_depth(proxies,c)
    triplets,graph=mine_triplets(np.exp(-dp),seed=4781)
    pair,triple=minimax_selection(dp,triplets)
    proxy_result=order_statistics(proxy_d,proxy_d,triplets,pair,triple,proxy_endpoints=True)
    proxy_result["graph"]=graph
    rng=np.random.RandomState(4780)
    classes=np.unique(labels)
    rows=[]; pair_rows=[]; triple_rows=[]; depth_rows=[]
    graphs=[]
    for batch in range(sample_batches):
        chosen=rng.choice(classes,32,replace=False)
        indices=np.concatenate([rng.choice(np.flatnonzero(labels==k),2,replace=False) for k in chosen])
        point=hier_mu[indices]; gold=labels[indices]
        ds=poincare_distances(point,point,c,same=True)
        distance=poincare_distances(point,proxies,c)
        triples,g=mine_triplets(np.exp(-ds)+(gold[:,None]==gold[None,:]),seed=4780+batch)
        p,t=minimax_selection(distance,triples)
        rows.append(triples+64*batch); pair_rows.append(p); triple_rows.append(t)
        depth_rows.append(origin_depth(point,c)); graphs.append(g)
    result=order_statistics(np.concatenate(depth_rows),proxy_d,np.concatenate(rows),np.concatenate(pair_rows),np.concatenate(triple_rows))
    result["batches"]=sample_batches; result["graphs"]=graphs
    return {"semantics":"independent frozen clean/eval audit: fixed shared-ID CB64 batches, 32 classes without replacement and 2 distinct objects per class without replacement; each model mines its own reciprocal graph and triplets; deterministic all-proxy minimax argmin; not the training replacement sampler or a Gumbel replay; no optimization or test input",
            "topk":20,"draws_per_eligible_anchor":50,"seed":4780,"sample":result,"proxy":proxy_result}


def nearest_indices(matrix, k, *, largest=False):
    k=min(k,matrix.shape[1]); signed=-matrix if largest else matrix
    picked=np.argpartition(signed,k-1,axis=1)[:,:k]
    order=np.argsort(np.take_along_axis(signed,picked,axis=1),axis=1,kind="stable")
    return np.take_along_axis(picked,order,axis=1)


def neighborhood_stats(indices, labels):
    classes=np.unique(labels)
    count=np.stack([(labels[indices]==k).sum(1) for k in classes],axis=1)
    p=count/indices.shape[1]
    entropy=-(np.where(p>0,p*np.log(np.maximum(p,1e-300)),0.)).sum(1)
    mode=count.argmax(1)
    return {"dominant_class":classes[mode],"purity":count.max(1)/indices.shape[1],"entropy":entropy,
            "distinct_classes":np.count_nonzero(count,axis=1)}


def hotspot_summary(indices,n):
    counts=np.bincount(indices.reshape(-1),minlength=n)
    quartets=np.unique(np.sort(indices,axis=1),axis=0,return_counts=True)[1]
    return {"objects_covered":int(np.count_nonzero(counts)),"pool_objects":n,
            "coverage_percent":float(100*np.count_nonzero(counts)/n),"maximum_object_slots":int(counts.max()),
            "top4_object_slot_share":float(np.sort(counts)[-4:].sum()/indices.size),
            "distinct_neighbor_sets":len(quartets),"maximum_repeated_neighbor_set":int(quartets.max())}


def depth_breadth_association(depths, entropy, repetitions=64, seed=4780):
    """Descriptive depth/entropy association and shuffled-depth baseline.

    Correlated proxies are not independent observations; the permutation
    reference is a diagnostic of this fixed proxy set, not inferential proof.
    """
    depths=np.asarray(depths,dtype=np.float64)
    entropy=np.asarray(entropy,dtype=np.float64)
    if depths.shape!=entropy.shape or depths.ndim!=1 or not np.isfinite(depths).all() or not np.isfinite(entropy).all():
        raise ValueError("Depth and breadth must be aligned finite vectors")
    resolution=1e-3
    # Forward cap can leave several FP32 ulps of radius noise. Treat the
    # declared cap layer as tied, rather than turning those ulps into ranks.
    near_cap=np.abs(depths-4.6)<=resolution+1e-12
    robust_depth=np.rint(depths/resolution)*resolution
    robust_depth[near_cap]=4.6
    levels,counts=np.unique(robust_depth,return_counts=True)
    dominant=int(counts.max()) if len(counts) else 0
    base={"depth_rank_resolution":resolution,"depth_rank_units":"physical origin hyperbolic distance",
          "depth_rank_policy":"snap depths within1e-3 of4.6 to the cap layer; round other depths to the nearest1e-3 before ranking",
          "raw_depth_span":float(np.ptp(depths)) if len(depths) else None,
          "near_cap_proxy_count":int(near_cap.sum()),"robust_depth_levels":int(len(levels)),
          "dominant_depth_group_count":dominant,"non_dominant_proxy_count":int(len(depths)-dominant),
          "robust_depth_groups":[{"depth":float(level),"proxies":int(count)} for level,count in zip(levels,counts)]}
    if len(depths)<3 or np.ptp(depths)<=resolution or len(levels)<2 or np.std(entropy)<=1e-10:
        return {**base,"available":False,"reason":"depth span or semantic breadth is nearly constant at the declared depth resolution"}
    try:
        from scipy.stats import rankdata, spearmanr
    except ImportError:
        return {**base,"available":False,"reason":"SciPy unavailable; geometry remains valid"}
    raw_rho,raw_pvalue=spearmanr(depths,entropy)
    base.update(raw_spearman_descriptive_legacy=float(raw_rho),raw_pvalue_descriptive_legacy=float(raw_pvalue))
    # A single off-shell proxy does not replicate a depth/breadth pattern.
    # Two well populated levels remain valid: no three-level rule is imposed.
    if len(depths)-dominant<2:
        return {**base,"available":False,"reason":"insufficient depth replication: fewer than two proxies outside the dominant tied layer; raw ranks are legacy diagnostics only"}
    rho,pvalue=spearmanr(robust_depth,entropy)
    x=rankdata(robust_depth)-rankdata(robust_depth).mean()
    y=rankdata(entropy)-rankdata(entropy).mean()
    denominator=np.linalg.norm(x)*np.linalg.norm(y)
    rng=np.random.RandomState(seed)
    null=np.asarray([np.dot(rng.permutation(x),y)/denominator for _ in range(repetitions)])
    null_summary=describe(null)
    null_summary.update(p025=float(np.quantile(null,.025)),p975=float(np.quantile(null,.975)))
    return {**base,"available":True,"spearman_depth_vs_nn32_entropy":float(rho),
            "pvalue_descriptive_only":float(pvalue),
            "depth_shuffle_null":null_summary,"depth_shuffle_repetitions":repetitions,
            "depth_shuffle_seed":seed,
            "depth_shuffle_absolute_exceedance_fraction_descriptive":float((1+np.count_nonzero(np.abs(null)>=abs(rho)))/(repetitions+1)),
            "expected_broad_shallow_pattern":"negative correlation; class neighborhood breadth is not independent morphological ground truth",
            "null_limit":"fixed-set shuffled proxy-depth baseline; proxy dependencies and supervised labels preclude an independent morphology significance claim"}


def analyze_proxies(cache,hier_mu,labels,names,meta,centers,*,sample_batches=8):
    proxies=np.asarray(cache["proxy_ball"],dtype=np.float64)
    depths=origin_depth(proxies,.1); punit=unit_directions(proxies); hunit=unit_directions(hier_mu)
    cosine=punit@hunit.T
    angular32=nearest_indices(cosine,32,largest=True)
    actual_dist=poincare_distances(proxies,hier_mu,.1)
    actual32=nearest_indices(actual_dist,32)
    angular=neighborhood_stats(angular32,labels); actual=neighborhood_stats(actual32,labels)
    nn4=angular32[:,:4]
    center_cos=np.clip(punit@centers.T,-1.,1.)
    near_class=center_cos.argmax(1); near_angle=np.degrees(np.arccos(center_cos[np.arange(len(proxies)),near_class]))
    usages={name:np.asarray(cache[name]) for name in ("sample_usage","proxy_usage","sample_effective_usage","proxy_effective_usage") if name in cache}
    for values in usages.values():
        if values.shape != (len(proxies),): raise ValueError("Usage/proxy count mismatch")
    tangent=np.asarray(cache["proxy_tangent"],dtype=np.float64) if "proxy_tangent" in cache else None
    raw_norm=np.linalg.norm(tangent,axis=1) if tangent is not None else None
    rows=[]
    for i in range(len(proxies)):
        row={**meta,"proxy_id":i,"depth":float(depths[i]),"raw_tangent_norm":float(raw_norm[i]) if raw_norm is not None else None,
             "at_depth_cap_tol_1e3":bool(abs(depths[i]-4.6)<=1e-3),"nearest_class_id":int(near_class[i]),"nearest_class_name":names[near_class[i]],
             "nearest_class_center_angle":float(near_angle[i]),"angular_nn4_same_class":bool(np.all(labels[nn4[i]]==labels[nn4[i,0]])),
             "angular_nn32_dominant_class":names[int(angular["dominant_class"][i])],"angular_nn32_purity":float(angular["purity"][i]),
             "angular_nn32_entropy_nats":float(angular["entropy"][i]),"angular_nn32_distinct_classes":int(angular["distinct_classes"][i]),
             "distance_nn32_purity":float(actual["purity"][i]),"distance_nn32_entropy_nats":float(actual["entropy"][i])}
        row.update({name:int(v[i]) for name,v in usages.items()}); rows.append(row)
    null_rng=np.random.RandomState(4780)
    null=[]
    for _ in range(64):
        shuffled=null_rng.permutation(labels)
        null.append(float(np.all(shuffled[nn4]==shuffled[nn4[:,:1]],axis=1).mean()))
    bands=[]
    for low,high in zip((0,2,3,3.5,4,4.599,4.601),(2,3,3.5,4,4.599,4.601,8)):
        mask=(depths>=low)&(depths<high)
        bands.append({**meta,"depth_left":low,"depth_right":high,"proxies":int(mask.sum()),"percent":float(100*mask.mean()),
                      "mean_nn32_class_count":float(angular["distinct_classes"][mask].mean()) if mask.any() else None,
                      "mean_nn32_entropy":float(angular["entropy"][mask].mean()) if mask.any() else None,
                      "mean_nn32_purity":float(angular["purity"][mask].mean()) if mask.any() else None})
    correlations=depth_breadth_association(depths,angular["entropy"])
    summary={"available":True,"proxy_count":len(proxies),"c":.1,"depth":describe(depths),
             "cap_near_count_tol_1e3":int(np.count_nonzero(np.abs(depths-4.6)<=1e-3)),
             "raw_above_tangent_cap_count":int(np.count_nonzero(raw_norm>2.3)) if raw_norm is not None else None,
             "raw_tangent_norm":describe(raw_norm) if raw_norm is not None else None,
             "usage":{name:usage_summary(usages.get(name)) for name in ("sample_usage","proxy_usage","sample_effective_usage","proxy_effective_usage")},
             "nearest_class_center_angles":describe(near_angle),"nearest_class_center_coverage":int(len(np.unique(near_class))),
             "angular_nn4_same_class_proxy_fraction":float(np.all(labels[nn4]==labels[nn4[:,:1]],axis=1).mean()),
             "label_permutation_null_nn4_same_class_proxy_fraction":describe(null),"label_null_repetitions":64,
             "angular_nn32_purity":describe(angular["purity"]),"angular_nn32_entropy":describe(angular["entropy"]),
             "angular_nn4_hotspots":hotspot_summary(nn4,len(labels)),"distance_nn4_hotspots":hotspot_summary(actual32[:,:4],len(labels)),
             "angular_vs_distance_nn4_identical_sets_fraction":float(np.all(np.sort(nn4,axis=1)==np.sort(actual32[:,:4],axis=1),axis=1).mean()),
             "depth_vs_class_breadth":correlations,
             "semantic_limit":"class binding and geometric depth order do not prove independent morphology or a semantic taxonomy",
             "frozen_orders":frozen_orders(hier_mu,proxies,labels,sample_batches=sample_batches)}
    return summary,rows,bands,depths


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--expected-count",type=int,default=9840)
    parser.add_argument("--expected-classes",type=int,default=40)
    parser.add_argument("--sample-batches",type=int,default=8)
    args=parser.parse_args(argv)
    if args.sample_batches<1: parser.error("sample-batches must be positive")
    manifest=json.loads(args.manifest.read_text(encoding="utf-8-sig"))
    names=manifest.get("class_names",CLASS_NAMES)
    if isinstance(names,dict): names=[names.get(str(i),names.get(i)) for i in range(len(names))]
    if len(names)!=args.expected_classes or any(not isinstance(n,str) for n in names):
        raise ValueError("Class names do not match declared canonical label classes")
    observations=manifest["observations"]
    selected,policy=reference_and_pairs(manifest,args.manifest,names)
    args.out.mkdir(parents=True,exist_ok=True)
    expected=None; seen=set(); results=[]
    collections={name:[] for name in ("class_intra_angles","class_pair_angles","class_centroid_angles","angle_histograms","representative_pair_angles","representative_pair_histograms","depth_histograms","depth_cdf","proxy_stats","proxy_depth_bands")}
    started=time.monotonic()
    for spec in observations:
        meta=observation_meta(spec); key=meta["observation_id"]
        if key in seen: raise ValueError("Duplicate observation")
        seen.add(key); cache_path=resolve_cache(args.manifest,spec)
        before=cache_path.stat(); cache_sha=digest(cache_path)
        if spec.get("cache_sha256") and spec["cache_sha256"]!=cache_sha: raise ValueError("Cache SHA differs")
        print(json.dumps({"stage":"loading","observation":key,"elapsed_seconds":round(time.monotonic()-started,1)}),flush=True)
        with np.load(cache_path,allow_pickle=False) as cache:
            mu=np.asarray(cache["native_mu"],dtype=np.float64)
            labels=integer_vector(cache["labels"],"labels")
            ids=integer_vector(cache["canonical_ids"],"canonical_ids")
            if len(mu)!=args.expected_count or len(labels)!=len(mu) or len(ids)!=len(mu) or len(np.unique(ids))!=len(ids):
                raise ValueError("Expected full unique clean pool and matching labels/IDs")
            if not np.array_equal(np.unique(labels),np.arange(args.expected_classes)): raise ValueError("Unexpected label mapping")
            if expected is None: expected=(ids.copy(),labels.copy())
            elif not np.array_equal(ids,expected[0]) or not np.array_equal(labels,expected[1]):
                raise ValueError("Caches do not share canonical object identities and labels")
            c=float(spec["c_native"]); native_d=origin_depth(mu,c)
            unit=unit_directions(mu)
            print(json.dumps({"stage":"angles","observation":key}),flush=True)
            angle_summary,tables,centers=analyze_angles(unit,labels,names,meta,selected)
            for name,rows in zip(("class_intra_angles","class_pair_angles","class_centroid_angles","angle_histograms","representative_pair_angles","representative_pair_histograms"),tables): collections[name].extend(rows)
            domains={"native_whole":{"c":c,"depth":describe(native_d)}}
            hist,cdf=depth_outputs(native_d,c,{**meta,"domain":"native_whole"})
            collections["depth_histograms"].extend(hist); collections["depth_cdf"].extend(cdf)
            proxy={"available":False,"reason":"HIER disabled/no proxy cache; not a zero-depth proxy population"}
            if "proxy_ball" in cache:
                if float(spec.get("c_hier",.1)) != .1 or "hier_mu" not in cache: raise ValueError("HIER proxy audit needs its explicit same-domain whole view")
                hier=np.asarray(cache["hier_mu"],dtype=np.float64)
                if hier.shape!=mu.shape: raise ValueError("HIER/native rows differ")
                # A radial view must preserve directions, never be silently
                # treated as a new learned angular representation.
                if not np.allclose(unit_directions(hier),unit,atol=3e-6,rtol=3e-6): raise ValueError("HIER view changed whole directions")
                hd=origin_depth(hier,.1); domains["hier_whole_view"]={"c":.1,"depth":describe(hd),"view_only_not_native_learning":True}
                hist,cdf=depth_outputs(hd,.1,{**meta,"domain":"hier_whole_view"})
                collections["depth_histograms"].extend(hist); collections["depth_cdf"].extend(cdf)
                print(json.dumps({"stage":"proxies_and_frozen_orders","observation":key}),flush=True)
                proxy,rows,bands,pd=analyze_proxies(cache,hier,labels,names,meta,centers,sample_batches=args.sample_batches)
                collections["proxy_stats"].extend(rows); collections["proxy_depth_bands"].extend(bands)
                hist,cdf=depth_outputs(pd,.1,{**meta,"domain":"proxy"})
                collections["depth_histograms"].extend(hist); collections["depth_cdf"].extend(cdf)
            after=cache_path.stat()
            if before.st_size!=after.st_size or before.st_mtime_ns!=after.st_mtime_ns or digest(cache_path)!=cache_sha:
                raise RuntimeError("Cache changed during read-only analysis")
            results.append({**meta,"native_c":c,"objects":len(mu),"dimensions":mu.shape[1],"cache_sha256":cache_sha,
                            "source_checkpoint_sha256":spec.get("source_checkpoint_sha256",spec.get("checkpoint_sha256")),
                            "canonical_ids_sha256":array_digest(ids),"labels_sha256":array_digest(labels),
                            "domains":domains,"angles":angle_summary,"proxies":proxy})
        for name,rows in collections.items(): csv_write(args.out/(name+".csv"),rows)
        summary={"schema":"clean-embedding-audit85-1","status":"running","mode":"CPU-only immutable cached geometry; no model/test forward, optimizer, backward or BN updates",
                 "depth_histogram":"percent of points in bins centered on multiples of 0.1 physical origin-distance units; first bin starts at zero",
                 "angle_unit":"degrees","angle_histogram_bin_width":.25,"depth_order_tolerances":list(TOLERANCES),
                 "class_names":names,"representative_pairs_policy":policy,"observations":results,
                 "interpretation_limits":["Native c1 whole and c.1 proxies are separate domains; ancestry is evaluated only within c.1 HIER-view/proxy domain", "Frozen deterministic minimax is not a training Gumbel replay", "Pairwise/population statistics do not provide independent multi-seed evidence", "Coordinate axes and proxy IDs are not aligned across independently trained models", "Geometric depth and class binding do not establish morphology/semantic hierarchy"]}
        (args.out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
        print(json.dumps({"stage":"completed_observation","observation":key,"elapsed_seconds":round(time.monotonic()-started,1)}),flush=True)
    summary.update(status="completed",elapsed_seconds=time.monotonic()-started)
    (args.out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
    (args.out/"representative_pairs_policy.json").write_text(json.dumps(policy,ensure_ascii=False,indent=2),encoding="utf-8")


if __name__=="__main__":
    main()
