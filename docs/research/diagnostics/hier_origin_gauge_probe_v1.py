"""CPU origin-gauge control for the distance-only HIER terms.

All8856 cached whole vectors and512 proxies receive a common c1 Mobius left
translation, using Geoopt's unprojected exact-curvature math. Full whole-proxy
and proxy-proxy matrices first pass an FP64 isometry gate. Source FP32 loss
and hard-Gumbel selectors are then replayed with fixed original triplets and
identical noise; finite-precision selector changes are measured, not forbidden.

This is a coordinate/math control without model forward or optimizer update.
It does not assert that the whole classifier, HyCoRe radial term, bounded
proxy parameterization, clipping or AdamW share this symmetry. Translated
proxies can exceed the source.999 safety radius; they remain inside the open
unit ball and are NOT projected, so this is not a realizable training proposal.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

import hier_proxy_objective_probe_v1 as source
from hier_whole_mechanism_v1 import fixed_ancestor_loss
from inter_hierarchy_MN40.hier_proxy_scratch_v5.relations import poincare_distance
from inter_hierarchy_MN40.hier_proxy_scratch_v6.hier_loss import ghhc_loss, expmap0_c1


def translate(points, center):
    from geoopt.manifolds.stereographic import math as geo_math
    result=geo_math.mobius_add(center.to(points)[None,:],points,k=points.new_tensor(-1.),dim=-1)
    if not bool(torch.isfinite(result).all()) or not bool((result.double().square().sum(-1)<1).all()):
        raise RuntimeError("Unprojected Mobius translation left the open unit ball")
    return result


def depth64(points):
    radius=points.detach().double().norm(dim=-1)
    if bool((radius>=1).any()):
        raise RuntimeError("Depth undefined outside ball")
    return 2*torch.atanh(radius)


def matrix_distance(points, proxies, chunk=256):
    rows=[]
    for start in range(0,len(points),chunk):
        rows.append(poincare_distance(points[start:start+chunk],proxies))
    return torch.cat(rows)


def error_stats(reference, current, atol, relative_floor=1e-6):
    error=(current-reference).abs().double()
    relative=error/reference.abs().double().clamp_min(relative_floor)
    if not bool(torch.isfinite(error).all()):
        raise RuntimeError("Nonfinite distance/loss comparison")
    return {"entries":error.numel(),"max_absolute_error":float(error.max()) if error.numel() else 0.,
            "mean_absolute_error":float(error.mean()) if error.numel() else 0.,
            "max_relative_error_floor1e_6":float(relative.max()) if error.numel() else 0.,
            "absolute_tolerance":atol,"absolute_gate_passed":bool((error<=atol).all())}


@torch.no_grad()
def replay(dist_whole_proxy,dist_proxy_proxy,sample_triplets,proxy_triplets,seed):
    generator=torch.Generator(device="cpu").manual_seed(seed)
    sample_loss,sample_stats,sample_details=ghhc_loss(dist_whole_proxy,sample_triplets,
        tau=.1,margin=.1,generator=generator,return_details=True)
    proxy_loss,proxy_stats,proxy_details=ghhc_loss(dist_proxy_proxy,proxy_triplets,
        tau=.1,margin=.1,generator=generator,return_details=True)
    return {"sample_loss":float(sample_loss),"proxy_loss":float(proxy_loss),
            "combined_loss":float(sample_loss+proxy_loss),
            "sample_stats":sample_stats,"proxy_stats":proxy_stats,
            "sample_details":sample_details,"proxy_details":proxy_details}


def domain_masks(details):
    pair,triple=details["pair_proxy_idx"],details["triple_proxy_idx"]
    valid=pair!=triple
    return {"all_draws":torch.ones_like(valid),"noncollision":valid,
            "active_noncollision":valid & (details["per_draw_loss"]>0)}


def order_counts(values, mask, tolerance):
    selected=values[mask]
    return {"draws":selected.numel(),"positive":int((selected>tolerance).sum()),
            "negative":int((selected<-tolerance).sum()),
            "approximately_equal":int((selected.abs()<=tolerance).sum()),
            "mean":float(selected.mean()) if selected.numel() else None}


def fixed_identity_depth_changes(points_before,points_after,proxies_before,proxies_after,
                                 triplets,details,tolerance):
    """Keep baseline ancestry and collision/active masks fixed: pure gauge."""
    p,q=details["pair_proxy_idx"],details["triple_proxy_idx"]
    i,j,k=triplets.unbind(-1)
    d0,d1=depth64(points_before),depth64(points_after)
    p0,p1=depth64(proxies_before),depth64(proxies_after)
    quantities={
        "pair_minus_triple_depth":(p0[p]-p0[q],p1[p]-p1[q]),
        "pair_min_endpoint_gap":(torch.minimum(d0[i],d0[j])-p0[p],torch.minimum(d1[i],d1[j])-p1[p]),
        "triple_min_endpoint_gap":(torch.minimum(torch.minimum(d0[i],d0[j]),d0[k])-p0[q],
                                   torch.minimum(torch.minimum(d1[i],d1[j]),d1[k])-p1[q])}
    result={}
    for domain,mask in domain_masks(details).items():
        result[domain]={}
        count=int(mask.sum())
        for name,(before,after) in quantities.items():
            old_sign=torch.where(before>tolerance,1,torch.where(before<-tolerance,-1,0))
            new_sign=torch.where(after>tolerance,1,torch.where(after<-tolerance,-1,0))
            result[domain][name]={"before":order_counts(before,mask,tolerance),
                "after":order_counts(after,mask,tolerance),
                "order_changed_count":int(((old_sign!=new_sign)&mask).sum()),
                "strict_positive_to_negative":int(((old_sign==1)&(new_sign==-1)&mask).sum()),
                "strict_negative_to_positive":int(((old_sign==-1)&(new_sign==1)&mask).sum()),
                "order_changed_fraction":float((old_sign[mask]!=new_sign[mask]).double().mean()) if count else None}
    return result


def selector_changes(before,after):
    result={}
    for domain,mask in domain_masks(before).items():
        count=int(mask.sum())
        result[domain]={"draws":count}
        for role in ("pair","triple"):
            old,new=before[role+"_proxy_idx"],after[role+"_proxy_idx"]
            result[domain][role+"_changed_count"]=int(((old!=new)&mask).sum())
            result[domain][role+"_changed_fraction"]=float((old[mask]!=new[mask]).double().mean()) if count else None
    return result


def add_counts(accumulator,values):
    """Pool actual draw counts; per-batch means remain in detailed reports."""
    for domain,quantities in values.items():
        target=accumulator.setdefault(domain,{})
        for name,row in quantities.items():
            out=target.setdefault(name,{"draws":0,"before_positive":0,"before_negative":0,"before_equal":0,
                "after_positive":0,"after_negative":0,"after_equal":0,"order_changed_count":0,
                "strict_positive_to_negative":0,"strict_negative_to_positive":0})
            out["draws"]+=row["before"]["draws"]
            for prefix in ("before","after"):
                out[prefix+"_positive"]+=row[prefix]["positive"]
                out[prefix+"_negative"]+=row[prefix]["negative"]
                out[prefix+"_equal"]+=row[prefix]["approximately_equal"]
            for key in ("order_changed_count","strict_positive_to_negative","strict_negative_to_positive"):
                out[key]+=row[key]


def finish_counts(accumulator):
    for quantities in accumulator.values():
        for row in quantities.values():
            denominator=row["draws"]
            for key in ("before_positive","before_negative","after_positive","after_negative","order_changed_count"):
                row[key+"_fraction"]=row[key]/denominator if denominator else None
    return accumulator


def self_test():
    torch.set_num_threads(2)
    # Valid, satisfied distance ordering while fixed-origin depth order reverses.
    whole=torch.tensor([[.55,0.],[.45,0.],[-.45,0.]],dtype=torch.float64)
    proxy=torch.tensor([[.05,0.],[-.12,0.]],dtype=torch.float64)
    center=torch.tensor([.2,0.],dtype=torch.float64)
    moved_whole,moved_proxy=translate(whole,center),translate(proxy,center)
    before=poincare_distance(whole,proxy)
    after=poincare_distance(moved_whole,moved_proxy)
    torch.testing.assert_close(before,after,atol=1e-12,rtol=1e-12)
    torch.testing.assert_close(translate(moved_whole,-center),whole,atol=1e-12,rtol=1e-12)
    triplets=torch.tensor([[0,1,2]])
    pair,triple=torch.tensor([0]),torch.tensor([1])
    loss0=fixed_ancestor_loss(before,triplets,pair,triple)
    loss1=fixed_ancestor_loss(after,triplets,pair,triple)
    torch.testing.assert_close(loss0,loss1,atol=1e-12,rtol=1e-12)
    gap0=depth64(proxy)[0]-depth64(proxy)[1]
    gap1=depth64(moved_proxy)[0]-depth64(moved_proxy)[1]
    assert float(loss0)==0. and gap0<0 and gap1>0
    # Same shapes/order/seeds reproduce the source FP32 stochastic replay.
    small_whole=whole.float()
    small_proxy=torch.tensor([[.05,0.],[-.12,0.],[0.,.1]],dtype=torch.float32)
    sample=triplets.repeat(8,1)
    proxy_triplets=torch.tensor([[0,1,2]]*8)
    dw,dp=poincare_distance(small_whole,small_proxy),poincare_distance(small_proxy)
    first,repeat=replay(dw,dp,sample,proxy_triplets,11),replay(dw,dp,sample,proxy_triplets,11)
    for component in ("sample","proxy"):
        for role in ("pair","triple"):
            assert torch.equal(first[component+"_details"][role+"_proxy_idx"],repeat[component+"_details"][role+"_proxy_idx"])
    print(json.dumps({"self_test":"passed","fixed_distance_hinge_loss":float(loss0),
        "pair_minus_triple_depth_before":float(gap0),"pair_minus_triple_depth_after":float(gap1),
        "checks":["Geoopt unprojected c1 isometry","inverse translation","fixed loss same/depth order reversed",
                  "source FP32 shared-generator replay repeat"]}),flush=True)


def arguments():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache",type=Path)
    p.add_argument("--cache-metadata",type=Path)
    p.add_argument("--checkpoint",type=Path)
    p.add_argument("--run-dir",type=Path)
    p.add_argument("--batches",type=int,default=4)
    p.add_argument("--seed",type=int,default=22)
    p.add_argument("--plan-epoch",type=int,default=0)
    p.add_argument("--threads",type=int,default=2)
    p.add_argument("--hotspot-ids",default="2883,6009,7985,9123")
    p.add_argument("--fp64-distance-atol",type=float,default=2e-8)
    p.add_argument("--fp32-distance-atol",type=float,default=2e-3)
    p.add_argument("--fp32-loss-atol",type=float,default=2e-3)
    p.add_argument("--depth-order-tolerance",type=float,default=1e-3)
    p.add_argument("--self-test",action="store_true")
    args=p.parse_args()
    if not args.self_test:
        if not all((args.cache,args.checkpoint,args.run_dir)):
            p.error("cache, matching checkpoint and new run-dir are required")
        if not 1<=args.batches<=16 or args.seed!=22 or args.plan_epoch<0 or args.threads<1:
            p.error("Require seed22/batches1..16 and nonnegative epoch/positive threads")
        if min(args.fp64_distance_atol,args.fp32_distance_atol,args.fp32_loss_atol,args.depth_order_tolerance)<=0:
            p.error("Numerical tolerances must be positive")
        args.hotspot_ids=[int(s) for s in args.hotspot_ids.split(",") if s.strip()]
        if not args.hotspot_ids:
            p.error("Provide at least one hotspot ID to define translation direction")
        args.cache_metadata=args.cache_metadata or args.cache.with_name("summary.json")
    return args


@torch.no_grad()
def main():
    args=arguments()
    if args.self_test:
        self_test()
        return
    torch.set_num_threads(args.threads)
    args.run_dir.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter()
    manifest={"status":"running","format":"hier-origin-gauge-probe-v1","device":"cpu",
        "diagnostic_only":True,"optimizer_updates":0,"backbone_forwards":0,
        "started_utc":datetime.now(timezone.utc).isoformat(),"script_sha256":source.sha_file(__file__),
        "utility_script_sha256":source.sha_file(Path(__file__).with_name("hier_proxy_objective_probe_v1.py")),
        "configuration":{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        "semantics":["Coordinate control of HIER distances only; source fixed triplets and common pair/triple Gumbel noise.",
            "FP64 isometry is gated before any FP32 replay claim.",
            "No transformed mining, no backbone/BN/CE/intra/optimizer evaluation.",
            "Origin-depth comparisons hold original selected proxy IDs and domain masks fixed.",
            "Source.999 proxy cap is not reapplied; beyond-cap translated vectors are flagged.",
            "HyCoRe fixes a task-relevant origin; this does not assert a whole-objective symmetry.",
            "FP32 selector changes are measured separately from pure-gauge depth changes.",
            "Depth order uses an explicit1e-3 default tie tolerance to avoid boundary roundoff rankings."]}
    try:
        manifest["commit"]=subprocess.check_output(["git","rev-parse","HEAD"],cwd=source.REPO,text=True).strip()
    except (OSError,subprocess.CalledProcessError):
        manifest["commit"]=None
    source.json_write(args.run_dir/"manifest.json",manifest)
    try:
        whole,labels,ids,tangent,saved,identity=source.validate_inputs(args.cache,args.checkpoint,args.cache_metadata)
        proxies=expmap0_c1(tangent).detach()
        whole64,proxy64=whole.double(),proxies.double()
        by_id={int(identity):row for row,identity in enumerate(ids)}
        if any(identity not in by_id for identity in args.hotspot_ids):
            raise RuntimeError("Hotspot ID outside matched train cache")
        hot=whole64[[by_id[identity] for identity in args.hotspot_ids]]
        direction=(hot/hot.norm(dim=-1,keepdim=True).clamp_min(1e-15)).mean(0)
        direction=direction/direction.norm().clamp_min(1e-15)
        if float(direction.norm())<.99:
            raise RuntimeError("Hotspot direction is degenerate")
        random=torch.randn(whole.shape[1],generator=torch.Generator().manual_seed(args.seed+900001),dtype=torch.float64)
        random=random/random.norm()
        centers={"hotspot_positive_r0_1":.1*direction,"hotspot_negative_r0_2":-.2*direction,
                 "random_r0_1":.1*random}
        plans,sampler_identity=source.make_plans(labels,ids,args.seed,args.batches,args.plan_epoch)
        source.json_write(args.run_dir/"batch_plans.json",{"sampler":sampler_identity,"plans":[p["metadata"] for p in plans]})
        original_wp64=matrix_distance(whole64,proxy64)
        original_pp64=poincare_distance(proxy64)
        original_wp32=matrix_distance(whole,proxies)
        original_pp32=poincare_distance(proxies)
        original={}
        for step,plan in enumerate(plans):
            at=plan["rows"]
            seed=args.seed+args.plan_epoch*10000019+step*10007
            mined_sample,mined_proxy=source.mine_for_mode(whole[at],labels[at],ids[at],proxies,"original",seed,.98)
            sample_triplets,proxy_triplets=mined_sample["triplets"],mined_proxy["triplets"]
            draw=replay(original_wp32[at],original_pp32,sample_triplets,proxy_triplets,seed)
            original[step]={"at":at,"seed":seed,"sample_triplets":sample_triplets,"proxy_triplets":proxy_triplets,
                "replay":draw,"sample_mining":mined_sample["stats"],"proxy_mining":mined_proxy["stats"]}
        manifest.update(identity=identity,sampler_identity=sampler_identity,
            distance_gate_domain="full8856x512 whole-proxy and512x512 proxy-proxy",
            batch_plan_sha256=source.sha_file(args.run_dir/"batch_plans.json"),
            translations={name:{"center_norm":float(c.norm()),"center_sha256":source.sha_tensor(c)} for name,c in centers.items()})
        source.json_write(args.run_dir/"manifest.json",manifest)
        results={}
        for name,center in centers.items():
            moved_whole64,moved_proxy64=translate(whole64,center),translate(proxy64,center)
            inverse_whole,inverse_proxy=translate(moved_whole64,-center),translate(moved_proxy64,-center)
            torch.testing.assert_close(inverse_whole,whole64,atol=2e-12,rtol=2e-12)
            torch.testing.assert_close(inverse_proxy,proxy64,atol=2e-12,rtol=2e-12)
            moved_wp64,moved_pp64=matrix_distance(moved_whole64,moved_proxy64),poincare_distance(moved_proxy64)
            fp64_gates={"whole_proxy":error_stats(original_wp64,moved_wp64,args.fp64_distance_atol),
                        "proxy_proxy":error_stats(original_pp64,moved_pp64,args.fp64_distance_atol)}
            if not all(row["absolute_gate_passed"] for row in fp64_gates.values()):
                raise RuntimeError("FP64 isometry gate failed for "+name)
            moved_whole32,moved_proxy32=translate(whole,center.float()),translate(proxies,center.float())
            moved_wp32,moved_pp32=matrix_distance(moved_whole32,moved_proxy32),poincare_distance(moved_proxy32)
            fp32_gates={"whole_proxy":error_stats(original_wp32,moved_wp32,args.fp32_distance_atol),
                        "proxy_proxy":error_stats(original_pp32,moved_pp32,args.fp32_distance_atol)}
            fp32_pass=all(row["absolute_gate_passed"] for row in fp32_gates.values())
            translation_report={"FP64_isometry_gates":fp64_gates,"FP32_distance_gates":fp32_gates,
                "FP32_isometry_tolerance_passed":fp32_pass,"inverse_FP64_verified":True,
                "direct_FP32_vs_FP64_cast_coordinate_max_error":{
                    "whole":float((moved_whole32-moved_whole64.float()).abs().max()),
                    "proxy":float((moved_proxy32-moved_proxy64.float()).abs().max())},
                "translated_proxy_beyond_source_0_999_count":int((moved_proxy64.norm(dim=-1)>.999).sum()),
                "translated_whole_depth_mean":float(depth64(moved_whole64).mean()),
                "translated_proxy_depth_mean":float(depth64(moved_proxy64).mean()),"batches":[]}
            aggregated={"sample":{},"proxy":{}}
            for step,reference in original.items():
                at=reference["at"]
                before=reference["replay"]
                after=replay(moved_wp32[at],moved_pp32,reference["sample_triplets"],reference["proxy_triplets"],reference["seed"])
                batch={"batch":step,"sample_mining_fixed":reference["sample_mining"],
                    "proxy_mining_fixed":reference["proxy_mining"],"same_noise_seed":reference["seed"],
                    "sample_triplets_sha256":source.sha_tensor(reference["sample_triplets"]),
                    "proxy_triplets_sha256":source.sha_tensor(reference["proxy_triplets"]),
                    "losses_before":{k:before[k] for k in ("sample_loss","proxy_loss","combined_loss")},
                    "losses_after":{k:after[k] for k in ("sample_loss","proxy_loss","combined_loss")},
                    "FP32_loss_absolute_deltas":{k:abs(after[k]-before[k]) for k in ("sample_loss","proxy_loss","combined_loss")},
                    "FP32_selector_changes":{},"fixed_identity_FP64_depth_changes":{},"FP64_fixed_ancestor_loss_deltas":{}}
                batch["FP32_loss_tolerance_passed"]=all(value<=args.fp32_loss_atol for value in batch["FP32_loss_absolute_deltas"].values())
                for component in ("sample","proxy"):
                    triplets=reference[component+"_triplets"]
                    details=before[component+"_details"]
                    db0,db1=(original_wp64[at],moved_wp64[at]) if component=="sample" else (original_pp64,moved_pp64)
                    p,q=details["pair_proxy_idx"],details["triple_proxy_idx"]
                    fixed_loss0=fixed_ancestor_loss(db0,triplets,p,q)
                    fixed_loss1=fixed_ancestor_loss(db1,triplets,p,q)
                    delta=float((fixed_loss1-fixed_loss0).abs())
                    if delta>6*args.fp64_distance_atol:
                        raise RuntimeError("FP64 fixed ancestry loss invariance failed")
                    batch["FP64_fixed_ancestor_loss_deltas"][component]=delta
                    batch["FP32_selector_changes"][component]=selector_changes(details,after[component+"_details"])
                    points0,points1=(whole64[at],moved_whole64[at]) if component=="sample" else (proxy64,moved_proxy64)
                    depth_changes=fixed_identity_depth_changes(points0,points1,proxy64,moved_proxy64,triplets,details,args.depth_order_tolerance)
                    batch["fixed_identity_FP64_depth_changes"][component]=depth_changes
                    add_counts(aggregated[component],depth_changes)
                translation_report["batches"].append(batch)
            translation_report["pooled_fixed_identity_depth_changes"]={k:finish_counts(v) for k,v in aggregated.items()}
            translation_report["FP32_replay_tolerance_passed"]=fp32_pass and all(b["FP32_loss_tolerance_passed"] for b in translation_report["batches"])
            translation_report["claim_scope"]="FP64 coordinate isometry verified; FP32 source replay within tolerance" if translation_report["FP32_replay_tolerance_passed"] else "FP64 mathematical control only; FP32 numerical replay did NOT meet tolerance"
            results[name]=translation_report
            source.json_write(args.run_dir/(name+".json"),translation_report)
            print(json.dumps({"translation":name,"FP64_gate":True,"FP32_replay_gate":translation_report["FP32_replay_tolerance_passed"],
                              "proxy_beyond_source_cap":translation_report["translated_proxy_beyond_source_0_999_count"]}),flush=True)
            del moved_wp64,moved_pp64,moved_wp32,moved_pp32,moved_whole32,moved_proxy32
        for key,path in (("cache",args.cache),("checkpoint",args.checkpoint),("cache_metadata",args.cache_metadata)):
            if source.sha_file(path)!=identity["hashes"][key]:
                raise RuntimeError("Source file changed: "+key)
        manifest.update(status="completed",finished_utc=datetime.now(timezone.utc).isoformat(),
                        wall_seconds=time.perf_counter()-started,source_files_unchanged=True)
        source.json_write(args.run_dir/"summary.json",{"manifest":manifest,"translations":results,
            "interpretation":"Distance-only HIER objectives do not explicitly anchor ancestry to the fixed origin. This coordinate control does not establish symmetry of the full HyCoRe model/training or validate morphology."})
        source.json_write(args.run_dir/"manifest.json",manifest)
    except BaseException as error:
        manifest.update(status="failed",error=repr(error),wall_seconds=time.perf_counter()-started)
        source.json_write(args.run_dir/"manifest.json",manifest)
        raise


if __name__=="__main__":
    main()
