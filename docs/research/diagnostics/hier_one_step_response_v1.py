"""One-step restored-momentum HyCoRe/HIER counterfactuals on one GPU.

Two sequential local32 train-mode forwards and low-memory VJPs reproduce a
global64 mean gradient. Each batch starts from one complete V6-H20 checkpoint.
Base, base+.5sampleST and base+.5sample_fixed receive independent copies of
the exact checkpoint model/optimizer/BN/RNG state and each takes one clipped
RiemannianSGD update. Proxies remain fixed. Clean1024 eval readout uses the OLD
checkpoint BN buffers in every arm, isolating parameter-step responses.

This is a conditional one-step diagnostic, not historical DDP/worker replay,
long-run training, official test evaluation, or independent morphology evidence.
No source checkpoint or data file is written.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

# The adjacent helper script is imported read-only; no shared helper is edited.
import hier_whole_mechanism_v1 as mech

ARMS = ("base", "base_sample_st", "base_sample_fixed")
SOURCE_FORMAT = "hycore-hier-v6-h20-selfk300-1"


def state_digest(value):
    """Semantic hash independent of tensor device and pickle serialization."""
    result = hashlib.sha256()
    def visit(item):
        if isinstance(item, torch.Tensor):
            result.update(b"tensor")
            result.update(mech.array_hash(item).encode())
        elif isinstance(item, np.ndarray):
            result.update(b"ndarray")
            result.update(mech.array_hash(item).encode())
        elif isinstance(item, dict):
            result.update(b"dict")
            for key in sorted(item, key=lambda key: (type(key).__name__, repr(key))):
                visit(key)
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            result.update(type(item).__name__.encode())
            for child in item:
                visit(child)
        else:
            result.update((type(item).__name__ + ":" + repr(item)).encode())
    visit(value)
    return result.hexdigest()


def validate_optimizer_layout(named, saved):
    """Match every saved momentum tensor to canonical V6 parameter order."""
    layout = [(group_index,identity) for group_index,group in enumerate(saved["param_groups"])
              for identity in group["params"]]
    identities = [identity for _,identity in layout]
    if len(layout)!=len(named) or identities!=list(range(len(named))):
        raise RuntimeError("Expected canonical serialized V6 optimizer parameter order")
    if set(saved["state"])!=set(identities):
        raise RuntimeError("Every trainable parameter must retain historical optimizer state")
    report = []
    for (group_index,identity),(name,parameter) in zip(layout,named):
        state = saved["state"][identity]
        momentum = state.get("momentum_buffer")
        if not isinstance(momentum,torch.Tensor) or momentum.shape!=parameter.shape or momentum.dtype!=parameter.dtype:
            raise RuntimeError("Momentum shape/dtype mismatch at "+name)
        mech.finite_tensor(momentum,"historical momentum "+name)
        report.append({"parameter":name,"group":group_index,"serialized_id":identity,
                       "shape":list(parameter.shape),"dtype":str(parameter.dtype),
                       "momentum_sha256":mech.array_hash(momentum),
                       "momentum_norm":float(momentum.detach().double().norm())})
    return report


def build_optimizer(parameters, saved):
    import geoopt
    optimizer = geoopt.optim.RiemannianSGD(parameters, lr=.1, momentum=.9, weight_decay=2e-4)
    optimizer.load_state_dict(copy.deepcopy(saved))
    if state_digest(optimizer.state_dict()) != state_digest(saved):
        raise RuntimeError("RiemannianSGD complete optimizer state did not restore exactly")
    return optimizer


def assign_vector_gradient(named, vector):
    count = sum(parameter.numel() for _, parameter in named)
    if vector.numel() != count:
        raise RuntimeError("VJP/optimizer parameter vector layout mismatch")
    offset = 0
    for _, parameter in named:
        stop = offset + parameter.numel()
        parameter.grad = vector[offset:stop].reshape(parameter.shape).to(parameter).clone()
        offset = stop


def clipped_step(backbone, named, gradient, optimizer):
    """The production order: norm1 clip, then one actual RiemannianSGD step."""
    parameters = [parameter for _, parameter in named]
    optimizer.zero_grad(set_to_none=True)
    assign_vector_gradient(named, gradient)
    for parameter in parameters:
        mech.finite_tensor(parameter.grad, "assigned parameter gradient")
    norm_before = torch.nn.utils.clip_grad_norm_(parameters, 1.)
    norm_after = torch.stack([parameter.grad.detach().float().norm().square() for parameter in parameters]).sum().sqrt()
    if not bool(torch.isfinite(norm_before)) or not bool(torch.isfinite(norm_after)) or float(norm_after) > 1.00001:
        raise RuntimeError("Nonfinite or excessive clipped parameter norm")
    before = torch.cat([parameter.detach().flatten().cpu() for parameter in parameters])
    optimizer.step()
    after = torch.cat([parameter.detach().flatten().cpu() for parameter in parameters])
    for parameter in parameters:
        mech.finite_tensor(parameter.detach(), "one-step updated parameter")
    return {
        "preclip_norm": float(norm_before), "postclip_norm": float(norm_after),
        "clip_scale": min(1., 1. / (float(norm_before) + 1e-6)),
        "clip_applied": bool(norm_before > 1.),
        "actual_parameter_displacement_norm": float((after - before).norm()),
        "gradient_assignment_sha256": mech.array_hash(gradient),
        "parameter_before_sha256": mech.array_hash(before),
        "parameter_after_sha256": mech.array_hash(after),
    }


@torch.no_grad()
def clean_readout(backbone, clean_cloud, old_buffers, common_rng, device):
    """Same clean1024 cloud; no crop/alias, fixed OLD BN and common eval RNG."""
    mech.restore_buffers(backbone, old_buffers)
    mech.restore_selected_rng(common_rng, device)
    backbone.eval()
    buffer_hash = state_digest(mech.snapshot_buffers(backbone))
    rows = {"mu": [], "logits": []}
    for start in (0, 32):
        mu, logits = backbone(clean_cloud[start:start+32].transpose(1, 2).contiguous())
        mech.finite_tensor(mu, "readout whole")
        mech.finite_tensor(logits, "readout logits")
        if not bool((mu.double().square().sum(-1) < 1).all()):
            raise RuntimeError("Readout outside c1 ball")
        rows["mu"].append(mu.detach().cpu())
        rows["logits"].append(logits.detach().cpu())
    if state_digest(mech.snapshot_buffers(backbone)) != buffer_hash:
        raise RuntimeError("Eval readout changed a BN buffer")
    return {key: torch.cat(value) for key, value in rows.items()}


def distribution(value):
    value = torch.as_tensor(value).detach().double().reshape(-1)
    if not value.numel():
        return {"n":0, "mean":None, "median":None, "min":None, "max":None,
                "positive_fraction":None, "negative_fraction":None}
    mech.finite_tensor(value, "readout response")
    return {"n":value.numel(), "mean":float(value.mean()), "median":float(value.median()),
            "min":float(value.min()), "max":float(value.max()),
            "positive_fraction":float((value>0).double().mean()),
            "negative_fraction":float((value<0).double().mean())}


def response_arrays(before, after, labels):
    mu0, mu1 = before["mu"].double(), after["mu"].double()
    r0, r1 = mu0.norm(dim=-1), mu1.norm(dim=-1)
    d0, d1 = 2*torch.atanh(r0), 2*torch.atanh(r1)
    u0 = mu0/r0[:,None].clamp_min(1e-15)
    u1 = mu1/r1[:,None].clamp_min(1e-15)
    angle = torch.rad2deg(torch.acos((u0*u1).sum(-1).clamp(-1,1)))
    hyperbolic = 2*torch.asinh((mu1-mu0).norm(dim=-1) / torch.sqrt((1-r0.square())*(1-r1.square())))
    pred0, pred1 = before["logits"].argmax(-1), after["logits"].argmax(-1)
    labels = torch.as_tensor(labels)
    return {"depth_before":d0, "depth_after":d1, "depth_delta":d1-d0,
            "radius_delta":r1-r0, "angular_movement_degrees":angle,
            "hyperbolic_movement":hyperbolic,
            "prediction_changed":pred1!=pred0, "correct_before":pred0==labels,
            "correct_after":pred1==labels, "prediction_before":pred0, "prediction_after":pred1}


def response_summary(arrays, labels, ids, hotspot_ids):
    labels, ids = torch.as_tensor(labels), torch.as_tensor(ids)
    hotspot = torch.tensor([int(identity) in hotspot_ids for identity in ids], dtype=torch.bool)
    groups = {"all64":torch.ones(len(ids),dtype=torch.bool), "hotspot":hotspot, "other":~hotspot}
    groups.update({"class/"+mech.NAMES[int(c)]:labels==c for c in labels.unique()})
    result = {}
    for group, mask in groups.items():
        entry = {key:distribution(arrays[key][mask]) for key in
                 ("depth_before","depth_after","depth_delta","radius_delta",
                  "angular_movement_degrees","hyperbolic_movement")}
        entry["prediction_changed_count"] = int(arrays["prediction_changed"][mask].sum())
        entry["correct_before_count"] = int(arrays["correct_before"][mask].sum())
        entry["correct_after_count"] = int(arrays["correct_after"][mask].sum())
        entry["positions"] = int(mask.sum())
        result[group] = entry
    return result


def validate_source(saved):
    if saved.get("format") != SOURCE_FORMAT or saved.get("model_selection_only",False):
        raise RuntimeError("A complete V6-H20 checkpoint is required")
    required = {"net","proxy","optimizer","scheduler","rank_states","train_ids",
                "validation_ids","split_sha256","training_config","completed_epochs"}
    if not required.issubset(saved) or saved.get("diagnostic_only",False):
        raise RuntimeError("Missing complete production model/optimizer/rank state")
    config = saved["training_config"]
    expected = {"seed":22,"epochs":300,"steps_per_epoch":200,"c":1,"D":256,
                "sample_K":20,"proxy_K":20,"P":512,"T":50,"margin":.1,"tau":.1,
                "self_negative":False,"lambda_hier_after_warmup":.5,
                "architecture":"Hype_pointMLP","num_classes":40,"global_batch":64,
                "world_size":2,"classes_per_rank":16,"instances_per_class":2,
                "part_overwrites_whole":True,"FPS_first_stage":512,
                "extra_HIER_tangent_cap":False,"HIER_backward_hook":False,
                "precision":"FP32","label_smoothing":.2,
                "alpha_contrastive":.01,"alpha_radial":.01,
                "contrastive_margin":4.,"radial_margin":"1000 / child_count",
                "whole_count_range":[800,1024],"child_count_range":[200,600],
                "BN":"ordinary local32 training; child and whole update; rank0 buffer broadcast"}
    if any(config.get(key)!=value for key,value in expected.items()):
        raise RuntimeError("Checkpoint is not the reviewed V6 H20 protocol")
    completed = int(saved["completed_epochs"])
    if int(saved.get("epoch",-1)) != completed or not 20 <= completed <= 300:
        raise RuntimeError("Unexpected complete-checkpoint epoch identity")
    if len(saved["rank_states"])!=2 or [s.get("rank") for s in saved["rank_states"]] != [0,1]:
        raise RuntimeError("Both ordered rank states are required")
    scheduler = saved["scheduler"]
    if scheduler.get("last_epoch")!=completed or scheduler.get("T_max")!=300:
        raise RuntimeError("Checkpoint scheduler identity is inconsistent")
    expected_optimizer={"name":"RiemannianSGD","lr":.1,"min_lr":.005,
                        "momentum":.9,"weight_decay":2e-4,"norm_clip":1.}
    if config.get("model_optimizer")!=expected_optimizer:
        raise RuntimeError("Source model optimizer/config clipping differs from V6")
    groups = saved["optimizer"]["param_groups"]
    if not groups or not saved["optimizer"]["state"] or any(
            group.get("momentum")!=.9 or group.get("weight_decay")!=2e-4 or group.get("lr",0)<=0
            for group in groups):
        raise RuntimeError("Original RiemannianSGD momentum state/settings required")


def self_test():
    """CPU test exercises real RSGD, inherited momentum, clipping and resets."""
    import geoopt
    torch.set_num_threads(2)
    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.euclidean = torch.nn.Parameter(torch.tensor([.4,-.3,.2]))
            self.ball = geoopt.ManifoldParameter(torch.tensor([[.12,.08]]),manifold=geoopt.PoincareBall(c=1.))
    toy=Toy()
    named=list(toy.named_parameters())
    optimizer=geoopt.optim.RiemannianSGD(toy.parameters(),lr=.017,momentum=.9,weight_decay=2e-4)
    historical=torch.tensor([.3,-.2,.1,.4,-.5])
    for _ in range(2):
        assign_vector_gradient(named,historical)
        optimizer.step()
    source_net=copy.deepcopy(toy.state_dict())
    source_opt=copy.deepcopy(optimizer.state_dict())
    source_hash=state_digest(source_opt)
    layout=validate_optimizer_layout(named,source_opt)
    assert len(layout)==len(named)
    corrupted=copy.deepcopy(source_opt)
    first_identity=corrupted["param_groups"][0]["params"][0]
    corrupted["state"][first_identity]["momentum_buffer"]=torch.zeros(7)
    try:
        validate_optimizer_layout(named,corrupted)
    except RuntimeError:
        pass
    else:
        raise AssertionError("Corrupted momentum shape was accepted")
    gradients={"base":torch.tensor([2.,-1.,.5,.3,-.7]),
               "base_sample_st":torch.tensor([2.8,-.8,.2,.9,-.6]),
               "base_sample_fixed":torch.tensor([2.1,-.9,.6,.2,-.5])}
    outputs={}
    for arm in ARMS:
        toy.load_state_dict(source_net)
        restored=build_optimizer([p for _,p in named],source_opt)
        row=clipped_step(toy,named,gradients[arm],restored)
        assert row["postclip_norm"]<=1.00001
        outputs[arm]=torch.cat([p.detach().flatten() for _,p in named]).clone()
    toy.load_state_dict(source_net)
    repeat=build_optimizer([p for _,p in named],source_opt)
    clipped_step(toy,named,gradients["base"],repeat)
    torch.testing.assert_close(torch.cat([p.detach().flatten() for _,p in named]),outputs["base"],atol=0,rtol=0)
    # Independent direct production-order update is the reference.
    toy.load_state_dict(source_net)
    direct=build_optimizer([p for _,p in named],source_opt)
    assign_vector_gradient(named,gradients["base"])
    torch.nn.utils.clip_grad_norm_(toy.parameters(),1.)
    direct.step()
    torch.testing.assert_close(torch.cat([p.detach().flatten() for _,p in named]),outputs["base"],atol=0,rtol=0)
    toy.load_state_dict(source_net)
    fresh=geoopt.optim.RiemannianSGD(toy.parameters(),lr=.017,momentum=.9,weight_decay=2e-4)
    clipped_step(toy,named,gradients["base"],fresh)
    assert not torch.equal(torch.cat([p.detach().flatten() for _,p in named]),outputs["base"])
    assert source_hash==state_digest(source_opt)
    assert not torch.equal(outputs["base"],outputs["base_sample_st"])
    assert not torch.equal(outputs["base_sample_fixed"],outputs["base_sample_st"])
    before={"mu":torch.tensor([[.2,0.],[0.,.3]]),"logits":torch.tensor([[2.,0.],[0.,2.]])}
    after={"mu":torch.tensor([[.3,0.],[0.,.2]]),"logits":before["logits"].clone()}
    response=response_arrays(before,after,torch.tensor([0,1]))
    assert response["depth_delta"][0]>0 and response["depth_delta"][1]<0
    assert not response["prediction_changed"].any()
    print({"self_test":"passed","checks":["real RSGD including manifold parameter",
          "full momentum restore exact","independent arm reset repeat exact",
          "direct norm1-clip/SGD match","fresh optimizer demonstrably differs",
          "source optimizer immutable","per-parameter momentum shape/order gate","signed actual-depth response"]},flush=True)


def arguments():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint",type=Path)
    parser.add_argument("--data-dir",type=Path)
    parser.add_argument("--run-dir",type=Path)
    parser.add_argument("--gpu",type=int,default=0)
    parser.add_argument("--batches",type=int,default=2)
    parser.add_argument("--seed",type=int,default=22)
    parser.add_argument("--plan-epoch",type=int,default=0)
    parser.add_argument("--input-mode",choices=("clean","augmented"),default="augmented")
    parser.add_argument("--hotspot-ids",default="")
    parser.add_argument("--self-test",action="store_true")
    args=parser.parse_args()
    if not args.self_test:
        if not all((args.checkpoint,args.data_dir,args.run_dir)):
            parser.error("checkpoint, data-dir and new run-dir are required")
        if not 1<=args.batches<=16 or args.seed!=22 or args.plan_epoch<0 or args.gpu<0:
            parser.error("Require seed22, batches1..16, nonnegative plan-epoch/gpu")
        args.hotspot_ids=[int(s) for s in args.hotspot_ids.split(",") if s.strip()]
        if args.hotspot_ids and (len(args.hotspot_ids)<2 or len(set(args.hotspot_ids))!=len(args.hotspot_ids)):
            parser.error("Hotspot IDs must contain at least2 distinct IDs")
    return args


def main():
    args=arguments()
    if args.self_test:
        self_test()
        return
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    if torch.distributed.is_initialized():
        raise RuntimeError("Single-process diagnostic only")
    args.run_dir.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter()
    manifest={"status":"starting","diagnostic_only":True,"optimizer_updates":0,
              "checkpoint":str(args.checkpoint),"checkpoint_sha256":mech.digest(args.checkpoint),
              "script_sha256":mech.digest(__file__),"started_utc":datetime.now(timezone.utc).isoformat(),
              "command":[sys.executable,*sys.argv],"logical_gpu":args.gpu,
              "CUDA_VISIBLE_DEVICES":os.environ.get("CUDA_VISIBLE_DEVICES"),
              "arms":list(ARMS),"batches":args.batches,"hotspot_ids":args.hotspot_ids,
              "new_HIER_warmup_epochs":0,"sample_weight":.5,"proxy_updates":0,
              "readout_BN":"checkpoint OLD rank0 buffers, identical in all arms",
              "readout_input":"fixed original first1024 clean whole points",
              "gradient_condition":"source_train: part then overwritten whole; ordinary local32 BN",
              "limitations":["Conditional independent one-step panels, not a rolling trajectory.",
                "Input augmentation and local32 RNG are fixed diagnostic streams, not historical worker replay.",
                "Global64 mean VJP is summed over two local32 groups without extra DDP factor.",
                "Component VJPs are added in FP32; this does not promise bitwise historical backward.",
                "Sample ST and fixed share identical mining/Gumbel choices and scalar hinges.",
                "ST-base/fixed-base differences include the intervention's change to global clipping.",
                "Readout restores OLD BN, isolating parameter response rather than full training BN drift.",
                "Saved optimizer LR is the next continuation-step LR after the checkpoint epoch.",
                "The base arm retains historical H20 model momentum; it is a one-step objective switch, not a from-scratch B0.",
                "Per-arm repeated no-update readout quantifies baseline numerical noise before interpreting step responses.",
                "No official test or independent morphology evidence."]}
    try:
        manifest["commit"]=subprocess.check_output(["git","rev-parse","HEAD"],cwd=mech.ROOT,text=True).strip()
    except (OSError,subprocess.CalledProcessError):
        manifest["commit"]=None
    mech.save_json(args.run_dir/"manifest.json",manifest)
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("PointMLP requires CUDA; use --self-test for CPU-only optimizer checks")
        torch.cuda.set_device(args.gpu)
        device=torch.device("cuda",args.gpu)
        from models.pointmlp import Hype_pointMLP
        from inter_hierarchy_MN40.hier_proxy_v2_online.train import load_shards
        from inter_hierarchy_MN40.hier_proxy_scratch_v3.train import freeze_curvature
        from inter_hierarchy_MN40.hier_proxy_scratch_v5.sampler import SourceClassBatchSampler
        from inter_hierarchy_MN40.hier_proxy_scratch_v5.distributed import shared_crop_counts
        from inter_hierarchy_MN40.hier_proxy_scratch_v6.hier_loss import HIERLoss
        saved=torch.load(args.checkpoint,map_location="cpu",weights_only=False)
        validate_source(saved)
        source_net=copy.deepcopy(saved["net"])
        source_optimizer=copy.deepcopy(saved["optimizer"])
        source_optimizer_sha=state_digest(source_optimizer)
        train_ids=np.asarray(saved["train_ids"],dtype=np.int64)
        mech.seed_selected(args.seed,device)
        backbone=Hype_pointMLP().to(device)
        backbone.load_state_dict(source_net,strict=True)
        freeze_curvature(backbone)
        named=[(name,p) for name,p in backbone.named_parameters() if p.requires_grad]
        momentum_layout=validate_optimizer_layout(named,source_optimizer)
        old_buffers=mech.snapshot_buffers(backbone)
        rank0=saved["rank_states"][0]
        if state_digest(old_buffers)!=state_digest(rank0["BN_buffers"]):
            raise RuntimeError("Source net/rank0 BN buffer identity mismatch")
        original_rng=mech.capture_selected_rng(device)
        source_rng=rank0["rng"]
        common_rng={"python":source_rng["python"],"numpy":source_rng["numpy"],
                    "torch":source_rng["torch"],"cuda":source_rng["cuda"][0]}
        source_model_sha=mech.module_hash(backbone)
        proxy=HIERLoss(seed=args.seed).to(device)
        proxy.load_state_dict(saved["proxy"],strict=True)
        proxy_hash=mech.module_hash(proxy)
        with torch.no_grad():
            proxy_points=proxy.proxies().detach()
        points,labels,shards=load_shards(args.data_dir,"train")
        if len(train_ids)!=8856 or len(np.unique(train_ids))!=8856 or len(labels)!=9840:
            raise RuntimeError("Expected unique V6 train8856 from ModelNet40 train9840")
        if np.any(train_ids<0) or np.any(train_ids>=len(labels)):
            raise RuntimeError("Train IDs outside dataset")
        shard_hashes={name:mech.digest(args.data_dir/name) for name in shards}
        sampler=SourceClassBatchSampler(labels[train_ids],seed=22,steps=200)
        momentum_elements=sum(state["momentum_buffer"].numel() for state in source_optimizer["state"].values() if "momentum_buffer" in state)
        manifest.update(status="running",checkpoint_epoch=int(saved["completed_epochs"]),
                        checkpoint_source_commit=saved.get("commit"),training_config=saved["training_config"],
                        scheduler_identity=saved["scheduler"],optimizer_identity={
                            "sha256":source_optimizer_sha,"algorithm":"geoopt.RiemannianSGD",
                            "param_groups":[{k:v for k,v in group.items() if k!="params"} for group in source_optimizer["param_groups"]],
                            "historical_state_parameter_count":len(source_optimizer["state"]),
                            "historical_momentum_elements":momentum_elements,
                            "per_parameter_momentum_layout":momentum_layout},
                        source_model_sha256=source_model_sha,proxy_sha256=proxy_hash,
                        train_ids_sha256=mech.array_hash(train_ids),split_sha256=saved["split_sha256"],
                        input_shard_sha256=shard_hashes,gpu_name=torch.cuda.get_device_name(device),
                        torch_version=torch.__version__,sampler=sampler.state_dict())
        mech.save_json(args.run_dir/"manifest.json",manifest)
        all_reports=[]
        for step in range(args.batches):
            backbone.load_state_dict(source_net,strict=True)
            mech.restore_buffers(backbone,old_buffers)
            mech.restore_selected_rng(common_rng,device)
            backbone.zero_grad(set_to_none=True)
            indices,ids,gold,modified=mech.make_plan(sampler,step,args.plan_epoch,train_ids,labels,args.hotspot_ids)
            counts=shared_crop_counts(args.seed,args.plan_epoch,step)
            cloud_cpu,centers=mech.prepare_inputs(points,ids,args.seed,args.plan_epoch,step,args.input_mode,counts)
            cloud=cloud_cpu.to(device)
            clean_cloud=torch.from_numpy(points[ids,:1024].copy()).to(device)
            initial_readout=clean_readout(backbone,clean_cloud,old_buffers,common_rng,device)
            group_seed=args.seed+args.plan_epoch*10000019+step*1009
            relation_seed=args.seed+args.plan_epoch*10000019+step*10007
            targets=torch.from_numpy(gold).to(device)
            identities=torch.from_numpy(ids).to(device)
            outputs,replay_states,bn_deltas=mech.collect_global(backbone,cloud,centers,counts,"source_train",old_buffers,group_seed,device)
            leaf=mech.leaf_components(outputs,targets,identities,proxy_points,relation_seed)
            vectors,replayed_named,replay_checks=mech.replay_parameter_vjps(backbone,cloud,centers,counts,
                "source_train",old_buffers,replay_states,outputs,leaf["leaf_gradients"],device)
            if [name for name,_ in replayed_named]!=[name for name,_ in named]:
                raise RuntimeError("Replay and optimizer parameter names/order differ")
            base=vectors["ce"]+vectors["weighted_contrastive"]+vectors["weighted_radial"]
            gradients={"base":base,"base_sample_st":base+vectors["weighted_sample_st"],
                       "base_sample_fixed":base+vectors["weighted_sample_fixed"]}
            report={"batch":step,"ids":ids.tolist(),"labels":gold.tolist(),"subset_indices":indices.tolist(),
                    "negative_ids_global_flip":ids[::-1].tolist(),"forced_panel_changes":modified,
                    "input_cloud_sha256":mech.array_hash(cloud_cpu),"clean_cloud_sha256":mech.array_hash(clean_cloud),
                    "whole_count":counts[0],"child_count":counts[1],
                    "whole_centers":centers[0].tolist(),"child_centers":centers[1].tolist(),
                    "relation_seed":relation_seed,"BN_updates_per_local32":bn_deltas,
                    "source_train_geometry":mech.geometry(outputs),"losses_weighted":leaf["losses"],
                    "mining":leaf["mining"]["stats"],"sample_stats":leaf["sample_stats"],
                    "VJP_replay_max_abs_errors":replay_checks,
                    "parameter_gradient_groups":mech.parameter_summary(vectors,named),"arms":{}}
            after_readouts={}
            for arm in ARMS:
                backbone.load_state_dict(source_net,strict=True)
                mech.restore_buffers(backbone,old_buffers)
                mech.restore_selected_rng(common_rng,device)
                backbone.zero_grad(set_to_none=True)
                if mech.module_hash(backbone)!=source_model_sha:
                    raise RuntimeError("Independent arm model/BN did not restore")
                optimizer=build_optimizer([p for _,p in named],source_optimizer)
                before=clean_readout(backbone,clean_cloud,old_buffers,common_rng,device)
                start_max_errors={key:float((before[key]-initial_readout[key]).abs().max()) for key in before}
                for key in before:
                    torch.testing.assert_close(before[key],initial_readout[key],atol=2e-5,rtol=1e-4)
                no_update_response=response_arrays(initial_readout,before,gold)
                no_update_summary=response_summary(no_update_response,gold,ids,args.hotspot_ids)
                # Restore the arm RNG after its start-point gate; no stochastic
                # optimizer operation is expected, but starting identity stays explicit.
                mech.restore_selected_rng(common_rng,device)
                update=clipped_step(backbone,named,gradients[arm],optimizer)
                after=clean_readout(backbone,clean_cloud,old_buffers,common_rng,device)
                response=response_arrays(initial_readout,after,gold)
                report["arms"][arm]={"restore_optimizer_sha256":source_optimizer_sha,
                    "restore_model_sha256":source_model_sha,"start_readout_match":True,
                    "no_update_readout_max_abs_error":start_max_errors,
                    "no_update_readout_response":no_update_summary,
                    "update":update,"actual_clean_oldBN_response":response_summary(response,gold,ids,args.hotspot_ids)}
                after_readouts[arm]=after
                manifest["optimizer_updates"]+=1
                arrays={k:v.numpy() for k,v in response.items()}
                arrays.update(ids=ids,labels=gold,mu_before=initial_readout["mu"].numpy(),
                              mu_after=after["mu"].numpy(),logits_before=initial_readout["logits"].numpy(),
                              logits_after=after["logits"].numpy())
                np.savez_compressed(args.run_dir/f"batch_{step:03d}_{arm}.npz",**arrays)
                optimizer.zero_grad(set_to_none=True)
                del optimizer,before,after,response,no_update_response
            report["increments_vs_base"]={}
            for arm in ARMS[1:]:
                increment=response_arrays(after_readouts["base"],after_readouts[arm],gold)
                report["increments_vs_base"][arm]=response_summary(increment,gold,ids,args.hotspot_ids)
            if state_digest(source_optimizer)!=source_optimizer_sha or mech.module_hash(proxy)!=proxy_hash:
                raise RuntimeError("Source optimizer/proxy changed")
            backbone.load_state_dict(source_net,strict=True)
            mech.restore_buffers(backbone,old_buffers)
            mech.restore_selected_rng(common_rng,device)
            backbone.zero_grad(set_to_none=True)
            if mech.module_hash(backbone)!=source_model_sha:
                raise RuntimeError("Batch final restoration failed")
            report["checks"]={"same_startpoint":True,"optimizer_momentum_restored":True,
                "fixed_old_BN_readout":True,"finite_update_and_readout":True,"source_model_restored":True,
                "proxy_fixed":True,"source_optimizer_copy_unchanged":True}
            mech.save_json(args.run_dir/f"batch_{step:03d}.json",report)
            all_reports.append(report)
            manifest["completed_batches"]=step+1
            mech.save_json(args.run_dir/"manifest.json",manifest)
            print({"batch":step,"status":"passed","optimizer_updates":manifest["optimizer_updates"],
                   "clip_scales":{arm:report["arms"][arm]["update"]["clip_scale"] for arm in ARMS},
                   "mean_depth_delta":{arm:report["arms"][arm]["actual_clean_oldBN_response"]["all64"]["depth_delta"]["mean"] for arm in ARMS}},flush=True)
            del vectors,gradients,base,leaf,outputs,cloud,clean_cloud,after_readouts
        mech.restore_selected_rng(original_rng,device)
        if mech.digest(args.checkpoint)!=manifest["checkpoint_sha256"]:
            raise RuntimeError("Source checkpoint changed")
        if {name:mech.digest(args.data_dir/name) for name in shards}!=shard_hashes:
            raise RuntimeError("Input shards changed")
        if mech.module_hash(backbone)!=source_model_sha or mech.module_hash(proxy)!=proxy_hash:
            raise RuntimeError("Final model/proxy restore failed")
        if any(p.grad is not None for p in backbone.parameters()):
            raise RuntimeError("Final model gradient buffers not empty")
        manifest.update(status="completed",finished_utc=datetime.now(timezone.utc).isoformat(),
                        wall_seconds=time.perf_counter()-started)
        mech.save_json(args.run_dir/"summary.json",{"manifest":manifest,"batches":all_reports,
            "sign_convention":"Positive depth_delta is actual outward clean-eval motion after one optimizer step; increment_vs_base compares arm outputs directly.",
            "checks":{"source_checkpoint_unchanged":True,"data_shards_unchanged":True,
                      "source_model_restored":True,"proxy_unchanged":True,"optimizer_momentum_used":True}})
        mech.save_json(args.run_dir/"manifest.json",manifest)
    except BaseException as error:
        manifest.update(status="failed",error=repr(error),wall_seconds=time.perf_counter()-started)
        mech.save_json(args.run_dir/"manifest.json",manifest)
        raise


if __name__=="__main__":
    main()
