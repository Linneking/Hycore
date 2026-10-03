"""Bounded dual-GPU checks only. No epoch loop, validation selection or main training."""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'pointnet2_ops_lib'))
sys.path.insert(0, str(REPO / 'inter_hierarchy_MN40'))
os.environ.setdefault('HDF5_USE_FILE_LOCKING', 'FALSE')
import numpy as np
import geoopt
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from models.pointmlp import Hype_pointMLP
from hier_proxy_v2_online.train import load_shards, PointClouds
from hier_proxy_scratch_v4.train import stratified_split, split_identity
from hier_proxy_scratch_v3.train import freeze_curvature
from .sampler import SourceClassBatchSampler
from .base_protocol import HyCoReTrainingForward, global_base_losses
from .distributed import (gather_without_grad, shared_crop_counts,
                          broadcast_parameters, sync_parameter_gradients)
from .hier_loss import HIERLoss
from .sweep_k import examine_case, self_negative_audit, proxy_lr_audit
from .parameter_audit import audit_parameter_gradients


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def checkpoint_sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def cpu_case(values, ids, stage, step, source_identity):
    return {'stage': stage, 'step': step, 'source_identity': source_identity,
            **{key: values[key].detach().cpu().clone() for key in ('mu','nu','logits','gold')},
            'ids': ids.detach().cpu().clone()}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir', type=Path, required=True)
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--reference', type=Path)
    p.add_argument('--initial-checkpoint', type=Path,
                   help='Read-only representation reference for bounded joint updates; fresh optimizer')
    p.add_argument('--joint-steps', type=int, default=16)
    p.add_argument('--probe-batches', type=int, default=4)
    p.add_argument('--seed', type=int, default=22)
    p.add_argument('--proxy-lr', type=float, default=.01)
    p.add_argument('--topk', type=int, default=8,
                   help='Shared sample/proxy K, including self')
    p.add_argument('--skip-cpu-analysis', action='store_true',
                   help='Only run bounded GPU checks and save features')
    p.add_argument('--parameter-gradient-audit', action='store_true',
                   help='Discarded first/last forwards audit actual global parameter components')
    p.add_argument('--workers', type=int, default=2)
    args=p.parse_args()
    if not 1<=args.joint_steps<=24 or not 1<=args.probe_batches<=6:
        p.error('Diagnostic bounds: at most24 optimizer steps and6 reference batches')
    if not 3<=args.topk<=32:
        p.error('Shared diagnostic K must be between3 and32')
    torch.set_num_threads(2)
    rank=int(os.environ['RANK']); local=int(os.environ['LOCAL_RANK'])
    if int(os.environ['WORLD_SIZE'])!=2: raise RuntimeError('Exactly two ranks required')
    torch.cuda.set_device(local); device=torch.device('cuda',local)
    dist.init_process_group('nccl', timeout=dt.timedelta(minutes=5))
    if rank==0: args.run_dir.mkdir(parents=True, exist_ok=False)
    dist.barrier()
    manifest={'status':'running', 'diagnostic_only':True, 'main_training':False,
              'started_utc':dt.datetime.now(dt.timezone.utc).isoformat(),
              'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
              'command_argv':[sys.executable,*sys.argv],
              'config':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
              'fixed':{'c':1,'D':256,'P':512,'margin':.1,'tau':.1,'lambda_H':.5,
                       'global_CE_intra_inter':64,'local_encoder_BN':32,'sample_proxy_K':args.topk,
                       'T':50,'nominal_steps':200,'extra_cap':False,'HIER_backward_hook':False,
                       'source_self_negative':True,'proxy_initialization':'random_tangent',
                       'optimizer_state':'fresh'},
              'visible_gpus':os.environ.get('CUDA_VISIBLE_DEVICES'),
              'runtime':{'torch':torch.__version__,'geoopt':geoopt.__version__},
              'reference_identity':None,'initial_checkpoint_identity':None}
    try:
        points, labels, shards=load_shards(args.data_dir,'train')
        train_ids,val_ids=stratified_split(labels,args.seed,.1)
        manifest.update(split_sha256=split_identity(train_ids,val_ids), train_size=len(train_ids),
                        validation_size=len(val_ids), validation_read=False,test_read=False,
                        training_shards=shards)
        seed_all(args.seed)
        backbone=Hype_pointMLP().to(device); freeze_curvature(backbone)
        case_source_identity='random_seed'+str(args.seed)
        if args.initial_checkpoint:
            initial=torch.load(args.initial_checkpoint,map_location='cpu',weights_only=False)
            backbone.load_state_dict({k.removeprefix('module.'):v for k,v in initial['net'].items()},strict=True)
            freeze_curvature(backbone)
            case_source_identity={'sha256':checkpoint_sha(args.initial_checkpoint),
                                  'epoch':initial.get('epoch'),
                                  'role':'representation-reference joint diagnostic; fresh optimizer, not V5 main initialization'}
            manifest['initial_checkpoint_identity']=case_source_identity
            del initial
        model=DDP(HyCoReTrainingForward(backbone),device_ids=[local],broadcast_buffers=True)
        proxy=HIERLoss(seed=args.seed).to(device); broadcast_parameters(proxy)
        optimizer=geoopt.optim.RiemannianSGD([x for x in model.parameters() if x.requires_grad],
                                           lr=.1,momentum=.9,weight_decay=2e-4)
        popt=torch.optim.AdamW(proxy.parameters(),lr=args.proxy_lr,weight_decay=.01,eps=1e-8)
        dataset=PointClouds(points[train_ids],labels[train_ids],1024,'train',args.seed)
        sampler=SourceClassBatchSampler(labels[train_ids],rank=rank,seed=args.seed,steps=200)
        loader=DataLoader(dataset,batch_sampler=sampler,num_workers=args.workers,pin_memory=True)
        if rank==0: write_json(args.run_dir/'manifest.json',manifest)
        rows=[]; cases=[]
        model.train(); torch.cuda.reset_peak_memory_stats(device)
        for step,(cloud,gold,subset_ids) in enumerate(loader):
            if step>=args.joint_steps: break
            seed_all(args.seed+step*1009+rank*1000003)
            whole_count,child_count=shared_crop_counts(args.seed,0,step)
            optimizer.zero_grad(set_to_none=True); popt.zero_grad(set_to_none=True)
            parameter_gradient={}
            if args.parameter_gradient_audit and step in (0,args.joint_steps-1):
                parameter_gradient=audit_parameter_gradients(
                    model,proxy,cloud.to(device),gold.to(device),
                    torch.as_tensor(train_ids[subset_ids.numpy()],device=device),
                    whole_count,child_count,topk=args.topk,seed=args.seed+step*10007)
                # Restore augmentation/crop-center RNG for the actual step.
                seed_all(args.seed+step*1009+rank*1000003)
            bn=[m for m in backbone.modules() if isinstance(m,torch.nn.modules.batchnorm._BatchNorm)]
            before=[int(m.num_batches_tracked) for m in bn]
            dist.barrier(); torch.cuda.synchronize(device); started=time.perf_counter()
            local_out=model(cloud.to(device),whole_count,child_count)
            values=global_base_losses(local_out,gold.to(device))
            ids=gather_without_grad(torch.as_tensor(train_ids[subset_ids.numpy()],device=device))
            gen=torch.Generator(device=device).manual_seed(args.seed+step*10007)
            h_loss,h_stats=proxy(values['mu'],values['gold'],topk=args.topk,seed=args.seed+step*10007,
                                 data_ids=ids,generator=gen)
            gradient_stats={}
            if step in (0,args.joint_steps-1):
                grads={}
                for name,loss in [('ce',values['ce']),('intra',values['base']-values['ce']),
                                  ('weighted_hier',.5*h_loss)]:
                    grads[name]=torch.autograd.grad(loss,local_out['mu'],retain_graph=True)[0]
                    gradient_stats[name+'_local_mu_grad_norm']=float(grads[name].norm())
                for a,b in [('ce','weighted_hier'),('intra','weighted_hier')]:
                    ga,gb=grads[a].flatten(),grads[b].flatten()
                    gradient_stats[a+'_'+b+'_cosine']=float(torch.dot(ga,gb)/(ga.norm()*gb.norm()).clamp_min(1e-30))
            loss=values['base']+.5*h_loss
            loss.backward(); sync_parameter_gradients(proxy.parameters())
            if not torch.isfinite(loss) or any(x.grad is not None and not torch.isfinite(x.grad).all()
                                               for x in list(model.parameters())+list(proxy.parameters())):
                raise RuntimeError('Nonfinite loss/gradient in uncapped ordinary-backward diagnostic')
            model_norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1)
            proxy_grad=proxy.tangent_proxies.grad
            proxy_grad_norm=float(proxy_grad.norm())
            proxy_gradient_over10=int((proxy_grad.abs()>10).sum())
            optimizer.step(); popt.step()
            twin=proxy.tangent_proxies.detach().clone(); dist.broadcast(twin,src=0)
            proxy_diff=(twin-proxy.tangent_proxies.detach()).abs().max()
            dist.all_reduce(proxy_diff,op=dist.ReduceOp.MAX)
            if float(proxy_diff)>1e-7: raise RuntimeError('Proxy replicas diverged')
            delta=[int(m.num_batches_tracked)-v for m,v in zip(bn,before)]
            if any(v!=2 for v in delta): raise RuntimeError('Restored child/whole BN must update twice')
            torch.cuda.synchronize(device)
            elapsed=torch.tensor(time.perf_counter()-started,device=device)
            dist.all_reduce(elapsed,op=dist.ReduceOp.MAX)
            rank_observations=[None,None]
            dist.all_gather_object(rank_observations,{
                'rank':rank,'input':local_out['input_stats'],'BN_update_deltas':delta,
                'peak_allocated_MiB':torch.cuda.max_memory_allocated(device)/1024**2,
                'gradient':gradient_stats})
            row={'step':step,'step_seconds_max_rank':float(elapsed),'input':local_out['input_stats'],
                 'global_batch':64,'ce':float(values['ce'].detach()),
                 'intra_contrastive':float(values['intra_contrastive'].detach()),
                 'intra_radial':float(values['intra_radial'].detach()),
                 'loss':float(loss.detach()),'BN_update_deltas':delta,
                 'model_preclip_norm':float(model_norm),'proxy_grad_norm':proxy_grad_norm,
                 'shadow_proxy_gradient_elements_over10':proxy_gradient_over10,
                 'proxy_replica_max_difference':float(proxy_diff),
                 'peak_allocated_MiB':torch.cuda.max_memory_allocated(device)/1024**2,
                 'hier':h_stats,'gradient':gradient_stats,'parameter_gradient':parameter_gradient,
                 'rank_observations':rank_observations}
            rows.append(row)
            if rank==0:
                write_json(args.run_dir/f'step_{step:03d}.json',row)
                if step<2 or step>=args.joint_steps-2:
                    stage='reference_start_short_updates' if args.initial_checkpoint else 'random_start_short_updates'
                    cases.append(cpu_case(values,ids,stage,step,case_source_identity))
                print(json.dumps({'step':step,'seconds':float(elapsed),'loss':float(loss.detach()),
                                  'anchors':h_stats['sample']['eligible_anchors']},allow_nan=False),flush=True)
        if args.reference:
            ref=torch.load(args.reference,map_location='cpu',weights_only=False)
            state={k.removeprefix('module.'):v for k,v in ref['net'].items()}
            backbone.load_state_dict(state,strict=True); freeze_curvature(backbone)
            identity={'sha256':checkpoint_sha(args.reference),'epoch':ref.get('epoch'),
                      'role':'V4 representation reference, not a V5 initialization or performance baseline'}
            manifest['reference_identity']=identity
            model.train()
            with torch.no_grad():
                for step,(cloud,gold,subset_ids) in enumerate(loader):
                    if step>=args.probe_batches: break
                    seed_all(args.seed+step*1009+rank*1000003)
                    wc,cc=shared_crop_counts(args.seed,0,step)
                    local_out=model(cloud.to(device),wc,cc)
                    values=global_base_losses(local_out,gold.to(device))
                    ids=gather_without_grad(torch.as_tensor(train_ids[subset_ids.numpy()],device=device))
                    if rank==0: cases.append(cpu_case(values,ids,'v4_epoch20_reference',step,identity))
        dist.barrier()
        # Finish collectives before rank0's potentially slower CPU analysis.
        # Otherwise rank1 can time out at a barrier while rank0 analyzes caches.
        dist.destroy_process_group()
        del optimizer, popt, model, backbone, proxy
        torch.cuda.empty_cache()
        if rank==0:
            cache={'identity':{'commit':manifest['commit'],'split_sha256':manifest['split_sha256'],
                               'source':'training subsets only; restored whole overwrite'},
                   'config':manifest['fixed'],'proxy_count':512,'cases':cases}
            torch.save(cache,args.run_dir/'training_features.pt')
            print('GPU_CHECKS_COMPLETED; CPU_RELATION_ANALYSIS',flush=True)
            sweep=[]; selfk=[]; lr_cases=[]
            if not args.skip_cpu_analysis:
                cpu_proxy=HIERLoss(seed=args.seed)
                sweep=[r for case in cases for r in examine_case(case,cpu_proxy,seed=args.seed)]
                selfk=[self_negative_audit(case,cpu_proxy,k=args.topk,seed=args.seed) for case in cases]
                lr_cases=proxy_lr_audit(cases[-1],seed=args.seed,k=args.topk)
            times=[r['step_seconds_max_rank'] for r in rows[3:]] or [r['step_seconds_max_rank'] for r in rows]
            median=statistics.median(times)
            summary={'diagnostic_only':True,'joint_optimizer_steps':args.joint_steps,
                     'median_step_seconds_after3':median,'estimated_epoch138_seconds':138*median,
                     'estimated_epoch200_seconds':200*median,'relative200_to138':200/138,
                     'time_estimate_limit':'short-run compute; excludes cold imports, IO changes, validation, long-term drift',
                     'rows':rows,'k_sweep':sweep,'self_negative':selfk,'proxy_lr':lr_cases}
            write_json(args.run_dir/'diagnostic.json',summary)
            manifest.update(status='completed',finished_utc=dt.datetime.now(dt.timezone.utc).isoformat())
            write_json(args.run_dir/'manifest.json',manifest)
            print('DIAGNOSTIC_COMPLETED',flush=True)
    except BaseException as exc:
        if rank==0:
            manifest.update(status='failed',error=repr(exc)); write_json(args.run_dir/'manifest.json',manifest)
        raise
    finally:
        if dist.is_initialized(): dist.destroy_process_group()


if __name__=='__main__': main()
