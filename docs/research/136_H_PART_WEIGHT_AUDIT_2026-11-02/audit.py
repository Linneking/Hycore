"""CPU-only part-weight controls, exact last-M update decomposition and log audit."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time


def weights_from_fit(y, starts, counts):
    import torch
    n=torch.bincount(y.reshape(-1),minlength=50).double();w=torch.zeros(50,dtype=torch.float64)
    for start,count in zip(starts,counts):
        f=n[start:start+count];present=f>0
        raw=torch.zeros_like(f);raw[present]=(f.max()/f[present]).sqrt().clamp_max(5)
        w[start:start+count]=raw/((raw*f).sum()/f.sum())
    return w.float(),n.long()


def weighted_loss(z,y,w):
    import torch.nn.functional as F
    return (F.cross_entropy(5*z,y,reduction='none')*w[y]).mean()/5


def draw(rng,counts,categories):
    import torch
    import numpy as np
    uniform=torch.rand(32,generator=rng);objdraw=torch.randint(8,(32,),generator=rng)
    points=torch.randint(2048,(32,256),generator=rng)
    pool=torch.stack([torch.nonzero(categories==c).flatten() for c in range(16)])
    result={}
    for prior,prob in [('N',torch.tensor(np.asarray(counts)/sum(counts))),('U',torch.full((16,),1/16))]:
        cls=torch.searchsorted(prob.cumsum(0),uniform.double()).clamp_max(15)
        result[prior]=(pool[cls,objdraw][:,None]*2048+points).reshape(-1)
    return result


def log_audit(root):
    groups={};sizes={};import math
    for arm in ['r0','r1']:
        path=Path(root)/arm/'geometry.jsonl';sizes[arm]=path.stat().st_size
        with path.open() as f:
            while f.tell()<sizes[arm]:
                line=f.readline()
                if not line.endswith('\n'):break
                d=json.loads(line);g=d.get('gradients',{})
                if not g:continue
                band=(d['epoch']-1)//20;cats=d['diagnostics']['anchor_classes']
                for cohort in ['all']+['contains_'+str(c) for c in [1,5,10,13,14] if c in cats]:
                    key=f'{arm}:e{band*20+1}-{band*20+20}:{cohort}'
                    row=groups.setdefault(key,{'n':0,'contr_over_ce':[],'hier_over_ce':[],'ce_h_l2':[],'contr_h_l2':[],'hier_h_l2':[]})
                    row['n']+=1;ce=g['ce']['nodes']['h']['l2']
                    for loss in ['ce','contr','hier']:
                        value=g[loss]['nodes'].get('h',{}).get('l2')
                        if value is not None and math.isfinite(value):row[loss+'_h_l2'].append(value)
                        if loss!='ce' and value is not None and ce and math.isfinite(value/ce):row[loss+'_over_ce'].append(value/ce)
    return {'bytes_at_open':sizes,'cohorts':{k:{name:({'n':len(v),'median':statistics.median(v),'min':min(v),'max':max(v)} if v else None) if isinstance(v,list) else v for name,v in row.items()} for k,row in groups.items()},
            'limit':'sampled batch-level norm association only; no vectors/cosines or class-specific gradient attribution'}


def run(a):
    os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']='2';os.environ['MKL_NUM_THREADS']='2'
    try:os.nice(10)
    except AttributeError:pass
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    sys.path.insert(0,a.source);sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'132_H_FINAL_M_SCALE_2026-11-02'))
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'131_H_CAUSAL_DIAGNOSTICS_2026-11-02'))
    import torch
    import geoopt
    # import the earlier metric helper by explicit file identity (this file also named audit.py).
    import importlib.util
    helper=Path(__file__).resolve().parents[1]/'131_H_CAUSAL_DIAGNOSTICS_2026-11-02/audit.py'
    spec=importlib.util.spec_from_file_location('part_stats_helper',helper);stats=importlib.util.module_from_spec(spec);spec.loader.exec_module(stats)
    stats.PARTS=list(range(50));dump,sha=stats.dump,stats.sha
    from run_scale import final_only_optimizer,frozen_optimizer_equal,scale_loss
    from segmentation_ShapeNetPart.hycore_seg_v6.model import NativeTail2
    from segmentation_ShapeNetPart.hycore_seg_v1.utils import state_hash,seed_all
    from segmentation_ShapeNetPart.hycore_seg_v1.metrics import SegMetrics,PART_STARTS,PART_COUNTS
    torch.set_num_threads(2);torch.set_num_interop_threads(1);seed_all(4780);started=time.time()
    prior=Path(a.prior);manifest=json.loads((prior/'manifest.json').read_text());reference=json.loads((prior/'result.json').read_text())
    assert sha(prior/'frozen_mu.pt')==manifest['feature_sha256']
    cache=torch.load(prior/'frozen_mu.pt',map_location='cpu',weights_only=False)
    entries=json.loads(Path(a.runs_json).read_text(encoding='utf-8-sig'));cp=Path(next(e['checkpoint'] for e in entries if e['name']=='Hrepeat_best'))
    cp_sha=sha(cp);assert cp_sha==manifest['checkpoint_sha256']
    ck=torch.load(cp,map_location='cpu',weights_only=False);cfg=ck['config'];base=NativeTail2('h',k=cfg['k'],dropout=cfg['dropout']).eval();base.load_state_dict(ck['model'],strict=True)
    original=state_hash(base.state_dict());frozen=lambda m:state_hash({k:v for k,v in m.state_dict().items() if k not in ['mlr._weight','mlr._bias']})
    frozen_hash=frozen(base);initial_optimizer=copy.deepcopy(ck['optimizer'])
    mu,y,c=cache['fit'];flat_mu,flat_y=mu.reshape(-1,128),y.reshape(-1)
    weights,n=weights_from_fit(y,PART_STARTS,PART_COUNTS);check_n=torch.bincount(cache['check'][1].reshape(-1),minlength=50)
    support={str(k):{'fit':int(n[k]),'check':int(check_n[k]),'weight':float(weights[k]),'testable':bool(n[k]>0 and check_n[k]>0)} for k in range(50)}
    for start,count in zip(PART_STARTS,PART_COUNTS):torch.testing.assert_close((weights[start:start+count].double()*n[start:start+count]).sum()/n[start:start+count].sum(),torch.ones((),dtype=torch.float64),atol=1e-6,rtol=1e-6)
    dump(out/'manifest.json',dict(started=started,pid=os.getpid(),device='cpu',threads=2,seed=4780,steps=a.steps,checkpoint_sha256=cp_sha,
        cache_sha256=manifest['feature_sha256'],source_commit=subprocess.check_output(['git','-C',a.source,'rev-parse','HEAD'],text=True).strip(),
        tool_commit=subprocess.check_output(['git','-C',str(Path(__file__).parent),'rev-parse','HEAD'],text=True).strip(),support=support,
        note='CPU N5/U5 are hardware-matched replays of prior GPU controls; only NW5/UW5 are new weighted conditions'))

    def margins(m,x,target,categories):
        return stats.scores(m.mlr(x),target,categories)

    # Exact raw-parameter update decomposition and actual margin counterfactuals.
    decomposition=[];rng=torch.Generator().manual_seed(4780)
    for batch in range(3):
        idx=draw(rng,manifest['class_counts'],c)['U'];x,yy=flat_mu[idx],flat_y[idx]
        m=copy.deepcopy(base);opt,head_ids=final_only_optimizer(m,ck,geoopt.optim.RiemannianSGD)
        ps=[m.mlr._weight,m.mlr._bias];old=[p.detach().clone() for p in ps];group=opt.param_groups[0]
        assert group['dampening']==0 and not group['nesterov']
        assert all(not isinstance(p,(geoopt.ManifoldParameter,geoopt.ManifoldTensor)) and 'momentum_buffer' in opt.state[p] for p in ps)
        z=m.mlr(x);losses=torch.nn.functional.cross_entropy(5*z,yy,reduction='none')/5
        g=torch.autograd.grad(losses.mean(),ps,retain_graph=True)
        own_gradients={k:torch.autograd.grad((losses*(yy==k)).mean(),ps,retain_graph=True)
                       for k in [4,42,43,31,33,34,6,37] if (yy==k).any()}
        momentum=[-group['lr']*group['momentum']*opt.state[p]['momentum_buffer'].clone() for p in ps]
        decay=[-group['lr']*group['weight_decay']*p for p in old]
        reconstructed=[mom+wd-group['lr']*grad for mom,wd,grad in zip(momentum,decay,g)]
        for p,grad in zip(ps,g):p.grad=grad.clone()
        opt.step();actual=[p.detach()-o for p,o in zip(ps,old)]
        for v,r in zip(actual,reconstructed):torch.testing.assert_close(v,r,rtol=1e-3,atol=1e-7)
        for p,o,r in zip(ps,old,reconstructed):torch.testing.assert_close(p,o+r,rtol=1e-6,atol=1e-7)
        assert frozen_optimizer_equal(opt.state_dict(),initial_optimizer,head_ids,1)
        parts={}
        for k in [4,42,43,31,33,34,6,37]:
            mask=yy==k
            if not mask.any():continue
            own=own_gradients[k]
            components={'own':[-group['lr']*v for v in own],
                'others':[-group['lr']*(total-v) for total,v in zip(g,own)],'momentum':momentum,'decay':decay,'full':actual}
            def part_margin(delta):
                with torch.no_grad():
                    for p,o,v in zip(ps,old,delta):p.copy_(o+v)
                    zz=m.mlr(x);correct=zz[mask,k];wrong=zz[mask].clone();wrong[:,k]=-float('inf')
                    owner=next(j for j,(s,nc) in enumerate(zip(PART_STARTS,PART_COUNTS)) if s<=k<s+nc)
                    lo,hi=PART_STARTS[owner],PART_STARTS[owner]+PART_COUNTS[owner]
                    return float((correct-wrong[:,lo:hi].max(-1).values).mean())
            zero=[torch.zeros_like(p) for p in ps];before=part_margin(zero)
            parts[str(k)]={'n':int(mask.sum()),'before_margin':before,'components':{name:{'update_l2':float(sum(v.square().sum() for v in values).sqrt()),'margin_delta':part_margin(values)-before} for name,values in components.items()}}
        decomposition.append({'batch':batch,'max_reconstruction_error':max(float((x-r).abs().max()) for x,r in zip(actual,reconstructed)),'parts':parts})
    dump(out/'update_decomposition.json',decomposition)

    def evaluate(m):
        result={}
        with torch.no_grad():
            for split,(xx,yy,cc) in cache.items():
                z=torch.cat([m.mlr(part) for part in xx.split(4)])
                meter=SegMetrics();pred=z.argmax(-1);meter.update(pred.numpy(),yy.numpy(),cc.numpy());s=stats.scores(z,yy,cc)
                for k,v in s.items():
                    pr=int((pred==int(k)).sum());v['pred_all']=pr;v['precision']=v['tp']/pr if pr else None;v['iou']=v['tp']/(v['n']+pr-v['tp'])
                result[split]={'raw_ce':float(torch.nn.functional.cross_entropy(z.reshape(-1,50),yy.reshape(-1))),'metrics':meter.result(),'parts':s}
        return result
    arms={}
    for label in ['N5','U5','NW5','UW5']:
        m=copy.deepcopy(base);opt,ids=final_only_optimizer(m,ck,geoopt.optim.RiemannianSGD);arms[label]=(m,opt,ids)
    history=[];rng=torch.Generator().manual_seed(4780);digest=hashlib.sha256();milestones={0,1,10,50,100,200,400,1000,2000,a.steps}
    for step in range(a.steps+1):
        if step in milestones:
            record={'step':step,'arms':{}}
            for label,(m,opt,ids) in arms.items():
                assert frozen(m)==frozen_hash and frozen_optimizer_equal(opt.state_dict(),initial_optimizer,ids,step)
                record['arms'][label]=evaluate(m)
            history.append(record);dump(out/'history.json',history);dump(out/'status.json',{'state':'fitting','step':step,'at':time.time()});print(step,'done',flush=True)
        if step==a.steps:break
        batches=draw(rng,manifest['class_counts'],c)
        for indices in batches.values():digest.update(indices.numpy().tobytes())
        for label,(m,opt,_) in arms.items():
            idx=batches[label[0]];xx,yy=flat_mu[idx],flat_y[idx];opt.zero_grad(set_to_none=True)
            z=m.mlr(xx);loss=weighted_loss(z,yy,weights) if 'W' in label else scale_loss(z,yy,5)
            assert torch.isfinite(loss);loss.backward();opt.step()
    assert digest.hexdigest()==reference['paired_batches_sha256']
    assert sha(cp)==cp_sha and state_hash(ck['model'])==original
    for label,(m,opt,_) in arms.items():torch.save({'mlr':m.mlr.state_dict(),'optimizer':opt.state_dict()},out/(label+'_final.pth'))
    dump(out/'log_audit.json',log_audit(a.production))
    result={'state':'complete','history':history,'support':support,'paired_hash_match':True,'frozen_states_unchanged':True,'checkpoint_unchanged':True,
        'elapsed_seconds':time.time()-started,'updates':decomposition,'note':'CPU-matched controls; pooled-part metrics do not replace full test metrics'}
    dump(out/'result.json',result);dump(out/'status.json',{'state':'complete','step':a.steps,'at':time.time()})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['source','prior','runs-json','production','out']:p.add_argument('--'+k,required=True)
    p.add_argument('--steps',type=int,default=4000);a=p.parse_args()
    try:run(a)
    except Exception:
        import traceback
        if Path(a.out).exists():Path(a.out,'failure.json').write_text(json.dumps({'traceback':traceback.format_exc()},indent=2))
        raise
