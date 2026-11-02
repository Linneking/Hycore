"""Matched final-M scale test on fixed, unnormalized ball features."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def tree_equal(a,b):
    import torch
    if isinstance(a,torch.Tensor):
        return isinstance(b,torch.Tensor) and a.dtype==b.dtype and a.shape==b.shape and torch.equal(a.cpu(),b.cpu())
    if isinstance(a,dict):
        return isinstance(b,dict) and a.keys()==b.keys() and all(tree_equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):
        return type(a)==type(b) and len(a)==len(b) and all(tree_equal(x,y) for x,y in zip(a,b))
    return a==b


def scale_loss(z,y,scale):
    import torch.nn.functional as F
    return F.cross_entropy(z,y)/scale if scale==1 else F.cross_entropy(z*scale,y)/scale


def final_only_optimizer(model,checkpoint,optimizer_cls):
    trainable=[model.mlr._weight,model.mlr._bias]
    ids={id(p) for p in trainable}
    for p in model.parameters():p.requires_grad_(id(p) in ids)
    cfg=checkpoint['config']
    # Keep original parameter ordering: shrinking the group would mis-map momentum.
    opt=optimizer_cls(model.parameters(),lr=cfg['lr'],momentum=cfg['momentum'],weight_decay=cfg['weight_decay'])
    opt.load_state_dict(copy.deepcopy(checkpoint['optimizer']))
    state_ids=[k for group in opt.state_dict()['param_groups'] for k in group['params']]
    objects=[p for group in opt.param_groups for p in group['params']]
    head_ids={k for p,k in zip(objects,state_ids) if id(p) in ids}
    assert len(head_ids)==2 and sum(p.requires_grad for p in model.parameters())==2
    assert tree_equal(opt.state_dict(),checkpoint['optimizer'])
    return opt,head_ids


def frozen_optimizer_equal(current,initial,head_ids,steps=0):
    if len(current['param_groups'])!=len(initial['param_groups']):return False
    for cur,old in zip(current['param_groups'],initial['param_groups']):
        # RSGD increments a global group counter even for frozen parameter groups.
        if not tree_equal({k:v for k,v in cur.items() if k!='step'},
                          {k:v for k,v in old.items() if k!='step'}):return False
        if 'step' in cur or 'step' in old:
            if cur.get('step',0)!=old.get('step',0)+steps:return False
    return all(
        tree_equal(current['state'].get(k),initial['state'].get(k))
        for k in set(current['state'])|set(initial['state']) if k not in head_ids)


def run(a):
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    os.environ['CUDA_VISIBLE_DEVICES']=str(a.gpu)
    sys.path.insert(0,a.source)
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'131_H_CAUSAL_DIAGNOSTICS_2026-11-02'))
    import torch
    import numpy as np
    import h5py
    import geoopt
    from audit import dump,sha,scores
    from segmentation_ShapeNetPart.hycore_seg_v1.train import ensure_idle
    from segmentation_ShapeNetPart.hycore_seg_v1.utils import seed_all,state_hash
    from segmentation_ShapeNetPart.hycore_seg_v1.metrics import SegMetrics
    from segmentation_ShapeNetPart.hycore_seg_v6.model import NativeTail2
    gpu=ensure_idle(a.gpu);torch.set_num_threads(4);seed_all(4780)
    started=time.time()
    runs=json.loads(Path(a.runs_json).read_text(encoding='utf-8-sig'))
    entry=next(e for e in runs if e['name']=='Hrepeat_best')
    cp=Path(entry['checkpoint']);cp_sha=sha(cp)
    ck=torch.load(cp,map_location='cpu',weights_only=False);cfg=ck['config']
    groups=json.loads(Path(a.groups).read_text())['groups']
    assert len(groups['fit'])==56 and len(groups['check'])==28
    assert not set(groups['fit'])&set(groups['check'])
    model=NativeTail2(arm='h',k=cfg.get('k',40),dropout=cfg.get('dropout',.5)).cuda().eval()
    model.load_state_dict(ck['model'],strict=True)
    original=state_hash(model.state_dict())
    frozen=lambda m:state_hash({k:v for k,v in m.state_dict().items() if k not in ['mlr._weight','mlr._bias']})
    frozen_hash=frozen(model)
    cache={}
    dump(out/'status.json',dict(state='extracting',at=time.time()))
    with h5py.File(Path(a.data)/'trainval.h5','r') as f:
        for split,indices in groups.items():
            idx=np.asarray(indices);assert np.all(np.diff(idx)>0)
            points=torch.from_numpy(f['data'][idx,:2048].astype('float32')).cuda()
            y=torch.from_numpy(f['pid'][idx,:2048].astype('int64')).cuda()
            c=torch.from_numpy(f['label'][idx].reshape(-1).astype('int64')).cuda()
            features=[];original_logits=[]
            with torch.no_grad():
                for start in range(0,len(idx),8):
                    pp=points[start:start+8].transpose(1,2).contiguous()
                    cc=torch.nn.functional.one_hot(c[start:start+8],16).float()
                    u=model.forward_features(pp,cc)
                    mu=model.emb2(model.manifold.expmap0(u))
                    features.append(mu);original_logits.append(model.mlr(mu))
            mu=torch.cat(features).detach()
            assert torch.isfinite(mu).all() and mu.norm(dim=-1).max()<=1.001
            cache[split]=(mu,y,c)
            torch.testing.assert_close(model.mlr(mu),torch.cat(original_logits),rtol=1e-5,atol=1e-6)
            del points
    assert original==state_hash(model.state_dict())
    torch.save({k:tuple(t.cpu() for t in value) for k,value in cache.items()},out/'frozen_mu.pt')
    arms={}
    for label,scale in [('A',1),('B',5)]:
        m=copy.deepcopy(model).eval()
        opt,head_ids=final_only_optimizer(m,ck,geoopt.optim.RiemannianSGD)
        assert state_hash(m.state_dict())==original
        arms[label]=dict(model=m,optimizer=opt,head_ids=head_ids,scale=scale)
    assert tree_equal(arms['A']['optimizer'].state_dict(),arms['B']['optimizer'].state_dict())
    with torch.no_grad():
        torch.testing.assert_close(arms['A']['model'].mlr(cache['fit'][0]),
                                   arms['B']['model'].mlr(cache['fit'][0]),rtol=0,atol=0)
    del model
    initial_optimizer=copy.deepcopy(ck['optimizer'])
    git=lambda path:subprocess.check_output(['git','-C',str(path),'rev-parse','HEAD'],text=True).strip()
    manifest=dict(started=started,pid=os.getpid(),seed=4780,gpu=gpu,
        source_commit=git(a.source),tool_commit=git(Path(__file__).parent),script_sha256=sha(__file__),
        checkpoint_sha256=cp_sha,checkpoint_epoch=ck['epoch'],model_sha256=original,
        frozen_state_sha256=frozen_hash,teacher=None,optimizer=cfg,
        actual_optimizer_groups=[{k:v for k,v in group.items() if k!='params'} for group in arms['A']['optimizer'].state_dict()['param_groups']],
        steps=a.steps,batch_points=8192,scales={'A':1,'B':5},groups=groups,
        groups_manifest_sha256=sha(a.groups),data_sha256=sha(Path(a.data)/'trainval.h5'),
        feature_sha256=sha(out/'frozen_mu.pt'),torch=torch.__version__,geoopt=geoopt.__version__,
        python=sys.version,device=torch.cuda.get_device_name(),
        normalization='none',selection='fixed final step; check objects never choose weights',
        frozen_initial_match=True,optimizer_initial_match=True)
    dump(out/'manifest.json',manifest)
    history=[];batch_hash=hashlib.sha256()
    train_mu,train_y,_=cache['fit'];flat_mu=train_mu.reshape(-1,128);flat_y=train_y.reshape(-1)
    rng=torch.Generator(device='cpu').manual_seed(4780)
    milestones={0,1,10,50,100,200,400,1000,2000,a.steps}
    last={}
    for step in range(a.steps+1):
        if step in milestones:
            record=dict(step=step,arms={},at=time.time())
            for label,arm in arms.items():
                m=arm['model'];opt=arm['optimizer'];row=dict(splits={},last_update=last.get(label),
                    raw_row_norm=m.mlr._weight.detach().norm(dim=-1).tolist(),
                    effective_row_norm=m.mlr.weight.detach().norm(dim=-1).tolist(),
                    bias_parameter_norm=float(m.mlr._bias.detach().norm()))
                with torch.no_grad():
                    for split,(mu,y,c) in cache.items():
                        z=m.mlr(mu)
                        ce=torch.nn.functional.cross_entropy(z.reshape(-1,50),y.reshape(-1))
                        objective=scale_loss(z.reshape(-1,50),y.reshape(-1),arm['scale'])
                        meter=SegMetrics();meter.update(z.argmax(-1).cpu().numpy(),y.cpu().numpy(),c.cpu().numpy())
                        row['splits'][split]=dict(raw_ce=float(ce),objective=float(objective),
                            parts=scores(z,y,c),focus_metrics=meter.result(),
                            output_boundary_fraction=float((z.norm(dim=-1)>=.996-1e-6).float().mean()))
                assert frozen(m)==frozen_hash
                assert frozen_optimizer_equal(opt.state_dict(),initial_optimizer,arm['head_ids'],step)
                record['arms'][label]=row
                torch.save(dict(step=step,scale=arm['scale'],mlr=m.mlr.state_dict(),optimizer=opt.state_dict()),
                           out/(f'{label}_step{step:05d}.pth'))
            history.append(record);dump(out/'history.json',history)
            dump(out/'status.json',dict(state='fitting' if step<a.steps else 'checking',step=step,at=time.time()))
            print(json.dumps(dict(step=step,check={label:row['splits']['check']['raw_ce'] for label,row in record['arms'].items()})),flush=True)
        if step==a.steps:break
        indices=torch.randint(len(flat_y),(8192,),generator=rng)
        batch_hash.update(indices.numpy().tobytes())
        ii=indices.cuda();mu=flat_mu[ii];y=flat_y[ii]
        for label,arm in arms.items():
            m=arm['model'];opt=arm['optimizer'];opt.zero_grad(set_to_none=True)
            before=[m.mlr._weight.detach().clone(),m.mlr._bias.detach().clone()]
            loss=scale_loss(m.mlr(mu),y,arm['scale'])
            assert torch.isfinite(loss)
            loss.backward()
            grad=float(torch.stack([p.grad.square().sum() for p in [m.mlr._weight,m.mlr._bias]]).sum().sqrt())
            opt.step()
            updates=[p.detach()-old for p,old in zip([m.mlr._weight,m.mlr._bias],before)]
            last[label]=dict(objective=float(loss.detach()),gradient_l2=grad,
                             parameter_update_l2=float(torch.stack([x.square().sum() for x in updates]).sum().sqrt()))
    assert sha(cp)==cp_sha and state_hash(ck['model'])==original
    result=dict(state='complete',history=history,checkpoint_unchanged=True,
        frozen_state_unchanged=True,frozen_optimizer_unchanged=True,paired_batches_sha256=batch_hash.hexdigest(),
        ended=time.time(),elapsed_seconds=time.time()-started,peak_memory_bytes=torch.cuda.max_memory_allocated(),
        final_step=a.steps,checkpoint_epoch=ck['epoch'],checkpoint_sha256=cp_sha,tool_commit=manifest['tool_commit'],
        actual_optimizer_groups=manifest['actual_optimizer_groups'])
    dump(out/'result.json',result);dump(out/'status.json',dict(state='complete',step=a.steps,at=time.time()))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['source','data','runs-json','groups','out']:p.add_argument('--'+key,required=True)
    p.add_argument('--gpu',type=int,default=2);p.add_argument('--steps',type=int,default=4000)
    a=p.parse_args()
    if a.steps<1:p.error('steps must be positive')
    try:run(a)
    except Exception:
        import traceback
        if Path(a.out).exists():
            Path(a.out,'failure.json').write_text(json.dumps(dict(traceback=traceback.format_exc()),indent=2))
        raise
