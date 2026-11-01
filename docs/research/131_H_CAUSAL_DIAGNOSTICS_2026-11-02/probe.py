"""Frozen representation probes; object-disjoint fitting/check subsets of trainval.

The encoder previously saw trainval. This assesses readout accessibility, not
unseen-object generalization or task-level improvement. Only probe weights train.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time
from audit import dump,sha,scores


def run(a):
    out=Path(a.out); out.mkdir(parents=True,exist_ok=False)
    os.environ['CUDA_VISIBLE_DEVICES']=str(a.gpu); sys.path.insert(0,a.source)
    import torch
    import numpy as np
    import h5py
    from segmentation_ShapeNetPart.hycore_seg_v1.train import ensure_idle
    from segmentation_ShapeNetPart.hycore_seg_v1.utils import state_hash,seed_all
    from segmentation_ShapeNetPart.hycore_seg_v1.model import DGCNNPartSeg
    from segmentation_ShapeNetPart.hycore_seg_v6.model import NativeTail2
    ensure_idle(a.gpu); torch.set_num_threads(4); seed_all(4780)
    rng=np.random.default_rng(4780)
    groups={}; cache={}
    with h5py.File(Path(a.data)/'trainval.h5','r') as f:
        cats=f['label'][:].reshape(-1)
        chosen={c:rng.choice(np.flatnonzero(cats==c),12,replace=False) for c in [1,13,10,14,5,2,11]}
        for split,sl in [('fit',slice(0,8)),('check',slice(8,12))]:
            idx=np.sort(np.concatenate([v[sl] for v in chosen.values()]))
            groups[split]=idx.tolist()
            cache[split]=(torch.from_numpy(f['data'][idx,:2048].astype('float32')).cuda(),
                torch.from_numpy(f['pid'][idx,:2048].astype('int64')).cuda(),
                torch.from_numpy(cats[idx].astype('int64')).cuda())
    assert not set(groups['fit'])&set(groups['check'])
    dump(out/'manifest.json',dict(seed=4780,groups=groups,script_sha256=sha(__file__),
        source_checkpoint_list_sha256=sha(a.runs_json),started=time.time(),encoder_updates=0,
        probe_steps=400,batch_points=8192,optimizer='Adam lr=.01',
        normalization='per-channel fit RMS, without centering; invertible scaling',
        caveat='All objects were used in original encoder training; shape split only isolates probe fitting.'))
    runs=json.loads(Path(a.runs_json).read_text(encoding='utf-8-sig'))
    for entry in runs:
        if entry['name'] not in ['DGCNN_best','H0_best','Hrepeat_best']: continue
        name=entry['name']; dump(out/'status.json',dict(state='extracting',model=name))
        ck=torch.load(entry['checkpoint'],map_location='cpu',weights_only=False)
        cfg=ck['config']; kw=dict(k=cfg.get('k',40),dropout=cfg.get('dropout',.5))
        model=DGCNNPartSeg(head='euclidean',**kw) if entry['kind']=='dgcnn' else NativeTail2(arm='h',**kw)
        model.load_state_dict(ck['model'],strict=True); del ck
        before=state_hash(model.state_dict()); model.cuda().eval()
        features={}; original={}
        for split,(pts,y,c) in cache.items():
            ulist=[];vlist=[];zlist=[]
            for start in range(0,len(pts),8):
                pp=pts[start:start+8].transpose(1,2).contiguous()
                cc=torch.nn.functional.one_hot(c[start:start+8],16).float()
                with torch.no_grad():
                    if entry['kind']=='dgcnn':
                        captured=[]
                        handle=model.conv9.register_forward_hook(lambda mod,args,output:captured.append(output))
                        try: zz,v=model(pp,cc,return_features=True)
                        finally: handle.remove()
                        u=captured[0].transpose(1,2).contiguous(); z=zz.transpose(1,2)
                    else:
                        u=model.forward_features(pp,cc)
                        v=model.emb2(model.manifold.expmap0(u)); z=model.mlr(v)
                ulist.append(u);vlist.append(v);zlist.append(z)
            features[split]={'U':torch.cat(ulist).detach(),'V128':torch.cat(vlist).detach()}
            original[split]=scores(torch.cat(zlist),y,c)
        assert before==state_hash(model.state_dict())
        del model; torch.cuda.empty_cache()
        for node in ['U','V128']:
            x=features['fit'][node].reshape(-1,features['fit'][node].shape[-1])
            scale=x.square().mean(0).sqrt().clamp_min(1e-6)
            features['fit'][node]=features['fit'][node]/scale
            features['check'][node]=features['check'][node]/scale
        result=dict(checkpoint_sha256=sha(entry['checkpoint']),encoder_state_unchanged=True,
                    original=original,probe={})
        for seed in [4780,4781]:
            seed_all(seed)
            heads={node+('_affine' if bias else '_linear'):
                torch.nn.Linear(features['fit'][node].shape[-1],50,bias=bias).cuda()
                for node in ['U','V128'] for bias in [False,True]}
            optimizer=torch.optim.Adam([p for h in heads.values() for p in h.parameters()],lr=.01)
            curve=[]
            for step in range(401):
                if step in [0,100,200,400]:
                    stats={}
                    with torch.no_grad():
                        for key,h in heads.items():
                            node=key.split('_')[0]; stats[key]={}
                            for split,(_,y,c) in cache.items():
                                z=h(features[split][node]); s=scores(z,y,c)
                                loss=torch.nn.functional.cross_entropy(z.reshape(-1,50),y.reshape(-1))
                                stats[key][split]=dict(ce=float(loss),parts=s)
                    curve.append(dict(step=step,stats=stats))
                    dump(out/'status.json',dict(state='probe_fitting',model=name,seed=seed,step=step))
                if step==400: break
                idx=torch.randint(cache['fit'][1].numel(),(8192,),device='cuda')
                y=cache['fit'][1].reshape(-1)[idx]
                optimizer.zero_grad(set_to_none=True)
                losses=[]
                for key,h in heads.items():
                    node=key.split('_')[0]
                    x=features['fit'][node].reshape(-1,features['fit'][node].shape[-1])[idx]
                    losses.append(torch.nn.functional.cross_entropy(h(x),y))
                sum(losses).backward();optimizer.step()
            result['probe'][str(seed)]=curve
        dump(out/(name+'.json'),result)
        print(name,'probe complete',flush=True)
        del features
        torch.cuda.empty_cache()
    dump(out/'status.json',dict(state='complete',at=time.time()))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['source','data','runs-json','out']:p.add_argument('--'+key,required=True)
    p.add_argument('--gpu',type=int,default=2);a=p.parse_args()
    try:run(a)
    except Exception:
        import traceback
        if Path(a.out).exists():dump(Path(a.out)/'failure.json',dict(traceback=traceback.format_exc()))
        raise
