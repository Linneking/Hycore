"""Matched one-step RSGD interventions in disposable model/optimizer copies."""
import argparse
import copy
import json
import os
from pathlib import Path
import sys
import time
from audit import dump,sha,scores


def run(a):
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    os.environ['CUDA_VISIBLE_DEVICES']=str(a.gpu);sys.path.insert(0,a.source)
    import torch
    import numpy as np
    import h5py
    import geoopt
    from segmentation_ShapeNetPart.hycore_seg_v1.train import ensure_idle
    from segmentation_ShapeNetPart.hycore_seg_v1.utils import seed_all,restore_rng,state_hash
    from segmentation_ShapeNetPart.hycore_seg_v6.model import NativeTail2
    from segmentation_ShapeNetPart.hycore_seg_v6.losses import HyCoReSegLoss
    ensure_idle(a.gpu);torch.set_num_threads(4)
    audit_manifest=json.loads(Path(a.batches).read_text())
    batches=[]
    with h5py.File(Path(a.data)/'trainval.h5','r') as f:
        for batch in audit_manifest['batches']:
            idx=np.array(batch['indices']);order=np.argsort(idx);inverse=np.argsort(order);ix=idx[order]
            pts=torch.from_numpy(f['data'][ix,:2048].astype('float32')[inverse]).cuda().transpose(1,2).contiguous()
            y=torch.from_numpy(f['pid'][ix,:2048].astype('int64')[inverse]).cuda()
            c=torch.from_numpy(f['label'][ix].reshape(-1).astype('int64')[inverse]).cuda()
            batches.append((pts,y,c))
    entry=next(e for e in json.loads(Path(a.runs_json).read_text(encoding='utf-8-sig')) if e['name']=='Hrepeat_best')
    cp=Path(entry['checkpoint']);ck=torch.load(cp,map_location='cpu',weights_only=False);cfg=ck['config']
    checkpoint_sha=sha(cp);original_hash=state_hash(ck['model'])
    result=dict(checkpoint_sha256=checkpoint_sha,epoch=ck['epoch'],seed=4780,
        model_state_sha256=original_hash,script_sha256=sha(__file__),
        source_audit_manifest_sha256=sha(a.batches),conditions={},
        scope='one local training-mode step, matching stored momentum/weight decay/LR and sampling RNG; no claim of long-run effect')
    for b,(pts,y,c) in enumerate(batches):
        baseline=None;pair=None
        for condition in ['ce','ce_contr','ce_hier','ce_both']:
            dump(out/'status.json',dict(state='running',batch=b,condition=condition))
            seed_all(4780)
            model=NativeTail2(arm='h',k=cfg.get('k',40),dropout=cfg.get('dropout',.5)).cuda()
            model.load_state_dict(ck['model'],strict=True)
            opt=geoopt.optim.RiemannianSGD(model.parameters(),lr=cfg['lr'],momentum=cfg['momentum'],weight_decay=cfg['weight_decay'])
            opt.load_state_dict(copy.deepcopy(ck['optimizer']))
            criterion=HyCoReSegLoss(feature_dim=256).cuda()
            criterion.generator.set_state(ck['part_generator'].cpu())
            criterion.negative_generator.set_state(ck['negative_generator'].cpu())
            restore_rng(ck['rng'])
            model.train();onehot=torch.nn.functional.one_hot(c,16).float()
            z,u=model(pts,onehot,return_features=True);z=z.transpose(1,2).contiguous()
            reg,info=criterion(u,c)
            ce=torch.nn.functional.cross_entropy(z.reshape(-1,50),y.reshape(-1))
            contr,hier=criterion.last_components['contr'],criterion.last_components['hier']
            loss=ce+(contr if condition in ['ce_contr','ce_both'] else 0)+(hier if condition in ['ce_hier','ce_both'] else 0)
            signature=(info['part_indices_sha256'],info['negative_indices'])
            if pair is None:pair=signature
            else:assert pair==signature
            loss.backward()
            gradient_norm=float(torch.stack([p.grad.detach().square().sum() for p in model.parameters() if p.grad is not None]).sum().sqrt())
            model.eval()
            checks=[b,(b+1)%len(batches)];before=[];after=[]
            with torch.no_grad():
                for j in checks:
                    xp,yy,cc=batches[j]
                    before.append(scores(model(xp,torch.nn.functional.one_hot(cc,16).float()).transpose(1,2),yy,cc))
            # Before/after use the identical BN buffers produced by this one train forward.
            bn_before={n:t.detach().clone() for n,t in model.named_buffers()}
            if baseline is None:baseline=before
            else:assert baseline==before, 'Unmatched pre-step forward'
            lr=opt.param_groups[0]['lr'];opt.step()
            with torch.no_grad():
                for j in checks:
                    xp,yy,cc=batches[j]
                    after.append(scores(model(xp,torch.nn.functional.one_hot(cc,16).float()).transpose(1,2),yy,cc))
            assert all(torch.equal(t,bn_before[n]) for n,t in model.named_buffers())
            row=dict(lr=lr,ce=float(ce.detach()),contr=float(contr.detach()),hier=float(hier.detach()),
                gradient_norm=gradient_norm,before=before,after=after,check_batches=checks,
                sampling_sha256=info['part_indices_sha256'],bn_fixed_during_update_measurement=True)
            result['conditions'][f'b{b}_{condition}']=row
            dump(out/'result.json',result)
            print(b,condition,'done',flush=True)
            del model,opt,criterion,z,u,reg,ce,contr,hier,loss
            torch.cuda.empty_cache()
    assert sha(cp)==checkpoint_sha and state_hash(ck['model'])==original_hash
    result['input_checkpoint_unchanged']=True;dump(out/'result.json',result)
    dump(out/'status.json',dict(state='complete',at=time.time()))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['source','data','runs-json','out','batches']:p.add_argument('--'+key,required=True)
    p.add_argument('--gpu',type=int,default=2);a=p.parse_args()
    try:run(a)
    except Exception:
        import traceback
        if Path(a.out).exists():dump(Path(a.out)/'failure.json',dict(traceback=traceback.format_exc()))
        raise
