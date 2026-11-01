"""Fixed train-object timeline, isolated BN probe and supervised head gradients.

No optimizer steps. Input checkpoints/data are read-only; all BN buffers restored.
The head gradient probe detaches U: it measures the head->U path, not earlier layers.
"""
import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def dump(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n', encoding='utf8')


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(4194304), b''):
            h.update(chunk)
    return h.hexdigest()


@contextlib.contextmanager
def batch_bn(model):
    import torch
    states = [(m, m.training, m.track_running_stats) for m in model.modules()
              if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
    try:
        for m, _, _ in states:
            m.training = True
            m.track_running_stats = False
        yield
    finally:
        for m, training, track in states:
            m.training, m.track_running_stats = training, track


PARTS = [4, 5, 18, 30, 31, 32, 33, 34, 35, 41, 42, 43, 46, 6, 7, 36, 37]
STARTS = [0,4,6,8,12,16,19,22,24,28,30,36,38,41,44,47]
COUNTS = [4,2,2,4,4,3,3,2,4,2,6,2,3,3,3,3]


def scores(z, y, cats):
    import torch
    owner = torch.repeat_interleave(torch.arange(16, device=z.device),
                                   torch.tensor(COUNTS, device=z.device))
    correct = z.gather(-1, y[..., None]).squeeze(-1)
    wrong = z.clone().scatter_(-1, y[..., None], -float('inf'))
    wrong.masked_fill_(owner[None,None,:] != cats[:,None,None], -float('inf'))
    margin = correct - wrong.max(-1).values
    pred = z.argmax(-1)
    prob = z.softmax(-1).gather(-1,y[...,None]).squeeze(-1)
    result = {}
    for k in PARTS:
        mask = y == k
        if mask.any():
            result[str(k)] = dict(n=int(mask.sum()), tp=int(((pred==k)&mask).sum()),
                pred=int(((pred==k)&(cats[:,None]==owner[k])).sum()),
                recall=float((pred[mask]==k).float().mean()),
                margin=float(margin[mask].mean()), probability=float(prob[mask].mean()))
    return result


def vector_stats(g, x, mask):
    import torch
    g, x = g[mask].detach(), x[mask].detach()
    direction = x / x.norm(dim=-1,keepdim=True).clamp_min(1e-20)
    radial = (g*direction).sum(-1)
    tangential = g-radial[...,None]*direction
    return dict(l2=float(g.norm()), point_mean_l2=float(g.norm(dim=-1).mean()),
        radial_l2=float(radial.norm()), radial_signed_mean=float(radial.mean()),
        angular_l2=float(tangential.norm()), activation_norm=float(x.norm(dim=-1).mean()))


def head_probe(model, u, y, cats):
    import torch
    u = u.detach().requires_grad_(True)
    e = model.manifold.expmap0(u)
    m = model.emb2(e)
    z = model.mlr(m)
    nodes = [u,e,m,z]
    params = [p for layer in (model.emb2, model.mlr) for p in layer.parameters()]
    names = ['emb2.'+n for n,_ in model.emb2.named_parameters()] + ['mlr.'+n for n,_ in model.mlr.named_parameters()]
    losses = torch.nn.functional.cross_entropy(z.reshape(-1,50),y.reshape(-1),reduction='none').reshape_as(y)
    total = torch.autograd.grad(losses.mean(), nodes+params, retain_graph=True)
    output = dict(ce=float(losses.mean()), parts={}, effective_rows={})
    for label, layer in [('emb2',model.emb2),('mlr',model.mlr)]:
        output['effective_rows'][label] = dict(raw=layer._weight.detach().norm(dim=-1).tolist(),
            effective=layer.weight.detach().norm(dim=-1).tolist(), bias_parameter_norm=float(layer._bias.detach().norm()))
    for k in PARTS:
        mask = y == k
        if not mask.any():
            continue
        # This is the actual contribution with denominator B*N, not a reweighted part loss.
        loss = (losses*mask).sum()/y.numel()
        gs = torch.autograd.grad(loss, params, retain_graph=True)
        par = {}
        for name, g, full in zip(names,gs,total[4:]):
            rest = full-g
            par[name] = dict(part_norm=float(g.norm()), other_norm=float(rest.norm()),
                total_norm=float(full.norm()), cosine_part_other=float((g*rest).sum()/(g.norm()*rest.norm()).clamp_min(1e-30)))
        output['parts'][str(k)] = dict(n=int(mask.sum()), batch_point_fraction=float(mask.float().mean()),
            nodes={name:vector_stats(g,x,mask) for name,g,x in zip(['U','EXP','M128','Z50'],total[:4],nodes)}, parameters=par)
    return output


def run(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(a.gpu)
    sys.path.insert(0,a.source)
    import numpy as np
    import h5py
    import torch
    import geoopt
    from segmentation_ShapeNetPart.hycore_seg_v1.train import ensure_idle
    from segmentation_ShapeNetPart.hycore_seg_v1.utils import state_hash,seed_all
    from segmentation_ShapeNetPart.hycore_seg_v1.model import DGCNNPartSeg
    from segmentation_ShapeNetPart.hycore_seg_v6.model import NativeTail2
    ensure_idle(a.gpu)
    torch.set_num_threads(4)
    seed_all(4780)
    runs=json.loads(Path(a.runs_json).read_text(encoding='utf-8-sig'))
    manifest=dict(started=time.time(), seed=4780, gpu=a.gpu, script_sha=sha(__file__),
        source_commit=subprocess.check_output(['git','-C',a.source,'rev-parse','HEAD'],text=True).strip(),
        torch=torch.__version__,geoopt=geoopt.__version__,optimizer_steps=0, teacher=None,
        diagnostic_batches='three fixed train batches, four/category of seven focus categories plus four random others',
        timeline_epochs=[0,1,5,10,20,30,40,50,100,150,200])
    with h5py.File(Path(a.data)/'trainval.h5','r') as f:
        cats=f['label'][:].reshape(-1)
        rng=np.random.default_rng(4780)
        batches=[]
        for b in range(3):
            idx=np.concatenate([rng.choice(np.flatnonzero(cats==c),4,replace=False) for c in [1,13,10,14,5,2,11]]+
                               [rng.choice(np.flatnonzero(~np.isin(cats,[1,13,10,14,5,2,11])),4,replace=False)])
            rng.shuffle(idx)
            # h5py requires sorted indices; materialize then restore identical batch order.
            order=np.argsort(idx); inverse=np.argsort(order); sidx=idx[order]
            pts=f['data'][sidx,:2048].astype('float32')[inverse]
            yy=f['pid'][sidx,:2048].astype('int64')[inverse]
            batches.append((torch.from_numpy(pts).cuda().transpose(1,2).contiguous(),
                torch.from_numpy(yy).cuda(),torch.from_numpy(cats[idx].astype('int64')).cuda()))
            manifest.setdefault('batches',[]).append(dict(indices=idx.tolist(),ids=f['ids'].asstr()[sidx][inverse].tolist()))
    manifest['data_sha256']=sha(Path(a.data)/'trainval.h5')
    dump(out/'manifest.json',manifest)
    jobs=[]
    for entry in runs:
        if entry['name'] not in ['DGCNN_best','H0_best','Hrepeat_best']:
            continue
        jobs.append((entry['name'],entry['kind'],Path(entry['checkpoint']),True))
        if entry['name']!='DGCNN_best':
            for ep in manifest['timeline_epochs']:
                cp=Path(entry['checkpoint']).parent/('e0.pth' if ep==0 else f'epoch_{ep:03d}.pth')
                if cp.exists(): jobs.append((entry['name'].replace('_best','')+f'_e{ep}',entry['kind'],cp,False))
    results={}
    for name,kind,cp,final in jobs:
        dump(out/'status.json',dict(state='running',job=name,at=time.time()))
        ck=torch.load(cp,map_location='cpu',weights_only=False)
        config=ck['config']; kw=dict(k=config.get('k',40),dropout=config.get('dropout',.5))
        model=(DGCNNPartSeg(head='euclidean',**kw) if kind=='dgcnn' else NativeTail2(arm='h',**kw))
        model.load_state_dict(ck['model'],strict=True)
        initial=state_hash(model.state_dict()); epoch=ck.get('epoch'); del ck
        model.cuda().eval()
        item=dict(epoch=epoch,checkpoint_sha256=sha(cp),batches=[])
        for b,(points,y,c) in enumerate(batches):
            onehot=torch.nn.functional.one_hot(c,16).float()
            row={}
            with torch.no_grad():
                z=model(points,onehot).transpose(1,2)
                row['eval']=scores(z,y,c)
                if final:
                    with batch_bn(model):
                        z=model(points,onehot).transpose(1,2)
                        row['batch_bn']=scores(z,y,c)
                    if kind!='dgcnn':
                        u=model.forward_features(points,onehot)
            if final and kind!='dgcnn':
                row['gradient']=head_probe(model,u,y,c)
            item['batches'].append(row)
        assert initial==state_hash(model.state_dict()), 'Model/BN state mutated'
        item['state_unchanged']=True
        dump(out/(name+'.json'),item)
        results[name]=dict(epoch=epoch,state_unchanged=True)
        dump(out/'index.json',results)
        print(name,epoch,'done',flush=True)
        del model
        torch.cuda.empty_cache()
    dump(out/'status.json',dict(state='complete',at=time.time(),jobs=len(jobs)))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['source','data','runs-json','out']: p.add_argument('--'+key,required=True)
    p.add_argument('--gpu',type=int,default=2)
    a=p.parse_args()
    try: run(a)
    except Exception:
        import traceback
        if Path(a.out).exists(): dump(Path(a.out)/'failure.json',dict(traceback=traceback.format_exc()))
        raise
