"""Read-only production FP32 reverse retrieval and exact source proxy mapping.

Input is a completed full-pool snapshot spec. Writes a fresh private directory;
never re-encodes an object, changes a checkpoint or invokes an optimizer.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys
from collections import Counter
import numpy as np


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for b in iter(lambda:stream.read(1<<20),b''): h.update(b)
    return h.hexdigest()


def run(specfile,output,epochs):
    import torch
    sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
    from tools.hier_postrun_audit.mechanisms import _kernel
    torch.set_num_threads(2)
    out=Path(output)
    out.mkdir(parents=True,exist_ok=False)
    specs=json.loads(Path(specfile).read_text())['snapshots']
    selected=[s for s in specs if s['epoch'] in epochs]
    if {s['epoch'] for s in selected}!=set(epochs): raise ValueError('Missing requested actual checkpoint')
    records=[]
    for s in selected:
        before=sha(s['cache'])
        with np.load(s['cache'],allow_pickle=False) as z:
            a={k:z[k].copy() for k in z.files}
        if s.get('version')!='v8': raise ValueError('This explicit production adapter is V8 only')
        _,_,distance,expmap,_=_kernel({'version':'v8'})
        tangent=torch.from_numpy(a['proxy_tangent']).float()
        whole=torch.from_numpy(a['mu']).float()
        with torch.no_grad():
            proxy=expmap(tangent)
            rows=[]
            for start in range(0,len(proxy),16):
                d=distance(proxy[start:start+16],whole)
                rows.append(torch.topk(d,k=16,largest=False,sorted=True).indices.cpu().numpy())
        rows=np.concatenate(rows)
        coverage=len(np.unique(rows[:,:4]))
        count=Counter(tuple(sorted(row)) for row in rows[:,:4].tolist())
        path=out/('V8_e%03d.npz'%s['epoch'])
        metadata=dict(mapping_config={'clip_r':2.3},split='train',epoch=s['epoch'],
            source_cache_sha256=before,proxy_map='actual production V8 to_ball FP32 CPU',
            distance='actual V8 official Mobius eps1e-5; CPU torch.topk ties, not GPU tie-order guarantee')
        np.savez_compressed(path,mu=a['mu'],proxy_ball=proxy.numpy(),proxy_tangent=a['proxy_tangent'],
            sample_ids=a['sample_ids'],labels=a['labels'],proxy_ids=a['proxy_ids'],
            c=np.asarray(s['c']),metadata_json=np.asarray(json.dumps(metadata)),
            production_top16_indices=rows)
        if sha(s['cache'])!=before: raise RuntimeError('Frozen input changed')
        records.append(dict(name='V8_e%03d'%s['epoch'],epoch=s['epoch'],cache=str(path),
            kind='HyCoRe_V8',split='train',source_cache_sha256=before,converted_sha256=sha(path),
            production_coverage4=coverage,production_max_repeated_quartet=max(count.values()),
            CPU_operator=True,encoder_forwards=0,optimizer_updates=0))
        print(json.dumps(records[-1]),flush=True)
    (out/'spec.json').write_text(json.dumps({'models':records},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--spec',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--epochs',required=True)
    a=p.parse_args()
    run(a.spec,a.output,[int(v) for v in a.epochs.split(',')])
