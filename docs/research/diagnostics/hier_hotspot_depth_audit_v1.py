"""CPU proxy->whole top4 audit, equalizing WHOLE radii only.

Never changes proxies, directions, source cache, or training. Same candidate IDs
and label-free retrieval in every control. Output must be a new directory.
"""
from __future__ import annotations
import argparse
from collections import Counter
import csv
import datetime as dt
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import numpy as np


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()


def stats(x):
    x=np.asarray(x,dtype=float)
    if not len(x):return {'n':0}
    return dict(n=len(x),min=float(x.min()),median=float(np.median(x)),max=float(x.max()),
                mean=float(x.mean()),std_population=float(x.std()))


def score_matrix(proxy,whole,c=1.):
    """Hyperbolic order score. Query-only positive factors can be omitted.

    cosh(sqrt(c)*d)=1+2c*||p-x||²/[(1-c||p||²)(1-c||x||²)].
    Float64 dot products are validated against stable coordinate differences.
    """
    pn=np.sum(proxy**2,axis=1);wn=np.sum(whole**2,axis=1)
    sq=pn[:,None]+wn[None,:]-2*(proxy@whole.T)
    if (sq < -1e-12).any():raise ValueError('Invalid squared distance')
    return np.maximum(sq,0)/(1-c*wn[None,:])


def top4(score):
    # Candidate axis sorted by stable sample ID before calling. Exact ties by ID.
    return np.argsort(score,axis=1,kind='stable')[:,:4]


def summarize(rows,ids,labels,pdepth,pdirection,used,wdepth):
    quartets=Counter(tuple(sorted(ids[r].tolist())) for r in rows)
    counts=Counter(ids[rows].reshape(-1).tolist())
    leading=[]
    for quartet,n in quartets.most_common(12):
        members=np.array([i for i,r in enumerate(rows) if tuple(sorted(ids[r].tolist()))==quartet])
        lo=members[np.argmin(pdepth[members])];hi=members[np.argmax(pdepth[members])]
        angle=float(np.degrees(np.arccos(np.clip(pdirection[lo]@pdirection[hi],-1,1))))
        leading.append(dict(sample_ids=list(quartet),count=n,
            proxy_ids=members.tolist(),proxy_depth=stats(pdepth[members]),
            sample_used_proxy_count=int(used[members].sum()),sample_inactive_proxy_count=int((~used[members]).sum()),
            shallow_depth_le3_count=int((pdepth[members]<=3).sum()),
            deep_depth_ge5_count=int((pdepth[members]>=5).sum()),
            extreme_proxy_pair=dict(min_depth_proxy_id=int(lo),min_depth=float(pdepth[lo]),
                 max_depth_proxy_id=int(hi),max_depth=float(pdepth[hi]),
                 depth_gap=float(pdepth[hi]-pdepth[lo]),angular_separation_deg=angle)))
    gaps=[]
    for q,n in quartets.items():
        m=[i for i,r in enumerate(rows) if tuple(sorted(ids[r].tolist()))==q]
        gaps.append(float(np.ptp(pdepth[m])))
    idrow={int(v):i for i,v in enumerate(ids)}
    hot=[dict(sample_id=int(i),class_label=int(labels[idrow[i]]),slots=n,
              original_whole_depth=float(wdepth[idrow[i]]),
              original_depth_percentile=float(np.mean(wdepth<=wdepth[idrow[i]]))) for i,n in counts.most_common(12)]
    groups={}
    for name,mask in [('sample_used',used),('sample_inactive',~used),
                      ('shallow_depth_le3',pdepth<=3),('deep_depth_ge5',pdepth>=5)]:
        r=rows[mask]
        cq=Counter(tuple(sorted(ids[x].tolist())) for x in r)
        groups[name]=dict(proxy_count=int(mask.sum()),covered_sample_count=len(np.unique(ids[r])),
                         distinct_quartets=len(cq),max_repeated_quartet=max(cq.values(),default=0))
    return dict(proxy_count=len(rows),slot_count=int(rows.size),covered_sample_count=len(counts),
        unique_quartets=len(quartets),max_repeated_quartet=max(quartets.values()),
        top_four_sample_slot_share=sum(sorted(counts.values(),reverse=True)[:4])/rows.size,
        groups=groups,leading_quartets=leading,hottest_samples=hot,
        quartets_with_proxy_depth_span_ge2=sum(g>=2 for g in gaps),
        quartets_with_proxy_depth_span_ge3=sum(g>=3 for g in gaps))


def analyze(repo,cache,normalized,output,epoch=300):
    sys.path.insert(0,str(repo))
    from tools.hier_postrun_audit.geometry import expmap0,ball_geometry,poincare_distance
    out=Path(output)
    if out.exists():raise FileExistsError('Fresh output directory required')
    out.mkdir(parents=True)
    cache=Path(cache);source_sha=sha(cache)
    with np.load(cache,allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
    if float(a['c'])!=1:raise ValueError('Current adapter requires c1')
    so=np.argsort(a['sample_ids'],kind='stable');po=np.argsort(a['proxy_ids'],kind='stable')
    ids=a['sample_ids'][so];labels=a['labels'][so];whole=a['mu'][so].astype(float)
    pids=a['proxy_ids'][po]
    if not np.array_equal(pids,np.arange(len(pids))):raise ValueError('Explicit contiguous production parameter IDs required')
    if len(np.unique(ids))!=len(ids):raise ValueError('Repeated candidate ID')
    proxy=expmap0(a['proxy_tangent'][po],1,numeric_radius_fraction=.999)
    wg,pg=ball_geometry(whole,1),ball_geometry(proxy,1)
    logs=json.loads(Path(normalized).read_text())
    if len(logs)!=1:raise ValueError('Exactly one normalized run required')
    log=logs[0]
    rec=[r for r in log['proxy_usage'] if r['epoch']==epoch and (r['component'],r['domain'],r['role'])==('sample','noncollision','combined')]
    if len(rec)!=1:raise ValueError('Epoch usage absent/ambiguous')
    lookup=dict(zip(rec[0]['proxy_ids'],rec[0]['counts']));used=np.array([lookup[int(i)]>0 for i in pids])
    result=dict(schema='hier_hotspot_depth_audit_v1',epoch=epoch,
        started_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        identity=dict(cache_sha256=source_sha,cache_file=cache.name,normalized_sha256=sha(normalized),
                      input_sha256=str(a['input_sha256']),code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),
                      script_sha256=sha(__file__),run_key=log['run_id']),
        protocol=dict(c=1,seed=22,candidate_count=len(ids),proxy_count=len(pids),display_topk=4,
                      retrieval='full geodesic ranking, no label boost/prefilter',GPU=None,teacher=None,
                      activation_scope='same epoch sample noncollision combined hard ancestors',
                      input_condition=str(a['input_mode'])),
        controls={},validation={})
    raw_rows=top4(score_matrix(proxy,whole))
    exact=top4(poincare_distance(proxy[:16],whole,1))
    if not np.array_equal(exact,raw_rows[:16]):raise RuntimeError('Dot score differs from stable geodesic top4')
    result['validation']['stable_distance_first16_proxies_agree']=True
    direction_rows=top4(1-pg['direction']@wg['direction'].T)
    targets={'whole_equal_median':float(np.median(wg['depth'])),
             'whole_equal_depth2':2.,'whole_equal_depth6':6.}
    rowsets={'raw':raw_rows,'direction':direction_rows}
    for name,depth in targets.items():
        moved=wg['direction']*np.tanh(depth/2)
        rowsets[name]=top4(score_matrix(proxy,moved))
        if not np.array_equal(rowsets[name],direction_rows):raise RuntimeError('Whole-only equal-radius order not identical to direction')
        result['controls'][name]=dict(target_whole_depth=depth,proxies_unchanged=True,
            whole_directions_unchanged=True,exact_top4_agreement_with_direction=1.)
    for name,rows in rowsets.items():
        result['controls'].setdefault(name,{})['retrieval']=summarize(rows,ids,labels,pg['depth'],pg['direction'],used,wg['depth'])
    header=['proxy_id','origin_depth','sample_used_this_epoch','raw_top4_ids','equal_whole_top4_ids','direction_top4_ids']
    with (out/'proxy_neighbours.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.writer(f);writer.writerow(header)
        for i in pids:
            writer.writerow([int(i),float(pg['depth'][i]),bool(used[i]),
                json.dumps(ids[raw_rows[i]].tolist()),json.dumps(ids[rowsets['whole_equal_median'][i]].tolist()),json.dumps(ids[direction_rows[i]].tolist())])
    result['read_only']=dict(source_cache_unchanged=sha(cache)==source_sha,optimizer_updates=0,encoder_forwards=0,test_forwards=0)
    if not result['read_only']['source_cache_unchanged']:raise RuntimeError('Source changed')
    (out/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    np.savez_compressed(out/'plot_arrays.npz',proxy_ids=pids,proxy_depth=pg['depth'],used=used,
        raw_quartet_ids=ids[raw_rows],equal_whole_quartet_ids=ids[rowsets['whole_equal_median']],whole_depth=wg['depth'])
    print(json.dumps({name:dict(coverage=v['retrieval']['covered_sample_count'],quartets=v['retrieval']['unique_quartets'],
                   max_repeated=v['retrieval']['max_repeated_quartet']) for name,v in result['controls'].items()}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for flag in ['repo','cache','normalized','output']:p.add_argument('--'+flag,required=True)
    p.add_argument('--epoch',type=int,default=300)
    a=p.parse_args();analyze(a.repo,a.cache,a.normalized,a.output,a.epoch)
