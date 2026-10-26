"""Small numerical and accounting contracts; no model or optimizer creation."""
from __future__ import annotations
import collections
import hashlib
import json
import math
from pathlib import Path
import numpy as np

ROLES = ('S', 'C', 'T')
NAMES = {'S':'Same-class pair', 'C':'Cross-class pair', 'T':'Triple', 'P':'Pair union', 'Used':'Selected union', 'All':'All 512'}
CLASSES = ('airplane bathtub bed bench bookshelf bottle bowl car chair cone cup curtain desk door dresser flower_pot glass_box guitar keyboard lamp laptop mantel monitor night_stand person piano plant radio range_hood sink sofa stairs stool table tent toilet tv_stand vase wardrobe xbox').split()


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4<<20),b''):h.update(b)
    return h.hexdigest()


def write(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def stats(x,weights=None):
    x=np.asarray(x,dtype=np.float64).reshape(-1)
    if weights is None:
        if not len(x):return {'count':0,'median':None,'p25':None,'p75':None,'min':None,'max':None,'mean':None}
        if not np.isfinite(x).all():raise ValueError('Nonfinite statistic')
        q=np.quantile(x,[0,.05,.25,.5,.75,.95,1])
        return dict(count=len(x),min=float(q[0]),p05=float(q[1]),p25=float(q[2]),median=float(q[3]),p75=float(q[4]),p95=float(q[5]),max=float(q[6]),mean=float(x.mean()))
    w=np.asarray(weights,dtype=np.float64).reshape(-1)
    if len(x)!=len(w) or np.any(w<0) or not np.isfinite(w).all():raise ValueError('Invalid weights')
    keep=w>0
    if not keep.any():return dict(count=0,mass=0.,median=None,p25=None,p75=None,min=None,max=None,mean=None)
    x,w=x[keep],w[keep];order=np.argsort(x,kind='stable');x,w=x[order],w[order]
    cumulative=np.cumsum(w);q=x[np.searchsorted(cumulative,np.array([.25,.5,.75])*cumulative[-1],side='left')]
    return dict(count=len(x),mass=float(w.sum()),min=float(x[0]),p25=float(q[0]),median=float(q[1]),p75=float(q[2]),max=float(x[-1]),mean=float(np.average(x,weights=w)))


def unit(x):
    x=np.asarray(x,dtype=np.float64);norm=np.linalg.norm(x,axis=1)
    if np.any(norm<=1e-8) or not np.isfinite(x).all():raise ValueError('Undefined direction')
    return x/norm[:,None]


def average_ranks(x):
    x=np.asarray(x);order=np.argsort(x,kind='stable');s=x[order];start=np.r_[0,np.flatnonzero(np.diff(s)!=0)+1];end=np.r_[start[1:],len(s)]
    result=np.empty(len(x),float)
    for a,b in zip(start,end):result[order[a:b]]=(a+b-1)/2
    return result


def depth(x,c=.1):
    r=np.linalg.norm(np.asarray(x,dtype=np.float64),axis=1)*math.sqrt(c)
    if np.any(r>=1):raise ValueError('Outside ball')
    return 2/math.sqrt(c)*np.arctanh(r)


def nearest_centers(native,labels,proxies):
    u=unit(native);centers=np.stack([u[np.asarray(labels)==c].mean(0) for c in range(40)])
    centers=unit(centers);angles=np.degrees(np.arccos(np.clip(unit(proxies)@centers.T,-1,1)))
    nearest=angles.argmin(1)
    return nearest,angles[np.arange(len(proxies)),nearest],centers


def roles(counts):
    counts=np.asarray(counts)
    if counts.ndim!=2 or counts.shape[0]!=3:raise ValueError('Require S,C,T counts')
    mask=counts>0;bits=mask.astype(np.int64).T@np.array([1,2,4])
    active=bits>0
    return mask,bits,active


def role_summary(counts,pure):
    mask,bits,active=roles(counts);m=int(active.sum());pure=np.asarray(pure,dtype=bool)
    result=dict(selected=m,unused=int((~active).sum()),all_proxies=len(pure),pure_all_count=int(pure.sum()),pure_all_fraction=float(pure.mean()),
                pure_selected_count=int((pure&active).sum()),pure_selected_fraction=float((pure&active).sum()/m) if m else None,groups={},exclusive=[])
    for i,role in enumerate(ROLES):
        n=int(mask[i].sum());result['groups'][role]=dict(proxies=n,of_selected=n/m if m else None,of512=n/len(pure),pure_count=int((pure&mask[i]).sum()),pure_fraction=float((pure&mask[i]).sum()/n) if n else None)
    for b in range(1,8):
        n=int((bits==b).sum());result['exclusive'].append(dict(bits=b,roles='+'.join(ROLES[i] for i in range(3) if b&(1<<i)),count=n,of_selected=n/m if m else None))
    assert sum(x['count'] for x in result['exclusive'])==m
    return result


def gini(x):
    x=np.sort(np.asarray(x,dtype=np.float64));total=x.sum()
    return float((2*np.dot(np.arange(1,len(x)+1),x)/(len(x)*total))-(len(x)+1)/len(x)) if total else None


def coverage(nn4,selected,n):
    selected=np.asarray(selected,dtype=np.int64);neighbors=np.asarray(nn4)[selected]
    if len(selected)==0:return dict(proxies=0,unique_objects=0,coverage_fraction=0.,capacity_normalized=None,counts=np.zeros(n,dtype=np.int64))
    if any(len(set(row))!=4 for row in neighbors):raise ValueError('Top4 must have distinct IDs')
    h=np.bincount(neighbors.ravel(),minlength=n);positive=h[h>0];slots=4*len(selected);prob=positive/slots
    quartets=collections.Counter(tuple(sorted(row)) for row in neighbors)
    return dict(proxies=len(selected),slots=slots,unique_objects=len(positive),coverage_fraction=len(positive)/n,
                capacity_normalized=len(positive)/min(n,slots),repeated_slot_fraction=1-len(positive)/slots,max_exposure=int(h.max()),
                positive_median=float(np.median(positive)),top1_fraction=float(h.max()/slots),top10_fraction=float(np.sort(h)[-10:].sum()/slots),
                effective_objects=float(np.exp(-np.sum(prob*np.log(prob)))),gini_all=gini(h),gini_covered=gini(positive),
                unique_quartets=len(quartets),max_quartet_multiplicity=max(quartets.values()),counts=h)


def null_maxima(nn4,selected,labels,repeats=1000,seed=4780):
    """Preserve each proxy's quartet label multiset and distinct-ID constraint."""
    labels=np.asarray(labels);rng=np.random.RandomState(seed);hist=np.zeros((repeats,len(labels)),dtype=np.int16);rr=np.arange(repeats)
    for proxy in np.asarray(selected,dtype=int):
        quartet=labels[nn4[proxy]]
        for c in np.unique(quartet):
            pool=np.flatnonzero(labels==c);k=int((quartet==c).sum())
            picks=rng.randint(len(pool),size=(repeats,k))
            duplicate=np.any(np.diff(np.sort(picks,axis=1),axis=1)==0,axis=1) if k>1 else np.zeros(repeats,bool)
            while duplicate.any():
                picks[duplicate]=rng.randint(len(pool),size=(int(duplicate.sum()),k))
                duplicate=np.any(np.diff(np.sort(picks,axis=1),axis=1)==0,axis=1)
            for j in range(k):np.add.at(hist,(rr,pool[picks[:,j]]),1)
    return hist.max(1).astype(np.int64)


def depth_bands(h):
    """An independent, predeclared three-bandwidth KDE stability diagnostic."""
    h=np.asarray(h,dtype=float);grid=np.linspace(h.min()-.05,h.max()+.05,1024);bw=1.06*np.std(h)*len(h)**(-.2)
    if bw<=1e-10:return dict(stable_two_bands=False,threshold=None,reason='constant depth')
    findings=[];curves=[]
    for scale in (.8,1.,1.2):
        z=(grid[:,None]-h[None,:])/(bw*scale);den=np.exp(-.5*z*z).mean(1)/(bw*scale*np.sqrt(2*np.pi));curves.append(den)
        peaks=np.flatnonzero((den[1:-1]>den[:-2])&(den[1:-1]>=den[2:]))+1;peaks=peaks[den[peaks]>=.1*den.max()]
        threshold=None
        if len(peaks)==2:
            lo,hi=peaks;valley=lo+np.argmin(den[lo:hi+1]);t=float(grid[valley]);mass=float((h<t).mean())
            if .1<=mass<=.9 and min(den[lo],den[hi])-den[valley]>=.05*den.max():threshold=t
        findings.append(dict(bandwidth_scale=scale,significant_modes=len(peaks),threshold=threshold))
    ts=[x['threshold'] for x in findings];stable=all(t is not None for t in ts) and max(ts)-min(ts)<=.1
    return dict(stable_two_bands=bool(stable),threshold=ts[1] if stable else None,bandwidth=float(bw),method='Silverman KDE x0.8/1/1.2, 10% peak/mass, 5% valley contrast, threshold span<=0.1',checks=findings,grid=grid.tolist(),density=[x.tolist() for x in curves])


def examples(mask,h,counts,pure):
    """Prespecified depth-quartile cases, purity diversity where available."""
    selected=[];used=set()
    for ri,(role,quota) in enumerate((('S',2),('C',1),('T',1))):
        order=np.flatnonzero(mask[ri]);order=order[np.lexsort((order,h[order]))]
        for quartile,pool in enumerate(np.array_split(order,4)):
            picked=[]
            for purity in ([True,False] if quota==2 else [None]):
                choices=[int(p) for p in pool if p not in used and p not in picked and (purity is None or bool(pure[p])==purity)]
                if not choices:choices=[int(p) for p in pool if p not in used and p not in picked]
                if not choices:choices=[int(p) for p in pool if p not in picked]
                if not choices:continue
                med=np.median(counts[ri,choices]);p=min(choices,key=lambda x:(abs(counts[ri,x]-med),x));picked.append(p)
            for p in picked:selected.append(dict(role=role,proxy_id=p,depth_quartile=quartile+1,reused_across_roles=p in used));used.add(p)
    return selected
