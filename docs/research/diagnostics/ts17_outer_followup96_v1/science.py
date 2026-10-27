"""Small pure-array contracts for the three approved outer-proxy questions."""
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'ts17_proxy_roles96_v1'))
from core import sha,write,unit,depth,stats,average_ranks,CLASSES


def sample_pairs(raw,labels,seed=4780,per_class=100):
    keep=raw['distinct']&raw['same']
    pairs=np.unique(np.sort(np.column_stack((raw['i'][keep],raw['j'][keep])),axis=1),axis=0)
    if np.any(pairs[:,0]==pairs[:,1]) or np.any(labels[pairs[:,0]]!=labels[pairs[:,1]]):raise ValueError('Invalid same-class pairs')
    rng=np.random.RandomState(seed);chosen=[];available=[]
    for c in range(40):
        candidates=pairs[labels[pairs[:,0]]==c];available.append(len(candidates))
        ix=np.sort(rng.choice(len(candidates),min(per_class,len(candidates)),replace=False))
        chosen.extend(candidates[ix])
    return np.asarray(chosen,dtype=np.int64),available


def at_depth(direction,h):
    u=unit(direction);h=np.asarray(h,dtype=np.float64).reshape(-1)
    if len(h)!=len(u) or np.any(h<=0) or not np.isfinite(h).all():raise ValueError('Invalid radial replacement')
    return u*(np.tanh(np.sqrt(.1)*h/2)/np.sqrt(.1))[:,None]


def angular_rows(a,b):
    return np.degrees(np.arccos(np.clip(np.sum(unit(a)*unit(b),axis=1),-1,1)))


def rank_correlation(a,b):
    a=np.asarray(a);b=np.asarray(b);ok=np.isfinite(a)&np.isfinite(b)
    if ok.sum()<3 or np.unique(a[ok]).size<2 or np.unique(b[ok]).size<2:return None
    return float(np.corrcoef(average_ranks(a[ok]),average_ranks(b[ok]))[0,1])


def nearest_angle(proxies,whole,labels):
    centers=unit(np.stack([unit(whole[labels==c]).mean(0) for c in range(40)]))
    angles=np.degrees(np.arccos(np.clip(unit(proxies)@centers.T,-1,1)))
    return angles.min(1),angles.argmin(1),centers


def equal_class_mean(values,pair_labels):
    return float(np.mean([np.mean(values[pair_labels==c]) for c in np.unique(pair_labels)]))


def choose_cases(gaps,classes):
    result=[];seen_classes=set();used=set();order=np.argsort(gaps,kind='stable')
    for third in np.array_split(order,3):
        target=float(np.median(gaps[third]));priority=sorted(third,key=lambda i:(abs(gaps[i]-target),int(i)))
        for _ in range(2):
            candidates=[int(i) for i in priority if int(i) not in used]
            diverse=[i for i in candidates if int(classes[i]) not in seen_classes]
            if not candidates:continue
            i=(diverse or candidates)[0];result.append(i);used.add(i);seen_classes.add(int(classes[i]))
    return result
