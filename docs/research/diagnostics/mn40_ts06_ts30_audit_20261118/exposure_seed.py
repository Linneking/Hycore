"""CPU-only conditional exposure checks; alternative seeds never train models."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from scipy.stats import chi2, rankdata, spearmanr
from audit import NAMES, SourceClassBatchSampler, write
from analyze import arrays, csv_file

SEEDS=(0,1,2,3,7,22,42,123,2026,2027,4780,4781)


def read(path):
    with Path(path).open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))


def bh(p):
    p=np.asarray(p);order=np.argsort(p);q=np.minimum.accumulate((p[order]*len(p)/np.arange(1,len(p)+1))[::-1])[::-1]
    result=np.empty(len(p));result[order]=np.minimum(q,1);return result


def class_percentiles(values,labels):
    pct=np.empty(len(values))
    for c in range(40):
        mask=labels==c;pct[mask]=(rankdata(values[mask])-.5)/mask.sum()
    return pct


def main():
    p=argparse.ArgumentParser();p.add_argument('--inputs',type=Path,required=True)
    p.add_argument('--delivery',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--simulations',type=int,default=4000);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    ident=arrays(a.inputs/'identity.npz');s=arrays(a.inputs/'TS30_sampling.npz');labels=ident['train_labels']
    rng=np.random.default_rng(20261118);all_rows=[];tails=[];counts_by_tag={}
    for tag,key in (('best','best_counts'),('e300','final_counts')):
        counts=s[key];counts_by_tag[tag]=counts
        errors=read(a.delivery/f'{tag}_train_objects.csv')
        assert [r['id'] for r in errors]==ident['train_ids'].tolist()
        cb_wrong=np.array([r['TS30_prediction']!=r['label'] for r in errors])
        sh_wrong=np.array([r['TS06_prediction']!=r['label'] for r in errors])
        for c,name in enumerate(NAMES):
            mask=labels==c;v=counts[mask];n=len(v);total=int(v.sum());mu=v.mean()
            simulated=rng.multinomial(total,np.full(n,1/n),size=a.simulations)
            stat=float(((v-mu)**2/mu).sum());null=((simulated-mu)**2/mu).sum(1)
            row=dict(tag=tag,name=name,n=n,total_draws=total,minimum=int(v.min()),mean=float(mu),maximum=int(v.max()),
                cv=float(v.std()/mu),expected_uniform_cv=float(np.sqrt((1-1/n)/mu)),
                overdispersion=stat/(n-1),p_overdispersion_mc=float((1+(null>=stat).sum())/(1+a.simulations)),
                p_overdispersion_chi2=float(chi2.sf(stat,n-1)),
                uniform_min_95_low=float(np.quantile(simulated.min(1),.025)),uniform_min_95_high=float(np.quantile(simulated.min(1),.975)),
                uniform_max_95_low=float(np.quantile(simulated.max(1),.025)),uniform_max_95_high=float(np.quantile(simulated.max(1),.975)),
                cb_clean_train_wrong=int(cb_wrong[mask].sum()),sh_clean_train_wrong=int(sh_wrong[mask].sum()))
            all_rows.append(row)
        pct=class_percentiles(counts,labels)
        for event,event_mask in (('cb_wrong',cb_wrong),('sh_correct_cb_wrong',cb_wrong&~sh_wrong)):
            for bin_name,bmask in (('low10',pct<=.1),('high10',pct>=.9)):
                observed=int((event_mask&bmask).sum());expected=0.;null=np.zeros(a.simulations,dtype=int)
                for c in range(40):
                    mask=labels==c;n=int(mask.sum());k=int(event_mask[mask].sum());b=int(bmask[mask].sum())
                    expected+=k*b/n;null+=rng.hypergeometric(k,n-k,b,size=a.simulations)
                tails.append(dict(tag=tag,event=event,bin=bin_name,objects=int(bmask.sum()),errors=observed,
                    error_rate=float(observed/bmask.sum()),class_conditional_expected_errors=expected,
                    p_two_sided_mc=float((1+(np.abs(null-expected)>=abs(observed-expected)).sum())/(a.simulations+1))))
    for row,q in zip(all_rows,bh([r['p_overdispersion_mc'] for r in all_rows])):row['q_bh_80']=float(q)
    for row,q in zip(tails,bh([r['p_two_sided_mc'] for r in tails])):row['q_bh_8']=float(q)
    csv_file(a.out/'within_class.csv',all_rows);csv_file(a.out/'tail_errors.csv',tails)

    blocks=np.stack([s['epoch_counts'][j*100:(j+1)*100].sum(0) for j in range(3)])
    pct=np.stack([class_percentiles(x,labels) for x in blocks])
    persistence=dict(rank_correlations={f'{i+1}_{j+1}':float(spearmanr(pct[i],pct[j]).statistic) for i,j in ((0,1),(0,2),(1,2))})
    for tail,mask in (('high10',pct>=.9),('low10',pct<=.1)):
        expected=sum(np.prod(mask[:,labels==c].sum(1))/(int((labels==c).sum())**2) for c in range(40))
        persistence[tail]=dict(all_three_blocks=int(mask.all(0).sum()),independent_assignment_expectation=float(expected))

    seed_rows=[];seed_counts=[]
    for seed in SEEDS:
        sampler=SourceClassBatchSampler(labels,world_size=1,classes_per_rank=16,instances_per_class=2,steps=200,seed=seed)
        count=np.zeros(len(labels),dtype=np.int64);best_count=None
        for epoch in range(300):
            sampler.set_epoch(epoch);batch=np.array(list(sampler));count+=np.bincount(batch.ravel(),minlength=len(labels))
            if epoch==261:best_count=count.copy()
        if seed==4780:
            assert np.array_equal(count,s['final_counts']) and np.array_equal(best_count,s['best_counts'])
        pooled=0.;max_z=0.
        row=dict(seed=seed,mode='sampler_only_no_model_training')
        for c,name in enumerate(NAMES):
            v=count[labels==c];mu=v.mean();pooled+=float(((v-mu)**2/mu).sum())
            max_z=max(max_z,float(np.abs((v-mu)/np.sqrt(mu*(1-1/len(v)))).max()))
            if name in ('table','dresser','vase','chair','bowl','xbox'):
                row[name+'_min']=int(v.min());row[name+'_mean']=float(mu);row[name+'_max']=int(v.max())
        row.update(pooled_overdispersion=pooled/(len(labels)-40),max_abs_within_class_z=max_z)
        seed_rows.append(row);seed_counts.append(count);print('sample seed',seed,'completed',flush=True)
    csv_file(a.out/'sampler_seeds.csv',seed_rows)
    np.savez_compressed(a.out/'sampler_seed_counts.npz',seeds=SEEDS,counts=np.stack(seed_counts),ids=ident['train_ids'])

    # Count high/low-exposure training neighbours of the already classified test objects.
    percentiles=class_percentiles(s['best_counts'],labels);pid={str(k):i for i,k in enumerate(ident['train_ids'])}
    neighbours=read(a.delivery/'neighbours.csv');by_query={}
    for r in neighbours:
        if int(r['rank'])<=5:by_query.setdefault(r['test_id'],[]).append(percentiles[pid[r['train_id']]])
    objects=read(a.delivery/'best_test_objects.csv');group={r['id']:r['group'] for r in objects}
    neighbour_tails={}
    for g in ('sh_only','cb_only','both_wrong','both_correct'):
        v=[p for k,vs in by_query.items() if group[k]==g for p in vs]
        neighbour_tails[g]=dict(training_neighbour_slots=len(v),low10_fraction=float(np.mean(np.array(v)<=.1)),high10_fraction=float(np.mean(np.array(v)>=.9)))
    out=dict(simulations=a.simulations,conditional_null='within each class condition on the recorded total draws; uniform independent objects, multinomial',
        within_class=all_rows,tail_errors=tails,persistent_id_preference=persistence,sampler_seeds=seed_rows,
        significant_dispersion=[r for r in all_rows if r['q_bh_80']<.05],significant_error_tails=[r for r in tails if r['q_bh_8']<.05],
        neighbour_tails=neighbour_tails,no_model_training=True)
    write(a.out/'summary.json',out)
    print(json.dumps({k:v for k,v in out.items() if k not in ('within_class','sampler_seeds')}),flush=True)


if __name__=='__main__':main()
