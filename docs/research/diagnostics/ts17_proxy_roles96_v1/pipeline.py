"""Fixed-cache GPU geometry/readout. No encoder forward, backward or optimizer."""
from __future__ import annotations
import argparse,csv,json,os,sys,time
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
REPO=HERE.parents[3]
sys.path[:0]=[str(HERE),str(REPO),str(REPO/'inter_hierarchy_MN40')]
from core import *

CP_SHA='d10144896c69ffa215a6622e4077a05749e3547c0fd606a546b0f750ae4af61b'
CACHE_SHA='b0142fac9d4ce59ddfacf44a9600ccbcce322e9311544a0dca25ce0e5642bfde'


def csvout(path,rows):
    if not rows:return
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)


def readout(settings):
    import torch
    from inter_hierarchy_MN40.hier_proxy_scratch_v8.geometry import official_distance,to_ball
    from inter_hierarchy_MN40.hier_proxy_cycle_v9c.geometry import whole_to_hier
    from inter_hierarchy_MN40.hier_proxy_scratch_v5.sampler import SourceClassBatchSampler
    from inter_hierarchy_MN40.hier_proxy_scratch_v5.relations import mine_reciprocal_triplets
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    root=Path(settings['root']);private=root/'private';public=root/'public';private.mkdir();public.mkdir()
    state=root/'analysis_status.json';started=time.time()
    def progress(stage,**kw):write(state,dict(status='running',stage=stage,elapsed_seconds=time.time()-started,**kw))
    progress('input_verification')
    cp,cache=Path(settings['checkpoint']),Path(settings['cache'])
    if sha(cp)!=CP_SHA or sha(cache)!=CACHE_SHA:raise RuntimeError('Checkpoint/cache SHA changed')
    saved=torch.load(cp,map_location='cpu',weights_only=False)
    if saved.get('completed_epochs')!=273:raise RuntimeError('Only best e273 is allowed')
    with np.load(cache,allow_pickle=False) as z:data={k:z[k].copy() for k in ('native_mu','hier_mu','labels','canonical_ids','proxy_ball','proxy_tangent','nn4')}
    ids=data['canonical_ids'];labels=data['labels'].astype(np.int64);native=data['native_mu'];tangent=data['proxy_tangent']
    if native.shape!=(9840,256) or tangent.shape!=(512,256) or not np.array_equal(ids,np.arange(9840)):raise RuntimeError('Cache population/ID contract')
    if not np.array_equal(saved['proxy']['tangent_proxies'].cpu().numpy(),tangent):raise RuntimeError('Proxy checkpoint/cache mismatch')
    device=torch.device('cuda:0');p=512;n=9840
    with torch.no_grad():
        whole=whole_to_hier(torch.from_numpy(native).to(device));proxy=to_ball(torch.from_numpy(tangent).to(device),hook=False)
        shell=whole.cpu().numpy();ball=proxy.cpu().numpy()
        shell_error=float(np.max(np.abs(shell-data['hier_mu'])));proxy_error=float(np.max(np.abs(ball-data['proxy_ball'])))
        if shell_error>3e-6 or proxy_error>3e-6:raise RuntimeError('Actual forward mapping differs from trusted cache')
        if np.max(np.abs(depth(shell)-4.6))>1e-5:raise RuntimeError('Whole must be depth4.6')
        h=depth(ball);near,theta,centers=nearest_centers(native,labels,ball)
        progress('distance_tables')
        dsp=official_distance(whole,proxy).detach()
        # Literal official numerics are weakly asymmetric. Reverse retrieval
        # must be computed explicitly, never by transposing the forward table.
        dpw=official_distance(proxy,whole,block=4).cpu().numpy()
        nn4=np.argsort(dpw,axis=1,kind='stable')[:,:4]
        old=data['nn4'];old_cost=np.sort(np.take_along_axis(dpw,old,axis=1),axis=1);new_cost=np.sort(np.take_along_axis(dpw,nn4,axis=1),axis=1)
        top4_cost_error=float(np.max(np.abs(old_cost-new_cost)))
        if top4_cost_error>2e-5:raise RuntimeError('Top4 cannot reproduce trusted85 nearest distances')
        pure=np.all(labels[nn4]==labels[nn4[:,0]][:,None],axis=1)
        counts=np.zeros((3,p),np.int64);all_counts=counts.copy();effective=counts.copy()
        expected=np.zeros((3,p),np.float64);expected_effective=expected.copy();per_seed=np.zeros((5,3,p),np.int64)
        raw={k:[] for k in ('i','j','k','pair','triple','same','distinct','seed','batch')}
        audit_counts=dict(draws=0,distinct_draws=0,collisions_all=0,collisions_distinct=0,near_tie_pair=0,near_tie_triple=0,expected_collisions=0.)
        alignment=np.zeros((3,2),np.int64)

        def choose(distance,triplets):
            i,j,k=triplets.T
            ac=torch.maximum(distance[i],distance[j]);bc=torch.maximum(ac,distance[k])
            a=ac.argmin(1);b=bc.argmin(1)
            pa=torch.softmax(-ac/.1,dim=1);pb=torch.softmax(-bc/.1,dim=1)
            ta=torch.topk(ac,2,dim=1,largest=False).values;tb=torch.topk(bc,2,dim=1,largest=False).values
            return a,b,pa,pb,int(((ta[:,1]-ta[:,0])<=1e-5).sum()),int(((tb[:,1]-tb[:,0])<=1e-5).sum())

        for si,seed in enumerate(range(4780,4785)):
            sampler=SourceClassBatchSampler(labels,world_size=2,classes_per_rank=16,instances_per_class=2,steps=200,seed=seed,rank=0)
            for step,plan in enumerate(sampler.iter_global_plans(0)):
                batch=np.asarray(plan.indices);ix=torch.as_tensor(batch,device=device);local=whole[ix];gold=labels[batch]
                ds=official_distance(local);sim=torch.exp(-ds)+(torch.as_tensor(gold,device=device)[:,None]==torch.as_tensor(gold,device=device)[None,:]).float()
                mined=mine_reciprocal_triplets(sim,20,50,seed*10000+step,True,labels=gold,data_ids=batch)
                triples=mined['triplets'];arr=triples.cpu().numpy();row_ids=batch[arr]
                a,b,pa,pb,tiea,tieb=choose(dsp[ix],triples);aa=a.cpu().numpy();bb=b.cpu().numpy()
                same=gold[arr[:,0]]==gold[arr[:,1]];distinct=(row_ids[:,0]!=row_ids[:,1])&(row_ids[:,0]!=row_ids[:,2])&(row_ids[:,1]!=row_ids[:,2]);valid=distinct&(aa!=bb)
                audit_counts['draws']+=len(aa);audit_counts['distinct_draws']+=int(distinct.sum());audit_counts['collisions_all']+=int((aa==bb).sum());audit_counts['collisions_distinct']+=int((distinct&(aa==bb)).sum());audit_counts['near_tie_pair']+=tiea;audit_counts['near_tie_triple']+=tieb
                audit_counts['expected_collisions']+=float((pa*pb)[torch.as_tensor(distinct,device=device)].sum())
                for ri,(which,group) in enumerate(((aa,same),(aa,~same),(bb,np.ones(len(aa),bool)))):
                    all_counts[ri]+=np.bincount(which[group],minlength=p)
                    values=np.bincount(which[group&distinct],minlength=p);counts[ri]+=values;per_seed[si,ri]+=values
                    effective[ri]+=np.bincount(which[group&valid],minlength=p)
                    mask=torch.as_tensor(group&distinct,device=device);prob=pa if ri<2 else pb;other=pb if ri<2 else pa
                    expected[ri]+=prob[mask].sum(0,dtype=torch.float64).cpu().numpy()
                    expected_effective[ri]+=(prob*(1-other))[mask].sum(0,dtype=torch.float64).cpu().numpy()
                    hit=(near[which]==gold[arr[:,0]])|(near[which]==gold[arr[:,1]])
                    if ri==0:hit=near[which]==gold[arr[:,0]]
                    if ri==2:hit=hit|(near[which]==gold[arr[:,2]])
                    alignment[ri]+=[int((hit&group&distinct).sum()),int((group&distinct).sum())]
                for key,value in zip(('i','j','k','pair','triple','same','distinct'),(row_ids[:,0],row_ids[:,1],row_ids[:,2],aa,bb,same,distinct)):raw[key].append(np.asarray(value,dtype=bool if key in ('same','distinct') else np.int32))
                raw['seed'].append(np.full(len(aa),si,np.uint8));raw['batch'].append(np.full(len(aa),step,np.uint8))
                if step%25==0:progress('sample_roles',seed=seed,batch=step+1,total_draws=audit_counts['draws'])
        # The proxy-internal term is deliberately outside S/C/T denominators.
        progress('proxy_internal_roles');dp=official_distance(proxy);internal=np.zeros((2,p),np.int64)
        internal_expected=np.zeros((2,p),np.float64)
        for seed in range(4780,4785):
            tr=mine_reciprocal_triplets(torch.exp(-dp),20,50,seed,True)['triplets']
            for chunk in tr.split(4096):
                a,b,pa,pb,_,_=choose(dp,chunk)
                internal[0]+=np.bincount(a.cpu().numpy(),minlength=p);internal[1]+=np.bincount(b.cpu().numpy(),minlength=p)
                internal_expected[0]+=pa.sum(0,dtype=torch.float64).cpu().numpy();internal_expected[1]+=pb.sum(0,dtype=torch.float64).cpu().numpy()

    raw={k:np.concatenate(v) for k,v in raw.items()};np.savez_compressed(private/'relations.npz',**raw)
    mask,bits,active=roles(counts);sets={r:np.flatnonzero(mask[i]) for i,r in enumerate(ROLES)};sets.update(P=np.flatnonzero(mask[:2].any(0)),Used=np.flatnonzero(active),All=np.arange(512))
    bands=depth_bands(h);threshold=bands['threshold'];split=(h>=threshold) if threshold is not None else None
    group_stats=[]
    for ri,r in enumerate(ROLES):
        for weighting,w in (('unique_proxy',(counts[ri]>0).astype(float)),('draw_frequency',counts[ri]),('effective_noncollision',effective[ri]),('gumbel_expected',expected[ri]),('gumbel_expected_noncollision',expected_effective[ri])):
            group_stats.append(dict(role=r,weighting=weighting,depth=stats(h,w),nearest_class_angle=stats(theta,w),near_cap_0p1_mass=float(w[h>=4.5].sum()/w.sum()) if w.sum() else None,near_cap_0p5_mass=float(w[h>=4.1].sum()/w.sum()) if w.sum() else None,outer_band_mass=float(w[split].sum()/w.sum()) if split is not None and w.sum() else None))
    paired=[]
    for r,g in (('S',raw['same']),('C',~raw['same'])):
        for cohort,ok in (('all_distinct',raw['distinct']&g),('noncollision',raw['distinct']&g&(raw['pair']!=raw['triple']))):
            gap=h[raw['pair'][ok]]-h[raw['triple'][ok]]
            paired.append(dict(role=r,cohort=cohort,draws=len(gap),gap=stats(gap),positive=float((gap>1e-3).mean()) if len(gap) else None,negative=float((gap< -1e-3).mean()) if len(gap) else None,tied=float((np.abs(gap)<=1e-3).mean()) if len(gap) else None,joint_order=float(((gap>1e-3)&(h[raw['pair'][ok]]<4.6)).mean()) if len(gap) else None))
    progress('top4_hotspot_null');coverage_rows=[];exposure={};nulls={}
    for offset,(r,selected) in enumerate(sets.items()):
        result=coverage(nn4,selected,n);exposure[r]=result.pop('counts')
        if r!='All' and len(selected):
            vals=null_maxima(nn4,selected,labels,1000,4780+offset);nulls[r]=vals
            result.update(null_max_p50=float(np.median(vals)),null_max_p95=float(np.quantile(vals,.95)),null_max_p99=float(np.quantile(vals,.99)),null_max_empirical_p=float((1+(vals>=result['max_exposure']).sum())/1001))
        coverage_rows.append(dict(role=r,**result))
    representative=examples(mask,h,counts,pure)
    seed_stats=[]
    cumulative=np.zeros_like(counts)
    for si,seed in enumerate(range(4780,4785)):
        cumulative+=per_seed[si];s=role_summary(per_seed[si],pure)
        seed_stats.append(dict(seed=seed,selected=s['selected'],groups=s['groups'],cumulative_selected=role_summary(cumulative,pure)['selected']))
    preference=np.divide(counts[0]+counts[1],counts.sum(0),out=np.zeros(512,float),where=counts.sum(0)>0)
    rankh=average_ranks(h[active]);rankp=average_ranks(preference[active])
    spearman=float(np.corrcoef(rankh,rankp)[0,1]) if active.sum()>2 and np.std(preference[active])>0 else None
    proxy_rows=[]
    for i in range(p):
        proxy_rows.append(dict(proxy_id=i,depth=float(h[i]),gap_to_cap=float(4.6-h[i]),nearest_class=int(near[i]),nearest_class_name=CLASSES[near[i]],nearest_class_angle=float(theta[i]),role_bits=int(bits[i]),role_names='+'.join(ROLES[r] for r in range(3) if mask[r,i]),top4_pure=bool(pure[i]),top4_class_if_pure=CLASSES[labels[nn4[i,0]]] if pure[i] else '',pair_preference=float(preference[i]),internal_pair=int(internal[0,i]),internal_triple=int(internal[1,i]),**{r+'_draws':int(counts[j,i]) for j,r in enumerate(ROLES)},**{r+'_expected':float(expected[j,i]) for j,r in enumerate(ROLES)}))
    csvout(public/'proxy_statistics.csv',proxy_rows)
    csvout(public/'role_combinations.csv',role_summary(counts,pure)['exclusive'])
    csvout(public/'coverage_hotspots.csv',coverage_rows)
    csvout(public/'role_distribution.csv',[dict(role=g['role'],weighting=g['weighting'],**{'depth_'+k:v for k,v in g['depth'].items()},**{'angle_'+k:v for k,v in g['nearest_class_angle'].items()},outer_band_mass=g['outer_band_mass']) for g in group_stats])
    csvout(public/'paired_depth_gap.csv',[dict(role=x['role'],cohort=x['cohort'],**{k:v for k,v in x.items() if k not in ('gap','role','cohort')},**{'gap_'+k:v for k,v in x['gap'].items()}) for x in paired])
    csvout(private/'top4_neighbors.csv',[dict(proxy_id=i,rank=j+1,object_id=int(ids[k]),label=int(labels[k]),class_name=CLASSES[labels[k]],distance=float(dpw[i,k])) for i in range(p) for j,k in enumerate(nn4[i])])
    csvout(private/'hot_objects.csv',[dict(role=r,object_id=int(ids[i]),label=int(labels[i]),exposure=int(v)) for r,values in exposure.items() for i,v in enumerate(values) if v])
    np.savez_compressed(private/'readout.npz',h=h,theta=theta,near=near,counts=counts,all_counts=all_counts,effective=effective,expected=expected,expected_effective=expected_effective,per_seed=per_seed,nn4=nn4,pure=pure,labels=labels,ids=ids,bits=bits,internal=internal,internal_expected=internal_expected,**{'exposure_'+k:v for k,v in exposure.items()},**{'null_'+k:v for k,v in nulls.items()})
    summary=dict(status='completed',run_id='TS17',best_epoch=273,objects=n,proxies=p,seed_values=list(range(4780,4785)),batches_per_seed=200,
        selection='deterministic exact minimax; original reciprocal miner; analytic independent Gumbel probability sensitivity',
        geometry=dict(c=.1,whole_depth=stats(depth(shell)),proxy_depth=stats(h),proxy_raw_cap_count=int((np.linalg.norm(tangent,axis=1)>2.3).sum())),
        source_verification=dict(checkpoint_sha256=CP_SHA,cache_sha256=CACHE_SHA,shell_max_abs_error=shell_error,proxy_max_abs_error=proxy_error,top4_cached_cost_max_abs_error=top4_cost_error,top4_cached_exact_rows_fraction=float(np.mean(np.all(np.sort(nn4,axis=1)==np.sort(old,axis=1),axis=1)))),
        draws=audit_counts,roles=role_summary(counts,pure),raw_including_repeated_objects=role_summary(all_counts,pure),noncollision_roles=role_summary(effective,pure),group_distributions=group_stats,paired=paired,depth_bands=bands,depth_pair_preference_rank_correlation=spearman,alignment=[dict(role=r,matched=int(alignment[i,0]),draws=int(alignment[i,1]),fraction=float(alignment[i,0]/alignment[i,1]) if alignment[i,1] else None) for i,r in enumerate(ROLES)],coverage=coverage_rows,seed_stability=seed_stats,examples=representative,
        proxy_internal=dict(draws_per_role=int(internal[0].sum()),pair_depth=stats(h,internal[0]),triple_depth=stats(h,internal[1])),
        new_network_forwards=0,backward_calls=0,optimizer_updates=0,BN_updates=0,test_objects_used=0,elapsed_seconds=time.time()-started)
    if sha(cp)!=CP_SHA or sha(cache)!=CACHE_SHA:raise RuntimeError('Read-only source changed during analysis')
    assert counts[:2].sum()==counts[2].sum()==audit_counts['distinct_draws']
    assert effective[:2].sum()==effective[2].sum()==audit_counts['distinct_draws']-audit_counts['collisions_distinct']
    assert np.allclose(expected[:2].sum(),expected[2].sum(),rtol=1e-5)
    write(public/'summary.json',summary);write(state,dict(status='completed',elapsed_seconds=time.time()-started,all_input_SHA_unchanged=True,network_forwards=0,optimizer_updates=0))
    print(json.dumps(dict(status='completed',roles=summary['roles'],paired=paired),ensure_ascii=False),flush=True)


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--settings',type=Path,required=True);args=a.parse_args()
    readout(json.loads(args.settings.read_text()))
