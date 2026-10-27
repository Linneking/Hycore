"""Approved frozen pair competition, two replacements, and e0/e5 direction probes."""
from __future__ import annotations
import json,sys,time,hashlib,argparse
from pathlib import Path
import numpy as np
from science import *
REPO=Path(__file__).resolve().parents[4]
sys.path[:0]=[str(REPO),str(REPO/'inter_hierarchy_MN40'),str(REPO/'classification_ModelNet40'),str(REPO/'pointnet2_ops_lib')]
sys.path.insert(0,str(REPO/'docs/research/diagnostics/ts17_proxy_roles96_v1'))
from pipeline import csvout,CP_SHA,CACHE_SHA


def main():
    a=argparse.ArgumentParser();a.add_argument('--settings',type=Path,required=True);args=a.parse_args();s=json.loads(args.settings.read_text())
    root=Path(s['root']);private=root/'private';public=root/'public';private.mkdir();public.mkdir();start=time.time()
    def progress(stage,**kw):write(root/'worker_status.json',dict(status='running',stage=stage,elapsed_seconds=time.time()-start,**kw))
    import torch,h5py
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    from inter_hierarchy_MN40.hier_proxy_scratch_v8.geometry import official_distance,to_ball,configure_backbone
    from inter_hierarchy_MN40.hier_proxy_cycle_v9c.geometry import whole_to_hier
    from inter_hierarchy_MN40.models.pointmlp import Hype_pointMLP
    sys.path.insert(0,str(REPO/'docs/research/diagnostics/v9_embedding_review_20261021'))
    from collect import state_hash
    progress('source_verification');material=Path(s['material']);old=Path(s['previous_root'])
    before={name:sha(material/name) for name in ['initialization_e0.pth','checkpoint_epoch_005.pth','manifest.json']}
    if before['initialization_e0.pth']!='444c7063c0fad6d8d87138fe9a266e1e5ca59efe2912c4c9ff87598d7d88bf81' or before['checkpoint_epoch_005.pth']!='c698920bf81b5662120ba8f9f388f091f42194d0b4c91b34f6b70e494bc81bdf':raise ValueError('Transferred early checkpoint differs from original8001 SHA')
    history=np.zeros((273,2,3,2,512),np.int64);source_hashes={}
    for epoch in range(1,274):
        f=material/('metrics_epoch_%03d.json'%epoch);source_hashes[f.name]=sha(f);record=json.loads(f.read_text())
        if record['epoch']!=epoch or record['completed_optimizer_steps']!=200*epoch:raise ValueError('Wrong TS17 history epoch/update')
        for ci,component in enumerate(('sample','proxy')):
            source=record['structure'][component]['selected_ancestor_usage']
            for di,domain in enumerate(('all_draws','noncollision','active_noncollision')):
                for ri,role in enumerate(('pair','triple')):
                    row=source[domain+'/'+role];v=np.asarray(row['counts'],np.int64)
                    if v.shape!=(512,) or (v<0).any() or v.sum()!=row['selection_count']:raise ValueError('Historical count contract')
                    history[epoch-1,ci,di,ri]=v
        if epoch==5:expected_e5_sha=record['checkpoint']['last_sha256']
    if expected_e5_sha!=before['checkpoint_epoch_005.pth']:raise ValueError('Original e5 checkpoint SHA differs from its saved epoch record')
    if sha(s['checkpoint'])!=CP_SHA or sha(s['cache'])!=CACHE_SHA:raise ValueError('Fixed best/cache identity')
    cp0=torch.load(material/'initialization_e0.pth',map_location='cpu',weights_only=False)
    cp5=torch.load(material/'checkpoint_epoch_005.pth',map_location='cpu',weights_only=False)
    best=torch.load(s['checkpoint'],map_location='cpu',weights_only=False)
    manifest=json.loads((material/'manifest.json').read_text())
    if cp0['initialized_model_sha256']!=manifest['initialized_model_sha256'] or cp5['completed_epochs']!=5:raise ValueError('Wrong e0/e5 lineage')
    for cp in (cp0,cp5,best):
        config=cp['training_config'];opt=config['proxy_optimizer']
        if config['arm']!='b05' or config['seed']!=4780 or opt['lr']!=.005 or opt['eps']!=1e-4 or config['steps_per_epoch']!=200:raise ValueError('Not TS17 A1 CB64')
    with np.load(s['cache']) as f:cache={k:f[k] for k in ('native_mu','hier_mu','labels','canonical_ids','proxy_ball','proxy_tangent')}
    labels=cache['labels'].astype(np.int64);ids=cache['canonical_ids'];assert np.array_equal(ids,np.arange(9840))
    old_summary=json.loads((old/'public/summary.json').read_text());threshold=old_summary['depth_bands']['threshold']
    with np.load(old/'private/relations.npz') as f:raw={k:f[k] for k in ('i','j','same','distinct')}
    pairs,available=sample_pairs(raw,labels);pair_labels=labels[pairs[:,0]];del raw
    with np.load(old/'private/readout.npz') as f:old_counts=f['counts'].copy()
    group=np.full(512,-1,np.int64);h=depth(cache['proxy_ball']);outer=h>=threshold;used=old_counts.any(0)
    group[outer&~used]=0;group[~outer&used]=1;group[~outer&~used]=2;group[outer&used]=3
    group_names=['outer_unused','inner_used','inner_unused','outer_used']
    if [int((group==i).sum()) for i in range(4)]!=[162,290,59,1]:raise ValueError('96 proxy ID groups differ')
    if not np.array_equal(best['proxy']['tangent_proxies'].numpy(),cache['proxy_tangent']):raise ValueError('Best proxy rows changed')
    generator=torch.Generator(device='cpu').manual_seed(4780)
    expected_init=torch.randn(512,256,generator=generator)/np.sqrt(256)*2.3*.9
    init=cp0['proxy']['tangent_proxies'];initial_exact=torch.equal(init,expected_init)
    if not initial_exact:raise ValueError('e0 is not the original independent seed4780 initialization')
    # Inputs are byte-identical to the 85/96 clean cache.
    progress('input_loading');clouds=[];gold=[]
    for p in sorted(Path(s['data_dir']).glob('ply_data_train*.h5')):
        with h5py.File(p,'r') as f:clouds.append(f['data'][:,:1024].astype(np.float32));gold.append(f['label'][:].reshape(-1).astype(np.int64))
    clouds=np.concatenate(clouds);gold=np.concatenate(gold);input_sha=hashlib.sha256(clouds.tobytes()).hexdigest()
    if input_sha!='dc82f898decf86fed478184b9a6e72f8fa95d4165b0150a449a2fa5e178f27f7' or not np.array_equal(labels,gold):raise ValueError('Input/cache mismatch')
    rng=np.random.RandomState(4780);probe_ids=np.sort(np.concatenate([rng.choice(np.flatnonzero(labels==c),20,False) for c in range(40)]))
    device=torch.device('cuda:0');native_probe={};state_checks={};forwards=0
    # A bounded best32 regression precedes early feature extraction.
    for name,cp,selected in (('best_gate',best,np.arange(32)),('e0',cp0,probe_ids),('e5',cp5,probe_ids)):
        progress('frozen_forward',checkpoint=name,objects=len(selected));torch.manual_seed(22);np.random.seed(22)
        model=configure_backbone(Hype_pointMLP());model.load_state_dict({k.removeprefix('module.'):v for k,v in cp['net'].items()},strict=True)
        model.requires_grad_(False);model.eval();model.to(device);state_before=state_hash(model);chunks=[]
        with torch.no_grad():
            for off in range(0,len(selected),32):
                batch=torch.from_numpy(clouds[selected[off:off+32]].copy()).to(device)
                mu,_=model(batch.transpose(1,2).contiguous());chunks.append(mu.cpu().numpy());forwards+=len(batch)
        values=np.concatenate(chunks);assert np.isfinite(values).all()
        unchanged=state_before==state_hash(model) and all(p.grad is None for p in model.parameters())
        if not unchanged:raise RuntimeError('Frozen model or BN mutated')
        state_checks[name]=dict(unchanged=True,objects=len(selected),state_sha256=state_before)
        if name=='best_gate':
            delta=np.abs(values-cache['native_mu'][:32]);angles=angular_rows(values,cache['native_mu'][:32])
            gate=dict(passed=bool(delta.max()<=2e-4 and delta.mean()<=2e-6 and np.quantile(angles,.99)<=.05),max_abs=float(delta.max()),mean_abs=float(delta.mean()),angle_p99=float(np.quantile(angles,.99)),objects=32)
            if not gate['passed']:raise ValueError('Best32 frozen forward cannot reproduce trusted cache')
        else:native_probe[name]=values
        del model;torch.cuda.empty_cache()
    native_probe['best']=cache['native_mu'][probe_ids]
    progress('pair_competition',pairs=len(pairs))
    with torch.no_grad():
        whole=whole_to_hier(torch.from_numpy(cache['native_mu']).to(device));proxy=to_ball(torch.from_numpy(cache['proxy_tangent']).to(device),hook=False)
        assert float((whole-torch.from_numpy(cache['hier_mu']).to(device)).abs().max())<3e-6
        d=official_distance(whole,proxy);pc=torch.maximum(d[pairs[:,0]],d[pairs[:,1]])
        inner_ids=np.flatnonzero(~outer);outer_ids=np.flatnonzero(outer);cost=pc.cpu().numpy()
        I=inner_ids[cost[:,inner_ids].argmin(1)];O=outer_ids[cost[:,outer_ids].argmin(1)];n=len(pairs);ix=np.arange(n)
        cI=cost[ix,I];cO=cost[ix,O];winner=np.where(cI<cO,I,np.where(cI>cO,O,np.minimum(I,O)))
        if not np.array_equal(winner,cost.argmin(1)):raise ValueError('Inner/outer winner does not equal full512 argmin')
        outer_probability=torch.softmax(-pc/.1,1)[:,outer_ids].sum(1).cpu().numpy();dn=d.cpu().numpy()
        ball=proxy.cpu().numpy();whole_np=whole.cpu().numpy()
        altered=[at_depth(ball[O],h[I]).astype(np.float32),at_depth(ball[I],h[O]).astype(np.float32)]
        altered_costs=[]
        for values in altered:
            results=[]
            for off in range(0,n,64):
                p=torch.from_numpy(values[off:off+64]).to(device)
                di=official_distance(whole[pairs[off:off+64,0]],p).diag();dj=official_distance(whole[pairs[off:off+64,1]],p).diag()
                results.append(torch.maximum(di,dj).cpu().numpy())
            altered_costs.append(np.concatenate(results))
        p0=to_ball(init.to(device),hook=False).cpu().numpy();p5=to_ball(cp5['proxy']['tangent_proxies'].to(device),hook=False).cpu().numpy()
    # CPU summaries retain exact matched candidates, not a reoptimized outer pool.
    thetaI=np.column_stack([angular_rows(whole_np[pairs[:,q]],ball[I]) for q in range(2)])
    thetaO=np.column_stack([angular_rows(whole_np[pairs[:,q]],ball[O]) for q in range(2)])
    gap=cO-cI;gap_depth=altered_costs[0]-cI;gap_angle=altered_costs[1]-cI;loses=gap>1e-5
    classes=[]
    for c in range(40):
        m=pair_labels==c
        classes.append(dict(class_id=c,class_name=CLASSES[c],available_unique_pairs=available[c],pairs=int(m.sum()),outer_wins=int((m&(gap< -1e-5)).sum()),ties=int((m&(np.abs(gap)<=1e-5)).sum()),gap_median=stats(gap[m])['median'],inner_max_angle_median=stats(thetaI[m].max(1))['median'],outer_max_angle_median=stats(thetaO[m].max(1))['median'],outer_probability_mean=float(outer_probability[m].mean()) if m.any() else None))
    pair_rows=[]
    for q in range(n):
        pair_rows.append(dict(pair_index=q,class_name=CLASSES[pair_labels[q]],inner_id=int(I[q]),outer_id=int(O[q]),inner_depth=float(h[I[q]]),outer_depth=float(h[O[q]]),inner_angle_i=float(thetaI[q,0]),inner_angle_j=float(thetaI[q,1]),outer_angle_i=float(thetaO[q,0]),outer_angle_j=float(thetaO[q,1]),inner_distance_i=float(dn[pairs[q,0],I[q]]),inner_distance_j=float(dn[pairs[q,1],I[q]]),outer_distance_i=float(dn[pairs[q,0],O[q]]),outer_distance_j=float(dn[pairs[q,1],O[q]]),inner_cost=float(cI[q]),outer_cost=float(cO[q]),gap=float(gap[q]),depth_only_gap=float(gap_depth[q]),direction_only_gap=float(gap_angle[q]),outer_pool_probability=float(outer_probability[q])))
    example_indices=choose_cases(gap,pair_labels);examples=[dict(pair_rows[q],object_i=int(pairs[q,0]),object_j=int(pairs[q,1])) for q in example_indices]
    theta0,near0,centers0=nearest_angle(p0,native_probe['e0'],labels[probe_ids])
    theta05,near05,centers5=nearest_angle(p0,native_probe['e5'],labels[probe_ids])
    theta5,_,_=nearest_angle(p5,native_probe['e5'],labels[probe_ids])
    theta_best,_,_=nearest_angle(ball,native_probe['best'],labels[probe_ids])
    rotation5=angular_rows(p0,p5);rotation_best=angular_rows(p0,ball);h0=depth(p0);h5=depth(p5)
    early=history[:5,:,0].sum(0);last=history[-1,:,0];groups=[]
    for gi,name in enumerate(group_names):
        m=group==gi;record=dict(group=name,proxies=int(m.sum()),initial_angle=stats(theta0[m]),initial_to_e5_angle=stats(theta05[m]),initial_depth=stats(h0[m]),rotation_e5=stats(rotation5[m]),rotation_best=stats(rotation_best[m]),depth_change=stats((h-h0)[m]),best_angle=stats(theta_best[m]))
        for ci,component in enumerate(('sample','internal')):
            for ri,role in enumerate(('pair','triple')):
                for period,values in (('first5',early),('epoch273',last)):
                    total=values[ci,ri].sum();count=values[ci,ri,m].sum()
                    record[period+'_'+component+'_'+role+'_share']=float(count/total) if total else None
                    record[period+'_'+component+'_'+role+'_per_proxy_vs_uniform']=float(count/total/m.sum()*512) if total else None
        groups.append(record)
    strata=np.empty(512,int);strata_rows=[]
    for qi,indices in enumerate(np.array_split(np.argsort(h0,kind='stable'),4)):
        strata[indices]=qi
        sorted_ix=indices[np.argsort(theta0[indices],kind='stable')]
        for tag,members in zip(('near_initial_direction','far_initial_direction'),np.array_split(sorted_ix,2)):
            strata_rows.append(dict(initial_depth_quartile=qi+1,direction_group=tag,proxies=len(members),depth_min=float(h0[members].min()),depth_max=float(h0[members].max()),angle_median=float(np.median(theta0[members])),first5_sample_pair_per_proxy_vs_uniform=float(early[0,0,members].sum()/early[0,0].sum()/len(members)*512),outer_unused_fraction=float((group[members]==0).mean())))
    proxy_rows=[]
    for p in range(512):
        proxy_rows.append(dict(proxy_id=p,group=group_names[group[p]],initial_depth=float(h0[p]),e5_depth=float(h5[p]),best_depth=float(h[p]),initial_depth_quartile=int(strata[p]+1),initial_nearest_angle=float(theta0[p]),initial_nearest_class=CLASSES[near0[p]],initial_to_e5_nearest_angle=float(theta05[p]),initial_to_e5_nearest_class=CLASSES[near05[p]],e5_nearest_angle=float(theta5[p]),best_nearest_angle_probe=float(theta_best[p]),rotation_e5=float(rotation5[p]),rotation_best=float(rotation_best[p]),**{period+'_'+comp+'_'+role:int(values[ci,ri,p]) for period,values in (('first5',early),('epoch273',last)) for ci,comp in enumerate(('sample','internal')) for ri,role in enumerate(('pair','triple'))}))
    historical=[]
    for epoch in range(273):
        for gi,name in enumerate(group_names):
            m=group==gi
            for ci,comp in enumerate(('sample','internal')):
                for di,domain in enumerate(('all_draws','noncollision','active_noncollision')):
                    for ri,role in enumerate(('pair','triple')):
                        total=int(history[epoch,ci,di,ri].sum());count=int(history[epoch,ci,di,ri,m].sum())
                        historical.append(dict(epoch=epoch+1,group=name,proxies=int(m.sum()),component=comp,domain=domain,role=role,count=count,total=total,share=count/total if total else None,per_proxy_vs_uniform=count/total/m.sum()*512 if total else None))
    control=[]
    for name,values in (('original',gap),('depth_only',gap_depth),('direction_only',gap_angle)):
        good=loses&(values< -1e-5)
        fractions=[float(good[pair_labels==c].sum()/loses[pair_labels==c].sum()) for c in np.unique(pair_labels) if loses[pair_labels==c].any()]
        control.append(dict(state=name,gap=stats(values),class_equal_mean_gap=equal_class_mean(values,pair_labels),original_losses=int(loses.sum()),rescues=int(good.sum()),class_equal_rescue_fraction=float(np.mean(fractions)),improvement=stats(gap-values)))
    identities=dict(source_files=before,original_e5_epoch_SHA_matches=True,initial_proxy_tensor_exact_reconstruction=initial_exact,initial_proxy_tensor_sha256=hashlib.sha256(init.numpy().tobytes()).hexdigest(),best_checkpoint_sha256=CP_SHA,best_cache_sha256=CACHE_SHA,history_metric_SHA=source_hashes,input_sha256=input_sha)
    for name,digest in before.items():assert sha(material/name)==digest
    for name,digest in source_hashes.items():assert sha(material/name)==digest
    assert sha(s['cache'])==CACHE_SHA and sha(s['checkpoint'])==CP_SHA and hashlib.sha256(clouds.tobytes()).hexdigest()==input_sha
    csvout(public/'class_competition.csv',classes);csvout(private/'all_pair_accounts.csv',pair_rows);csvout(public/'representative_pair_accounts.csv',[pair_rows[i] for i in example_indices]);csvout(public/'proxy_initial_history.csv',proxy_rows);csvout(public/'initial_depth_control.csv',strata_rows);csvout(public/'history_groups.csv',historical)
    np.savez_compressed(private/'readout.npz',pairs=pairs,pair_labels=pair_labels,h=h,h0=h0,h5=h5,group=group,thetaI=thetaI,thetaO=thetaO,gap=gap,gap_depth=gap_depth,gap_angle=gap_angle,outer_probability=outer_probability,history=history,theta0=theta0,theta05=theta05,rotation5=rotation5,rotation_best=rotation_best,probe_ids=probe_ids,probe_e0=native_probe['e0'],probe_e5=native_probe['e5'])
    write(private/'case_manifest.json',examples);write(private/'source_identity.json',identities)
    summary=dict(status='completed',run='TS17/B12',epoch=273,code_commit=s['commit'],pairs=len(pairs),classes_present=int(len(np.unique(pair_labels))),class_equal_outer_win_fraction=equal_class_mean((gap< -1e-5).astype(float),pair_labels),class_equal_outer_probability=equal_class_mean(outer_probability,pair_labels),inner_max_angle=stats(thetaI.max(1)),outer_max_angle=stats(thetaO.max(1)),pair_gap=stats(gap),controls=control,groups=groups,strata=strata_rows,examples=[{k:v for k,v in x.items() if k not in ('object_i','object_j')} for x in examples],initial_depth=stats(h0),initial_angle_early_pair_spearman=rank_correlation(theta0,early[0,0]),initial_to_e5_angle_early_pair_spearman=rank_correlation(theta05,early[0,0]),depth_stratum_correlations=[dict(quartile=q+1,spearman=rank_correlation(theta0[strata==q],early[0,0,strata==q])) for q in range(4)],best_gate=gate,frozen_models=state_checks,new_forward_objects=forwards,early_probe_objects_per_epoch=800,backward_calls=0,optimizer_updates=0,BN_updates=0,test_objects=0,source_unchanged=True,identity_summary={k:v for k,v in identities.items() if k!='history_metric_SHA'},elapsed_seconds=time.time()-start)
    # Public identity uses filenames and hashes, never host paths.
    write(public/'summary.json',summary);write(public/'source_hashes.json',identities);write(root/'worker_status.json',dict(status='completed',elapsed_seconds=time.time()-start,source_unchanged=True,new_forward_objects=forwards,optimizer_updates=0))
    print(json.dumps(dict(status='completed',pairs=len(pairs),controls=control,groups=groups,spearman=summary['initial_angle_early_pair_spearman']),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
