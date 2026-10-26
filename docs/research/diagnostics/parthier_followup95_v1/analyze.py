"""Approved five-question frozen readout; consumes caches, never an encoder."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import torch

from docs.research.diagnostics.parthier_embedding_audit94_v1 import metrics as old
from docs.research.diagnostics.parthier_embedding_audit94_v1.collect import sha, idle_gpu
from docs.research.diagnostics.parthier_embedding_audit94_v1.pipeline import checked_export_identity
from .core import (SCALES, STATE_NAMES, Columns, aligned_official_distance, angles,
                   angle_matrix, midpoint, states, gradient_probe, gradient_summary,
                   feasible_pair, hotspot)


def cat_stats(columns): return {k:old.describe(v) for k,v in columns.items()}


def supply_stats(value):
    x=np.asarray(old.array(value),dtype=np.float64).reshape(-1);x=x[np.isfinite(x)]
    return {**old.describe(x),'p10':float(np.quantile(x,.1)) if len(x) else None}


def state_table(gaps, margin):
    s=states(gaps,margin)
    return {'count':len(s),'per_edge':[[int((s[:,i]==j).sum()) for j in range(3)] for i in range(s.shape[1])],
            'all_margin_pass':int((s==2).all(1).sum()),
            'all_basic_pass':int((s>0).all(1).sum()),
            'near_zero_count':[int((np.abs(gaps[:,i])<1e-3).sum()) for i in range(s.shape[1])],
            'near_margin_count':[int((np.abs(gaps[:,i]-margin)<1e-3).sum()) for i in range(s.shape[1])]}


class SelectionAudit:
    def __init__(self, queue, panels):
        self.q=len(queue['features']);self.live=192;self.n=self.q+panels*self.live
        self.ids=np.full(self.n,-1,dtype=np.int64);self.scale=np.full(self.n,-1,dtype=np.int64)
        self.depth=np.full(self.n,np.nan);self.source=np.r_[np.ones(self.q,dtype=np.int8),np.zeros(panels*self.live,dtype=np.int8)]
        self.ids[:self.q]=old.array(queue['object_ids']);self.scale[:self.q]=old.array(queue['scales'])
        self.depth[:self.q]=old.array(old.origin_depth(queue['features']))
        self.exposures=np.zeros(self.n,dtype=np.int64)
        self.role={r:{'actual':np.zeros(self.n,dtype=np.int64),'valid':np.zeros(self.n,dtype=np.int64),
                      'expected':np.zeros(self.n),'equal_depth_expected':np.zeros(self.n)} for r in ('A','B')}

    def slots(self,panel,ids,depth):
        live=np.arange(self.live)+self.q+panel*self.live
        self.ids[live]=np.repeat(old.array(ids),3);self.scale[live]=np.tile(np.arange(3),len(ids))
        self.depth[live]=old.array(depth[:self.live])
        return np.r_[live,np.arange(self.q)]

    def add(self,role,slots,chosen,valid,prob,equal_prob):
        item=self.role[role]
        ix=slots[old.array(chosen)]
        np.add.at(item['actual'],ix,1);np.add.at(item['valid'],ix[old.array(valid)],1)
        item['expected'][slots]+=old.array(prob.sum(0).double())
        item['equal_depth_expected'][slots]+=old.array(equal_prob.sum(0).double())

    def finish(self,records):
        groups=[];hot=[];top=[]
        for role,item in self.role.items():
            count_key=role.lower()+'_group';depth_key=role.lower()+'_depth';pct_key=role.lower()+'_percentile'
            for source in (0,1):
                for scale in range(3):
                    group=source*3+scale;mask=(self.source==source)&(self.scale==scale)
                    selected=(records[count_key]==group)&records['valid'].astype(bool)
                    row={'role':role,'source':'queue' if source else 'current','scale':SCALES[scale],
                         'candidate_entries':int(mask.sum()),'query_exposures':int(self.exposures[mask].sum()),
                         'actual_all':int(item['actual'][mask].sum()),'actual_valid':int(item['valid'][mask].sum()),
                         'expected_all':float(item['expected'][mask].sum()),
                         'equal_depth_expected_all':float(item['equal_depth_expected'][mask].sum()),
                         'candidate_depth':supply_stats(self.depth[mask]),'selected_depth':supply_stats(records[depth_key][selected]),
                         'selected_within_group_percentile':supply_stats(records[pct_key][selected])}
                    groups.append(row)
                source_mask=self.source==source
                h={'role':role,'source':'queue' if source else 'current',
                   **hotspot(item['actual'][source_mask],item['expected'][source_mask],self.exposures[source_mask])}
                h['valid_selection']=hotspot(item['valid'][source_mask],item['expected'][source_mask],self.exposures[source_mask])['actual']
                h['valid_expected_note']='Expected concentration shown only for ALL draws, not collision-conditioned.'
                keys=(self.ids[source_mask]*3+self.scale[source_mask]).astype(np.int64)
                unique,inv=np.unique(keys,return_inverse=True)
                aa=np.bincount(inv,weights=item['actual'][source_mask]);ee=np.bincount(inv,weights=item['expected'][source_mask])
                xx=np.bincount(inv,weights=self.exposures[source_mask])
                h['object_scale_merged']=hotspot(aa,ee,xx);hot.append(h)
                eligible=np.where(source_mask)[0]
                order=eligible[np.argsort(-item['actual'][eligible],kind='stable')[:10]]
                for rank,slot in enumerate(order,1):
                    top.append({'role':role,'source':h['source'],'rank':rank,'object_id':int(self.ids[slot]),
                        'scale':SCALES[int(self.scale[slot])],'slot':int(slot),'depth':float(self.depth[slot]),
                        'actual_all':int(item['actual'][slot]),'actual_valid':int(item['valid'][slot]),
                        'expected_all':float(item['expected'][slot]),'query_exposures':int(self.exposures[slot]),
                        'per_million_exposure':float(1e6*item['actual'][slot]/self.exposures[slot])})
            if abs(item['expected'].sum()-len(records['valid']))>2:
                raise RuntimeError('Softmax expected counts lost probability mass')
            if item['actual'].sum()!=len(records['valid']) or item['valid'].sum()!=records['valid'].sum():
                raise RuntimeError('Hotspot counts differ from selection denominator')
        return {'groups':groups,'hotspots':hot,'top_entries':top}


def candidate_percentiles(depth,groups):
    result=torch.empty_like(depth)
    for g in range(6):
        mask=groups==g;v=depth[mask];s=v.sort().values
        if len(v): result[mask]=50*(torch.searchsorted(s,v,right=False)+torch.searchsorted(s,v,right=True)).to(depth.dtype)/len(v)
    return result


def case_card(kind,label,panel,trips,index,ids,labels,mu,candidates,scales,candidate_ids,depth,distance,detail):
    positions=trips[index];i,j,k=positions.tolist();a=int(detail['pair_idx'][index]);b=int(detail['triple_idx'][index])
    d=distance[positions]
    hw=old.origin_depth(mu[positions])
    card={'kind':kind,'model':label,'panel':panel,'relation_row':int(index),
          'object_ids':old.array(ids[positions]).tolist(),'classes':old.array(labels[positions]).tolist(),
          'whole_depths':old.array(hw).tolist(),'whole_angles':{
              'ij':float(angles(mu[i:i+1],mu[j:j+1])[0]),'ik':float(angles(mu[i:i+1],mu[k:k+1])[0]),
              'jk':float(angles(mu[j:j+1],mu[k:k+1])[0])},'A':{},'B':{}}
    for role,chosen in (('A',a),('B',b)):
        card[role]={'index':chosen,'object_id':int(candidate_ids[chosen]),'scale':SCALES[int(scales[chosen])],
                    'source':'current' if chosen<192 else 'queue','depth':float(depth[chosen]),
                    'whole_distances':old.array(d[:,chosen]).tolist()}
    card['ord_gaps']=[float(hw[:2].min()-depth[a]),float(depth[a]-depth[b])]
    card['hier_gaps']=old.array(torch.stack((d[0,b]-d[0,a],d[1,b]-d[1,a],d[2,a]-d[2,b]))).tolist()
    if kind.startswith('hier'):
        result=feasible_pair(d)
        if result['exists']:
            aa,bb=result['a'],result['b']
            result['A']={'object_id':int(candidate_ids[aa]),'scale':SCALES[int(scales[aa])],
                         'source':'current' if aa<192 else 'queue','depth':float(depth[aa])}
            result['B']={'object_id':int(candidate_ids[bb]),'scale':SCALES[int(scales[bb])],
                         'source':'current' if bb<192 else 'queue','depth':float(depth[bb])}
        card['hier_only_candidate_feasibility']=result
    return card


@torch.no_grad()
def relation_panel(store,audit,cases,label,panel,mu,parts,labels,ids,queue,seed):
    current=parts.reshape(-1,parts.shape[-1]);candidates=torch.cat((current,queue['features']))
    candidate_ids=torch.cat((ids.repeat_interleave(3),queue['object_ids']))
    scales=torch.cat((torch.arange(3,device=mu.device).repeat(len(mu)),queue['scales']))
    groups=scales+3*torch.cat((torch.zeros(len(current),device=mu.device,dtype=torch.long),
                              torch.ones(len(queue['features']),device=mu.device,dtype=torch.long)))
    hd,cd=old.origin_depth(mu),old.origin_depth(candidates)
    mined=old.mine_sample_triplets(mu,labels,topk=20,t_per_anchor=50,seed=seed,exclude_self_negative=True,data_ids=ids)
    trips=mined['triplets']
    if not len(trips): return
    i,j,k=trips.unbind(-1)
    distance=old.official_distance(mu,candidates,c=1.)
    generator=torch.Generator(device=mu.device).manual_seed(seed)
    hloss,oloss,detail=old._selection_losses(distance,trips,hd,cd,generator=generator)
    a,b,valid=detail['pair_idx'],detail['triple_idx'],detail['valid']
    di,dj,dk=distance[i],distance[j],distance[k]
    pcost=torch.maximum(di,dj);tcost=torch.maximum(pcost,dk)
    pa,pb=torch.softmax(-pcost/.1,-1),torch.softmax(-tcost/.1,-1)
    common_h=torch.quantile(cd,.5)
    equal_candidates=(candidates/candidates.norm(dim=-1,keepdim=True))*torch.tanh(common_h/2)
    eqd=old.official_distance(mu,equal_candidates,c=1.)
    epcost=torch.maximum(eqd[i],eqd[j]);etcost=torch.maximum(epcost,eqd[k])
    epa,epb=torch.softmax(-epcost/.1,-1),torch.softmax(-etcost/.1,-1)
    slots=audit.slots(panel,ids,cd);audit.exposures[slots]+=len(trips)
    audit.add('A',slots,a,valid,pa,epa);audit.add('B',slots,b,valid,pb,epb)
    percentile=candidate_percentiles(cd,groups)
    og=torch.stack((torch.minimum(hd[i],hd[j])-cd[a],cd[a]-cd[b]),1)
    hg=torch.stack((distance[i,b]-distance[i,a],distance[j,b]-distance[j,a],distance[k,a]-distance[k,b]),1)
    torch.testing.assert_close((torch.relu(.1-hg).sum(1)*valid).mean(),hloss,rtol=1e-5,atol=3e-6)
    torch.testing.assert_close((torch.relu(.2-og).sum(1)*valid).mean(),oloss,rtol=1e-5,atol=3e-6)
    wa=angle_matrix(mu,mu);ac=angle_matrix(mu,candidates)
    wd=old.official_distance(mu,mu,c=1.)
    mid,mid_h=midpoint(mu[i],mu[j]);mid32=mid.to(mu)
    mid_cost=torch.maximum(aligned_official_distance(mu[i],mid32),aligned_official_distance(mu[j],mid32))
    row=torch.arange(len(trips),device=mu.device)
    cpa,cpb=pcost[row,a],tcost[row,b]
    store.add(panel=torch.full_like(i,panel),relation_row=row,valid=valid,same_ij=(labels[i]==labels[j]),
        same_object_ij=(ids[i]==ids[j]),object_i=ids[i],object_j=ids[j],object_k=ids[k],
        class_i=labels[i],class_j=labels[j],class_k=labels[k],a_index=a,b_index=b,
        a_group=groups[a],b_group=groups[b],a_object=candidate_ids[a],b_object=candidate_ids[b],
        h_i=hd[i],h_j=hd[j],h_k=hd[k],a_depth=cd[a],b_depth=cd[b],
        theta_ij=wa[i,j],theta_ik=wa[i,k],theta_jk=wa[j,k],
        theta_iA=ac[i,a],theta_jA=ac[j,a],theta_kB=ac[k,b],
        d_ij=wd[i,j],d_ik=wd[i,k],d_jk=wd[j,k],j_closer_i=(wd[i,j]<wd[i,k]),
        ij_depth_difference=(hd[i]-hd[j]).abs(),midpoint_depth=mid_h,midpoint_cost=mid_cost,
        a_minus_midpoint=cd[a]-mid_h,angle_midpoint_inward=(hd[i]+hd[j]).double()/2-mid_h,
        a_percentile=percentile[a],b_percentile=percentile[b],
        a_cost=cpa,b_cost=cpb,a_min_cost=pcost.min(1).values,b_min_cost=tcost.min(1).values,
        a_cost_excess=cpa-pcost.min(1).values,b_cost_excess=cpb-tcost.min(1).values,
        a_rank=(pcost<cpa[:,None]).sum(1)+1,b_rank=(tcost<cpb[:,None]).sum(1)+1,
        a_selection_probability=pa[row,a],b_selection_probability=pb[row,b],
        ord_gaps=og,hier_gaps=hg)
    basic=(hg>0).all(1);failed=~(hg>=.1).all(1)
    selectors={'hier_same_basic_wrong':valid&failed&~basic&(labels[i]==labels[j]),
               'hier_cross_basic_wrong':valid&failed&~basic&(labels[i]!=labels[j]),
               'hier_only_short':valid&failed&basic}
    if label=='TS18_e300':
        selectors.update(ord_A_not_shallow=valid&(og[:,0]<=0),ord_B_not_shallow=valid&(og[:,1]<=0),
                         ord_only_short=valid&(og>0).all(1)&~(og>=.2).all(1))
    for kind,mask in selectors.items():
        if kind not in cases and bool(mask.any()):
            ix=int(torch.where(mask)[0][0])
            cases[kind]=case_card(kind,label,panel,trips,ix,ids,labels,mu,candidates,scales,candidate_ids,cd,distance,detail)


def summarize_relations(records,audit,reference):
    valid=records['valid'].astype(bool);og=records['ord_gaps'][valid];hg=records['hier_gaps'][valid]
    out={'draws':len(valid),'valid_draws':int(valid.sum()),'collisions':int((~valid).sum()),
         'same_ij_fraction_all':float(records['same_ij'].mean()),
         'same_ij_fraction_valid':float(records['same_ij'][valid].mean()),
         'ord':state_table(og,.2),'hier':state_table(hg,.1),'selection':audit.finish(records)}
    observed={'ord_basic':float((og>0).all(1).mean()),'ord_margin':float((og>=.2).all(1).mean()),
              'hier_basic':float((hg>0).all(1).mean()),'hier_margin':float((hg>=.1).all(1).mean())}
    expected={'ord_basic':reference['ord']['joint_basic_fraction'],'ord_margin':reference['ord']['joint_margin_fraction'],
              'hier_basic':reference['hier']['joint_basic_fraction'],'hier_margin':reference['hier']['joint_margin_fraction']}
    errors={k:abs(observed[k]-expected[k]) for k in observed}
    gate={'observed':observed,'reference':expected,'absolute_rate_errors':errors,
          'counts_exact':all(out[k]==reference[k] for k in ('draws','valid_draws','collisions')),
          'rate_tolerance':1e-5,'rates_exact':all(e==0 for e in errors.values())}
    gate['passed']=gate['counts_exact'] and max(errors.values())<=1e-5
    if not gate['passed']: raise RuntimeError('Frozen replay differs from94: '+json.dumps(gate))
    out['replay_gate']=gate
    st=states(og,.2);out['ord']['cross_3x3']=np.bincount(st[:,0]*3+st[:,1],minlength=9).reshape(3,3).tolist()
    failure=(states(hg,.1)!=2).astype(np.int8)
    patterns=(failure*np.array([1,2,4],np.int8)).sum(1)
    out['hier']['failure_patterns']=np.bincount(patterns,minlength=8).tolist()
    out['ord_groups']=[];out['geometry_groups']=[];out['choice_quality']=[];out['pair_bins']=[]
    for role in ('a','b'):
        for group in range(6):
            m=valid&(records[role+'_group']==group);n=int(m.sum());fails=int((~(records['ord_gaps'][m]>=.2).all(1)).sum())
            out['ord_groups'].append({'role':role.upper(),'source':'queue' if group>=3 else 'current',
                'scale':SCALES[group%3],'count':n,'joint_failures':fails,'failure_fraction':fails/n if n else None,
                'share_of_all_ord_failures':fails/int((~(og>=.2).all(1)).sum()) if n else 0.})
        for view,m in (('all',np.ones(len(valid),bool)),('valid',valid)):
            out['choice_quality'].append({'role':role.upper(),'view':view,
                **cat_stats({name:records[role+'_'+name][m] for name in ('rank','cost','cost_excess','selection_probability','percentile')})})
    geometric=('h_i','h_j','h_k','a_depth','b_depth','theta_ij','theta_ik','theta_jk','d_ij','d_ik','d_jk',
               'ij_depth_difference','midpoint_depth','a_minus_midpoint','angle_midpoint_inward','theta_iA','theta_jA','j_closer_i','same_object_ij')
    for same in (True,False):
        for passed in (True,False):
            mask=valid&(records['same_ij']==same)&((records['hier_gaps']>=.1).all(1)==passed)
            out['geometry_groups'].append({'same_ij':same,'hier_pass':passed,'count':int(mask.sum()),
                                          **cat_stats({k:records[k][mask] for k in geometric})})
        for lo,hi in ((0,5),(5,15),(15,30),(30,180.0001)):
            for dlo,dhi in ((0,.2),(.2,.5),(.5,float('inf'))):
                mask=valid&(records['same_ij']==same)&(records['theta_ij']>=lo)&(records['theta_ij']<hi)
                mask&=(records['ij_depth_difference']>=dlo)&(records['ij_depth_difference']<dhi)
                out['pair_bins'].append({'same_ij':same,'angle_low':lo,'angle_high':min(hi,180),
                    'depth_diff_low':dlo,'depth_diff_high':dhi if math.isfinite(dhi) else None,'count':int(mask.sum()),
                    **cat_stats({k:records[k][mask] for k in ('a_depth','midpoint_depth','a_minus_midpoint','a_cost_excess','theta_iA','theta_jA')})})
    return out


def population(cache,device):
    whole=torch.from_numpy(cache['whole']).to(device)
    parts=torch.from_numpy(cache['parts']).to(device)
    clean=torch.from_numpy(cache['clean']).to(device)
    hw=old.origin_depth(whole);hp=old.origin_depth(parts)
    theta=angles(parts,whole[:,None,:].expand_as(parts))
    clean_theta=angles(parts,clean[:,None,:].expand_as(parts))
    cg=torch.stack((hp[:,1]-hp[:,0],hp[:,2]-hp[:,1],hw-hp[:,2]),1)
    state=states(old.array(cg),.2)
    pattern=((state!=2)*np.array([1,2,4])).sum(1)
    model={'angles':{},'chain':{**state_table(old.array(cg),.2),
           'failure_patterns':np.bincount(pattern,minlength=8).tolist()},'object_examples':[],'chain_examples':[]}
    for index,g in enumerate(SCALES):
        x=old.array(theta[:,index]);cx=old.array(clean_theta[:,index])
        counts,bins=np.histogram(x[np.isfinite(x)],bins=np.arange(0,181,5))
        model['angles'][str(g)]={**old.describe(x),'undefined':int((~np.isfinite(x)).sum()),
            'clean_angle':old.describe(cx),'hist_counts':counts.tolist(),'hist_edges':bins.tolist()}
        assert counts.sum()+model['angles'][str(g)]['undefined']==9840
    depths=old.array(torch.cat((hp,hw[:,None]),1));t=old.array(theta)
    for i in (0,1968,3936,5904,7872,9839):
        model['object_examples'].append({'object_id':int(cache['ids'][i]),'class':int(cache['labels'][i]),
                                        'part_whole_angles':t[i].tolist(),'depths_200_400_600_whole':depths[i].tolist()})
    if cache.get('label')=='TS18_e300':
        gap=old.array(cg)
        selectors={'last_wrong_or_tied':gap[:,2]<=0,
                   'only_last_short':(state[:,:2]==2).all(1)&(state[:,2]==1),
                   'earlier_edges_fail':(state[:,:2]!=2).any(1)}
        for name,mask in selectors.items():
            ix=np.where(mask)[0]
            if len(ix):
                i=int(ix[0]);model['chain_examples'].append({'kind':name,'object_id':int(cache['ids'][i]),
                    'class':int(cache['labels'][i]),'depths_200_400_600_whole':depths[i].tolist(),
                    'chain_gaps':gap[i].tolist(),'states':state[i].tolist(),'part_whole_angles':t[i].tolist()})
    return model,whole,parts,{'ids':cache['ids'],'labels':cache['labels'],'angles':t,
                            'clean_angles':old.array(clean_theta),'depths':depths,'chain_gaps':old.array(cg)}


def run_model(label,cache,inputs,out,device,reference,batches):
    from geoopt import PoincareBall
    start=time.monotonic();cache['label']=label
    model,whole,parts,objects=population(cache,device)
    np.savez_compressed(out/'private'/(label+'_objects.npz'),**objects)
    labels=torch.from_numpy(cache['labels']).to(device)
    sampler=old.SourceClassBatchSampler(cache['labels'],steps=batches,seed=4780)
    measures={g:Columns() for g in SCALES}
    gradients={(g,t):Columns() for g in SCALES for t in ('contrastive','radial')}
    store=Columns();cases={};queue=audit=None
    if label.startswith('TS18'):
        queue,qinfo=old.load_queue(inputs/(label+'_queue.pt'),device,9840)
        if qinfo['source_sha256']!=reference['checkpoints'][label]['source_sha256']:
            raise ValueError('Saved queue and checkpoint identity differ')
        audit=SelectionAudit(queue,batches)
    ball=PoincareBall(c=1.).to(device=device,dtype=whole.dtype)
    for panel in range(batches):
        ix=torch.tensor(sampler.global_plan(panel).indices,device=device,dtype=torch.long)
        w,p,y=whole[ix],parts[ix],labels[ix]
        if bool((y==y.flip(0)).any()):raise RuntimeError('Negative labels collide')
        for col,g in enumerate(SCALES):
            positive=ball.dist(w,p[:,col]);negative=ball.dist(w,p[:,col].flip(0))
            radial_gap=ball.dist0(w)-ball.dist0(p[:,col])
            measures[g].add(positive_angle=angles(w,p[:,col]),negative_angle=angles(w,p[:,col].flip(0)),
                positive_distance=positive,negative_distance=negative,distance_gap=negative-positive,
                radial_gap=radial_gap,contrastive_active=(positive-negative+4>0),
                radial_active=(1000/g-radial_gap>0))
            probe=gradient_probe(w,p[:,col],g)
            for term in ('contrastive','radial'):
                gradients[(g,term)].add(**{k:v for k,v in probe[term].items() if k!='weighted_batch_loss'})
        if queue is not None:
            seed=4780+(panel+1)*1009+70000027
            relation_panel(store,audit,cases,label,panel,w,p,y,ix,queue,seed)
        if (panel+1)%10==0:print(f'{label}: {panel+1}/{batches}',flush=True)
    model['intra']={};model['gradient']={}
    for g in SCALES:
        measured=measures[g].finish();model['intra'][str(g)]=cat_stats(measured)
        model['gradient'][str(g)]={}
        for term in ('contrastive','radial'):
            columns=gradients[(g,term)].finish()
            summary=gradient_summary(columns)
            if term=='radial':
                for k in ('whole_tangent_norm','part_tangent_norm'):
                    if summary[k]['max']>1e-8:
                        raise RuntimeError('Native radial loss unexpectedly has tangential gradient')
            model['gradient'][str(g)][term]=summary
            measured.update({term+'_'+k:v for k,v in columns.items()})
        np.savez_compressed(out/'private'/(label+f'_intra{g}.npz'),**measured)
    if audit is not None:
        records=store.finish()
        model['relations']=summarize_relations(records,audit,reference['models'][label]['relations'])
        model['cases']=list(cases.values())
        np.savez_compressed(out/'private'/(label+'_relations.npz'),**records)
        for role,values in audit.role.items():
            np.savez_compressed(out/'private'/(label+'_'+role+'_candidate_slots.npz'),
                ids=audit.ids,scales=audit.scale,depth=audit.depth,source=audit.source,exposures=audit.exposures,**values)
    model['wall_seconds']=time.monotonic()-start
    print(f'{label}: completed {model["wall_seconds"]:.1f}s',flush=True)
    return model


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--settings',type=Path,required=True)
    args=parser.parse_args();settings=json.loads(args.settings.read_text(encoding='utf-8-sig'))
    inputs=Path(settings['inputs']);out=Path(settings['root'])/'analysis'
    out.mkdir(exist_ok=False);(out/'private').mkdir()
    transfer=json.loads((Path(settings['root'])/'input_transfer.json').read_text())['files_sha256']
    before={name:sha(inputs/name) for name in transfer}
    if before!=transfer:raise ValueError('Transferred cache identity changed')
    identity,export_gates=checked_export_identity(inputs)
    reference=json.loads((inputs/'reference94.json').read_text())
    torch.set_num_threads(4);torch.set_grad_enabled(False)
    # Keep production FP32 for exact seeded replay; FP64 only for angles and gradient probes.
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    uuid=idle_gpu(settings['gpu'])
    if torch.cuda.is_initialized():raise RuntimeError('GPU initialized before final resource check')
    device=torch.device('cuda:0')
    result={'version':'parthier-followup95-v1','analysis_commit':settings['commit'],'config':{
        'count':9840,'batches':200,'seed':4780,'K':20,'T':50,'queue_capacity':2048,
        'current_candidates':192,'tau':.1,'hier_margin':.1,'ord_chain_margin':.2,
        'angle_direction_undefined_norm':1e-6,'gradient_angle_zero_tolerance':1e-10,
        'gradient_space':'Detached output-coordinate Euclidean descent; both endpoints and shared flip negatives',
        'optimizer_steps':0,'encoder_forwards':0,'model_parameter_backward':0,
        'candidate_equal_depth':'Independent copy; batch median native depth; fixed directions/ijk',
        'hotspot_expected':'ALL draws only, not collision-conditioned',
        'device':'cuda:0','server':'8000','torch':torch.__version__},
        'inputs':{'export_gates':export_gates,'shared_identity':identity,'files_sha256':before},'models':{}}
    for label in old.LABELS:
        with np.load(inputs/(label+'.npz'),allow_pickle=False) as data:cache=old.validate_cache(data,9840)
        result['models'][label]=run_model(label,cache,inputs,out,device,reference,settings['batches'])
        old.write_json(out/'summary.partial.json',result)
        torch.cuda.empty_cache()
    if {name:sha(inputs/name) for name in transfer}!=before:raise RuntimeError('An input was mutated')
    result['acceptance']={'all_inputs_unchanged':True,'replay_gates_passed':True,
        'no_encoder_or_optimizer':True,'output_leaf_gradients_only':True,'models':4,
        'part_whole_angle_populations':12,'server':'8000'}
    old.write_json(out/'summary.json',result)
    print('ANALYSIS_COMPLETE',flush=True)


if __name__=='__main__':main()
