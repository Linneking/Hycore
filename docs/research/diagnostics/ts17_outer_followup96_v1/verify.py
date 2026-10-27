"""Independent CPU recount/numerical closure; writes only to a new QA directory."""
from __future__ import annotations
import argparse,json,csv,sys
from pathlib import Path
import numpy as np
from science import *


def literal_rows(x,y):
    """Float64 rowwise Mobius formula, retaining the original denominator eps."""
    x=np.asarray(x,np.float64);y=np.asarray(y,np.float64);c=float(np.float32(.1))
    xy=-np.sum(x*y,axis=1);xx=np.sum(x*x,1);yy=np.sum(y*y,1)
    num=-(1+2*c*xy+c*yy)[:,None]*x+(1-c*xx)[:,None]*y
    difference=num/(1+2*c*xy+c*c*xx*yy+1e-5)[:,None]
    return 2/np.sqrt(c)*np.arctanh(np.clip(np.sqrt(c)*np.linalg.norm(difference,axis=1),-1+1e-5,1-1e-5))


def main():
    a=argparse.ArgumentParser();a.add_argument('--settings',type=Path,required=True);a.add_argument('--output',type=Path,required=True);args=a.parse_args();s=json.loads(args.settings.read_text());root=Path(s['root']);out=args.output;out.mkdir(exist_ok=False)
    summary=json.loads((root/'public/summary.json').read_text());z=dict(np.load(root/'private/readout.npz'));cache=dict(np.load(s['cache']));checks={};details={}
    def check(name,result):
        if not bool(result):raise AssertionError(name)
        checks[name]=True
    published=json.loads((root/'public/artifact_hashes.json').read_text());check('all_original_public_hashes',all(sha(root/'public'/f)==v for f,v in published.items()))
    pairs=z['pairs'];labels=cache['labels'].astype(int);check('4000_unique_unordered_pairs',pairs.shape==(4000,2) and len(np.unique(pairs,axis=0))==4000 and np.all(pairs[:,0]<pairs[:,1]))
    check('40_equal_classes',np.array_equal(np.bincount(labels[pairs[:,0]],minlength=40),np.full(40,100)) and np.array_equal(labels[pairs[:,0]],labels[pairs[:,1]]))
    rows=list(csv.DictReader((root/'private/all_pair_accounts.csv').open(encoding='utf-8-sig')));I=np.array([int(r['inner_id']) for r in rows]);O=np.array([int(r['outer_id']) for r in rows]);ball=cache['proxy_ball'];whole=cache['hier_mu'];h=depth(ball)
    check('fixed_inner_outer_membership',(h[I]<3.678016487278603).all() and (h[O]>=3.678016487278603).all())
    def costs(p):return np.maximum(literal_rows(whole[pairs[:,0]],p),literal_rows(whole[pairs[:,1]],p))
    cI=costs(ball[I]);cO=costs(ball[O]);cost_depth=costs(at_depth(ball[O],h[I]).astype(np.float32));cost_direction=costs(at_depth(ball[I],h[O]).astype(np.float32))
    for name,values,key in (('actual_cost_gap',cO-cI,'gap'),('radial_replacement_gap',cost_depth-cI,'gap_depth'),('direction_replacement_gap',cost_direction-cI,'gap_angle')):
        error=float(np.max(np.abs(values-z[key])));details[name+'_max_error']=error;check(name,error<5e-5)
    check('no_actual_or_virtual_outer_flip',min((cO-cI).min(),(cost_depth-cI).min(),(cost_direction-cI).min())>1e-5)
    hdepth=at_depth(ball[O],h[I]);dird=at_depth(ball[I],h[O])
    check('only_depth_direction_preserved',np.max(np.abs(unit(hdepth)-unit(ball[O])))<1e-12)
    check('only_direction_depth_preserved',np.max(np.abs(depth(dird)-h[O]))<1e-12)
    check('proxy_population_depth_matches',np.max(np.abs(h-z['h']))<1e-7)
    original_groups=np.full(512,-1)
    with np.load(Path(s['previous_root'])/'private/readout.npz') as old:used=old['counts'].any(0)
    original_groups[(h>=3.678016487278603)&~used]=0;original_groups[(h<3.678016487278603)&used]=1;original_groups[(h<3.678016487278603)&~used]=2;original_groups[(h>=3.678016487278603)&used]=3
    check('same512_ID_groups',np.array_equal(original_groups,z['group']))
    source_identity=json.loads((root/'public/source_hashes.json').read_text());material=Path(s['material'])
    for epoch in range(1,274):
        name='metrics_epoch_%03d.json'%epoch;check_name='history_'+str(epoch)
        check(check_name+'_SHA',sha(material/name)==source_identity['history_metric_SHA'][name])
        raw=json.loads((material/name).read_text())
        for ci,component in enumerate(('sample','proxy')):
            for di,domain in enumerate(('all_draws','noncollision','active_noncollision')):
                for ri,role in enumerate(('pair','triple')):
                    check(check_name+'_'+component+'_'+domain+'_'+role,np.array_equal(raw['structure'][component]['selected_ancestor_usage'][domain+'/'+role]['counts'],z['history'][epoch-1,ci,di,ri]))
    check('historical_opportunity_denominators',(z['history'][:,:,0].sum(-1)>0).all())
    early=z['history'][:5,0,0,0].sum(0);cold=z['group']==0
    rho_h=rank_correlation(z['h0'],early);rho_a=rank_correlation(z['theta0'],early)
    check('reported_initial_angle_correlation',np.isclose(rho_a,summary['initial_angle_early_pair_spearman']))
    quartet=[]
    for qi,indices in enumerate(np.array_split(np.argsort(z['h0'],kind='stable'),4)):
        quartet.append(dict(quartile=qi+1,proxies=len(indices),initial_depth_min=float(z['h0'][indices].min()),initial_depth_max=float(z['h0'][indices].max()),outer_unused=int(cold[indices].sum()),outer_unused_fraction=float(cold[indices].mean()),first5_sample_pair_share=float(early[indices].sum()/early.sum()),first5_sample_pair_per_proxy_vs_uniform=float(early[indices].sum()/early.sum()/len(indices)*512)))
    check('quartile_total512_and162',sum(x['proxies'] for x in quartet)==512 and sum(x['outer_unused'] for x in quartet)==162)
    result=dict(passed=True,check_count=len(checks),checks=checks,numerical_errors=details,initial_depth_vs_first5_sample_pair_spearman=rho_h,initial_angle_vs_first5_sample_pair_spearman=rho_a,initial_depth_quartiles=quartet,original_results_unchanged=True,CPU_only=True,network_forwards=0,optimizer_updates=0)
    write(out/'review_qa.json',result)
    sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'ts17_proxy_roles96_v1'));from pipeline import csvout
    csvout(out/'initial_depth_quartiles.csv',quartet)
    print(json.dumps({k:v for k,v in result.items() if k!='checks'},ensure_ascii=False))


if __name__=='__main__':main()
