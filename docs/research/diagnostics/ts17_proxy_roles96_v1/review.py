"""Read-only CPU reconciliation of completed readout into a NEW result directory.

No GPU selection, encoder, optimizer or new relation sampling is performed.
The original audit directory and its public artifact hashes remain unchanged.
"""
from __future__ import annotations
import argparse,json,shutil,subprocess,datetime
from pathlib import Path
import numpy as np
from core import *
from pipeline import csvout,CP_SHA,CACHE_SHA
from publish import figures,cloud_cards,report,figure,plt,COLORS,mdtable,num,pct


def main():
    a=argparse.ArgumentParser();a.add_argument('--settings',type=Path,required=True);a.add_argument('--output',type=Path,required=True);args=a.parse_args()
    settings=json.loads(args.settings.read_text());original=Path(settings['root']);out=args.output
    out.mkdir(exist_ok=False);(out/'private').mkdir();(out/'public').mkdir()
    old_hashes=json.loads((original/'public/artifact_hashes.json').read_text())
    def original_unchanged():
        assert all(sha(original/'public'/name)==digest for name,digest in old_hashes.items())
        assert sha(settings['checkpoint'])==CP_SHA and sha(settings['cache'])==CACHE_SHA
    original_unchanged()
    s=json.loads((original/'public/summary.json').read_text())
    with np.load(original/'private/readout.npz') as f:z={k:f[k] for k in f.files}
    h,counts,pure=z['h'],z['counts'],z['pure'];mask,bits,used=roles(counts)
    checks={}
    def check(name,value):
        if not bool(value):raise AssertionError(name)
        checks[name]=True
    with np.load(original/'private/relations.npz') as f:
        raw={k:f[k] for k in ('i','j','k','pair','triple','same','distinct','seed')}
    distinct=(raw['i']!=raw['j'])&(raw['i']!=raw['k'])&(raw['j']!=raw['k'])
    check('three_distinct_ID_definition',np.array_equal(distinct,raw['distinct']))
    check('same_class_definition',np.array_equal(z['labels'][raw['i']]==z['labels'][raw['j']],raw['same']))
    for ri,(which,group) in enumerate(((raw['pair'],raw['same']),(raw['pair'],~raw['same']),(raw['triple'],np.ones(len(distinct),bool)))):
        check(ROLES[ri]+'_counts_from_raw_relations',np.array_equal(np.bincount(which[distinct&group],minlength=512),counts[ri]))
        check(ROLES[ri]+'_effective_from_raw_relations',np.array_equal(np.bincount(which[distinct&group&(raw['pair']!=raw['triple'])],minlength=512),z['effective'][ri]))
    check('pure_top4_labels',np.array_equal(np.all(z['labels'][z['nn4']]==z['labels'][z['nn4'][:,0]][:,None],axis=1),pure))
    check('top4_unique_IDs',all(len(set(row))==4 for row in z['nn4']))
    check('roles_summary',role_summary(counts,pure)==s['roles'])
    check('seed_counts_sum',np.array_equal(z['per_seed'].sum(0),counts))
    check('expected_probability_mass',np.allclose(z['expected'].sum(1),counts.sum(1),rtol=1e-6))
    check('expected_noncollision_mass',np.isclose(z['expected_effective'][0].sum()+z['expected_effective'][1].sum(),z['expected_effective'][2].sum(),rtol=1e-6))
    corrections=[]
    for g in s['group_distributions']:
        if g['weighting']=='unique_proxy':
            m=mask[ROLES.index(g['role'])]
            before=g['depth']['median'];g['depth']=stats(h[m]);g['nearest_class_angle']=stats(z['theta'][m])
            corrections.append(dict(role=g['role'],previous_depth_median=before,final_depth_median=g['depth']['median']))
    check('boxplot_unique_quantiles',all(np.isclose(g['depth']['median'],np.median(h[mask[ROLES.index(g['role'])]])) for g in s['group_distributions'] if g['weighting']=='unique_proxy'))
    for record in s['paired']:
        m=distinct&(raw['same'] if record['role']=='S' else ~raw['same'])
        if record['cohort']=='noncollision':m &= raw['pair']!=raw['triple']
        values=h[raw['pair'][m]]-h[raw['triple'][m]]
        check('paired_'+record['role']+'_'+record['cohort'],len(values)==record['draws'] and np.isclose(np.median(values),record['gap']['median']) and np.isclose(np.mean(values>1e-3),record['positive']))
    threshold=s['depth_bands']['threshold'];outer=h>=threshold if threshold is not None else None
    band_rows=[]
    if outer is not None:
        for name,m in [('All512',np.ones(512,bool)),('Selected',used),('Unused',~used)]+[(r,mask[i]) for i,r in enumerate(ROLES)]:
            band_rows.append(dict(set=name,total=int(m.sum()),inner=int((m&~outer).sum()),outer=int((m&outer).sum()),outer_fraction=float((m&outer).sum()/m.sum())))
    s['depth_band_populations']=band_rows
    internal_rows=[]
    for ri,r in enumerate(('internal_pair','internal_triple')):
        for label,w in (('unique_proxy',z['internal'][ri]>0),('draw_frequency',z['internal'][ri]),('gumbel_expected',z['internal_expected'][ri])):
            internal_rows.append(dict(role=r,weighting=label,**{'depth_'+k:v for k,v in role_stats(h,w,label=='unique_proxy').items()},outer_band_mass=float(w[outer].sum()/w.sum()) if outer is not None else None))
    s['proxy_internal_distributions']=internal_rows
    seed_rows=[]
    for si,seed in enumerate(range(4780,4785)):
        for ri,r in enumerate(ROLES):
            m=z['per_seed'][si,ri]>0
            seed_rows.append(dict(seed=seed,role=r,proxies=int(m.sum()),depth_median=float(np.median(h[m])),angle_median=float(np.median(z['theta'][m])),outer_count=int((m&outer).sum()) if outer is not None else None))
    s['seed_role_medians']=seed_rows
    cap_rows=[]
    for ri,r in enumerate(ROLES):
        cap_rows.append(dict(role=r,**{'gap_to_4p6_'+k:v for k,v in stats(4.6-h[mask[ri]]).items()},within_0p1=int((mask[ri]&(h>=4.5)).sum()),within_0p5=int((mask[ri]&(h>=4.1)).sum())))
    s['cap_distance_by_role']=cap_rows
    original_status=json.loads((original/'status.json').read_text())
    current_commit=subprocess.check_output(['git','-C',settings['repo'],'rev-parse','HEAD'],text=True).strip()
    s['execution']=dict(completed=True,readout_commit=original_status['commit'],postprocess_commit=current_commit,started_utc=original_status['started_utc'],readout_finished_utc=original_status['finished_utc'],readout_GPU_index=original_status['gpu']['index'],postprocess_GPU_used=False,review_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    s['statistical_conventions']=dict(unique_proxy_quantiles='numpy linear interpolation, consistent with boxplots',weighted_quantiles='inverse empirical weighted CDF',review='Only unique-node quartile convention reconciled; relation choices, frequencies, source features and top4 unchanged')
    for path in (original/'public').glob('*.csv'):shutil.copy2(path,out/'public'/path.name)
    csvout(out/'public/role_distribution.csv',[dict(role=g['role'],weighting=g['weighting'],**{'depth_'+k:v for k,v in g['depth'].items()},**{'angle_'+k:v for k,v in g['nearest_class_angle'].items()},outer_band_mass=g['outer_band_mass']) for g in s['group_distributions']])
    for filename,rows in (('depth_bands_by_role.csv',band_rows),('internal_role_distribution.csv',internal_rows),('per_seed_role_stats.csv',seed_rows),('cap_distance.csv',cap_rows)):
        csvout(out/'public'/filename,rows)
    write(out/'public/summary.json',s)
    figures(s,z,out,analysis_root=original);digest=cloud_cards(s,z,settings,out);report(s,out)
    fig,axs=plt.subplots(2,2,figsize=(13,9))
    rows=band_rows[:3];x=np.arange(len(rows));axs[0,0].bar(x,[v['inner'] for v in rows],label='Inner depth band');axs[0,0].bar(x,[v['outer'] for v in rows],bottom=[v['inner'] for v in rows],label='Outer depth band');axs[0,0].set_xticks(x,labels=[v['set'] for v in rows]);axs[0,0].set(ylabel='Distinct proxies',title=f'Independent depth threshold {threshold:.3f}');axs[0,0].legend()
    for off,weighting in ((-.18,'draw_frequency'),(.18,'gumbel_expected')):
        rows=[g for g in s['group_distributions'] if g['weighting']==weighting]
        axs[0,1].bar(np.arange(3)+off,[100*v['outer_band_mass'] for v in rows],.36,label=weighting)
    axs[0,1].set_xticks(np.arange(3),labels=ROLES);axs[0,1].set(ylabel='Usage mass in outer band (%)',title='Sample-driven roles rarely select outer band');axs[0,1].legend(fontsize=8)
    entries=[h[raw['pair'][distinct&(raw['same']==same)&(raw['pair']!=raw['triple'])]]-h[raw['triple'][distinct&(raw['same']==same)&(raw['pair']!=raw['triple'])]] for same in (True,False)]
    axs[1,0].boxplot(entries,labels=['Same-class pair','Cross-class pair'],showfliers=False);axs[1,0].axhline(0,color='black',ls='--');axs[1,0].set(ylabel='h(pair) - h(triple)',title='Same relation; distinct IDs and A != B')
    for off,weighting in ((-.18,'draw_frequency'),(.18,'gumbel_expected')):
        rows=[g for g in internal_rows if g['weighting']==weighting];axs[1,1].bar(np.arange(2)+off,[100*v['outer_band_mass'] for v in rows],.36,label=weighting)
    axs[1,1].set_xticks(np.arange(2),labels=['Proxy-internal pair','Proxy-internal triple']);axs[1,1].set(ylabel='Usage mass in outer band (%)',title='Separate proxy-internal readout, not S/C/T');axs[1,1].legend(fontsize=8)
    figure(fig,out/'public/figures','06_depth_bands_and_paired_gaps')
    outer_all=next(v for v in band_rows if v['set']=='All512');outer_used=next(v for v in band_rows if v['set']=='Selected')
    lead=['## 本次结论','',
        '**两段深度分布存在，但不支持“较深一段是pair、较浅一段是triple”这一整体分工假说。** '+f"以独立谷值{threshold:.3f}分段，外层共{outer_all['outer']}个代理，其中仅{outer_used['outer']}个在本次样本驱动读出中入选；其余{outer_all['outer']-outer_used['outer']}个没有入选。同类pair、异类pair和triple基本都使用内层代理。这里的外层仅是相对全体代理的较深一段，离4.6 cap仍有距离。",'',
        '**在已入选代理内部，pair比triple深的使用偏好成立。** 同一ijk、A≠B时，同类pair更深的比例73.89%，异类pair94.31%，深度差中位分别0.034和0.200；但S的25个代理全属于C，C的142个全属于T（T共291个），不存在仅pair的独占集合。应描述成重叠代理的相对使用偏好，不能据此称为两个独立语义层。','',
        '最近类角度随S→C→T增大，top4也存在超过类组成随机对照的热点。这些支持方向使用差异与检索集中现象；仅凭类标签、径向次序和展示卡，还不能证明独立形态层次。','',
        mdtable(['集合','内段代理数','外段代理数','合计'],[[v['set'],v['inner'],v['outer'],v['total']] for v in band_rows]),'',
        '![深浅段、同关系深度差和代理内部项](figures/06_depth_bands_and_paired_gaps.png)','',
        '原tau0.1 Gumbel概率下，外段的S/C/T期望选择质量分别为'+ '／'.join(pct(g['outer_band_mass']) for g in s['group_distributions'] if g['weighting']=='gumbel_expected')+'，仍不足以将外段解释为样本pair层。有限确定性读出的“未入选”不等于训练历史中从未使用；代理内部项的冻结读出另见附表，不与样本角色混算。','',
        '执行状态：8000已完成，GPU已释放；本次只读best e273，复用匹配的正常嵌入后重新构建4.6视图。后处理只统一不同代理统计与箱形图的四分位定义，没有改变选择结果、top4或频次。','']
    path=out/'public/report.md';body=path.read_text(encoding='utf-8');first,rest=body.split('\n',1);path.write_text(first+'\n\n'+'\n'.join(lead)+rest,encoding='utf-8')
    with path.open('a',encoding='utf-8') as f:
        f.write('\n附加表：[深浅段计数](depth_bands_by_role.csv)、[代理内部项](internal_role_distribution.csv)、[各seed中位数](per_seed_role_stats.csv)、[距cap统计](cap_distance.csv)。不同代理的分位数采用线性插值，频次／概率加权分位数采用经验加权CDF。\n')
        f.write(f"\n读出源码 `{s['execution']['readout_commit']}`；CPU整理源码 `{current_commit}`。见[验收](acceptance.json)与[独立复算](review_qa.json)。\n")
    original_unchanged();check('original_results_and_source_SHA_unchanged',True)
    write(out/'public/review_qa.json',dict(passed=True,checks=checks,unique_quantile_reconciliation=corrections,original_public_files_verified=len(old_hashes),source_data_unchanged=True))
    write(out/'public/acceptance.json',dict(passed=True,**s['execution'],CPU_contracts=12,raw_recount_checks=len(checks),source_SHA_unchanged=True,original_results_unchanged=True,pointcloud_input_sha256=digest,network_forwards=0,backward_calls=0,optimizer_updates=0,BN_updates=0,test_objects=0))
    write(out/'public/artifact_hashes.json',{str(f.relative_to(out/'public')):sha(f) for f in (out/'public').rglob('*') if f.is_file() and f.name!='artifact_hashes.json'})
    write(out/'status.json',dict(status='completed',CPU_only=True,public_files=len(list((out/'public').rglob('*'))),original_results_unchanged=True))
    print(json.dumps(dict(status='completed',output=str(out),checks=len(checks),depth_bands=band_rows,internal=internal_rows),ensure_ascii=False))


if __name__=='__main__':main()
