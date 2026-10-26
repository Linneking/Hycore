"""CPU figures, 16 top4 cloud cards and a report, all on the audit server."""
from __future__ import annotations
import argparse,hashlib,json,sys,csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from core import *
from pipeline import csvout

COLORS=['#0072b2','#d55e00','#009e73']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})


def figure(fig,out,name,layout_top=1.):
    fig.tight_layout(rect=(0,0,1,layout_top));fig.savefig(out/(name+'.png'),dpi=180,bbox_inches='tight');fig.savefig(out/(name+'.svg'),bbox_inches='tight');plt.close(fig)


def pct(x):return '—' if x is None else f'{100*x:.2f}%'
def num(x):return '—' if x is None else f'{x:.3f}'
def mdtable(head,rows):return '\n'.join(['| '+' | '.join(head)+' |','| '+' | '.join(['---']*len(head))+' |']+['| '+' | '.join(map(str,row))+' |' for row in rows])


def figures(s,z,root,analysis_root=None):
    out=root/'public/figures';out.mkdir();h=z['h'];theta=z['theta'];counts=z['counts'];mask,bits,active=roles(counts)
    fig,axs=plt.subplots(2,2,figsize=(13,9))
    axs[0,0].hist(h,bins=np.arange(2.9,4.61,.025),weights=np.full(len(h),100/len(h)),color='#777777');axs[0,0].axvline(4.6,color='red',ls='--',label='Whole shell / cap4.6');axs[0,0].set(xlabel='Origin hyperbolic depth (c=0.1)',ylabel='All 512 proxies (%)',title='Saved best e273: full proxy population');axs[0,0].legend()
    for i,r in enumerate(ROLES):
        values=np.sort(h[mask[i]])
        if len(values):axs[0,1].plot(values,np.arange(1,len(values)+1)/len(values)*100,label=NAMES[r],color=COLORS[i])
    axs[0,1].set(xlabel='Depth',ylabel='Unique selected proxy ECDF (%)',title='Each proxy counted once within a role');axs[0,1].legend()
    for ax,value,title,ylabel in ((axs[1,0],h,'Depth by geometric role','Origin hyperbolic depth'),(axs[1,1],theta,'Nearest class-direction center angle','Degrees')):
        entries=[value[mask[i]] for i in range(3)];labels=[NAMES[r]+f'\nn={int(mask[i].sum())}' for i,r in enumerate(ROLES)]
        plot=ax.boxplot(entries,labels=labels,showfliers=True,patch_artist=True)
        for patch,color in zip(plot['boxes'],COLORS):patch.set_facecolor(color);patch.set_alpha(.45)
        ax.set(title=title,ylabel=ylabel)
    fig.suptitle('01 TS17 best: 4.6-shell queries, saved proxy depths; roles may overlap');figure(fig,out,'01_depth_angle')
    fig,axs=plt.subplots(2,2,figsize=(14,9));order=np.argsort(h);total=counts[:,order].sum(0);composition=np.divide(counts[:,order],total,out=np.zeros((3,512)),where=total>0)
    axs[0,0].stackplot(np.arange(512),100*composition,colors=COLORS,labels=[NAMES[r] for r in ROLES]);axs[0,0].set(xlabel='Proxy rank, shallow to deep',ylabel='S/C/T selection composition (%)',title='Unused proxies have zero height');axs[0,0].legend(fontsize=7)
    with np.load((analysis_root or root)/'private/relations.npz') as raw:
        for r,same,color in (('S',True,COLORS[0]),('C',False,COLORS[1])):
            keep=raw['distinct']&(raw['same']==same)&(raw['pair']!=raw['triple']);values=np.sort(h[raw['pair'][keep]]-h[raw['triple'][keep]])
            if len(values):
                ix=np.unique(np.linspace(0,len(values)-1,min(5000,len(values))).astype(int));axs[0,1].plot(values[ix],100*(ix+1)/len(values),label=NAMES[r],color=color)
    axs[0,1].axvline(0,color='black',ls='--');axs[0,1].set(xlabel='Paired depth gap h(pair)-h(triple)',ylabel='Noncollision distinct-object ECDF (%)');axs[0,1].legend()
    combos=s['roles']['exclusive'];axs[1,0].bar([x['roles'] for x in combos],[100*x['of_selected'] for x in combos],color='#777777');axs[1,0].set(ylabel='Of selected proxy union (%)',title='Seven mutually exclusive role combinations')
    for i,r in enumerate(ROLES):
        values=np.argsort(h);weight=z['expected'][i,values];axs[1,1].plot(h[values],100*np.cumsum(weight)/weight.sum(),label=NAMES[r],color=COLORS[i])
    axs[1,1].set(xlabel='Depth',ylabel='Analytic Gumbel expected usage ECDF (%)',title='Original tau=0.1; no arbitrary activation threshold');axs[1,1].legend()
    figure(fig,out,'02_role_overlap_and_probability')
    fig,axs=plt.subplots(2,2,figsize=(14,9));cov=[r for r in s['coverage'] if r['role']!='All'];x=np.arange(len(cov))
    axs[0,0].bar(x-.18,[100*r['coverage_fraction'] for r in cov],.36,label='Unique objects / 9840');axs[0,0].bar(x+.18,[100*r['capacity_normalized'] for r in cov],.36,label='Unique objects / max possible slots');axs[0,0].set_xticks(x,labels=[r['role'] for r in cov]);axs[0,0].set(ylabel='Coverage (%)');axs[0,0].legend(fontsize=7)
    for r in ('S','C','T','P','Used'):
        values=np.sort(z['exposure_'+r])[::-1];values=values[values>0];axs[0,1].plot(np.arange(1,len(values)+1),values,label=r)
    axs[0,1].set(xlabel='Object rank among covered IDs',ylabel='Number of distinct proxies selecting object',yscale='log');axs[0,1].legend()
    for i,r in enumerate(cov):
        vals=z['null_'+r['role']];axs[1,0].boxplot(vals,positions=[i],widths=.5,showfliers=False);axs[1,0].scatter(i,r['max_exposure'],marker='x',color='red',s=45)
    axs[1,0].set_xticks(x,labels=[r['role'] for r in cov]);axs[1,0].set(ylabel='Maximum object exposure',title='Class-composition-matched null; red x = actual')
    purity=[s['roles']['pure_all_fraction'],s['roles']['pure_selected_fraction']]+[s['roles']['groups'][r]['pure_fraction'] for r in ROLES]
    axs[1,1].bar(['All512','Selected','S','C','T'],[100*v if v is not None else 0 for v in purity],color=['#999999','#555555']+COLORS);axs[1,1].set(ylabel='Pure-class top4 proxies within each set (%)',ylim=(0,105));figure(fig,out,'03_top4_coverage_and_hotspots')
    fig,axs=plt.subplots(1,2,figsize=(13,4));x=np.arange(5)
    for i,r in enumerate(ROLES):axs[0].plot(x,[100*t['groups'][r]['of_selected'] for t in s['seed_stability']],marker='o',label=NAMES[r],color=COLORS[i])
    axs[0].set_xticks(x,labels=list(range(4780,4785)));axs[0].set(xlabel='Audit seed, 200 batches each',ylabel='Role membership / seed selected union (%)');axs[0].legend()
    axs[1].plot(x,[100*t['cumulative_selected']/512 for t in s['seed_stability']],marker='o');axs[1].set_xticks(x,labels=[200,400,600,800,1000]);axs[1].set(xlabel='Cumulative frozen readout batches',ylabel='Selected proxy union / 512 (%)');figure(fig,out,'04_seed_stability')


def cloud_cards(s,z,settings,root):
    import h5py
    clouds=[];labels=[]
    for p in sorted(Path(settings['data_dir']).glob('ply_data_train*.h5')):
        with h5py.File(p,'r') as f:clouds.append(f['data'][:,:1024].astype(np.float32));labels.append(f['label'][:].reshape(-1).astype(np.int64))
    clouds=np.concatenate(clouds);labels=np.concatenate(labels)
    digest=hashlib.sha256(clouds.tobytes()).hexdigest()
    if clouds.shape!=(9840,1024,3) or digest!='dc82f898decf86fed478184b9a6e72f8fa95d4165b0150a449a2fa5e178f27f7' or not np.array_equal(labels,z['labels']):raise RuntimeError('Figure objects do not match cached network inputs')
    cards=root/'public/cases';cards.mkdir();nn4=z['nn4'];bits=z['bits'];h=z['h'];theta=z['theta'];near=z['near'];counts=z['counts']
    rows=[];counter=collections.Counter()
    def draw(ax,obj):
        points=clouds[obj];ax.scatter(points[:,0],points[:,2],points[:,1],s=.65,c='#334455',alpha=.75,linewidths=0,rasterized=True)
        ax.set(xlim=(-1.05,1.05),ylim=(-1.05,1.05),zlim=(-1.05,1.05));ax.set_box_aspect((1,1,1));ax.view_init(elev=18,azim=45);ax.set_axis_off();ax.set_title(f'{CLASSES[labels[obj]]}\nID {obj}',fontsize=9,pad=0)
    for example in s['examples']:
        r=example['role'];p=example['proxy_id'];counter[r]+=1;name=f'{r}_{counter[r]:02d}_proxy{p:03d}';fig=plt.figure(figsize=(12,3.5))
        for j,obj in enumerate(nn4[p]):draw(fig.add_subplot(1,4,j+1,projection='3d'),int(obj))
        roles_text='+'.join(ROLES[i] for i in range(3) if bits[p]&(1<<i))
        fig.suptitle(f'{NAMES[r]} | proxy {p} | depth {h[p]:.3f} | roles {roles_text} | nearest {CLASSES[near[p]]}: {theta[p]:.2f} deg\nS/C/T draws {counts[:,p].tolist()} | depth quartile {example["depth_quartile"]}',fontsize=10)
        figure(fig,cards,name,layout_top=.80)
        rows.append(dict(**example,depth=float(h[p]),nearest_class=CLASSES[near[p]],angle=float(theta[p]),figure='cases/'+name+'.png',top4_labels=[CLASSES[labels[o]] for o in nn4[p]],top4_ids=nn4[p].tolist()))
    for r in ROLES:
        items=[x for x in s['examples'] if x['role']==r];fig=plt.figure(figsize=(12,2.65*len(items)))
        for row,item in enumerate(items):
            p=item['proxy_id']
            for j,obj in enumerate(nn4[p]):
                ax=fig.add_subplot(len(items),4,row*4+j+1,projection='3d');draw(ax,int(obj))
                if j==0:ax.text2D(-.12,.45,f'P{p}\nh={h[p]:.3f}\n{theta[p]:.2f} deg',transform=ax.transAxes,fontsize=9)
        fig.suptitle(NAMES[r]+' | saved proxy top4 from all 9840 depth4.6 queries',fontsize=12);figure(fig,root/'public/figures','05_'+r+'_cloud_examples',layout_top=1-.6/(2.65*len(items)))
    write(root/'private/case_manifest.json',rows)
    write(root/'public/case_summary.json',[{k:v for k,v in x.items() if k!='top4_ids'} for x in rows])
    return digest


def report(s,root):
    out=root/'public';groups={g['role']:g for g in s['group_distributions'] if g['weighting']=='unique_proxy'};weighted={g['role']:g for g in s['group_distributions'] if g['weighting']=='draw_frequency'}
    role=s['roles'];draw=s['draws'];lines=['# TS17 best代理角色与top4形态审查结果','',
        '仅TS17/B12 best e273，固定保存权重、全9840训练对象。正常网络嵌入复用85已验收缓存；选代理和反查样本统一使用c0.1深度4.6视图，代理保留原深度。五个seed各200批正常CB64，共1000批。没有编码器前向、反传、参数更新、BN更新或test读出。','',
        '## 1. 先看“双层是否对应pair／triple”','',
        f"全部512代理深度范围{num(s['geometry']['proxy_depth']['min'])}–{num(s['geometry']['proxy_depth']['max'])}，中位{num(s['geometry']['proxy_depth']['median'])}；raw参数越cap为{s['geometry']['proxy_raw_cap_count']}/512。独立三带宽KDE的两段稳定性判定为 **{s['depth_bands']['stable_two_bands']}**，分界为{num(s['depth_bands']['threshold'])}。判定规则先于角色分类固定；不强迫连续分布二分。",'',
        mdtable(['角色','不同代理数','深度中位 [Q1,Q3]','最近类中心角中位 [Q1,Q3]（度）','选择频次加权深度中位'],[[NAMES[r],role['groups'][r]['proxies'],f"{num(groups[r]['depth']['median'])} [{num(groups[r]['depth']['p25'])},{num(groups[r]['depth']['p75'])}]",f"{num(groups[r]['nearest_class_angle']['median'])} [{num(groups[r]['nearest_class_angle']['p25'])},{num(groups[r]['nearest_class_angle']['p75'])}]",num(weighted[r]['depth']['median'])] for r in ROLES]),'',
        '箱形图按每组不同代理各计一次；同一代理可以跨组出现。最近类是全量40类单位方向中心中夹角最小的一类，不是最近对象。','',
        '![深度和角度](figures/01_depth_angle.png)','',
        mdtable(['角色','同ijk、A≠B关系数','h(pair)−h(triple)中位','pair更深比例（>0.001）','近零比例','逆序比例'],[[NAMES[x['role']],x['draws'],num(x['gap']['median']),pct(x['positive']),pct(x['tied']),pct(x['negative'])] for x in s['paired'] if x['cohort']=='noncollision']),'',
        f"pair倾向与深度的Spearman相关（平均秩处理并列）为{num(s['depth_pair_preference_rank_correlation'])}。完整合法索引抽样{draw['draws']}条，其中三个不同对象ID的主分析关系{draw['distinct_draws']}条；重复对象关系占{pct(1-draw['distinct_draws']/draw['draws'])}。主分析A=B占{pct(draw['collisions_distinct']/draw['distinct_draws'])}，保留在角色分布中，非碰撞次序另表。",'',
        '![角色与原Gumbel概率敏感性](figures/02_role_overlap_and_probability.png)','',
        '主选择为确定性minimax，不能称训练随机轨迹重放。原tau0.1下独立Gumbel选择的解析概率及非碰撞期望权重另行累计，详见role_distribution.csv的gumbel_expected两种口径；不以随机曾抽中一次定义活跃代理。','',
        '## 2. 活跃代理、角色交叠与纯类top4','',
        f"本次合并活跃集合M_used=**{role['selected']}**，未入选{role['unused']}。主角色比例排除未入选代理；‘未入选’仅指本次有限冻结审查。",'',
        mdtable(['角色','数量','占活跃并集','占全512','组内纯类top4比例'],[[NAMES[r],role['groups'][r]['proxies'],pct(role['groups'][r]['of_selected']),pct(role['groups'][r]['of512']),pct(role['groups'][r]['pure_fraction'])] for r in ROLES]),'',
        f"全512中纯类top4为{role['pure_all_count']}/512（{pct(role['pure_all_fraction'])}）；活跃代理中为{role['pure_selected_count']}/{role['selected']}（{pct(role['pure_selected_fraction'])}）。纯类指top4的四个GT标签完全相同。两个分母的分子各自限制在相应集合，不能混除。",'',
        mdtable(['互斥角色组合','代理数','占活跃并集'],[[x['roles'],x['count'],pct(x['of_selected'])] for x in role['exclusive']]),'',
        '七种组合合计100%；S/C/T三个重叠集合的比例不要求合计100%。','',
        '## 3. top4覆盖与热点','',
        mdtable(['角色集合','不同代理','覆盖对象数','占9840','占槽位覆盖上限','最大曝光','类组成匹配随机最大曝光p95','随机尾概率'],[[x['role'],x['proxies'],x['unique_objects'],pct(x['coverage_fraction']),pct(x['capacity_normalized']),x['max_exposure'],num(x.get('null_max_p95')),num(x.get('null_max_empirical_p'))] for x in s['coverage']]),'',
        '每个代理只贡献一次top4，从全9840个等深whole按双曲距离选取四个不同对象。即使全512代理都参与，只有2048个槽位，覆盖上限20.81%；覆盖低于100%本身不等于塌缩。热点曝光按不同代理计数。随机对照固定各代理四个邻居的标签组成，在相应类内重选不同对象1000次；尾概率=(1+超过实测最大值次数)/1001。真实热点也可能反映方向上常见的对象，不单凭集中度宣称语义失败。','',
        '![覆盖与热点](figures/03_top4_coverage_and_hotspots.png)','',
        '## 4. 16个代理的四点云形态卡','',
        '同类pair八例、异类pair四例、triple四例。按角色内深度四分位取例，层内优先兼顾纯类与混类，随后固定选择规则；未按形态好看程度筛选。跨角色复用代理会明确标记。每例是同一代理的top4对象并排，相机、原始坐标和点数一致。每卡标明代理深度、多重角色、使用量及最近类夹角。','',
        '![同类pair八例](figures/05_S_cloud_examples.png)','', '![异类pair四例](figures/05_C_cloud_examples.png)','', '![triple四例](figures/05_T_cloud_examples.png)','',
        '单卡高清图见[cases目录](cases/)，逐卡说明见[case_summary.json](case_summary.json)。对象ID对应关系、完整邻居表和逐关系原件只保留在服务器私有目录。','',
        '## 5. 稳定性、代理内部目标与边界','',
        '![五组审查seed](figures/04_seed_stability.png)','',
        f"代理内部K20/T50另有每角色{s['proxy_internal']['draws_per_role']}次读出，pair/triple使用频次加权深度中位为{num(s['proxy_internal']['pair_depth']['median'])}/{num(s['proxy_internal']['triple_depth']['median'])}。该项参与原训练，但没有样本同异类标签，不混入S/C/T分母；冻结选择关联不能单独证明分层由哪一损失造成。",'',
        '最终判断需同时结合深度分布、角色交叠、同ijk的深度次序和Gumbel概率复核；边际箱形图差异不等于互斥的pair/triple两类节点。类中心夹角和标签纯度也不是独立形态层次证明。','',
        '## 6. 数据、身份与验收','',
        f"权重SHA `{s['source_verification']['checkpoint_sha256']}`；缓存SHA `{s['source_verification']['cache_sha256']}`。前向映射重建误差：whole {s['source_verification']['shell_max_abs_error']:.3g}、代理 {s['source_verification']['proxy_max_abs_error']:.3g}；top4与85可信邻居距离最大差{s['source_verification']['top4_cached_cost_max_abs_error']:.3g}。输入/权重哈希在计算前后均未变。",'',
        '完整统计：[summary.json](summary.json)、[代理明细](proxy_statistics.csv)、[角色分布](role_distribution.csv)、[角色交叠](role_combinations.csv)、[同关系深度差](paired_depth_gap.csv)、[覆盖与热点](coverage_hotspots.csv)。角色明细保留512个代理，包括未入选者；报告主比例依计划排除未入选。原模型/BN/优化器未构造或更新。','']
    (out/'report.md').write_text('\n'.join(lines),encoding='utf-8')


def main():
    p=argparse.ArgumentParser();p.add_argument('--settings',type=Path,required=True);a=p.parse_args();settings=json.loads(a.settings.read_text());root=Path(settings['root'])
    s=json.loads((root/'public/summary.json').read_text())
    with np.load(root/'private/readout.npz') as f:z={k:f[k] for k in f.files}
    figures(s,z,root);digest=cloud_cards(s,z,settings,root);report(s,root)
    write(root/'publish_status.json',dict(status='completed',input_sha256=digest,case_cards=len(s['examples']),png_files=len(list((root/'public').rglob('*.png'))),network_forwards=0))


if __name__=='__main__':main()
