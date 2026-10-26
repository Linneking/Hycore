"""Five figure groups and a compact Chinese report from completed audit95."""
from __future__ import annotations
import argparse,csv,hashlib,json,math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ORDER=('TS02_best','TS02_e300','TS18_best','TS18_e300')
PH=('TS18_best','TS18_e300')
NAMES={'TS02_best':'HyCoRe best e189','TS02_e300':'HyCoRe e300','TS18_best':'PartHier best e201','TS18_e300':'PartHier e300'}
ZH={'TS02_best':'原版best','TS02_e300':'原版e300','TS18_best':'PartHier best','TS18_e300':'PartHier e300'}
SCALES=(200,400,600)
COLORS=('#0072b2','#009e73','#d55e00')
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})

def pct(x): return '—' if x is None else f'{100*x:.2f}%'
def num(x,n=2): return '—' if x is None else f'{x:.{n}f}'
def save(fig,out,name):
    fig.tight_layout();fig.savefig(out/(name+'.png'),dpi=180,bbox_inches='tight');fig.savefig(out/(name+'.svg'),bbox_inches='tight');plt.close(fig)
def csvout(out,name,rows):
    if not rows:return
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with (out/(name+'.csv')).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def flatstats(item,prefix='',fields=('count','median','p90','mean','p10','min','max')):
    return {prefix+k:item.get(k) for k in fields if k in item}


def export_tables(models,out):
    groups=[];hot=[];tops=[];choices=[];bins=[];angs=[];intra=[];grad=[];og=[];oc=[];cs=[];hf=[];hs=[];ge=[]
    for label,m in models.items():
        for scale,a in m['angles'].items():
            angs.append({'model':label,'scale':scale,'undefined':a['undefined'],**flatstats(a,fields=('count','p5','p25','median','p75','p90','p95')),'clean_median':a['clean_angle']['median']})
        for scale,x in m['intra'].items():
            intra.append({'model':label,'scale':scale,**{k+'_mean':v['mean'] for k,v in x.items()},
                          **{k+'_median':v['median'] for k,v in x.items() if k not in ('contrastive_active','radial_active')}})
            for term,t in m['gradient'][scale].items():
                grad.append({'model':label,'scale':scale,'term':term,**{k:t[k] for k in ('total','defined','undefined','increase_fraction','decrease_fraction','stationary_fraction')},
                             **{k+'_mean':t[k]['mean'] for k in ('angle_rate','whole_tangent_norm','part_tangent_norm','whole_gradient_norm','part_gradient_norm')},
                             **{k+'_max':t[k]['max'] for k in ('whole_tangent_norm','part_tangent_norm')}})
        for edge,counts in zip(('200_400','400_600','600_whole'),m['chain']['per_edge']):
            cs.append({'model':label,'edge':edge,'count':m['chain']['count'],**dict(zip(('wrong_or_tied','ordered_short','margin_pass'),counts))})
        if 'relations' not in m:continue
        r=m['relations'];sel=r['selection']
        for g in sel['groups']:
            row={k:v for k,v in g.items() if not isinstance(v,dict)}
            for field in ('candidate_depth','selected_depth','selected_within_group_percentile'):row.update(flatstats(g[field],field+'_'))
            groups.append({'model':label,**row})
        for h in sel['hotspots']:
            hot.append({'model':label,'role':h['role'],'source':h['source'],'exposed_entries':h['exposed_entries'],'used_entries':h['used_entries'],
                'actual_top1_fraction':h['actual']['top1_fraction'],'expected_top1_fraction':h['expected']['top1_fraction'],
                'actual_top5pct_fraction':h['actual']['top5pct_fraction'],'expected_top5pct_fraction':h['expected']['top5pct_fraction'],
                'merged_object_scale_top1_fraction':h['object_scale_merged']['actual']['top1_fraction'],
                'per_million_exposure_p90':h['per_million_exposure']['p90']})
        tops.extend({'model':label,**x} for x in sel['top_entries'])
        for q in r['choice_quality']:
            row={'model':label,'role':q['role'],'view':q['view']}
            for key,value in q.items():
                if isinstance(value,dict):row.update(flatstats(value,key+'_'))
            choices.append(row)
        for b in r['pair_bins']:
            row={'model':label,**{k:v for k,v in b.items() if not isinstance(v,dict)}}
            for key,value in b.items():
                if isinstance(value,dict):row.update(flatstats(value,key+'_',('median','p90')))
            bins.append(row)
        og.extend({'model':label,**g} for g in r['ord_groups'])
        for a,row in enumerate(r['ord']['cross_3x3']):
            for b,n in enumerate(row):oc.append({'model':label,'A_minij_state':a,'B_A_state':b,'count':n,'percentage':100*n/r['valid_draws']})
        for pattern,n in enumerate(r['hier']['failure_patterns']):hf.append({'model':label,'failure_bits_i1_j2_k4':pattern,'count':n,'percentage':100*n/r['valid_draws']})
        for edge,counts in zip(('i','j','k'),r['hier']['per_edge']):hs.append({'model':label,'edge':edge,'count':r['valid_draws'],**dict(zip(('wrong_or_tied','ordered_short','margin_pass'),counts))})
        for g in r['geometry_groups']:
            row={'model':label,'same_ij':g['same_ij'],'hier_pass':g['hier_pass'],'count':g['count']}
            for key,value in g.items():
                if isinstance(value,dict):row.update(flatstats(value,key+'_',('mean','median','p90')))
            ge.append(row)
    for name,rows in (('candidate_supply',groups),('hotspots',hot),('top_hotspots',tops),('choice_quality',choices),('pair_geometry_bins',bins),('part_whole_angles',angs),
        ('intra',intra),('gradients',grad),('ord_groups',og),('ord_cross',oc),('chain_states',cs),('hier_patterns',hf),('hier_states',hs),('geometry_groups',ge)):
        csvout(out,name,rows)


def plots(models,analysis,out):
    fig,axes=plt.subplots(3,2,figsize=(13,12))
    for col,label in enumerate(PH):
        r=models[label]['relations'];groups=[x for x in r['selection']['groups'] if x['role']=='A'];n=r['draws'];x=np.arange(6)
        for offset,key,name,color in ((-1,'actual_all','Actual all draws','#0072b2'),(0,'expected_all','Softmax expected','#009e73'),(1,'equal_depth_expected_all','Equal candidate depth','#d55e00')):
            axes[0,col].bar(x+offset*.24,[100*g[key]/n for g in groups],.24,label=name,color=color)
        axes[0,col].set_xticks(x,labels=['cur200','cur400','cur600','Q200','Q400','Q600']);axes[0,col].set_ylabel('A selections / expectations (%)');axes[0,col].set_title(NAMES[label]);axes[0,col].legend(fontsize=7)
        h=models[label]['relations']['selection']['hotspots'];labels=[];a=[];e=[]
        for item in h:
            labels.append(item['role']+' '+item['source']);a.append(100*item['actual']['top1_fraction']);e.append(100*item['expected']['top1_fraction'])
        xx=np.arange(len(h));axes[1,col].bar(xx-.18,a,.36,label='Actual top1');axes[1,col].bar(xx+.18,e,.36,label='Expected top1')
        axes[1,col].set_xticks(xx,labels=labels);axes[1,col].set_ylabel('Share of source selections (%)');axes[1,col].legend(fontsize=7)
        with np.load(analysis/'private'/(label+'_relations.npz')) as data:
            mask=data['valid']&data['same_ij'];ix=np.flatnonzero(mask)[::40]
            axes[2,col].scatter(data['theta_ij'][ix],data['a_depth'][ix],s=4,alpha=.25,label='Selected A')
            axes[2,col].scatter(data['theta_ij'][ix],data['midpoint_depth'][ix],s=4,alpha=.25,label='Continuous midpoint')
        axes[2,col].set_xlabel('Actual same-class i/j angle (degrees)');axes[2,col].set_ylabel('Depth');axes[2,col].legend(fontsize=7)
    fig.suptitle('01 Selection mechanism | source-matched supply, probability, exposure and geometry',fontsize=12);save(fig,out,'01_selection')

    fig,axes=plt.subplots(2,4,figsize=(16,7),sharey='row')
    for col,label in enumerate(ORDER):
        m=models[label];bs=[]
        for g in SCALES:
            a=m['angles'][str(g)];bs.append({'label':str(g),'q1':a['p25'],'med':a['median'],'q3':a['p75'],'whislo':a['p5'],'whishi':a['p95'],'fliers':[]})
            centers=(np.array(a['hist_edges'][:-1])+np.array(a['hist_edges'][1:]))/2
            axes[1,col].step(centers,100*np.array(a['hist_counts'])/a['count'],where='mid',label='P'+str(g),color=COLORS[SCALES.index(g)])
        axes[0,col].bxp(bs,showfliers=False);axes[0,col].set_title(NAMES[label]);axes[0,col].set_ylim(0,180)
        axes[1,col].set_xlim(0,180);axes[1,col].set_xlabel('Angle to own overwritten whole (degrees)');axes[1,col].legend(fontsize=7)
    axes[0,0].set_ylabel('Angle (degrees); box P25-P75, whiskers P5-P95');axes[1,0].set_ylabel('Objects per 5-degree bin (%)')
    fig.suptitle('02 Same-object part / whole directions | 9,840 paired objects per scale',fontsize=12);save(fig,out,'02_part_whole_angles')

    fig,axes=plt.subplots(2,2,figsize=(12,8));x=np.arange(3)
    for n,label in enumerate(ORDER):
        m=models[label];offset=(n-1.5)*.18
        fields=((0,0,'contrastive_active'),(0,1,'radial_active'))
        for row,col,field in fields:axes[row,col].bar(x+offset,[100*m['intra'][str(g)][field]['mean'] for g in SCALES],.18,label=NAMES[label])
        for col,term in enumerate(('contrastive','radial')):
            axes[1,col].bar(x+offset,[100*m['gradient'][str(g)][term]['increase_fraction'] for g in SCALES],.18,label=NAMES[label])
    for ax,title in zip(axes.flat,('Contrastive hinge active','Radial hinge active','Contrastive: positive angle locally increases','Radial: positive angle locally increases')):
        ax.set_xticks(x,labels=['P200','P400','P600']);ax.set_ylim(0,103);ax.set_ylabel('Output events (%)');ax.set_title(title);ax.grid(axis='y',alpha=.15)
    for ax in axes[1]:ax.set_ylabel('Defined angle-derivative events (%)')
    axes[0,0].legend(fontsize=7);fig.suptitle('03 Intra goals and detached output-coordinate gradients | not optimizer steps',fontsize=12);save(fig,out,'03_intra_gradients')

    fig,axes=plt.subplots(2,2,figsize=(12,9))
    for col,label in enumerate(PH):
        r=models[label]['relations'];cross=np.array(r['ord']['cross_3x3']);perc=100*cross/r['valid_draws']
        axes[0,col].imshow(perc,vmin=0,vmax=100,cmap='Blues')
        for a in range(3):
            for b in range(3):axes[0,col].text(b,a,f'{perc[a,b]:.2f}%',ha='center',va='center',color='white' if perc[a,b]>50 else 'black')
        axes[0,col].set_xticks([0,1,2],labels=['wrong/tied','ordered <.2','>=.2']);axes[0,col].set_yticks([0,1,2],labels=['wrong/tied','ordered <.2','>=.2'])
        axes[0,col].set_xlabel('B shallower than A');axes[0,col].set_ylabel('A shallower than min(i,j)');axes[0,col].set_title(NAMES[label]+' | ORD')
        counts=np.array(models[label]['chain']['per_edge']);base=np.zeros(3)
        for state,color,name in zip(range(3),('#d55e00','#e69f00','#009e73'),('wrong/tied','ordered, short','margin pass')):
            y=100*counts[:,state]/9840;axes[1,col].bar(np.arange(3),y,bottom=base,label=name,color=color);base+=y
        axes[1,col].set_xticks([0,1,2],labels=['200->400','400->600','600->whole']);axes[1,col].set_ylim(0,103);axes[1,col].set_ylabel('Objects (%)');axes[1,col].set_title('CHAIN: own-object edges');axes[1,col].legend(fontsize=7)
    fig.suptitle('04 ORD and CHAIN failures are separate relations',fontsize=12);save(fig,out,'04_order_failures')

    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for col,label in enumerate(PH):
        r=models[label]['relations'];axes[0,col].bar(np.arange(8),100*np.array(r['hier']['failure_patterns'])/r['valid_draws'],color='#0072b2')
        axes[0,col].set_xticks(range(8),labels=['pass','i','j','ij','k','ik','jk','ijk']);axes[0,col].set_ylabel('Valid relations (%)');axes[0,col].set_title(NAMES[label]+' | Failed components')
        with np.load(analysis/'private'/(label+'_relations.npz')) as data:
            valid=data['valid'];same=data['same_ij'];passed=(data['hier_gaps']>=.1).all(1)
            arrays=[data['theta_ij'][valid&(same==s)&(passed==p)] for s,p in ((True,True),(True,False),(False,True),(False,False))]
        axes[1,col].boxplot(arrays,labels=['same/pass','same/fail','cross/pass','cross/fail'],showfliers=False,whis=(5,95))
        axes[1,col].set_ylabel('Actual i/j angle (degrees)');axes[1,col].tick_params(axis='x',labelsize=8)
    fig.suptitle('05 HIER failures | all three whole/part comparisons, no proxy/proxy loss',fontsize=12);save(fig,out,'05_hier_failures')


def examples(models):
    lines=['# 代表反例','', '以下为固定重放中的确定性首个对应类型实例，不代表总体抽样。HIER候选可行性只检查这6例，不修改原A/B或通过率。','']
    for label in PH:
        for case in models[label]['cases']:
            lines.extend(['## '+ZH[label]+' / '+case['kind'],'',f"对象ID(i,j,k)：{case['object_ids']}；类别：{case['classes']}；batch={case['panel']}。",
                '', '| 节点 | 深度 | 尺度/来源 |','|---|---:|---|'])
            for name,h in zip(('i','j','k'),case['whole_depths']):lines.append(f'| {name} | {num(h,4)} | whole |')
            for name in ('A','B'):
                n=case[name];lines.append(f"| {name} | {num(n['depth'],4)} | p{n['scale']} / {n['source']} / ID{n['object_id']} |")
            lines+=['','| whole | 到A | 到B |','|---|---:|---:|']
            for ix,name in enumerate(('i','j','k')):lines.append(f"| {name} | {num(case['A']['whole_distances'][ix],4)} | {num(case['B']['whole_distances'][ix],4)} |")
            lines+=['',f"ord两gap：{[round(x,5) for x in case['ord_gaps']]}；HIER三gap：{[round(x,5) for x in case['hier_gaps']]}。",
                    f"whole夹角ij/ik/jk：{[round(case['whole_angles'][k],2) for k in ('ij','ik','jk')]}°。"]
            if 'hier_only_candidate_feasibility' in case:
                f=case['hier_only_candidate_feasibility'];lines+=['',f"同池HIER可行A≠B对数：{f['feasible_pairs']}。"]
                if f['exists']:lines.append(f"一组可行替代：A=p{f['A']['scale']}({f['A']['source']})，B=p{f['B']['scale']}({f['B']['source']})；三gap={[round(x,5) for x in f['hier_gaps']]}。这组替代未要求ord。")
            lines+=['']
    lines+=['## chain代表对象（PartHier e300）','']
    for c in models['TS18_e300']['chain_examples']:
        lines.extend([f"- {c['kind']}：ID{c['object_id']}，类{c['class']}；四深度={[round(x,5) for x in c['depths_200_400_600_whole']]}；三gap={[round(x,5) for x in c['chain_gaps']]}；三角度={[round(x,2) for x in c['part_whole_angles']]}°。"])
    lines+=['','## 固定对象：三个part对自身whole夹角','', '| 权重 | ID | 类 | p200 | p400 | p600 |','|---|---:|---:|---:|---:|---:|']
    for label in ORDER:
        for c in models[label]['object_examples']:lines.append('| '+ZH[label]+' | '+str(c['object_id'])+' | '+str(c['class'])+' | '+' | '.join(num(x)+'°' for x in c['part_whole_angles'])+' |')
    return '\n'.join(lines)+'\n'


def report(models):
    lines=['# PartHier后续五问审查','', '全部统计与绘图在8000完成，复用原4份冻结缓存与配套Q2048，没有编码前向、网络反传、参数更新或新训练。原200批CB64/K20/T50/seed4780，先核对94读出再解释新结果。','',
      '## 1. A为什么偏浅p200，热点从哪里来？','',
      'A评分为max(d(i,q),d(j,q))，不是只靠近一个whole。下表统一到Q-p200内部比较；候选数按条目计、选择按事件计，百分位是在本batch同来源/尺度中计算。','',
      '| 权重 | Q-p200候选深度中位 | 选中A深度中位 | 选中A的组内深度百分位中位 | p200预期选择占比 | 候选统一深度后 |','|---|---:|---:|---:|---:|---:|']
    for label in PH:
        r=models[label]['relations'];g=next(x for x in r['selection']['groups'] if x['role']=='A' and x['source']=='queue' and x['scale']==200)
        ga=[x for x in r['selection']['groups'] if x['role']=='A' and x['scale']==200]
        lines.append(f"| {ZH[label]} | {num(g['candidate_depth']['median'])} | {num(g['selected_depth']['median'])} | {num(g['selected_within_group_percentile']['median'])}% | {pct(sum(x['expected_all'] for x in ga)/r['draws'])} | {pct(sum(x['equal_depth_expected_all'] for x in ga)/r['draws'])} |")
    lines+=['', '统一深度只改变候选副本的半径，方向和原ijk不变；它检查当前评分的径向敏感性，不代表新训练的因果结果。队列和当前eval的裁剪/增强/BN/历史参数条件不同，不能拿全9840 p200中位数直接解释队列选择。',
      '', '热点同时给真实次数、softmax预期与曝光率，见[热点表](tables/hotspots.csv)。Top1是一条最热条目，Top5%是曝光条目中最热5%的合计；对象/尺度合并不代表同一裁剪几何part。概率预期只与全部抽签比较。',
      '', '![选择机制](figures/01_selection.png)','', '## 2. 同一对象的part与whole夹角','',
      '| 权重 | p200中位 | p400中位 | p600中位 |','|---|---:|---:|---:|']
    for label in ORDER:lines.append('| '+ZH[label]+' | '+' | '.join(num(models[label]['angles'][str(g)]['median'])+'°' for g in SCALES)+' |')
    lines+=['', '每尺度各9840个配对对象，whole为训练配对的覆写whole。箱体P25–P75、须P5–P95；下排保留角度分布，近零方向单列。固定6个对象的逐尺度数据见[实例表](examples.md)。',
      '', '![同对象夹角](figures/02_part_whole_angles.png)','', '## 3. 统一4与角度自由度','',
      '**4约束正负双曲距离差；1000/N约束原点深度差。转角不能直接满足后者。** 下表是PartHier e300的输出空间局部梯度：','',
      '| 尺度 | 对比项活跃 | 径向项活跃 | 对比项局部使正角度增大 | 径向项局部使正角度增大 |','|---|---:|---:|---:|---:|']
    m=models['TS18_e300']
    for g in SCALES:
        x=m['intra'][str(g)];gr=m['gradient'][str(g)]
        lines.append(f"| {g} | {pct(x['contrastive_active']['mean'])} | {pct(x['radial_active']['mean'])} | {pct(gr['contrastive']['increase_fraction'])} | {pct(gr['radial']['increase_fraction'])} |")
    lines+=['', '保留同一个part兼任flip负例的梯度贡献，同时考虑whole/part双方移动；每尺度权重0.01。完全同向/反向时角度导数无定义，单列数量；增大/减小比例以导数有定义事件为分母。探针只对复制出的输出叶子求导，不连接主干、不执行优化器；这是局部驱动，不是实际训练一步或margin消融的效果。径向切向分量验收通过；完整正负角度、距离与梯度见[表](tables/gradients.csv)。',
      '', '![intra和梯度](figures/03_intra_gradients.png)','', '## 4. ord与chain的反例不同','',
      'ord分别检查min(hi,hj)−hA与hA−hB；chain检查同对象三条相邻边。状态分错序/并列、方向正确但间隔不足、达到设计间隔。','',
      '| 权重 | ord第一项不达0.2 | ord第二项不达0.2 | ord联合失败 | chain联合失败 |','|---|---:|---:|---:|---:|']
    for label in PH:
        m=models[label];r=m['relations'];n=r['valid_draws']
        lines.append(f"| {ZH[label]} | {pct(1-r['ord']['per_edge'][0][2]/n)} | {pct(1-r['ord']['per_edge'][1][2]/n)} | {pct(1-r['ord']['all_margin_pass']/n)} | {pct(1-m['chain']['all_margin_pass']/9840)} |")
    lines+=['', '**p600≈whole直接对应chain末边；不能把它当作ord失败的定义。** ord按A/B尺度与来源的失败贡献另列[分组表](tables/ord_groups.csv)，三种ord与chain代表反例见[实例](examples.md)。',
      '', '![ord和chain反例](figures/04_order_failures.png)','', '## 5. HIER失败的ijk长什么样？','',
      '只检查i/j更近A、k更近B三项whole–part距离；没有代理–代理距离损失。图中i/j/k组合表示对应哪些项未达到0.1，pass为全通过。','',
      '| 权重 | 基本方向联合通过 | 0.1联合通过 | 原抽签数 | 碰撞数 |','|---|---:|---:|---:|---:|']
    for label in PH:
        r=models[label]['relations'];lines.append(f"| {ZH[label]} | {pct(r['hier']['all_basic_pass']/r['valid_draws'])} | {pct(r['hier']['all_margin_pass']/r['valid_draws'])} | {r['draws']} | {r['collisions']} |")
    cards=[c for l in PH for c in models[l]['cases'] if c['kind'].startswith('hier')]
    feasible=sum(c['hier_only_candidate_feasibility']['exists'] for c in cards)
    lines += ['',f'展示的{len(cards)}个HIER反例中，{feasible}个在同一2240候选池里存在另一组可行A≠B；只检查这些例子，不外推总体，也不修改原选择或通过率。完整五深度/六距离/替代说明见[反例卡](examples.md)。',
       '', '通过/失败组按同类/异类分开，三whole深度、三对夹角与距离、A/B深度及j是否实际比k更近i见[几何表](tables/geometry_groups.csv)。',
       '', '![HIER失败分布](figures/05_hier_failures.png)','',
       '原Q包含历史train-mode特征，本次是保存状态审查；原版并非匹配训练消融。结果解释当前几何和选择机制，不单独证明训练因果或形态树。输入SHA、冻结重放及无更新验收见[summary.json](summary.json)。']
    return '\n'.join(lines)+'\n'


def main():
    p=argparse.ArgumentParser();p.add_argument('--analysis-dir',type=Path,required=True);p.add_argument('--out-dir',type=Path,required=True);args=p.parse_args()
    summary=json.loads((args.analysis_dir/'summary.json').read_text(encoding='utf-8'));models=summary['models']
    assert summary['acceptance']['all_inputs_unchanged'] and summary['acceptance']['replay_gates_passed']
    args.out_dir.mkdir(exist_ok=False);figs=args.out_dir/'figures';tables=args.out_dir/'tables';figs.mkdir();tables.mkdir()
    export_tables(models,tables);plots(models,args.analysis_dir,figs)
    (args.out_dir/'report.md').write_text(report(models),encoding='utf-8')
    (args.out_dir/'examples.md').write_text(examples(models),encoding='utf-8')
    (args.out_dir/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    assert len(list(figs.glob('*.png')))==5 and len(list(figs.glob('*.svg')))==5
    acceptance={'status':'completed','figure_groups':5,'new_training':False,'server':'8000',
       'files_sha256':{str(f.relative_to(args.out_dir)):hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(args.out_dir.rglob('*')) if f.is_file()}}
    (args.out_dir/'acceptance.json').write_text(json.dumps(acceptance,indent=2),encoding='utf-8')
    print('PUBLICATION_COMPLETE',flush=True)


if __name__=='__main__':main()
