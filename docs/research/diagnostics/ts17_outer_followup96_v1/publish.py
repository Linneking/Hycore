"""CPU publication of the approved three-question frozen audit."""
from __future__ import annotations
import argparse,json,csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from science import *
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
COLORS=['#d55e00','#0072b2','#999999','#009e73']
NAMES=['Outer, unused in96 (162)','Inner, used in96 (290)','Inner, unused in96 (59)','Outer, used in96 (1)']


def save(fig,out,name):
    fig.tight_layout();fig.savefig(out/(name+'.png'),dpi=180,bbox_inches='tight');fig.savefig(out/(name+'.svg'),bbox_inches='tight');plt.close(fig)


def table(head,rows):return '\n'.join(['| '+' | '.join(head)+' |','| '+' | '.join(['---']*len(head))+' |']+['| '+' | '.join(map(str,row))+' |' for row in rows])
def f(x,n=3):return '—' if x is None else f'{x:.{n}f}'
def p(x):return '—' if x is None else f'{x*100:.3f}%'


def main():
    a=argparse.ArgumentParser();a.add_argument('--root',type=Path,required=True);args=a.parse_args();root=args.root;out=root/'public';figures=out/'figures';figures.mkdir()
    s=json.loads((out/'summary.json').read_text());z=dict(np.load(root/'private/readout.npz'))
    fig,axs=plt.subplots(1,2,figsize=(12,4.3))
    axs[0].boxplot([z['thetaI'].max(1),z['thetaO'].max(1)],labels=['Best inner I','Best outer O'],showfliers=False);axs[0].set(ylabel='Max angle to the same i/j (degrees)',title='Fixed same-class pairs; actual candidate angles')
    axs[1].scatter(z['thetaO'].max(1)-z['thetaI'].max(1),z['gap'],s=4,alpha=.25);axs[1].axhline(0,color='black',ls='--');axs[1].axvline(0,color='gray',ls='--');axs[1].set(xlabel='Outer minus inner max angle (degrees)',ylabel='Outer minus inner minimax distance',title='Positive vertical gap: outer loses')
    save(fig,figures,'01_pair_competition')
    fig,axs=plt.subplots(1,2,figsize=(12,4.4));values=[z[k] for k in ('gap','gap_depth','gap_angle')]
    axs[0].boxplot(values,labels=['Actual O','O: depth of I','O: direction of I'],showfliers=False);axs[0].axhline(0,color='black',ls='--');axs[0].set(ylabel='Candidate cost minus original I cost',title='Same original I/O; no outer-pool re-search')
    axs[1].bar(['Actual O','Depth only','Direction only'],[100*x['class_equal_rescue_fraction'] for x in s['controls']],color=['#888888',COLORS[0],COLORS[1]]);axs[1].set(ylabel='O beats I among original losses (%)',title='Equal-class rescue rate; tolerance 1e-5')
    save(fig,figures,'02_direction_depth_replacements')
    fig,axs=plt.subplots(1,2,figsize=(13,4.7));hist=z['history']
    for ri,role in enumerate(('pair','triple')):
        ax=axs[ri]
        for gi in (0,1):
            m=z['group']==gi
            for ci,component in enumerate(('Sample','Proxy-internal')):
                total=hist[:,ci,0,ri].sum(1);mass=hist[:,ci,0,ri][:,m].sum(1);value=mass/total/m.sum()*512
                ax.plot(np.arange(1,274),value,color=COLORS[gi],ls='-' if ci==0 else '--',lw=1.2,label=NAMES[gi]+' / '+component)
        ax.axhline(1,color='gray',lw=.7);ax.set(xlabel='Actual training epoch',ylabel='Usage per proxy / uniform512 baseline',title='Actual hard-Gumbel '+role+' selections',xticks=[1,5,10,100,200,273]);ax.legend(fontsize=6)
    save(fig,figures,'03_actual_history')
    fig,axs=plt.subplots(2,2,figsize=(12,8.8));groups=z['group'];early=hist[:5,0,0,0].sum(0)
    axs[0,0].boxplot([z['theta0'][groups==g] for g in (0,1,2)],labels=['Outer unused','Inner used','Inner unused'],showfliers=False);axs[0,0].set(ylabel='Initial angle to nearest e0 class direction (deg)',title='Clean/eval e0 probe; not learned semantics')
    for gi in range(4):
        m=groups==gi;axs[0,1].scatter(z['theta0'][m],early[m]+1,s=10,alpha=.55,color=COLORS[gi],label=NAMES[gi])
    axs[0,1].set(xlabel='Initial nearest-direction angle (degrees)',ylabel='First5 sample pair selections + 1',yscale='log',title='Initial angle versus subsequent actual usage');axs[0,1].legend(fontsize=6)
    axs[1,0].boxplot([z['rotation_best'][groups==g] for g in (0,1,2)],labels=['Outer unused','Inner used','Inner unused'],showfliers=False);axs[1,0].set(ylabel='e0 to best proxy direction change (degrees)',title='Same saved proxy IDs')
    for off,kind,color in ((-.16,'near_initial_direction',COLORS[1]),(.16,'far_initial_direction',COLORS[0])):
        rows=[x for x in s['strata'] if x['direction_group']==kind]
        axs[1,1].bar(np.arange(4)+off,[x['first5_sample_pair_per_proxy_vs_uniform'] for x in rows],.32,color=color,label=kind)
    axs[1,1].set_xticks(np.arange(4),labels=['Q1','Q2','Q3','Q4']);axs[1,1].set(xlabel='Matched initial-depth quartile',ylabel='First5 sample-pair usage / uniform512',title='Near/far halves within each depth stratum');axs[1,1].legend(fontsize=7)
    save(fig,figures,'04_initial_direction')
    g=s['groups'];ctrl=s['controls'];lines=['# TS17外层代理落选原因：冻结审查结果','',
        f"仅TS17/B12 best e273；同类i/j按无序对象对去重，每类最多100对，实际{s['classes_present']}类、{s['pairs']}对。所有whole用于距离选择前统一c0.1深度4.6，代理保留实际深度；内外段固定沿用96分界。",'',
        '## 1. 同一对i/j，外层为什么输','',
        table(['指标','结果'],[['内层候选较大夹角中位（度）',f(s['inner_max_angle']['median'])],['外层候选较大夹角中位（度）',f(s['outer_max_angle']['median'])],['外层落后距离中位',f(s['pair_gap']['median'])],['外层胜出率（类别等权）',p(s['class_equal_outer_win_fraction'])],['全部外层Gumbel概率质量均值（类别等权）',p(s['class_equal_outer_probability'])]]),'',
        '夹角是代理与这对i/j的实际方向夹角，取较大值；选择评分仍是两个原双曲距离的较大值。最近类别中心未被拿来代替这对样本。','',
        '![真实候选比较](figures/01_pair_competition.png)','',
        '## 2. 只换深度，或只换方向','',
        table(['状态','相对I的评分差中位','评分改善中位','原最佳O对I翻盘率'],[[x['state'],f(x['gap']['median']),f(x['improvement']['median']),p(x['class_equal_rescue_fraction'])] for x in ctrl]),'',
        '真实O、只换成I的深度（保O方向）、只换成I的方向（保O深度），始终比较同一对i/j和固定I/O。翻盘只针对原外层落败关系，阈值1e-5、各类等权；没有重新搜索外层池。方向和深度相互影响，两个改善量不应相加成百分之百归因。','',
        '![两次只改一项](figures/02_direction_depth_replacements.png)','',
        '## 3. 历史真的一直冷落吗','',
        table(['best分组','代理数','前5轮样本pair份额','e273样本pair份额','前5轮单代理相对均匀机会','前5轮内部pair份额'],[[x['group'],x['proxies'],p(x['first5_sample_pair_share']),p(x['epoch273_sample_pair_share']),f(x['first5_sample_pair_per_proxy_vs_uniform']),p(x['first5_internal_pair_share'])] for x in g]),'',
        '完整读取e1–273逐轮实际hard-Gumbel计数。主图按组内单个代理归一，1为全部512平均机会；橙色为外段未入选162个，蓝色为内段入选290个，虚线为代理内部项。其他60代理仍在明细和汇总表中。选择次数不等于非零损失次数；全部、非碰撞、有效非碰撞分别保留。','',
        '![真实训练使用历史](figures/03_actual_history.png)','',
        '## 4. 初始方向是否与后续机会有关','',
        table(['best分组','初始最近方向角中位','初始深度中位','转动到e5中位','转动到best中位','best深度−初始深度中位'],[[x['group'],f(x['initial_angle']['median']),f(x['initial_depth']['median']),f(x['rotation_e5']['median']),f(x['rotation_best']['median']),f(x['depth_change']['median'])] for x in g]),'',
        f"全部512代理，e0最近方向角与前5轮实际样本pair使用量的Spearman相关为 **{f(s['initial_angle_early_pair_spearman'])}**。初始深度四组内分别为："+'、'.join(f(x['spearman']) for x in s['depth_stratum_correlations'])+'。这只是同一次训练的关联，不是初始化唯一成因的证明。','',
        'e0、e5各对同一800个对象（每类20个）做冻结clean/eval前向，best复用原缓存同ID。e0方向尚无已学成类别语义；初始代理到e5类别方向的夹角仅作回顾对应，不能拿来独立预测前5轮结果。实际初始深度不同，已分层控制。','',
        '![初始化方向及位移](figures/04_initial_direction.png)','',
        table(['初始深度组','方向较近／较远','代理数','前5轮单代理机会／均匀基线'],[[x['initial_depth_quartile'],x['direction_group'],x['proxies'],f(x['first5_sample_pair_per_proxy_vs_uniform'])] for x in s['strata']]),'',
        '## 5. 六对具体选择账单','',
        '按外层落后分数小／中／大三档各取两例，层内优先不同类，再按固定分位和ID；未按外观筛选。I/O都是该对关系原始最优候选。','']
    for idx,e in enumerate(s['examples']):
        lines += [f"### 例{idx+1}：{e['class_name']}",'',table(['候选','代理ID','深度','与i夹角','与j夹角','到i距离','到j距离','选择评分'],[['I',e['inner_id'],f(e['inner_depth']),f(e['inner_angle_i']),f(e['inner_angle_j']),f(e['inner_distance_i']),f(e['inner_distance_j']),f(e['inner_cost'])],['O',e['outer_id'],f(e['outer_depth']),f(e['outer_angle_i']),f(e['outer_angle_j']),f(e['outer_distance_i']),f(e['outer_distance_j']),f(e['outer_cost'])]]),'',f"原评分差{f(e['gap'])}；只换深度后{f(e['depth_only_gap'])}；只换方向后{f(e['direction_only_gap'])}。",'']
    lines += ['## 6. 验收与文件','',f"原e5权重SHA与训练第5轮保存记录相同，e0代理逐位匹配独立seed4780初始化。best32前向回归通过，最大坐标误差{s['best_gate']['max_abs']:.3g}。e0/e5/best回归共前向{s['new_forward_objects']}个对象，无网络反传、优化器、BN更新或test读出；源权重和输入前后SHA不变。",'',
        '[512代理初始化／历史明细](proxy_initial_history.csv)、[初始深度控制](initial_depth_control.csv)、[40类候选比较](class_competition.csv)、[六对账单](representative_pair_accounts.csv)、[完整分组历史](history_groups.csv)、[完整统计](summary.json)、[源身份](source_hashes.json)。逐对对象ID、特征、逐代理逐轮计数留在服务器私有目录。','',
        '冻结换位回答当前几何原因；历史及初始化关联回答可能的路径依赖。两者都不单独构成随机初始化干预的因果实验。没有新增训练。','']
    (out/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    write(out/'artifact_hashes.json',{str(x.relative_to(out)):sha(x) for x in out.rglob('*') if x.is_file() and x.name!='artifact_hashes.json'})
    print(json.dumps(dict(status='completed',figures=4,public_files=len(list(out.rglob('*'))))))


if __name__=='__main__':main()
