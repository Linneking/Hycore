"""Render frozen radial diagnosis, using only the generated scalar/1D arrays."""
from pathlib import Path
import argparse
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager


def plot(folder):
    folder = Path(folder)
    j = json.loads((folder/'summary.json').read_text(encoding='utf-8'))
    font = Path('C:/Windows/Fonts/msyh.ttc')
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams['font.family'] = font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams.update({'font.size':11,'axes.unicode_minus':False,'svg.fonttype':'path'})
    with np.load(folder/'radial_diagnostic_arrays.npz',allow_pickle=False) as z:
        pd,wd,mask=z['proxy_depth'],z['whole_depth'],z['selected_mask']
    fig,axes=plt.subplots(1,2,figsize=(14,5.7))
    for values,label,color in [(wd,'whole：8856 个干净输入','#2474b5'),
                              (pd[mask],'第 5 轮样本项选中：265 个','#32a252'),
                              (pd[~mask],'第 5 轮样本项未选：247 个','#d75048')]:
        x=np.sort(values);axes[0].plot(x,np.arange(1,len(x)+1)/len(x),label=label,color=color,lw=2)
    axes[0].set(xlabel='原点双曲深度（c=1）',ylabel='累积分布',title='同一检查点的径向分布')
    axes[0].legend(loc='upper left',fontsize=10);axes[0].grid(alpha=.2)
    names=['saved','unused_shrunk_to_selected_median','all_at_selected_median','selected_expanded_to_unused_median']
    labels=['保存位置','仅把未选组\n移到浅层中位深度','全部代理\n移到浅层中位深度','仅把选中组\n移到深层中位深度']
    x=np.arange(len(names));width=.34
    for role,offset,color in [('pair',-width/2,'#2474b5'),('triple',width/2,'#32a252')]:
        values=np.array([j['fixed_query_radial_controls'][name][role]['expected_not_selected_group_probability_given_noncollision_mean']*100 for name in names])
        axes[1].bar(x+offset,values,width,label=role,color=color)
        for xx,v in zip(x+offset,values):
            axes[1].text(xx,max(v,0)+1.2,'≈0%' if v<.00001 else f'{v:.1f}%',ha='center',fontsize=9)
    axes[1].set_xticks(x,labels,fontsize=9);axes[1].set_ylim(0,110)
    axes[1].set(ylabel='原未选组的理论选择概率（%）',title='固定方向和查询，仅改变代理半径')
    axes[1].legend();axes[1].grid(axis='y',alpha=.2)
    fig.suptitle('V7 第 5 轮：深层代理的祖先选择竞争',fontsize=16)
    fig.text(.5,.005,'冻结 clean/eval 64 个对象，512 个相关查询；非碰撞条件概率；无训练更新。',ha='center',fontsize=10)
    fig.tight_layout(rect=[0,.035,1,.94])
    for ext in ['png','svg']:fig.savefig(folder/('e5_radial_selection_controls.'+ext),dpi=160)
    plt.close(fig)
    t=j['trajectory_by_fixed_e5_group']
    fig,ax=plt.subplots(figsize=(11,4.8))
    for key,label,color in [('whole_clean','whole 干净输入中位深度','#2474b5'),
                            ('fixed_e5_selected','固定 e5 选中组的中位深度','#32a252'),
                            ('fixed_e5_not_selected','固定 e5 未选组的中位深度','#d75048')]:
        ax.plot([r['epoch'] for r in t],[r[key]['median'] for r in t],'-o',label=label,color=color,lw=2,ms=4)
    ax.axhline(6,ls=':',color='gray',label='V7 参数约束：代理最大深度约 6')
    ax.set(xlabel='训练轮数',ylabel='原点双曲深度（c=1）',title='第 1 轮已发生分化；第 5 轮的固定分组沿训练追踪')
    ax.legend(loc='lower right',fontsize=10);ax.grid(alpha=.2)
    fig.text(.5,.005,'固定组内 62 个未选代理在后期重新成为样本祖先；185 个直到 e300 未回归。',ha='center',fontsize=10)
    fig.tight_layout(rect=[0,.04,1,1])
    for ext in ['png','svg']:fig.savefig(folder/('fixed_e5_groups_trajectory.'+ext),dpi=160)
    plt.close(fig)
    print('Two scientific figures generated in PNG and SVG')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('folder');plot(p.parse_args().folder)
