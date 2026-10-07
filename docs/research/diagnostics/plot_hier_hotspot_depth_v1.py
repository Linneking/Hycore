"""Scientific figure: which proxy depths share each leading top4 quartet."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt,font_manager


def plot(root):
    root=Path(root);j=json.loads((root/'summary.json').read_text(encoding='utf-8'))
    font=Path('C:/Windows/Fonts/msyh.ttc')
    if font.exists():
        font_manager.fontManager.addfont(str(font));plt.rcParams['font.family']=font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams.update({'font.size':11,'axes.unicode_minus':False,'svg.fonttype':'path'})
    with np.load(root/'plot_arrays.npz',allow_pickle=False) as z:
        ids,depth=z['proxy_ids'],z['proxy_depth']
    fig,axes=plt.subplots(1,2,figsize=(14,6.4),sharex=True,sharey=True)
    colors=['#c64646','#286eac']
    for ax,name,title in zip(axes,['raw','whole_equal_median'],['whole 保留原始深度','只把 whole 统一深度；代理位置不变']):
        r=j['controls'][name]['retrieval'];leaders=r['leading_quartets'][:2]
        members=set(i for g in leaders for i in g['proxy_ids'])
        other=np.array([i for i in ids if i not in members])
        ax.scatter(depth[other],other,s=11,c='#aaa',alpha=.5,label='其余四例组合')
        for q,color in zip(leaders,colors):
            p=np.array(q['proxy_ids']);label='/'.join(map(str,q['sample_ids']))
            ax.scatter(depth[p],p,s=16,c=color,alpha=.7,label=f"{label}（{len(p)} 个代理）")
        ax.axvline(3,color='gray',ls=':',lw=.8);ax.axvline(5,color='gray',ls=':',lw=.8)
        ax.set_title(f"{title}\n覆盖 {r['covered_sample_count']} 个样本；最大重复 {r['max_repeated_quartet']}/512",fontsize=13)
        ax.set_xlabel('代理原点双曲深度（c=1）');ax.grid(alpha=.12)
        ax.legend(loc='upper center',bbox_to_anchor=(.5,-.15),fontsize=9)
        ax.set_xlim(1.95,6.15)
    axes[0].set_ylabel('代理 ID（512 个固定参数行）')
    for p,offset in [(264,(40,10)),(81,(-140,8))]:
        axes[0].annotate(f"代理 {p}：{depth[p]:.3f}",xy=(depth[p],p),xytext=offset,
                        textcoords='offset points',arrowprops={'arrowstyle':'->','color':'black'},fontsize=10)
    fig.suptitle('V7 e300：同一热点四例可以横跨浅、深代理',fontsize=17)
    fig.text(.5,.006,'所有比较均为 8856 个固定训练对象、512 个代理的无标签最近四例；没有训练更新。',ha='center',fontsize=10)
    fig.tight_layout(rect=[0,.04,1,.94]);fig.subplots_adjust(bottom=.23)
    for ext in ['png','svg']:fig.savefig(root/('proxy_hotspots_by_depth.'+ext),dpi=160)
    plt.close(fig)
    print('Hotspot-depth scientific figure generated')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root');plot(p.parse_args().root)
