"""Build a read-only, aggregate V8/V7/original-HIER scientific review.

Inputs are completed hier_postrun_audit artifacts and the original-HIER
geometry audit. No torch, encoder forward, optimizer or checkpoint mutation.
Usage: python v8_system_review_v1.py --work WORK --references VIS --out OUT
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import html
import json
import shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

COLORS=['#156b9c','#ca5935','#25876c','#8a5fa3','#b18827','#6a7680']
NAMES='airplane bathtub bed bench bookshelf bottle bowl car chair cone cup curtain desk door dresser flower_pot glass_box guitar keyboard lamp laptop mantel monitor night_stand person piano plant radio range_hood sink sofa stairs stool table tent toilet tv_stand vase wardrobe xbox'.split()
def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def dump(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def short(s):return s.replace('V7-HIER64-K20-W0','V7-HIER64').replace('V8-HIER64-K20-W0','V8-HIER64')
def fmt(x,n=3):return '—' if x is None or not np.isfinite(x) else f'{x:.{n}f}'
def ecdf(ax,x,label,color,ls='-'):
 x=np.sort(np.asarray(x));ax.plot(x,np.arange(1,len(x)+1)/len(x),label=label,color=color,ls=ls,lw=2)
def stats(x):
 x=np.asarray(x);return dict(zip(['min','p05','p25','median','p75','p95','max'],map(float,np.quantile(x,[0,.05,.25,.5,.75,.95,1]))))|dict(mean=float(np.mean(x)),std=float(np.std(x)),variance=float(np.var(x)))
def export_csv(p,rows):
 if not rows:return
 with Path(p).open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
class Report:
 def __init__(self,out):self.out=out;self.md=[];self.htm=[];self.links=[]
 def title(self,t):self.md+=['# '+t,''];self.htm+=['<h1>'+html.escape(t)+'</h1>']
 def h(self,t):self.md+=['## '+t,''];self.htm+=['<h2>'+html.escape(t)+'</h2>']
 def p(self,t):self.md +=[t,''];self.htm+=['<p>'+html.escape(t)+'</p>']
 def math(self,t):self.md+=['$$',t,'$$',''];self.htm+=['<pre class="equation">'+html.escape(t)+'</pre>']
 def table(self,headers,rows):
  self.md+=['| '+' | '.join(map(str,headers))+' |','| '+' | '.join(['---']*len(headers))+' |']
  self.md+=['| '+' | '.join(map(str,r))+' |' for r in rows];self.md+=['']
  self.htm+=['<div class="table"><table><thead><tr>'+''.join('<th>'+html.escape(str(x))+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in r)+'</tr>' for r in rows)+'</tbody></table></div>']
 def fig(self,name,caption):
  p='figures/'+name+'.png';self.links.append(p);self.md+=['!['+caption+']('+p+')','',caption,''];self.htm+=['<figure><img loading="lazy" src="'+p+'" alt="'+html.escape(caption)+'"><figcaption>'+html.escape(caption)+'</figcaption></figure>']
 def link(self,path,label):self.links.append(path);self.md+=['['+label+']('+path+')',''];self.htm+=['<p><a href="'+path+'">'+html.escape(label)+'</a></p>']
 def finish(self):
  (self.out/'report.md').write_text('\n'.join(self.md),encoding='utf-8')
  css='body{font:17px/1.8 "Microsoft YaHei",sans-serif;color:#20323d;background:#f4f6f8;margin:0}main{max-width:1200px;padding:40px 48px;margin:auto;background:white}h1{font-size:32px}h2{margin-top:50px;border-bottom:2px solid #156b9c;padding-bottom:8px}p{max-width:1080px}figure{margin:30px 0}img{width:100%;height:auto}figcaption{font-size:15px;color:#4b5f6a}.table{overflow:auto}table{border-collapse:collapse;font-size:14px;width:100%}td,th{border:1px solid #d5dfe5;padding:8px;text-align:right}td:first-child,th:first-child{text-align:left}th{background:#eef3f7}.equation{white-space:pre-wrap;background:#f4f6f8;padding:18px}a{color:#156b9c} @media print{main{padding:0}h2{break-after:avoid}figure{break-inside:avoid}}'
  (self.out/'report.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>V8系统审查</title><style>'+css+'</style><main>'+''.join(self.htm)+'</main></html>',encoding='utf-8')
  missing=[x for x in self.links if not (self.out/x).exists()]
  assert not missing,missing
  return len(self.links)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--work',type=Path,required=True);ap.add_argument('--references',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
 W,V,O=args.work,args.references,args.out;O.mkdir(parents=True,exist_ok=True);(O/'figures').mkdir(exist_ok=True);(O/'tables').mkdir(exist_ok=True)
 for f in ['C:/Windows/Fonts/msyh.ttc','/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc']:
  if Path(f).exists():font_manager.fontManager.addfont(f);plt.rcParams['font.family']=[font_manager.FontProperties(fname=f).get_name(),'DejaVu Sans'];break
 plt.rcParams.update({'axes.unicode_minus':False,'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':120,'savefig.dpi':170,'svg.fonttype':'none','svg.hashsalt':'v8-reviewed-20261016'})
 S=W/'v8_audit_system_20261016_r1';G=V/'original_hier_full_geometry_20261008/analysis';D=W/'v8_direction_20261016_r1'
 sources=[S/'normalized_runs.json',S/'structure/structure_summary.json',S/'mechanisms/mechanism_summary.json',S/'system_manifest.json',D/'evolution.json']
 before={str(p):sha(p) for p in sources}
 runs=read(sources[0]);structure=read(sources[1]);mech=read(sources[2]);assert structure['status']==mech['status']=='completed'
 evo=read(D/'evolution.json');v7evo=read(V/'hier_v7_direction_hotspots_20261008/evolution_r2/evolution.json')
 dr=[read(p/'summary.json') for p in sorted(D.glob('e[0-9][0-9][0-9]'))];epochs=[x['epoch'] for x in dr]
 geometry=[]
 for name,label in [('HIER_cub_train','原版 CUB best'),('HIER_cars_train','原版 Cars best'),('HIER_sop_train','原版 SOP best'),('V7_e187','V7 best e187'),('V7_e300','V7 last e300')]:
  p=G/name;geometry.append(dict(name=name,label=label,summary=read(p/'summary.json'),arrays=np.load(p/'plot_arrays.npz')))
 for name,label in [('V8_e005','V8 e5'),('V8_e221','V8 best e221'),('V8_e300','V8 last e300')]:
  p=W/'geometry_v8'/name;geometry.append(dict(name=name,label=label,summary=read(p/'summary.json'),arrays=np.load(p/'plot_arrays.npz')))
 by={x['name']:x for x in geometry};v8=by['V8_e300'];vg=v8['summary'];va=v8['arrays'];v7=by['V7_e300'];g7=v7['summary'];a7=v7['arrays']
 randoms=read(V/'original_hier_full_geometry_20261008/random_directions/summary.json')['models']+read(W/'random_v8/summary.json')['models'];rnd={x['name']:x for x in randoms}
 def save(fig,name):
  fig.savefig(O/'figures'/f'{name}.png',bbox_inches='tight');svg=O/'figures'/f'{name}.svg';fig.savefig(svg,bbox_inches='tight',metadata={'Date':None});plt.close(fig)
  svg.write_text('\n'.join(line.rstrip() for line in svg.read_text(encoding='utf-8').splitlines())+'\n',encoding='utf-8')
 def use(run,domain='noncollision',role='combined',component='sample'):
  return [x for x in run['proxy_usage'] if x['domain']==domain and x['role']==role and x['component']==component]
 def metric(run,key):return [x['metrics'].get(key,np.nan) for x in run['epochs']]
 # Aggregates exclude raw identities, source paths and per-object arrays.
 radial=[]
 for x in geometry:
  for role in ['whole','proxy']:
   radial.append(dict(model=x['label'],population=role,**stats(x['arrays'][role+'_d0'])))
 export_csv(O/'tables/radial_statistics.csv',radial)
 cover=[]
 for x in geometry:
  s=x['summary'];r=rnd.get(x['name']);trials=r['trials'] if r else []
  cover.append(dict(model=x['label'],N=s['identity']['N'],D=s['identity']['D'],c=s['identity']['c'],P=s['identity']['P'],raw=s['retrieval']['raw']['different_objects_top4'],equal_whole=s['retrieval']['direction']['different_objects_top4'],raw_max_quartet=s['retrieval']['raw']['maximum_repeated_quartet'],raw_hot4_share=s['retrieval']['raw']['top4_hot_object_slot_share'],bottom_1pct_share=s['retrieval']['raw']['bottom_depth_1pct_slot_share'],proxy_R=s['proxy_direction']['R'],random_raw_min=min(t['retrieval']['raw']['different_objects_top4'] for t in trials) if trials else None,random_raw_max=max(t['retrieval']['raw']['different_objects_top4'] for t in trials) if trials else None,random_equal_min=min(t['retrieval']['direction']['different_objects_top4'] for t in trials) if trials else None,random_equal_max=max(t['retrieval']['direction']['different_objects_top4'] for t in trials) if trials else None))
 export_csv(O/'tables/geometry_and_controls.csv',cover)
 # Figure 1: validation is not a new evaluation; final test is from saved logs.
 f,ax=plt.subplots(1,2,figsize=(13,4.4))
 for i,r in enumerate(runs):
  ax[0].plot([x['epoch'] for x in r['epochs']],metric(r,'val_oa_pct'),label=short(r['display_name']),color=COLORS[i],alpha=.85,lw=1.2)
  t=r['identity']['final_test'];oa=t.get('test_oa');ax[1].bar(i,oa,color=COLORS[i]);ax[1].text(i,oa+.04,f'{oa:.4f}%',ha='center')
 ax[0].set(xlabel='实际 epoch',ylabel='validation OA (%)',ylim=(85,96),title='完整 984 验证集：已有训练记录');ax[0].legend(fontsize=9)
 ax[1].set(xticks=range(len(runs)),xticklabels=[short(r['display_name']) for r in runs],ylim=(91.5,93.5),ylabel='test OA (%)',title='验证选模后保存的一次 test（单 seed）');f.tight_layout();save(f,'01_performance')
 # Figure 2: clean vs augmented train, proxy vs whole, raw tangent vs output.
 f,ax=plt.subplots(2,2,figsize=(13,8.8))
 for k,label,col in [('whole_origin_depth','clean whole 中位数',COLORS[0]),('proxy_origin_depth','代理输出中位数',COLORS[1])]:
  med=[x[k]['median'] for x in dr];lo=[x[k]['p05'] for x in dr];hi=[x[k]['p95'] for x in dr]
  ax[0,0].plot(epochs,med,'o-',label=label,color=col);ax[0,0].fill_between(epochs,lo,hi,color=col,alpha=.15)
 ax[0,0].axhline(4.6,ls=':',color='gray',label='forward cap 深度 4.6');ax[0,0].legend(fontsize=9);ax[0,0].set(ylabel='原点双曲深度 d₀',title='V8 冻结 clean 分布：实线中位数，阴影5–95%')
 mainrun=runs[1]
 ax[0,1].plot([x['epoch'] for x in mainrun['epochs']],metric(mainrun,'whole_depth_mean'),color=COLORS[2],label='训练：随机增强/crop + train BN')
 ax[0,1].plot(epochs,[x['whole_origin_depth']['mean'] for x in dr],'o-',color=COLORS[0],label='审查：固定1024点 + eval BN')
 ax[0,1].set(title='不同输入与BN条件，不能混成一条曲线',ylabel='whole 平均 d₀');ax[0,1].legend(fontsize=8)
 ax[1,0].plot(epochs,[x['proxy_saved_tangent_norm']['median'] for x in dr],'o-',color=COLORS[1],label='保存的切空间参数范数中位数');ax[1,0].axhline(2.3,color='gray',ls='--',label='前向 tangent cap 2.3');ax[1,0].set(title='forward cap 不等于参数投影',xlabel='epoch',ylabel='切空间范数');ax[1,0].legend(fontsize=9)
 for i,role in enumerate(['whole','proxy']):ecdf(ax[1,1],va[role+'_d0'],role,COLORS[i])
 ax[1,1].set(title='V8 e300：深代理也可能被训练选中',xlabel='d₀',ylabel='累计比例');ax[1,1].legend();f.tight_layout();save(f,'02_radial_evolution')
 # Figure 3: each original dataset retains its own identity.
 f,ax=plt.subplots(2,3,figsize=(14,8))
 cases=[('HIER_cub_train','HIER_cars_train','HIER_sop_train'),('V7_e187','V8_e221'),('V7_e300','V8_e300')]
 for j,group in enumerate(cases):
  for i,name in enumerate(group):
   x=by[name]
   for row,role in enumerate(['whole','proxy']):ecdf(ax[row,j],x['arrays'][role+'_d0'],x['label'],COLORS[i])
  for row,role in enumerate(['whole','proxy']):ax[row,j].set(xlabel='d₀',ylabel=role+' 累计比例',xlim=(2,6.4) if row==0 else (1.7,6.3),title=[('原版 best：三组 whole 近乎同深' if row==0 else '原版 best：代理分布不同'),'各自 validation best','固定 e300'][j]);ax[row,j].legend(fontsize=8)
 f.tight_layout();save(f,'03_radial_cross_models')
 # Figure 4: true training usage, active hinge and effective count separated.
 f,ax=plt.subplots(1,3,figsize=(14,4.2))
 usage_rows=[]
 for i,r in enumerate(runs[:2]):
  for domain,ls in [('noncollision','-'),('active_noncollision','--')]:
   u=use(r,domain);ax[0].plot([x['epoch'] for x in u],[x['used_proxy_count'] for x in u],ls,color=COLORS[i],label=short(r['display_name'])+' '+('非碰撞' if domain=='noncollision' else '正hinge'))
   for x in u:usage_rows.append(dict(model=short(r['display_name']),domain=domain,epoch=x['epoch'],used=x['used_proxy_count'],effective=x['effective_proxy_count']))
  u=use(r);ax[1].plot([x['epoch'] for x in u],[x['effective_proxy_count'] for x in u],color=COLORS[i],label=short(r['display_name']))
  last={x['epoch']:x for x in u}[300];counts=np.asarray(last['counts']);order=np.argsort(counts)[::-1];ax[2].plot(np.arange(1,513),np.cumsum(counts[order])/counts.sum(),color=COLORS[i],label=short(r['display_name']))
 ax[0].set(xlabel='epoch',ylabel='代理数',ylim=(0,530),title='sample pair∪triple：实际训练选择');ax[0].legend(fontsize=7)
 ax[1].set(xlabel='epoch',ylabel='exp(选择频率熵)',title='512个被用过，不代表均匀使用');ax[1].legend(fontsize=8)
 ax[2].plot([1,512],[1/512,1],':',color='gray');ax[2].set(xlabel='按频率由高到低累计代理数',ylabel='累计选择份额',title='e300 使用频率集中程度');ax[2].legend();f.tight_layout();save(f,'04_training_activation');export_csv(O/'tables/training_usage_aggregate.csv',usage_rows)
 # Figure 5: raw / equal whole / random direction are distinct controls.
 f,ax=plt.subplots(1,2,figsize=(14,5.2));selected=[cover[i] for i in [0,1,2,4,7]];x=np.arange(len(selected));width=.35
 ax[0].bar(x-width/2,[z['raw'] for z in selected],width,label='原始双曲 NN4',color=COLORS[0]);ax[0].bar(x+width/2,[z['equal_whole'] for z in selected],width,label='仅统一whole深度',color=COLORS[1]);ax[0].set(xticks=x,xticklabels=[z['model'].replace('原版 ','').replace(' last e300',' e300') for z in selected],ylabel='512×4槽中不同对象数',title='固定代理位置；不改变训练或分类器');ax[0].legend(fontsize=9)
 for i,z in enumerate(selected):
  a,b=z['random_equal_min'],z['random_equal_max'];ax[1].plot([i,i],[a,b],lw=5,color=COLORS[2]);ax[1].scatter(i,z['equal_whole'],color=COLORS[1],zorder=3)
 ax[1].plot([],[],lw=5,color=COLORS[2],label='随机代理方向 + 等深：5seed范围');ax[1].scatter([],[],color=COLORS[1],label='已学习方向 + 等深');ax[1].set(xticks=x,xticklabels=[z['model'].replace('原版 ','').replace(' last e300',' e300') for z in selected],ylabel='不同对象数',title='随机方向保留每个代理的半径');ax[1].legend(fontsize=8);f.tight_layout();save(f,'05_coverage_counterfactuals')
 # Figure 6: temporal concentration and radius ranks.
 f,ax=plt.subplots(1,3,figsize=(15,4.4))
 ax[0].plot(epochs,[x['raw_top4']['covered_samples'] for x in dr],'o-',color=COLORS[0],label='raw NN4');ax[0].plot(epochs,[x['direction_top4']['covered_samples'] for x in dr],'o-',color=COLORS[1],label='等whole深度/方向NN4');ax[0].set(xlabel='epoch',ylabel='不同对象数',title='V8 早期覆盖增长没有保持至末期');ax[0].legend(fontsize=8)
 ax[1].plot(epochs,[100*x['raw_top4_slot_fraction_bottom_radius_1pct'] for x in dr],'o-',color=COLORS[0],label='最浅1%样本槽份额');ax[1].plot(epochs,[100*x['raw_top4']['top4_sample_slot_share'] for x in dr],'o-',color=COLORS[1],label='最热4个样本槽份额');ax[1].set(xlabel='epoch',ylabel='NN4槽份额 (%)',title='只统计固定全量样本池的反向检索');ax[1].legend(fontsize=8)
 slots=va['raw_whole_top4_slots'];ax[2].scatter(va['whole_d0'],slots,s=8,alpha=.35,color=COLORS[0]);ax[2].set(xlabel='V8 e300 whole d₀',ylabel='被512代理列入NN4的次数',title='热点与浅层显著关联');f.tight_layout();save(f,'06_hotspot_evolution')
 # Figure 7: genuine high-D class separation and proxy dispersion.
 f,ax=plt.subplots(1,3,figsize=(15,4.4))
 ax[0].plot(epochs,[x['whole_directions']['loo_direction_accuracy_macro'] for x in dr],'o-',color=COLORS[0],label='whole 留一类别中心 macro acc');ax[0].set(xlabel='epoch',ylabel='比例',ylim=(0,1.03),title='whole 方向类别分离很早形成');ax[0].legend(fontsize=8)
 ax[1].plot(epochs,[x['proxy_direction_dispersion']['all_proxy_common_resultant_length'] for x in dr],'o-',color=COLORS[1],label='代理 R');ax[1].plot(epochs,[x['whole_directions']['class_balanced_common_resultant_length'] for x in dr],'o-',color=COLORS[0],label='whole 类平衡 R');ax[1].set(xlabel='epoch',ylabel='R = ‖均值单位方向‖',title='代理出现共同分量，未跟随whole同等分散');ax[1].legend(fontsize=8)
 v7active_affinity=None
 for i,name in enumerate(['HIER_cub_train','HIER_cars_train','HIER_sop_train','V7_e300','V8_e300']):
  a=by[name]['summary']['proxy_class_affinity'];rr=[v['angle_deg']['p50'] for v in a['random_direction_reference']];angle=a['nearest_center_angle_deg']['p50']
  if name=='V7_e300':
   za=np.load(V/'hier_v7_direction_hotspots_20261008/evolution_r2/e300/arrays.npz');used=za['sample_used'].astype(bool);angle=float(np.median(za['all__nearest_center_angle_deg'][used]));centers=za['class_centers'];rr=[]
   for seed in range(1001,1006):
    rd=np.random.default_rng(seed).normal(size=(int(used.sum()),centers.shape[1]));rd/=np.linalg.norm(rd,axis=1,keepdims=True);rr.append(float(np.median(np.degrees(np.arccos(np.clip((rd@centers.T).max(axis=1),-1,1))))))
   v7active_affinity=dict(model='V7 e300 sample-used',condition='完整池',proxy_count=int(used.sum()),nearest_angle_median=angle,random_angle_median_min=min(rr),random_angle_median_max=max(rr),inside_direction_cone90=int(za['all__inside_nearest_class_whole_cone90'][used].sum()))
  ax[2].scatter(i,angle,color=COLORS[0],label='当轮sample-used代理' if i==0 else None);ax[2].plot([i,i],[min(rr),max(rr)],lw=5,color=COLORS[2],label='同池同维同数量随机方向' if i==0 else None)
 ax[2].set(xticks=range(5),xticklabels=['CUB','Cars','SOP','V7 used267','V8 used512'],ylabel='最近类别中心夹角中位数 (°)',title='V7剔除当轮未用代理后比较');ax[2].legend(fontsize=8);f.tight_layout();save(f,'07_directions')
 # Figure 8: active proxy affinity, leave hotspots out, high-D cosine map.
 arr=np.load(D/'e300/arrays.npz');cos=arr['proxy_to_whole_class_center_cosine'];nearest=arr['all__nearest_class'];order=np.lexsort((arr['all__nearest_center_angle_deg'],nearest))
 f,ax=plt.subplots(1,2,figsize=(14,6),gridspec_kw={'width_ratios':[1.4,1]})
 im=ax[0].imshow(cos[order],aspect='auto',cmap='coolwarm',vmin=-.4,vmax=.7,interpolation='nearest');ax[0].set(xlabel='40个类别方向中心',ylabel='512代理（按相对最近类别排序）',title='V8 e300 全部512个均为sample-used');f.colorbar(im,ax=ax[0],label='高维方向余弦')
 clean_modes=['all','exclude_bottom_radius_1pct','exclude_raw_top4_union','exclude_both'];labels=['完整池','去掉最浅1%','去掉raw覆盖并集','两者均去掉']
 vals=[arr[k+'__nearest_center_angle_deg'] for k in clean_modes]
 ax[1].boxplot(vals,tick_labels=labels,showfliers=False,orientation='horizontal');ax[1].set(xlabel='最近类别中心夹角 (°)',title='移除热点后重新计算类别中心与亲和');f.tight_layout();save(f,'08_proxy_affinity_controls')
 # Figure 9: class bias and class radial comparison with matched B64.
 # Snapshot class rows are self-describing; use direct baseline cache for identity consistency.
 bp=W/'v8_audit_inputs_20261016/B64/feature_exports/e300_whole_cache.npz';bz=np.load(bp);mu=bz['mu'];bc=float(bz['c']);bd=2/np.sqrt(bc)*np.arctanh(np.minimum(np.linalg.norm(mu,axis=1)*np.sqrt(bc),1-1e-12));bl=bz['labels'].reshape(-1)
 cl=va['class_labels'].astype(int);pop=np.array([(va['labels']==k).sum() for k in cl],float);ct=np.array([slots[va['labels']==k].sum() for k in cl],float);share=ct/slots.sum();ix=np.argsort(share)[-12:][::-1]
 f,ax=plt.subplots(1,2,figsize=(14,5));xx=np.arange(len(ix));ax[0].barh(xx-.18,100*share[ix],.36,color=COLORS[0],label='raw NN4槽');ax[0].barh(xx+.18,100*pop[ix]/pop.sum(),.36,color=COLORS[2],label='完整对象池');ax[0].set(yticks=xx,yticklabels=[NAMES[cl[j]] for j in ix],xlabel='份额 (%)',title='V8 e300 热点类别（前12）');ax[0].invert_yaxis();ax[0].legend(fontsize=8)
 v8med=np.array([np.median(va['whole_d0'][va['labels']==k]) for k in cl]);bmed=np.array([np.median(bd[bl==k]) for k in cl]);ax[1].scatter(bmed,v8med,s=30,color=COLORS[0]);mn=min(bmed.min(),v8med.min());ax[1].plot([mn,4.6],[mn,4.6],':',color='gray');
 for j in ix[:4]:ax[1].annotate(NAMES[cl[j]],(bmed[j],v8med[j]),fontsize=8,xytext=(4,4),textcoords='offset points')
 ax[1].set(xlabel='V8-B64 e300 类别whole深度中位数',ylabel='V8-HIER e300 类别whole深度中位数',title='相同8856对象：加入HIER后类深度变化');f.tight_layout();save(f,'09_hot_classes_and_baseline')
 classrows=[dict(class_name=NAMES[int(k)],count=int(pop[j]),pool_share=float(pop[j]/pop.sum()),raw_nn4_share=float(share[j]),V8_median=float(v8med[j]),B64_median=float(bmed[j])) for j,k in enumerate(cl)];export_csv(O/'tables/class_depth_and_hotspots.csv',classrows)
 # Figure 10: conditional pair/triple roles, never fixed two-level identities.
 f,ax=plt.subplots(1,2,figsize=(13,4.5))
 for i,r in enumerate(runs[:2]):
  for role,ls in [('pair','-'),('triple','--')]:
   vals=metric(r,'sample_noncollision_'+role+'_depth_mean');ax[0].plot([x['epoch'] for x in r['epochs']],vals,ls,color=COLORS[i],label=short(r['display_name'])+' '+role)
 masks={}
 for role in ['pair','triple']:
  u=[x for x in use(mainrun,role=role) if x['epoch']==300][0];masks[role]=np.asarray(u['counts'])>0
 ax[1].bar([0,1,2],[masks['pair'].sum(),masks['triple'].sum(),np.sum(masks['pair']&masks['triple'])],color=[COLORS[0],COLORS[1],COLORS[2]]);ax[1].set(xticks=[0,1,2],xticklabels=['pair used','triple used','两角色交集'],ylim=(0,550),ylabel='代理数',title='V8 e300 sample 非碰撞集合');
 for i in range(3):ax[1].text(i,520,'512',ha='center')
 ax[0].set(xlabel='epoch',ylabel='抽样加权平均 d₀',title='pair/triple是每个三元组的角色');ax[0].legend(fontsize=8);f.tight_layout();save(f,'10_role_depths')
 # Figure 11: independent shape evidence, conditional bootstrap uncertainty.
 chosen=[('V7-HIER64-K20-W0',187),('V7-HIER64-K20-W0',300),('V8-HIER64-K20-W0',221),('V8-HIER64-K20-W0',300),('V8-B64',132),('V8-B64',300)]
 selected_shape=[next(x for x in structure['snapshots'] if x['display_name']==n and x['epoch']==ep) for n,ep in chosen];shaperows=[]
 f,ax=plt.subplots(1,3,figsize=(15,5))
 for j,key in enumerate(['within_class_spearman','chance_adjusted_overlap','full_pool_chamfer_enrichment']):
  for i,x in enumerate(selected_shape):
   s=x['shape']['center_unit_rms']['whole_relations']['hyperbolic'][key];m=s['class_balanced_mean'];lo,hi=s['class_bootstrap_95ci'];ax[j].errorbar(i,m,yerr=[[m-lo],[hi-m]],fmt='o',capsize=3,color=COLORS[i//2]);shaperows.append(dict(model=short(x['display_name']),epoch=x['epoch'],mode='center_unit_rms',metric=key,mean=m,ci_low=lo,ci_high=hi))
  ax[j].set(xticks=range(6),xticklabels=['V7\nbest','V7\ne300','V8\nbest','V8\ne300','B64\nbest','B64\ne300'],title=['类内形态距离 Spearman','扣除随机期望的近邻重叠','全池Chamfer近邻富集'][j]);ax[j].axhline(0,color='gray',ls=':',lw=1)
 f.suptitle('独立形态：同一320查询对象、40类；中心化/单位RMS；按类别bootstrap 95%区间',fontsize=12);f.tight_layout();save(f,'11_independent_shape');export_csv(O/'tables/independent_shape.csv',shaperows)
 paired=[]
 for le,re in [(221,132),(300,300)]:
  left=next(x for x in selected_shape if x['display_name']=='V8-HIER64-K20-W0' and x['epoch']==le);right=next(x for x in selected_shape if x['display_name']=='V8-B64' and x['epoch']==re)
  la=left['shape']['center_unit_rms']['whole_relations']['hyperbolic']['objects'];ra=right['shape']['center_unit_rms']['whole_relations']['hyperbolic']['objects'];lm={x['sample_id']:x for x in la};rm={x['sample_id']:x for x in ra};assert set(lm)==set(rm)
  for key in ['within_class_spearman','chance_adjusted_overlap','chamfer_enrichment']:
   delta=[]
   for k in range(40):
    q=[float(lm[i][key])-float(rm[i][key]) for i in lm if lm[i]['label']==k and lm[i][key] is not None and rm[i][key] is not None];delta.append(float(np.mean(q)))
   rng=np.random.default_rng(20261016);bs=np.mean(np.asarray(delta)[rng.integers(0,40,(2000,40))],axis=1);lo,hi=np.quantile(bs,[.025,.975]);paired.append(dict(V8_epoch=le,B64_epoch=re,metric=key,paired_class_mean_difference=float(np.mean(delta)),ci_low=float(lo),ci_high=float(hi),bootstrap=2000))
 export_csv(O/'tables/paired_shape_differences.csv',paired)
 # Figure 12: full shared-encoder gradients, no optimizer step.
 probes=[read(W/f'backbone_e{ep}.json') for ep in [200,221,300]];proberows=[]
 f,ax=plt.subplots(1,2,figsize=(12,4.2))
 for p in probes:
  z=p['parameter_groups']['shared_encoder']['hier_vs_base'];proberows.append(dict(epoch=p['epoch'],weighted_hier_over_base=z['norm_ratio'],cosine=z['cosine'],optimizer_updates=p['read_only']['optimizer_updates'],replay_passed=p['replay_validation']['passed']))
 ax[0].plot([p['epoch'] for p in probes],[r['weighted_hier_over_base'] for r in proberows],'o-',color=COLORS[0]);ax[0].set(xlabel='epoch',ylabel='‖λ∇HIER‖ / ‖∇(CE+intra)‖',title='真实共享编码器参数：不是feature梯度')
 ax[1].plot([p['epoch'] for p in probes],[r['cosine'] for r in proberows],'o-',color=COLORS[2]);ax[1].axhline(0,color='gray',ls=':');ax[1].set(xlabel='epoch',ylabel='两项共享编码器梯度余弦',title='固定64对象/eval BN/原crop别名行为');f.tight_layout();save(f,'12_encoder_gradients');export_csv(O/'tables/encoder_gradients.csv',proberows)
 # Figure 13: fixed queries, different Gumbel draws; training counts above use logs.
 f,ax=plt.subplots(1,3,figsize=(14,4.5));mrows=[]
 for i,version in enumerate(['v7','v8']):
  x=next(x for x in mech['snapshots'] if x['version']==version and x['epoch']==300)
  for j,comp in enumerate(['sample','proxy']):
   ns=x['components'][comp]['noise_summary'];sv=ns['scalar_variation'];ag=ns['independent_noise_pair_agreement'];row=dict(model=version,component=comp,active=sv['active_fraction_noncollision']['mean'],collision=sv['collision_rate']['mean'],pair_agreement=ag['pair_proxy_ids_agreement']['mean'],triple_agreement=ag['triple_proxy_ids_agreement']['mean']);mrows.append(row)
   xx=i*3+j
   ax[0].bar(xx,row['active'],color=COLORS[i],alpha=1 if j==0 else .55);ax[1].bar(xx,row['collision'],color=COLORS[i],alpha=1 if j==0 else .55);ax[2].bar(xx-.15,row['pair_agreement'],.3,color=COLORS[i]);ax[2].bar(xx+.15,row['triple_agreement'],.3,color=COLORS[i],alpha=.4)
 for a,t in zip(ax,['非碰撞查询中正hinge比例','pair/triple碰撞比例','独立噪声祖先ID一致率（深/浅：pair/triple）']):a.set(xticks=[0,1,3,4],xticklabels=['V7\nsample','V7\nproxy','V8\nsample','V8\nproxy'],title=t,ylim=(0,1))
 f.suptitle('e300固定64三元组、8次Gumbel；冻结条件响应，不等于训练一轮比例',fontsize=12);f.tight_layout();save(f,'13_gumbel_mechanisms');export_csv(O/'tables/frozen_mechanisms.csv',mrows)
 # Figure 14: same-object clean/baseline depth ranks; no private IDs in figure.
 ids=va['sample_ids'].astype(int);bi={int(k):i for i,k in enumerate(bz['sample_ids'].astype(int))};assert set(ids)==set(bi);bb=bd[[bi[int(k)] for k in ids]];ranks=np.argsort(np.argsort(va['whole_d0'],kind='stable'),kind='stable')+1;branks=np.argsort(np.argsort(bb,kind='stable'),kind='stable')+1;hot=np.argsort(slots)[-4:][::-1]
 f,ax=plt.subplots(1,2,figsize=(13,4.5));ax[0].scatter(100*ranks/len(ids),100*branks/len(ids),s=4,alpha=.12,color=COLORS[0]);ax[0].scatter(100*ranks[hot]/len(ids),100*branks[hot]/len(ids),s=50,color=COLORS[1],label='V8最热4对象');ax[0].set(xlabel='V8深度百分位',ylabel='B64深度百分位',title='同一对象在无HIER对照里是否也浅？');ax[0].legend(fontsize=8)
 cc=np.load(W/'v8_audit_20261016_main_r1/feature_exports/candidate_clouds.npz');cloudmap={int(k):i for i,k in enumerate(cc['sample_ids'])};
 # Gallery gets a separate figure to keep each object legible.
 ax[1].bar(np.arange(4)-.18,100*ranks[hot]/len(ids),.36,color=COLORS[0],label='V8');ax[1].bar(np.arange(4)+.18,100*branks[hot]/len(ids),.36,color=COLORS[2],label='B64');ax[1].set(xticks=np.arange(4),xticklabels=[NAMES[int(va['labels'][i])] for i in hot],ylabel='由浅到深的百分位',title='最热4例：排名来自完整8856池');ax[1].legend();f.tight_layout();save(f,'14_same_object_depth_ranks')
 f=plt.figure(figsize=(13,3.5))
 for j,i in enumerate(hot):
  a=f.add_subplot(1,4,j+1,projection='3d');c=cc['clouds'][cloudmap[int(ids[i])]][::2];a.scatter(c[:,0],c[:,2],c[:,1],s=1,c=c[:,1],cmap='viridis');a.set_box_aspect((1,1,1));a.set_axis_off();a.view_init(20,45);a.set_title(f'{NAMES[int(va["labels"][i])]}\nNN4槽 {int(slots[i])}；d₀={va["whole_d0"][i]:.3f}',fontsize=10)
 f.suptitle('V8 e300 最热四例：原始1024点的可视化（显示隔点512），标签并非形态层级标注',fontsize=11);f.tight_layout();save(f,'15_hotspot_pointclouds')
 hotrows=[dict(rank=j+1,class_name=NAMES[int(va['labels'][i])],slots=int(slots[i]),V8_depth=float(va['whole_d0'][i]),V8_depth_percentile=float(100*ranks[i]/len(ids)),B64_depth_percentile=float(100*branks[i]/len(ids))) for j,i in enumerate(hot)];export_csv(O/'tables/hottest_four_aggregate.csv',hotrows)
 # Figure 16: per-proxy temporal sample binding, with actual current-used mask.
 temporal=[]
 for prev,curr in zip(dr[:-1],dr[1:]):
  pa=np.load(D/f'e{prev["epoch"]:03}/arrays.npz');ca=np.load(D/f'e{curr["epoch"]:03}/arrays.npz');assert np.array_equal(pa['proxy_ids'],ca['proxy_ids'])
  used=ca['sample_used'].astype(bool);rawjac=[];dirjac=[]
  for p,q in zip(pa['raw_top4_ids'],ca['raw_top4_ids']):rawjac.append(len(set(p)&set(q))/len(set(p)|set(q)))
  for p,q in zip(pa['direction_top4_ids'],ca['direction_top4_ids']):dirjac.append(len(set(p)&set(q))/len(set(p)|set(q)))
  temporal.append(dict(from_epoch=prev['epoch'],to_epoch=curr['epoch'],gap=curr['epoch']-prev['epoch'],raw_nn4_mean_jaccard=float(np.mean(rawjac)),direction_nn4_mean_jaccard=float(np.mean(dirjac)),current_used_proxy_count=int(used.sum()),nearest_class_retention_current_used=float(np.mean(pa['all__nearest_class'][used]==ca['all__nearest_class'][used])) if used.any() else None))
 f,ax=plt.subplots(1,2,figsize=(14,4.5));tx=np.arange(len(temporal));tl=[f'{r["from_epoch"]}→{r["to_epoch"]}' for r in temporal]
 for k,label,color in [('raw_nn4_mean_jaccard','raw NN4',COLORS[0]),('direction_nn4_mean_jaccard','等whole深度/方向NN4',COLORS[1])]:ax[0].plot(tx,[r[k] for r in temporal],'o-',label=label,color=color)
 ax[0].set(xticks=tx,xticklabels=tl,ylim=(0,1),ylabel='相同代理四例集合平均Jaccard',title='相邻已保存快照：对象身份严格配对');ax[0].tick_params(axis='x',rotation=45);ax[0].legend(fontsize=8)
 ax[1].plot(tx,[r['nearest_class_retention_current_used'] for r in temporal],'o-',color=COLORS[2]);ax[1].set(xticks=tx,xticklabels=tl,ylim=(0,1.03),ylabel='相对最近类别保留比例',title='只统计后一快照当轮sample-used代理');ax[1].tick_params(axis='x',rotation=45);f.tight_layout();save(f,'16_binding_stability');export_csv(O/'tables/temporal_binding.csv',temporal)
 # Figure 17: radial derivatives of clipped proxies vs the two interior proxies.
 cap=arr['proxy_saved_tangent_norm']>2.3;pr=probes[-1];caprows=[]
 f,ax=plt.subplots(1,2,figsize=(13,4.5));groups=[];groupnames=[]
 for j,comp in enumerate(['sample','proxy']):
  per=pr['proxy_tangent_partials'][comp]['per_id'];nn=np.array([p['norm'] for p in per]);rr=np.array([p['signed_radial'] for p in per]);frac=np.abs(rr)/np.maximum(nn,1e-30)
  ax[0].scatter(np.flatnonzero(cap),np.maximum(nn[cap],1e-16),s=7,alpha=.5,color=COLORS[j],label=comp+'：越cap');ax[0].scatter(np.flatnonzero(~cap),np.maximum(nn[~cap],1e-16),s=55,marker='x',color=COLORS[j],label=comp+'：cap内')
  for mask,label in [(cap,'越cap'),(~cap,'cap内')]:
   groups.append(np.maximum(frac[mask],1e-10));groupnames.append(comp+'\n'+label)
   caprows.append(dict(component=comp,group=label,proxy_count=int(mask.sum()),gradient_norm_median=float(np.median(nn[mask])),radial_fraction_median=float(np.median(frac[mask])),nonzero_gt_1e12=int((nn[mask]>1e-12).sum())))
 ax[0].set(yscale='log',xlabel='稳定代理行号（同一run内）',ylabel='单代理原切空间梯度范数（未加权）',title='e300固定64对象：两个分项仍有角向梯度');ax[0].legend(fontsize=8)
 ax[1].boxplot(groups,tick_labels=groupnames,showfliers=False);ax[1].set(yscale='log',ylabel='|径向导数| / 梯度范数',title='510个越cap代理与2个内部代理分开');f.tight_layout();save(f,'17_cap_gradient_groups');export_csv(O/'tables/cap_group_gradients.csv',caprows)
 # Figure 18: actual training lifetimes, one stable proxy row across 300 epochs.
 f,ax=plt.subplots(1,2,figsize=(14,6));life=[]
 for j,r in enumerate(runs[:2]):
  u=use(r);matrix=np.stack([x['counts'] for x in u],axis=1);order=np.argsort(matrix[:,-1],kind='stable')[::-1];im=ax[j].imshow(np.log10(1+matrix[order]),aspect='auto',origin='upper',extent=(.5,300.5,512.5,.5),cmap='viridis',vmin=0,vmax=5);ax[j].set(xlabel='epoch',ylabel='稳定代理行（按e300使用频率排序）',title=short(r['display_name'])+' 实际sample非碰撞选择次数');life.append(dict(model=short(r['display_name']),never_used=int((matrix.sum(axis=1)==0).sum()),used_in_every_epoch=int((matrix>0).all(axis=1).sum()),unused_at_e300=int((matrix[:,-1]==0).sum())))
 f.subplots_adjust(right=.86,wspace=.25);cax=f.add_axes([.89,.2,.018,.6]);f.colorbar(im,cax=cax,label='log10(1 + pair/triple合计选择次数)');save(f,'18_proxy_lifetimes');export_csv(O/'tables/proxy_lifetimes_aggregate.csv',life)
 # Additional high-D temporal metrics in aggregate JSON, no per-object IDs.
 evo_rows=[]
 for x in dr:
  evo_rows.append(dict(epoch=x['epoch'],whole_median=x['whole_origin_depth']['median'],whole_std=x['whole_origin_depth']['std_population'],proxy_median=x['proxy_origin_depth']['median'],proxy_std=x['proxy_origin_depth']['std_population'],tangent_median=x['proxy_saved_tangent_norm']['median'],whole_direction_loo=x['whole_directions']['loo_direction_accuracy_macro'],proxy_R=x['proxy_direction_dispersion']['all_proxy_common_resultant_length'],sample_used=x['usage'].get('sample_used_count'),raw_coverage=x['raw_top4']['covered_samples'],equal_coverage=x['direction_top4']['covered_samples'],bottom1_share=x['raw_top4_slot_fraction_bottom_radius_1pct']))
 export_csv(O/'tables/evolution_snapshots.csv',evo_rows)
 affinity=[]
 for x in geometry:
  a=x['summary']['proxy_class_affinity'];rr=[z['angle_deg']['p50'] for z in a['random_direction_reference']]
  affinity.append(dict(model=x['label'],condition='完整池',proxy_count=512,nearest_angle_median=a['nearest_center_angle_deg']['p50'],random_angle_median_min=min(rr),random_angle_median_max=max(rr),inside_direction_cone90=a['proxy_count_in_nearest_class_direction_cone90']))
 for k,label in zip(clean_modes,labels):
  affinity.append(dict(model='V8 e300 sample-used',condition=label,proxy_count=512,nearest_angle_median=float(np.median(arr[k+'__nearest_center_angle_deg'])),random_angle_median_min=None,random_angle_median_max=None,inside_direction_cone90=int(arr[k+'__inside_nearest_class_whole_cone90'].sum())))
 affinity.append(v7active_affinity)
 export_csv(O/'tables/proxy_class_affinity.csv',affinity)
 # Public report assembled from the numeric records above.
 report=Report(O);report.title('V8系统审查：与V7及原版HIER的几何、激活和热点对照')
 report.p('审查日期：2026-10-16。主范围为已经完成300轮的V8-HIER64及其V8-B64同协议无HIER对照；V7与原版CUB/Cars/SOP best作为历史参考。仅读取既有训练记录、训练集冻结前向和缓存；未更新参数、未新增验证/测试评估。V8-B0-32-C01单卡shuffle对照在本报告采集时仍训练中，另列进度，不伪填最终结果。')
 progress=read(W/'B0_progress.json') if (W/'B0_progress.json').exists() else None
 if progress:
  b=progress['best'];report.p(f'单卡B0进度快照（UTC {progress["captured_utc"]}）：{progress["status"]}，e{progress["epoch"]}第{progress["step"]}/{progress["steps_per_epoch"]}步；当前best e{b["epoch"]}，val OA {b["val_oa"]:.4f}%。尚无final test；本报告的完整几何与主结论不含这一未完成组。')
 report.h('1. 可以先读的判断')
 report.p('V8消除了V7末期大批代理未被样本祖先选择的现象，反向NN4覆盖从27提升到231，但没有解决热点，也没有形成类似CUB/Cars那样的内部代理分布。V8末期510/512代理触及forward cap，代理输出几乎同深4.6；clean whole则分散在2.51–4.60，中位数3.74。最浅1%的whole仍占72.17%的NN4槽。')
 report.p('V8-HIER test OA为92.7877%，同协议关闭HIER的B64为92.9092%，相差−0.1216个百分点，即3/2468个对象。独立形态指标存在局部差异，但不足以主张稳定结构收益或分类收益。下一步应优先解释“whole径向离散 + 代理cap薄壳 + 方向共同分量”的联合状态，而不是再次把代理休眠当成唯一原因。')
 report.h('2. 审查范围、口径与可比性')
 report.table(['对象','训练/审查范围','用于什么比较'],[['V8-HIER64','300轮日志；12个checkpoint，均8856训练对象','时间演变、训练真实usage、冻结机制和形态'],['V8-B64','300轮；12个checkpoint，均8856对象','相同V8几何与batch协议，仅关闭HIER'],['V7-HIER64','既有300轮日志及11个全池快照','版本结果对照；曲率、cap、hook及权重不同'],['原版CUB / Cars / SOP','分别为5864 / 8054 / 59551个训练对象，best缓存','保留不同数据、512维、类别数及训练协议边界'],['V8-B0-32-C01','单卡shuffle32；采集时仍在训练','仅列现有进度；不当作已完成几何审查']])
 report.p('“训练使用”指实际训练hard Gumbel祖先日志；另区分非碰撞选择与正hinge选择。“反向NN4”指每个代理到完整样本池查四个最近对象，一共2048槽；它既不是训练三元组采样，也不是一轮训练覆盖。hard选择未选中仍可能收到ST软选择梯度，proxy项查询端点也能收到梯度。静态whole→proxy最近邻占用率同样不等于训练激活。')
 report.p('所有ModelNet40快照采用相同固定对象身份与clean first1024/eval BN。主干真实训练仍使用随机增强、随机whole/part与train BN；两套深度曲线必须分开。原版三个数据集的最佳权重不能推断其训练演变；跨数据集覆盖绝对数、角度与聚类纯度也不能当作统一任务性能排名。')
 report.h('3. 分类效果：先看同协议无HIER对照')
 performance=[]
 for r in runs:
  b=r['identity']['best'];t=r['identity']['final_test'];performance.append(dict(model=short(r['display_name']),best_epoch=b['epoch'],val_oa=b['val_oa'],test_oa=t['test_oa'],test_aa=t['test_aa']))
 report.table(['模型','best轮次','val OA %','test OA %','test AA %'],[[p['model'],p['best_epoch'],fmt(p['val_oa'],4),fmt(p['test_oa'],4),fmt(p['test_aa'],4)] for p in performance]);report.fig('01_performance','图1：保存的完整验证曲线与最终一次test。审查没有重跑test；不同版本曲线不构成单因素因果实验。')
 report.p('V8相对V7的OA上升同时混合了c=1→0.1、官方cap/hook、intra margin和HIER权重变化，不能全部归于inter目标。B64是当前最直接的目标开关对照：初始化、划分、seed、global64和300×200步一致。两机硬件/运行位置仍不同，且只有一个seed；0.1216个百分点差异不支持显著胜负判断。')
 report.h('4. 半径演化：V8没有复制出原版whole薄壳')
 report.math(r'd_0(x)=\frac{2}{\sqrt c}\operatorname{artanh}(\sqrt c\|x\|),\quad q=\sqrt c\|x\|,\quad \rho=d_0/2.\qquad \|v\|\le2.3\Longrightarrow d_0\lesssim4.6.')
 report.p('切空间参数范数、球坐标半径、原点双曲距离是三个量。V8 c≈0.1时，cap后的坐标半径约1.965、归一化半径q≈0.621、双曲深度4.6；不是到了数值球边界q=0.999。原版1e−5等数值细节会带来微小偏差。forward cap只截断参与映射的向量；它不会把优化器保存的原参数模长投回2.3。')
 report.fig('02_radial_evolution','图2：V8从初始化到e300的clean径向分布、训练/clean条件差异和参数范数。阴影是5–95%分位；稀疏快照之间的连线仅供观察。')
 report.p('V8初始化whole已几乎在4.6薄壳；e5仍接近薄壳，随后clean whole向内且分散，e221中位数3.999、e300为3.741。代理反而很快顶到4.6，末期保存参数范数中位数6.923。两者都受同一个cap约束，仍能学出不同实际分布：cap是上限，既不是目标半径，也不是whole和代理的相对深度保证。')
 report.fig('03_radial_cross_models','图3：原版三个best分别显示；V7/V8各自best与固定e300分别比较。原版whole标准差仅约3–6×10⁻⁷；V8末期whole标准差约0.350。')
 tab=[]
 for x in geometry:
  if x['name']=='V8_e005':continue
  for role in ['whole','proxy']:
   s=stats(x['arrays'][role+'_d0']);tab.append([x['label'],role,fmt(s['min']),fmt(s['p05']),fmt(s['median']),fmt(s['p95']),fmt(s['max']),f'{s["std"]:.3g}'])
 report.table(['模型','对象','min','p05','median','p95','max','总体std'],tab);report.link('tables/radial_statistics.csv','完整径向统计CSV：含均值、方差与四分位数')
 report.h('5. 激活、休眠与pair/triple：V7的问题确实缓解了')
 report.fig('04_training_activation','图4：来自300轮真实训练日志的非碰撞与正hinge祖先使用；有效代理数与累计选择频率用于识别“全用过但仍不均匀”。')
 end7=next(x for x in use(runs[0]) if x['epoch']==300);end8=next(x for x in use(mainrun) if x['epoch']==300);act8=next(x for x in use(mainrun,'active_noncollision') if x['epoch']==300)
 report.p(f'固定e300：V7 sample非碰撞祖先并集为{end7["used_proxy_count"]}个，V8为{end8["used_proxy_count"]}个；V8正hinge祖先并集为{act8["used_proxy_count"]}个。V8非碰撞频率熵对应有效代理数为{end8["effective_proxy_count"]:.1f}，仍低于512。不存在“512个都出现过，所以训练均匀”的推论。')
 report.fig('18_proxy_lifetimes','图4补充：每个稳定代理行的300轮真实使用轨迹。每列对应一个训练epoch，不从快照插值激活。')
 report.p('两个版本在已记录300轮中都没有从未被sample非碰撞选过的代理；V7末期245个当轮未用，属于后期休眠，不能描述为整个训练始终未受训练。V8有506个代理每个epoch都被使用，末期未用数为0。')
 report.fig('10_role_depths','图5：sample pair与triple抽样加权深度，以及V8两角色集合交集。此处是训练日志，不是全量NN4。')
 report.p('V8 e300 sample非碰撞pair/triple深度均值4.5991/4.5398，差约0.0593；两角色都覆盖512代理，交集也是512。不能将它们解释成互不重叠的两层代理。CUB/Cars原版同样角色高度重叠，SOP几乎同深。V7曾出现的浅层sample-used与深层未用分裂，没有以同样形式出现在V8末期；V8更主要的径向问题变成绝大多数代理挤在输出cap薄壳。')
 report.h('6. 热点：改善明显，径向偏置与方向冗余仍同时存在')
 report.table(['模型','raw覆盖','等whole深度覆盖','最大重复四例','最热4例槽份额'],[[z['model'],z['raw'],z['equal_whole'],z['raw_max_quartet'],f'{100*z["raw_hot4_share"]:.2f}%'] for z in selected]);report.fig('05_coverage_counterfactuals','图6：等深控制仅改whole半径、保留全部方向与代理；随机方向控制则改变代理方向并保留各自半径。二者不能混称。')
 report.p('原版CUB/Cars whole本来几乎等深，因此统一深度后覆盖基本不变；CUB严格稳定距离为1646，旧官方FP32为1647，差1来自极近邻排序。原版SOP仍是432，说明等深也可能存在方向热点。V7为27→221，V8为231→1051；因此V8当前几何中的径向差异确实显著改变反向近邻排序。这个冻结控制没有证明训练中全部热点由半径单独造成。')
 report.p('V8 e300随机代理方向保留半径，raw覆盖只有53–62，低于已训练方向231；配合等whole深度才得到1476–1545，高于已训练方向1051。这同时表明：学习到的方向已部分抵消径向偏置，但仍有相对于随机基准的覆盖冗余。不能因此直接重置代理方向，随机方向扩大覆盖不等于改善分类或形态关系。')
 report.fig('06_hotspot_evolution','图7：V8覆盖随训练收缩；e300最浅1%对象占72.17%反向NN4槽，最热4例占24.85%。')
 report.p('e5 raw覆盖748，best e221为236，e300为231；等深覆盖分别1345、1045、1051。热点并不是只有末期某几个深代理才造成；早期已有径向敏感性，训练继续后方向与半径两方面的集中共同影响覆盖。最大重复四例由V7 e300的287组降到V8的34组，但仍应保留热点警报。')
 report.fig('09_hot_classes_and_baseline','图8：热点槽类别份额与样本池份额，以及V8-HIER/B64相同对象池的类别深度中位数。类别偏置不等于语义上更高层。')
 report.fig('14_same_object_depth_ranks','图9：使用同一对象ID配对V8与B64，比较最热样本在无HIER对照中的深度百分位。B64也是V8几何，并非最初c=1的HyCoRe。')
 report.table(['热度排名','类别','槽数','V8 d₀','V8深度百分位','B64深度百分位'],[[x['rank'],x['class_name'],x['slots'],fmt(x['V8_depth']),fmt(x['V8_depth_percentile'],2)+'%',fmt(x['B64_depth_percentile'],2)+'%'] for x in hotrows]);report.fig('15_hotspot_pointclouds','图10：最热四个对象的点云。用于辨识对象形态，不从四个例子推导全类别机制。')
 report.p('最热四例为plant、两个vase和bookshelf，分别是全池由浅到深第5、1、2、21个对象；并不是恰好最浅的前四个。在B64里这四例深度百分位约5.27%、2.12%、0.21%、1.69%，本来偏浅，但V8中更加极端；不能把全部差异都说成HIER凭空制造，也不能把B64等同最初c=1的原始HyCoRe。')
 report.h('7. 类别方向：whole分开了，代理有亲和但仍处于类别方向外围')
 report.math(r'u=x/\|x\|,\quad m_k=\frac1{N_k}\sum_{i:y_i=k}u_i,\quad \hat m_k=m_k/\|m_k\|,\quad \theta_{pk}=\arccos(u_p^\top\hat m_k),\quad R=\|P^{-1}\sum_p u_p\|.')
 report.p('留一类别中心准确率是训练集方向结构诊断，中心排除查询对象；它不是独立测试准确率。最近类别是与40个类别方向中心夹角最小者，任何代理都会得到一个相对最近标签，不代表位于该类簇内。R是平均单位方向的模长，不是某一对代理的余弦。')
 report.fig('07_directions','图11：whole方向类别分离、代理共同方向与各数据集匹配的随机方向角度基准。原版512维、类别数不同，所以不能单看角度大小排名。')
 report.p(f'V7角度面板已按当轮sample-used筛为{v7active_affinity["proxy_count"]}个，并重新生成同数量随机方向基准；中位数为{v7active_affinity["nearest_angle_median"]:.2f}°。V8此时512个均sample-used。原版三个best的现有真实图像无更新窗口中sample有效祖先也均为512个。全代理方向集中度与使用组亲和是分别定义的指标，不将V7深层未用群的角度混入使用组结论。')
 report.p('V8 whole留一方向macro准确率由初始化约0.405升到e5的0.807、e20的0.926、e100的0.977及e300的0.998；类别方向分离确实发生。代理R则由0.041升到e300的0.430，出现共同方向分量，尚不及原版SOP的0.648，但远高于CUB/Cars的0.044/0.047。')
 report.p('V8 e300所有代理都是当轮sample-used，其最近类别中心夹角中位数74.39°，相同全池/256维随机方向参考约82.7°，所以不能说代理完全没学到类别亲和。然而512代理都不落入最近类别包含90%样本的方向锥；它们更像位于类别簇外围的关系节点。CUB/Cars原版也无代理进入该方向锥，不能把“必须进入细类簇”当作HIER的既定设计目标。方向锥也不等于双曲凸包。')
 report.fig('08_proxy_affinity_controls','图12：仅使用实际sample-used代理（V8此时为全部512）分析高维类别余弦，并剔除热点/最浅对象后重新计算类别中心。绝非二维投影的簇归属推断。')
 report.p('去除最浅1%、raw NN4覆盖并集和二者并集的亲和对照用于检查热点污染；完整数值随表保存。类别标签仍是单层人工标签，较高top-k同类纯度或最近类别稳定本身不能验证“发现了真实形态层级”。')
 report.fig('16_binding_stability','图12补充：相同代理的最近四例与相对最近类别随时间的保留率。快照间隔不等，不能当作单位epoch变化率；相对最近类别稳定也不是簇内归属。')
 lastt=temporal[-1];report.p(f'e240→e300：相对最近类别在当轮sample-used代理中的保留率为{lastt["nearest_class_retention_current_used"]:.1%}；raw NN4平均Jaccard为{lastt["raw_nn4_mean_jaccard"]:.3f}，方向NN4为{lastt["direction_nn4_mean_jaccard"]:.3f}。代理类别方向偏好与精确最近样本的稳定性是两个层面。高维方向亲和确有演化，但与whole形成一致类内软结构仍需独立形态证据。')
 report.h('8. 独立形态检查：尚无一致的结构收益')
 report.p('采用同一320训练查询对象（40类各8例）、完整8856候选池和FPS64点云Chamfer信号，分别检查原坐标与中心化/单位RMS。这里展示去整体尺度影响后的结果；类内Spearman来自每类其余7查询，近邻重叠已扣除4/7的随机期望。全池富集还使用完整候选池。误差条为固定图上200次类别bootstrap，不代表多seed训练置信区间。')
 report.fig('11_independent_shape','图13：三个互补形态指标。选择best与固定e300分别展示，避免用一个更好的数值概括全部结构。')
 report.p('e300 V8/B64：类内Spearman为0.1269/0.1291，扣随机近邻重叠为0.0849/0.0576，全池富集为0.2558/0.2122。best时V8/B64相应为0.1496/0.1599、0.1195/0.0977、0.2987/0.3164。存在指标与选模轮次依赖，不能认定HIER带来一致形态改进；V8类内Spearman也低于V7 best的0.2135。Chamfer是独立几何信号，但仍不是人工层级真值。')
 report.table(['V8/B64轮次','形态指标','配对差值 V8−B64','按类别配对bootstrap 95%区间'],[[str(x['V8_epoch'])+'/'+str(x['B64_epoch']),x['metric'],fmt(x['paired_class_mean_difference'],4),'['+fmt(x['ci_low'],4)+', '+fmt(x['ci_high'],4)+']'] for x in paired])
 report.p('差值额外按同一查询对象配对后，在40个类别均值上做2000次bootstrap；比只看两组误差条是否重叠更合适。该区间仍只描述当前两个固定模型和这一形态指标，不包含重训随机性或best轮次选择不确定性。')
 report.p('其中固定e300的全池Chamfer富集差值+0.0436，配对区间约[+0.0166,+0.0695]不含0，是当前固定模型上的局部正面证据；但best对比的该指标变为负差，另外两项区间含0，因此仍不能扩展为普遍结构收益。')
 report.h('9. 损失与梯度：没有被hook完全压没，也不等同于优化器实际位移')
 report.fig('12_encoder_gradients','图14：通过两遍feature-adjoint/VJP回传获得13227200个共享编码器参数的真实偏导；重放特征最大差0，RNG/BN/参数保持，零optimizer更新。')
 report.table(['轮次','加权HIER/base梯度范数比','梯度余弦','参数更新'],[[x['epoch'],fmt(x['weighted_hier_over_base'],4),fmt(x['cosine'],4),x['optimizer_updates']] for x in proberows])
 report.p('e221/e300 HIER/base编码器梯度比约0.326/0.344，余弦约0.145/0.374。HIER确实能影响编码器，不能把无分类收益简单归因于hook把信号全部抹掉。proxy–proxy项对编码器梯度为0，是因为它只依赖代理参数，不是故障。此为固定64对象、固定crop/eval BN、保留原HyCoRe child别名覆盖行为的条件探针，不能替代train BN全轨迹或实际Adam/RSGD更新分析。源manifest没有独立train-label SHA字段，探针另记录真实载入标签与输入哈希；重放一致性与不变性已验收。')
 report.fig('13_gumbel_mechanisms','图15：相同查询与不同Gumbel噪声的冻结损失响应；与图4真实整轮训练激活统计分开。')
 report.p('V8 e300固定sample查询的pair ID独立噪声一致率约0.752，triple约0.156：共同祖先选择仍有很强随机性。sample非碰撞正hinge比例约0.220，proxy约0.139；这既不等于512代理都没约束，也不能把选择ID变动全部解释为结构不稳定。代理切空间梯度在两个分项均可非零；多数越cap代理的径向可优化性被压制，角向信号仍存在。少数未cap代理可主导整体径向梯度范数，不能用整体范数代表510个cap代理各自情况。')
 report.fig('17_cap_gradient_groups','图15补充：按原参数是否超过2.3，拆分两个分项的梯度。正径向导数对应朴素负梯度下降的向内趋势；不能直接等同AdamW实际位移。')
 report.h('10. 与原版HIER的联系，以及下一步可检验的假设')
 report.p('CUB/Cars：whole几乎严格同深，代理稍浅且方向分散；SOP：whole同样同深，但大多数代理cap且方向共同分量强，仍存在热点；V7：whole分散并伴深层未使用代理群；V8：whole后期分散、几乎所有代理cap，但全部仍被sample使用。四种状态说明热点、代理休眠、cap饱和和角色层次是相关但不等价的维度。')
 report.table(['假设','当前支持','尚不能证明','优先验证办法'],[['whole深度离散影响反向检索','V8仅等whole深度231→1051','不能推出统一半径训练会提高分类','保持候选与查询不变，先比较角向/径向敏感性及类内形态'],['代理共同方向造成覆盖冗余','V8 R=.430；等深学习方向覆盖低于随机','随机方向覆盖高不是结构收益','保留代理Gram的整体旋转、去均值控制，并同时评价形态/纯度'],['forward cap改变径向学习','510/512越cap，代理输出几乎同深','不能说角向停止训练或只有proxy项激活','按cap组拆sample/proxy的径向和角向梯度、频率与Adam状态'],['当前HIER关系不够贴合分类/形态','同协议无OA增益，形态指标不一致','不足以否定所有inter目标','固定V8主协议，先定位采样关系与独立形态一致性，再设计单因素试验']])
 report.p('本次不启动新训练矩阵，也不基于test挑方案。建议优先做冻结机制控制，只有当覆盖改善同时保留或提高独立形态指标时，再提出新的训练计划。若补训练，应一次只改一个关键机制，并保留V8-B64的同协议对照。')
 report.h('11. 可复现入口、文件与检查记录')
 report.p('审查代码已为V8显式适配最终共享表示cap/hook、官方mobius+eps距离、c≈0.1及gamma500/delta2；遗留c=1路径保持独立。原始训练commit：V8主模型14f8a72，B64为85fc812；审查实现ad53d36/7ebc081。154项现有及V8合同测试通过。官方CPU FP32 NN4与独立稳定距离在e5/e221/e300覆盖分别同为748/236/231，降低算子误用造成结论的风险；并不保证每个极近邻排序逐位等同CUDA AMP。')
 report.p('全系统包含：逐轮性能/损失/LR/裁剪、几何分布、真实选择与生命周期、全池近邻与热点、角色与碰撞、Gumbel重复、分项梯度、Adam状态、独立形态及真实编码器梯度。主/对照共24个新全量训练集特征快照；结合V7为35个系统快照；8个固定机制快照、12个独立形态快照、3个编码器探针。全量缓存/点云/权重/日志/身份表留私有，不进入公开报告或Git。')
 for name,label in [('geometry_and_controls.csv','几何与冻结控制总表'),('evolution_snapshots.csv','V8快照时间演变'),('training_usage_aggregate.csv','真实训练祖先使用汇总'),('class_depth_and_hotspots.csv','类别深度与热点份额'),('independent_shape.csv','独立形态与bootstrap区间'),('paired_shape_differences.csv','同对象配对形态差值'),('encoder_gradients.csv','编码器梯度结果'),('frozen_mechanisms.csv','冻结Gumbel机制汇总'),('hottest_four_aggregate.csv','最热四例匿名统计'),('temporal_binding.csv','样本绑定与最近类别稳定性'),('cap_group_gradients.csv','cap组分项梯度'),('proxy_lifetimes_aggregate.csv','代理生命周期汇总'),('proxy_class_affinity.csv','代理类别方向亲和与热点剔除对照')]:report.link('tables/'+name,label)
 links=report.finish()
 summary=dict(format='hier-v8-reviewed-aggregate-v1',performance=performance,geometry=cover,radial=radial,evolution=evo_rows,encoder_gradients=proberows,independent_shape=shaperows,paired_shape_differences=paired,proxy_class_affinity=affinity,hottest_four=hotrows,temporal_binding=temporal,cap_group_gradients=caprows,proxy_lifetimes=life,B0_progress=progress,source_files_unchanged=all(sha(p)==before[str(p)] for p in sources),optimizer_updates=0,new_test_forwards=0,figure_count=18,checked_report_links=links,limits=['Single seed; cross-version geometry/protocol changes are confounded','B0-32 still training at acquisition; no final result imputed','Frozen clean/evalBN diagnostics are not real optimizer trajectories','Original datasets best-only, different D/classes/protocols'])
 assert summary['source_files_unchanged'];dump(O/'summary.json',summary);dump(O/'private_provenance.json',dict(inputs=before,script_sha256=sha(__file__)))
 print(json.dumps(dict(status='completed',output=str(O),figures=18,links=links,source_files_unchanged=True),ensure_ascii=False))
if __name__=='__main__':main()
