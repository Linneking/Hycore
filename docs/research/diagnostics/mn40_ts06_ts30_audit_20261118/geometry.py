"""Independent geometric neighbours, matched controls, and point-cloud panels."""
import argparse
import csv
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree

from audit import NAMES, data_arrays, load_json, write
from analyze import csv_file


def rows(path):
    with Path(path).open(encoding="utf-8-sig") as f: return list(csv.DictReader(f))


def normalize(points):
    x = points - points.mean(1, keepdims=True)
    return x / np.sqrt((x*x).sum(2).mean(1))[:, None, None]


def descriptors(x):
    cov = np.einsum("npi,npj->nij", x, x) / x.shape[1]
    eig = np.linalg.eigvalsh(cov)
    radial = np.quantile(np.linalg.norm(x, axis=2), np.linspace(.05,.95,10), axis=1).T
    axis = np.quantile(x, [.05,.25,.5,.75,.95], axis=1).transpose(1,0,2).reshape(len(x),-1)
    return np.concatenate((eig, radial, axis, np.ptp(x,axis=1)),axis=1)


def main():
    p=argparse.ArgumentParser();p.add_argument("--inputs",type=Path,required=True)
    p.add_argument("--predictions",type=Path,required=True);p.add_argument("--analysis",type=Path,required=True)
    p.add_argument("--data",type=Path,required=True);p.add_argument("--out",type=Path,required=True)
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    data=data_arrays(args.data,load_json(args.inputs/'manifest.json')['data'])
    tr,yt,train_ids=data['train'];te,y,test_ids=data['test']
    tr=normalize(tr);te=normalize(te)
    dt,de=descriptors(tr),descriptors(te);scale=np.maximum(dt.std(0),1e-6)
    dt/=scale;de/=scale
    sub=np.linspace(0,1023,256,dtype=int)
    trsub=tr[:,sub];tesub=te[:,sub]
    trees=[cKDTree(x) for x in trsub]
    s=np.load(args.inputs/'TS06_sampling.npz')['best_counts'];c=np.load(args.inputs/'TS30_sampling.npz')['best_counts']
    class_s=np.array([s[yt==i].mean() for i in range(40)])
    class_c=np.array([c[yt==i].mean() for i in range(40)])
    class_sd=np.array([max(c[yt==i].std(),1e-12) for i in range(40)])
    records=rows(args.analysis/'best_test_objects.csv')
    stable=np.load(args.analysis/'stability.npz')
    target=np.array([i for i,r in enumerate(records) if r['group']!='both_correct' or stable['persistent_sh'][i] or stable['persistent_cb'][i]])
    control={}
    for i in target:
        eligible=np.array([j for j,r in enumerate(records) if y[j]==y[i] and r['group']=='both_correct'])
        if len(eligible):control[int(i)]=int(eligible[np.argmin(((de[eligible]-de[i])**2).sum(1))])
    query=np.unique(np.r_[target,list(control.values())]).astype(int)
    neighbours={};table=[]
    for i in query:
        eligible=np.flatnonzero(yt==y[i])
        candidate=eligible[np.argsort(((dt[eligible]-de[i])**2).sum(1),kind='stable')[:20]]
        qt=cKDTree(tesub[i]);dist=[]
        for j in candidate:
            dist.append(float(np.mean(trees[j].query(tesub[i],workers=1)[0]**2)+np.mean(qt.query(trsub[j],workers=1)[0]**2)))
        order=np.argsort(dist,kind='stable')[:10];indices=candidate[order]
        neighbours[int(i)]=(indices,np.asarray(dist)[order])
        for rank,(j,d) in enumerate(zip(indices,np.asarray(dist)[order]),1):
            table.append(dict(test_id=str(test_ids[i]),class_name=NAMES[int(y[i])],group=records[i]['group'],
                persistent_sh=bool(stable['persistent_sh'][i]),persistent_cb=bool(stable['persistent_cb'][i]),
                rank=rank,train_id=str(train_ids[j]),chamfer256=float(d),sh_best_count=int(s[j]),cb_best_count=int(c[j]),
                cb_class_relative=float(c[j]/class_c[y[i]]),cb_within_class_z=float((c[j]-class_c[y[i]])/class_sd[y[i]])))
    csv_file(args.out/'neighbours.csv',table)
    samples=[]
    for i in query:
        ix=neighbours[int(i)][0]
        row=dict(id=str(test_ids[i]),class_name=NAMES[int(y[i])],group=records[i]['group'],is_target=bool(i in target),
            persistent_sh=bool(stable['persistent_sh'][i]),persistent_cb=bool(stable['persistent_cb'][i]))
        for k in (5,10):
            row.update({f'k{k}_sh_mean':float(s[ix[:k]].mean()),f'k{k}_cb_mean':float(c[ix[:k]].mean()),
                f'k{k}_cb_class_relative':float(c[ix[:k]].mean()/class_c[y[i]]),
                f'k{k}_cb_z':float(((c[ix[:k]]-class_c[y[i]])/class_sd[y[i]]).mean())})
        samples.append(row)
    csv_file(args.out/'neighbour_exposure.csv',samples)
    byid={r['id']:r for r in samples};summary={}
    for group in ('sh_only','cb_only','both_wrong','both_correct'):
        ss=[r for r in samples if r['group']==group]
        summary[group]=dict(n=len(ss),k5_cb_class_relative=float(np.mean([r['k5_cb_class_relative'] for r in ss])) if ss else None,
            k5_cb_z=float(np.mean([r['k5_cb_z'] for r in ss])) if ss else None)
    paired=[]
    for i,j in control.items():
        if records[i]['group'] not in ('sh_only','cb_only','both_wrong'):continue
        a,b=byid[str(test_ids[i])],byid[str(test_ids[j])]
        paired.append(dict(id=str(test_ids[i]),control_id=str(test_ids[j]),group=records[i]['group'],class_name=NAMES[int(y[i])],
            delta_k5_cb_z=a['k5_cb_z']-b['k5_cb_z'],delta_k10_cb_z=a['k10_cb_z']-b['k10_cb_z']))
    csv_file(args.out/'matched_controls.csv',paired)
    summary['matched_controls']={}
    for group in ('sh_only','cb_only','both_wrong'):
        ss=[r for r in paired if r['group']==group]
        per=defaultdict(list)
        for row in ss:per[row['class_name']].append(row['delta_k5_cb_z'])
        vals=np.array([np.mean(v) for v in per.values()]);rng=np.random.default_rng(4780)
        boot=np.array([rng.choice(vals,len(vals),replace=True).mean() for _ in range(4000)]) if len(vals) else np.array([0.])
        summary['matched_controls'][group]=dict(n=len(ss),class_count=len(vals),mean_delta_z=float(np.mean([r['delta_k5_cb_z'] for r in ss])) if ss else None,
            class_mean_delta_z=float(vals.mean()) if len(vals) else None,class_bootstrap_ci95=np.quantile(boot,[.025,.975]).tolist(),
            mean_delta_k10_z=float(np.mean([r['delta_k10_cb_z'] for r in ss])) if ss else None)
    summary['method']='same-class top20 independent fixed-shape descriptor candidates, reranked by symmetric squared Chamfer on fixed 256 points; approximate candidate neighbours, not exhaustive global nearest'
    summary['query_count']=len(query);summary['target_count']=len(target);summary['unique_control_count']=len(set(control.values()))

    # Fixed selection order: largest class net loss/gain, then canonical ID.
    classes=rows(args.analysis/'best_classes.csv');net={r['name']:int(r['net_sh_advantage']) for r in classes}
    chosen=[]
    for group,persistent in (('sh_only',stable['persistent_sh']),('cb_only',stable['persistent_cb']),('both_wrong',stable['persistent_both_wrong'])):
        pool=[i for i in target if records[i]['group']==group and persistent[i]]
        pool.sort(key=lambda i:((-net[NAMES[int(y[i])]]) if group!='cb_only' else net[NAMES[int(y[i])]],str(test_ids[i])))
        chosen.extend(pool[:4 if group!='both_wrong' else 3])
    gallery=[]
    for i in chosen:
        ix,dist=neighbours[int(i)];show=[None,*ix[:3]]
        fig=plt.figure(figsize=(15,7.5))
        for col,j in enumerate(show):
            cloud=tesub[i] if j is None else trsub[j]
            title=(str(test_ids[i])+f"\n{records[i]['group']}: SH={records[i]['TS06_prediction']} CB={records[i]['TS30_prediction']}") if j is None else f"{train_ids[j]}\nSH count={s[j]}, CB count={c[j]}, d={dist[col-1]:.4f}"
            for view,(az,el) in enumerate(((35,20),(125,15))):
                ax=fig.add_subplot(2,4,view*4+col+1,projection='3d')
                ax.scatter(cloud[:,0],cloud[:,2],cloud[:,1],c=cloud[:,1],s=4,cmap='viridis',alpha=.85)
                ax.view_init(el,az);ax.set_box_aspect((1,1,1));ax.set_axis_off()
                if view==0:ax.set_title(title,fontsize=8)
        filename=f"case_{i:04d}.png";fig.suptitle(f"True class: {NAMES[int(y[i])]} | late SH-only={stable['sh_only_count'][i]}/6 CB-only={stable['cb_only_count'][i]}/6",fontsize=12)
        fig.tight_layout();fig.savefig(args.out/filename,dpi=150);plt.close(fig)
        gallery.append(dict(test_id=str(test_ids[i]),class_name=NAMES[int(y[i])],group=records[i]['group'],png=filename,
            sh_only_epochs=int(stable['sh_only_count'][i]),cb_only_epochs=int(stable['cb_only_count'][i]),
            nearest_train=[str(train_ids[j]) for j in ix[:3]],nearest_cb_counts=c[ix[:3]].tolist(),nearest_sh_counts=s[ix[:3]].tolist()))
    summary['gallery']=gallery
    html='<meta charset="utf-8"><title>TS06 TS30 object audit</title><style>body{font-family:sans-serif;margin:24px}img{width:100%;max-width:1600px}section{margin-bottom:36px}</style><h1>TS06 / TS30 stable disagreements and training neighbours</h1><p>Frozen best models. Counts stop at each best checkpoint. Independent candidate-filtered geometry; no causal labels.</p>'
    for r in gallery:html+=f'<section><h2>{r["test_id"]} | {r["class_name"]} | {r["group"]}</h2><img src="{r["png"]}"></section>'
    (args.out/'gallery.html').write_text(html,encoding='utf-8')
    write(args.out/'summary.json',summary)

    # Scientific figures use the actual counters and paired predictions.
    fig,axes=plt.subplots(1,2,figsize=(15,6))
    ratios=np.log2([float(r['cb_to_sh_exposure_ratio']) for r in classes]);delta=np.array([int(r['net_sh_advantage']) for r in classes])
    ax=axes[0];ax.scatter(ratios,[float(r['delta_oa_class']) for r in classes],s=[int(r['test_size']) for r in classes],c=delta,cmap='coolwarm',edgecolors='black',linewidths=.3)
    ax.axhline(0,color='gray',lw=.7);ax.axvline(0,color='gray',lw=.7);ax.set_xlabel('log2(CB / SH training exposure per object, best cutoff)');ax.set_ylabel('Test class OA: SH - CB (percentage points)')
    for j in np.argsort(np.abs(delta))[-10:]:ax.annotate(classes[j]['name'],(ratios[j],float(classes[j]['delta_oa_class'])),fontsize=8)
    ax=axes[1];order=np.argsort(delta);ax.barh(np.arange(40),delta[order],color=np.where(delta[order]>0,'#b44b41','#407f9d'));ax.set_yticks(np.arange(40));ax.set_yticklabels([classes[j]['name'] for j in order],fontsize=7);ax.set_xlabel('Net correct objects: SH - CB (total = 12)')
    fig.tight_layout();fig.savefig(args.out/'class_exposure_error.png',dpi=170);plt.close(fig)
    print('Geometry completed',len(query),'queries',len(gallery),'panels',flush=True)


if __name__=='__main__':main()
