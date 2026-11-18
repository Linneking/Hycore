"""Verify original mesh names and publish a private, readable audit delivery."""
import argparse
import csv
from collections import Counter
from pathlib import Path
import numpy as np
from audit import NAMES, data_arrays, load_json, write, source
from analyze import csv_file
from geometry import normalize, rows


def main():
    p=argparse.ArgumentParser()
    for name in ('inputs','predictions','analysis','geometry','data','out'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    arrays=np.load(a.inputs/'identity.npz');mapping={};map_sha={}
    for split in ('train','test'):
        for identifier,label in zip(arrays[split+'_ids'],arrays[split+'_labels']):
            _,shard,row=str(identifier).split('/')
            filename=shard.replace(split,split+'_').replace('.h5','_id2file.json')
            if filename not in mapping:
                mapping[filename]=load_json(a.data/filename);map_sha[filename]=source.digest(a.data/filename)
            mesh=mapping[filename][int(row)]
            assert mesh.split('/')[0]==NAMES[int(label)],(identifier,mesh,label)
            mapping[str(identifier)]=mesh
    identity={str(k):v for k,v in mapping.items() if str(k).startswith(('train/','test/'))}
    csv_file(a.out/'object_identity.csv',(dict(id=k,mesh=v) for k,v in identity.items()))
    for path in a.analysis.glob('*.csv'):
        content=rows(path)
        if not content:continue
        for row in content:
            for key in ('id','control_id'):
                if key in row and row[key] in identity:row[key.replace('id','mesh')]=identity[row[key]]
        csv_file(a.out/path.name,content)
    neigh=rows(a.geometry/'neighbours.csv')
    for r in neigh:r.update(test_mesh=identity[r['test_id']],train_mesh=identity[r['train_id']])
    csv_file(a.out/'neighbours.csv',neigh)
    for filename in ('neighbour_exposure.csv','matched_controls.csv'):
        rr=rows(a.geometry/filename)
        for r in rr:
            r['mesh']=identity[r['id']]
            if 'control_id' in r:r['control_mesh']=identity[r['control_id']]
        csv_file(a.out/filename,rr)

    objects=rows(a.analysis/'best_test_objects.csv');byid={r['id']:r for r in objects}
    stable=np.load(a.analysis/'stability.npz');y=arrays['test_labels']
    persistent=[]
    for i,identifier in enumerate(arrays['test_ids']):
        if stable['persistent_sh'][i] or stable['persistent_cb'][i]:
            r=dict(byid[str(identifier)],mesh=identity[str(identifier)])
            for arm in ('TS06','TS30','TS31'):r[arm+'_late_predictions']='|'.join(NAMES[int(v)] for v in stable[arm][:,i])
            persistent.append(r)
    csv_file(a.out/'persistent_candidates.csv',persistent)
    summary=load_json(a.analysis/'summary.json');geo=load_json(a.geometry/'summary.json')
    common_all=np.all((stable['TS06']!=y)&(stable['TS30']!=y)&(stable['TS31']!=y),axis=0)
    csv_file(a.out/'all_three_wrong_all_six_late.csv',(dict(id=str(i),mesh=identity[str(i)],label=NAMES[int(c)]) for i,c,yes in zip(arrays['test_ids'],y,common_all) if yes))
    summary['all_three_wrong_all_six_late']=int(common_all.sum())
    summary['mesh_mapping_verified_objects']=len(identity);summary['mesh_mapping_sha256']=map_sha

    # Show all persistent candidates, then one canonical representative of each
    # top three net-loss and top three common-error classes. No visual selection.
    chosen=[r['id'] for r in persistent]
    cls=rows(a.analysis/'best_classes.csv')
    for c in sorted(cls,key=lambda r:(-int(r['net_sh_advantage']),r['name']))[:3]:
        r=next(r for r in objects if r['label']==c['name'] and r['group']=='sh_only');chosen.append(r['id'])
    hard=Counter(r['label'] for r in objects if r['all_three_wrong']=='True')
    for c,_ in hard.most_common(3):
        r=next(r for r in objects if r['label']==c and r['all_three_wrong']=='True');chosen.append(r['id'])
    chosen=list(dict.fromkeys(chosen))
    data=data_arrays(a.data,load_json(a.inputs/'manifest.json')['data'])
    tr=normalize(data['train'][0]);te=normalize(data['test'][0])
    tid={str(v):i for i,v in enumerate(arrays['train_ids'])};qid={str(v):i for i,v in enumerate(arrays['test_ids'])}
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plates=[]
    for identifier in chosen:
        i=qid[identifier];r=byid[identifier]
        nn=sorted([n for n in neigh if n['test_id']==identifier],key=lambda n:int(n['rank']))[:3]
        assert len(nn)==3
        clouds=[te[i],*[tr[tid[n['train_id']]] for n in nn]]
        bound=max(float(np.abs(x).max()) for x in clouds)*1.05
        fig=plt.figure(figsize=(15,7.5))
        for col,cloud in enumerate(clouds):
            title=(f"TEST: {identity[identifier]}\nSH={r['TS06_prediction']} CB={r['TS30_prediction']} CB32x300={r['TS31_prediction']}") if col==0 else f"TRAIN: {nn[col-1]['train_mesh']}\nSH={nn[col-1]['sh_best_count']} CB={nn[col-1]['cb_best_count']} draws; d={float(nn[col-1]['chamfer256']):.4f}"
            for view,(az,el) in enumerate(((35,20),(125,15))):
                ax=fig.add_subplot(2,4,view*4+col+1,projection='3d')
                ax.scatter(cloud[:,0],cloud[:,2],cloud[:,1],c=cloud[:,1],s=2,cmap='viridis',alpha=.85)
                ax.set_xlim(-bound,bound);ax.set_ylim(-bound,bound);ax.set_zlim(-bound,bound)
                ax.set_box_aspect((1,1,1));ax.view_init(el,az);ax.set_axis_off()
                if view==0:ax.set_title(title,fontsize=8)
        sh=int(stable['sh_only_count'][i]);cb=int(stable['cb_only_count'][i])
        fig.suptitle(f"{identifier} | true={r['label']} | best group={r['group']} | late SH-only={sh}/6, CB-only={cb}/6",fontsize=11)
        fig.text(.5,.01,'1024 clean input points; equal physical axis scales. Neighbours use fixed-256-point Chamfer. Draw counts stop at each best.',ha='center',fontsize=8)
        fig.tight_layout(rect=(0,.03,1,.95));filename=f'case_{i:04d}.png';fig.savefig(a.out/filename,dpi=150);plt.close(fig)
        plates.append(dict(id=identifier,mesh=identity[identifier],label=r['label'],group=r['group'],png=filename,
            late_sh_only=sh,late_cb_only=cb,SH_prediction=r['TS06_prediction'],CB_prediction=r['TS30_prediction'],
            nearest_train=[n['train_mesh'] for n in nn],nearest_sh_counts=[int(n['sh_best_count']) for n in nn],
            nearest_cb_counts=[int(n['cb_best_count']) for n in nn]))
    summary['gallery']=plates;summary['geometry']=geo
    write(a.out/'summary.json',summary)
    html='<meta charset="utf-8"><title>TS06 TS30 audit</title><style>body{font-family:sans-serif;margin:24px}img{width:100%;max-width:1600px}section{margin-bottom:30px}</style><h1>TS06 SH32 / TS30 CB32 object audit</h1><p>All four persistent candidates, representatives of top three net-loss classes, and top three all-H0-error classes. Fixed data-based selection; 1024-point plots with equal physical axis scales. Counts at best checkpoints.</p>'
    for r in plates:html+=f'<section><h2>{r["mesh"]} | {r["group"]}</h2><img src="{r["png"]}"></section>'
    (a.out/'gallery.html').write_text(html,encoding='utf-8')
    print('Delivery complete',len(identity),'verified mesh names',len(plates),'plates',flush=True)


if __name__=='__main__':main()
