"""Matched read-only DGCNN/TAN e200 evaluation and table/lamp diagnosis.

All private paths/object IDs/predictions stay in the new server output directory.
Public summary contains only aggregates and provenance hashes. Never updates
parameters, BN buffers, optimizer state or checkpoints.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

NAMES = ('airplane','bag','cap','car','chair','earphone','guitar','knife',
         'lamp','laptop','motorbike','mug','pistol','rocket','skateboard','table')
STARTS = (0,4,6,8,12,16,19,22,24,28,30,36,38,41,44,47)
COUNTS = (4,2,2,4,4,3,3,2,4,2,6,2,3,3,3,3)
FOCUS = (8,15)
FOCUS_PARTS = (24,25,26,27,47,48,49)
BINS = ((1,16),(17,64),(65,256),(257,2048))


def ratio(a,b):
    return float(a/b) if b else None


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def write(path,value):
    path=Path(path)
    tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    os.replace(tmp,path)


def object_parts(pred,target,cat):
    rows=[]
    for k in range(STARTS[cat],STARTS[cat]+COUNTS[cat]):
        gt=int((target==k).sum()); pr=int((pred==k).sum())
        tp=int(((target==k)&(pred==k)).sum()); union=gt+pr-tp
        rows.append(dict(part=k,gt=gt,pred=pr,tp=tp,iou=tp/union if union else 1.,
                         present=bool(gt),absent_both=union==0))
    return rows


def boundary_mask(points,target,k=16):
    """Spatial 16-NN semantic boundary, GT-only observational definition."""
    from scipy.spatial import cKDTree
    n=len(points)
    if n<2:return np.zeros(n,dtype=bool)
    # Query k+1 to remove the self index explicitly even when points duplicate.
    neighbours=cKDTree(points).query(points,k=min(k+1,n),workers=1)[1]
    if neighbours.ndim==1:neighbours=neighbours[:,None]
    masks=[]
    for i,indices in enumerate(neighbours):
        other=indices[indices!=i][:k]
        masks.append(bool(np.any(target[other]!=target[i])))
    return np.asarray(masks,dtype=bool)


class FocusStatistics:
    def __init__(self):
        self.confusion={str(c):np.zeros((50,50),dtype=np.int64) for c in FOCUS}
        self.rows=[]
        self.regions={str(c):{r:dict(points=0,errors=0) for r in ('boundary','interior')} for c in FOCUS}
        self.norms={str(c):{r:dict(points=0,sum=0.,square_sum=0.) for r in ('correct','incorrect')} for c in FOCUS}

    def update(self,pred,target,cat,idx,object_id,points=None,tangent_norm=None):
        cat=int(cat)
        parts=object_parts(pred,target,cat)
        row=dict(index=int(idx),id=str(object_id),category=cat,shape_miou=float(np.mean([r['iou'] for r in parts])),parts=parts)
        if cat not in FOCUS:return row
        self.confusion[str(cat)]+=np.bincount(target*50+pred,minlength=2500).reshape(50,50)
        if points is not None:
            boundary=boundary_mask(points,target)
            for label,mask in [('boundary',boundary),('interior',~boundary)]:
                a=self.regions[str(cat)][label]
                a['points']+=int(mask.sum());a['errors']+=int((pred[mask]!=target[mask]).sum())
            row['boundary_points']=int(boundary.sum())
            row['boundary_errors']=int((pred[boundary]!=target[boundary]).sum())
            row['interior_errors']=int((pred[~boundary]!=target[~boundary]).sum())
        if tangent_norm is not None:
            for label,mask in [('correct',pred==target),('incorrect',pred!=target)]:
                z=np.asarray(tangent_norm[mask],dtype=np.float64)
                a=self.norms[str(cat)][label]
                a['points']+=len(z);a['sum']+=float(z.sum());a['square_sum']+=float((z*z).sum())
        self.rows.append(row)
        return row

    def summary(self):
        result={}
        for cat in FOCUS:
            rows=[r for r in self.rows if r['category']==cat]
            conf=self.confusion[str(cat)]
            parts={}
            for k in range(STARTS[cat],STARTS[cat]+COUNTS[cat]):
                allparts=[next(p for p in r['parts'] if p['part']==k) for r in rows]
                present=[p for p in allparts if p['gt']>0]
                gt=int(conf[k,:].sum());pr=int(conf[:,k].sum());tp=int(conf[k,k])
                bins={}
                for lo,hi in BINS:
                    pp=[p for p in present if lo<=p['gt']<=hi]
                    bins[f'{lo}_{hi}']={'gt_present_objects':len(pp),
                        'zero_recall_objects':sum(p['tp']==0 for p in pp),
                        'no_prediction_objects':sum(p['pred']==0 for p in pp),
                        'mean_iou_gt_present':ratio(sum(p['iou'] for p in pp),len(pp)),
                        'pooled_recall':ratio(sum(p['tp'] for p in pp),sum(p['gt'] for p in pp))}
                parts[str(k)]={'gt_points':gt,'pred_points_within_category':pr,'tp':tp,
                    'recall':ratio(tp,gt),'precision':ratio(tp,pr),'pooled_iou':ratio(tp,gt+pr-tp),
                    'gt_present_objects':len(present),'absent_both_objects':sum(p['absent_both'] for p in allparts),
                    'zero_recall_objects_given_present':sum(p['tp']==0 for p in present),
                    'no_prediction_objects_given_present':sum(p['pred']==0 for p in present),
                    'mean_iou_gt_present':ratio(sum(p['iou'] for p in present),len(present)),
                    'mean_iou_all_objects_absent_both_is_one':ratio(sum(p['iou'] for p in allparts),len(allparts)),
                    'gt_point_count_bins':bins}
            regions={r:{**a,'error_rate':ratio(a['errors'],a['points'])} for r,a in self.regions[str(cat)].items()}
            norms={r:{'points':a['points'],'mean':ratio(a['sum'],a['points']),
                      'std':float(np.sqrt(max(0.,a['square_sum']/a['points']-(a['sum']/a['points'])**2))) if a['points'] else None}
                   for r,a in self.norms[str(cat)].items()}
            legal=list(range(STARTS[cat],STARTS[cat]+COUNTS[cat]))
            result[NAMES[cat]]={'objects':len(rows),'parts':parts,'confusion_gt_rows_pred_columns_50x50':conf.tolist(),
                'illegal_prediction_points':int(conf[:,[k for k in range(50) if k not in legal]].sum()),
                'boundary_definition':'any_different_GT_among_16_spatial_nearest_other_points',
                'regions':regions,'tangent_norm_by_correctness':norms}
        return result


def paired_summary(left,right):
    a={r['index']:r for r in left};b={r['index']:r for r in right}
    if a.keys()!=b.keys():raise ValueError('Paired evaluation object sets differ')
    output={}
    for cat in FOCUS:
        indices=[i for i,r in a.items() if r['category']==cat]
        if any(a[i]['id']!=b[i]['id'] or a[i]['category']!=b[i]['category'] for i in indices):
            raise ValueError('Paired object identities differ')
        if not indices:
            output[NAMES[cat]]={'objects':0,'mean_iou_difference':None,'median_iou_difference':None,
                'improved_objects':0,'worse_objects':0,'tied_objects':0,'difference_quantiles':{},'parts':{}}
            continue
        d=np.asarray([b[i]['shape_miou']-a[i]['shape_miou'] for i in indices])
        parts={}
        for k in range(STARTS[cat],STARTS[cat]+COUNTS[cat]):
            pairs=[(next(p for p in a[i]['parts'] if p['part']==k),next(p for p in b[i]['parts'] if p['part']==k)) for i in indices]
            if any(x['gt']!=y['gt'] for x,y in pairs):raise ValueError('GT point counts differ')
            pairs=[(x,y) for x,y in pairs if x['gt']>0]
            changes=[y['iou']-x['iou'] for x,y in pairs]
            parts[str(k)]={'gt_present_objects':len(pairs),'mean_iou_difference_given_gt_present':ratio(sum(changes),len(changes)),
                'zero_recall_recovered_objects':sum(x['tp']==0 and y['tp']>0 for x,y in pairs),
                'zero_recall_newly_lost_objects':sum(x['tp']>0 and y['tp']==0 for x,y in pairs)}
        output[NAMES[cat]]={'objects':len(indices),'mean_iou_difference':float(d.mean()),'median_iou_difference':float(np.median(d)),
            'improved_objects':int((d>1e-12).sum()),'worse_objects':int((d < -1e-12).sum()),'tied_objects':int((np.abs(d)<=1e-12).sum()),
            'difference_quantiles':dict(zip(['p10','p25','p50','p75','p90'],map(float,np.quantile(d,[.1,.25,.5,.75,.9])))),
            'parts':parts}
    return output


def metric_diff(actual,expected):
    diffs={k:actual[k]-expected[k] for k in ('instance_miou','class_miou','point_accuracy')}
    diffs['per_class']={k:actual['per_class'][k]['miou']-expected['per_class'][k]['miou'] for k in NAMES}
    maximum=max(abs(v) for k,v in diffs.items() if k!='per_class')
    maximum=max(maximum,max(abs(v) for v in diffs['per_class'].values()))
    return {'differences':diffs,'max_absolute_difference':maximum,'within_1e_minus5':maximum<=1e-5}


def run(args):
    out=Path(args.out)
    out.mkdir(parents=True,exist_ok=False)
    try:
        os.environ['CUDA_VISIBLE_DEVICES']=str(args.gpu)
        os.environ['OMP_NUM_THREADS']='4';os.environ['MKL_NUM_THREADS']='4'
        sys.path.insert(0,str(Path(args.source).resolve(strict=True)))
        import torch
        import h5py
        from segmentation_ShapeNetPart.hycore_seg_v1.train import ensure_idle
        from segmentation_ShapeNetPart.hycore_seg_v1.metrics import SegMetrics
        from segmentation_ShapeNetPart.hycore_seg_v1.model import DGCNNPartSeg
        from segmentation_ShapeNetPart.hycore_seg_v1.utils import state_hash,seed_all
        from segmentation_ShapeNetPart.hycore_seg_v10.model import NativeTail2
        gpu=ensure_idle(args.gpu)
        torch.set_num_threads(4)
        source=Path(args.source).resolve(strict=True)
        commit=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()
        if subprocess.check_output(['git','-C',str(source),'status','--porcelain','--','segmentation_ShapeNetPart'],text=True).strip():
            raise ValueError('Model source must be clean')
        data=Path(args.data).resolve(strict=True)
        manifest_hash=sha(data/'manifest.json')
        dm=json.loads((data/'manifest.json').read_text())
        data_hash=sha(data/'test.h5')
        if data_hash!=dm['splits']['test']['file_sha256']:raise ValueError('Test data hash differs')
        entries=json.loads(Path(args.runs_json).read_text(encoding='utf-8-sig'))
        if len(entries)!=2 or [e['kind'] for e in entries]!=['dgcnn','tan']:
            raise ValueError('Require ordered dgcnn then tan entries for paired comparison')
        names=[e['name'] for e in entries]
        if len(set(names))!=2 or any(Path(n).name!=n or n in ('.','..') for n in names):raise ValueError('Unsafe run names')
        write(out/'private_manifest.json',{'args':vars(args),'gpu':gpu,'source_commit':commit,'entries':entries})
        allrows={};public={'source_commit':commit,'script_sha256':sha(__file__),'data_manifest_sha256':manifest_hash,'test_sha256':data_hash,'runs':{}}
        with h5py.File(data/'test.h5','r') as f:
            cats=f['label'][:].reshape(-1).astype(np.int64)
            ids=f['ids'].asstr()[:]
            if len(cats)!=2874:raise ValueError('Expected full2874 test objects')
            for entry in entries:
                ckpath=Path(entry['checkpoint']).resolve(strict=True)
                ckhash=sha(ckpath)
                ck=torch.load(ckpath,map_location='cpu',weights_only=False)
                if ck['epoch']!=200:raise ValueError('Only fixed epoch200 checkpoints allowed')
                if ck['data_manifest_sha256']!=manifest_hash:raise ValueError('Checkpoint data identity differs')
                cfg=ck['config'];seed_all(int(cfg.get('seed',4780)))
                kwargs={'k':int(cfg.get('k',40)),'dropout':float(cfg.get('dropout',.5))}
                model=DGCNNPartSeg(head='euclidean',**kwargs) if entry['kind']=='dgcnn' else NativeTail2('h',**kwargs)
                model.load_state_dict(ck['model'],strict=True)
                updates=ck.get('updates')
                if updates!=87400:raise ValueError('Expected87400 checkpoint updates')
                before=state_hash(model.state_dict());del ck;gc.collect()
                model=model.cuda().eval();hooked={}
                hook=None
                if entry['kind']=='tan':
                    def capture(module,inputs,output):hooked['mu']=output.detach()
                    hook=model.emb2.register_forward_hook(capture)
                    weight=model.mlr.weight.detach()
                    readout_weights={'effective_row_norms':weight.norm(dim=1).cpu().tolist(),
                        'raw_weight_row_norms':model.mlr._weight.detach().norm(dim=1).cpu().tolist(),
                        'raw_bias_row_norms':model.mlr._bias.detach().norm(dim=1).cpu().tolist(),
                        'effective_row_norm_min':float(weight.norm(dim=1).min()),'effective_row_norm_max':float(weight.norm(dim=1).max())}
                else:readout_weights=None
                meter=SegMetrics();stats=FocusStatistics();records=[];predictions=[]
                cosine_meter=SegMetrics();cosine_stats=FocusStatistics();cosine_records=[];cosine_predictions=[]
                directory=out/entry['name'];directory.mkdir()
                began=time.monotonic()
                with (directory/'private_objects.jsonl').open('x',encoding='utf-8') as objects,torch.inference_mode():
                    for offset in range(0,len(cats),args.batch_size):
                        stop=min(offset+args.batch_size,len(cats));indices=np.arange(offset,stop)
                        points=f['data'][offset:stop,:2048].astype(np.float32)
                        target=f['pid'][offset:stop,:2048].astype(np.int64)
                        tensor=torch.from_numpy(points).cuda();cat=cats[offset:stop]
                        logits=model(tensor.transpose(1,2).contiguous(),torch.nn.functional.one_hot(torch.from_numpy(cat).cuda(),16).float()).transpose(1,2).contiguous()
                        if logits.shape!=(len(indices),2048,50) or not bool(torch.isfinite(logits).all()):raise FloatingPointError('Invalid model scores')
                        pred=logits.argmax(-1).cpu().numpy();meter.update(pred,target,cat)
                        norms=None
                        cosine_pred=None
                        if entry['kind']=='tan':
                            tangent=model.manifold.logmap0(hooked.pop('mu'))
                            norms=tangent.norm(dim=-1).cpu().numpy()
                            cosine_scores=torch.nn.functional.linear(
                                torch.nn.functional.normalize(tangent,dim=-1),
                                torch.nn.functional.normalize(model.mlr._weight.detach(),dim=-1))
                            cosine_pred=cosine_scores.argmax(-1).cpu().numpy()
                            cosine_meter.update(cosine_pred,target,cat)
                            cosine_predictions.append(cosine_pred.astype(np.uint8))
                        for b,i in enumerate(indices):
                            row=stats.update(pred[b],target[b],cat[b],i,ids[i],points=points[b] if cat[b] in FOCUS else None,
                                             tangent_norm=norms[b] if norms is not None else None)
                            records.append(row);objects.write(json.dumps(row,allow_nan=False)+'\n')
                            if cosine_pred is not None:
                                cosine_records.append(cosine_stats.update(cosine_pred[b],target[b],cat[b],i,ids[i],
                                    points=points[b] if cat[b] in FOCUS else None,tangent_norm=norms[b]))
                        predictions.append(pred.astype(np.uint8))
                        if offset==0 or offset//args.batch_size%25==0:
                            write(out/'status.json',{'state':'evaluating','run':entry['name'],'objects_done':stop,'total':len(cats)})
                if hook:hook.remove()
                metrics=meter.result()
                expected=entry.get('expected_metrics')
                if expected is None:
                    rows=[json.loads(s) for s in (ckpath.parent/'metrics.jsonl').read_text().splitlines() if s]
                    expected=next(r['test'] for r in rows if r['epoch']==200)
                diff=metric_diff(metrics,expected)
                if not diff['within_1e_minus5']:raise AssertionError('Fixed checkpoint metrics do not match historical e200')
                after=state_hash(model.state_dict())
                if before!=after or sha(ckpath)!=ckhash:raise AssertionError('Read-only identity changed')
                np.savez_compressed(directory/'private_predictions.npz',predictions=np.concatenate(predictions),categories=cats)
                result={'checkpoint_sha256':ckhash,'checkpoint_epoch':200,'checkpoint_updates':updates,
                    'model_before_sha256':before,'model_after_sha256':after,'model_unchanged':True,'metrics':metrics,
                    'historical_metric_check':diff,'focus':stats.summary(),'readout_weights':readout_weights,
                    'elapsed_seconds':time.monotonic()-began,'prediction_rule':'raw_50class_argmax','model_mode':'eval','grad_mode':'inference_mode'}
                if entry['kind']=='tan':
                    cosine_result={'interpretation':'frozen_same_features_no_training_no_label_dependent_score',
                        'formula':'normalize(LOG0(mu)) dot normalize(raw_mlr_weight); fixed positive scale irrelevant to argmax',
                        'metrics':cosine_meter.result(),'focus':cosine_stats.summary(),
                        'paired_cosine_minus_original_tan':paired_summary(records,cosine_records),
                        'normalized_effective_vs_raw_weight_max_abs':float((
                            torch.nn.functional.normalize(model.mlr.weight.detach(),dim=-1)
                            -torch.nn.functional.normalize(model.mlr._weight.detach(),dim=-1)).abs().max())}
                    public['frozen_cosine_counterfactual']=cosine_result
                    write(directory/'frozen_cosine_counterfactual.json',cosine_result)
                    with (directory/'private_cosine_objects.jsonl').open('x') as stream:
                        for row in cosine_records:stream.write(json.dumps(row,allow_nan=False)+'\n')
                    np.savez_compressed(directory/'private_cosine_predictions.npz',predictions=np.concatenate(cosine_predictions),categories=cats)
                public['runs'][entry['name']]=result;allrows[entry['name']]=records
                write(directory/'summary.json',result);write(out/'summary.json',public)
                print(json.dumps({'completed_model':entry['name'],'objects':metrics['objects'],'instance_miou':metrics['instance_miou'],'class_miou':metrics['class_miou'],'historical_match':True}),flush=True)
                del model;gc.collect();torch.cuda.empty_cache()
        public['paired_second_minus_first']=paired_summary(allrows[names[0]],allrows[names[1]])
        public['complete']=True
        write(out/'summary.json',public);write(out/'status.json',{'state':'complete','runs':names})
        print(json.dumps({'complete':True,'source_commit':commit,'paired_second_minus_first':public['paired_second_minus_first']}),flush=True)
    except Exception as exc:
        write(out/'failure.json',{'error':repr(exc)})
        write(out/'status.json',{'state':'failed','error':repr(exc)})
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',required=True);p.add_argument('--data',required=True)
    p.add_argument('--runs-json',required=True);p.add_argument('--out',required=True)
    p.add_argument('--gpu',type=int,required=True);p.add_argument('--batch-size',type=int,default=16)
    a=p.parse_args()
    if a.batch_size<1:p.error('batch-size must be positive')
    run(a)
