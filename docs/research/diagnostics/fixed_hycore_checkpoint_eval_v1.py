"""Read-only evaluation of explicitly fixed, user-owned HyCoRe checkpoints.

No training, model selection, voting, optimizer, or source-result mutation.
The spec supplies private paths; public outputs must strip those paths/IDs.
Original models did train on all9840, so the modern984 subset is *in-sample*,
not a retrospectively invented held-out validation set.
"""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False), encoding='utf-8')


def array_sha(array):
    h = hashlib.sha256()
    h.update(str(array.dtype).encode())
    h.update(str(array.shape).encode())
    h.update(array.tobytes())
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', type=Path, required=True)
    p.add_argument('--spec', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--gpu', type=int, required=True)
    p.add_argument('--max-seconds', type=int, default=900)
    a = p.parse_args()
    started = time.monotonic()
    spec = json.loads(a.spec.read_text(encoding='utf-8-sig'))
    if not spec.get('models'):
        raise ValueError('Explicit fixed checkpoint list required')
    source_paths = [Path(x['checkpoint']).resolve() for x in spec['models']]
    out = a.out.resolve()
    if any(out == s.parent or s.parent in out.parents for s in source_paths):
        raise ValueError('Outputs must be outside source run directories')
    out.mkdir(parents=True, exist_ok=False)
    manifest = dict(status='started', started_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                    optimizer_updates=0, backward_calls=0, voting=False,
                    checkpoint_selection='fixed before this evaluation',
                    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    write(out/'manifest.json', manifest)
    sys.path.insert(0, str(a.repo.resolve()))
    from tools.hier_postrun_audit.extract import (
        idle_gpu, load_cpu_checkpoint, sha256, state_digest)
    uuid = idle_gpu(a.gpu)
    os.environ['CUDA_VISIBLE_DEVICES'] = uuid
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    import numpy as np
    import torch
    if torch.cuda.is_initialized():
        raise RuntimeError('CUDA initialized before UUID pinning')
    from inter_hierarchy_MN40.hier_proxy_scratch_v5.diagnose import (
        Hype_pointMLP, load_shards, seed_all, split_identity, stratified_split)
    from inter_hierarchy_MN40.hier_proxy_scratch_v8.controls import evaluate_clean
    from inter_hierarchy_MN40.hier_proxy_scratch_v8.geometry import configure_backbone
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    train, labels, train_shards = load_shards(Path(spec['data_dir']), 'train')
    test, test_labels, test_shards = load_shards(Path(spec['data_dir']), 'test')
    train_ids, val_ids = stratified_split(labels, 22, .1)
    split_sha = split_identity(train_ids, val_ids)
    if (len(labels), len(train_ids), len(val_ids), len(test_labels)) != (9840,8856,984,2468):
        raise ValueError('Unexpected HDF5 population')
    if split_sha != spec['expected_split_sha256']:
        raise ValueError('Dataset order/labels produced a different modern split')
    manifest.update(gpu_uuid=uuid, split_sha256=split_sha, train_n=len(train_ids),
                    validation_n=len(val_ids), original_train_n=len(labels), test_n=len(test_labels),
                    training_shards=train_shards, test_shards=test_shards,
                    test_input_sha256=array_sha(np.ascontiguousarray(test[:,:1024])),
                    test_labels_sha256=array_sha(test_labels),
                    modern_validation_input_sha256=array_sha(np.ascontiguousarray(train[val_ids,:1024])))
    # Private input identities; these do not enter the reviewed aggregate report.
    np.savez_compressed(out/'private_split_ids.npz', train_ids=train_ids,
                        validation_ids=val_ids, labels=labels)
    rows=[]
    first_allocation=True
    try:
        for entry in spec['models']:
            if time.monotonic()-started>a.max_seconds:
                raise TimeoutError('Evaluation budget reached')
            cp=Path(entry['checkpoint']);before_sha=sha256(cp);saved=load_cpu_checkpoint(cp)
            epoch=int(saved.get('epoch',saved.get('completed_epochs',-1)))
            if epoch!=int(entry['expected_epoch']):
                raise ValueError('Actual checkpoint epoch differs from the fixed identity')
            original=entry['version']=='original'
            if not original:
                if saved.get('split_sha256')!=split_sha:
                    raise ValueError('Checkpoint has a different training/validation split')
                for key,want in [('train_ids',train_ids),('validation_ids',val_ids)]:
                    if key in saved and not np.array_equal(np.asarray(saved[key]),want):
                        raise ValueError('Saved split IDs differ')
            seed_all(22)
            model=Hype_pointMLP()
            if entry['version']=='v8':
                if saved.get('format') not in ['hycore-hier-v8-c01-official-geometry-1','hycore-v8-c01-controls-1']:
                    raise ValueError('Unrecognized V8 format')
                model=configure_backbone(model)
            elif entry['version'] not in ['v7','original']:
                raise ValueError('Unknown geometry adapter')
            net={k.removeprefix('module.'):v for k,v in saved['net'].items()}
            model.load_state_dict(net,strict=True)
            model.requires_grad_(False);model.eval()
            curvature=float(model.manifold.c.detach().cpu())
            if not np.isclose(curvature,.1 if entry['version']=='v8' else 1,rtol=1e-5):
                raise ValueError('Wrong loaded curvature')
            if first_allocation:
                if idle_gpu(a.gpu)!=uuid:raise RuntimeError('GPU became busy before allocation')
                first_allocation=False
            model=model.to('cuda:0');before_state=state_digest(model.state_dict())
            args=SimpleNamespace(eval_batch_size=int(entry['eval_batch_size']),workers=0,
                                 seed=22,smoke=False,smoke_eval_batches=None)
            # Evaluate the fixed model only; never choose a checkpoint by these scores.
            validation=evaluate_clean(model,train[val_ids],labels[val_ids],args,'cuda:0','val')
            test_result=evaluate_clean(model,test,test_labels,args,'cuda:0','test')
            if state_digest(model.state_dict())!=before_state:
                raise RuntimeError('Model or BN buffers changed')
            if any(p.grad is not None for p in model.parameters()):
                raise RuntimeError('Unexpected gradient buffer')
            if sha256(cp)!=before_sha:raise RuntimeError('Source checkpoint changed')
            counts=np.bincount(labels,minlength=40)
            val_counts=np.bincount(labels[val_ids],minlength=40)
            row=dict(name=entry['name'],version=entry['version'],epoch=epoch,
                     epoch_convention='original source zero-based metadata' if original else 'one-based completed epochs',
                     checkpoint_sha256=before_sha,checkpoint_path=str(cp),
                     checkpoint_format=saved.get('format'),curvature=curvature,
                     split_sha256=split_sha,validation_role='in_sample_modern984_diagnostic' if original else 'held_out_validation',
                     input_condition='raw first1024; no augment/no part; eval BN; FP32; no voting',
                     eval_batch_size=args.eval_batch_size,validation=validation,test=test_result,
                     recorded_checkpoint_best=saved.get('best'),
                     original_recorded_best_test_oa=saved.get('best_test_acc') if original else None,
                     original_recorded_best_test_aa=saved.get('best_test_acc_avg') if original else None,
                     labels_per_class=counts.tolist(),modern_train_per_class=(counts-val_counts).tolist(),
                     modern_validation_per_class=val_counts.tolist(),
                     test_per_class=np.bincount(test_labels,minlength=40).tolist(),
                     read_only=dict(checkpoint_unchanged=True,model_and_BN_unchanged=True,
                                    no_gradient_buffers=True,backward_calls=0,optimizer_updates=0))
            write(out/(entry['name']+'.json'),row);rows.append(row)
            print(json.dumps(dict(name=entry['name'],epoch=epoch,validation_role=row['validation_role'],
                                  val_oa=validation['val_oa'],test_oa=test_result['test_oa'])),flush=True)
            del model,saved,net
            torch.cuda.empty_cache()
        manifest.update(status='completed',models=rows,elapsed_seconds=time.monotonic()-started,
                        completed_utc=dt.datetime.now(dt.timezone.utc).isoformat())
    except BaseException as e:
        manifest.update(status='failed',error=repr(e),elapsed_seconds=time.monotonic()-started)
        write(out/'manifest.json',manifest)
        raise
    write(out/'manifest.json',manifest)


if __name__=='__main__':main()
