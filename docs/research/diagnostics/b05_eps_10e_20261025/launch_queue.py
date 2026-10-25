"""Two fresh B05 ten-epoch epsilon-only diagnostics on one idle GPU pair.

Both small/large epsilon smokes precede both full diagnostics. Training never
reads test/evaluation data. After2000 updates per arm, CPU analysis is automatic.
"""
from __future__ import annotations
import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[3]
RELATIVE=HERE.relative_to(REPO)
for item in (HERE,REPO,REPO/'inter_hierarchy_MN40'):
    sys.path.insert(0,str(item))
import runner
from inter_hierarchy_MN40.hier_proxy_cycle_v9c.queue import freeze_source as freeze_parent,TRAIN_ORDER,TRAIN_SHA256
from inter_hierarchy_MN40.hier_proxy_scratch_v8.queue import now,write,query_gpus,idle_gpus,device_environment,run_logged,cache_data


def freeze_source(repo,root,commit):
    snapshot=freeze_parent(repo,root,commit)
    tracked=set(subprocess.check_output(['git','-C',str(repo),'ls-files','-z','--',RELATIVE.as_posix()]).decode().split('\0'))
    target=snapshot/RELATIVE
    target.mkdir(parents=True,exist_ok=False)
    identity=json.loads((root/'source_identity.json').read_text())
    for source in (repo/RELATIVE).glob('*.py'):
        relative=RELATIVE/source.name
        if relative.as_posix() not in tracked: raise RuntimeError('Only committed diagnostic sources may run')
        shutil.copy2(source,target/source.name)
        identity['files'][relative.as_posix()]=hashlib.sha256((target/source.name).read_bytes()).hexdigest()
    identity['diagnostic_pair']=dict(seed=4780,epochs_observed=10,steps_per_epoch=200,schedule_epochs=300,
        batch=64,sample_K=20,proxy_K=20,proxy_initial_lr=.005,epsilons=[1e-8,1e-4],selection=None)
    write(root/'source_identity.json',identity)
    return snapshot


def finite_tree(value,torch):
    if isinstance(value,torch.Tensor): return bool(torch.isfinite(value).all())
    if isinstance(value,dict): return all(finite_tree(v,torch) for v in value.values())
    if isinstance(value,(list,tuple)): return all(finite_tree(v,torch) for v in value)
    return True


def read_rows(path):
    with path.open() as stream: return [json.loads(line) for line in stream if line.strip()]


def accept_run(directory,variant,epochs,steps):
    import torch
    manifest=json.loads((directory/'manifest.json').read_text())
    config=manifest['training_config']
    expected=dict(seed=4780,epochs=300,diagnostic_epochs=epochs,steps_per_epoch=steps,
        world_size=2,global_batch=64,classes_per_rank=16,instances_per_class=2,
        sample_K=20,proxy_K=20,T=50,P=512,D=256,base_c=.1,native_extra_cap=2.3,
        native_GS=True,hier_c=.1,hier_view_depth=4.6,hier_enabled=True,
        contrastive_margin=2.,radial_gamma=500.,eps_variant=variant)
    if any(config.get(k)!=v for k,v in expected.items()): raise RuntimeError('Actual config differs from approved epsilon-only B05')
    if not (manifest['status']=='completed' and manifest['completed_epochs']==epochs
        and manifest['completed_optimizer_steps']==epochs*steps and manifest['train_size']==9840
        and manifest['validation_size']==0 and manifest['selection_size']==0
        and not manifest['official_test_read'] and not manifest['test_used_for_selection']
        and manifest['best'] is None and manifest['resume_identity'] is None):
        raise RuntimeError('Actual run is not the complete fresh train-only diagnostic')
    if (directory/'best.pth').exists() or (directory/'final_test.json').exists(): raise RuntimeError('Diagnostic unexpectedly selected/evaluated a model')
    saved=torch.load(directory/'last.pth',map_location='cpu',weights_only=False)
    if saved['completed_optimizer_steps']!=epochs*steps or len(saved['rank_states'])!=2: raise RuntimeError('Wrong actual update/rank count')
    for rank,state in enumerate(saved['rank_states']):
        counters=[int(v) for k,v in state['BN_buffers'].items() if k.endswith('num_batches_tracked')]
        if state['rank']!=rank or not counters or set(counters)!={2*epochs*steps}: raise RuntimeError('BN/probe restoration differs from two actual forwards per update')
        if not {'sampler','private_generators','rng'}.issubset(state): raise RuntimeError('Incomplete recovery state')
    if not all(saved.get(k) is not None and finite_tree(saved[k],torch) for k in ('net','optimizer','proxy','proxy_optimizer')): raise RuntimeError('Missing/nonfinite actual model or optimizer')
    optimizer=runner.verify_optimizer_payload(saved,epochs,variant)
    model_group,=saved['optimizer']['param_groups']
    if not (model_group['initial_lr']==.1 and model_group['momentum']==.9 and model_group['weight_decay']==2e-4
        and saved['scheduler']['eta_min']==.005 and saved['scheduler']['last_epoch']==epochs):
        raise RuntimeError('Actual backbone RSGD or300-epoch schedule differs from B05')
    adam_steps={int(v['step']) for v in saved['proxy_optimizer']['state'].values()}
    if adam_steps!={epochs*steps}: raise RuntimeError('AdamW steps differ')
    observations=read_rows(directory/'proxy_steps.jsonl')
    ordinary=read_rows(directory/'steps.jsonl')
    if len(observations)!=epochs*steps or len(ordinary)!=epochs*steps: raise RuntimeError('Missing per-step observations')
    for epoch in range(epochs+1):
        record=json.loads((directory/f'observer_epoch_{epoch:03d}.json').read_text())
        if record['proxy_count']!=512 or len(record['proxy_depth'])!=512 or len(record['depth_delta_init'])!=512: raise RuntimeError('Incomplete proxy depth/movement distribution')
        if epoch and record['step_count']!=steps: raise RuntimeError('Observer epoch differs from actual steps')
        if epoch and not (directory/f'parameter_audit_epoch_{epoch:03d}.json').exists(): raise RuntimeError('Missing epoch HIER/backbone gradient audit')
    if any(r['actual_optimizer_step']!=i+1 or r['eps']!=runner.EPSILONS[variant] for i,r in enumerate(observations)): raise RuntimeError('Observer optimizer identity differs')
    if any(r['prediction_error_max_coordinate']>2e-6 for r in observations): raise RuntimeError('Actual AdamW step accounting failed')
    if any(r['proxy_replica_max_difference']>1e-7 or abs(r['hier_whole_depth']-4.6)>1e-4 for r in ordinary): raise RuntimeError('Replicas or constructed shell differ')
    final=json.loads((directory/f'observer_epoch_{epochs:03d}.json').read_text())
    return dict(accepted=True,variant=variant,epochs=epochs,steps=epochs*steps,
        initialized_model_sha256=manifest['initialized_model_sha256'],initial_proxy_parameter_sha256=observations[0]['initial_proxy_parameter_sha256'],
        initial_proxy_gradient_sha256=observations[0]['initial_proxy_gradient_sha256'],
        training_points_sha256=manifest['train_points_sha256'],training_labels_sha256=manifest['train_labels_sha256'],
        first_batch=manifest['first_batch_identity'],actual_optimizer=optimizer,BN_updates_per_rank=2*epochs*steps,
        final_cap_count=final['cap_count'],final_depth=final['proxy_depth_summary'],
        maximum_update_accounting_error=max(r['prediction_error_max_coordinate'] for r in observations))


def verify_pair(left,right):
    import torch
    a=torch.load(left/'initialization_e0.pth',map_location='cpu',weights_only=False)
    b=torch.load(right/'initialization_e0.pth',map_location='cpu',weights_only=False)
    for kind in ('net','proxy'):
        if set(a[kind])!=set(b[kind]) or any(not torch.equal(a[kind][k],b[kind][k]) for k in a[kind]):
            raise RuntimeError('Paired initial model/proxies are not exactly equal')
    ma,mb=(json.loads((p/'manifest.json').read_text()) for p in (left,right))
    for key in ('train_points_sha256','train_labels_sha256','split_sha256','first_batch_identity'):
        if ma[key]!=mb[key]: raise RuntimeError('Paired input/crop identity differs: '+key)
    x,y=(read_rows(p/'proxy_steps.jsonl')[0] for p in (left,right))
    return dict(initial_parameters_bitwise_equal=True,input_and_first_batch_equal=True,
                first_proxy_gradient_sha_equal=x['initial_proxy_gradient_sha256']==y['initial_proxy_gradient_sha256'],
                first_proxy_gradient_norms=[x['proxy_gradient_norm'],y['proxy_gradient_norm']],
                note='Initial gradient hash is recorded; later gradients may diverge after epsilon changes the optimization trajectory.')


def safe_query(state,path):
    try: return query_gpus()
    except (subprocess.TimeoutExpired,subprocess.CalledProcessError,OSError) as exc:
        state.update(gpu_query_error=repr(exc),updated_utc=now());write(path,state);return []


def wait_release(chosen,state,path):
    deadline=time.monotonic()+120
    while True:
        found={r['uuid'] for r in idle_gpus(safe_query(state,path))}
        if all(r['uuid'] in found for r in chosen): return
        if time.monotonic()>deadline: raise RuntimeError('Reserved pair not idle between diagnostic processes; no other process interrupted')
        time.sleep(2)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo',type=Path,required=True)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--data-dir',type=Path,required=True)
    parser.add_argument('--expected-commit',required=True)
    parser.add_argument('--source-branch',required=True)
    parser.add_argument('--analysis-deps',type=Path)
    args=parser.parse_args()
    args.repo,args.data_dir=args.repo.resolve(),args.data_dir.resolve()
    args.root.mkdir(parents=True,exist_ok=False)
    path=args.root/'queue_status.json'
    state=dict(status='preparing',pid=os.getpid(),created_utc=now(),commit=args.expected_commit,branch=args.source_branch,
        seed=4780,diagnostic_epochs=10,steps_per_epoch=200,schedule_epochs=300,global_batch=64,
        proxy_initial_lr=.005,epsilons=[1e-8,1e-4],source_checkpoint=None,teacher=None,test_read=False,
        order=['small','large'],target_gpus=[0,1],smokes={},diagnostics={})
    write(path,state)
    locks=[]
    try:
        snapshot=freeze_source(args.repo,args.root,args.expected_commit)
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
            PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1',HDF5_USE_FILE_LOCKING='FALSE',
            V9C_SOURCE_COMMIT=args.expected_commit,V9C_SOURCE_BRANCH=args.source_branch,
            HYCORE_TRAIN_SHARD_ORDER=json.dumps(TRAIN_ORDER))
        env['PYTHONPATH']=':'.join((str(snapshot),str(snapshot/'inter_hierarchy_MN40'),str(snapshot/'pointnet2_ops_lib'),env.get('PYTHONPATH','')))
        check_env=dict(env)
        if args.analysis_deps: check_env['PYTHONPATH']=str(args.analysis_deps)+':'+check_env['PYTHONPATH']
        tests=[sys.executable,'-m','unittest','discover','-s',str(snapshot/RELATIVE),'-p','test_*.py','-v']
        if run_logged(tests,args.root/'cpu_checks.log',check_env,snapshot,timeout=1200)!=0: raise RuntimeError('CPU diagnostic contracts failed')
        cached=cache_data(args.data_dir,args.root.name,args.root)
        data=json.loads((args.root/'training_data_cache.json').read_text())
        if {r['file']:r['sha256'] for r in data['train_files']}!=TRAIN_SHA256: raise RuntimeError('Training HDF5 identity differs')
        state.update(status='waiting_for_idle_pair',cpu_checks_passed=True,training_files_match=True,updated_utc=now());write(path,state)
        while True:
            rows=safe_query(state,path);found={r['index']:r for r in idle_gpus(rows)}
            state.update(gpu_observation=rows,updated_utc=now());write(path,state)
            if not all(g in found for g in (0,1)):
                time.sleep(30);continue
            chosen=[found[0],found[1]]
            try:
                for row in chosen:
                    lock=open(f"/tmp/hycore_{os.getuid()}_gpu_{row['index']}.lock",'a+');locks.append(lock)
                    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                break
            except BlockingIOError:
                for lock in locks: lock.close()
                locks=[];time.sleep(30)
        time.sleep(5);wait_release(chosen,state,path)
        env.update(device_environment(chosen))
        base=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=2',str(snapshot/RELATIVE/'runner.py'),
            '--data-dir',str(cached),'--workers','8']
        for smoke in (True,False):
            for variant in ('small','large'):
                directory=args.root/(('smoke_' if smoke else '')+variant)
                command=base+['--eps',variant,'--run-dir',str(directory)]
                if smoke: command+=['--smoke','--diagnostic-epochs','2','--steps-per-epoch','2']
                state.update(status='smoke_running' if smoke else 'diagnostic_running',active_variant=variant,
                    active_dir=str(directory),active_command=command,updated_utc=now());write(path,state)
                if run_logged(command,args.root/(directory.name+'.log'),env,snapshot,timeout=1200 if smoke else None)!=0:
                    raise RuntimeError(directory.name+' failed; inspect its saved log')
                accepted=accept_run(directory,variant,2 if smoke else 10,2 if smoke else 200)
                state['smokes' if smoke else 'diagnostics'][variant]=accepted
                write(args.root/(directory.name+'_acceptance.json'),accepted)
                if variant=='large':
                    pair=verify_pair(args.root/('smoke_small' if smoke else 'small'),directory)
                    state['smoke_pair_identity' if smoke else 'pair_identity']=pair
                state.update(updated_utc=now());write(path,state)
                wait_release(chosen,state,path)
        state.update(status='cpu_analysis_running',updated_utc=now());write(path,state)
        # GPU locks can be released once the actual diagnostics are accepted.
        for lock in locks: lock.close()
        locks=[]
        analysis_env=dict(env,CUDA_VISIBLE_DEVICES='',MPLBACKEND='Agg')
        if args.analysis_deps: analysis_env['PYTHONPATH']=str(args.analysis_deps)+':'+analysis_env['PYTHONPATH']
        command=[sys.executable,str(snapshot/RELATIVE/'analyze.py'),'--small-run',str(args.root/'small'),
            '--large-run',str(args.root/'large'),'--out',str(args.root/'public')]
        if run_logged(command,args.root/'analysis.log',analysis_env,snapshot,timeout=1200)!=0: raise RuntimeError('Diagnostics completed but CPU analysis failed')
        state.update(status='completed',finished_utc=now(),updated_utc=now());write(path,state)
    except BaseException as exc:
        state.update(status='failed',error=repr(exc),updated_utc=now());write(path,state);raise
    finally:
        for lock in locks: lock.close()


if __name__=='__main__': main()
