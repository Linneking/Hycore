"""One fresh native CB32 H0 run; idle GPU, independent smoke, immutable source."""
from __future__ import annotations
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
RELATIVE = HERE.relative_to(REPO)
for item in (HERE, REPO, REPO / 'inter_hierarchy_MN40'):
    sys.path.insert(0, str(item))
import runner
from inter_hierarchy_MN40.hier_proxy_cycle_v9c.queue import freeze_source as freeze_parent, TRAIN_SHA256, TRAIN_ORDER
from inter_hierarchy_MN40.hier_proxy_scratch_v8.queue import now, write, query_gpus, idle_gpus, device_environment, run_logged, cache_data


def freeze_source(repo, root, commit):
    snapshot = freeze_parent(repo, root, commit)
    tracked = set(subprocess.check_output(['git','-C',str(repo),'ls-files','-z','--',RELATIVE.as_posix()]).decode().split('\0'))
    identity = json.loads((root/'source_identity.json').read_text())
    destination = snapshot / RELATIVE
    destination.mkdir(parents=True, exist_ok=False)
    for source in sorted((repo/RELATIVE).glob('*.py')):
        relative = RELATIVE / source.name
        if relative.as_posix() not in tracked:
            raise RuntimeError('All launch sources must be committed')
        target = snapshot / relative
        shutil.copy2(source, target)
        identity['files'][relative.as_posix()] = hashlib.sha256(target.read_bytes()).hexdigest()
    identity['successor'] = dict(alias='TS26', baseline='B15', name=runner.NAME, epochs=300,
        steps_per_epoch=200, seed=4780, source_checkpoint=None, teacher=None)
    write(root/'source_identity.json', identity)
    return snapshot


def finite_tree(value, torch):
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(finite_tree(v, torch) for v in value.values())
    if isinstance(value, (list,tuple)):
        return all(finite_tree(v, torch) for v in value)
    return True


def accept_run(directory, smoke):
    import torch
    epochs, steps = (2,2) if smoke else (300,200)
    manifest = json.loads((directory/'manifest.json').read_text())
    config = manifest['training_config']
    expected = dict(hier_enabled=False, P=0, HIER_calls_per_step=0, world_size=1, global_batch=32,
        classes_per_rank=16, instances_per_class=2, base_c=1., native_c=1., native_GS=False,
        native_extra_cap=None, contrastive_margin=4., radial_gamma=1000.,
        proxy_optimizer=None, seed=4780, epochs=epochs, steps_per_epoch=steps)
    if any(config.get(k) != v for k,v in expected.items()):
        raise RuntimeError('Saved configuration differs from native H0 CB32 contract')
    if not (manifest['status']=='completed' and manifest['completed_epochs']==epochs
            and manifest['train_size']==9840 and manifest['validation_size']==0
            and manifest['official_test_read'] is (not smoke) and manifest['resume_identity'] is None
            and manifest['initialized_model_sha256']==runner.INITIAL_HASH
            and manifest['selection_domain']==('smoke_train' if smoke else 'test')):
        raise RuntimeError('Completion/selection/initialization contract failed')
    if not smoke and manifest['selection_size'] != 2468:
        raise RuntimeError('Official test must be complete')
    saved = torch.load(directory/'last.pth', map_location='cpu', weights_only=False)
    if saved['completed_optimizer_steps'] != epochs*steps or len(saved['rank_states']) != 1:
        raise RuntimeError('Optimizer/rank recovery count differs')
    if any(saved.get(k) is not None for k in ('proxy','proxy_optimizer','proxy_scheduler','best_proxy')):
        raise RuntimeError('H0 constructed hierarchy state')
    state, = saved['rank_states']
    counters = [int(v) for k,v in state['BN_buffers'].items() if k.endswith('num_batches_tracked')]
    if not counters or set(counters) != {2*epochs*steps}:
        raise RuntimeError('BN must update twice per optimizer step')
    if not {'rng','sampler','private_generators','BN_buffers'}.issubset(state):
        raise RuntimeError('Missing complete recovery state')
    if state['private_generators']['gumbel'] is not None:
        raise RuntimeError('H0 allocated a Gumbel generator')
    for key in ('net','optimizer'):
        if saved.get(key) is None or not finite_tree(saved[key],torch):
            raise RuntimeError('Missing/nonfinite model or optimizer')
    group, = saved['optimizer']['param_groups']
    scheduler = saved['scheduler']
    if not (group['initial_lr']==.1 and group['momentum']==.9 and group['weight_decay']==2e-4
            and scheduler['T_max']==epochs and scheduler['last_epoch']==epochs
            and scheduler['eta_min']==.005 and math.isclose(group['lr'],.005,abs_tol=1e-12)):
        raise RuntimeError('Original RSGD/cosine contract failed')
    metrics = json.loads((directory/f'metrics_epoch_{epochs:03d}.json').read_text())
    if metrics['lambda_hier'] != 0:
        raise RuntimeError('H0 recorded nonzero HIER weight')
    if smoke:
        actual = torch.load(directory/'actual_training_epoch_001.pth',map_location='cpu',weights_only=False)
        if not (actual['native_c']==1. and actual['rank_order']==[0] and len(actual['ids'])==32
                and actual['proxy_parameter_count']==0 and not actual['hier_view_present']):
            raise RuntimeError('Actual train forward differs from CB32 native H0')
        counts = torch.unique(actual['labels'], return_counts=True)[1]
        if len(counts)!=16 or not bool((counts==2).all()):
            raise RuntimeError('Actual train batch is not 16 classes x2')
    if not (directory/'best.pth').is_file():
        raise RuntimeError('Missing best checkpoint')
    return dict(accepted=True, epochs=epochs, updates=epochs*steps, BN_updates=2*epochs*steps,
                HIER_constructed=False, initial_sha256=runner.INITIAL_HASH, official_test_read=not smoke)


def run_training(command, log, env, snapshot, directory, smoke):
    completed_at = None
    with log.open('x') as stream:
        child = subprocess.Popen(command, cwd=snapshot, env=env, stdin=subprocess.DEVNULL,
                                 stdout=stream, stderr=subprocess.STDOUT)
        while child.poll() is None:
            path = directory/'manifest.json'
            if path.is_file():
                try:
                    done = json.loads(path.read_text()).get('status')=='completed'
                except (OSError,ValueError):
                    done = False
                if done and completed_at is None:
                    completed_at = time.monotonic()
            if completed_at is not None and time.monotonic()-completed_at > 120:
                write(directory.parent/(directory.name+'_saved_completion.json'),accept_run(directory,smoke))
                child.terminate()  # Only this queue's verified-complete direct child.
                child.wait(timeout=30)
                return 0
            time.sleep(2)
        return child.returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo',type=Path,required=True)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--data-dir',type=Path,required=True)
    parser.add_argument('--expected-commit',required=True)
    parser.add_argument('--source-branch',required=True)
    args=parser.parse_args()
    args.repo=args.repo.resolve(); args.data_dir=args.data_dir.resolve()
    args.root.mkdir(parents=True,exist_ok=False)
    status_path=args.root/'queue_status.json'
    state=dict(status='preparing_cpu_only',queue_pid=os.getpid(),created_utc=now(),commit=args.expected_commit,
        branch=args.source_branch,alias='TS26',canonical_name=runner.NAME,seed=4780,epochs=300,
        steps_per_epoch=200,production_started=False,source_checkpoint=None,teacher=None)
    write(status_path,state)
    try:
        snapshot=freeze_source(args.repo,args.root,args.expected_commit)
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',
            OPENBLAS_NUM_THREADS='2',PYTHONUNBUFFERED='1',HDF5_USE_FILE_LOCKING='FALSE',
            V9C_SOURCE_COMMIT=args.expected_commit,V9C_SOURCE_BRANCH=args.source_branch,
            HYCORE_TRAIN_SHARD_ORDER=json.dumps(TRAIN_ORDER),
            PYTHONPYCACHEPREFIX='/tmp/hycore_cb32_h0_pycache_'+args.root.name)
        env['PYTHONPATH']=':'.join((str(snapshot),str(snapshot/'inter_hierarchy_MN40'),env.get('PYTHONPATH','')))
        commands=[
            [sys.executable,'-m','unittest','discover','-s',str(snapshot/RELATIVE),'-p','test_contracts.py','-v'],
            [sys.executable,'-m','unittest','inter_hierarchy_MN40.hier_proxy_scratch_v9.tests.test_base_protocol','-v']]
        for index,command in enumerate(commands):
            log=args.root/f'cpu_contracts_{index}.log'
            if run_logged(command,log,env,snapshot,timeout=1200)!=0 or 'skipped=' in log.read_text():
                raise RuntimeError('CPU contracts failed/skipped; no GPU allocated')
        cached=cache_data(args.data_dir,args.root.name,args.root)
        data=json.loads((args.root/'training_data_cache.json').read_text())
        if {v['file']:v['sha256'] for v in data['train_files']} != TRAIN_SHA256:
            raise RuntimeError('Training HDF5 identity differs')
        state.update(status='waiting_for_idle_gpu',cpu_checks_passed=True,training_files_match=True,updated_utc=now())
        write(status_path,state)
        attempts=0
        while True:
            rows=query_gpus(); state.update(gpu_observation=rows,updated_utc=now());write(status_path,state)
            candidates=idle_gpus(rows)
            if not candidates:
                time.sleep(60);continue
            chosen=candidates[0]
            lock=open(f"/tmp/hycore_{os.getuid()}_gpu_{chosen['index']}.lock",'a+')
            try:
                try:
                    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:
                    time.sleep(60);continue
                time.sleep(10)
                if not any(v['uuid']==chosen['uuid'] for v in idle_gpus(query_gpus())):
                    continue
                env.update(device_environment([chosen]));attempts+=1
                base=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=1',
                    str(snapshot/RELATIVE/'runner.py'),'--hier','off','--data-dir',str(cached)]
                smoke=args.root/f'smoke_{attempts:02d}'
                state.update(status='smoke_running',gpus=[chosen],updated_utc=now());write(status_path,state)
                command=base+['--run-dir',str(smoke),'--epochs','2','--steps-per-epoch','2',
                    '--smoke','--skip-final-test','--smoke-eval-batches','1','--audit-epochs','1']
                if run_training(command,args.root/f'smoke_{attempts:02d}.log',env,snapshot,smoke,True)!=0:
                    raise RuntimeError('Independent train-only smoke failed; no production launch')
                write(args.root/'smoke_acceptance.json',accept_run(smoke,True))
                state.update(smoke_passed=True,updated_utc=now());write(status_path,state)
                time.sleep(5)
                if not any(v['uuid']==chosen['uuid'] for v in idle_gpus(query_gpus())):
                    continue
                production=args.root/'main';command=base+['--run-dir',str(production)]
                state.update(status='production_running',production_started=True,production_dir=str(production),
                    production_command=command,production_started_utc=now(),updated_utc=now());write(status_path,state)
                code=run_training(command,args.root/'production.log',env,snapshot,production,False)
                if code==0:
                    state['completion_acceptance']=accept_run(production,False)
                state.update(status='completed' if code==0 else 'production_failed',exit_code=code,
                    finished_utc=now(),updated_utc=now());write(status_path,state)
                return
            finally:
                lock.close()
    except BaseException as exc:
        state.update(status='failed',error=repr(exc),updated_utc=now());write(status_path,state)
        raise


if __name__=='__main__':
    main()
