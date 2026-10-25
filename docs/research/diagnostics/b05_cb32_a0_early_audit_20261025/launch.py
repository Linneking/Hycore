"""One fresh CB32 A0 early rerun, then immutable audit85 geometry on e0..e10."""
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
REL=HERE.relative_to(REPO)
RUNNER_DIR=HERE
for p in (REPO,REPO/'inter_hierarchy_MN40'):sys.path.insert(0,str(p))
from inter_hierarchy_MN40.hier_proxy_cycle_v9c.queue import freeze_source as freeze_parent,TRAIN_ORDER,TRAIN_SHA256
from inter_hierarchy_MN40.hier_proxy_scratch_v8.queue import now,write,query_gpus,idle_gpus,device_environment,cache_data


def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()


def freeze(repo,root,commit):
    snapshot=freeze_parent(repo,root,commit)
    identity=json.loads((root/'source_identity.json').read_text())
    dirs=[Path('docs/research/diagnostics')/s for s in (
        's4780_native_b05_20261023','b05_singlecard_a0_a1_20261025',
        'b05_cb32_a0_early_audit_20261025','embedding_audit85','v9_embedding_review_20261021')]
    for relative in dirs:
        tracked=set(subprocess.check_output(['git','-C',str(repo),'ls-files','-z','--',relative.as_posix()]).decode().split('\0'))
        (snapshot/relative).mkdir(parents=True,exist_ok=True)
        for p in sorted((repo/relative).glob('*.py')):
            target=relative/p.name
            if target.as_posix() not in tracked:raise RuntimeError('Unpublished diagnostic source: '+str(target))
            shutil.copy2(p,snapshot/target);identity['files'][target.as_posix()]=sha(snapshot/target)
    identity['audit_scope']={'fresh_A0_CB32':True,'observed_epochs':10,'schedule_epochs':300,
        'sample_K':10,'proxy_K':20,'lr':.01,'eps':1e-8,'historical_missing_weights_reconstructed':False}
    write(root/'source_identity.json',identity)
    return snapshot


def finite(value,torch):
    if isinstance(value,torch.Tensor):return bool(torch.isfinite(value).all())
    if isinstance(value,dict):return all(finite(x,torch) for x in value.values())
    if isinstance(value,(list,tuple)):return all(finite(x,torch) for x in value)
    return True


def init_pair(directory,original):
    import torch
    a=torch.load(directory/'initialization_e0.pth',map_location='cpu',weights_only=False)
    b=torch.load(original/'initialization_e0.pth',map_location='cpu',weights_only=False)
    for kind in ('net','proxy'):
        if set(a[kind])!=set(b[kind]) or any(not torch.equal(a[kind][k],b[kind][k]) for k in a[kind]):raise RuntimeError('Historical initialization differs')
    m=json.loads((directory/'manifest.json').read_text());old=json.loads((original/'manifest.json').read_text())
    for key in ('train_points_sha256','train_labels_sha256','split_sha256','first_batch_identity'):
        if m.get(key)!=old.get(key):raise RuntimeError('Historical first-batch/input differs: '+key)
    return {'initial_network_and_proxy_bitwise_equal':True,'training_data_and_first_batch_equal':True}


def accept(directory,original,epochs,steps):
    import torch
    runner=module(RUNNER_DIR/'runner.py','early_accept_runner')
    m=json.loads((directory/'manifest.json').read_text());c=m['training_config']
    required=dict(variant='a0',seed=4780,epochs=300,observed_epochs=epochs,steps_per_epoch=steps,
        global_batch=32,world_size=1,classes_per_rank=16,instances_per_class=2,sample_K=10,proxy_K=20,
        P=512,D=256,base_c=.1,native_extra_cap=2.3,native_GS=True,hier_view_depth=4.6,radial_gamma=500.,contrastive_margin=2.)
    if any(c.get(k)!=v for k,v in required.items()):raise RuntimeError('Actual early rerun config differs')
    if not (m['status']=='completed' and m['completed_epochs']==epochs and m['completed_optimizer_steps']==epochs*steps
            and m['resume_identity'] is None):raise RuntimeError('Incomplete/nonfresh rerun')
    for e in range(epochs+1):
        p=directory/('initialization_e0.pth' if not e else 'checkpoint_epoch_%03d.pth'%e)
        s=torch.load(p,map_location='cpu',weights_only=False)
        if s['completed_epochs']!=e or s['completed_optimizer_steps']!=e*steps:raise RuntimeError('Wrong retained epoch')
        runner.verify_optimizer_payload(s,e)
        if not all(finite(s[k],torch) for k in ('net','proxy','optimizer','proxy_optimizer')):raise RuntimeError('Nonfinite saved state')
        if e and {int(x['step']) for x in s['proxy_optimizer']['state'].values()}!={e*steps}:raise RuntimeError('Adam step count differs')
        rank,=s['rank_states'];bn=[int(v) for k,v in rank['BN_buffers'].items() if k.endswith('num_batches_tracked')]
        if not bn or set(bn)!={e*steps*2} or not all(k in rank for k in ('sampler','rng','private_generators')):raise RuntimeError('BN/random recovery differs')
        del s
    if not m['official_test_read'] and steps==200:raise RuntimeError('Formal historical test protocol was changed')
    result=dict(accepted=True,epochs=epochs,steps=epochs*steps,BN_updates=epochs*steps*2,
        all_epoch_checkpoints_retained=True,schedule_epochs=300,actual_proxy_lr=.01,actual_proxy_eps=1e-8)
    if steps==200:result.update(init_pair(directory,original))
    return result


def run(command,log,env,cwd,state,path,*,timeout=12000,completed_manifest=None,completion_gate=None):
    """Bounded owned child; completion is not confused with process exit."""
    started=time.monotonic();completed_at=None
    with log.open('x') as f:
        p=subprocess.Popen(command,stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,env=env,cwd=cwd)
        state.update(active_pid=p.pid,active_command=command,updated_utc=now());write(path,state)
        while p.poll() is None:
            if completed_manifest and completed_manifest.exists():
                try:completed=json.loads(completed_manifest.read_text()).get('status')=='completed'
                except (OSError,ValueError):completed=False
                if completed and completed_at is None:completed_at=time.monotonic()
            if completed_at and time.monotonic()-completed_at>120:
                if completion_gate is None:raise RuntimeError('No checkpoint gate for completed child teardown')
                completion_gate()  # No termination without actual saved-state acceptance.
                # torchrun is our direct child and propagates TERM only to its
                # own workers. No GPU-wide/user-wide/process-group signalling.
                p.terminate();state.update(completed_child_exit_timeout=True,updated_utc=now());write(path,state)
                try:p.wait(timeout=30)
                except subprocess.TimeoutExpired:raise RuntimeError('Owned completed child still blocked in OS cleanup')
                return
            if time.monotonic()-started>timeout:raise RuntimeError('Owned child timeout; inspect retained logs before any intervention')
            time.sleep(2)
        if p.returncode:raise RuntimeError(log.name+' exited '+str(p.returncode))


def wait_idle(gpu,state,path):
    while True:
        try:
            rows=query_gpus();found={r['index']:r for r in idle_gpus(rows)}
            if gpu in found:return found[gpu]
            state.update(gpu_observation=rows,updated_utc=now());write(path,state)
        except (OSError,subprocess.SubprocessError) as e:
            state.update(gpu_query_error=repr(e),updated_utc=now());write(path,state)
        time.sleep(10)


def compare_saved(replay,original):
    import torch
    results={}
    for epoch in (1,5):
        a=torch.load(replay/('checkpoint_epoch_%03d.pth'%epoch),map_location='cpu',weights_only=False)
        b=torch.load(original/('checkpoint_epoch_%03d.pth'%epoch),map_location='cpu',weights_only=False)
        fields={}
        for kind in ('net','proxy'):
            differences=[(a[kind][k].double()-b[kind][k].double()).abs() for k in a[kind]]
            fields[kind]={'bitwise_equal':all(torch.equal(a[kind][k],b[kind][k]) for k in a[kind]),
                'max_abs_difference':max(float(d.max()) for d in differences),
                'mean_abs_difference':sum(float(d.sum()) for d in differences)/sum(d.numel() for d in differences)}
        results[str(epoch)]=fields
    return results


def main():
    global RUNNER_DIR
    parser=argparse.ArgumentParser();parser.add_argument('--settings',type=Path,required=True);args=parser.parse_args()
    s=json.loads(args.settings.read_text(encoding='utf-8-sig'));repo=Path(s['repo']);root=Path(s['root']);original=Path(s['original_dir'])
    root.mkdir(parents=True,exist_ok=False);path=root/'status.json'
    state={'status':'preparing','pid':os.getpid(),'started_utc':now(),'source_commit':s['commit'],
        'scope':'B05 CB32 A0 fresh same-recipe10 rerun plus immutable audit85; no PartHier',
        'replay_optimizer_updates':2000,'audit_optimizer_updates':0,'historical_checkpoint_recovery':False}
    write(path,state);lock=None
    try:
        snapshot=freeze(repo,root,s['commit']);RUNNER_DIR=snapshot/REL;env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',
            OPENBLAS_NUM_THREADS='2',PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1',HDF5_USE_FILE_LOCKING='FALSE',
            V9C_SOURCE_COMMIT=s['commit'],V9C_SOURCE_BRANCH=s['branch'],HYCORE_TRAIN_SHARD_ORDER=json.dumps(TRAIN_ORDER))
        env['PYTHONPATH']=':'.join((str(snapshot),str(snapshot/'inter_hierarchy_MN40'),str(snapshot/'pointnet2_ops_lib'),env.get('PYTHONPATH','')))
        checkenv=dict(env,PYTHONPATH=str(s['analysis_deps'])+':'+env['PYTHONPATH'])
        for sub,name in ((REL,'early'),(Path('docs/research/diagnostics/embedding_audit85'),'audit85')):
            run([sys.executable,'-m','unittest','discover','-s',str(snapshot/sub),'-p','test_*.py','-v'],root/(name+'_CPU.log'),checkenv,snapshot,state,path,timeout=1200)
        cached=cache_data(Path(s['data_dir']),root.name,root)
        data=json.loads((root/'training_data_cache.json').read_text())
        if {r['file']:r['sha256'] for r in data['train_files']}!=TRAIN_SHA256:raise RuntimeError('Training shard identity differs')
        state.update(status='waiting_for_idle_gpu',updated_utc=now());write(path,state)
        chosen=wait_idle(s['gpu'],state,path)
        while True:
            lock=open('/tmp/hycore_%d_gpu_%d.lock'%(os.getuid(),s['gpu']),'a+')
            try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);break
            except BlockingIOError:
                lock.close();lock=None;time.sleep(10)
        chosen=wait_idle(s['gpu'],state,path);env.update(device_environment([chosen]));state['GPU']=chosen
        base=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=1',str(snapshot/REL/'runner.py'),
            '--variant','a0','--hier','on','--data-dir',str(cached),'--workers','8','--eval-batch-size','16']
        for smoke in (True,False):
            d=root/('smoke' if smoke else 'replay');epochs,steps=(2,2) if smoke else (10,200)
            command=base+['--run-dir',str(d)]
            if smoke:command+=['--smoke','--observed-epochs','2','--steps-per-epoch','2','--skip-final-test']
            state.update(status='smoke_running' if smoke else 'replay_running',active_dir=str(d),updated_utc=now());write(path,state)
            run(command,root/(d.name+'.log'),env,snapshot,state,path,completed_manifest=d/'manifest.json',
                completion_gate=lambda:accept(d,original,epochs,steps))
            accepted=accept(d,original,epochs,steps);write(root/(d.name+'_acceptance.json'),accepted)
            state[d.name+'_accepted']=True;write(path,state);wait_idle(s['gpu'],state,path)
        comparison=compare_saved(root/'replay',original);write(root/'original_replay_comparison.json',comparison)
        collector=module(snapshot/'docs/research/diagnostics/v9_embedding_review_20261021/collect.py','early_collector')
        observations=[]
        for e in range(11):
            observations.append(dict(key='B05_CB32_A0_REPLAY',stage='e%03d'%e,expected_epoch=e,epoch=e,primary=True,
                kind='cycle_hier',c_native=.1,c_hier=.1,checkpoint=str(root/'replay'/('initialization_e0.pth' if e==0 else 'checkpoint_epoch_%03d.pth'%e)),run_dir=str(root/'replay')))
        for e in (1,5):observations.append(dict(key='B05_CB32_A0_ORIGINAL',stage='e%03d'%e,expected_epoch=e,epoch=e,primary=False,
            kind='cycle_hier',c_native=.1,c_hier=.1,checkpoint=str(original/('checkpoint_epoch_%03d.pth'%e)),run_dir=str(original)))
        stage=root/'staged';stage.mkdir();collector.stage({'observations':observations},stage)
        staged=json.loads((stage/'staged_manifest.json').read_text())['observations']
        for row in staged:row.pop('run_dir',None)
        state.update(status='audit85_golden_regression',updated_utc=now());write(path,state)
        golden=s['golden'];wait_idle(s['gpu'],state,path)
        run([sys.executable,str(snapshot/'docs/research/diagnostics/embedding_audit85/gate.py'),'--repo',str(snapshot),
            '--data-dir',str(cached),'--checkpoint',golden['checkpoint'],'--expected-checkpoint-sha',golden['checkpoint_sha256'],
            '--reference-cache',golden['cache'],'--expected-input-sha',golden['input_sha256'],'--gpu',str(s['gpu']),
            '--out',str(root/'golden_regression.json')],root/'golden_regression.log',env,snapshot,state,path,timeout=1200)
        if not json.loads((root/'golden_regression.json').read_text())['passed']:raise RuntimeError('Audit85 coordinate gate failed')
        wait_idle(s['gpu'],state,path);spec=root/'export_spec.json';write(spec,{'repo':str(snapshot),'data_dir':str(cached),'observations':staged})
        state.update(status='exporting_e0_to_e10',updated_utc=now());write(path,state)
        run([sys.executable,str(snapshot/'docs/research/diagnostics/v9_embedding_review_20261021/collect.py'),'export','--spec',str(spec),
            '--out',str(root/'caches'),'--gpu',str(s['gpu'])],root/'export.log',env,snapshot,state,path,timeout=12000)
        exported=json.loads((root/'caches/export_manifest.json').read_text())
        if exported['status']!='completed' or exported['input_sha256']!=golden['input_sha256']:raise RuntimeError('Incomplete or changed canonical population')
        observations=exported['observations']
        for row in observations:row.update(cache=str(root/'caches'/row['cache']),cache_sha256=sha(root/'caches'/row['cache']),run_id=row['key'])
        reference=dict(s['representative_reference']);reference.update(primary=False,run_id='TS13')
        if sha(reference['cache'])!=reference['cache_sha256']:raise RuntimeError('Fixed TS13 representative reference changed')
        observations.append(reference)
        manifest={'observations':observations,'representative_reference':'TS13__best','replay_equivalence':comparison,
            'replay_kind':'fresh same-recipe; not historical missing checkpoint recovery','training_protocol':'CB32 K10/20;300-cosine observed10',
            'frozen_readout_protocol':'unchanged audit85 CB64 K20 common diagnosis, not actual training mining',
            'replay_run_dir':str(root/'replay'),'original_run_dir':str(original)}
        write(root/'analysis_manifest.json',manifest)
        wait_idle(s['gpu'],state,path);lock.close();lock=None
        state.update(status='CPU_audit85_statistics',updated_utc=now());write(path,state)
        analysisenv=dict(checkenv,CUDA_VISIBLE_DEVICES='',MPLBACKEND='Agg')
        run([sys.executable,str(snapshot/REL/'postprocess.py'),'--manifest',str(root/'analysis_manifest.json'),'--out',str(root/'public')],
            root/'postprocess.log',analysisenv,snapshot,state,path,timeout=12000)
        state.update(status='completed',finished_utc=now(),updated_utc=now());write(path,state)
    except BaseException as e:
        state.update(status='failed',error=repr(e),updated_utc=now());write(path,state);raise
    finally:
        if lock is not None:lock.close()


if __name__=='__main__':main()
