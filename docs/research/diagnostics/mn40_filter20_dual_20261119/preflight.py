"""Actual idle-GPU smoke gates; all smoke runs are fresh train-only 2x2."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[3]))
from inter_hierarchy_MN40.hier_proxy_scratch_v8.queue import device_environment,idle_gpus,query_gpus,now,write


def load(path):
    import torch
    return torch.load(path,map_location='cpu',weights_only=False)


def rows(path):
    return [json.loads(x) for x in path.read_text().splitlines()]


def verify(root, full_reference='legacy_ref', cb_reference='cb_ref'):
    import torch
    names=['legacy_ref','full_off','dual_on','sh_h0','sh_on','cb_ref','cb_h0','cb_off','cb_on']
    names += [n for n in (full_reference,cb_reference) if n not in names]
    m={name:json.loads((root/name/'manifest.json').read_text()) for name in names}
    s={name:load(root/name/'last_checkpoint.pth') for name in names}
    for name in names:
        assert m[name]['status']=='completed' and m[name]['completed_epochs']==2,name
        assert not m[name]['official_test_read'],name
        assert s[name]['metrics']['optimizer_steps']==4,name
        for value in s[name]['net'].values():
            assert bool(torch.isfinite(value).all()),name
    full=list(dict.fromkeys(['legacy_ref',full_reference,'full_off','dual_on']))
    sh=['sh_h0','sh_on'];cb=list(dict.fromkeys(['cb_ref',cb_reference,'cb_h0','cb_off','cb_on']))
    for group in [full,sh,cb]:
        assert len({m[n]['initialized_model_sha256'] for n in group})==1,group
        fingerprints=[]
        for n in group:
            trace=rows(root/n/'input_trace.jsonl')
            fingerprints.append([(r['cloud'],r['labels'],r['whole'],r['part']) for r in trace])
        assert all(x==fingerprints[0] for x in fingerprints),group
        first_losses=[rows(root/n/'input_trace.jsonl')[0]['base_losses'] for n in group]
        assert all(x==first_losses[0] for x in first_losses),('first-step base losses',group)
    # The unchanged reference itself drifts between GPU runs in the historical
    # benchmark protocol. Do not treat its evolved parameters as a golden seed
    # replay; retain strict checks for the matching full reference and CB H0/off.
    for a,b in [(full_reference,'full_off'),('cb_h0','cb_off')]:
        for name,value in s[a]['net'].items():
            assert torch.allclose(value,s[b]['net'][name],rtol=2e-6,atol=2e-7),(a,b,name)
    from inter_hierarchy_MN40.hier_proxy_scratch_v8.geometry import to_ball
    from inter_hierarchy_MN40.hier_proxy_cycle_v9c.geometry import FinalGeometry
    identity_log=type('IdentityLog',(),{'logmap0':staticmethod(lambda x:x)})()
    fake=type('GeometryInput',(),{'manifold':identity_log})()
    for dtype in (torch.float32,torch.float64):
        generator=torch.Generator().manual_seed(4780)
        x=torch.randn(32,256,generator=generator,dtype=dtype).requires_grad_()
        upstream=torch.randn(32,256,generator=generator,dtype=dtype)
        old=to_ball(x,c=.1,clip_r=2.3,hook=True)
        old_grad=torch.autograd.grad(old,x,upstream)[0]
        y=x.detach().clone().requires_grad_();new=FinalGeometry()(fake,(),y)
        new_grad=torch.autograd.grad(new,y,upstream)[0]
        assert torch.equal(old,new) and torch.equal(old_grad,new_grad),dtype
    for name in ['dual_on','sh_on','cb_on']:
        state=s[name]
        assert 'proxy' in state and torch.isfinite(state['proxy']['tangent_proxies']).all(),name
        assert all(int(v['step'])==4 for v in state['proxy_optimizer']['state'].values()),name
    for name in ['full_off','dual_on']:
        counters=[int(v) for k,v in s[name]['net'].items() if k.endswith('num_batches_tracked')]
        assert counters and set(counters)=={8},(name,set(counters))
    aux=rows(root/'dual_on/aux_trace.jsonl')
    assert len(aux)==4
    for row in aux:
        from collections import Counter
        assert sorted(Counter(row['labels']).values())==[2]*16
    for name in ['sh_h0','sh_on','cb_h0','cb_on']:
        assert m[name]['dataset_identity']['train']['size']==4298,name
        assert m[name]['training_config']['num_classes']==20,name
    assert len(s['sh_h0']['metrics']['sample_coverage']['id_draw_counts'])==4298
    repeats={}
    for initial,repeated in [('legacy_ref',full_reference),('cb_ref',cb_reference)]:
        repeats[initial]=dict(reference_used=repeated,max_state_difference=max(float((x-s[repeated]['net'][k]).abs().max())
            for k,x in s[initial]['net'].items() if x.is_floating_point()))
    return dict(passed=True,reference_repeats=repeats,cross_run_parameter_bitwise_reproducibility=False,
        checks=['train-only smokes including declared reference repeats','initialization matches',
        'SH/CB paired augmented inputs and crops match','legacy full-data off equivalence',
        'first-step base losses match references','FP32/FP64 cap GS functional equivalence',
        'same-entry CB H0/off equivalence','four updates each','finite networks and proxy updates',
        'auxiliary BN counter restoration','fresh CB16x2','filtered 20-class identities'],
        smoke_names=names,updated_at=now())


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--data-dir',type=Path,required=True);p.add_argument('--gpu',type=int,required=True)
    p.add_argument('--source-commit',required=True)
    p.add_argument('--verify-only',action='store_true')
    p.add_argument('--full-reference',default='legacy_ref');p.add_argument('--cb-reference',default='cb_ref')
    p.add_argument('--verification-commit');a=p.parse_args()
    if a.verify_only:
        result=verify(a.root,a.full_reference,a.cb_reference)
        result.update(source_commit=a.source_commit,verification_commit=a.verification_commit)
        destination=a.root/'acceptance_verified.json'
        if destination.exists():raise FileExistsError(destination)
        write(destination,result)
        print(json.dumps(result),flush=True)
        return
    a.root.mkdir(parents=True,exist_ok=False)
    tasks=[('legacy_ref',None,'reference'),('full_off','TS53','off'),('dual_on','TS53','enabled'),
        ('sh_h0','TS51','enabled'),('sh_on','TS52','enabled'),('cb_ref','TS54','reference'),
        ('cb_h0','TS54','enabled'),('cb_off','TS55','off'),('cb_on','TS55','enabled')]
    record=dict(state='waiting_for_idle_gpu',source_commit=a.source_commit,pid=os.getpid(),updated_at=now())
    write(a.root/'status.json',record)
    try:
        with open('/tmp/hycore_%d_gpu_%d.lock'%(os.getuid(),a.gpu),'a+') as lock:
            while True:
                try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:time.sleep(15);continue
                first={r['index']:r for r in idle_gpus(query_gpus())}
                if a.gpu in first:
                    time.sleep(2);second={r['index']:r for r in idle_gpus(query_gpus())}
                    if a.gpu in second and first[a.gpu]['uuid']==second[a.gpu]['uuid']:
                        gpu=second[a.gpu];break
                fcntl.flock(lock,fcntl.LOCK_UN);time.sleep(15)
            env={**os.environ,**device_environment([gpu]),'PYTHONHASHSEED':'4780',
                'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2',
                'MPLBACKEND':'Agg','PYTHONDONTWRITEBYTECODE':'1'}
            for name,arm,mode in tasks:
                record.update(state='running_smoke',smoke=name,updated_at=now());write(a.root/'status.json',record)
                if arm is None:
                    command=[sys.executable,'-u',str(HERE.parent/'mn40_ts33_35_20261113/shuffle_runner.py'),
                        '--mode','reference','--variant','c01-cap','--data-dir',str(a.data_dir),
                        '--run-dir',str(a.root/name),'--smoke','--epochs','2','--steps-per-epoch','2']
                else:
                    command=[sys.executable,'-u',str(HERE/'runner.py'),'--arm',arm,'--mode',mode,
                        '--data-dir',str(a.data_dir),'--output',str(a.root/name),'--smoke',
                        '--source-commit',a.source_commit]
                with (a.root/(name+'.log')).open('x') as log:
                    subprocess.run(command,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,check=True)
                print('completed smoke',name,flush=True)
        result=verify(a.root);result['source_commit']=a.source_commit
        write(a.root/'acceptance.json',result);write(a.root/'status.json',dict(state='passed',**result))
        print(json.dumps(result),flush=True)
    except BaseException as error:
        write(a.root/'acceptance.json',dict(passed=False,source_commit=a.source_commit,error=repr(error),updated_at=now()))
        write(a.root/'status.json',dict(state='failed',error=repr(error),updated_at=now()))
        raise


if __name__=='__main__':main()
