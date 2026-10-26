"""Single-dispatch audit95: validate, use an idle GPU, publish, release."""
from __future__ import annotations
import argparse,datetime as dt,json,os
from pathlib import Path
import subprocess,sys,traceback

from docs.research.diagnostics.parthier_embedding_audit94_v1.collect import idle_gpu,write,sha

MODULE='docs.research.diagnostics.parthier_followup95_v1'
def now():return dt.datetime.now(dt.timezone.utc).isoformat()

def main():
    p=argparse.ArgumentParser();p.add_argument('--settings',type=Path,required=True);args=p.parse_args()
    s=json.loads(args.settings.read_text(encoding='utf-8-sig'));root=Path(s['root']);source=Path(s['source'])
    status=root/'status.json'
    if status.exists():raise FileExistsError('Already dispatched; inspect existing status instead')
    state={'status':'running','phase':'starting','pid':os.getpid(),'commit':s['commit'],'started_utc':now(),'children':[]}
    def update(phase=None,**values):
        if phase:state['phase']=phase
        state.update(values);state['updated_utc']=now();write(status,state)
    env=os.environ.copy();env.update(PYTHONPATH=str(s['runtime'])+os.pathsep+str(source),CUDA_VISIBLE_DEVICES='',
        PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',MPLCONFIGDIR=str(root/'mpl_cache'))
    def child(name,module,arguments,child_env):
        with (root/(name+'.log')).open('xb') as log:
            proc=subprocess.Popen([sys.executable,'-u','-m',module,*arguments],cwd=source,env=child_env,
                                  stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
        row={'name':name,'pid':proc.pid,'started_utc':now(),'status':'running'};state['children'].append(row);update()
        code=proc.wait();row.update(returncode=code,status='completed' if code==0 else 'failed',ended_utc=now());update()
        if code:raise RuntimeError(name+' failed; inspect its log')
    update('starting')
    try:
        import fcntl
        with open('/tmp/hycore_%d_gpu_%d.lock'%(os.getuid(),s['gpu']),'a+') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            uuid=idle_gpu(s['gpu'])
            (root/'nvidia_before.txt').write_text(subprocess.check_output(['nvidia-smi'],text=True))
            compute_env=env.copy();compute_env.update(CUDA_VISIBLE_DEVICES=uuid,CUDA_DEVICE_ORDER='PCI_BUS_ID')
            update('analysis',gpu=s['gpu'],gpu_uuid=uuid)
            child('analysis',MODULE+'.analyze',['--settings',str(args.settings.resolve())],compute_env)
        update('report')
        child('report',MODULE+'.publish',['--analysis-dir',str(root/'analysis'),'--out-dir',str(root/'public')],env)
        acceptance=json.loads((root/'public/acceptance.json').read_text())
        if acceptance['figure_groups']!=5:raise RuntimeError('Incomplete publication')
        (root/'nvidia_after.txt').write_text(subprocess.check_output(['nvidia-smi'],text=True))
        write(root/'completion.json',{'status':'completed','commit':s['commit'],'ended_utc':now(),
            'summary_sha256':sha(root/'analysis/summary.json'),'publication':acceptance,'server':'8000',
            'encoder_forwards':0,'optimizer_steps':0,'output_leaf_gradients_only':True})
        update('completed',status='completed',ended_utc=now())
    except Exception as error:
        update(status='failed',error_type=type(error).__name__,error=str(error),ended_utc=now());traceback.print_exc();return 1
    return 0

if __name__=='__main__':raise SystemExit(main())
