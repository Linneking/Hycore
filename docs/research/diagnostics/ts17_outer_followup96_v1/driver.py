"""Pin published source/binary, gate CPU accounting, use one idle GPU, then plot."""
from __future__ import annotations
import argparse,os,sys,json,subprocess,fcntl,time,tarfile,io
from pathlib import Path
from datetime import datetime,timezone


def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')


def main():
    p=argparse.ArgumentParser();p.add_argument('--settings',type=Path,required=True);a=p.parse_args();s=json.loads(a.settings.read_text());root=Path(s['root']);repo=Path(s['repo']);lock=None
    state=dict(status='preparing',pid=os.getpid(),commit=s['commit'],started_utc=datetime.now(timezone.utc).isoformat());write(root/'status.json',state)
    try:
        sys.path[:0]=[str(repo),str(repo/'inter_hierarchy_MN40')]
        from inter_hierarchy_MN40.hier_proxy_cycle_v9c.queue import freeze_source
        from inter_hierarchy_MN40.hier_proxy_scratch_v8.queue import query_gpus,idle_gpus
        snapshot=freeze_source(repo,root,s['commit'])
        paths=['docs/research/diagnostics/ts17_outer_followup96_v1','docs/research/diagnostics/ts17_proxy_roles96_v1','docs/research/diagnostics/v9_embedding_review_20261021']
        blob=subprocess.check_output(['git','-C',str(repo),'archive',s['commit']]+paths)
        with tarfile.open(fileobj=io.BytesIO(blob)) as t:t.extractall(snapshot)
        source=snapshot/'docs/research/diagnostics/ts17_outer_followup96_v1'
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',MPLCONFIGDIR=str(root/'mplconfig'))
        env['PYTHONPATH']=':'.join([str(snapshot),str(snapshot/'inter_hierarchy_MN40'),str(snapshot/'classification_ModelNet40'),str(snapshot/'pointnet2_ops_lib'),s['runtime'],s['plot_deps']])
        def run(cmd,name):
            state.update(status=name,updated_utc=datetime.now(timezone.utc).isoformat());write(root/'status.json',state)
            with (root/(name+'.log')).open('x') as out:
                child=subprocess.Popen(cmd,env=env,cwd=snapshot,stdin=subprocess.DEVNULL,stdout=out,stderr=subprocess.STDOUT);state['child_pid']=child.pid;write(root/'status.json',state);code=child.wait(timeout=3600)
            if code:raise RuntimeError(name+' failed '+str(code))
        run([sys.executable,'-m','unittest','discover','-s',str(source),'-p','test_science.py','-v'],'cpu_contracts')
        chosen=None
        for gpu in idle_gpus(query_gpus()):
            if gpu['index'] not in (0,1):continue
            handle=open('/tmp/hycore_%s_gpu_%s.lock'%(os.getuid(),gpu['index']),'a+')
            try:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:handle.close();continue
            chosen=gpu;lock=handle;break
        if chosen is None:raise RuntimeError('No completely idle GPU0/1; nothing interrupted')
        time.sleep(2);again={g['index']:g for g in idle_gpus(query_gpus())}
        if chosen['index'] not in again or again[chosen['index']]['uuid']!=chosen['uuid']:raise RuntimeError('GPU became busy')
        state['gpu']=chosen;env['CUDA_VISIBLE_DEVICES']=chosen['uuid']
        run([sys.executable,str(source/'worker.py'),'--settings',str(a.settings)],'frozen_audit')
        env['CUDA_VISIBLE_DEVICES']='';lock.close();lock=None;state['GPU_released_before_plotting']=True
        run([sys.executable,str(source/'publish.py'),'--root',str(root)],'publishing')
        state.update(status='completed',finished_utc=datetime.now(timezone.utc).isoformat(),optimizer_updates=0,BN_updates=0,backward_calls=0);write(root/'status.json',state)
    except BaseException as e:
        state.update(status='failed',error=repr(e));write(root/'status.json',state);raise
    finally:
        if lock is not None:lock.close()


if __name__=='__main__':main()
