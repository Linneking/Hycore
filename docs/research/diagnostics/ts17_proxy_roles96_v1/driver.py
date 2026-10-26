"""Publish-pinned source snapshot, CPU gates, idle GPU readout, then CPU plots."""
from __future__ import annotations
import argparse,fcntl,hashlib,io,json,os,subprocess,sys,tarfile,time
from pathlib import Path


def write(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


def main():
    p=argparse.ArgumentParser();p.add_argument('--settings',type=Path,required=True);a=p.parse_args();s=json.loads(a.settings.read_text())
    root=Path(s['root']);repo=Path(s['repo']);status=root/'status.json';state=dict(status='preparing',started_utc=__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),pid=os.getpid(),commit=s['commit'])
    write(status,state);lock=None
    try:
        if subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()!=s['commit']:raise RuntimeError('Published source commit differs')
        if subprocess.check_output(['git','-C',str(repo),'status','--porcelain'],text=True).strip():raise RuntimeError('Server source must be clean')
        paths=['docs/research/diagnostics/ts17_proxy_roles96_v1','inter_hierarchy_MN40/hier_proxy_scratch_v5','inter_hierarchy_MN40/hier_proxy_scratch_v6','inter_hierarchy_MN40/hier_proxy_scratch_v8','inter_hierarchy_MN40/hier_proxy_cycle_v9c']
        blob=subprocess.check_output(['git','-C',str(repo),'archive',s['commit']]+paths);snapshot=root/'source';snapshot.mkdir()
        with tarfile.open(fileobj=io.BytesIO(blob)) as t:
            if any(m.name.startswith('/') or '..' in Path(m.name).parts for m in t.getmembers()):raise RuntimeError('Unsafe archive member')
            t.extractall(snapshot)
        source=snapshot/'docs/research/diagnostics/ts17_proxy_roles96_v1'
        identity={str(f.relative_to(snapshot)):hashlib.sha256(f.read_bytes()).hexdigest() for f in snapshot.rglob('*.py')}
        write(root/'source_identity.json',dict(commit=s['commit'],files=identity))
        env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONUNBUFFERED='1',CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',MPLCONFIGDIR=str(root/'mplconfig'))
        env['PYTHONPATH']=':'.join((str(snapshot),str(snapshot/'inter_hierarchy_MN40'),s['runtime'],s['plot_deps']))
        def run(cmd,name,timeout=3600):
            state.update(status=name,updated_utc=__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat());write(status,state)
            with (root/(name+'.log')).open('x') as f:
                process=subprocess.Popen(cmd,cwd=snapshot,env=env,stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
                state.update(child_pid=process.pid);write(status,state)
                code=process.wait(timeout=timeout)
            if code:raise RuntimeError(name+' failed with exit '+str(code))
        run([sys.executable,'-m','unittest','discover','-s',str(source),'-p','test_core.py','-v'],'cpu_contracts',600)
        sys.path[:0]=[str(snapshot),str(snapshot/'inter_hierarchy_MN40')]
        from inter_hierarchy_MN40.hier_proxy_scratch_v8.queue import query_gpus,idle_gpus
        available=idle_gpus(query_gpus());chosen=None
        for candidate in available:
            if candidate['index'] not in (0,1):continue
            handle=open('/tmp/hycore_%s_gpu_%s.lock'%(os.getuid(),candidate['index']),'a+')
            try:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:handle.close();continue
            chosen=candidate;lock=handle;break
        if chosen is None:raise RuntimeError('No completely idle approved GPU; no process interrupted')
        time.sleep(2)
        checked={x['index']:x for x in idle_gpus(query_gpus())}
        if chosen['index'] not in checked or checked[chosen['index']]['uuid']!=chosen['uuid']:raise RuntimeError('GPU no longer idle')
        state['gpu']=chosen;write(status,state);env['CUDA_VISIBLE_DEVICES']=chosen['uuid']
        run([sys.executable,str(source/'pipeline.py'),'--settings',str(a.settings)],'analysis',3600)
        lock.close();lock=None;env['CUDA_VISIBLE_DEVICES']=''
        state['GPU_readout_released']=True;write(status,state)
        run([sys.executable,str(source/'publish.py'),'--settings',str(a.settings)],'publishing',1800)
        from importlib.util import spec_from_file_location,module_from_spec
        spec=spec_from_file_location('core_verify96',source/'core.py');m=module_from_spec(spec);spec.loader.exec_module(m)
        if m.sha(s['checkpoint'])!='d10144896c69ffa215a6622e4077a05749e3547c0fd606a546b0f750ae4af61b' or m.sha(s['cache'])!='b0142fac9d4ce59ddfacf44a9600ccbcce322e9311544a0dca25ce0e5642bfde':raise RuntimeError('Source changed')
        acceptance=dict(passed=True,source_SHA_unchanged=True,code_commit=s['commit'],cpu_contracts_passed=True,network_forwards=0,backward_calls=0,optimizer_updates=0,BN_updates=0,test_objects=0,GPU_released_before_plotting=True)
        m.write(root/'public/acceptance.json',acceptance)
        files={str(f.relative_to(root/'public')):m.sha(f) for f in (root/'public').rglob('*') if f.is_file()}
        m.write(root/'public/artifact_hashes.json',files)
        state.update(status='completed',finished_utc=__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),acceptance=acceptance,public_files=len(files));write(status,state)
    except BaseException as e:
        state.update(status='failed',error=repr(e));write(status,state);raise
    finally:
        if lock is not None:lock.close()


if __name__=='__main__':main()
