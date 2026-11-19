"""One persistent, idle-only single-GPU lane for a source-matched arm."""
from __future__ import annotations
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
from inter_hierarchy_MN40.hier_proxy_scratch_v8.queue import query_gpus,idle_gpus,device_environment
sys.path.insert(0,str(HERE))
from support import write,now


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--data-dir',type=Path,required=True)
    p.add_argument('--gpu',type=int,required=True)
    p.add_argument('--arm',choices=['SO-H0','SO-HIER'],required=True)
    p.add_argument('--gate',type=Path,required=True)
    p.add_argument('--source-commit',required=True)
    args=p.parse_args()
    actual=subprocess.check_output(['git','-C',str(HERE.parents[3]),'rev-parse','HEAD'],text=True).strip()
    if actual!=args.source_commit:raise RuntimeError('Frozen source commit mismatch')
    args.root.mkdir(parents=True,exist_ok=True)
    folder=args.root/args.arm
    folder.mkdir(exist_ok=False)
    status=folder/'queue.json'
    with (args.root/('lane_gpu%d.lock'%args.gpu)).open('a+') as lane:
        fcntl.flock(lane,fcntl.LOCK_EX|fcntl.LOCK_NB)
        row=dict(state='waiting_for_preflight',arm=args.arm,gpu=args.gpu,pid=None,
            manager_pid=os.getpid(),source_commit=args.source_commit,created_at=now(),updated_at=now())
        write(status,row)
        while True:
            gate=json.loads(args.gate.read_text()) if args.gate.exists() else {}
            if gate.get('source_commit')==args.source_commit and gate.get('passed') is True:break
            if gate.get('passed') is False:raise RuntimeError('Preflight rejected this source')
            time.sleep(10)
        row.update(state='waiting_for_idle_gpu',updated_at=now());write(status,row)
        with open('/tmp/hycore_%d_gpu_%d.lock'%(os.getuid(),args.gpu),'a+') as device:
            while True:
                try:fcntl.flock(device,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:time.sleep(15);continue
                first={r['index']:r for r in idle_gpus(query_gpus())}
                if args.gpu in first:
                    time.sleep(2)
                    second={r['index']:r for r in idle_gpus(query_gpus())}
                    if args.gpu in second and first[args.gpu]['uuid']==second[args.gpu]['uuid']:
                        gpu=second[args.gpu];break
                fcntl.flock(device,fcntl.LOCK_UN);time.sleep(15)
            env=dict(os.environ,**device_environment([gpu]),PYTHONHASHSEED='4780',
                OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MPLBACKEND='Agg',PYTHONDONTWRITEBYTECODE='1')
            command=[sys.executable,'-u',str(HERE/'runner.py'),'--arm',args.arm,
                '--data-dir',str(args.data_dir),'--output',str(folder/'main'),'--source-commit',args.source_commit]
            with (folder/'main.log').open('x') as log:
                child=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,env=env)
                row.update(state='running',pid=child.pid,gpu_uuid=gpu['uuid'],started_at=now(),updated_at=now());write(status,row)
                code=child.wait()
            row.update(state='completed' if code==0 else 'failed',exit_code=code,updated_at=now());write(status,row)
            if code:raise RuntimeError(args.arm+' failed; output preserved')


if __name__=='__main__':main()
