"""Persistent idle-only lane; never shadows Python's standard queue module."""
from __future__ import annotations
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[3]))
from inter_hierarchy_MN40.hier_proxy_scratch_v8.queue import device_environment, idle_gpus, now, query_gpus, write


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--data-dir',type=Path,required=True)
    parser.add_argument('--gpu',type=int,required=True)
    parser.add_argument('--arms',nargs='+',required=True)
    parser.add_argument('--gate',type=Path,required=True)
    parser.add_argument('--source-commit',required=True)
    args=parser.parse_args()
    args.root.mkdir(parents=True,exist_ok=True)
    lane=args.root/('lane_gpu%d.json'%args.gpu)
    # One lane controller per output/GPU. Exiting never touches someone else's process.
    with (args.root/('lane_gpu%d.lock'%args.gpu)).open('a+') as lane_lock:
        fcntl.flock(lane_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for arm in args.arms:
            folder=args.root/arm
            folder.mkdir(exist_ok=True)
            if (folder/'main').exists():
                raise FileExistsError('Production already exists: '+str(folder))
            write(folder/'queue.json',dict(state='waiting_for_preceding_arm',arm=arm,
                gpu=args.gpu,manager_pid=os.getpid(),source_commit=args.source_commit,updated_at=now()))
        for arm in args.arms:
            folder=args.root/arm
            folder.mkdir(exist_ok=True)
            output=folder/'main'
            if output.exists():
                raise FileExistsError('Refusing existing production directory: '+str(output))
            status=folder/'queue.json'
            record=dict(state='waiting_for_preflight',arm=arm,gpu=args.gpu,pid=None,
                manager_pid=os.getpid(),source_commit=args.source_commit,updated_at=now())
            write(status,record);write(lane,record)
            while True:
                gate=json.loads(args.gate.read_text()) if args.gate.exists() else {}
                if gate.get('source_commit')==args.source_commit and gate.get('passed') is True:
                    break
                if gate.get('passed') is False:
                    raise RuntimeError('Preflight rejected: '+str(gate))
                time.sleep(15)
            record.update(state='waiting_for_idle_gpu',updated_at=now())
            write(status,record);write(lane,record)
            with open('/tmp/hycore_%d_gpu_%d.lock'%(os.getuid(),args.gpu),'a+') as device_lock:
                while True:
                    try:
                        fcntl.flock(device_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                    except BlockingIOError:
                        time.sleep(15);continue
                    first={r['index']:r for r in idle_gpus(query_gpus())}
                    if args.gpu in first:
                        time.sleep(2)
                        second={r['index']:r for r in idle_gpus(query_gpus())}
                        if args.gpu in second and first[args.gpu]['uuid']==second[args.gpu]['uuid']:
                            gpu=second[args.gpu];break
                    fcntl.flock(device_lock,fcntl.LOCK_UN);time.sleep(15)
                env={**os.environ,**device_environment([gpu]),'PYTHONHASHSEED':'4780',
                    'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2',
                    'MPLBACKEND':'Agg','PYTHONDONTWRITEBYTECODE':'1'}
                command=[sys.executable,'-u',str(HERE/'runner.py'),'--arm',arm,
                    '--data-dir',str(args.data_dir),'--output',str(output),
                    '--source-commit',args.source_commit]
                with (folder/'main.log').open('x') as log:
                    process=subprocess.Popen(command,env=env,stdin=subprocess.DEVNULL,
                        stdout=log,stderr=subprocess.STDOUT)
                    record.update(state='running',pid=process.pid,started_at=now(),updated_at=now(),gpu_uuid=gpu['uuid'])
                    write(status,record);write(lane,record)
                    code=process.wait()
                record.update(state='completed' if code==0 else 'failed',exit_code=code,updated_at=now())
                write(status,record);write(lane,record)
                if code:raise RuntimeError(arm+' failed; no subsequent lane job is started')
        write(lane,dict(state='completed',arms=args.arms,gpu=args.gpu,updated_at=now()))


if __name__=='__main__':main()
