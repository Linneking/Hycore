"""Queue whole two-GPU official HIER jobs without claiming occupied GPUs."""
import csv
import datetime
import hashlib
import itertools
import json
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path


root = Path(__file__).resolve().parent
source = root / 'source'
datasets = sys.argv[1:]
gpu_allowlist = os.environ.get('HIER_GPU_ALLOWLIST')
allowed_gpus = {int(value) for value in gpu_allowlist.split(',')} if gpu_allowlist else None


def update(path, **values):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(values, indent=2) + '\n', encoding='utf-8')
    temp.replace(path)


def query(fields, kind):
    lines = subprocess.check_output(['nvidia-smi', f'--query-{kind}={fields}',
                                     '--format=csv,noheader,nounits'], text=True, timeout=10)
    return [[part.strip() for part in row] for row in csv.reader(lines.splitlines()) if row]


def idle():
    busy = {row[0] for row in query('gpu_uuid,pid', 'compute-apps')}
    idle_gpus = {int(idx) for idx, uuid, mem, load in query('index,uuid,memory.used,utilization.gpu', 'gpu')
                 if uuid not in busy and mem.isdigit() and int(mem) < 128 and load == '0'}
    return idle_gpus if allowed_gpus is None else idle_gpus & allowed_gpus


def configuration(dataset, tag):
    script = (source / 'scripts' / 'Resnet50' / f'hier_{dataset}.sh').read_text()
    args = shlex.split(script.split('hier/train.py', 1)[1].replace('\\\n', ' '))
    assert args[args.index('--lambda2') + 1] == '0.5'
    if tag == 'off':
        args[args.index('--lambda2') + 1] = '0'
    return args


jobs = []
for dataset, tag in itertools.product(datasets, ('on', 'off')):
    run = root / f'{dataset}_{tag}'
    args = configuration(dataset, tag)
    jobs.append((dataset, tag, run, args))
    update(run / 'state.json', status='waiting_for_idle_gpu_pair', dataset=dataset, hier=tag,
           args=args, source=(root / 'source.sha256').read_text().strip(), seed=0)

active = {}
previous_idle = set()
while jobs or active:
    for pair, (proc, dataset, tag, run, args) in list(active.items()):
        result = proc.poll()
        if result is None:
            continue
        best = run / 'checkpoint_best.pth'
        log = run / dataset / f'resnet50_{dataset}_{tag}_log.txt'
        epochs = int(args[args.index('--epochs') + 1])
        complete = result == 0 and best.is_file() and log.is_file()
        complete = complete and len(log.read_text().splitlines()) == epochs
        update(run / 'state.json', status='complete' if complete else 'failed',
               dataset=dataset, hier=tag, gpu=list(pair), returncode=result,
               best_exists=best.is_file(), epochs_expected=epochs,
               checkpoint_sha256=hashlib.sha256(best.read_bytes()).hexdigest() if complete else None)
        del active[pair]
    unavailable = {idx for pair in active for idx in pair}
    try:
        current_idle = idle() - unavailable
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        print(f'GPU query unavailable: {exc}', flush=True)
        current_idle = set()
    eligible = sorted(current_idle & previous_idle)
    while len(eligible) >= 2 and jobs:
        index = next((i for i, (ds, _, _, _) in enumerate(jobs)
                      if ds not in {job[1] for job in active.values()}), None)
        if index is None:
            break
        dataset, tag, run, args = jobs.pop(index)
        pair = tuple(eligible[:2])
        del eligible[:2]
        current_idle.difference_update(pair)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=','.join(map(str, pair)))
        command = [sys.executable, '-m', 'torch.distributed.run', '--standalone',
                   '--nproc-per-node=2', 'hier/train.py', *args, '--data_path', str(root / 'data'),
                   '--output_dir', str(run), '--run_name', f'{dataset}_{tag}', '--saveckp_freq', '0']
        with (run / 'training.log').open('x') as log:
            proc = subprocess.Popen(command, cwd=source, env=env, stdout=log, stderr=subprocess.STDOUT)
        update(run / 'state.json', status='running', dataset=dataset, hier=tag, gpu=list(pair), pid=proc.pid,
               start=datetime.datetime.now(datetime.timezone.utc).isoformat(), command=command, seed=0,
               source=(root / 'source.sha256').read_text().strip(),
               initialization='ImageNet ResNet50 V1, fresh PA and HIER parameters')
        active[pair] = (proc, dataset, tag, run, args)
    previous_idle = current_idle
    if jobs or active:
        time.sleep(15)
