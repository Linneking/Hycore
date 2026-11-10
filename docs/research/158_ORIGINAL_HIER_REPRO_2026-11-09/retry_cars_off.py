"""Launch one fresh Cars PA-only run using the verified official-source deployment."""
import datetime
import hashlib
import json
import os
import selectors
import shlex
import subprocess
import sys
import time
from pathlib import Path


run = Path(__file__).resolve().parent
base = Path(sys.argv[1]).resolve()
requested_devices = sys.argv[2]
automatic = requested_devices == 'auto'
devices = '0,1,2' if automatic else requested_devices
gpu_ids = [int(value) for value in devices.split(',')]
assert len(set(gpu_ids)) == (3 if automatic else 2)
source = base / 'source'
script = source / 'scripts' / 'Resnet50' / 'hier_Cars.sh'
args = shlex.split(script.read_text().split('hier/train.py', 1)[1].replace('\\\n', ' '))
idx = args.index('--lambda2') + 1
assert args[idx] == '0.5'
args[idx] = '0'
data_root = Path(os.environ.get('HIER_DATA_ROOT', str(base / 'data'))).resolve()
assert (data_root / 'cars196' / 'cars_annos.mat').is_file()
assert (base / 'pretrained.sha256').read_text().startswith('0676ba61')
command = [sys.executable, '-m', 'torch.distributed.run', '--standalone',
           '--nproc-per-node=2', 'hier/train.py', *args, '--data_path', str(data_root),
           '--output_dir', str(run), '--run_name', 'Cars_off', '--saveckp_freq', '0']
state = dict(dataset='Cars', hier='off', gpu=gpu_ids, seed=0, command=command,
             source_sha256=(base / 'source.sha256').read_text().strip(),
             pretrained_sha256=(base / 'pretrained.sha256').read_text().split()[0])


def publish(status, **details):
    state.update(status=status, **details)
    path = run / '.state.tmp'
    path.write_text(json.dumps(state, indent=2) + '\n')
    path.replace(run / 'state.json')


def read_probe(command, separator=None):
    proc = subprocess.Popen(command, stdout=subprocess.PIPE)
    received = b''
    try:
        with selectors.DefaultSelector() as selector:
            selector.register(proc.stdout, selectors.EVENT_READ)
            until = time.monotonic() + 10
            while time.monotonic() < until and selector.select(max(0, until - time.monotonic())):
                chunk = os.read(proc.stdout.fileno(), 4096)
                if not chunk:
                    break
                received += chunk
                lines = received.decode(errors='replace').split('\n')[:-1]
                rows = [line.split(separator) for line in lines
                        if line.strip() and not line.lstrip().startswith('#')]
                if rows and {int(row[0]) for row in rows} == set(gpu_ids):
                    return rows
        raise RuntimeError(f'Incomplete GPU probe output: {received!r}')
    finally:
        proc.stdout.close()
        if proc.poll() is None:
            proc.terminate()


def available():
    rows = read_probe(['nvidia-smi', '-i', devices, '--query-gpu=index,memory.used,utilization.gpu',
                       '--format=csv,noheader,nounits'], ',')
    pmon = read_probe(['nvidia-smi', 'pmon', '-i', devices, '-c', '1'])
    busy = {int(row[0]) for row in pmon
            if row[1] != '-' and not (row[2] == 'G' and row[-1] in {'Xorg', 'gnome-shell'})}
    return [int(index) for index, mem, load in rows if int(index) not in busy
            and mem.strip().isdigit() and int(mem) < 128 and load.strip() == '0']


publish('waiting_for_idle_gpu_pair', candidate_gpus=gpu_ids)
try:
    while True:
        ready = available()
        if len(ready) >= 2:
            picked = ready[:2]
            break
        time.sleep(15)
    state['gpu'] = picked
    devices = ','.join(map(str, picked))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=devices,
               PYTHONPATH=str(base / 'deps'), TORCH_HOME=str(base / 'pretrained'),
               OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               PYTHONDONTWRITEBYTECODE='1', TORCH_NCCL_TRACE_BUFFER_SIZE='2048',
               TORCH_NCCL_DUMP_ON_TIMEOUT='1')
    with (run / 'training.log').open('x') as log:
        proc = subprocess.Popen(command, cwd=source, env=env, stdout=log, stderr=subprocess.STDOUT)
    publish('running', pid=proc.pid, start=datetime.datetime.now(datetime.timezone.utc).isoformat())
    code = proc.wait()
    best = run / 'checkpoint_best.pth'
    epochs = run / 'Cars' / 'resnet50_Cars_off_log.txt'
    count = len(epochs.read_text().splitlines()) if epochs.is_file() else 0
    publish('complete' if code == 0 and count == 100 and best.is_file() else 'failed',
            returncode=code, logged_epochs=count,
            best_sha256=hashlib.sha256(best.read_bytes()).hexdigest() if best.is_file() else None)
except Exception as error:
    publish('failed', error=repr(error))
    raise
