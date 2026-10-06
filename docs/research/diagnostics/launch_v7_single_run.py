"""Detach one authorized V7 from-scratch run after two fresh idle-GPU checks."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from inter_hierarchy_MN40.hier_proxy_scratch_v4.launch_matrix import gpu_inventory


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def atomic_json(path, value):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--python', type=Path, default=Path(sys.executable))
    parser.add_argument('--gpus', type=int, nargs=2, required=True)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--startup-check-seconds', type=float, default=10.)
    args = parser.parse_args()
    if len(set(args.gpus)) != 2 or min(args.gpus) < 0 or not 0 <= args.startup_check_seconds <= 60:
        parser.error('Two distinct GPUs and a bounded startup check are required')
    args.root = args.root.resolve()
    args.data_dir = args.data_dir.resolve(strict=True)
    args.python = args.python.resolve(strict=True)
    if args.root.exists():
        raise FileExistsError(args.root)
    if subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'],
                               cwd=REPO, text=True).strip():
        raise RuntimeError('Tracked server checkout must be clean before launch')
    inventory = gpu_inventory()
    if any(not inventory.get(gpu, {}).get('idle', False) for gpu in args.gpus):
        raise RuntimeError('Assigned GPUs are not completely idle')
    args.root.mkdir(parents=True, exist_ok=False)
    run_dir = args.root / 'V7_HIER64_K20_W0'
    command = [str(args.python), '-u', '-m', 'torch.distributed.run', '--standalone',
               '--nnodes=1', '--nproc-per-node=2', '--module',
               'inter_hierarchy_MN40.hier_proxy_scratch_v7.train',
               '--data-dir', str(args.data_dir), '--run-dir', str(run_dir)]
    if args.smoke:
        command += ['--smoke', '--epochs', '2', '--steps-per-epoch', '2',
                    '--warmup-epochs', '0', '--smoke-eval-batches', '1',
                    '--audit-epochs', '', '--save-every', '1', '--skip-final-test']
    state = {'status': 'dispatching', 'started_utc': now(),
             'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
             'branch': subprocess.check_output(['git', 'branch', '--show-current'], cwd=REPO, text=True).strip(),
             'gpu_inventory': inventory, 'smoke_only': args.smoke,
             'source_checkpoint': None, 'teacher': None, 'seed': 22, 'command': command,
             'protocol': 'V7-HIER64-K20-W0; from scratch; parameter cap depth6; validation selection'}
    atomic_json(args.root / 'launcher_state.json', state)
    try:
        fresh = gpu_inventory()
        if any(not fresh.get(gpu, {}).get('idle', False) for gpu in args.gpus):
            raise RuntimeError('GPU became busy before dispatch')
        environment = os.environ.copy()
        environment.update(CUDA_VISIBLE_DEVICES=','.join(map(str, args.gpus)),
                           CUDA_DEVICE_ORDER='PCI_BUS_ID',
                           HYCORE_PHYSICAL_GPUS=','.join(map(str, args.gpus)),
                           HYCORE_PHYSICAL_GPU_UUIDS=','.join(fresh[gpu]['uuid'] for gpu in args.gpus),
                           OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
                           PYTHONUNBUFFERED='1', TORCH_NCCL_ASYNC_ERROR_HANDLING='1')
        log_path = args.root / 'V7_console.log'
        with log_path.open('x', encoding='utf-8') as stream:
            process = subprocess.Popen(command, cwd=REPO, env=environment,
                                       stdin=subprocess.DEVNULL, stdout=stream,
                                       stderr=subprocess.STDOUT, start_new_session=True)
        state.update(status='dispatched', pid=process.pid, run_dir=str(run_dir),
                     console_log=str(log_path), physical_gpus=args.gpus)
        atomic_json(args.root / 'launcher_state.json', state)
        deadline = time.monotonic() + args.startup_check_seconds
        while time.monotonic() < deadline:
            if process.poll() is not None:
                if args.smoke and process.returncode == 0:
                    break
                raise RuntimeError('Training exited during startup: %s' % process.returncode)
            time.sleep(min(1., max(0., deadline-time.monotonic())))
        state.update(status='smoke_completed' if process.poll() == 0 else 'detached_process_alive',
                     checked_utc=now(), alive_at_launcher_check=process.poll() is None)
        atomic_json(args.root / 'launcher_state.json', state)
        print(json.dumps(state, allow_nan=False), flush=True)
    except BaseException as exc:
        state.update(status='dispatch_failed_or_partial', error=repr(exc), checked_utc=now())
        atomic_json(args.root / 'launcher_state.json', state)
        raise


if __name__ == '__main__':
    main()
