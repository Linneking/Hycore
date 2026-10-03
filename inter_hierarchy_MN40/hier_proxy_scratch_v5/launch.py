"""Launch the reviewed V5-H20 and B32 diagnostic on three idle GPUs.

Each job is detached from the launcher/SSH session. Existing run directories
are refused; launcher state and each training manifest record progress.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from ..hier_proxy_scratch_v4.launch_matrix import gpu_inventory

REPO = Path(__file__).resolve().parents[2]


def timestamp():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def write_json(path, value):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def command_for(args, arm):
    common = ['--data-dir', str(args.data_dir), '--run-dir', str(args.root / arm),
              '--epochs', '200', '--steps-per-epoch', '200', '--seed', str(args.seed),
              '--workers', str(args.workers), '--train-eval-every', '10',
              '--audit-epochs', '21,40,100,160,200']
    if arm == 'H20':
        return [str(args.python), '-u', '-m', 'torch.distributed.run', '--standalone',
                '--nnodes=1', '--nproc-per-node=2', '--module',
                'inter_hierarchy_MN40.hier_proxy_scratch_v5.train', *common,
                '--warmup-epochs', '20']
    return [str(args.python), '-u', '-m',
            'inter_hierarchy_MN40.hier_proxy_scratch_v5.train_b32', *common]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--python', type=Path, default=Path(sys.executable))
    p.add_argument('--data-dir', type=Path, required=True)
    p.add_argument('--main-gpus', type=int, nargs=2, required=True)
    p.add_argument('--baseline-gpu', type=int, required=True)
    p.add_argument('--seed', type=int, default=22)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--startup-check-seconds', type=float, default=20)
    args = p.parse_args()
    selected = [*args.main_gpus, args.baseline_gpu]
    if len(set(selected)) != 3 or min(selected) < 0 or args.workers < 0:
        p.error('three distinct nonnegative GPU indices and nonnegative workers required')
    if not 0 <= args.startup_check_seconds <= 60:
        p.error('startup check must be in [0,60] seconds')
    args.python = args.python.resolve(strict=True)
    args.data_dir = args.data_dir.resolve(strict=True)
    args.root = args.root.resolve()
    if not args.data_dir.is_dir() or not args.python.is_file():
        p.error('invalid python/data paths')
    if args.root.exists():
        raise FileExistsError(args.root)
    inventory = gpu_inventory()
    unavailable = [g for g in selected if not inventory.get(g, {}).get('idle', False)]
    if unavailable:
        raise RuntimeError(f'assigned GPUs are not completely idle: {unavailable}')
    args.root.mkdir(parents=True, exist_ok=False)
    state = {'status': 'dispatching', 'started_utc': timestamp(),
             'commit': subprocess.check_output(['git','rev-parse','HEAD'], cwd=REPO, text=True).strip(),
             'branch': subprocess.check_output(['git','branch','--show-current'], cwd=REPO, text=True).strip(),
             'config': {k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
             'gpu_inventory': inventory, 'runs': {},
             'comparison': 'B32 is an operator-stability diagnostic; it is not matched global64 B0.'}
    write_json(args.root / 'launcher_state.json', state)
    processes = {}
    try:
        for arm, physical in [('H20', args.main_gpus), ('B32', [args.baseline_gpu])]:
            fresh = gpu_inventory()
            if any(not fresh.get(g, {}).get('idle', False) for g in physical):
                raise RuntimeError(f'GPU became busy before {arm} dispatch')
            command = command_for(args, arm)
            environment = os.environ.copy()
            environment.update(CUDA_VISIBLE_DEVICES=','.join(map(str, physical)),
                               HYCORE_PHYSICAL_GPUS=','.join(map(str, physical)),
                               HYCORE_PHYSICAL_GPU_UUIDS=','.join(str(fresh[g]['uuid']) for g in physical),
                               OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
                               PYTHONUNBUFFERED='1', TORCH_NCCL_ASYNC_ERROR_HANDLING='1')
            log_path = args.root / (arm + '_console.log')
            with log_path.open('x', encoding='utf-8') as stream:
                process = subprocess.Popen(command, cwd=REPO, env=environment,
                                           stdin=subprocess.DEVNULL, stdout=stream,
                                           stderr=subprocess.STDOUT, start_new_session=True)
            processes[arm] = process
            state['runs'][arm] = {'status': 'dispatched', 'pid': process.pid,
                                  'physical_gpus': physical, 'command': command,
                                  'run_dir': str(args.root / arm), 'console_log': str(log_path),
                                  'started_utc': timestamp()}
            write_json(args.root / 'launcher_state.json', state)
            print(json.dumps({'arm': arm, **state['runs'][arm]}), flush=True)
        deadline = time.monotonic() + args.startup_check_seconds
        while time.monotonic() < deadline:
            for arm, process in processes.items():
                if process.poll() is not None:
                    state['runs'][arm].update(status='early_exit', exit_code=process.returncode)
                    raise RuntimeError(f'{arm} exited during startup: {process.returncode}')
            time.sleep(min(1.0, max(0.0, deadline - time.monotonic())))
        state.update(status='detached_processes_alive', checked_utc=timestamp())
        for record in state['runs'].values():
            record['alive_at_launcher_check'] = True
        write_json(args.root / 'launcher_state.json', state)
        print('DETACHED_DISPATCH_COMPLETE', flush=True)
    except BaseException as exc:
        state.update(status='dispatch_failed_or_partial', error=repr(exc), checked_utc=timestamp())
        write_json(args.root / 'launcher_state.json', state)
        # Preserve any already running authorized job; never kill another run.
        raise


if __name__ == '__main__':
    main()
