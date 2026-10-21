"""Freeze reviewed code and run one bounded train-only V9 diagnostic on idle GPUs."""
from __future__ import annotations
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from inter_hierarchy_MN40.hier_proxy_scratch_v8.queue import (
    now, write, query_gpus, idle_gpus, device_environment, run_logged,
)
from inter_hierarchy_MN40.hier_proxy_scratch_v9.queue import freeze_source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--expected-commit', required=True)
    parser.add_argument('--source-branch', required=True)
    parser.add_argument('--diagnostic-epochs', type=int, default=10)
    args = parser.parse_args()
    if not 5 <= args.diagnostic_epochs <= 10:
        parser.error('Authorized observation is 5..10 epochs')
    args.root.mkdir(parents=True, exist_ok=False)
    status_path = args.root / 'queue_status.json'
    state = {'status': 'preparing', 'queue_pid': os.getpid(), 'created_utc': now(),
             'commit': args.expected_commit, 'branch': args.source_branch,
             'diagnostic_epochs': args.diagnostic_epochs, 'steps_per_epoch': 200,
             'scheduler_epochs': 300, 'seed': 22, 'teacher': None,
             'source_checkpoint': None, 'test_read': False, 'selection': None}
    write(status_path, state)
    try:
        snapshot = freeze_source(args.repo, args.root, args.expected_commit)
        relative = Path('docs/research/diagnostics/v9_proxy_cap_cause_20261021')
        target = snapshot / relative
        target.mkdir(parents=True, exist_ok=False)
        identity = json.loads((args.root / 'source_identity.json').read_text())
        for source in (args.repo / relative).glob('*.py'):
            destination = target / source.name
            shutil.copy2(source, destination)
            identity['files'][str(relative / source.name)] = hashlib.sha256(destination.read_bytes()).hexdigest()
        write(args.root / 'source_identity.json', identity)
        env = dict(os.environ)
        env.update(OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
                   PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1',
                   HDF5_USE_FILE_LOCKING='FALSE',
                   V9_SOURCE_COMMIT=args.expected_commit, V9_SOURCE_BRANCH=args.source_branch)
        env['PYTHONPATH'] = str(snapshot) + ':' + str(snapshot / 'inter_hierarchy_MN40') + ':' + env.get('PYTHONPATH', '')
        env.pop('CUDA_VISIBLE_DEVICES', None)
        tests = [sys.executable, '-m', 'unittest', 'test_runner_contracts',
                 'test_optimizer_step_audit', 'test_same_graph_observer', '-v']
        if run_logged(tests, args.root / 'cpu_contracts.log', env, target, timeout=300) != 0:
            raise RuntimeError('Diagnostic CPU contracts failed')
        geometry_tests = [sys.executable, '-m', 'unittest',
            'inter_hierarchy_MN40.hier_proxy_scratch_v9.tests.test_geometry',
            'inter_hierarchy_MN40.hier_proxy_scratch_v9.tests.test_base_protocol', '-v']
        if run_logged(geometry_tests, args.root / 'geometry_contracts.log', env, snapshot, timeout=300) != 0:
            raise RuntimeError('Existing V9 geometry/base contracts failed')
        state.update(status='waiting_for_two_idle_gpus', cpu_checks_passed=True, updated_utc=now())
        write(status_path, state)
        while True:
            rows = query_gpus()
            candidates = idle_gpus(rows)
            state.update(gpu_observation=rows, updated_utc=now())
            write(status_path, state)
            if len(candidates) < 2:
                time.sleep(30)
                continue
            chosen, locks = candidates[:2], []
            try:
                for row in chosen:
                    lock = open(f"/tmp/hycore_{os.getuid()}_gpu_{row['index']}.lock", 'a+')
                    locks.append(lock)
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                time.sleep(5)
                confirmed = {row['uuid'] for row in idle_gpus(query_gpus())}
                if not all(row['uuid'] in confirmed for row in chosen):
                    continue
                env.update(device_environment(chosen))
                base = [sys.executable, '-m', 'torch.distributed.run', '--standalone',
                        '--nproc_per_node=2', str(target / 'runner.py'),
                        '--data-dir', str(args.data_dir), '--workers', '2']
                state.update(status='smoke_running', gpus=chosen, snapshot=str(snapshot), updated_utc=now())
                write(status_path, state)
                smoke = args.root / 'smoke'
                if run_logged(base + ['--run-dir', str(smoke), '--smoke', '--diagnostic-epochs', '2',
                                      '--steps-per-epoch', '2'], args.root / 'smoke.log', env, snapshot, timeout=600) != 0:
                    raise RuntimeError('GPU 2x2 diagnostic smoke failed; main not launched')
                manifest = json.loads((smoke / 'manifest.json').read_text())
                if not (manifest['status'] == 'completed' and manifest['completed_epochs'] == 2
                        and not manifest['official_test_read'] and not manifest['test_used_for_selection']
                        and manifest['selection_size'] == 0
                        and manifest['training_config']['epochs'] == 300):
                    raise RuntimeError('Diagnostic smoke manifest acceptance failed')
                state.update(smoke_passed=True, smoke_initialization=manifest['initialization'],
                             smoke_model_sha256=manifest['initialized_model_sha256'], updated_utc=now())
                write(status_path, state)
                confirmed = {row['uuid'] for row in idle_gpus(query_gpus())}
                if not all(row['uuid'] in confirmed for row in chosen):
                    raise RuntimeError('Chosen GPUs no longer idle after smoke; diagnostic not launched')
                main_run = args.root / 'main'
                command = base + ['--run-dir', str(main_run), '--diagnostic-epochs', str(args.diagnostic_epochs)]
                state.update(status='diagnostic_running', main_dir=str(main_run), main_command=command,
                             started_utc=now(), updated_utc=now())
                write(status_path, state)
                exit_code = run_logged(command, args.root / 'diagnostic.log', env, snapshot)
                state.update(status='completed' if exit_code == 0 else 'diagnostic_failed',
                             exit_code=exit_code, finished_utc=now(), updated_utc=now())
                write(status_path, state)
                return
            except BlockingIOError:
                time.sleep(30)
            finally:
                for lock in locks:
                    lock.close()
    except BaseException as exc:
        state.update(status='failed', error=repr(exc), updated_utc=now())
        write(status_path, state)
        raise


if __name__ == '__main__':
    main()
