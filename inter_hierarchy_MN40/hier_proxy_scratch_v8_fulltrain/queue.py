"""One fresh V8 full9840/test-selected job; never allocates occupied GPUs."""
from __future__ import annotations
import argparse
import fcntl
import json
import os
from pathlib import Path
import sys
import time
from ..hier_proxy_scratch_v8.queue import (
    now, write, query_gpus, idle_gpus, device_environment, freeze_source, run_logged, cache_data)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--expected-commit', required=True)
    parser.add_argument('--source-branch', required=True)
    parser.add_argument('--poll-seconds', type=int, default=30)
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=False)
    state = {'status': 'preparing', 'created_utc': now(), 'queue_pid': os.getpid(),
             'commit': args.expected_commit, 'branch': args.source_branch,
             'display_name': 'V8-HIER64-FULL9840-TESTSEL', 'epochs': 300, 'steps_per_epoch': 200,
             'train_size': 9840, 'validation_size': 0, 'selection': 'test2468 each epoch',
             'seed': 22, 'classes_per_rank': 16, 'instances_per_class': 2,
             'source_checkpoint': None, 'teacher': None, 'production_started': False}
    path = args.root / 'queue_status.json'
    write(path, state)
    try:
        snapshot = freeze_source(args.repo, args.root, args.expected_commit)
        env = dict(os.environ)
        env.update(OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
                   PYTHONUNBUFFERED='1', HDF5_USE_FILE_LOCKING='FALSE',
                   V8_SOURCE_COMMIT=args.expected_commit, V8_SOURCE_BRANCH=args.source_branch)
        env.setdefault('PYTHONPYCACHEPREFIX', '/tmp/hycore_v8_pycache_' + args.root.name)
        env['PYTHONPATH'] = str(snapshot) + ':' + env.get('PYTHONPATH', '')
        env.pop('CUDA_VISIBLE_DEVICES', None)
        cpu_commands = [
            [sys.executable, '-m', 'unittest', 'inter_hierarchy_MN40.hier_proxy_scratch_v8_fulltrain.tests.test_protocol', '-v'],
            [sys.executable, '-m', 'unittest', 'inter_hierarchy_MN40.hier_proxy_scratch_v8.tests.test_contract', '-v'],
            [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=2',
             '--module', 'inter_hierarchy_MN40.hier_proxy_scratch_v8.tests.ddp_contract']]
        for index, command in enumerate(cpu_commands):
            if run_logged(command, args.root / f'cpu_check_{index}.log', env, snapshot, timeout=1200) != 0:
                raise RuntimeError(f'CPU contract {index} failed; inspect its log')
        state.update(status='preparing_training_data', cpu_checks_passed=True, updated_utc=now())
        write(path, state)
        cached_data = cache_data(args.data_dir, args.root.name, args.root)
        state.update(status='waiting_for_two_idle_gpus', cpu_checks_passed=True, updated_utc=now())
        write(path, state)
        attempts = 0
        while True:
            rows = query_gpus()
            candidates = idle_gpus(rows)
            state.update(gpu_observation=rows, updated_utc=now())
            write(path, state)
            if len(candidates) < 2:
                time.sleep(args.poll_seconds)
                continue
            chosen = candidates[:2]
            locks = []
            try:
                for row in chosen:
                    lock = open(f"/tmp/hycore_{os.getuid()}_gpu_{row['index']}.lock", 'a+')
                    locks.append(lock)
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                time.sleep(10)
                still_idle = {row['uuid'] for row in idle_gpus(query_gpus())}
                if not all(row['uuid'] in still_idle for row in chosen):
                    continue
                env.update(device_environment(chosen))
                attempts += 1
                smoke = args.root / f'smoke_{attempts:02d}'
                base = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=2',
                        '--module', 'inter_hierarchy_MN40.hier_proxy_scratch_v8_fulltrain.train',
                        '--data-dir', str(cached_data)]
                state.update(status='smoke_running', gpus=chosen, smoke_dir=str(smoke), updated_utc=now())
                write(path, state)
                smoke_command = base + ['--run-dir', str(smoke), '--epochs', '2', '--steps-per-epoch', '2',
                                       '--smoke', '--skip-final-test', '--smoke-eval-batches', '1', '--audit-epochs', '1']
                if run_logged(smoke_command, args.root / f'smoke_{attempts:02d}.log', env, snapshot, timeout=3600) != 0:
                    raise RuntimeError('GPU smoke failed; production was not launched')
                manifest = json.loads((smoke / 'manifest.json').read_text())
                if (manifest['status'] != 'completed' or manifest['official_test_read'] or
                        manifest['train_size'] != 9840 or manifest['validation_size'] != 0 or
                        manifest['selection_domain'] != 'smoke_train' or
                        manifest['initialized_model_sha256'] != 'f94e52202df5aaae21f21b8b9c6beaa3925424c4e04a6dcc7d2b84b89af04b73'):
                    raise RuntimeError('Smoke acceptance failed')
                state.update(smoke_passed=True, smoke_commit=manifest['commit'], updated_utc=now())
                write(path, state)
                time.sleep(5)
                idle_after = {row['uuid'] for row in idle_gpus(query_gpus())}
                if not all(row['uuid'] in idle_after for row in chosen):
                    state.update(status='waiting_for_two_idle_gpus')
                    continue
                production = args.root / 'main'
                command = base + ['--run-dir', str(production)]
                state.update(status='production_running', production_started=True,
                             production_dir=str(production), production_command=command,
                             production_started_utc=now(), updated_utc=now())
                write(path, state)
                code = run_logged(command, args.root / 'production.log', env, snapshot)
                state.update(status='completed' if code == 0 else 'production_failed',
                             exit_code=code, finished_utc=now(), updated_utc=now())
                write(path, state)
                return
            except BlockingIOError:
                time.sleep(args.poll_seconds)
            finally:
                for lock in locks:
                    lock.close()
    except BaseException as exc:
        state.update(status='failed', error=repr(exc), updated_utc=now())
        write(path, state)
        raise


if __name__ == '__main__':
    main()
