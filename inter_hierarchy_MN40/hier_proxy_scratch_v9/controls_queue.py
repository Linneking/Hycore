"""Matched V9-B64: no hierarchy calls; CPU gates, idle-GPU smoke, fresh300."""
from __future__ import annotations
import argparse
import fcntl
import json
import hashlib
import shutil
import subprocess
import os
from pathlib import Path
import sys
import time
from ..hier_proxy_scratch_v8.queue import (
    now, write, query_gpus, idle_gpus, device_environment, freeze_source as freeze_legacy_source, run_logged, cache_data)

from .queue import freeze_source, safe_query, native_initial_identity


def verify_smoke(directory, expected_commit, expected_native_sha):
    manifest = json.loads((directory / 'manifest.json').read_text())
    config = manifest['training_config']
    if (manifest['status'] != 'completed' or manifest['official_test_read'] or
            manifest['completed_epochs'] != 2 or manifest['commit'] != expected_commit or
            manifest['train_size'] != 9840 or manifest['validation_size'] != 0 or
            manifest['selection_domain'] != 'smoke_train' or
            manifest['initialized_model_sha256'] != expected_native_sha or
            config['control'] != 'B64' or config['P'] != 0 or
            config['lambda_hier_after_warmup'] != 0 or config['proxy_optimizer'] is not None or
            config['c'] != 1. or config['native_GS'] or config['native_extra_cap'] is not None or
            config['radial_gamma'] != 1000 or config['contrastive_margin'] != 4):
        raise RuntimeError('Matched control smoke identity/configuration failed')
    steps = [json.loads(line) for line in (directory / 'steps.jsonl').read_text().splitlines()]
    if len(steps) != 4 or any(row['hier_calls'] != 0 or row['proxy_parameter_count'] != 0 or
                             row['lambda_hier'] != 0 or row['loss'] != row['base'] or
                             len(row['ids']) != 64 for row in steps):
        raise RuntimeError('Smoke is not a strict global64 base-only control')
    return {'status': 'passed', 'updates': 4, 'hier_calls': 0, 'proxy_parameter_count': 0,
            'initialized_model_sha256': manifest['initialized_model_sha256'],
            'official_test_read': False, 'commit': expected_commit}


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
             'display_name': 'V9-B64-C1-CAPnone-GS0-M1000_4-TS-S22', 'epochs': 300, 'steps_per_epoch': 200,
             'train_size': 9840, 'validation_size': 0, 'selection': 'test2468 each epoch',
             'seed': 22, 'classes_per_rank': 16, 'instances_per_class': 2,
             'source_checkpoint': None, 'teacher': None, 'production_started': False,
             'base_c': 1., 'native_extra_cap': None, 'native_GS': False,
             'radial_gamma': 1000., 'contrastive_margin': 4., 'hier_calls': 0}
    path = args.root / 'queue_status.json'
    write(path, state)
    try:
        snapshot = freeze_source(args.repo, args.root, args.expected_commit)
        env = dict(os.environ)
        env.update(OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
                   PYTHONUNBUFFERED='1', HDF5_USE_FILE_LOCKING='FALSE',
                   V9_SOURCE_COMMIT=args.expected_commit, V9_SOURCE_BRANCH=args.source_branch)
        env.setdefault('PYTHONPYCACHEPREFIX', '/tmp/hycore_v8_pycache_' + args.root.name)
        env['PYTHONPATH'] = str(snapshot) + ':' + env.get('PYTHONPATH', '')
        env.pop('CUDA_VISIBLE_DEVICES', None)
        cpu_commands = [
            [sys.executable, '-m', 'unittest', 'inter_hierarchy_MN40.hier_proxy_scratch_v8_fulltrain.tests.test_protocol', '-v'],
            [sys.executable, '-m', 'unittest', 'inter_hierarchy_MN40.hier_proxy_scratch_v9.tests.test_geometry',
             'inter_hierarchy_MN40.hier_proxy_scratch_v9.tests.test_base_protocol',
             'inter_hierarchy_MN40.hier_proxy_scratch_v9.tests.test_controls', '-v'],
            [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=2',
             '--module', 'inter_hierarchy_MN40.hier_proxy_scratch_v9.tests.ddp_control_contract']]
        for index, command in enumerate(cpu_commands):
            if run_logged(command, args.root / f'cpu_check_{index}.log', env, snapshot, timeout=1200) != 0:
                raise RuntimeError(f'CPU contract {index} failed; inspect its log')
        expected_native_sha = native_initial_identity(snapshot, env, args.root)
        if expected_native_sha != 'fd7251e0eb6b397ecaf3862e9cc00a6bc2be9b684329fdc31dc05915b388d29d':
            raise RuntimeError('Control initialization differs from the existing V9 main')
        state.update(status='preparing_training_data', cpu_checks_passed=True,
                     expected_native_initial_sha256=expected_native_sha, updated_utc=now())
        write(path, state)
        cached_data = cache_data(args.data_dir, args.root.name, args.root)
        state.update(status='waiting_for_two_idle_gpus', cpu_checks_passed=True, updated_utc=now())
        write(path, state)
        attempts = 0
        while True:
            rows = safe_query(state, path)
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
                still_idle = {row['uuid'] for row in idle_gpus(safe_query(state, path))}
                if not all(row['uuid'] in still_idle for row in chosen):
                    continue
                env.update(device_environment(chosen))
                attempts += 1
                smoke = args.root / f'smoke_{attempts:02d}'
                base = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=2',
                        '--module', 'inter_hierarchy_MN40.hier_proxy_scratch_v9.controls',
                        '--control', 'B64', '--data-dir', str(cached_data)]
                state.update(status='smoke_running', gpus=chosen, smoke_dir=str(smoke), updated_utc=now())
                write(path, state)
                smoke_command = base + ['--run-dir', str(smoke), '--epochs', '2', '--steps-per-epoch', '2',
                                       '--smoke', '--skip-final-test', '--smoke-eval-batches', '1', '--audit-epochs', '1']
                if run_logged(smoke_command, args.root / f'smoke_{attempts:02d}.log', env, snapshot, timeout=3600) != 0:
                    raise RuntimeError('GPU smoke failed; production was not launched')
                acceptance = verify_smoke(smoke, args.expected_commit, expected_native_sha)
                write(args.root / 'smoke_acceptance.json', acceptance)
                state.update(smoke_passed=True, smoke_commit=args.expected_commit,
                             smoke_native_initial_sha256=expected_native_sha, updated_utc=now())
                write(path, state)
                time.sleep(5)
                idle_after = {row['uuid'] for row in idle_gpus(safe_query(state, path))}
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
