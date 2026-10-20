"""V9 native-c1/HIER-shell-c.1 job: CPU gates, idle dual-GPU smoke, fresh300."""
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

def freeze_source(repo, root, expected):
    dirty = subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain', '--untracked-files=no'], text=True)
    if dirty.strip():
        raise RuntimeError('Tracked server checkout must be clean before source freeze')
    snapshot = freeze_legacy_source(repo, root, expected)
    relative = Path('classification_ModelNet40/hutil.py')
    target = snapshot / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(repo / relative, target)
    identity = json.loads((root / 'source_identity.json').read_text())
    identity['files'][str(relative)] = hashlib.sha256(target.read_bytes()).hexdigest()
    write(root / 'source_identity.json', identity)
    return snapshot


def safe_query(state, path):
    try:
        return query_gpus()
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, OSError) as exc:
        state.update(gpu_query_error=repr(exc), gpu_query_failed_utc=now())
        write(path, state)
        return []  # Unknown resources are never treated as idle.


def native_initial_identity(snapshot, env, root):
    command = [sys.executable, '-c',
        "import json; from inter_hierarchy_MN40.hier_proxy_scratch_v9.train import Hype_pointMLP,seed_all,state_hash; "
        "from inter_hierarchy_MN40.hier_proxy_scratch_v9.geometry import configure_backbone; "
        "seed_all(22); m=Hype_pointMLP(); before=state_hash(m); "
        "configure_backbone(m); after=state_hash(m); assert before==after; "
        "print('V9_NATIVE_SHA='+before)"]
    output = subprocess.check_output(command, env=env, cwd=snapshot, text=True, stderr=subprocess.STDOUT, timeout=300)
    (root / 'native_initial_identity.log').write_text(output)
    matching = [line.split('=', 1)[1] for line in output.splitlines() if line.startswith('V9_NATIVE_SHA=')]
    if len(matching) != 1 or len(matching[0]) != 64:
        raise RuntimeError('Native initialization identity contract failed')
    return matching[0]


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
             'display_name': 'V9-HIER64-Cbase1-Chier0.1-H4.6-GShier-M1000_4-TS-S22', 'epochs': 300, 'steps_per_epoch': 200,
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
                   V9_SOURCE_COMMIT=args.expected_commit, V9_SOURCE_BRANCH=args.source_branch)
        env.setdefault('PYTHONPYCACHEPREFIX', '/tmp/hycore_v8_pycache_' + args.root.name)
        env['PYTHONPATH'] = str(snapshot) + ':' + env.get('PYTHONPATH', '')
        env.pop('CUDA_VISIBLE_DEVICES', None)
        cpu_commands = [
            [sys.executable, '-m', 'unittest', 'inter_hierarchy_MN40.hier_proxy_scratch_v8_fulltrain.tests.test_protocol', '-v'],
            [sys.executable, '-m', 'unittest', 'inter_hierarchy_MN40.hier_proxy_scratch_v9.tests.test_geometry',
             'inter_hierarchy_MN40.hier_proxy_scratch_v9.tests.test_base_protocol',
             'inter_hierarchy_MN40.hier_proxy_scratch_v9.tests.test_train_contract', '-v'],
            [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=2',
             '--module', 'inter_hierarchy_MN40.hier_proxy_scratch_v9.tests.ddp_contract']]
        for index, command in enumerate(cpu_commands):
            if run_logged(command, args.root / f'cpu_check_{index}.log', env, snapshot, timeout=1200) != 0:
                raise RuntimeError(f'CPU contract {index} failed; inspect its log')
        expected_native_sha = native_initial_identity(snapshot, env, args.root)
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
                        '--module', 'inter_hierarchy_MN40.hier_proxy_scratch_v9.train',
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
                        manifest['initialized_model_sha256'] != expected_native_sha or
                        manifest['completed_epochs'] != 2 or
                        manifest['training_config']['native_GS'] or
                        manifest['training_config']['native_extra_cap'] is not None or
                        manifest['training_config']['base_c'] != 1. or
                        manifest['training_config']['hier_c'] != .1):
                    raise RuntimeError('Smoke acceptance failed')
                state.update(smoke_passed=True, smoke_commit=manifest['commit'],
                             smoke_native_initial_sha256=manifest['initialized_model_sha256'], updated_utc=now())
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
