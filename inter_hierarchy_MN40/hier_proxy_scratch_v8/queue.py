"""One detached V8 job: immutable source, CPU checks, idle-GPU smoke, production.

Never terminates anyone else's processes. No CUDA allocation until two
devices have no compute apps and zero utilization on consecutive queries.
"""
from __future__ import annotations
import argparse
import datetime as dt
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def write(path, value):
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(temp, path)


def query_gpus():
    rows = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.free,utilization.gpu',
                                    '--format=csv,noheader,nounits'], text=True, timeout=20)
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                                    '--format=csv,noheader,nounits'], text=True, timeout=20)
    occupied = {line.split(',')[0].strip() for line in apps.splitlines() if ',' in line}
    result = []
    for line in rows.splitlines():
        index, uuid, free, util = [field.strip() for field in line.split(',')]
        result.append({'index': int(index), 'uuid': uuid, 'free_MiB': int(free),
                       'utilization': int(util), 'compute_busy': uuid in occupied})
    return result


def idle_gpus(rows):
    return [row for row in rows if not row['compute_busy'] and row['utilization'] == 0
            and row['free_MiB'] >= 30000]


def freeze_source(repo, root, expected):
    actual = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
    if actual != expected:
        raise RuntimeError('Checkout commit changed before snapshot')
    paths = subprocess.check_output(['git', '-C', str(repo), 'ls-files', '-z', '--',
                                    'inter_hierarchy_MN40', 'pointnet2_ops_lib']).split(b'\0')
    snapshot = root / 'code'
    snapshot.mkdir(exist_ok=False)
    hashes = {}
    for encoded in paths:
        if not encoded:
            continue
        relative = Path(os.fsdecode(encoded))
        if relative.suffix != '.py':
            continue
        source = repo / relative
        target = snapshot / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        hashes[str(relative)] = hashlib.sha256(target.read_bytes()).hexdigest()
    sys.path.insert(0, str(repo / 'pointnet2_ops_lib'))
    spec = importlib.util.find_spec('pointnet2_ops._ext')
    if spec is None or not spec.origin or not spec.origin.endswith('.so'):
        raise RuntimeError('Verified pointnet2 binary extension is unavailable')
    ext = Path(spec.origin)
    destination = snapshot / 'pointnet2_ops_lib/pointnet2_ops' / ext.name
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ext, destination)
    hashes[str(destination.relative_to(snapshot))] = hashlib.sha256(destination.read_bytes()).hexdigest()
    write(root / 'source_identity.json', {'commit': actual, 'files': hashes,
                                        'compiled_extension_origin': str(ext), 'frozen_utc': now()})
    return snapshot


def run_logged(command, log, env, cwd, timeout=None):
    with log.open('w', encoding='utf-8') as stream:
        process = subprocess.Popen(command, cwd=cwd, env=env, stdout=stream, stderr=subprocess.STDOUT)
        try:
            return process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            # Only this job's process is ended. torchrun propagates termination to its own workers.
            process.terminate()
            try:
                process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise


def cache_data(source, name, root):
    cache = Path('/tmp') / ('hycore_v8_data_' + name)
    cache.mkdir(exist_ok=False)
    records = []
    train_files = sorted(source.glob('*train*.h5'))
    if not train_files:
        raise RuntimeError('No source training HDF5 files')
    for original in train_files:
        before = original.stat()
        target = cache / original.name
        shutil.copyfile(original, target)
        after = original.stat()
        if before.st_size != target.stat().st_size or before.st_mtime_ns != after.st_mtime_ns:
            raise RuntimeError('Training data changed while copied')
        records.append({'file': original.name, 'bytes': target.stat().st_size,
                        'sha256': hashlib.sha256(target.read_bytes()).hexdigest()})
    # Test bytes are not read here; evaluation follows these links once at the end.
    for original in source.glob('*test*.h5'):
        (cache / original.name).symlink_to(original.resolve())
    write(root / 'training_data_cache.json', {'source': str(source), 'cache': str(cache),
                                            'train_files': records, 'test_content_read': False})
    return cache


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
             'display_name': 'V8-HIER64-K20-W0', 'epochs': 300, 'steps_per_epoch': 200,
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
                gpu_ids = ','.join(str(row['index']) for row in chosen)
                env.update(CUDA_VISIBLE_DEVICES=gpu_ids, HYCORE_PHYSICAL_GPUS=gpu_ids,
                           HYCORE_PHYSICAL_GPU_UUIDS=','.join(row['uuid'] for row in chosen))
                attempts += 1
                smoke = args.root / f'smoke_{attempts:02d}'
                base = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=2',
                        '--module', 'inter_hierarchy_MN40.hier_proxy_scratch_v8.train',
                        '--data-dir', str(cached_data)]
                state.update(status='smoke_running', gpus=chosen, smoke_dir=str(smoke), updated_utc=now())
                write(path, state)
                smoke_command = base + ['--run-dir', str(smoke), '--epochs', '2', '--steps-per-epoch', '2',
                                       '--smoke', '--skip-final-test', '--smoke-eval-batches', '1', '--audit-epochs', '1']
                if run_logged(smoke_command, args.root / f'smoke_{attempts:02d}.log', env, snapshot, timeout=3600) != 0:
                    raise RuntimeError('GPU smoke failed; production was not launched')
                manifest = json.loads((smoke / 'manifest.json').read_text())
                if manifest['status'] != 'completed' or manifest['official_test_read']:
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
