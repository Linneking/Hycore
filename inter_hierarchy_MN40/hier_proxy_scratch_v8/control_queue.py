"""Detached, immutable V8 controls; one B64 job plus one shuffle32 job.

CPU acceptance precedes CUDA allocation. Both GPU smokes must finish before
production. Uses only three completely idle GPUs and never interrupts others.
"""
from __future__ import annotations
import argparse
import datetime as dt
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from .queue import (now, write, query_gpus, idle_gpus, device_environment,
                    freeze_source, run_logged, cache_data)


def command(control, data, directory, smoke=False):
    ranks = 2 if control == 'B64' else 1
    result = [sys.executable, '-m', 'torch.distributed.run', '--standalone',
              '--nproc_per_node=' + str(ranks), '--module',
              'inter_hierarchy_MN40.hier_proxy_scratch_v8.controls',
              '--control', control, '--data-dir', str(data), '--run-dir', str(directory)]
    if smoke:
        result += ['--smoke', '--epochs', '2', '--steps-per-epoch', '2',
                   '--skip-final-test', '--smoke-eval-batches', '1']
    return result


def verify_smoke(directory, control, commit):
    manifest = json.loads((directory / 'manifest.json').read_text())
    config = manifest['training_config']
    if (manifest['status'] != 'completed' or manifest['official_test_read'] or
            manifest['completed_epochs'] != 2 or manifest['commit'] != commit or
            config['control'] != control or config['lambda_hier_after_warmup'] != 0 or
            config['P'] != 0 or config['proxy_optimizer'] is not None or
            config['c'] != .1 or config['clip_r'] != 2.3 or
            config['radial_gamma'] != 500 or config['contrastive_margin'] != 2):
        raise RuntimeError('Control smoke identity/configuration failed')
    steps = [json.loads(line) for line in (directory / 'steps.jsonl').read_text().splitlines()]
    if len(steps) != 4 or any(row['hier_calls'] != 0 or row['proxy_parameter_count'] != 0 or
                             row['lambda_hier'] != 0 or row['loss'] != row['base'] for row in steps):
        raise RuntimeError('Control smoke is not a strict base-only run')
    if any(len(row['ids']) != (64 if control == 'B64' else 32) for row in steps):
        raise RuntimeError('Control smoke batch size differs')
    return {'control': control, 'status': 'passed', 'updates': 4,
            'initialized_model_sha256': manifest['initialized_model_sha256'],
            'split_sha256': manifest['split_sha256'], 'test_read': False,
            'last_checkpoint': str(directory / 'last.pth')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--expected-commit', required=True)
    parser.add_argument('--source-branch', required=True)
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=False)
    state = dict(status='preparing', created_utc=now(), queue_pid=os.getpid(),
                 commit=args.expected_commit, branch=args.source_branch,
                 controls=['V8-B64', 'V8-B0-32-C01'], production_started=False,
                 teacher=None, source_checkpoint=None)
    path = args.root / 'queue_status.json'
    write(path, state)
    try:
        snapshot = freeze_source(args.repo, args.root, args.expected_commit)
        env = dict(os.environ)
        env.update(PYTHONPATH=str(snapshot) + ':' + env.get('PYTHONPATH', ''),
            V8_SOURCE_COMMIT=args.expected_commit, V8_SOURCE_BRANCH=args.source_branch,
            OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
            PYTHONUNBUFFERED='1', HDF5_USE_FILE_LOCKING='FALSE',
            PYTHONPYCACHEPREFIX='/tmp/hycore_v8_controls_pycache_' + args.root.name)
        env.pop('CUDA_VISIBLE_DEVICES', None)
        cpu = [sys.executable, '-m', 'unittest',
               'inter_hierarchy_MN40.hier_proxy_scratch_v8.tests.test_controls', '-v']
        if run_logged(cpu, args.root / 'cpu_control_contract.log', env, snapshot, 1200) != 0:
            raise RuntimeError('Control CPU contracts failed')
        data = cache_data(args.data_dir, args.root.name, args.root)
        state.update(cpu_checks_passed=True, status='waiting_for_three_idle_gpus', updated_utc=now())
        write(path, state)
        attempts = 0
        while True:
            rows = query_gpus()
            state.update(gpu_observation=rows, updated_utc=now())
            write(path, state)
            candidates = idle_gpus(rows)
            if len(candidates) < 3:
                time.sleep(30)
                continue
            chosen, locks = candidates[:3], []
            try:
                for gpu in chosen:
                    lock = open(f"/tmp/hycore_{os.getuid()}_gpu_{gpu['index']}.lock", 'a+')
                    locks.append(lock)
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                time.sleep(10)
                available = {row['uuid'] for row in idle_gpus(query_gpus())}
                if not all(gpu['uuid'] in available for gpu in chosen):
                    continue
                attempts += 1
                groups = {'B64': chosen[:2], 'B0-32': chosen[2:]}
                acceptances = []
                for control in ('B64', 'B0-32'):
                    child_env = dict(env, **device_environment(groups[control]))
                    directory = args.root / f'smoke_{control}_{attempts:02d}'
                    state.update(status='smoke_running', smoke_control=control,
                                 gpus=groups, updated_utc=now())
                    write(path, state)
                    if run_logged(command(control, data, directory, True),
                                  directory.with_suffix('.log'), child_env, snapshot, 3600) !=0:
                        raise RuntimeError(control + ' GPU smoke failed; no production launched')
                    acceptances.append(verify_smoke(directory, control, args.expected_commit))
                if len({item['initialized_model_sha256'] for item in acceptances}) !=1:
                    raise RuntimeError('Controls do not share the identical seeded backbone')
                if len({item['split_sha256'] for item in acceptances}) !=1:
                    raise RuntimeError('Control data splits differ')
                write(args.root / 'smoke_acceptance.json', acceptances)
                time.sleep(5)
                available = {row['uuid'] for row in idle_gpus(query_gpus())}
                if not all(gpu['uuid'] in available for gpu in chosen):
                    state.update(status='waiting_for_three_idle_gpus')
                    continue
                children, streams, jobs = {}, [], {}
                for control in ('B64', 'B0-32'):
                    child_env = dict(env, **device_environment(groups[control]))
                    directory = args.root / control
                    cmd = command(control, data, directory)
                    stream = (args.root / (control + '_production.log')).open('x')
                    streams.append(stream)
                    child = subprocess.Popen(cmd, cwd=snapshot, env=child_env,
                        stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT)
                    children[control] = child
                    jobs[control] = dict(pid=child.pid, directory=str(directory), command=cmd,
                                         gpus=groups[control], status='running', started_utc=now())
                state.update(status='production_running', production_started=True,
                             jobs=jobs, updated_utc=now())
                write(path, state)
                while True:
                    for control, child in children.items():
                        code = child.poll()
                        if code is not None and jobs[control]['status'] == 'running':
                            jobs[control].update(status='completed' if code ==0 else 'failed',
                                                 exit_code=code, finished_utc=now())
                    state.update(jobs=jobs, updated_utc=now())
                    write(path, state)
                    if all(child.poll() is not None for child in children.values()):
                        break
                    time.sleep(10)
                for stream in streams:
                    stream.close()
                state.update(status='completed' if all(child.returncode ==0 for child in children.values())
                             else 'production_failed', finished_utc=now())
                write(path, state)
                return
            except BlockingIOError:
                time.sleep(30)
            finally:
                for lock in locks:
                    lock.close()
    except BaseException as exc:
        state.update(status='failed', error=repr(exc), updated_utc=now())
        write(path, state)
        raise


if __name__ == '__main__':
    main()
