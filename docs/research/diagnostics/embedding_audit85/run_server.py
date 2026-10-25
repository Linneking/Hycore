"""Audit85 collection on idle server GPUs; no training or test-set access.

Private settings identify selected immutable checkpoints/caches. Each run owns
a new directory. Published sources and the existing PointNet2 binary are frozen
before regression and clean inference. Existing result directories are read-only.
"""
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


def load_module(path, name):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--settings', type=Path, required=True)
    args = parser.parse_args()
    settings = json.loads(args.settings.read_text(encoding='utf-8-sig'))
    repo, root = Path(settings['repo']), Path(settings['root'])
    for path in (repo, repo / 'inter_hierarchy_MN40'):
        sys.path.insert(0, str(path))
    from inter_hierarchy_MN40.hier_proxy_cycle_v9c.queue import freeze_source
    from inter_hierarchy_MN40.hier_proxy_scratch_v8.queue import now, write, query_gpus, idle_gpus, run_logged
    collect_path = Path('docs/research/diagnostics/v9_embedding_review_20261021/collect.py')
    here = Path('docs/research/diagnostics/embedding_audit85')
    collector = load_module(repo / collect_path, 'audit85_legacy_collect')
    status = dict(status='preparing', pid=os.getpid(), started_utc=now(),
                  commit=settings['commit'], optimizer_updates=0, backward_calls=0,
                  test_forwarded=False, BN_recalibrated=False)
    state_path = root / 'collection_status.json'
    if state_path.exists():
        raise FileExistsError('An existing audit must not be restarted or overwritten')
    write(state_path, status)
    locks, processes, streams = [], [], []
    try:
        snapshot = freeze_source(repo, root, settings['commit'])
        identity = json.loads((root / 'source_identity.json').read_text())
        for relative_dir in (collect_path.parent, here):
            target_dir = snapshot / relative_dir
            target_dir.mkdir(parents=True, exist_ok=True)
            tracked = set(subprocess.check_output(['git', '-C', str(repo), 'ls-files', '-z', '--', relative_dir.as_posix()]).decode().split('\0'))
            for source in (repo / relative_dir).glob('*.py'):
                relative = relative_dir / source.name
                if relative.as_posix() not in tracked:
                    raise RuntimeError('Only published diagnostic sources may run')
                target = snapshot / relative
                shutil.copy2(source, target)
                identity['files'][relative.as_posix()] = collector.sha(target)
        write(root / 'source_identity.json', identity)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
                   OPENBLAS_NUM_THREADS='2', PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1',
                   HDF5_USE_FILE_LOCKING='FALSE')
        env['PYTHONPATH'] = ':'.join((str(snapshot), str(snapshot / 'inter_hierarchy_MN40'), str(snapshot / 'pointnet2_ops_lib'), env.get('PYTHONPATH', '')))
        staged = json.loads((root / 'staged/staged_manifest.json').read_text())
        rows = []
        for value in staged['observations']:
            row = dict(value)
            row['checkpoint'] = str(root / 'staged' / Path(row['checkpoint']).name)
            if collector.sha(row['checkpoint']) != row['staged_checkpoint_sha256']:
                raise RuntimeError('Transferred frozen weight SHA differs')
            row.pop('run_dir', None)  # Existing usage was captured at the actual source epoch.
            if row['key'] in ('TS14', 'B05_A1'):
                row['kind'] = 'cycle_hier'
            elif row['key'] == 'NATIVE_A1':
                row['kind'] = 'v9_hier'
            else:
                raise ValueError('Unexpected primary checkpoint')
            rows.append(row)
        if len(rows) != 6:
            raise ValueError('Exactly six missing primary endpoints are required')
        selected_gpus = settings.get('gpus', [0, 1])
        observed = {row['index']: row for row in idle_gpus(query_gpus())}
        if not all(gpu in observed for gpu in selected_gpus):
            raise RuntimeError('Requested audit GPU is occupied; nothing interrupted')
        for gpu in selected_gpus:
            lock = open(f'/tmp/hycore_{os.getuid()}_gpu_{gpu}.lock', 'a+')
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            locks.append(lock)
        time.sleep(5)
        checked = {row['index']: row for row in idle_gpus(query_gpus())}
        if not all(gpu in checked and checked[gpu]['uuid'] == observed[gpu]['uuid'] for gpu in selected_gpus):
            raise RuntimeError('Audit GPU became occupied before allocation')
        status.update(status='golden_regression', gpus=[checked[g] for g in selected_gpus], updated_utc=now())
        write(state_path, status)
        reference = settings['golden']
        command = [sys.executable, str(snapshot / here / 'gate.py'), '--repo', str(snapshot),
                   '--data-dir', settings['data_dir'], '--checkpoint', reference['checkpoint'],
                   '--expected-checkpoint-sha', reference['checkpoint_sha256'],
                   '--reference-cache', reference['cache'], '--expected-input-sha', settings['input_sha256'],
                   '--gpu', str(selected_gpus[0]), '--out', str(root / 'golden_regression.json')]
        if run_logged(command, root / 'golden_regression.log', env, snapshot, timeout=900) != 0:
            raise RuntimeError('Golden regression failed; no population export started')
        if not json.loads((root / 'golden_regression.json').read_text())['passed']:
            raise RuntimeError('Golden regression was not accepted')
        # The bounded gate exits and releases its CUDA context before export.
        for offset, gpu in enumerate(selected_gpus):
            part = rows[offset::len(selected_gpus)]
            spec = dict(repo=str(snapshot), data_dir=settings['data_dir'], observations=part)
            spec_path = root / ('export_spec_gpu%d.json' % gpu)
            write(spec_path, spec)
            command = [sys.executable, str(snapshot / collect_path), 'export', '--spec', str(spec_path),
                       '--out', str(root / ('gpu%d' % gpu)), '--gpu', str(gpu)]
            stream = (root / ('export_gpu%d.log' % gpu)).open('x')
            streams.append(stream)
            processes.append(subprocess.Popen(command, cwd=snapshot, env=env, stdin=subprocess.DEVNULL,
                                              stdout=stream, stderr=subprocess.STDOUT))
        status.update(status='exporting_six_primary_caches', worker_pids=[p.pid for p in processes], updated_utc=now())
        write(state_path, status)
        codes = [process.wait() for process in processes]
        if any(codes):
            raise RuntimeError('Frozen export failed: ' + repr(codes))
        import numpy as np
        old_root = Path(settings['old_cache_root'])
        old = json.loads((old_root / 'export_manifest.json').read_text())
        if old['input_sha256'] != settings['input_sha256']:
            raise RuntimeError('Legacy cached input differs')
        observations = []
        rename = {'cycle_off': ('TS13', True), 'orig_c01_gs': ('TS06_B05_H0', False),
                  'orig_history': ('TS01_HIST', False), 'v9_b64': ('TS11', False), 'v9_hier': ('TS12', False)}
        for source in old['observations']:
            if source['key'] not in rename or source['stage'] not in ('best', 'e300'):
                continue
            row = dict(source)
            row['key'], row['primary'] = rename[source['key']]
            row['run_id'] = row['key']
            row['cache'] = str(old_root / row['cache'])
            row['cache_reused'] = True
            if row['key'] == 'TS01_HIST':
                row['raw_epoch_metadata'] = row['epoch']
                row['epoch'] = row['epoch'] + 1
                row['label'] = 'Original historical (not R2)'
            observations.append(row)
        for gpu in selected_gpus:
            directory = root / ('gpu%d' % gpu)
            exported = json.loads((directory / 'export_manifest.json').read_text())
            if exported['status'] != 'completed' or exported['input_sha256'] != settings['input_sha256']:
                raise RuntimeError('Population export input or completion identity differs')
            for source in exported['observations']:
                row = dict(source, cache=str(directory / source['cache']), cache_reused=False, primary=True)
                row['run_id'] = row['key']
                observations.append(row)
        ids, labels = None, None
        for row in observations:
            with np.load(row['cache'], allow_pickle=False) as cache:
                if cache['native_mu'].shape != (9840, 256) or not np.isfinite(cache['native_mu']).all():
                    raise ValueError('Expected finite9840x256 whole cache')
                if ids is None:
                    ids, labels = cache['canonical_ids'].copy(), cache['labels'].copy()
                if not np.array_equal(ids, cache['canonical_ids']) or not np.array_equal(labels, cache['labels']):
                    raise ValueError('Canonical IDs/labels differ across caches')
                if row['key'] == 'TS13':
                    source = Path(settings['ts13_run']) / ('best.pth' if row['stage'] == 'best' else 'last.pth')
                    if collector.sha(source) != row['checkpoint_sha256']:
                        raise RuntimeError('TS13 reused cache differs from actual fixed weight')
            row['cache_sha256'] = collector.sha(row['cache'])
            row['source_checkpoint_sha256'] = row['checkpoint_sha256']
        order = {'TS13': 0, 'TS14': 1, 'B05_A1': 2, 'NATIVE_A1': 3, 'TS06_B05_H0': 4,
                 'TS01_HIST': 5, 'TS11': 6, 'TS12': 7}
        observations.sort(key=lambda r: (order[r['run_id']], r['stage'] != 'best'))
        if len(observations) != 16 or sum(r['primary'] for r in observations) != 8:
            raise ValueError('Expected eight primary and eight auxiliary observations')
        write(root / 'analysis_manifest.json', dict(observations=observations, input_sha256=settings['input_sha256'],
              canonical_id_policy='sorted HDF5 filename then row, first1024 raw points, full9840 clean/eval',
              representative_reference='TS13_best', collection_commit=settings['commit']))
        status.update(status='completed', observations=len(observations), primary_observations=8,
                      regression_passed=True, finished_utc=now(), updated_utc=now())
        write(state_path, status)
    except BaseException as exc:
        status.update(status='failed', error=repr(exc), updated_utc=now())
        write(state_path, status)
        raise
    finally:
        for process in processes:
            if process.poll() is None:
                process.wait()
        for stream in streams:
            stream.close()
        for lock in locks:
            lock.close()


if __name__ == '__main__':
    main()
