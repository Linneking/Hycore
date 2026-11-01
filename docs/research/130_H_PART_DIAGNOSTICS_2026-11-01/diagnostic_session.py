"""Pause an explicitly identified owned trainer and diagnose; leave H stopped.

All machine-specific locations are supplied by an untracked server JSON config.
Existing run artifacts are read-only. This supervisor does not allocate CUDA.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def write(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    os.replace(temporary, path)


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(4194304), b''):
            h.update(chunk)
    return h.hexdigest()


def owned_group(pid, expected_out):
    proc = Path('/proc') / str(pid)
    if proc.stat().st_uid != os.getuid() or os.getpgid(pid) != pid:
        raise RuntimeError('Trainer ownership/process-group identity differs')
    cmd = proc.joinpath('cmdline').read_bytes().decode().strip('\x00').split('\x00')
    if ('segmentation_ShapeNetPart.hycore_seg_v6.train' not in cmd
            or cmd[cmd.index('--out') + 1] != str(expected_out)
            or cmd[cmd.index('--physical-gpu') + 1] != '2'):
        raise RuntimeError('Trainer command does not match the authorized H/GPU2 run')
    for entry in Path('/proc').iterdir():
        if entry.name.isdigit():
            try:
                if os.getpgid(int(entry.name)) == pid and entry.stat().st_uid != os.getuid():
                    raise RuntimeError('Process group contains an unexpected owner')
            except ProcessLookupError:
                pass
    return cmd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    root = Path(cfg['root'])
    root.mkdir(exist_ok=False)
    write(root / 'session_config.json', cfg)
    pid, old = cfg['trainer_pid'], Path(cfg['trainer_out'])
    command = owned_group(pid, old)
    source = Path(cfg['source'])
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=source, text=True).strip()
    if commit != cfg['training_commit']:
        raise RuntimeError('Frozen training source differs')
    inputs = root / 'resume_inputs'
    inputs.mkdir()
    stopped = False
    paused = False
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES='2', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
    try:
        # Stop the whole owned group briefly while copying a consistent saved state.
        os.killpg(pid, signal.SIGSTOP)
        paused = True
        import torch
        for name in ('last.pth', 'best.pth', 'best_selection.json', 'manifest.json', 'status.json'):
            shutil.copy2(old / name, inputs / name)
        last = torch.load(inputs / 'last.pth', map_location='cpu', weights_only=False)
        best = torch.load(inputs / 'best.pth', map_location='cpu', weights_only=False)
        selected = json.loads((inputs / 'best_selection.json').read_text())
        if last['best_epoch'] != best['epoch'] or last['best_epoch'] != selected['selected_epoch']:
            raise RuntimeError('Checkpoint set was captured during a save; original training will continue')
        if last['source']['commit'] != commit or last['config']['physical_gpu'] != 2:
            raise RuntimeError('Saved H checkpoint identity differs')
        if last['epoch'] >= last['config']['epochs']:
            raise RuntimeError('H already completed; do not restart it')
        sys.path.insert(0, str(source))
        from segmentation_ShapeNetPart.hycore_seg_v6.train import arguments
        proposed = list(command[command.index('segmentation_ShapeNetPart.hycore_seg_v6.train') + 1:])
        proposed[proposed.index('--out') + 1] = str(root / 'manual_H' / 'main')
        if '--resume' in proposed:
            proposed[proposed.index('--resume') + 1] = str(inputs / 'last.pth')
        else:
            proposed += ['--resume', str(inputs / 'last.pth')]
        parsed = vars(arguments(proposed))
        previous = dict(last['config'])
        for field in ('out', 'resume', 'config', 'expected_initial_sha256'):
            previous.pop(field, None)
            parsed.pop(field, None)
        if previous != parsed or sha(Path(cfg['data']) / 'manifest.json') != last['data_manifest_sha256']:
            raise RuntimeError('Full resume configuration or dataset identity differs')
        hashes = {name: sha(inputs / name) for name in ('last.pth', 'best.pth')}
        for name, digest in hashes.items():
            if digest != sha(old / name):
                raise RuntimeError('Checkpoint copy hash mismatch')
        write(root / 'pause.json', {'at': now(), 'pid': pid, 'epoch': last['epoch'],
              'updates': last['updates'], 'hashes': hashes,
              'observed_status': json.loads((inputs / 'status.json').read_text())})
        del last, best
        os.killpg(pid, signal.SIGTERM)
        stopped = True
        try:
            os.killpg(pid, signal.SIGCONT)
        except ProcessLookupError:
            pass
        paused = False
        write(root / 'manual_resume.json', {'command': command, 'source': str(source),
              'commit': commit, 'checkpoint': str(inputs / 'last.pth'), 'gpu': 2,
              'authorization': 'User will start H manually after diagnostics', 'at': now()})
        for _ in range(20):
            try:
                os.killpg(pid, 0)
            except ProcessLookupError:
                break
            time.sleep(.5)
        else:
            # Only the already verified owned process group can be signalled here.
            os.killpg(pid, signal.SIGKILL)
        write(root / 'status.json', {'state': 'diagnosing', 'at': now()})
        diag = [cfg['python'], cfg['diagnostic_script'], '--source', str(source),
                '--data', cfg['data'], '--runs-json', cfg['runs_json'],
                '--out', str(root / 'diagnostic'), '--gpu', '2', '--batch-size', '16']
        with (root / 'diagnostic.log').open('x') as log:
            result = subprocess.run(diag, cwd=source, env=env, stdin=subprocess.DEVNULL,
                                    stdout=log, stderr=subprocess.STDOUT)
        write(root / 'diagnostic_exit.json', {'returncode': result.returncode, 'at': now()})
    except Exception as exc:
        write(root / 'failure.json', {'error': repr(exc), 'at': now()})
        raise
    finally:
        if paused:
            try:
                os.killpg(pid, signal.SIGCONT)
            except ProcessLookupError:
                pass
        if stopped:
            write(root / 'status.json', {'state': 'diagnostic_finished_H_stopped',
                                        'at': now(), 'manual_resume_required': True})


if __name__ == '__main__':
    main()
