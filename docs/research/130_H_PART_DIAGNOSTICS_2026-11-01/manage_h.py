"""User-operated start/stop for H after a diagnostic session; never auto-starts."""
import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from diagnostic_session import now, owned_group, sha, write


def main():
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=('start', 'stop'))
    p.add_argument('--session', required=True)
    args = p.parse_args()
    root = Path(args.session).resolve(strict=True)
    plan = json.loads((root / 'manual_resume.json').read_text())
    active_path = root / 'active_h.json'
    active = json.loads(active_path.read_text()) if active_path.exists() else None
    if args.action == 'stop':
        if not active or not Path('/proc', str(active['pid'])).exists():
            print('No recorded H process is running.')
            return
        pid, out = active['pid'], Path(active['out'])
        owned_group(pid, out)
        if not (out / 'last.pth').exists():
            # No complete epoch yet: retain the previous resume plan/fresh start.
            os.killpg(pid, signal.SIGTERM)
            time.sleep(2)
            try:
                os.killpg(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            print('Stopped H before its first new complete epoch; prior checkpoint retained.')
            return
        os.killpg(pid, signal.SIGSTOP)
        terminated = False
        try:
            import torch
            snap = root / ('manual_stop_' + str(time.time_ns()))
            snap.mkdir()
            for name in ('last.pth', 'best.pth', 'best_selection.json'):
                shutil.copy2(out / name, snap / name)
            state = torch.load(snap / 'last.pth', map_location='cpu', weights_only=False)
            best = torch.load(snap / 'best.pth', map_location='cpu', weights_only=False)
            selection = json.loads((snap / 'best_selection.json').read_text())
            if state['best_epoch'] != best['epoch'] or best['epoch'] != selection['selected_epoch']:
                raise RuntimeError('Checkpoint is being updated; H continues. Retry stop shortly.')
            plan.update(checkpoint=str(snap / 'last.pth'), at=now())
            write(root / 'manual_resume.json', plan)
            os.killpg(pid, signal.SIGTERM)
            terminated = True
            print('Stopped H; saved epoch', state['epoch'], 'updates', state['updates'])
        finally:
            try:
                os.killpg(pid, signal.SIGCONT)
            except ProcessLookupError:
                pass
        if terminated:
            time.sleep(2)
            try:
                os.killpg(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        return
    if active and Path('/proc', str(active['pid'])).exists():
        owned_group(active['pid'], Path(active['out']))
        print('H is already running; PID', active['pid'])
        return
    if json.loads((root / 'status.json').read_text())['state'] != 'diagnostic_finished_H_stopped':
        raise RuntimeError('Diagnostic session has not finished')
    source = Path(plan['source'])
    if subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=source, text=True).strip() != plan['commit']:
        raise RuntimeError('Frozen training commit changed')
    sys.path.insert(0, str(source))
    os.environ.update(CUDA_VISIBLE_DEVICES='2', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
    from segmentation_ShapeNetPart.hycore_seg_v1.train import ensure_idle
    from segmentation_ShapeNetPart.hycore_seg_v6.train import arguments
    import torch
    checkpoint = Path(plan['checkpoint'])
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    best = torch.load(checkpoint.parent / 'best.pth', map_location='cpu', weights_only=False)
    selection = json.loads((checkpoint.parent / 'best_selection.json').read_text())
    if state['best_epoch'] != best['epoch'] or best['epoch'] != selection['selected_epoch']:
        raise RuntimeError('Checkpoint/best identities differ')
    fresh = state['epoch'] >= state['config']['epochs']
    parent = root / ('manual_H_' + str(time.time_ns()))
    out = parent / 'main'
    command = list(plan['command'])
    command[command.index('--out') + 1] = str(out)
    if fresh and '--resume' in command:
        i = command.index('--resume')
        del command[i:i + 2]
    elif fresh:
        pass
    elif '--resume' in command:
        command[command.index('--resume') + 1] = str(checkpoint)
    else:
        command += ['--resume', str(checkpoint)]
    parsed = vars(arguments(command[command.index('segmentation_ShapeNetPart.hycore_seg_v6.train') + 1:]))
    previous = dict(state['config'])
    for field in ('out', 'resume', 'config', 'expected_initial_sha256'):
        previous.pop(field, None)
        parsed.pop(field, None)
    if previous != parsed or state['source']['commit'] != plan['commit']:
        raise RuntimeError('Resume configuration/source differs')
    if sha(Path(state['config']['data']) / 'manifest.json') != state['data_manifest_sha256']:
        raise RuntimeError('Dataset identity differs')
    ensure_idle(2)
    parent.mkdir(exist_ok=False)
    with (parent / 'production.log').open('x') as log:
        proc = subprocess.Popen(command, cwd=source, env=os.environ.copy(), stdin=subprocess.DEVNULL,
                                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    write(active_path, {'pid': proc.pid, 'out': str(out), 'command': command, 'started': now(),
                        'resume_epoch': 0 if fresh else state['epoch'],
                        'initialization': 'fresh' if fresh else 'resume',
                        'checkpoint_sha256': None if fresh else sha(checkpoint)})
    print('Started H; PID', proc.pid, 'initialization', 'fresh' if fresh else 'resume',
          'resume after epoch', 0 if fresh else state['epoch'], 'output', out)


if __name__ == '__main__':
    main()
