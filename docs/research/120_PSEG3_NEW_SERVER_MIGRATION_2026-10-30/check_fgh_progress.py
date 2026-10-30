"""Read-only F/G/H progress; no PyTorch, GPU allocation or training writes."""
import argparse
import json
import statistics
from datetime import datetime, timedelta, timezone
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--epoch', type=int, help='Also show a particular completed epoch')
args = parser.parse_args()
base = Path.home() / 'HyCoRe_runs'
matrix = base / 'pseg3_v6_fgh_migrated_s4780_20261030_A'
jobs = [('F', base / 'pseg3_v6_f_gpu3_resume_s4780_20261030_A/main'),
        ('G', matrix / 'g/main'), ('H', matrix / 'h/main')]
zone = timezone(timedelta(hours=8))
now = datetime.now(zone)

def read(path):
    return json.loads(path.read_text())

def rows(path):
    result = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            result.append(json.loads(line))
        except json.JSONDecodeError:
            continue  # A concurrently appended final row may still be incomplete.
    return result

def score(row):
    return f"{100*row['instance_miou']:.3f}% / {100*row['class_miou']:.3f}%"

print('Beijing time:', now.strftime('%Y-%m-%d %H:%M:%S'))
print('Scores: Test Instance mIoU / Class mIoU')
for name, path in jobs:
    print('\n' + '=' * 64)
    try:
        status, manifest = read(path/'status.json'), read(path/'manifest.json')
        history = rows(path/'metrics.jsonl')
        if name == 'F':
            history = [r for r in rows(matrix/'f/main/metrics.jsonl') if r['epoch'] <= 5] + history
        best = read(path/'best_selection.json')
        last = history[-1]
        print(f"{name} / GPU{manifest['gpu']['physical_index']} / {status['state']}")
        print(f"Epoch {status['epoch']}/200; step {status.get('step', '-')}; updates {status.get('updates', '-')}")
        print(f"Latest e{last['epoch']}: {score(last['test'])}")
        print(f"Best   e{best['selected_epoch']}: {score(best['metrics'])}")
        if args.epoch:
            selected = next((r for r in history if r['epoch'] == args.epoch), None)
            print(f"Requested e{args.epoch}: " + (score(selected['test']) if selected else 'not available'))
        print(f"Loss {last['train_loss']:.5f}; CE {last['train_ce']:.5f}; R {last['train_reg']:.5f}")
        recent = history[-10:]
        healthy = all(r['model_finite'] and r['optimizer_finite'] for r in recent)
        print('Recent parameters/optimizer finite:', healthy, '| failure file:', (path/'failure.json').exists())
        seconds = statistics.mean(r['seconds'] for r in recent)
        updates = max(status.get('updates', 0), last['updates'])
        remaining = max(0, 87400-updates)/437*seconds
        if status['state'] == 'complete':
            print('Completed.')
        else:
            finish = now + timedelta(seconds=remaining)
            print(f"Recent epoch {seconds:.1f}s; remaining ~{remaining/3600:.2f}h; ETA {finish:%m-%d %H:%M} Beijing (+/-30-60min)")
        print('Status updated:', status.get('at', '-'))
    except (OSError, KeyError, ValueError, IndexError) as exc:
        print(name, 'Read error:', exc)
