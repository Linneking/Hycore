"""Read-only progress for the successfully relocated F/G/H continuation."""
import argparse
import json
import statistics
from datetime import datetime, timedelta, timezone
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--root", type=Path, default=Path.home() / "HyCoRe_runs/pseg3_v6_fgh_resume_pathfixed_s4780_20261031_B")
args = parser.parse_args()
now = datetime.now(timezone(timedelta(hours=8)))
print("Beijing time:", now.strftime("%Y-%m-%d %H:%M:%S"))
print("Scores: Test Instance mIoU / Class mIoU; Class is from the same epoch.")


def rows(path):
    result = []
    if not path.exists():
        return result
    for line in path.read_text().splitlines():
        try:
            result.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return result


def score(item):
    return f"{100*item['instance_miou']:.3f}% / {100*item['class_miou']:.3f}%"


for arm in "fgh":
    print("\n" + "=" * 64)
    try:
        root = args.root / arm
        prepare = json.loads((root / "prepare.json").read_text())
        main = root / "main"
        status = json.loads((main / "status.json").read_text())
        history = rows(Path(prepare["source_history"]) / "source_metrics.jsonl")
        resumed = rows(main / "metrics.jsonl")
        history += resumed
        best = json.loads((main / "best_selection.json").read_text())
        last = history[-1]
        print(f"{arm.upper()} / GPU{prepare['gpu']} / {status['state']}")
        print(f"Epoch {status.get('epoch')}/200; step {status.get('step', '-')}; updates {status.get('updates', '-')}")
        print(f"Resumed after e{prepare['resume_epoch']}; completed new epochs: {len(resumed)}")
        print(f"Latest complete e{last['epoch']}: {score(last['test'])}")
        print(f"Best e{best['selected_epoch']}: {score(best['metrics'])}")
        print(f"Loss {last['train_loss']:.5f}; CE {last['train_ce']:.5f}; R {last['train_reg']:.5f}")
        print("Failure file:", (main / "failure.json").exists())
        if resumed:
            print("Recent model/optimizer finite:", all(
                row['model_finite'] and row['optimizer_finite'] for row in resumed[-10:]))
            seconds = statistics.mean(row['seconds'] for row in resumed[-10:])
            remaining = max(0, 87400 - status.get('updates', last['updates'])) / 437 * seconds
            if status['state'] != "complete":
                print(f"Recent epoch {seconds:.1f}s; remaining ~{remaining/3600:.2f}h; ETA {now + timedelta(seconds=remaining):%m-%d %H:%M} Beijing")
        print("Status updated:", status.get("at", "-"))
    except (OSError, KeyError, ValueError, IndexError) as error:
        print(arm.upper(), "Read error:", error)
