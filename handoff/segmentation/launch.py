"""Print or start one registered fresh experiment from the repository root."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    rows = json.loads(Path(__file__).with_name("experiments.json").read_text(encoding="utf-8"))["records"]
    matches = [r for r in rows if args.experiment in (r["id"], r["name"])]
    if len(matches) != 1:
        parser.error("Experiment must identify one registry record")
    row = matches[0]
    command = [sys.executable, "-m", row["module"], "--config", str(root / row["config_file"]),
               *row["original_flags"], "--data", str(Path(args.data).resolve()),
               "--out", str(Path(args.out).resolve()), "--physical-gpu", str(args.gpu)]
    print(shlex.join(command), flush=True)
    if args.run:
        if Path(args.out).exists():
            parser.error("Output directory already exists; provide a new path")
        return subprocess.call(command, cwd=root, env={**os.environ, "CUDA_VISIBLE_DEVICES": str(args.gpu)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
