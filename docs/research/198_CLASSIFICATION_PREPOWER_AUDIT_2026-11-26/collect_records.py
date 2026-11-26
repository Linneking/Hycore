"""Read selected existing experiment text/code files into a streamed private archive."""
import hashlib
import io
import json
import os
import re
from pathlib import Path
import sys
import tarfile
from datetime import datetime, timezone

request = json.load(sys.stdin)
home = Path.home()
excluded_dirs = {'data', 'dataset', 'datasets', '.git', '__pycache__', 'wandb',
                 'tensorboard', 'node_modules', 'site-packages', '.venv', 'venv'}
code_extensions = {'.py', '.sh', '.yaml', '.yml', '.toml', '.ini', '.cfg', '.md'}
named_text = {'args.txt', 'log.txt', 'out.txt', 'requirements.txt', 'metrics.csv',
              'epoch_curves.csv', 'metrics.jsonl', 'epochs.jsonl', 'cb_sampler.jsonl'}
limit = 8 * 1024 * 1024
total_limit = 512 * 1024 * 1024
records, skipped, missing = [], [], []
seen = set()
total = 0

with tarfile.open(fileobj=sys.stdout.buffer, mode='w|gz', compresslevel=3) as archive:
    for relative in request['roots']:
        top = home / relative
        if not top.exists():
            missing.append(relative)
            continue
        for folder, dirs, files in os.walk(top, followlinks=False):
            dirs[:] = sorted(d for d in dirs if d.lower() not in excluded_dirs and
                             not d.endswith(('.egg', '.dist-info')) and
                             not (Path(folder) / d).is_symlink())
            for name in sorted(files):
                path = Path(folder) / name
                if path.is_symlink() or not path.is_file():
                    continue
                if not (path.suffix.lower() in code_extensions or path.suffix.lower() in {'.json', '.log'}
                        or name in named_text):
                    continue
                rel = str(path.relative_to(home))
                if rel in seen:
                    continue
                seen.add(rel)
                epoch_match = re.fullmatch(r'(?:metrics|sampler)_epoch_(\d+)\.json', name)
                if epoch_match and int(epoch_match[1]) not in {1, 20, 100, 200, 300}:
                    skipped.append(dict(path=rel, reason='sparse historical epoch retention; scalar curves already local'))
                    continue
                size = path.stat().st_size
                if size > limit or total + size > total_limit:
                    skipped.append(dict(path=rel, bytes=size, reason='archive size limit'))
                    continue
                try:
                    content = path.read_bytes()
                except OSError as error:
                    skipped.append(dict(path=rel, reason=type(error).__name__))
                    continue
                entry = tarfile.TarInfo(rel)
                entry.size = len(content)
                entry.mtime = int(path.stat().st_mtime)
                archive.addfile(entry, io.BytesIO(content))
                records.append(dict(path=rel, bytes=len(content), sha256=hashlib.sha256(content).hexdigest()))
                total += len(content)
    index = dict(observed_utc=datetime.now(timezone.utc).isoformat(), scope=request.get('scope'),
                 total_bytes=total, files=records, skipped=skipped, missing_roots=missing,
                 exclusions='No dataset files, model weights, binary feature caches or build binaries; large historical epoch JSON retained at e1/e20/e100/e200/e300 only; oversized streams skipped explicitly')
    content = json.dumps(index, ensure_ascii=False, indent=2).encode('utf-8')
    entry = tarfile.TarInfo('OFFLINE_ARCHIVE_INDEX.json')
    entry.size = len(content)
    archive.addfile(entry, io.BytesIO(content))
