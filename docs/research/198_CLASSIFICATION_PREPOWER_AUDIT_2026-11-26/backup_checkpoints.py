"""Copy only a small, explicit set of completed checkpoints to private offline storage."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile
from datetime import datetime, timezone

parser = argparse.ArgumentParser()
parser.add_argument('--host', required=True)
parser.add_argument('--port', required=True)
parser.add_argument('--kind', choices=('dela', 'weights'), required=True)
parser.add_argument('--out', type=Path, required=True)
args = parser.parse_args()
if args.kind == 'dela':
    paths = ['HyCoRe_runs/de_la_pw_dmr_hier_20261125/formal_seed4780/' + arm + '/work/last.pt'
             for arm in ('h0', 'hier')]
else:
    paths = []
    for arm in ('SO-S12-HIER-L0.1', 'SO-S12-HIER-L1.0'):
        base = 'HyCoRe_runs/scanobject_weights_cb_20261126/runs/' + arm + '/'
        if arm.endswith('L1.0'):
            paths += [base + 'checkpoints/PointNet-toyproof-Offv_pointmlp_hycore/' + name
                      for name in ('best_checkpoint.pth', 'last_checkpoint.pth')]
        paths += [base + 'hier/' + name for name in ('best.pth', 'last.pth', 'first_hier_view.pth', 'initial_proxies.pth')]
remote = '''
from pathlib import Path
import hashlib,io,json,sys,tarfile
from datetime import datetime,timezone
paths=json.load(sys.stdin); records=[]
with tarfile.open(fileobj=sys.stdout.buffer,mode="w|gz",compresslevel=1) as archive:
 for rel in paths:
  p=Path.home()/rel
  content=p.read_bytes()
  item=tarfile.TarInfo(rel);item.size=len(content);item.mtime=int(p.stat().st_mtime)
  archive.addfile(item,io.BytesIO(content))
  records.append(dict(path=rel,bytes=len(content),sha256=hashlib.sha256(content).hexdigest()))
 content=json.dumps(dict(observed_utc=datetime.now(timezone.utc).isoformat(),files=records),indent=2).encode()
 item=tarfile.TarInfo("CHECKPOINT_INDEX.json");item.size=len(content);archive.addfile(item,io.BytesIO(content))
'''
script = base64.b64encode(remote.encode()).decode('ascii')
command = "nice -n 10 /usr/bin/python3 -c 'import base64;exec(base64.b64decode(\"" + script + "\"))'"
args.out.mkdir(parents=True, exist_ok=True)
target = args.out / ('checkpoints_' + args.kind + '.tar.gz')
with target.open('xb') as stream:
    process = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=12', '-p', args.port,
                              args.host, command], input=json.dumps(paths).encode(), stdout=stream,
                             stderr=subprocess.PIPE, timeout=900)
if process.returncode:
    raise SystemExit(process.stderr.decode(errors='replace'))
with tarfile.open(target) as archive:
    index = json.load(archive.extractfile('CHECKPOINT_INDEX.json'))
    for row in index['files']:
        content = archive.extractfile(row['path']).read()
        if len(content) != row['bytes'] or hashlib.sha256(content).hexdigest() != row['sha256']:
            raise RuntimeError('Checkpoint transfer integrity failure')
(args.out / ('checkpoints_' + args.kind + '_index.json')).write_text(json.dumps(index, indent=2), encoding='utf-8')
receipt = dict(verified_utc=datetime.now(timezone.utc).isoformat(), archive=target.name,
               archive_bytes=target.stat().st_size, file_count=len(index['files']),
               verified_all_file_hashes=True, sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
               files=index['files'], private_storage='local .codex-local offline_classification_20261126_1736',
               model_execution=False, training=False)
here = Path(__file__).resolve().parent
(here / ('checkpoint_receipt_' + args.kind + '.json')).write_text(json.dumps(receipt, indent=2), encoding='utf-8')
print(json.dumps(receipt), flush=True)
