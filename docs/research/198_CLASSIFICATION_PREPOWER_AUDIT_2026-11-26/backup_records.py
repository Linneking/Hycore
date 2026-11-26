"""Stream selected existing records home and verify every archived file locally."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tarfile
from datetime import datetime, timezone

parser = argparse.ArgumentParser()
parser.add_argument('--host', required=True)
parser.add_argument('--port', required=True)
parser.add_argument('--out', type=Path, required=True)
parser.add_argument('--label', default='records_curated')
args = parser.parse_args()
here = Path(__file__).resolve().parent
previous = here.parent / '197_CLASSIFICATION_REPORT_DRAFT_2026-11-26'
inventory = json.loads((previous / 'ts_inventory.json').read_text(encoding='utf-8-sig'))
roots = set()
for row in inventory['runs']:
    relative = row.get('original_run_directory_id') or row.get('run_directory_id', '')
    if relative and row.get('directory_relative_to') != 'code repository root':
        first = relative.split('/')[0]
        if re.fullmatch(r'[A-Za-z0-9_-]+', first):
            roots.add('HyCoRe_runs/' + first)
audit = json.loads((previous / 'non_ts_log_audit.json').read_text(encoding='utf-8-sig'))
for snapshot in audit['snapshots']:
    for row in snapshot['results']:
        roots.add('HyCoRe_runs/' + row['relative_log'].split('/')[0])
for first in ('de_la_pw_dmr_hier_20261125', 'scanobject_weights_cb_20261126',
              'de_la_v1_author_checkpoint_20261124_0106', 'point2vec_author_checkpoint_20261124_0127'):
    roots.add('HyCoRe_runs/' + first)
roots.update(['HyCoRe/classification_ModelNet40', 'HyCoRe/classification_ScanObject',
              'HyCoRe/inter_hierarchy_MN40', 'HyCoRe/docs/research/diagnostics'])
priority = ('scanobject_weights_cb_20261126', 'de_la_pw_dmr_hier_20261125',
            'scanobject_author_s', 'mn40_b05_', 'scanobject_dgcnn_hier_20261124')
def order(name):
    if any(token in name for token in priority):
        return (0, name)
    if name.startswith('HyCoRe_runs/'):
        return (1, name)
    return (2, name)
request = dict(scope='classification report offline discussion: prioritized existing text, code, metrics; no model execution',
               roots=sorted(roots, key=order))
script = base64.b64encode((here / 'collect_records.py').read_bytes()).decode('ascii')
command = "nice -n 10 /usr/bin/python3 -c 'import base64;exec(base64.b64decode(\"" + script + "\"))'"
args.out.mkdir(parents=True, exist_ok=True)
target = args.out / (args.label + '_' + args.port + '.tar.gz')
with target.open('xb') as stream:
    run = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=12', '-p', args.port,
                          args.host, command], input=json.dumps(request).encode('utf-8'),
                         stdout=stream, stderr=subprocess.PIPE, timeout=900)
if run.returncode:
    raise SystemExit('Read-only archive stream failed: ' + run.stderr.decode('utf-8', errors='replace'))
with tarfile.open(target) as archive:
    index = json.load(archive.extractfile('OFFLINE_ARCHIVE_INDEX.json'))
    for row in index['files']:
        content = archive.extractfile(row['path']).read()
        if len(content) != row['bytes'] or hashlib.sha256(content).hexdigest() != row['sha256']:
            raise RuntimeError('Archive verification failed for ' + row['path'])
private_index = args.out / (args.label + '_' + args.port + '_index.json')
private_index.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding='utf-8')
receipt = dict(verified_utc=datetime.now(timezone.utc).isoformat(), archive_name=target.name,
               archive_bytes=target.stat().st_size, archived_uncompressed_bytes=index['total_bytes'],
               file_count=len(index['files']), verified_all_file_hashes=True,
               sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
               skipped_count=len(index['skipped']), missing_root_count=len(index['missing_roots']),
               private_storage='local .codex-local offline_classification_20261126_1736',
               exclusions=index['exclusions'])
(here / (args.label + '_receipt_' + args.port + '.json')).write_text(json.dumps(receipt, indent=2), encoding='utf-8')
print(json.dumps(receipt), flush=True)
