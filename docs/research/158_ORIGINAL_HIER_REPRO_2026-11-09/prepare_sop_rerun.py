"""Prepare a new SOP-only run from the already verified HIER deployment."""

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path


base, run = (Path(arg).resolve() for arg in sys.argv[1:])
source_hash = hashlib.sha256((base / 'source.tar').read_bytes()).hexdigest()
assert source_hash == (base / 'source.sha256').read_text().strip()
pretrained = base / 'pretrained/hub/checkpoints/resnet50-0676ba61.pth'
pretrained_hash = hashlib.sha256(pretrained.read_bytes()).hexdigest()
assert pretrained_hash == (base / 'pretrained.sha256').read_text().split()[0]
assert pretrained_hash.startswith('0676ba61')

run.mkdir()
for tag in ('on', 'off'):
    (run / f'SOP_{tag}').mkdir()
(run / 'data').symlink_to(base / 'data', target_is_directory=True)
shutil.copytree(base / 'source', run / 'source', ignore=shutil.ignore_patterns('__pycache__'))
shutil.copyfile(Path(__file__).with_name('dispatch_matrix.py'), run / 'dispatch_matrix.py')
(run / 'source.sha256').write_text(source_hash + '\n')

helpers = run / 'source/hier/helpers.py'
original = helpers.read_text()
eval_loader = 'batch_size=500,\n        shuffle=False,\n        num_workers=multiprocessing.cpu_count() // world_size,'
assert original.count(eval_loader) == 1
patched = original.replace(eval_loader, 'batch_size=128,\n        shuffle=False,\n        num_workers=4,')
compile(patched, str(helpers), 'exec')
helpers.write_text(patched)

launcher_commit = subprocess.check_output(
    ['git', '-C', str(Path(__file__).parent), 'rev-parse', 'HEAD'], text=True).strip()
manifest = dict(source_archive_sha256=source_hash, imagenet_sha256=pretrained_hash,
                launcher_commit=launcher_commit, eval_loader_batch_per_rank=128,
                eval_loader_workers_per_rank=4,
                eval_helpers_sha256=hashlib.sha256(helpers.read_bytes()).hexdigest())
(run / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')

sys.path.insert(0, str(run / 'source/hier'))
from dataset import SOP

for split, expected in [('train', 59551), ('eval', 60502)]:
    dataset = SOP(str(run / 'data'), split)
    assert len(dataset) == expected and all(Path(path).is_file() for path in dataset.im_paths)
    print(f'DATA_OK SOP {split} {len(dataset)}', flush=True)
