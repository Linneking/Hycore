"""Verify the original Cars archive and materialize the validated annotation mapping."""
import csv
import hashlib
import shutil
import tarfile
from pathlib import Path


archive = Path('car_ims.tgz')
assert archive.stat().st_size == 1956628579
digest = hashlib.sha256()
with archive.open('rb') as stream:
    while True:
        chunk = stream.read(8 * 1024 * 1024)
        if not chunk:
            break
        digest.update(chunk)
assert digest.hexdigest() == '4d96e3e3e892c6b7e4a71959325c7de4804037d27ba5aaf1c756100fd94a5342'
raw = Path('raw').resolve()
raw.mkdir()
with tarfile.open(archive) as images:
    images.extractall(raw, filter='data')
cars = Path('data/cars196')
cars.mkdir(parents=True)
shutil.copyfile('cars_annos.mat', cars / 'cars_annos.mat')
with open('mapping.csv', encoding='utf-8-sig', newline='') as source:
    mapping = list(csv.DictReader(source))
assert len(mapping) == 16185
for row in mapping:
    relative = Path(row['target_path'])
    assert not relative.is_absolute() and '..' not in relative.parts
    dest = cars / relative
    origin = raw / 'car_ims' / Path(row['source_path']).name
    assert origin.is_file()
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.symlink_to(origin)
