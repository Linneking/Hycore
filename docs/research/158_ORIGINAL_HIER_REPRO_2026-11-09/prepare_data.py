"""Create the published dataset layout using previously verified local images."""
import csv
import shutil
import sys
from pathlib import Path


root = Path(__file__).resolve().parent
data = root / 'data'
data.mkdir(exist_ok=True)
if sys.argv[1] == '8001':
    home = Path.home()
    assets = home / 'HyCoRe_datasets' / 'original_hier_images_20261008' / 'extracted'
    (data / 'Stanford_Online_Products').symlink_to(assets / 'sop' / 'Stanford_Online_Products')
    cars = data / 'cars196'
    cars.mkdir()
    mapping = home / 'HyCoRe_runs' / 'original_hier_cars_identity_20261008' / 'mapping.csv'
    shutil.copyfile(home / 'HyCoRe_runs' / 'original_hier_image_inputs_20261008' /
                    'handoff' / 'data' / 'cars196' / 'cars_annos.mat', cars / 'cars_annos.mat')
    with mapping.open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 16185
    for row in rows:
        relative = Path(row['target_path'])
        source = Path(row['source_path'])
        assert not relative.is_absolute() and '..' not in relative.parts and source.is_file()
        dest = cars / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.symlink_to(source)
