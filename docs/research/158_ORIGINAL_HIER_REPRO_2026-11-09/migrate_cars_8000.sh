#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
base="$HOME/HyCoRe_runs/hier_r50_repro_20261110_a"
p="$HOME/miniforge3/envs/hycore_seg_v1/bin/python"
printf '{"status":"preparing_cars_data","dataset":"Cars","hier":"off"}\n' > state.json
trap 'echo "Preparation failed at line $LINENO" >&2; printf "{\"status\":\"preparation_failed\"}\n" > state.json' ERR
curl --fail --location --retry 4 --connect-timeout 20 --speed-limit 1024 --speed-time 90 \
    'https://hf-mirror.com/datasets/XiN0919/FGVC/resolve/921e8dce38536cd5982739d85d5e9235d2cb2a76/car_ims.tgz' \
    -o car_ims.tgz
"$p" prepare_cars_migration.py
export HIER_DATA_ROOT="$PWD/data"
PYTHONPATH="$base/source/hier:$base/deps" PYTHONDONTWRITEBYTECODE=1 "$p" - <<'PY'
import os
from pathlib import Path
from dataset import Cars
for split, expected in [('train', 8054), ('eval', 8131)]:
    ds = Cars(os.environ['HIER_DATA_ROOT'], split)
    assert len(ds) == expected and all(Path(path).is_file() for path in ds.im_paths)
    print('DATA_OK', split, len(ds), flush=True)
PY
exec "$p" retry_cars_off.py "$base" auto
