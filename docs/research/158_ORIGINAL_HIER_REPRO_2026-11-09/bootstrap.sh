#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
root="$PWD"
port="$1"
if [[ "$port" == 8000 ]]; then
    python="$HOME/miniforge3/envs/hycore_seg_v1/bin/python"
    datasets=(CUB)
else
    python="$HOME/anaconda3/envs/hycore/bin/python"
    datasets=(Cars SOP)
fi
for dataset in "${datasets[@]}"; do
    for mode in on off; do
        mkdir "${dataset}_${mode}"
        printf '{"status":"preparing_data_and_environment","dataset":"%s","hier":"%s"}\n' "$dataset" "$mode" > "${dataset}_${mode}/state.json"
    done
done
trap 'printf "Preparation failed at line %s\n" "$LINENO" >&2' ERR
mkdir source deps
tar -xf source.tar -C source
sha256sum source.tar | awk '{print $1}' > source.sha256
"$python" configure_source.py source
export PYTHONPATH="$root/deps"
export TORCH_HOME="$root/pretrained"
export PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1
mkdir -p "$TORCH_HOME/hub/checkpoints"
fetch() {
    if command -v curl >/dev/null; then
        curl --fail --location --retry 4 --connect-timeout 15 "$1" -o "$2"
    else
        wget -nv -t 4 -T 30 "$1" -O "$2"
    fi
}
"$python" prepare_data.py "$port"
if [[ "$port" == 8000 ]]; then
    (
        fetch https://s3.amazonaws.com/fast-ai-imageclas/CUB_200_2011.tgz cub.tgz
        "$python" - <<'PY'
import hashlib
import tarfile
from pathlib import Path
p = Path('cub.tgz')
assert p.stat().st_size == 1150585339
with p.open('rb') as f:
    assert hashlib.md5(f.read(1048576)).hexdigest() == 'd2acaa99439dff0483c7bbac1bfe2a92'
with tarfile.open(p) as archive:
    archive.extractall('data', filter='data')
PY
    ) > data_preparation.log 2>&1 &
    data_pid=$!
    "$python" -m pip install --disable-pip-version-check --no-deps --target "$root/deps" \
        --index-url https://download.pytorch.org/whl/cu128 'torchvision==0.23.0+cu128'
fi
"$python" -m pip install --disable-pip-version-check --no-deps --target "$root/deps" \
    'timm==0.4.12' 'pytorch-metric-learning==2.9.0' 'pandas==2.2.3' \
    python-dateutil pytz tzdata six 'Pillow==11.3.0' 'scikit-learn==1.6.1' joblib threadpoolctl
fetch https://download.pytorch.org/models/resnet50-0676ba61.pth \
    "$TORCH_HOME/hub/checkpoints/resnet50-0676ba61.pth"
sha256sum "$TORCH_HOME/hub/checkpoints/resnet50-0676ba61.pth" | grep '^0676ba61' > pretrained.sha256
if [[ "$port" == 8000 ]]; then wait "$data_pid"; fi
PYTHONPATH="$root/source/hier:$PYTHONPATH" "$python" - "${datasets[@]}" <<'PY'
import sys
from pathlib import Path
import torch
import torchvision
import train
from dataset import CUBirds, Cars, SOP
registry = {'CUB': (CUBirds, 5864, 5924), 'Cars': (Cars, 8054, 8131), 'SOP': (SOP, 59551, 60502)}
for name in sys.argv[1:]:
    cls, *counts = registry[name]
    for split, count in zip(('train', 'eval'), counts):
        ds = cls(str(Path('data').resolve()), split)
        assert len(ds) == count and all(Path(p).is_file() for p in ds.im_paths), (name, split, len(ds), count)
        print('DATA_OK', name, split, len(ds), flush=True)
print('ENV_OK', torch.__version__, torchvision.__version__, flush=True)
PY
exec "$python" dispatch_matrix.py "${datasets[@]}"
