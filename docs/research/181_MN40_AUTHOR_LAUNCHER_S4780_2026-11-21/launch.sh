#!/usr/bin/env bash
set -euo pipefail

author_commit=ef56a73020e309b1009c486fea1f601d3f7b3252
export TZ=Asia/Shanghai PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 MPLBACKEND=Agg

mode=$1
run_dir=$2
shift 2

if [[ "$mode" == prepare ]]; then
    author_repo=$1
    dataset=$2
    extension=$3
    reference_manifest=$4
    baseline_preflight=$5
    helpers=$(cd "$(dirname "$0")" && pwd)
    mkdir -- "$run_dir"
    mkdir -- "$run_dir/source" "$run_dir/runtime"
    git -C "$author_repo" archive "$author_commit" classification_ModelNet40 pointnet2_ops_lib |
        tar -xf - -C "$run_dir/source"
    mkdir -p "$run_dir/source/classification_ModelNet40/data"
    ln -s "$dataset" "$run_dir/source/classification_ModelNet40/data/modelnet40_ply_hdf5_2048"
    ln -s "$extension" "$run_dir/source/pointnet2_ops_lib/pointnet2_ops/$(basename "$extension")"
    cat > "$run_dir/runtime/sitecustomize.py" <<'PY'
import os


gpu_uuid = os.environ.get("MN40_AUTHOR_GPU_UUID")
if gpu_uuid:
    environ_type = type(os.environ)
    set_item = environ_type.__setitem__

    def set_visible_gpu(environ, key, value):
        if key == "CUDA_VISIBLE_DEVICES":
            value = gpu_uuid
        set_item(environ, key, value)

    environ_type.__setitem__ = set_visible_gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_uuid
PY
    cat >> "$run_dir/runtime/sitecustomize.py" <<'PY'

order_file = os.environ.get("MN40_AUTHOR_ORDER_FILE")
if order_file:
    from shard_order import install
    install(order_file)
PY
    cp "$helpers/shard_order.py" "$run_dir/runtime/"
    printf '%s\n' "$author_commit" > "$run_dir/author_commit.txt"
    git -C "$(dirname "$0")" rev-parse HEAD > "$run_dir/launcher_commit.txt"
    export PYTHONPATH="$run_dir/source/pointnet2_ops_lib:$PYTHONPATH"
    CUDA_VISIBLE_DEVICES="" python - "$run_dir" "$reference_manifest" "$baseline_preflight" <<'PY' > "$run_dir/preflight.json"
import glob
import hashlib
import json
from pathlib import Path
import sys

import geoopt
import h5py
import torch
from pointnet2_ops import _ext


def sha256(path):
    result = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


run_dir, reference_path, baseline_path = map(Path, sys.argv[1:])
source = run_dir / "source/classification_ModelNet40"
data = source / "data/modelnet40_ply_hdf5_2048"
reference = json.loads(reference_path.read_text())["dataset_identity"]
baseline = json.loads(baseline_path.read_text())
runtime = {"python": sys.version.split()[0], "torch": torch.__version__,
           "cuda": torch.version.cuda, "geoopt": geoopt.__version__,
           "pointnet2_extension_sha256": sha256(_ext.__file__)}
source_hashes = {name: sha256(source / name) for name in
                 ("launcher_pointmlp.sh", "main_pointmlp_hycore.py", "data.py")}
if source_hashes != baseline["source_sha256"]:
    raise RuntimeError("Author source differs from the completed author-entry control")
for key in ("torch", "cuda", "geoopt", "pointnet2_extension_sha256"):
    if runtime[key] != baseline["runtime"][key]:
        raise RuntimeError(f"Author control runtime differs in {key}")
identity = {}
configured = {}
expected_train = [f"ply_data_train{i}.h5" for i in (1, 4, 3, 2, 0)]
baseline_train = [f"ply_data_train{i}.h5" for i in (3, 0, 2, 4, 1)]
if baseline["data"]["train"]["author_glob_order"] != baseline_train:
    raise RuntimeError("Unexpected train order in completed author-entry control")
if reference["train"]["file_order"] != expected_train:
    raise RuntimeError("Unexpected train order in TS02 manifest")
for partition in ("train", "test"):
    files = [Path(name) for name in glob.glob(str(data / f"ply_data_{partition}*.h5"))]
    hashes = {path.name: sha256(path) for path in files}
    rows = 0
    for path in files:
        with h5py.File(path, "r") as handle:
            rows += len(handle["label"])
    if hashes != reference[partition]["file_sha256"] or rows != reference[partition]["size"]:
        raise RuntimeError(f"{partition} HDF5 files differ from TS02")
    prior = baseline["data"][partition]
    if hashes != prior["sha256"] or rows != prior["rows"]:
        raise RuntimeError(f"{partition} dataset differs from the author control")
    effective = expected_train if partition == "train" else prior["author_glob_order"]
    if len(effective) != len(files) or set(effective) != set(hashes):
        raise RuntimeError(f"{partition} permutation differs from detected files")
    configured[str(data / f"ply_data_{partition}*.h5")] = effective
    identity[partition] = {
        "rows": rows,
        "natural_order": [path.name for path in files],
        "baseline_author_order": prior["author_glob_order"],
        "effective_order": effective,
        "matches_ts02_order": effective == reference[partition]["file_order"],
        "sha256": hashes,
    }
(run_dir / "runtime" / "order.json").write_text(json.dumps(configured, indent=2) + "\n")
print(json.dumps({
    "data": identity,
    "runtime": runtime,
    "author_entry_baseline_runtime": baseline["runtime"],
    "source_sha256": source_hashes,
    "config": {"seed": 4780, "batch": 32, "epochs": 300, "steps_per_epoch": 307,
               "workers": 8, "hier": False, "teacher": None, "initial_checkpoint": None},
}, indent=2))
PY
    cd "$run_dir/source/classification_ModelNet40"
    MN40_AUTHOR_GPU_UUID=-1 PYTHONPATH="$run_dir/runtime:$PYTHONPATH" \
        python main_pointmlp_hycore.py --help > "$run_dir/author_help.txt"
    MN40_AUTHOR_ORDER_FILE="$run_dir/runtime/order.json" PYTHONPATH="$run_dir/runtime:$PYTHONPATH" \
        python "$run_dir/runtime/shard_order.py" "$run_dir/runtime/order.json"
    exit 0
fi

if [[ "$mode" != run || "$#" != 1 ]]; then
    exit 2
fi
gpu_index=$1
test -s "$run_dir/preflight.json" && test -s "$run_dir/author_help.txt"
export MN40_AUTHOR_ORDER_FILE="$run_dir/runtime/order.json"
gpu_uuid=$(nvidia-smi -i "$gpu_index" --query-gpu=uuid --format=csv,noheader)
export MN40_AUTHOR_GPU_UUID="$gpu_uuid" CUDA_VISIBLE_DEVICES="$gpu_uuid"
export PYTHONPATH="$run_dir/runtime:$run_dir/source/pointnet2_ops_lib:$PYTHONPATH"
python -c 'import os; os.environ["CUDA_VISIBLE_DEVICES"] = "0"; assert os.environ["CUDA_VISIBLE_DEVICES"] == os.environ["MN40_AUTHOR_GPU_UUID"]'
python "$run_dir/runtime/shard_order.py" "$MN40_AUTHOR_ORDER_FILE"
exec 8>"/tmp/hycore_$(id -u)_gpu_$gpu_index.lock"
flock -n 8
exec 9>"/tmp/hycore-$USER-$gpu_uuid.lock"
flock -n 9
cd "$run_dir/source/classification_ModelNet40"
test ! -e checkpoints/Hype_PointNet-Offv_pointmlp_hycore_var-4780
for check in 1 2; do
    apps=$(nvidia-smi -i "$gpu_uuid" --query-compute-apps=pid --format=csv,noheader)
    read -r memory utilization < <(nvidia-smi -i "$gpu_uuid" \
        --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits | tr -d ',')
    if [[ -n "$apps" || "$utilization" != 0 || "$memory" -gt 128 ]]; then
        printf 'GPU %s is occupied\n' "$gpu_index" >&2
        exit 1
    fi
    if [[ "$check" == 1 ]]; then sleep 2; fi
done
printf 'gpu_index=%s\ngpu_uuid=%s\nstarted_at=%s\ncommand=bash -e launcher_pointmlp.sh\n' \
    "$gpu_index" "$gpu_uuid" "$(date --iso-8601=seconds)" > "$run_dir/start.txt"
trap 'status=$?; printf "exit_code=%s\nended_at=%s\n" "$status" "$(date --iso-8601=seconds)" > "$run_dir/exit.txt"' EXIT
bash -e launcher_pointmlp.sh
