"""Run DeLA calibration, paired smoke, or training on reserved GPUs."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
DELA_COMMIT = "0f427c36c5a991a28397ffdaaf8c205b80d6b30d"
MODES = ("h0", "hier")
DATA_FILES = (
    "training_objectdataset_augmentedrot_scale75.h5",
    "test_objectdataset_augmentedrot_scale75.h5",
)


def fingerprint(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def recipe(args):
    source_commit = subprocess.check_output(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True, timeout=10,
    ).strip()
    code = ("dispatch.py", "run.py", "model.py", "loss.py")
    author_code = ("config.py", "delacls.py", "ScanObjectNN.py", "train.py")
    return dict(
        source_commit=source_commit,
        source_sha256={name: fingerprint(HERE / name) for name in code},
        dela_commit=DELA_COMMIT,
        dela_source_sha256={
            name: fingerprint(args.dela_root / "ScanObjectNN" / name) for name in author_code
        },
        data_sha256={name: fingerprint(args.data_root / "main_split" / name) for name in DATA_FILES},
        seed=args.seed,
        lambda_h=args.lambda_h,
        input_scale=args.input_scale,
    )


def idle_gpu(index, allow_display, query_gpus):
    gpu = next((row for row in query_gpus() if row["index"] == index), None)
    if gpu is None:
        raise RuntimeError(f"GPU {index} is not present")
    if gpu["compute_busy"] or gpu["utilization"]:
        return None
    if not allow_display:
        used = subprocess.check_output(
            ["nvidia-smi", "-i", gpu["uuid"], "--query-gpu=memory.used",
             "--format=csv,noheader,nounits"], text=True, timeout=20,
        )
        if int(used.strip()) != 0:
            return None
    return gpu


@contextmanager
def reserve(index, args, query_gpus):
    import fcntl

    deadline = time.monotonic() + args.wait_timeout
    lock_path = Path("/tmp") / f"hycore_{os.getuid()}_gpu_{index}.lock"
    with lock_path.open("a+") as lock:
        while True:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                pass
            else:
                try:
                    first = idle_gpu(index, args.allow_display, query_gpus)
                    if first is not None:
                        time.sleep(2)
                        second = idle_gpu(index, args.allow_display, query_gpus)
                        if second is not None and second["uuid"] == first["uuid"]:
                            yield second
                            return
                finally:
                    fcntl.flock(lock, fcntl.LOCK_UN)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RuntimeError(f"GPU {index} has no available idle reservation")
            time.sleep(min(15, remaining))


def run_worker(root, mode, phase, gpu, args, queue):
    folder = root / mode
    command = [
        sys.executable, "-u", str(HERE / "run.py"), "--dela-root", str(args.dela_root),
        "--data-root", str(args.data_root), "--output", str(folder / "work"), "--mode", mode,
        "--lambda-h", str(args.lambda_h), "--input-scale", str(args.input_scale),
        "--seed", str(args.seed),
    ]
    if phase != "train":
        command.append("--" + phase)
    env = dict(os.environ, **queue.device_environment([gpu]))
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    folder.mkdir(exist_ok=False)
    status = dict(mode=mode, phase=phase, gpu=gpu["index"], state="starting", started_at=queue.now())
    status_file = folder / "status.json"
    queue.write(status_file, status)
    try:
        with (folder / "worker.log").open("x", encoding="utf-8") as stream:
            child = subprocess.Popen(command, env=env, stdin=subprocess.DEVNULL,
                                     stdout=stream, stderr=subprocess.STDOUT)
            status.update(state="running", pid=child.pid)
            queue.write(status_file, status)
            status["exit_code"] = child.wait()
        if status["exit_code"] != 0:
            raise RuntimeError(f"{mode} {phase} exited with code {status['exit_code']}")
    except BaseException:
        status["state"] = "failed"
        raise
    else:
        status["state"] = "completed"
    finally:
        status["finished_at"] = queue.now()
        queue.write(status_file, status)


def compare_smoke(h0, hier):
    h0_rows, hier_rows = h0["rows"], hier["rows"]
    enough_rows = len(h0_rows) == len(hier_rows) == 2
    checks = dict(
        two_batches=enough_rows,
        same_recipe=all(h0[key] == hier[key] for key in ("seed", "input_scale", "lambda_h")),
        same_initial_model=h0["initial_model_sha"] == hier["initial_model_sha"],
        both_optimizers_updated_twice=h0["updates"] == hier["updates"] == 2,
    )
    if not enough_rows:
        return checks

    first_h0, first_hier = h0_rows[0], hier_rows[0]
    proxy_states = [hier["initial_proxy_sha"], *(row["proxy_sha"] for row in hier_rows)]
    checks.update(
        same_input_rows=all(
            left[key] == right[key]
            for left, right in zip(h0_rows, hier_rows)
            for key in ("input_sha", "ids_sha", "m", "step")
        ),
        both_part_sizes_tested=[row["m"] for row in h0_rows] == [384, 511],
        same_first_embeddings_and_logits=all(
            first_h0[key] == first_hier[key] for key in ("whole_sha", "part_sha", "logits_sha")
        ),
        same_first_base_losses=all(
            math.isclose(first_h0[key], first_hier[key], rel_tol=1e-6, abs_tol=1e-7)
            for key in ("ce", "coordinate", "l_h", "l_c", "base")
        ),
        same_first_bn_state=first_h0["bn_sha"] == first_hier["bn_sha"],
        same_main_cuda_rng=all(
            left["main_rng"] == right["main_rng"] and left["rng_unchanged"] and right["rng_unchanged"]
            for left, right in zip(h0_rows, hier_rows)
        ),
        finite_loss_and_geometry=all(
            math.isfinite(row[key])
            for row in h0_rows + hier_rows
            for key in ("ce", "coordinate", "l_h", "l_c", "base", "whole_depth_mean", "part_depth_mean")
        ),
        no_skipped_step=all(row["did_step"] for row in h0_rows + hier_rows),
        active_hier_relationships=all(row["hier_samples"] > 0 and row["hier_proxies"] > 0 for row in hier_rows),
        proxy_updates=h0["initial_proxy_sha"] is None and all(proxy_states) and len(set(proxy_states)) == 3,
        first_model_update_changed=first_h0["model_sha_after"] != first_hier["model_sha_after"],
    )
    return checks


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=("calibrate", "smoke", "train"))
    parser.add_argument("--dela-root", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--gpus", required=True, type=int, nargs="+")
    parser.add_argument("--seed", type=int, default=4780)
    parser.add_argument("--lambda-h", required=True, type=float)
    parser.add_argument("--input-scale", type=float, default=1.0)
    parser.add_argument("--smoke-root", type=Path, help="Accepted matched smoke for a full launch")
    parser.add_argument("--wait-timeout", type=float, default=0, help="Seconds to wait for a GPU")
    parser.add_argument("--allow-display", action="store_true", help="Allow idle desktop graphics memory")
    args = parser.parse_args()
    expected = 2 if args.phase == "train" else 1
    if len(args.gpus) != expected or len(set(args.gpus)) != expected or min(args.gpus) < 0:
        parser.error(f"{args.phase} requires {expected} distinct physical GPU indices")
    if args.phase == "train" and args.smoke_root is None:
        parser.error("--smoke-root is required for full training")
    if not math.isfinite(args.lambda_h) or args.lambda_h <= 0:
        parser.error("--lambda-h must be finite and positive")
    if not math.isfinite(args.input_scale) or args.input_scale <= 0:
        parser.error("--input-scale must be finite and positive")
    if not math.isfinite(args.wait_timeout) or args.wait_timeout < 0:
        parser.error("--wait-timeout must be finite and nonnegative")
    args.dela_root = args.dela_root.expanduser().resolve(strict=True)
    args.data_root = args.data_root.expanduser().resolve(strict=True)
    args.output = args.output.expanduser().resolve()
    return args


def main():
    args = parse_arguments()
    sys.path.insert(0, str(REPO))
    from inter_hierarchy_MN40.hier_proxy_scratch_v8 import queue

    identity = recipe(args)
    if args.phase == "train":
        acceptance_path = args.smoke_root.expanduser() / "acceptance.json"
        acceptance = json.loads(acceptance_path.read_text(encoding="utf-8"))
        if not acceptance["passed"] or acceptance["recipe"] != identity:
            raise RuntimeError("Smoke acceptance does not match the training recipe")
    args.output.mkdir(parents=True, exist_ok=False)
    queue.write(args.output / "recipe.json", identity)

    if args.phase == "calibrate":
        with reserve(args.gpus[0], args, queue.query_gpus) as gpu:
            run_worker(args.output, "h0", args.phase, gpu, args, queue)
        return
    if args.phase == "smoke":
        for mode in MODES:
            with reserve(args.gpus[0], args, queue.query_gpus) as gpu:
                run_worker(args.output, mode, args.phase, gpu, args, queue)
        traces = [json.loads((args.output / mode / "work" / "smoke.json").read_text(encoding="utf-8"))
                  for mode in MODES]
        checks = compare_smoke(*traces)
        accepted = all(checks.values())
        queue.write(args.output / "acceptance.json", dict(passed=accepted, recipe=identity, checks=checks))
        if not accepted:
            raise RuntimeError(f"DeLA paired smoke failed: {checks}")
        return

    with ExitStack() as reservations:
        gpus = [reservations.enter_context(reserve(index, args, queue.query_gpus)) for index in args.gpus]
        for gpu in gpus:
            current = idle_gpu(gpu["index"], args.allow_display, queue.query_gpus)
            if current is None or current["uuid"] != gpu["uuid"]:
                raise RuntimeError(f"GPU {gpu['index']} is no longer idle")
        with ThreadPoolExecutor(max_workers=2) as executor:
            workers = [executor.submit(run_worker, args.output, mode, args.phase, gpu, args, queue)
                       for mode, gpu in zip(MODES, gpus)]
            for worker in workers:
                worker.result()


if __name__ == "__main__":
    main()
