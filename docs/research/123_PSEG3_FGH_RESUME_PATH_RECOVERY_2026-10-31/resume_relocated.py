"""Resume existing v6 checkpoints after a verified data-path/GPU relocation.

The training source stays frozen. Every recovery uses a fresh output root, and
the original inputs and failed attempt remain available for inspection.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from datetime import datetime, timezone


def stamp():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def same(left, right):
    import numpy as np
    import torch
    if torch.is_tensor(left):
        return torch.is_tensor(right) and torch.equal(left, right)
    if isinstance(left, np.ndarray):
        return isinstance(right, np.ndarray) and np.array_equal(left, right)
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(
            same(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)):
        return isinstance(right, type(left)) and len(left) == len(right) and all(
            same(a, b) for a, b in zip(left, right))
    return left == right


def idle(gpu):
    rows = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,uuid,utilization.gpu,memory.used",
        "--format=csv,noheader,nounits"], text=True).splitlines()
    row = next(row for row in rows if int(row.split(",")[0]) == gpu)
    _, uuid, util, memory = [item.strip() for item in row.split(",")]
    processes = subprocess.check_output([
        "nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"],
        text=True).splitlines()
    pids = [line for line in processes if line.split(",")[0].strip() == uuid]
    assert not pids and int(util) == 0 and int(memory) < 500, (gpu, row, pids)
    return {"gpu": gpu, "uuid": uuid, "utilization": int(util), "memory_mib": int(memory)}


def prepare(args):
    import torch
    sys.path.insert(0, args.source)
    from segmentation_ShapeNetPart.hycore_seg_v6.train import arguments
    from segmentation_ShapeNetPart.hycore_seg_v1.utils import state_hash
    root, previous, source, data = map(Path, (args.root, args.previous_root, args.source, args.data))
    assert not root.exists(), "Refuse reusing an output directory"
    commit = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    assert not subprocess.check_output([
        "git", "-C", str(source), "status", "--porcelain", "--", "segmentation_ShapeNetPart"], text=True).strip()
    root.mkdir()
    jobs = []
    for arm, gpu in zip("fgh", args.gpus):
        prior, out = previous / arm, root / arm
        inputs = out / "resume_inputs"
        inputs.mkdir(parents=True)
        checkpoint_path = prior / "resume_inputs/last.source.pth"
        original = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        assert original["source"]["commit"] == commit
        assert original["data_manifest_sha256"] == sha(data / "manifest.json")
        assert original["updates"] == original["epoch"] * 437 < 87400
        assert original["config"]["arm"] == arm
        adapted = dict(original)
        adapted["config"] = dict(original["config"], data=str(data), physical_gpu=gpu)
        torch.save(adapted, inputs / "last.pth")
        restored = torch.load(inputs / "last.pth", map_location="cpu", weights_only=False)
        for key in original:
            if key != "config":
                assert same(original[key], restored[key]), (arm, key)
        changed = {key: [original["config"].get(key), restored["config"].get(key)]
                   for key in original["config"] if original["config"].get(key) != restored["config"].get(key)}
        assert set(changed) <= {"data", "physical_gpu"}
        for name in ("best.pth", "best_selection.json"):
            shutil.copy2(prior / "resume_inputs" / name, inputs / name)
        command = [args.python, "-m", "segmentation_ShapeNetPart.hycore_seg_v6.train",
                   "--config", str(source / f"segmentation_ShapeNetPart/hycore_seg_v6/configs/fgh_{arm}.json"),
                   "--data", str(data), "--out", str(out / "main"), "--arm", arm,
                   "--physical-gpu", str(gpu), "--protocol", "antao_testsel", "--epochs", "200",
                   "--resume", str(inputs / "last.pth"), "--expected-initial-sha256", original["initial_model_sha256"]]
        actual = vars(arguments(command[3:]))
        expected = dict(restored["config"])
        for field in ("out", "resume", "config", "expected_initial_sha256"):
            actual.pop(field, None)
            expected.pop(field, None)
        assert actual == expected, {key: [expected.get(key), actual.get(key)]
                                    for key in expected.keys() | actual.keys() if expected.get(key) != actual.get(key)}
        record = {"arm": arm, "gpu": gpu, "resume_epoch": original["epoch"], "updates": original["updates"],
                  "command": command, "checkpoint_sha256": sha(inputs / "last.pth"),
                  "source_checkpoint_sha256": sha(checkpoint_path), "changed_config_fields": changed,
                  "model_sha256": state_hash(restored["model"]), "source_history": str(prior / "source_history"),
                  "training_state_exact": True, "resume_configuration_exact": True}
        write(out / "prepare.json", record)
        jobs.append(record)
    assert not torch.cuda.is_initialized()
    acceptance = {"complete": True, "at": stamp(), "source": str(source), "commit": commit,
                  "data_manifest_sha256": sha(data / "manifest.json"), "jobs": jobs}
    write(root / "prepare.json", acceptance)
    print(json.dumps(acceptance), flush=True)


def dispatch(args):
    root = Path(args.root)
    accepted = read(root / "prepare.json")
    assert accepted["complete"]
    for job in accepted["jobs"]:
        assert not (root / job["arm"] / "submitted.json").exists()
        idle(job["gpu"])
    for job in accepted["jobs"]:
        out = root / job["arm"]
        command = [args.python, str(Path(__file__).resolve()), "worker", "--root", str(root), "--arm", job["arm"]]
        with (out / "launcher.log").open("x") as log:
            proc = subprocess.Popen(command, cwd=accepted["source"],
                                    env=dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1"),
                                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                    start_new_session=True)
        write(out / "submitted.json", {"pid": proc.pid, "command": command, "at": stamp()})
        print(job["arm"], proc.pid, flush=True)


def worker(args):
    root = Path(args.root)
    accepted = read(root / "prepare.json")
    out = root / args.arm
    job = read(out / "prepare.json")
    assert not (out / "main").exists()
    idle(job["gpu"])
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(job["gpu"]), PYTHONPATH=accepted["source"],
               PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4")
    write(out / "queue.json", {"state": "resume_starting", "at": stamp()})
    with (out / "production.log").open("x") as log:
        result = subprocess.run(job["command"], cwd=accepted["source"], env=env,
                                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
    write(out / "queue.json", {"state": "complete" if result.returncode == 0 else "failed",
                               "returncode": result.returncode, "at": stamp()})
    if result.returncode:
        raise SystemExit(result.returncode)


def verify(args):
    import torch
    root = Path(args.root)
    accepted = read(root / "prepare.json")
    records = []
    for job in accepted["jobs"]:
        out = root / job["arm"]
        saved = torch.load(out / "resume_inputs/last.pth", map_location="cpu", weights_only=False)
        restored = torch.load(out / "main/e0.pth", map_location="cpu", weights_only=False)
        keys = ["model", "optimizer", "scheduler", "rng", "train_generator", "eval_generator",
                "part_generator", "negative_generator", "epoch", "updates", "best_epoch", "best_score", "best_ce"]
        for key in keys:
            assert same(saved[key], restored[key]), (job["arm"], key)
        status = read(out / "main/status.json")
        assert not (out / "main/failure.json").exists()
        rows = []
        if (out / "main/metrics.jsonl").exists():
            for line in (out / "main/metrics.jsonl").read_text().splitlines():
                try: rows.append(json.loads(line))
                except json.JSONDecodeError: pass
        if rows:
            assert rows[0]["epoch"] == job["resume_epoch"] + 1
            assert rows[0]["steps"] == 437 and rows[0]["updates"] == (job["resume_epoch"] + 1) * 437
            assert rows[0]["model_finite"] and rows[0]["optimizer_finite"]
        records.append({"arm": job["arm"], "status": status, "resume_state_exact": True,
                        "complete_resumed_epochs": len(rows),
                        "first_epoch": ({"epoch": rows[0]["epoch"], "steps": rows[0]["steps"],
                                         "instance_miou": rows[0]["test"]["instance_miou"],
                                         "class_miou": rows[0]["test"]["class_miou"]} if rows else None)})
    assert not torch.cuda.is_initialized()
    result = {"at": stamp(), "jobs": records, "verification_cpu_only": True}
    write(root / "recovery_verification.json", result)
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("prepare", "dispatch", "worker", "verify"))
    parser.add_argument("--root", required=True)
    parser.add_argument("--previous-root")
    parser.add_argument("--source")
    parser.add_argument("--data")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--gpus", type=int, nargs=3, default=[0, 1, 2])
    parser.add_argument("--arm", choices=("f", "g", "h"))
    args = parser.parse_args()
    globals()[args.phase](args)


if __name__ == "__main__":
    main()
