"""Run the approved frozen audit through export, metrics, and publication.

Launch once with a private settings JSON. The two export workers use disjoint
idle GPUs; metrics starts only after both workers exit and all four exports
pass the shared-input gates. This module has no training or resume path.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from .collect import idle_gpu, now, sha, write


MODULE = "docs.research.diagnostics.parthier_embedding_audit94_v1"
LABEL_GROUPS = (("TS18_best", "TS18_e300"), ("TS02_best", "TS02_e300"))
EPOCHS = {"TS18_best": 201, "TS18_e300": 300, "TS02_best": 189, "TS02_e300": 300}


def checked_export_identity(cache):
    """All labels must describe identical inputs and an unchanged model."""
    records = json.loads((cache / "checkpoints.json").read_text(encoding="utf-8"))
    reference, crop = None, None
    gates = {}
    for label, epoch in EPOCHS.items():
        target = cache / (label + ".npz")
        manifest = json.loads((cache / (label + "_manifest.json")).read_text(encoding="utf-8"))
        if manifest.get("status") != "completed" or manifest.get("objects") != 9840:
            raise ValueError("Incomplete frozen export: " + label)
        if manifest.get("epoch") != epoch or manifest.get("source_sha256") != records[label]["source_sha256"]:
            raise ValueError("Export checkpoint identity mismatch: " + label)
        if not manifest.get("model_state_unchanged") or manifest.get("optimizer_updates") != 0 or manifest.get("backward_calls") != 0 or manifest.get("test_forwarded") is not False:
            raise ValueError("Frozen-state gate did not pass: " + label)
        numerical = manifest.get("gate", [])
        if len(numerical) != 5 or not all(item.get("passed") is True for item in numerical):
            raise ValueError("Native-model numerical gate missing or failed: " + label)
        if not target.is_file() or sha(target) != manifest["cache_sha256"]:
            raise ValueError("Frozen cache bytes changed: " + label)
        if reference is None:
            reference, crop = manifest["input"], manifest["crop_sha256"]
        elif manifest["input"] != reference or manifest["crop_sha256"] != crop:
            raise ValueError("Input/crop identity differs across models: " + label)
        gates[label] = {"epoch": epoch, "state_unchanged": True,
                        "source_sha256": manifest["source_sha256"],
                        "cache_sha256": manifest["cache_sha256"],
                        "native_gate_passed": True}
    return {**reference, "crop_sha256": crop, "shared_across": list(EPOCHS)}, gates


def command_env():
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    return env


class Pipeline:
    def __init__(self, settings_path):
        self.settings_path = Path(settings_path).resolve()
        self.settings = json.loads(self.settings_path.read_text(encoding="utf-8-sig"))
        self.root = Path(self.settings["root"]).resolve()
        self.source = Path(self.settings["source"]).resolve()
        self.gpus = self.settings["gpus"]
        if not self.root.is_dir() or not self.source.is_dir():
            raise ValueError("Prepared root and frozen source directories are required")
        if len(self.gpus) != 2 or len(set(self.gpus)) != 2 or any(not isinstance(gpu, int) or gpu < 0 for gpu in self.gpus):
            raise ValueError("Exactly two distinct physical GPUs are required")
        observations = self.settings["observations"]
        if len(observations) != 4 or {item["label"] for item in observations} != set(EPOCHS):
            raise ValueError("Exactly the four approved checkpoints are required")
        if any(row["epoch"] != EPOCHS[row["label"]] for row in observations):
            raise ValueError("Checkpoint epoch differs from the approved audit")
        self.status_path = self.root / "pipeline_status.json"
        if self.status_path.exists():
            raise FileExistsError("Pipeline was already dispatched; no duplicate launch")
        self.state = {"status": "running", "phase": "starting", "pid": os.getpid(),
                      "started_utc": now(), "updated_utc": now(), "children": [],
                      "source": str(self.source), "settings": str(self.settings_path)}
        write(self.status_path, self.state)

    def update(self, phase=None, **values):
        if phase is not None:
            self.state["phase"] = phase
        self.state.update(values)
        self.state["updated_utc"] = now()
        write(self.status_path, self.state)

    def spawn(self, name, arguments, *, env=None, **metadata):
        log_path = self.root / (name + ".log")
        # New roots use new log names; this never appends to old run results.
        with log_path.open("xb") as log:
            child = subprocess.Popen([sys.executable, "-u", "-m", *arguments],
                                     cwd=self.source, env=env or command_env(),
                                     stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        record = {"name": name, "pid": child.pid, "status": "running",
                  "started_utc": now(), "log": str(log_path), **metadata}
        self.state["children"].append(record)
        self.update()
        return child, record

    def wait_one(self, child, record):
        result = child.wait()
        record.update(status="completed" if result == 0 else "failed", returncode=result,
                      ended_utc=now())
        self.update()
        if result != 0:
            raise RuntimeError(record["name"] + " failed; see its private log")

    def run(self):
        self.update("prepare")
        child, record = self.spawn("prepare", [MODULE + ".collect", "prepare", "--settings", str(self.settings_path)])
        self.wait_one(child, record)

        # Collector workers also lock/check immediately before allocation.
        for gpu in self.gpus:
            idle_gpu(gpu)
        self.update("collect")
        workers = []
        try:
            for gpu, labels in zip(self.gpus, LABEL_GROUPS):
                child, record = self.spawn("collect_gpu%d" % gpu,
                    [MODULE + ".collect", "collect", "--settings", str(self.settings_path),
                     "--gpu", str(gpu), "--labels", *labels],
                    gpu=gpu, labels=list(labels))
                workers.append((child, record))
        except Exception:
            # A successfully started worker can finish its immutable export
            # even when the other worker could not be started.
            for child, record in workers:
                code = child.wait()
                record.update(status="completed" if code == 0 else "failed",
                              returncode=code, ended_utc=now())
            self.update()
            raise
        remaining = list(workers)
        while remaining:
            for child, record in list(remaining):
                result = child.poll()
                if result is not None:
                    record.update(status="completed" if result == 0 else "failed",
                                  returncode=result, ended_utc=now())
                    remaining.remove((child, record))
                    self.update()
            if remaining:
                time.sleep(1)
        if any(record["returncode"] != 0 for _, record in workers):
            raise RuntimeError("Frozen collection failed; metrics was not launched")

        self.update("export_gates")
        cache = self.root / "cache"
        identity, export_gates = checked_export_identity(cache)
        identity_path = cache / "input_identity.json"
        if identity_path.exists():
            raise FileExistsError("Common export identity already exists")
        write(identity_path, identity)
        write(self.root / "export_acceptance.json", export_gates)

        # The subprocess holds the CUDA context, while this parent retains
        # the same cooperative lock used by collection until metrics exits.
        import fcntl
        gpu = self.gpus[0]
        with open("/tmp/hycore_%d_gpu_%d.lock" % (os.getuid(), gpu), "a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            snapshot = subprocess.check_output(["nvidia-smi"], text=True)
            with (self.root / "metrics_nvidia_smi.txt").open("x", encoding="utf-8") as handle:
                handle.write(snapshot)
            uuid = idle_gpu(gpu)
            metrics_env = command_env()
            metrics_env["CUDA_VISIBLE_DEVICES"] = uuid
            metrics_env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            analysis = self.root / "analysis"
            if analysis.exists():
                raise FileExistsError("Analysis directory already exists")
            self.update("metrics", metrics_gpu=gpu, metrics_gpu_uuid=uuid)
            child, record = self.spawn("metrics", [MODULE + ".metrics", "--cache-dir", str(cache),
                    "--out-dir", str(analysis), "--device", "cuda:0"], env=metrics_env, gpu=gpu)
            self.wait_one(child, record)

        self.update("report")
        public = self.root / "public"
        if public.exists():
            raise FileExistsError("Public output directory already exists")
        child, record = self.spawn("report", [MODULE + ".report", "--analysis-dir", str(analysis),
                                              "--out-dir", str(public)])
        self.wait_one(child, record)
        publication = json.loads((public / "report_acceptance.json").read_text(encoding="utf-8"))
        if publication["figure_groups"] != 5 or len(publication["figure_files"]) != 10:
            raise RuntimeError("Publication figure set incomplete")
        if not (public / "report.md").is_file() or not (analysis / "summary.json").is_file():
            raise RuntimeError("Completed metrics/report files missing")
        acceptance = {"status": "completed", "ended_utc": now(), "labels": export_gates,
                      "shared_input": identity, "metrics_summary_sha256": sha(analysis / "summary.json"),
                      "report_acceptance": publication, "training_updates": 0}
        write(self.root / "pipeline_acceptance.json", acceptance)
        self.update("completed", status="completed", ended_utc=now())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings", required=True, type=Path)
    args = parser.parse_args(argv)
    pipeline = None
    try:
        pipeline = Pipeline(args.settings)
        pipeline.run()
    except Exception as error:
        if pipeline is not None:
            pipeline.update(status="failed", failed_phase=pipeline.state["phase"],
                            error_type=type(error).__name__, error=str(error), ended_utc=now())
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
