"""Run the ScanObjectNN DGCNN author entry with an optional V10 HIER branch."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(HERE.parent / "scanobjectnn_author_s12"))
import entry as author
from dispatch import reserve
from inter_hierarchy_MN40.hier_proxy_scratch_v8.queue import device_environment

support = author.support
SOURCE = support.SOURCE / "main_dgcnn_hycore.py"
SOURCE_SHA = "7521d795939b2ed6123bf6ca6bae17f52c1c9aebf4c71bb75591232b6296ca1c"
EPOCHS = 250
STEPS = 356


def published_program(source, hier, smoke):
    return author.published_program(source, hier=hier, smoke=smoke,
                                    base_loss="        loss = std_loss + 0.01*t_loss + 0.01*h_loss")


class DGCNNHierarchy(author.HierAddition):
    def __init__(self, workdir, seed):
        super().__init__(workdir)
        self.seed = seed

    def loss(self, base, whole, labels, manifold):
        if self.branch is None:
            import torch

            self.path.mkdir()
            support.SEED = self.seed
            self.branch = support.Hierarchy(True, self.path, torch, epochs=EPOCHS)
            self.branch.ids = None  # Distinct shuffled rows within this batch.
        return self.branch.add_loss(base, whole, labels, manifold)


def run_worker(args):
    if not os.environ.get("HYCORE_PHYSICAL_GPUS") or not os.environ.get("CUDA_VISIBLE_DEVICES", "").startswith("GPU-"):
        raise RuntimeError("Run via the idle-GPU dispatcher")
    head = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
    if head != args.source_commit:
        raise RuntimeError("Deployment commit mismatch")
    text = SOURCE.read_text(encoding="utf-8")
    if hashlib.sha256(text.encode()).hexdigest() != SOURCE_SHA:
        raise RuntimeError("Published DGCNN entry changed")
    program = published_program(text, args.hier, args.smoke)

    workdir = args.output.resolve()
    workdir.mkdir(parents=True, exist_ok=False)
    os.chdir(workdir)
    data = support.SOURCE / "data" / "h5_files" / "main_split"
    fingerprints = {split: support.sha_file(data / name) for split, (name, _, _) in support.DATA_FILES.items()}
    if fingerprints != {split: item[1] for split, item in support.DATA_FILES.items()}:
        raise RuntimeError("ScanObjectNN input fingerprint mismatch")
    author.source_environment()
    import geoopt
    import torch

    extension = next((REPO / "pointnet2_ops_lib" / "pointnet2_ops").glob("_ext*.so"))
    actual_args = ["--learning_rate", "0.1", f"--epoch={2 if args.smoke else EPOCHS}",
                   "--msg=Offv_dgcnn_hycore", "--workers=8", f"--seed={args.seed}"]
    sys.argv = [str(SOURCE), *actual_args]
    namespace = {"__name__": "__main__", "__file__": str(SOURCE)}
    if args.hier:
        namespace["hier_addon"] = DGCNNHierarchy(workdir, args.seed)
    if args.smoke:
        namespace["smoke_trace"] = author.SmokeTrace()
    manifest = dict(
        source_commit=head, author_commit=support.SOURCE_COMMIT, author_source_sha256=SOURCE_SHA,
        program_sha256=hashlib.sha256(program.encode()).hexdigest(), argv=actual_args,
        seed=args.seed, smoke=args.smoke, status="running", started_at=support.now(),
        teacher=None, initial_checkpoint=None, data_variant="PB_T50_RS / main_split",
        data_sha256=fingerprints, train_rows=11416, test_rows=2882,
        epochs=2 if args.smoke else EPOCHS, batch_size=32, steps_per_epoch=2 if args.smoke else STEPS,
        hier_enabled=args.hier, gpu=os.environ["HYCORE_PHYSICAL_GPUS"],
        gpu_uuid=os.environ["CUDA_VISIBLE_DEVICES"], pid=os.getpid(),
        python=platform.python_version(), torch=torch.__version__, cuda=torch.version.cuda,
        geoopt=geoopt.__version__, extension_sha256=support.sha_file(extension),
        hierarchy=dict(curvature=.1, tangent_cap=2.3, fixed_shell=False, GS=True, pool=32,
                       proxy_count=512, initialization=.9, sample_K=20, proxy_K=20, T=50,
                       sample_weight=.5, proxy_weight=.5, proxy_seed=args.seed,
                       gumbel_seed=args.seed + 300001, optimizer="AdamW .005 to .0005, eps 1e-4")
        if args.hier else None,
    )
    path = workdir / "manifest.json"
    support.write(path, manifest)
    try:
        exec(compile(program, str(SOURCE), "exec"), namespace)
        if args.smoke:
            namespace["smoke_trace"].finish(workdir)
        manifest["status"] = "completed"
    except BaseException as error:
        manifest.update(status="failed", error=repr(error))
        raise
    finally:
        manifest["finished_at"] = support.now()
        support.write(path, manifest)


def check_smokes(root):
    h0, hier = (json.loads((root / arm / "smoke_trace.json").read_text(encoding="utf-8"))
                for arm in ("H0", "HIER"))
    proxy = [json.loads(line) for line in
             (root / "HIER" / "hier" / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    keys = ("cloud", "labels", "whole", "part")
    first_losses = [trace["batches"][0]["base_losses"] for trace in (h0, hier)]
    checks = dict(
        initial_model=h0["initial_model_sha256"] == hier["initial_model_sha256"],
        batches=len(h0["batches"]) == len(hier["batches"]) == 4 and
        all(a[key] == b[key] for a, b in zip(h0["batches"], hier["batches"]) for key in keys),
        first_losses=all(math.isclose(a, b, abs_tol=1e-5, rel_tol=1e-5)
                         for a, b in zip(*first_losses)),
        main_rng=h0["rng"] == hier["rng"],
        proxy_steps=[row["optimizer_steps"] for row in proxy] == [2, 4],
        relations=all(row["stats"]["sample_triplets"] > 0 and row["stats"]["proxy_triplets"] > 0
                      for row in proxy),
    )
    passed = all(checks.values())
    support.write(root / "acceptance.json", dict(passed=passed, checks=checks,
                  initial_model_sha256=h0["initial_model_sha256"], first_base_losses=first_losses))
    if not passed:
        raise RuntimeError(f"DGCNN HIER preflight failed: {checks}")


def run_dispatch(args):
    folder = args.output.resolve()
    folder.mkdir(parents=True, exist_ok=False)
    state = dict(seed=args.seed, hier=args.hier, gpu=args.gpu, manager_pid=os.getpid(),
                 source_commit=args.source_commit, state="waiting_for_idle_gpu", started_at=support.now())
    queue = folder / "queue.json"
    support.write(queue, state)

    def worker(gpu, output, log_path, hier, smoke):
        env = dict(os.environ, **device_environment([gpu]), OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                   OPENBLAS_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1", MPLBACKEND="Agg")
        command = [sys.executable, "-u", str(Path(__file__).resolve()), "--worker",
                   "--output", str(output), "--source-commit", args.source_commit,
                   "--seed", str(args.seed)]
        if hier:
            command.append("--hier")
        if smoke:
            command.append("--smoke")
        with log_path.open("x", encoding="utf-8") as log:
            child = subprocess.Popen(command, env=env, stdin=subprocess.DEVNULL, stdout=log,
                                     stderr=subprocess.STDOUT)
            state.update(state="preflight" if smoke else "running", pid=child.pid,
                         gpu_uuid=gpu["uuid"], updated_at=support.now())
            support.write(queue, state)
            if code := child.wait():
                raise RuntimeError(f"{log_path.name} exited with status {code}")

    try:
        while True:
            with reserve(args.gpu) as gpu:
                def empty():
                    used = subprocess.check_output(
                        ["nvidia-smi", "-i", gpu["uuid"], "--query-gpu=memory.used,utilization.gpu",
                         "--format=csv,noheader,nounits"], text=True, timeout=20)
                    return tuple(int(value.strip()) for value in used.split(",")) == (0, 0)

                if empty():
                    time.sleep(2)
                    if empty():
                        preflight = folder / "preflight"
                        preflight.mkdir()
                        for arm, hier in (("H0", False), ("HIER", True)):
                            worker(gpu, preflight / arm, preflight / f"{arm}.log", hier, smoke=True)
                        check_smokes(preflight)
                        worker(gpu, folder / "main", folder / "main.log", args.hier, smoke=False)
                        break
            time.sleep(15)
        state.update(state="completed", exit_code=0)
    except BaseException as error:
        state.update(state="failed", error=repr(error))
        raise
    finally:
        state["updated_at"] = support.now()
        support.write(queue, state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--hier", action="store_true")
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.worker:
        run_worker(args)
    elif args.gpu is not None and not args.smoke:
        run_dispatch(args)
    else:
        parser.error("Dispatch requires --gpu. Smoke is internal to the dispatcher.")


if __name__ == "__main__":
    main()
