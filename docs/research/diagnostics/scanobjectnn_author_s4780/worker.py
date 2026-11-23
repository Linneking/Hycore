"""Run the approved ScanObjectNN author-entry pair with seed 4780."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import os
from pathlib import Path
import platform
import subprocess
import sys

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(HERE.parent / "scanobjectnn_author_s12"))
import entry as author

support = author.support
SEED = 4780
ARM_NAMES = ("SO-S4780-H0", "SO-S4780-HIER")
REFERENCE_COMMIT = "b876d85238a8e8e254df2adab9a4dd26ee721476"


def author_args(smoke):
    epochs = 2 if smoke else 300
    return [f"--epoch={epochs}" if arg == "--epoch=300" else arg
            for arg in author.POINTMLP_ARGS] + [f"--seed={SEED}"]


class HierAddition(author.HierAddition):
    def loss(self, base, whole, labels, manifold):
        if self.branch is None:
            import torch

            self.path.mkdir(exist_ok=False)
            support.SEED = SEED
            self.branch = support.Hierarchy(True, self.path, torch, epochs=300)
            # Natural shuffle uses distinct rows, so within-batch positions
            # preserve the sample identity rule of the seed-12 author run.
            self.branch.ids = None
        return self.branch.add_loss(base, whole, labels, manifold)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=ARM_NAMES, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    head = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
    if head != args.source_commit:
        raise RuntimeError("Deployment commit mismatch")
    if not os.environ.get("HYCORE_PHYSICAL_GPUS") or not os.environ.get("CUDA_VISIBLE_DEVICES", "").startswith("GPU-"):
        raise RuntimeError("Run through the idle-GPU dispatcher")

    workdir = args.output.resolve()
    workdir.mkdir(parents=True, exist_ok=False)
    os.chdir(workdir)
    data = support.SOURCE / "data" / "h5_files" / "main_split"
    data_sha = {split: support.sha_file(data / name) for split, (name, _, _) in support.DATA_FILES.items()}
    if data_sha != {split: record[1] for split, record in support.DATA_FILES.items()}:
        raise RuntimeError("Official loader input fingerprint mismatch")

    hier = args.arm == ARM_NAMES[1]
    actual_args = author_args(args.smoke)
    sys.argv = [str(author.SOURCE), *actual_args]
    context = {"__name__": "__main__", "__file__": str(author.SOURCE)}
    if hier:
        context["hier_addon"] = HierAddition(workdir)
    if args.smoke:
        context["smoke_trace"] = author.SmokeTrace()
    author.source_environment()
    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    program = author.published_program(support.source_text(), hier=hier, smoke=args.smoke)
    versions = {package: importlib.import_module(package).__version__ for package in
                ("torch", "geoopt", "numpy", "h5py", "sklearn", "scipy", "matplotlib")}
    extension = next((REPO / "pointnet2_ops_lib" / "pointnet2_ops").glob("_ext*.so"))
    manifest = dict(
        arm=args.arm, source_commit=head, reference_commit=REFERENCE_COMMIT,
        author_commit=support.SOURCE_COMMIT, author_source_sha256=support.TRAIN_SHA,
        program_sha256=hashlib.sha256(program.encode()).hexdigest(),
        entry="author main()", seed=SEED, argv=actual_args, smoke=args.smoke,
        started_at=support.now(), status="running", teacher=None, initial_checkpoint=None,
        data_variant="ScanObjectNN PB_T50_RS / main_split", data_sha256=data_sha,
        epochs=2 if args.smoke else 300, batch_size=32, steps_per_epoch=2 if args.smoke else 356,
        test_rows=64 if args.smoke else 2882, hier_enabled=hier,
        hierarchy=dict(curvature=.1, tangent_cap=2.3, GS=True, fixed_shell=False, proxy_count=512,
                       initialization=.9, sample_K=20, proxy_K=20, sample_weight=.5, proxy_weight=.5,
                       T=50, cands=32, optimizer="AdamW A1 (lr .005 to .0005, eps 1e-4)",
                       proxy_seed=SEED, gumbel_seed=SEED + 300001, first_relationship_seed=SEED)
        if hier else None,
        gpu=os.environ["HYCORE_PHYSICAL_GPUS"], pid=os.getpid(), gpu_uuid=os.environ["CUDA_VISIBLE_DEVICES"],
        python=platform.python_version(), libraries=versions, extension_sha256=support.sha_file(extension),
    )
    manifest_path = workdir / "manifest.json"
    support.write(manifest_path, manifest)
    try:
        exec(compile(program, str(author.SOURCE), "exec"), context)
        if args.smoke:
            context["smoke_trace"].finish(workdir)
        manifest.update(status="completed", finished_at=support.now())
    except BaseException as error:
        manifest.update(status="failed", error=repr(error), finished_at=support.now())
        raise
    finally:
        support.write(manifest_path, manifest)


if __name__ == "__main__":
    main()
