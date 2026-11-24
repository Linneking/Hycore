"""Execute the published ScanObjectNN main with an optional HIER loss."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import itertools
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import types

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(HERE.parent / "scanobjectnn_hier_pair_20261120"))
import support

SEED = 12
POINTMLP_ARGS = ["--learning_rate", "0.1", "--epoch=300",
                 "--msg=Offv_pointmlp_hycore", "--workers=8"]
ARM_NAMES = ("SO-S12-H0", "SO-S12-HIER")
SOURCE = support.SOURCE / "main_pointmlp_hycore.py"
PROXY_DIR = "hier"


def add_once(source, old, new):
    if source.count(old) != 1:
        raise RuntimeError("Unexpected published source block: " + old.strip())
    return source.replace(old, new, 1)


def published_program(source, hier=False, smoke=False,
                      base_loss="        loss = std_loss + 0.01*t_loss + 0.001*h_loss"):
    # The worker is already bound to a verified physical GPU by the queue.
    source = add_once(source, 'os.environ["CUDA_VISIBLE_DEVICES"] = "0"',
                      '# Device visibility supplied by the launch queue')
    if hier:
        loss = base_loss
        source = add_once(source, loss, loss +
                          "\n        loss = hier_addon.loss(loss, parent_mu, label, net.manifold2)")
        step = "        optimizer.step()"
        source = add_once(source, step, step + "\n        hier_addon.step()")
        logger = '        logger.append([epoch, optimizer.param_groups[0][\'lr\'],'
        source = add_once(source, logger, "        hier_addon.finish_epoch(epoch, is_best)\n" + logger)
    if smoke:
        model = "    net = net.to(device)"
        source = add_once(source, model, model + "\n    smoke_trace.initial(net)")
        train = "    for batch_idx, (data, label) in enumerate(trainloader):"
        source = add_once(source, train,
                          "    for batch_idx, (data, label) in enumerate(smoke_trace.limit(trainloader)):\n"
                          "        smoke_trace.inputs(data, label)")
        crop = "        pos_mu, _ = net(pos_data)"
        source = add_once(source, crop, "        smoke_trace.crops(data, pos_data)\n" + crop)
        loss = base_loss
        source = add_once(source, loss, loss + "\n        smoke_trace.base(std_loss, t_loss, h_loss)")
        evaluate = "        for batch_idx, (data, label) in enumerate(testloader):"
        source = add_once(source, evaluate,
                          "        for batch_idx, (data, label) in enumerate(smoke_trace.limit(testloader)):")
    return source


class HierAddition:
    def __init__(self, workdir):
        self.path = workdir / PROXY_DIR
        self.branch = None

    def loss(self, base, whole, labels, manifold):
        if self.branch is None:
            import torch
            self.path.mkdir(exist_ok=False)
            support.SEED = SEED
            self.branch = support.Hierarchy(True, self.path, torch, epochs=300)
            # Source shuffle is without replacement. Local positions therefore
            # give the same equality/self-exclusion as distinct HDF row IDs.
            self.branch.ids = None
        return self.branch.add_loss(base, whole, labels, manifold)

    def step(self):
        self.branch.step()

    def finish_epoch(self, epoch, is_best):
        branch = self.branch
        branch.scheduler.step()
        report = dict(epoch=epoch + 1, optimizer_steps=branch.updates,
                      stats=dict(branch.stats), last_geometry=branch.last, selected_best=is_best)
        state = dict(epoch=epoch, optimizer_steps=branch.updates, **branch.state())
        temporary = self.path / "last.pth.tmp"
        branch.torch.save(state, temporary)
        os.replace(temporary, self.path / "last.pth")
        if is_best:
            branch.torch.save(state, self.path / "best.pth")
        with (self.path / "metrics.jsonl").open("a", encoding="utf-8") as file:
            file.write(json.dumps(report, allow_nan=False) + "\n")
        branch.stats.clear()


class SmokeTrace:
    """Capture exact initial model and input identity during two short epochs."""
    def __init__(self):
        self.rows = []

    @staticmethod
    def limit(loader):
        return itertools.islice(loader, 2)

    def initial(self, model):
        self.initial_sha = support.state_sha(model)

    def inputs(self, points, labels):
        self.rows.append(dict(cloud=support.tensor_sha(points), labels=support.tensor_sha(labels)))

    def crops(self, whole, part):
        self.rows[-1].update(whole=support.tensor_sha(whole), part=support.tensor_sha(part))

    def base(self, *losses):
        self.rows[-1]["base_losses"] = [float(loss.detach()) for loss in losses]

    def finish(self, workdir):
        import random
        import numpy as np
        import torch
        rng = dict(python=hashlib.sha256(repr(random.getstate()).encode()).hexdigest(),
                   numpy=hashlib.sha256(repr(np.random.get_state()).encode()).hexdigest(),
                   torch_cpu=support.tensor_sha(torch.get_rng_state()),
                   torch_cuda=support.tensor_sha(torch.cuda.get_rng_state()))
        support.write(workdir / "smoke_trace.json",
                      dict(initial_model_sha256=self.initial_sha, batches=self.rows, rng=rng))


def source_environment():
    # The official module names are unchanged. This only resolves upstream's
    # leftover modelsnet alias to the same ScanObject implementation.
    sys.path[:0] = [str(support.SOURCE), str(REPO / "pointnet2_ops_lib")]
    modelsnet = types.ModuleType("modelsnet")
    modelsnet.__path__ = [str(support.SOURCE / "models")]
    sys.modules["modelsnet"] = modelsnet


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
    workdir = args.output.resolve()
    workdir.mkdir(parents=True, exist_ok=False)
    os.chdir(workdir)
    # In the production path the author's loader reads both files itself.
    # Check the mount and byte identities before its normal initialization.
    data = support.SOURCE / "data" / "h5_files" / "main_split"
    for filename, sha256, _ in support.DATA_FILES.values():
        if support.sha_file(data / filename) != sha256:
            raise RuntimeError("Official loader input fingerprint mismatch")
    hier = args.arm == "SO-S12-HIER"
    actual_args = list(POINTMLP_ARGS)
    if args.smoke:
        actual_args[2] = "--epoch=2"
    sys.argv = [str(SOURCE), *actual_args]
    context = {"__name__": "__main__", "__file__": str(SOURCE)}
    if hier:
        context["hier_addon"] = HierAddition(workdir)
    if args.smoke:
        context["smoke_trace"] = SmokeTrace()
    source_environment()
    # Set source-defined HDF5 behavior before importing its dependencies.
    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
    program = published_program(support.source_text(), hier=hier, smoke=args.smoke)
    # The server's Conda PyTorch build exposes a module version without
    # Python wheel metadata. Read the actual imported libraries instead.
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    versions = {p: importlib.import_module(p).__version__ for p in
                ("torch", "geoopt", "numpy", "h5py", "sklearn", "scipy", "matplotlib")}
    extension = next((REPO / "pointnet2_ops_lib" / "pointnet2_ops").glob("_ext*.so"))
    manifest = dict(arm=args.arm, source_commit=head, author_commit=support.SOURCE_COMMIT,
                    author_source_sha256=support.TRAIN_SHA, program_sha256=hashlib.sha256(program.encode()).hexdigest(),
                    entry="author main()", seed=SEED, argv=actual_args, smoke=args.smoke,
                    started_at=support.now(), status="running", teacher=None, initial_checkpoint=None,
                    data_variant="ScanObjectNN PB_T50_RS / main_split",
                    data_sha256={k: v[1] for k, v in support.DATA_FILES.items()},
                    batch_size=32, steps_per_epoch=2 if args.smoke else 356,
                    test_rows=64 if args.smoke else 2882, hier_enabled=hier,
                    hierarchy=dict(curvature=.1, tangent_cap=2.3, GS=True, fixed_shell=False,
                                   proxy_count=512, initialization=.9, sample_K=20, proxy_K=20,
                                   sample_weight=.5, proxy_weight=.5, T=50, cands=32,
                                   optimizer="AdamW A1 (lr .005 to .0005, eps 1e-4)")
                    if hier else None,
                    gpu=os.environ.get("HYCORE_PHYSICAL_GPUS"), pid=os.getpid(),
                    gpu_uuid=os.environ.get("CUDA_VISIBLE_DEVICES"),
                    python=platform.python_version(), libraries=versions,
                    extension_sha256=support.sha_file(extension))
    support.write(workdir / "manifest.json", manifest)
    try:
        exec(compile(program, str(SOURCE), "exec"), context)
        if args.smoke:
            context["smoke_trace"].finish(workdir)
        manifest.update(status="completed", finished_at=support.now())
    except BaseException as error:
        manifest.update(status="failed", error=repr(error), finished_at=support.now())
        raise
    finally:
        support.write(workdir / "manifest.json", manifest)


if __name__ == "__main__":
    main()
