"""Run the published ScanObjectNN DGCNN HIER pair with class-balanced batches."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch

from sampler import BalancedBatches


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
SEED = 4780
ARMS = ("DG-S4780-CB32-H0", "DG-S4780-CB32-HIER")
HOOK_MODULE = "scanobject_dgcnn_cb_hooks"
SAMPLER_LOG = "cb_sampler.jsonl"
PUBLISHED_RUNNER = HERE.parent / "scanobjectnn_dgcnn_hier_20261124" / "run.py"


class AuditedBatches(BalancedBatches):
    def __init__(self, labels, steps):
        super().__init__(labels, SEED, steps)
        self.labels = np.asarray(labels, dtype=np.int64).reshape(-1)
        if len(self.labels) != 11416 or np.unique(self.labels).tolist() != list(range(15)):
            raise ValueError("Unexpected ScanObjectNN training labels")

    def __iter__(self):
        # Mirror the original shuffled RandomSampler's iterator seed draw.
        # The CB row stream itself is independent of the Torch global RNG.
        torch.empty((), dtype=torch.int64).random_()
        digest = hashlib.sha256()
        appearances = np.zeros(15, dtype=np.int64)
        reverse_same_class_pairs = 0
        first_batches = []
        for step, rows in enumerate(super().__iter__(), 1):
            labels = self.labels[rows]
            counts = np.bincount(labels, minlength=15)
            if len(rows) != 32 or len(set(rows)) != 32 or sorted(counts) != [2] * 13 + [3, 3]:
                raise RuntimeError(f"Invalid CB32 batch at epoch {self.epoch + 1}, step {step}")
            digest.update(np.asarray(rows, dtype="<i4").tobytes())
            appearances += counts
            reverse_same_class_pairs += int(np.count_nonzero(labels[:16] == labels[::-1][:16]))
            if step <= 2:
                first_batches.append(dict(rows=rows, labels=labels.tolist(), class_counts=counts.tolist()))
            if step == len(self):
                record = dict(epoch=self.epoch + 1, steps=step, batch_rows_sha256=digest.hexdigest(),
                              class_appearances=appearances.tolist(), first_batches=first_batches,
                              reverse_same_class_pairs=reverse_same_class_pairs,
                              reverse_pairs=16 * step)
                with Path(SAMPLER_LOG).open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(record, allow_nan=False) + "\n")
            yield rows


def cb_train_loader(loader, dataset, workers, steps):
    sampler = AuditedBatches(dataset.label, steps)
    return loader(dataset, num_workers=workers, batch_sampler=sampler)


def with_cb32(source, add_once, steps):
    loader = (
        "    train_loader = DataLoader(ScanObjectNN(partition='training', num_points=args.num_points), "
        "num_workers=args.workers,\n"
        "                              batch_size=args.batch_size, shuffle=True, drop_last=True)"
    )
    replacement = (
        f"    from {HOOK_MODULE} import cb_train_loader\n"
        "    train_loader = cb_train_loader(DataLoader,\n"
        "        ScanObjectNN(partition='training', num_points=args.num_points), "
        f"args.workers, {steps})"
    )
    source = add_once(source, loader, replacement)
    epoch = "    for epoch in range(start_epoch, args.epoch):"
    return add_once(source, epoch, epoch + "\n        train_loader.batch_sampler.set_epoch(epoch)")


def load_published_runner():
    sys.path[:0] = [str(PUBLISHED_RUNNER.parent), str(REPO)]
    spec = importlib.util.spec_from_file_location("published_scanobject_dgcnn", PUBLISHED_RUNNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)

    published = load_published_runner()
    output = args.output.resolve()
    original_program = published.published_program
    original_write = published.support.write
    original_argv = sys.argv

    def balanced_program(source, hier, smoke):
        program = original_program(source, hier, smoke)
        return with_cb32(program, published.author.add_once, 2 if smoke else published.STEPS)

    def write_manifest(path, value):
        if path == output / "manifest.json":
            value = dict(value, arm=args.arm, sampler="CB32", sampler_seed=SEED,
                         sampler_log=SAMPLER_LOG, classes_per_batch=15)
        return original_write(path, value)

    try:
        sys.modules[HOOK_MODULE] = sys.modules[__name__]
        published.published_program = balanced_program
        published.support.write = write_manifest
        published.run_worker(SimpleNamespace(output=output, source_commit=args.source_commit,
                                              seed=SEED, hier=args.arm == ARMS[1], smoke=args.smoke))
    finally:
        published.published_program = original_program
        published.support.write = original_write
        sys.argv = original_argv
        sys.modules.pop(HOOK_MODULE, None)


if __name__ == "__main__":
    main()
