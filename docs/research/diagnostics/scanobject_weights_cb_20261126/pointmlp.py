"""Run the author seed-12 PointMLP with a common HIER loss multiplier."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "scanobjectnn_author_s12"))
import entry as author


WEIGHTS = {"SO-S12-HIER-L0.1": 0.1, "SO-S12-HIER-L1.0": 1.0}


class WeightedHierarchy(author.HierAddition):
    def __init__(self, workdir, weight):
        super().__init__(workdir)
        self.weight = weight

    def loss(self, base, whole, labels, manifold):
        # Isolate the existing 0.5 * (sample + proxy) term without subtracting
        # the base loss. Scale it before adding the unchanged native objective.
        half_regularizer = super().loss(base.new_zeros(()), whole, labels, manifold)
        return base + (2 * self.weight) * half_regularizer


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=WEIGHTS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    output = args.output.resolve()
    weight = WEIGHTS[args.arm]

    original_argv = sys.argv
    original_addon = author.HierAddition
    original_write = author.support.write

    def write_manifest(path, value):
        if path == output / "manifest.json":
            hierarchy = dict(value["hierarchy"], sample_weight=weight, proxy_weight=weight)
            value = dict(value, arm=args.arm, hierarchy=hierarchy)
        return original_write(path, value)

    author_args = [str(HERE / "pointmlp.py"), "--arm", "SO-S12-HIER",
                   "--output", str(output), "--source-commit", args.source_commit]
    if args.smoke:
        author_args.append("--smoke")
    try:
        author.HierAddition = lambda workdir: WeightedHierarchy(workdir, weight)
        author.support.write = write_manifest
        sys.argv = author_args
        author.main()
    finally:
        sys.argv = original_argv
        author.support.write = original_write
        author.HierAddition = original_addon


if __name__ == "__main__":
    main()
