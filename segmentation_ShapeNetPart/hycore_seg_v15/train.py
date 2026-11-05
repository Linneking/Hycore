"""Reuse TAN-R training with the declared prefix overwrite loss."""

from ..hycore_seg_v11 import train as reference
from .losses import HyCoReSegLoss


def main():
    reference.main(loss_type=HyCoReSegLoss, run_format="hycore_seg_v15")


if __name__ == "__main__":
    main()
