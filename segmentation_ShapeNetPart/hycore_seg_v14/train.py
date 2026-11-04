"""Use the matched tail trainer with the two declared structural variants."""

from ..hycore_seg_v13 import train as shared_train
from ..hycore_seg_v13.monitoring import GeometryMonitor
from .model import NativeTail2


class BallScoreMonitor(GeometryMonitor):
    score_node = "m3"
    score_coordinate_space = "poincare_ball_coordinates"


def main():
    args = shared_train.arguments(variants=("shape", "late_m"))
    monitor = GeometryMonitor if args.variant == "shape" else BallScoreMonitor
    shared_train.main(args, model_type=NativeTail2, monitor_type=monitor, run_format="hycore_seg_v14")


if __name__ == "__main__":
    main()
