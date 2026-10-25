"""Pure CPU checks for feature-regression thresholds and invariants."""
from pathlib import Path
import importlib.util
import unittest

import numpy as np

spec = importlib.util.spec_from_file_location("gate85", Path(__file__).with_name("gate.py"))
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


class GateMetricsTests(unittest.TestCase):
    def test_identical_embeddings_pass_and_prediction_mismatch_fails(self):
        points = np.array([[.2, .1, .3], [.4, -.2, .1]])
        self.assertTrue(gate.compare_embeddings(points, points, [0, 1], [0, 1])["passed"])
        self.assertFalse(gate.compare_embeddings(points, points, [0, 1], [1, 1])["passed"])

    def test_coordinate_threshold_is_checked_even_with_matching_predictions(self):
        reference = np.full((80, 5), .1)
        actual = reference.copy()
        actual[0, 0] += .001
        report = gate.compare_embeddings(actual, reference, np.zeros(80), np.zeros(80))
        self.assertFalse(report["passed"])
        self.assertFalse(report["checks"]["max_abs_coordinate"])

    def test_invalid_directions_ball_and_shapes_fail_loudly(self):
        for points in (np.zeros((2, 3)), np.ones((2, 3))*10, np.full((2, 3), np.nan)):
            with self.assertRaises(ValueError):
                gate.compare_embeddings(points, points, [0, 0], [0, 0])
        with self.assertRaises(ValueError):
            gate.compare_embeddings(np.ones((2, 3))*.1, np.ones((3, 3))*.1, [0, 0], [0, 0])


if __name__ == "__main__":
    unittest.main()
