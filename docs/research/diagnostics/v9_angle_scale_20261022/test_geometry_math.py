"""Independent geometric checks of the exact three-point minimax solution."""
from __future__ import annotations

import itertools
import unittest

import numpy as np

from geometry_math import pair_midpoint_depth, triple_center


def directions_2d(degrees):
    radians = np.deg2rad(degrees)
    return np.stack((np.cos(radians), np.sin(radians)), axis=-1)


class GeometryTests(unittest.TestCase):
    def test_pair_against_normalized_hyperboloid_sum(self):
        q = np.sqrt(.1)
        angles = np.array([0, 5, 10, 30, 60, 120, 180], dtype=float)
        for angle in angles:
            unit = directions_2d([0, angle])
            points = np.column_stack((np.full(2, np.cosh(q * 4.6)),
                                      np.sinh(q * 4.6) * unit))
            total = points.sum(axis=0)
            lorentz_sq = total[0] ** 2 - np.dot(total[1:], total[1:])
            midpoint = total / np.sqrt(lorentz_sq)
            measured = np.arccosh(max(midpoint[0], 1)) / q
            expected = pair_midpoint_depth(np.cos(np.deg2rad(angle)))
            self.assertAlmostEqual(float(expected), float(measured), places=7)

    def test_symmetric_three_in_eight_dimensions(self):
        result = triple_center(np.eye(8)[:3])
        np.testing.assert_allclose(result['weights'], np.ones(3)/3, atol=1e-12)
        self.assertAlmostEqual(float(result['support_cosine']), 1/np.sqrt(3), places=12)

    def test_obtuse_triangle_supported_by_two_endpoints(self):
        result = triple_center(directions_2d([-20, 0, 20]))
        np.testing.assert_allclose(result['weights'], [.5, 0, .5], atol=1e-10)
        self.assertAlmostEqual(float(result['depth']),
                               float(pair_midpoint_depth(np.cos(np.deg2rad(40)))), places=10)

    def test_origin_and_degenerate_cases(self):
        self.assertLess(float(triple_center(directions_2d([0, 120, 240]))['depth']), 1e-6)
        self.assertLess(float(triple_center(directions_2d([0, 180, 50]))['depth']), 1e-6)
        repeated = triple_center(directions_2d([0, 0, 40]))
        self.assertAlmostEqual(float(repeated['depth']),
                               float(pair_midpoint_depth(np.cos(np.deg2rad(40)))), places=10)
        self.assertAlmostEqual(float(triple_center(np.ones((3, 8)))['depth']), 4.6, places=10)

    def test_random_optimality_certificate_and_permutations(self):
        rng = np.random.default_rng(20261022)
        unit = rng.standard_normal((300, 3, 8))
        unit /= np.linalg.norm(unit, axis=-1, keepdims=True)
        result = triple_center(unit)
        self.assertLess(float(np.max(result['projection_residual'])), 1e-10)
        for order in itertools.permutations(range(3)):
            other = triple_center(unit[:, order])
            np.testing.assert_allclose(other['depth'], result['depth'], atol=1e-10)
        random_weights = rng.dirichlet(np.ones(3), size=500)
        gram = unit @ np.swapaxes(unit, -1, -2)
        candidate_sq = np.einsum('ki,bij,kj->bk', random_weights, gram, random_weights)
        self.assertTrue(np.all(result['support_cosine']**2 <= candidate_sq.min(axis=1) + 1e-10))
        for i, j in ((0, 1), (0, 2), (1, 2)):
            pair_depth = pair_midpoint_depth(gram[:, i, j])
            self.assertTrue(np.all(result['depth'] <= pair_depth + 1e-10))


if __name__ == '__main__':
    unittest.main()
