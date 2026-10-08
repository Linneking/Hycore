"""Independent mathematical/identity checks for cross-curvature frozen geometry."""
import unittest

import numpy as np

from hier_original_vs_v7_geometry_v1 import (
    class_geometry, direction_resultant, radial_coordinates, retrieval,
    stable_distance_matrix, stable_topk, unit_rows,
)


class GeometryChecks(unittest.TestCase):
    def test_radial_coordinates_and_curvature_scaling(self):
        rng = np.random.default_rng(10)
        u, _ = unit_rows(rng.normal(size=(12, 7)))
        rho = np.linspace(.02, .9, 12)
        for c in (.1, 1., 3.):
            result = radial_coordinates(u * (rho / np.sqrt(c))[:, None], c)
            np.testing.assert_allclose(result['rho'], rho, atol=1e-15)
            np.testing.assert_allclose(result['q'], 2 * np.arctanh(rho), atol=1e-14)
            np.testing.assert_allclose(result['d0'] * np.sqrt(c), result['q'], atol=1e-14)

    def test_distance_matches_independent_hyperbolic_cosine_law(self):
        rng = np.random.default_rng(1)
        up, _ = unit_rows(rng.normal(size=(8, 9)))
        ux, _ = unit_rows(rng.normal(size=(19, 9)))
        for c in (.1, 1., 3.):
            hp = rng.uniform(.2, 5., 8)
            hx = rng.uniform(.2, 5., 19)
            p = up * (np.tanh(np.sqrt(c) * hp / 2) / np.sqrt(c))[:, None]
            x = ux * (np.tanh(np.sqrt(c) * hx / 2) / np.sqrt(c))[:, None]
            expected = np.arccosh(np.cosh(np.sqrt(c) * hp)[:, None] * np.cosh(np.sqrt(c) * hx)[None, :] -
                                 np.sinh(np.sqrt(c) * hp)[:, None] * np.sinh(np.sqrt(c) * hx)[None, :] * (up @ ux.T)) / np.sqrt(c)
            np.testing.assert_allclose(stable_distance_matrix(p, x, c), expected, atol=2e-12)

    def test_uniform_whole_depth_order_equals_direction_for_each_proxy_radius(self):
        rng = np.random.default_rng(2)
        u, _ = unit_rows(rng.normal(size=(47, 13)))
        v, _ = unit_rows(rng.normal(size=(9, 13)))
        ids = rng.permutation(np.arange(47) + 50)
        for c in (.1, 1., 3.):
            p = v * (np.linspace(.1, .92, len(v)) / np.sqrt(c))[:, None]
            x = u * (.66 / np.sqrt(c))
            raw = stable_topk(stable_distance_matrix(p, x, c), ids, 16)
            direction = stable_topk(1 - v @ u.T, ids, 16)
            np.testing.assert_array_equal(raw, direction)

    def test_cutoff_ties_sort_by_stable_identity(self):
        values = np.array([[3., 0., 0., 1., 0., 0.], [1., 1., 1., 1., 1., 1.]])
        ids = np.array([9, 12, 3, 2, 1, 4])
        np.testing.assert_array_equal(stable_topk(values, ids, 3), [[4, 2, 5], [4, 3, 2]])

    def test_macro_cosines_match_explicit_pairs_unequal_classes(self):
        rng = np.random.default_rng(3)
        u, _ = unit_rows(rng.normal(size=(18, 6)))
        labels = np.repeat([2, 9, 15], [3, 5, 10])
        radial = radial_coordinates(u * .4, 1.)
        result, rows, arrays = class_geometry(u, labels, radial, np.arange(len(u)))
        within, between = [], []
        for cl in np.unique(labels):
            a = u[labels == cl]
            tri = np.triu_indices(len(a), 1)
            within.append((a @ a.T)[tri].mean())
        for ca, cb in [(2, 9), (2, 15), (9, 15)]:
            between.append((u[labels == ca] @ u[labels == cb].T).mean())
        self.assertAlmostEqual(result['within_class_cosine_macro'], np.mean(within), places=13)
        self.assertAlmostEqual(result['between_class_cosine_macro_pairs'], np.mean(between), places=13)
        self.assertEqual(arrays['centers'].shape, (3, 6))

    def test_proxy_resultant_is_not_pairwise_cosine(self):
        u = np.array([[1., 0.], [0., 1.]])
        result = direction_resultant(u)
        self.assertAlmostEqual(result['R'], np.sqrt(.5))
        self.assertAlmostEqual(result['mean_pairwise_cosine_excluding_self'], 0.)

    def test_retrieval_keeps_all_zero_occupancy_proxies(self):
        rng = np.random.default_rng(4)
        u, _ = unit_rows(rng.normal(size=(23, 7)))
        v = np.repeat(u[:1], 5, axis=0)
        x, p = .3 * u, .2 * v
        ids = np.arange(len(x))
        labels = np.arange(len(x)) % 3
        result, arrays = retrieval(p, x, u, v, 1., ids, labels, radial_coordinates(x, 1.)['d0'], proxy_chunk=2)
        self.assertEqual(result['raw']['occupancy']['fixed_P'], 5)
        self.assertEqual(len(arrays['raw_proxy_occupancy']), 5)
        self.assertEqual(result['raw']['occupancy']['occupied_proxy_count'], 1)
        self.assertEqual(result['raw']['maximum_repeated_quartet'], 5)
        np.testing.assert_array_equal(arrays['raw_top_index'], arrays['direction_top_index'])


if __name__ == '__main__':
    unittest.main()
