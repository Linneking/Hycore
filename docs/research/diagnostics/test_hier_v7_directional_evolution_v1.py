"""Mathematical checks for frozen angular geometry and shared alignment."""
import unittest
import numpy as np

from hier_v7_directional_evolution_v1 import (
    class_geometry, weighted_alignment, top_rows, usage_for_epoch, proxy_affinity,
    proxy_direction_groups,
)


class GeometryChecks(unittest.TestCase):
    def test_macro_pair_formula_matches_explicit_pairs_with_unequal_classes(self):
        rng = np.random.default_rng(22)
        x = rng.normal(size=(14, 7))
        x /= np.linalg.norm(x, axis=1)[:, None]
        y = np.repeat([0, 1, 2], [2, 5, 7])
        result, rows, _ = class_geometry(x, y, np.arange(3))
        within, between = [], []
        for label in range(3):
            v = x[y == label]
            a, b = np.triu_indices(len(v), 1)
            within.append(float((v @ v.T)[a, b].mean()))
        for a in range(3):
            for b in range(a+1, 3):
                between.append(float((x[y == a] @ x[y == b].T).mean()))
        self.assertAlmostEqual(np.mean(within), result['within_class_cosine_macro']['mean'], places=13)
        self.assertAlmostEqual(np.mean(between), result['between_class_cosine_macro_pairs']['mean'], places=13)
        np.testing.assert_allclose([r['within_cosine_excluding_self'] for r in rows], within, atol=1e-13)

    def test_leave_one_out_own_center_and_margin_match_explicit_refitting(self):
        rng = np.random.default_rng(15)
        x = rng.normal(size=(18, 5))
        x /= np.linalg.norm(x, axis=1)[:, None]
        y = np.repeat(np.arange(3), 6)
        result, _, arrays = class_geometry(x, y, np.arange(3))
        predicted = []
        margins = []
        for i in range(len(x)):
            fitted = []
            for c in range(3):
                mask = y == c
                mask[i] = False
                vector = x[mask].mean(axis=0)
                fitted.append(vector/np.linalg.norm(vector))
            score = x[i] @ np.array(fitted).T
            predicted.append(np.argmax(score))
            other = np.delete(score, y[i]).max()
            margins.append(np.degrees(np.arccos(np.clip(other, -1, 1)) - np.arccos(np.clip(score[y[i]], -1, 1))))
        np.testing.assert_array_equal(arrays['whole_loo_predicted_class_index'], predicted)
        np.testing.assert_allclose(arrays['whole_loo_angle_margin_deg'], margins, atol=1e-12)
        self.assertAlmostEqual(result['loo_direction_accuracy_micro'], np.mean(np.array(predicted) == y))

    def test_one_shared_alignment_recovers_joint_rotation_and_proxy_scores(self):
        rng = np.random.default_rng(4)
        whole = rng.normal(size=(22, 5))
        whole /= np.linalg.norm(whole, axis=1)[:, None]
        proxy = rng.normal(size=(8, 5))
        proxy /= np.linalg.norm(proxy, axis=1)[:, None]
        q, _ = np.linalg.qr(rng.normal(size=(5, 5)))
        next_whole, next_proxy = whole @ q, proxy @ q
        labels = np.repeat([0, 1], [5, 17])
        rotation, fit = weighted_alignment(whole, next_whole, labels)
        np.testing.assert_allclose(next_whole @ rotation, whole, atol=1e-12)
        np.testing.assert_allclose(next_proxy @ rotation, proxy, atol=1e-12)
        np.testing.assert_allclose((next_proxy @ rotation) @ whole.T, proxy @ whole.T, atol=1e-12)
        self.assertLess(fit['class_balanced_mean_squared_residual'], 1e-23)

    def test_tie_order_uses_sample_ids_not_array_order(self):
        rows = top_rows(np.ones((2, 6)), np.array([12, 4, 8, 15, 1, 3]), 4)
        np.testing.assert_array_equal(rows, [[4, 5, 1, 2], [4, 5, 1, 2]])

    def test_missing_usage_is_unknown_and_cannot_be_classified_inactive(self):
        records, selected = usage_for_epoch({'proxy_usage': []}, 0, np.arange(4))
        self.assertEqual(records, {})
        self.assertIsNone(selected)

    def test_mean_affinity_distinguishes_center_coherence_and_exclusion_pool(self):
        # Class0 tightly aligned near first axis; class1 has same center direction
        # but low resultant. Mean-vector affinity explicitly retains that fact.
        x = np.array([[1., 0.], [1., 0.], [.1, np.sqrt(.99)], [.1, -np.sqrt(.99)]])
        labels = np.array([0, 0, 1, 1])
        ids = np.array([1, 2, 3, 4])
        p = np.array([[1., 0.]])
        summary, arrays, rows = proxy_affinity(p, x, labels, ids, np.arange(2), {}, ks=(2, 4))
        self.assertEqual(int(arrays['mean_vector_nearest_class'][0]), 0)
        self.assertAlmostEqual(arrays['mean_vector_top2_cosine_margin'][0], .9)
        self.assertEqual(rows.shape, (1, 4))
        mask = labels == 1
        after, after_arrays, _ = proxy_affinity(p, x, labels, ids, np.arange(2), {}, mask=mask, ks=(2,))
        self.assertEqual(after['retained_class_count'], 1)
        self.assertEqual(int(after_arrays['nearest_class'][0]), 1)

    def test_proxy_group_dispersion_matches_brute_force_and_skips_singleton_pairs(self):
        rng = np.random.default_rng(5)
        proxy = rng.normal(size=(8, 4))
        proxy /= np.linalg.norm(proxy, axis=1)[:, None]
        assigned = np.array([0, 0, 0, 1, 1, 1, 1, 2])
        result, angles = proxy_direction_groups(proxy, assigned, np.arange(3))
        i, j = np.triu_indices(8, 1)
        self.assertAlmostEqual(result['all_proxy_pair_cosine_excluding_self'], (proxy @ proxy.T)[i, j].mean(), places=13)
        within = []
        for label in [0, 1]:
            values = proxy[assigned == label]
            a, b = np.triu_indices(len(values), 1)
            within.append((values @ values.T)[a, b].mean())
        self.assertAlmostEqual(result['assigned_group_within_cosine_macro']['mean'], np.mean(within), places=13)
        self.assertEqual(result['assigned_group_within_cosine_macro']['n'], 2)
        self.assertAlmostEqual(angles[-1], 0., places=5)


if __name__ == '__main__':
    unittest.main()
