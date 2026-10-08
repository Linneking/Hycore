"""Independent identity and neighborhood checks; no private cache dependencies."""
import unittest
from unittest.mock import patch

import numpy as np

import hier_original_augmentation_stability_v1 as audit


class ArrayArchive(dict):
    """Minimal read-only NPZ interface for in-memory identity tests."""
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class AugmentationIdentityChecks(unittest.TestCase):
    def setUp(self):
        # Deliberately nonconsecutive, unsorted identities; expected reference
        # positions are determined by identity rather than labels or array index.
        self.ids = np.array([101, 7, 90])
        self.labels = np.array([3, 8, 9])
        self.x = np.array([[.2, 0., 0.], [0., .3, 0.], [0., 0., .4]])
        self.proxies = np.array([[.1, 0., 0.], [0., .2, 0.]])
        self.archive = ArrayArchive(
            sample_ids=np.array([90, 101, 7]), labels=np.array([9, 3, 8]),
            mu=np.array([[0., 0., .4], [.2, 0., 0.], [0., .3, 0.]]),
            proxy_ball=self.proxies.copy(), c=np.asarray(.1),
        )

    def aligned(self, labels=None, proxies=None):
        # Mock only file transport; production validation/alignment runs normally.
        with patch.object(audit, 'sha256', return_value='synthetic-digest'), \
             patch.object(audit.np, 'load', return_value=self.archive):
            return audit.load_aligned(
                {'cache': 'synthetic_view.npz'}, self.ids,
                self.labels if labels is None else labels,
                self.proxies if proxies is None else proxies, .1,
            )

    def test_arbitrary_identity_permutation_restores_reference_order(self):
        result = self.aligned()
        np.testing.assert_array_equal(result['ids'], [101, 7, 90])
        np.testing.assert_array_equal(result['labels'], [3, 8, 9])
        np.testing.assert_array_equal(result['x'], [[.2, 0., 0.], [0., .3, 0.], [0., 0., .4]])
        np.testing.assert_array_equal(result['unit'], np.eye(3))

    def test_same_identity_with_wrong_label_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'different labels'):
            self.aligned(labels=np.array([3, 8, 10]))

    def test_even_small_proxy_coordinate_change_is_rejected(self):
        modified = self.proxies.copy()
        modified[0, 1] = 1e-8
        with self.assertRaisesRegex(ValueError, 'proxies differ'):
            self.aligned(proxies=modified)

    def test_jaccard_keeps_disjoint_rows_as_zero_in_mean(self):
        first = np.array([[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]])
        second = np.array([[1, 2, 5, 6], [5, 6, 7, 8], [4, 3, 2, 1]])
        result = audit.set_jaccard_rows(first, second)
        # Intersections/unions are 2/6, 0/8, 4/4; all three count equally.
        np.testing.assert_allclose(result, [1 / 3, 0., 1.], atol=1e-15)
        self.assertAlmostEqual(float(result.mean()), 4 / 9, places=15)


if __name__ == '__main__':
    unittest.main()
