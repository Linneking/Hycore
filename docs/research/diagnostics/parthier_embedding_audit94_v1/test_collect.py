import unittest
import numpy as np
from .collect import centers, clean_state, epoch_of


class InputIdentityChecks(unittest.TestCase):
    def test_prefix_collision_cannot_silently_replace_weight(self):
        with self.assertRaises(ValueError):
            clean_state({'module.a': 1, 'a': 2})
        self.assertEqual(clean_state({'module.a': 1, 'b': 2}), {'a': 1, 'b': 2})

    def test_completed_epoch_required_not_raw_zero_based_epoch(self):
        with self.assertRaises(ValueError):
            epoch_of({'epoch': 188})
        self.assertEqual(epoch_of({'epoch': 188, 'completed_epochs': 189}), 189)

    def test_center_does_not_depend_on_batch_partition(self):
        ids = np.arange(9840, dtype=np.int64)
        whole, part = centers(ids)
        self.assertTrue(((whole >= 0) & (whole < 1024)).all())
        self.assertTrue(((part >= 0) & (part < 1024)).all())
        self.assertTrue(np.array_equal(part, np.concatenate([centers(x)[1] for x in np.array_split(ids, 37)])))


if __name__ == '__main__':
    unittest.main()
