import unittest
import numpy as np
from ..protocol import full_training_ids, select_best


class FullTrainingProtocol(unittest.TestCase):
    def test_every_official_training_object_is_eligible_without_holdout(self):
        labels = np.tile(np.arange(40), 246)
        ids = full_training_ids(labels)
        np.testing.assert_array_equal(ids, np.arange(9840))
        self.assertEqual(len(set(ids.tolist())), 9840)

    def test_reject_heldout_training_pool_and_missing_classes(self):
        with self.assertRaises(ValueError):
            full_training_ids(np.tile(np.arange(40), 222)[:8856])
        with self.assertRaises(ValueError):
            full_training_ids(np.zeros(9840, dtype=int))

    def metrics(self, oa=94., aa=90., ce=.2, count=2468):
        return dict(test_oa=oa, test_aa=aa, test_ce=ce, test_count=count,
                    test_smoke_partial=False)

    def test_strict_test_oa_selection_not_ce_or_aa(self):
        best = select_best(None, self.metrics(), "test", 1)
        self.assertIs(select_best(best, self.metrics(94., 95., .1), "test", 2), best)
        self.assertIs(select_best(best, self.metrics(93.), "test", 3), best)
        improved = select_best(best, self.metrics(94.01, 88., .3), "test", 4)
        self.assertEqual(improved["epoch"], 4)

    def test_selection_rejects_partial_or_nonfinite_test(self):
        with self.assertRaises(ValueError):
            select_best(None, self.metrics(count=32), "test", 1)
        with self.assertRaises(ValueError):
            select_best(None, self.metrics(oa=float("nan")), "test", 1)

    def test_smoke_scores_training_objects_and_does_not_mislabel_validation(self):
        metrics = dict(smoke_train_oa=20., smoke_train_aa=15., smoke_train_ce=3.,
                       smoke_train_count=32)
        result = select_best(None, metrics, "smoke_train", 1)
        self.assertNotIn("test_oa", result)
        self.assertNotIn("val_oa", result)


if __name__ == "__main__":
    unittest.main()
