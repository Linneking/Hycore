import unittest

import torch

from inter_hierarchy_MN40.parthier_v1.feature_queue import PartFeatureQueue


def features(batch=2, dimension=4):
    values = torch.arange(batch * 3 * dimension, dtype=torch.float64).reshape(batch, 3, dimension)
    return (values / (values.numel() * 4)).requires_grad_()


class QueueTests(unittest.TestCase):
    def test_fifo_is_part_capacity_and_preserves_global_row_scale_order(self):
        queue = PartFeatureQueue(7)
        first, second = features(), features() + .001
        queue.enqueue(first, torch.tensor([11, 12]), 0)
        queue.enqueue(second, torch.tensor([21, 22]), 1)
        entries = queue.entries()
        self.assertEqual(len(queue), 7)
        self.assertEqual(entries["object_ids"].tolist(), [12, 21, 21, 21, 22, 22, 22])
        self.assertEqual(entries["scales"].tolist(), [2, 0, 1, 2, 0, 1, 2])
        self.assertEqual(entries["steps"].tolist(), [0, 1, 1, 1, 1, 1, 1])
        expected = torch.cat((first.detach().reshape(6, 4), second.detach().reshape(6, 4)))[-7:]
        torch.testing.assert_close(entries["features"], expected, rtol=0, atol=0)

    def test_enqueue_detaches_and_owns_feature_storage(self):
        queue = PartFeatureQueue(2048)
        parts = features()
        expected = parts.detach().clone()
        queue.enqueue(parts, torch.tensor([4, 5]), 2)
        with torch.no_grad():
            parts.zero_()
        self.assertFalse(queue.features.requires_grad)
        self.assertIsNone(queue.features.grad_fn)
        torch.testing.assert_close(queue.features, expected.reshape(6, 4), rtol=0, atol=0)

    def test_cpu_snapshot_restore_and_next_fifo_update_are_identical(self):
        queue = PartFeatureQueue(7)
        queue.enqueue(features(), torch.tensor([1, 2]), 4)
        snapshot = queue.state_dict()
        restored = PartFeatureQueue(7)
        restored.load_state_dict(snapshot)
        snapshot["features"].zero_()
        queue.enqueue(features() + .001, torch.tensor([3, 4]), 5)
        restored.enqueue(features() + .001, torch.tensor([3, 4]), 5)
        for name, tensor in queue.entries().items():
            self.assertEqual(restored.entries()[name].device.type, "cpu")
            torch.testing.assert_close(tensor, restored.entries()[name], rtol=0, atol=0)

    def test_disabled_empty_and_restore_validation(self):
        queue = PartFeatureQueue(0)
        queue.enqueue(features(), torch.tensor([1, 2]), 0)
        self.assertEqual(len(queue), 0)
        restored = PartFeatureQueue(0)
        restored.load_state_dict(queue.state_dict())
        self.assertEqual(len(restored), 0)
        with self.assertRaises(ValueError):
            PartFeatureQueue(1).load_state_dict(queue.state_dict())
        active = PartFeatureQueue(7)
        active.enqueue(features(), torch.tensor([1, 2]), 0)
        corrupt = active.state_dict()
        corrupt["scales"][0] = 3
        with self.assertRaises(ValueError):
            active.load_state_dict(corrupt)

    def test_enabled_empty_restore_and_invalid_shape_preserve_existing_state(self):
        queue = PartFeatureQueue(2048)
        restored = PartFeatureQueue(2048)
        restored.load_state_dict(queue.state_dict())
        self.assertEqual(restored.features.shape, (0, 0))
        restored.enqueue(features(), torch.tensor([4, 5]), 2)
        self.assertEqual(restored.features.shape, (6, 4))
        before = restored.state_dict()
        corrupt = restored.state_dict()
        corrupt["features"] = torch.tensor(1.)
        with self.assertRaises(ValueError):
            restored.load_state_dict(corrupt)
        with self.assertRaises(ValueError):
            restored.enqueue(features(), torch.tensor([6, 7]), 1)
        for name in ("features", "object_ids", "scales", "steps"):
            torch.testing.assert_close(getattr(restored, name), before[name], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
