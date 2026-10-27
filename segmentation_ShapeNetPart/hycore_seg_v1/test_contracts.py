"""Scientific/data/state contracts, run on CPU before server training."""

import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import geoopt
import h5py
import numpy as np
import torch
from torch.nn import functional as F

from .data import ShapeNetPart
from .losses import HyCoReSegLoss, choose_parts, negative_indices, relation_losses
from .metrics import SegMetrics, shape_ious
from .model import DGCNNPartSeg
from .utils import restore_rng, rng_state
from .vendor.dgcnn_model import DGCNN_partseg


class Contracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_baseline_matches_vendor_eval(self):
        torch.manual_seed(11)
        vendor = DGCNN_partseg(SimpleNamespace(k=4, emb_dims=1024, dropout=.5), 50).eval()
        model = DGCNNPartSeg(k=4).eval()
        model.load_state_dict(vendor.state_dict(), strict=True)
        x, label = torch.randn(2, 3, 24), F.one_hot(torch.tensor([0, 3]), 16).float()
        with torch.no_grad():
            torch.testing.assert_close(model(x, label), vendor(x, label), atol=0, rtol=0)

    def test_baseline_matches_vendor_train_gradients_and_bn(self):
        torch.manual_seed(12)
        vendor = DGCNN_partseg(SimpleNamespace(k=4, emb_dims=1024, dropout=.5), 50).train()
        model = DGCNNPartSeg(k=4).train()
        model.load_state_dict(vendor.state_dict())
        x, labels = torch.randn(3, 3, 24), F.one_hot(torch.tensor([0, 1, 3]), 16).float()
        target = torch.randint(50, (3, 24))
        torch.manual_seed(77)
        a = vendor(x, labels)
        F.cross_entropy(a, target).backward()
        torch.manual_seed(77)
        b = model(x, labels)
        F.cross_entropy(b, target).backward()
        torch.testing.assert_close(a, b, atol=0, rtol=0)
        for (_, left), (_, right) in zip(vendor.named_parameters(), model.named_parameters()):
            if left.grad is not None:
                torch.testing.assert_close(left.grad, right.grad, atol=0, rtol=0)
        for name, value in vendor.state_dict().items():
            torch.testing.assert_close(value, model.state_dict()[name], atol=0, rtol=0)

    def test_head_preserves_initial_backbone_weight_and_parameter_count(self):
        torch.manual_seed(14)
        baseline = DGCNNPartSeg(k=4)
        torch.manual_seed(14)
        hyperbolic = DGCNNPartSeg("mobius", k=4)
        self.assertEqual(sum(p.numel() for p in baseline.parameters() if p.requires_grad),
                         sum(p.numel() for p in hyperbolic.parameters() if p.requires_grad))
        for name, value in baseline.state_dict().items():
            torch.testing.assert_close(value, hyperbolic.state_dict()[name], atol=0, rtol=0)

    def test_hyperbolic_head_shape_and_finite_gradients(self):
        model = DGCNNPartSeg("mobius", k=4)
        logits, features = model(torch.randn(2, 3, 24), torch.eye(16)[:2], True)
        self.assertEqual(logits.shape, (2, 50, 24))
        self.assertEqual(features.shape, (2, 24, 128))
        logits.square().mean().backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))

    def test_geometry_against_analytic_unit_ball_distance(self):
        ball = geoopt.PoincareBall(c=torch.tensor(1., dtype=torch.float64))
        x = torch.tensor([[.1, .2], [-.2, .3]], dtype=torch.float64)
        y = torch.tensor([[.25, -.1], [.1, .2]], dtype=torch.float64)
        expected = torch.acosh(1 + 2 * (x - y).square().sum(-1) /
                               ((1 - x.square().sum(-1)) * (1 - y.square().sum(-1))))
        torch.testing.assert_close(ball.dist(x, y), expected, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(ball.dist0(x), 2 * torch.atanh(x.norm(dim=-1)), rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(ball.logmap0(ball.expmap0(x)), x, rtol=1e-12, atol=1e-12)

    def test_radial_gradient_sign_for_independent_leaves(self):
        ball = geoopt.PoincareBall(c=1.).double()
        whole = torch.tensor([[.2, 0.]], dtype=torch.float64, requires_grad=True)
        part = torch.tensor([[.1, 0.]], dtype=torch.float64, requires_grad=True)
        radial, _, _ = relation_losses(ball, whole, part, torch.tensor([0]), torch.tensor([False]), 2., 4.)
        radial.backward()
        self.assertGreater(float((whole.grad * whole).sum()), 0)
        self.assertLess(float((part.grad * part).sum()), 0)

    def test_feature_selection_and_gradients_no_input_mutation(self):
        values = torch.tensor([[[0.], [1.], [4.], [10.], [11.]],
                               [[-3.], [-1.], [2.], [3.], [8.]]], requires_grad=True)
        original = values.detach().clone()
        mean, indices, centers, size = choose_parts(values, torch.Generator().manual_seed(9), 2, 2)
        for i in range(2):
            distances = (values[i, :, 0].detach() - values[i, centers[i], 0].detach()).square()
            distances[centers[i]] = -1
            torch.testing.assert_close(indices[i], distances.argsort(stable=True)[:2])
        mean.sum().backward()
        expected = torch.zeros_like(values)
        expected.scatter_(1, indices[:, :, None], .5)
        torch.testing.assert_close(values.grad, expected)
        torch.testing.assert_close(values, original)
        self.assertEqual(size, 2)

    def test_only_different_category_negatives(self):
        categories = torch.tensor([2, 2, 4, 7])
        negative, valid = negative_indices(categories, torch.Generator().manual_seed(3))
        self.assertTrue(valid.all())
        self.assertTrue((categories[negative] != categories).all())
        _, valid = negative_indices(torch.tensor([2, 2]), torch.Generator())
        self.assertFalse(valid.any())

    def test_user_margin_and_reduction(self):
        criterion = HyCoReSegLoss(part_min=2, part_max=2)
        loss, info = criterion(torch.randn(3, 8, 4) * .05, torch.tensor([1, 1, 1]))
        self.assertEqual(info["gamma"], 500.)
        self.assertEqual(info["delta"], 4.)
        self.assertEqual(info["contr"], 0.)
        self.assertAlmostEqual(float(loss), .01 * info["hier"], places=5)
        for n, gamma in ((200, 5.), (400, 2.5), (600, 1000/600)):
            self.assertEqual(criterion.gamma_base / n, gamma)

    def test_disabled_regularizer_does_not_sample(self):
        criterion = HyCoReSegLoss(alpha=0, beta=0)
        state = criterion.generator.get_state().clone()
        with patch(__package__ + ".losses.choose_parts", side_effect=AssertionError("must not sample")):
            loss, info = criterion(torch.randn(2, 24, 8), torch.tensor([0, 1]))
        self.assertEqual(float(loss), 0.)
        self.assertTrue(info["disabled"])
        torch.testing.assert_close(state, criterion.generator.get_state())

    def test_whole_part_pool_before_exp_and_finite_backbone_gradient(self):
        criterion = HyCoReSegLoss(part_min=4, part_max=4)
        features = (torch.randn(3, 12, 8) * .1).requires_grad_()
        saved_generator = criterion.generator.get_state()
        loss, info = criterion(features, torch.tensor([0, 1, 2]))
        loss.backward()
        self.assertTrue(torch.isfinite(features.grad).all())
        self.assertGreater(float(features.grad.norm()), 0.)
        criterion.generator.set_state(saved_generator)
        pooled, _, _, _ = choose_parts(features.detach(), criterion.generator, 4, 4)
        expected = criterion.ball.dist0(criterion.ball.expmap0(pooled)).mean()
        self.assertAlmostEqual(info["part_depth"], float(expected), places=6)

    def test_metrics_object_macro_empty_union_and_illegal_prediction(self):
        target = np.array([[4, 4], [0, 1], [0, 1]])
        predictions = np.array([[4, 4], [0, 1], [0, 0]])
        categories = np.array([1, 0, 0])
        scores = shape_ious(predictions, target, categories)
        np.testing.assert_allclose(scores, [1., 1., .625])
        metric = SegMetrics()
        metric.update(predictions, target, categories)
        result = metric.result()
        self.assertAlmostEqual(result["instance_miou"], .875)
        self.assertAlmostEqual(result["class_miou"], .90625)
        self.assertAlmostEqual(shape_ious([[0, 0]], [[4, 4]], [1])[0], .5)

    def test_dataset_point_label_permutation_and_eval_repeat(self):
        with tempfile.TemporaryDirectory() as temp:
            for split in ("train", "val"):
                with h5py.File(Path(temp) / (split + ".h5"), "w") as f:
                    xyz = np.arange(24, dtype=np.float32).reshape(1, 8, 3)
                    f["data"], f["pid"], f["label"] = xyz, np.arange(8).reshape(1, 8), [0]
                    f.create_dataset("ids", data=["synset/object"], dtype=h5py.string_dtype())
            train_set = ShapeNetPart(temp, "train", 8)
            xyz, _, parts, _ = train_set[0]
            np.testing.assert_array_equal(xyz[:, 0] / 3, parts)
            val_set = ShapeNetPart(temp, "val", 8)
            np.testing.assert_array_equal(val_set[0][0], val_set[0][0])
            train_set.close()
            val_set.close()

    def test_rsgd_equals_sgd_without_momentum_on_euclidean_parameters(self):
        a = torch.nn.Parameter(torch.tensor([.2, -.3]))
        b = torch.nn.Parameter(a.detach().clone())
        first = torch.optim.SGD([a], lr=.1, momentum=0, weight_decay=1e-4)
        second = geoopt.optim.RiemannianSGD([b], lr=.1, momentum=0, weight_decay=1e-4)
        for _ in range(3):
            first.zero_grad(); second.zero_grad()
            a.square().sum().backward(); b.square().sum().backward()
            first.step(); second.step()
        torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_rsgd_initial_momentum_is_a_recorded_protocol_difference(self):
        a = torch.nn.Parameter(torch.tensor([.2, -.3], dtype=torch.float64))
        b = torch.nn.Parameter(a.detach().clone())
        initial = a.detach().clone()
        first = torch.optim.SGD([a], lr=.1, momentum=.9, weight_decay=1e-4)
        second = geoopt.optim.RiemannianSGD([b], lr=.1, momentum=.9, weight_decay=1e-4)
        a.square().sum().backward(); b.square().sum().backward()
        first.step(); second.step()
        expected_sgd = initial - .1 * (2 * initial + 1e-4 * initial)
        # Geoopt 0.5.1 initializes the buffer with the gradient, then updates it.
        expected_rsgd = initial - .1 * (1.9 * 2 * initial + 1e-4 * initial)
        torch.testing.assert_close(a, expected_sgd, rtol=0, atol=1e-16)
        torch.testing.assert_close(b, expected_rsgd, rtol=0, atol=1e-16)
        self.assertFalse(torch.equal(a, b))

    def test_complete_state_next_step_restore(self):
        torch.manual_seed(88)
        network = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Dropout(.5), torch.nn.Linear(4, 2))
        optimizer = torch.optim.SGD(network.parameters(), .1, momentum=.9)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 200, eta_min=.001)
        data, target = torch.randn(3, 4), torch.tensor([0, 1, 0])
        def step(net, opt):
            opt.zero_grad(); F.cross_entropy(net(data), target).backward(); opt.step()
        step(network, optimizer); scheduler.step()
        saved = copy.deepcopy({"model": network.state_dict(), "opt": optimizer.state_dict(),
                               "scheduler": scheduler.state_dict(), "rng": rng_state()})
        step(network, optimizer)
        replica = copy.deepcopy(network)
        replica.load_state_dict(saved["model"])
        opt = torch.optim.SGD(replica.parameters(), .1, momentum=.9)
        opt.load_state_dict(saved["opt"])
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 200, eta_min=.001)
        sch.load_state_dict(saved["scheduler"])
        restore_rng(saved["rng"])
        step(replica, opt)
        for a, b in zip(network.parameters(), replica.parameters()):
            torch.testing.assert_close(a, b, atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
