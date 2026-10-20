"""CPU gates for V9's unchanged, uncapped original-HyCoRe base objective.

The reference loss functions are compiled from the released hutil.py AST;
we do not copy their formulas into an independently drifting test oracle.
"""
from __future__ import annotations

import ast
import contextlib
from pathlib import Path
import random
import unittest
from unittest import mock

try:
    import torch
    from torch import nn
    import torch.nn.functional as F
    import geoopt
except ModuleNotFoundError as exc:
    if exc.name not in {"torch", "geoopt"}:
        raise
    torch = None


def released_functions():
    """Load only CPU math/cropping functions, avoiding unrelated model imports."""
    source_path = Path(__file__).resolve().parents[3] / "classification_ModelNet40" / "hutil.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8-sig"))
    names = {"knn", "get_children_np", "hype_triplet_losses", "cal_loss"}
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    if len(nodes) != len(names):
        raise AssertionError("released hutil CPU reference functions are missing")

    class SourceBall(geoopt.PoincareBall):
        def __init__(self, *, c, dim):
            super().__init__(c=c)
            self.dim = dim

    namespace = {"torch": torch, "F": F, "random": random, "PoincareBall": SourceBall}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source_path), "exec"), namespace)
    return namespace


@unittest.skipIf(torch is None, "CPU Torch/Geoopt gate runs in the server environment")
class V9BaseProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from inter_hierarchy_MN40.hier_proxy_scratch_v9 import base_protocol
        cls.base = base_protocol
        cls.reference = released_functions()
        torch.set_num_threads(2)

    def fixture(self, dtype=None):
        dtype = torch.float64 if dtype is None else dtype
        generator = torch.Generator().manual_seed(22)
        mu = torch.randn(64, 256, generator=generator, dtype=dtype) * .002
        nu = torch.randn(64, 256, generator=generator, dtype=dtype) * .002
        # Global flip crosses ranks. A rank-local flip gives different distances.
        mu[:32, 0] += .25
        mu[32:, 0] -= .25
        nu[:32, 0] += .15
        nu[32:, 0] -= .15
        logits = torch.randn(64, 40, generator=generator, dtype=dtype)
        return tuple(value.requires_grad_() for value in (mu, nu, logits)), torch.arange(32).repeat_interleave(2)

    @staticmethod
    def outputs(mu, nu, logits, n=400):
        return {"mu": mu, "nu": nu, "logits": logits,
                "whole_count": 900, "child_count": n}

    def source_loss(self, mu, nu, logits, labels, n, *, align_ball_dtype=False):
        # Production is FP32. For the extra FP64 diagnostic, align the
        # reference manifold dtype with the implementation before softplus(c).
        # Merely upcasting a ball's stored inverse-softplus parameter changes
        # its represented c by FP32 rounding; this is not a loss discrepancy.
        ball_type = self.reference["PoincareBall"]
        def aligned_ball(**kwargs):
            return ball_type(**kwargs).to(device=mu.device, dtype=mu.dtype)
        scope = mock.patch.dict(self.reference, {"PoincareBall": aligned_ball}) if align_ball_dtype else contextlib.nullcontext()
        with scope:
            values = self.reference["hype_triplet_losses"](
                mu, nu, hier_margin=1000. / n, contr_margin=4, ball_dim=256)
        ce = self.reference["cal_loss"](logits, labels)
        return ce + .01 * values[4] + .01 * values[5], ce, values

    def test_c1_full_loss_and_gradients_match_released_hutil_all_part_sizes(self):
        for n in (200, 400, 600):
            with self.subTest(part_points=n):
                inputs, labels = self.fixture()
                actual = self.base.global_base_losses(self.outputs(*inputs, n=n), labels)
                reference_inputs = tuple(value.detach().clone().requires_grad_() for value in inputs)
                expected, ce, components = self.source_loss(*reference_inputs, labels, n, align_ball_dtype=True)
                for key, reference in zip(("whole_radius", "part_radius", "positive_distance",
                                           "negative_distance", "intra_contrastive", "intra_radial"), components):
                    torch.testing.assert_close(actual[key], reference, rtol=1e-13, atol=1e-13)
                torch.testing.assert_close(actual["ce"], ce, rtol=0, atol=0)
                torch.testing.assert_close(actual["base"], expected, rtol=1e-13, atol=1e-13)
                for gradient, reference in zip(torch.autograd.grad(actual["base"], inputs),
                                                torch.autograd.grad(expected, reference_inputs)):
                    self.assertTrue(bool(torch.isfinite(gradient).all()))
                    self.assertGreater(float(gradient.norm()), 0.)
                    torch.testing.assert_close(gradient, reference, rtol=1e-10, atol=1e-12)
                self.assertEqual(actual["global_batch_size"], 64)
                self.assertTrue(torch.equal(actual["gold"], labels))

    def test_production_fp32_matches_unmodified_released_loss_and_gradients(self):
        # No dtype alignment or source formula changes in this production gate.
        for n in (200, 400, 600):
            with self.subTest(part_points=n):
                inputs, labels = self.fixture(dtype=torch.float32)
                actual = self.base.global_base_losses(self.outputs(*inputs, n=n), labels)
                reference_inputs = tuple(value.detach().clone().requires_grad_() for value in inputs)
                expected, ce, components = self.source_loss(*reference_inputs, labels, n)
                for key, reference in zip(("whole_radius", "part_radius", "positive_distance",
                                           "negative_distance", "intra_contrastive", "intra_radial"), components):
                    torch.testing.assert_close(actual[key], reference, rtol=0, atol=0)
                torch.testing.assert_close(actual["ce"], ce, rtol=0, atol=0)
                torch.testing.assert_close(actual["base"], expected, rtol=0, atol=0)
                for gradient, reference in zip(torch.autograd.grad(actual["base"], inputs),
                                                torch.autograd.grad(expected, reference_inputs)):
                    self.assertTrue(bool(torch.isfinite(gradient).all()))
                    torch.testing.assert_close(gradient, reference, rtol=1e-6, atol=1e-10)

    def test_contrastive_negative_is_global_flip_not_rank_local_flip(self):
        (mu, nu, logits), labels = self.fixture()
        actual = self.base.global_base_losses(self.outputs(mu, nu, logits), labels)
        ball = geoopt.PoincareBall(c=1.).double()
        global_distance = ball.dist(mu, nu.flip(0)).mean()
        local_negative = torch.cat((nu[:32].flip(0), nu[32:].flip(0)))
        local_distance = ball.dist(mu, local_negative).mean()
        torch.testing.assert_close(actual["negative_distance"], global_distance)
        self.assertGreater(float((global_distance - local_distance).detach().abs()), .1)

    def test_two_rank_concatenation_uses_the_same_global64_loss(self):
        full, labels = self.fixture()
        reference_inputs = tuple(value.detach().clone().requires_grad_() for value in full)
        expected = self.base.global_base_losses(self.outputs(*reference_inputs), labels)
        local = tuple(value[:32] for value in full)
        gathered = [torch.cat((a, b[32:]), 0) for a, b in zip(local, full)]
        with mock.patch.object(self.base, "gather_with_grad", side_effect=gathered), \
                mock.patch.object(self.base, "gather_without_grad",
                                  side_effect=(labels, torch.tensor([[900, 400], [900, 400]]))):
            actual = self.base.global_base_losses(self.outputs(*local), labels[:32])
        torch.testing.assert_close(actual["base"], expected["base"], rtol=0, atol=0)
        self.assertEqual(actual["global_batch_size"], 64)
        for gradient, reference in zip(torch.autograd.grad(actual["base"], full),
                                        torch.autograd.grad(expected["base"], reference_inputs)):
            torch.testing.assert_close(gradient, reference, rtol=0, atol=0)

    def test_released_max_hinge_exact_zero_keeps_half_gradient(self):
        class TieBall:
            def __init__(self, **_kwargs):
                pass
            def to(self, **_kwargs):
                return self
            def dist(self, x, y):
                return x[:, 0] + y[:, 0]
            def dist0(self, x):
                return x[:, 1]

        mu = torch.zeros(64, 256, dtype=torch.float64)
        nu = torch.zeros_like(mu)
        mu[:, 1] = 1000. / 400
        nu[:32, 0], nu[32:, 0] = -2., 2.
        mu.requires_grad_()
        nu.requires_grad_()
        logits = torch.zeros(64, 40, dtype=torch.float64, requires_grad=True)
        labels = torch.arange(32).repeat_interleave(2)
        inputs = (mu, nu, logits)
        references = tuple(value.detach().clone().requires_grad_() for value in inputs)
        with mock.patch.object(geoopt, "PoincareBall", TieBall), \
                mock.patch.dict(self.reference, {"PoincareBall": TieBall}):
            actual = self.base.global_base_losses(self.outputs(*inputs), labels)
            expected, _, _ = self.source_loss(*references, labels, 400)
            gradients = torch.autograd.grad(actual["base"], inputs)
            expected_gradients = torch.autograd.grad(expected, references)
        self.assertEqual(float(actual["intra_radial"].detach()), 0.)
        self.assertEqual(float(actual["intra_contrastive"].detach()), 4.)
        for gradient, reference in zip(gradients, expected_gradients):
            torch.testing.assert_close(gradient, reference, rtol=0, atol=0)
        half_weight = .01 / (2 * 64)
        torch.testing.assert_close(gradients[0][:, 1], torch.full((64,), -half_weight, dtype=mu.dtype),
                                   rtol=0, atol=0)
        torch.testing.assert_close(gradients[1][:, 1], torch.full((64,), half_weight, dtype=nu.dtype),
                                   rtol=0, atol=0)

    def test_rejects_wrong_global_batch_and_same_class_flip(self):
        inputs, labels = self.fixture()
        with self.assertRaisesRegex(ValueError, "expected global64"):
            self.base.global_base_losses(self.outputs(*(value[:32] for value in inputs)), labels[:32])
        with self.assertRaisesRegex(RuntimeError, "shares the anchor class"):
            self.base.global_base_losses(self.outputs(*inputs), torch.zeros_like(labels))

    def test_rejects_rank_specific_whole_or_child_counts(self):
        inputs, labels = self.fixture()
        for other in ((900, 300), (850, 400)):
            with self.subTest(other_rank_counts=other):
                with mock.patch.object(self.base, "gather_without_grad",
                                       side_effect=(labels, torch.tensor([[900, 400], other]))):
                    with self.assertRaisesRegex(RuntimeError, "counts must be shared"):
                        self.base.global_base_losses(self.outputs(*inputs), labels)

    def test_crop_alias_and_mutation_match_released_operator(self):
        points = torch.randn(4, 3, 32, generator=torch.Generator().manual_seed(33))
        original = points.clone()
        saved_rng = random.getstate()
        try:
            random.seed(22)
            ref_margin, ref_child, ref_last = self.reference["get_children_np"](
                original, starting=31, kmin=7, kmax=7)
            random.seed(22)
            random.randint(7, 7)  # Released function consumes the point-count draw first.
            margin, child, last = self.base.get_children_alias(points, 7)
        finally:
            random.setstate(saved_rng)
        self.assertEqual((margin, last), (ref_margin, ref_last))
        torch.testing.assert_close(points, original, rtol=0, atol=0)
        torch.testing.assert_close(child, ref_child, rtol=0, atol=0)
        self.assertEqual(child.untyped_storage().data_ptr(), points.untyped_storage().data_ptr())

    def test_shared_bn_part_then_whole_and_no_base_cap_or_gs(self):
        class ProbeBackbone(nn.Module):
            def __init__(self):
                super().__init__()
                self.bn = nn.BatchNorm1d(3)
                self.embedding = nn.Parameter(torch.tensor([.995, 0., 0.]))
                self.classifier = nn.Linear(3, 40)
                self.calls = []
            def forward(self, points, emb=False):
                batch_feature = self.bn(points.mean(-1))
                mu = self.embedding[None].expand(len(points), -1) + batch_feature * 0
                self.calls.append((bool(emb), points.detach().clone(), mu))
                return mu, self.classifier(mu)

        backbone = ProbeBackbone()
        initial = {key: value.detach().clone() for key, value in backbone.state_dict().items()}
        model = self.base.HyCoReTrainingForward(backbone).train()
        self.assertIs(model.backbone, backbone)
        self.assertEqual(set(model.state_dict()), {"backbone." + key for key in initial})
        for key, value in initial.items():
            torch.testing.assert_close(model.state_dict()["backbone." + key], value, rtol=0, atol=0)
        output = model(torch.randn(4, 32, 3), 16, 7)
        self.assertEqual([call[0] for call in backbone.calls], [True, False])
        self.assertEqual(int(backbone.bn.num_batches_tracked), 2)
        self.assertEqual([call[1].shape[-1] for call in backbone.calls], [7, 16])
        torch.testing.assert_close(backbone.calls[0][1], backbone.calls[1][1][:, :, :7], rtol=0, atol=0)
        self.assertTrue(output["input_stats"]["child_aliases_whole"])
        self.assertTrue(output["input_stats"]["whole_aliases_source"])
        self.assertTrue(output["input_stats"]["whole_changed_by_child"])
        self.assertIs(output["nu"], backbone.calls[0][2])
        self.assertIs(output["mu"], backbone.calls[1][2])
        self.assertGreater(float(geoopt.PoincareBall(c=1.).dist0(output["mu"]).min()), 4.6)
        gradient, = torch.autograd.grad(output["mu"].sum() + output["nu"].sum(), backbone.embedding)
        torch.testing.assert_close(gradient, torch.full_like(gradient, 8.), rtol=0, atol=0)
        self.assertFalse(any(module._forward_hooks for module in model.modules()))

    def test_restored_c1_margins_and_geometry_are_distinct_from_v8(self):
        from inter_hierarchy_MN40.hier_proxy_scratch_v8.base_protocol import intra_losses as v8_intra
        (mu, nu, _), _labels = self.fixture()
        original = self.base.intra_losses(mu, nu, 400)
        v8 = v8_intra(mu, nu, 400)
        self.assertGreater(float((original["whole_depth"] - v8["whole_depth"]).abs()), .005)
        self.assertGreater(float((original["intra_contrastive"] - v8["intra_contrastive"]).detach().abs()), 1.)
        self.assertGreater(float((original["intra_radial"] - v8["intra_radial"]).detach().abs()), 1.)
        for n in (200, 400, 600):
            radial = self.base.intra_losses(mu, mu, n)["intra_radial"]
            torch.testing.assert_close(radial, radial.new_tensor(1000. / n), rtol=1e-14, atol=1e-14)


if __name__ == "__main__":
    unittest.main(verbosity=2)