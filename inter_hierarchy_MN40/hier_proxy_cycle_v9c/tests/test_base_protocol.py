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
            super().__init__(c=.1)
            self.dim = dim

    namespace = {"torch": torch, "F": F, "random": random, "PoincareBall": SourceBall}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source_path), "exec"), namespace)
    return namespace


@unittest.skipIf(torch is None, "CPU Torch/Geoopt gate runs in the server environment")
class CycleBaseProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from inter_hierarchy_MN40.hier_proxy_cycle_v9c import base_protocol
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
                mu, nu, hier_margin=500. / n, contr_margin=2, ball_dim=256)
        ce = self.reference["cal_loss"](logits, labels)
        return ce + .01 * values[4] + .01 * values[5], ce, values

    def test_c01_full_loss_and_gradients_match_released_hutil_all_part_sizes(self):
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
        ball = geoopt.PoincareBall(c=.1).double()
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
        mu[:, 1] = 500. / 400
        nu[:32, 0], nu[32:, 0] = -1., 1.
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
        self.assertEqual(float(actual["intra_contrastive"].detach()), 2.)
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
        self.assertEqual((margin, last), (ref_margin * .5, ref_last))
        torch.testing.assert_close(points, original, rtol=0, atol=0)
        torch.testing.assert_close(child, ref_child, rtol=0, atol=0)
        self.assertEqual(child.untyped_storage().data_ptr(), points.untyped_storage().data_ptr())

    def test_half_margins_are_thresholds_not_loss_coefficient_changes(self):
        (mu, nu, _), _labels = self.fixture()
        for count in (200, 400, 600):
            radial = self.base.intra_losses(mu, mu, count)["intra_radial"]
            torch.testing.assert_close(radial, radial.new_tensor(500. / count), rtol=1e-13, atol=1e-13)

    def test_wrapper_part_first_whole_alias_bn_base_unchanged_and_pre_gs_source(self):
        import copy
        from .test_geometry import ProbeBackbone, gs_nodes
        from ..geometry import configure_backbone, whole_to_hier
        plain = configure_backbone(ProbeBackbone().double())
        branched = copy.deepcopy(plain)
        a = self.base.HyCoReTrainingForward(plain, hier_enabled=False).train()
        b = self.base.HyCoReTrainingForward(branched, hier_enabled=True).train()
        points = torch.randn(4, 32, 3, dtype=torch.float64)
        state = random.getstate()
        out_a = a(points.clone(), 16, 7)
        random.setstate(state)
        out_b = b(points.clone(), 16, 7)
        self.assertNotIn("hier_source_mu", out_a)
        for name in ("mu", "nu", "logits"):
            torch.testing.assert_close(out_a[name], out_b[name], rtol=0, atol=0)
        for model in (plain, branched):
            self.assertEqual([call[0] for call in model.calls], [True, False])
            self.assertEqual([call[1].shape[-1] for call in model.calls], [7, 16])
            self.assertEqual(int(model.bn.num_batches_tracked), 2)
            torch.testing.assert_close(model.calls[0][1], model.calls[1][1][:, :, :7], rtol=0, atol=0)
            self.assertIsNone(model.v9c_geometry.pre_gs)
            self.assertFalse(model.v9c_geometry.capture_enabled)
        self.assertTrue(out_b["input_stats"]["child_aliases_whole"])
        self.assertTrue(out_b["input_stats"]["whole_changed_by_child"])
        self.assertEqual(gs_nodes(out_b["mu"]), 1)
        self.assertEqual(gs_nodes(out_b["nu"]), 1)
        self.assertEqual(gs_nodes(out_b["hier_source_mu"]), 0)
        hier = whole_to_hier(out_b["hier_source_mu"])
        self.assertEqual(gs_nodes(hier), 1)
        base_grad_a, = torch.autograd.grad(out_a["mu"].square().sum() + out_a["logits"].square().sum(), plain.emb.weight)
        base_grad_b, = torch.autograd.grad(out_b["mu"].square().sum() + out_b["logits"].square().sum(), branched.emb.weight, retain_graph=True)
        torch.testing.assert_close(base_grad_a, base_grad_b, rtol=1e-12, atol=1e-14)
        probe = torch.arange(hier.numel(), dtype=hier.dtype).reshape_as(hier)
        hier_grad, = torch.autograd.grad((hier * probe).sum(), branched.emb.weight)
        self.assertGreater(float(hier_grad.norm()), 1e-5)
        self.assertTrue(bool(torch.isfinite(hier_grad).all()))

    def test_global_hier_source_gather_preserves_pre_gs_branch(self):
        from ..geometry import whole_to_hier
        (mu, nu, logits), labels = self.fixture()
        source = mu.detach().clone().requires_grad_()
        outputs = self.outputs(mu, nu, logits)
        outputs["hier_source_mu"] = source
        values = self.base.global_base_losses(outputs, labels)
        torch.testing.assert_close(values["hier_source_mu"], source, rtol=0, atol=0)
        loss = whole_to_hier(values["hier_source_mu"])[:, 0].sum()
        gradient, = torch.autograd.grad(loss, source)
        self.assertGreater(float(gradient.norm()), 0.)
        # Base loss consumes only its GS path, not the parallel HIER source.
        self.assertIsNone(torch.autograd.grad(values["base"], source, allow_unused=True)[0])

    def test_capture_cleared_after_forward_failure_and_eval_does_not_retain_graph(self):
        from .test_geometry import ProbeBackbone
        from ..geometry import configure_backbone
        backbone = configure_backbone(ProbeBackbone()).eval()
        wrapper = self.base.HyCoReTrainingForward(backbone, hier_enabled=True)
        with mock.patch.object(backbone.classifier, "forward", side_effect=RuntimeError("sentinel")):
            with self.assertRaisesRegex(RuntimeError, "sentinel"):
                wrapper(torch.randn(4, 32, 3), 16, 7)
        self.assertIsNone(backbone.v9c_geometry.pre_gs)
        self.assertFalse(backbone.v9c_geometry.capture_enabled)
        backbone(torch.randn(4, 3, 16))
        self.assertIsNone(backbone.v9c_geometry.pre_gs)

    def test_telemetry_c01_domain_and_disabled_hier(self):
        from ..telemetry import EpochTelemetry
        from ..geometry import HIER_RADIUS
        ids = torch.arange(64)
        labels = torch.arange(32).repeat_interleave(2)
        telemetry = EpochTelemetry(ids, labels, hier_enabled=False)
        mu = torch.zeros(64, 256)
        mu[:, 0] = HIER_RADIUS
        telemetry.update(ids=ids, labels=labels, logits=labels, mu=mu, nu=mu * .5,
                         scalar_metrics={"hier_weight": 0.})
        summary = telemetry.summary()
        self.assertEqual(summary["geometry"]["whole"]["c"], .1)
        self.assertAlmostEqual(summary["geometry"]["whole"]["depth"]["mean"], 4.6, places=5)
        self.assertTrue(summary["geometry"]["whole"]["inverse_metric_hook_applied"])
        self.assertFalse(summary["geometry"]["hier_view_domain"]["enabled"])
        self.assertIsNone(summary["geometry"]["hier_view_domain"]["target_whole_depth"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
