"""CPU contracts for the new hook-only V8 ablation; no data or GPU forwards."""
from argparse import Namespace
from types import SimpleNamespace
import unittest

import torch

from ..v8_nohook import (
    CANONICAL_NAME, CHECKPOINT_VERSION, EXPECTED_INITIAL_SHA256,
    FinalGeometryNoHook, configure_backbone, fixed_configuration,
    isolated_runner_bindings, validate_variant,
)
from ...hier_proxy_scratch_v8.geometry import (
    C, FinalGeometry, configure_backbone as configure_prior, to_ball,
)


def args(**changes):
    result = Namespace(control="B0-32", seed=22, epochs=300, steps_per_epoch=307,
        workers=2, eval_batch_size=32, train_eval_every=10, save_every=20,
        audit_epochs=(1,5,20,40,99,100,160,200,240,300), smoke=False,
        smoke_eval_batches=1, warmup_epochs=0, lambda_hier=0.)
    for key, value in changes.items():
        setattr(result, key, value)
    return result


class NoHookContracts(unittest.TestCase):
    def test_forward_identical_backward_matches_removed_metric(self):
        torch.manual_seed(17)
        raw = torch.randn(32, 16, dtype=torch.float64) * .45
        raw[0] *= 4
        old = raw.clone().requires_grad_()
        new = raw.clone().requires_grad_()
        mapped_old, mapped_new = to_ball(old), to_ball(new, hook=False)
        torch.testing.assert_close(mapped_old, mapped_new, atol=0, rtol=0)
        weights = torch.randn_like(mapped_old)
        old_grad, = torch.autograd.grad((mapped_old * weights).sum(), old)
        new_grad, = torch.autograd.grad((mapped_new * weights).sum(), new)
        scale = (1 - C * mapped_new.detach().square().sum(-1, keepdim=True)).square() / 4
        torch.testing.assert_close(old_grad, new_grad * scale, atol=1e-12, rtol=1e-10)
        self.assertGreater(float((new_grad - old_grad).abs().max()), .01)

    def test_final_module_has_identical_values_and_state(self):
        import geoopt
        ball = geoopt.PoincareBall(c=.1).double()
        module = SimpleNamespace(manifold=ball)
        tangent = torch.linspace(.05, .9, 48, dtype=torch.float64).reshape(3,16)
        mu = ball.expmap0(tangent)
        old, new = FinalGeometry(), FinalGeometryNoHook()
        torch.testing.assert_close(old(module, (), mu), new(module, (), mu), atol=0, rtol=0)
        self.assertEqual(old.state_dict(), new.state_dict())

    def test_protocol_diff_is_only_hook_and_explicit_identity(self):
        from ...hier_proxy_scratch_v8_fulltrain.controls import fixed_configuration as baseline
        old = baseline(args())
        new = fixed_configuration(args())
        for key in old:
            if key not in ("canonical_name", "display_name", "HIER_backward_hook", "hook_scope"):
                self.assertEqual(old[key], new[key], key)
        self.assertFalse(new["HIER_backward_hook"])
        self.assertEqual(new["canonical_name"], CANONICAL_NAME)
        self.assertEqual(new["checkpoint_format"], CHECKPOINT_VERSION)
        self.assertEqual(new["P"], 0)
        self.assertEqual(new["proxy_optimizer"], None)
        self.assertEqual(new["radial_gamma"], 500)
        self.assertEqual(new["contrastive_margin"], 2)

    def test_initialization_identity_and_no_parameter_additions(self):
        from ...hier_proxy_scratch_v5.diagnose import Hype_pointMLP, seed_all
        from ...hier_proxy_scratch_v4.train import state_hash
        seed_all(22)
        old = configure_prior(Hype_pointMLP())
        seed_all(22)
        new = configure_backbone(Hype_pointMLP())
        self.assertEqual(state_hash(old), EXPECTED_INITIAL_SHA256)
        self.assertEqual(state_hash(new), EXPECTED_INITIAL_SHA256)
        for name, value in old.state_dict().items():
            torch.testing.assert_close(value, new.state_dict()[name], atol=0, rtol=0)
        self.assertEqual(len(new.emb._forward_hooks), 1)
        with self.assertRaises(RuntimeError):
            configure_backbone(new)

    def test_reused_loss_and_sampler_symbols_are_untouched(self):
        from ...hier_proxy_scratch_v8_fulltrain import controls as runner
        loss, sampler = runner.control_base_losses, runner.ShuffleBatchSampler
        old_format, old_geometry = runner.CHECKPOINT_VERSION, runner.configure_backbone
        with isolated_runner_bindings(runner):
            self.assertIs(runner.control_base_losses, loss)
            self.assertIs(runner.ShuffleBatchSampler, sampler)
            self.assertEqual(runner.CHECKPOINT_VERSION, CHECKPOINT_VERSION)
            self.assertIs(runner.configure_backbone, configure_backbone)
        self.assertEqual(runner.CHECKPOINT_VERSION, old_format)
        self.assertIs(runner.configure_backbone, old_geometry)
        batches = list(sampler(9840,22))
        self.assertEqual(len(batches),307)
        self.assertEqual(len(set(sum(batches,[]))),9824)

    def test_rejects_protocol_drift_and_accepts_train_only_smoke(self):
        for key, value in (("control","B64"),("seed",4780),("workers",8),
                           ("eval_batch_size",16),("epochs",2),("steps_per_epoch",200)):
            with self.assertRaises(ValueError):
                validate_variant(args(**{key:value}))
        self.assertTrue(validate_variant(args(smoke=True,epochs=2,steps_per_epoch=2)).smoke)

    def test_baseline_bindings_are_restored_after_exception(self):
        def identity():
            return args()
        runner = SimpleNamespace(arguments=identity, fixed_configuration=lambda a:{},
                                 configure_backbone=object(), CHECKPOINT_VERSION="old")
        old = dict(vars(runner))
        with self.assertRaisesRegex(RuntimeError,"deliberate"):
            with isolated_runner_bindings(runner):
                self.assertEqual(runner.arguments().control,"B0-32")
                raise RuntimeError("deliberate")
        self.assertEqual(vars(runner),old)


if __name__ == "__main__":
    unittest.main()
