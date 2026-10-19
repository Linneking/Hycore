"""Versioned V8 FULL9840 B0-32 ablation: remove only InverseMetric backward.

The completed FULL controls runner is reused, with temporary process-local
bindings. Existing production files, checkpoint formats, and run directories
are never changed. All forward geometry, losses, seeds, samplers and logging
remain the V8 B0-32 protocol.
"""
from __future__ import annotations

from contextlib import contextmanager
import torch
from torch import nn

from ..hier_proxy_scratch_v8.geometry import C, freeze_curvature, to_ball

CHECKPOINT_VERSION = "hycore-v8-b0-32-full9840-no-inversemetric-20261020-1"
CANONICAL_NAME = "V8-B0-32-C01-NOHOOK-FULL9840-TESTSEL"
EXPECTED_INITIAL_SHA256 = "f94e52202df5aaae21f21b8b9c6beaa3925424c4e04a6dcc7d2b84b89af04b73"


class FinalGeometryNoHook(nn.Module):
    """The same final logmap, cap and expmap; ordinary autograd backward."""
    def forward(self, module, inputs, mu):
        tangent = module.manifold.logmap0(mu)
        return to_ball(tangent, hook=False)


def configure_backbone(backbone):
    """Install exactly the V8 geometry except for InverseMetric.apply."""
    if hasattr(backbone, "v8_geometry"):
        raise RuntimeError("Final geometry must be installed exactly once")
    backbone.manifold = type(backbone.manifold)(dim=512, c=C)
    backbone.manifold2 = type(backbone.manifold2)(dim=256, c=C)
    backbone.emb.manifold = backbone.manifold
    backbone.classifier.manifold = backbone.manifold2
    backbone.v8_geometry = FinalGeometryNoHook()
    backbone.emb.register_forward_hook(backbone.v8_geometry)
    freeze_curvature(backbone)
    return backbone


def validate_variant(args):
    if args.control != "B0-32":
        raise ValueError("This version is exclusively V8 FULL9840 B0-32")
    fixed = {"seed": 22, "workers": 2, "eval_batch_size": 32,
             "train_eval_every": 10, "save_every": 20,
             "warmup_epochs": 0, "lambda_hier": 0.}
    if any(getattr(args, key) != value for key, value in fixed.items()):
        raise ValueError("Only InverseMetric may change; seed/workers/eval/save/loss protocol is fixed")
    if not args.smoke and (args.epochs, args.steps_per_epoch) != (300, 307):
        raise ValueError("Production must complete300 epochs x307 shuffle32 steps")
    return args


def fixed_configuration(args, baseline=None):
    validate_variant(args)
    if baseline is None:
        from ..hier_proxy_scratch_v8_fulltrain.controls import fixed_configuration as baseline
    config = baseline(args)
    config.update(canonical_name=CANONICAL_NAME, display_name=CANONICAL_NAME,
                  variant="V8 B0-32; only InverseMetric backward disabled",
                  checkpoint_format=CHECKPOINT_VERSION,
                  HIER_backward_hook=False,
                  hook_scope="disabled for whole and part; cap/expmap/logmap unchanged",
                  expected_initialized_model_sha256=EXPECTED_INITIAL_SHA256,
                  reused_runner="hier_proxy_scratch_v8_fulltrain.controls",
                  changed_from_v8_full_b0=["InverseMetric backward disabled"])
    return config


@contextmanager
def isolated_runner_bindings(runner):
    """Scope changed bindings to this new entrypoint and restore on failure."""
    names = ("arguments", "fixed_configuration", "configure_backbone", "CHECKPOINT_VERSION")
    original = {name: getattr(runner, name) for name in names}
    runner.arguments = lambda: validate_variant(original["arguments"]())
    runner.fixed_configuration = lambda args: fixed_configuration(args, original["fixed_configuration"])
    runner.configure_backbone = configure_backbone
    runner.CHECKPOINT_VERSION = CHECKPOINT_VERSION
    try:
        yield runner
    finally:
        for name, value in original.items():
            setattr(runner, name, value)


def main():
    from ..hier_proxy_scratch_v8_fulltrain import controls as runner
    with isolated_runner_bindings(runner):
        runner.main()


if __name__ == "__main__":
    main()
