"""Run an approved B13 control behind the historical single-GPU idle gate."""
from __future__ import annotations

from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
PARENT = REPO / "docs/research/diagnostics/b05_cb32_a1_followup_20261026/launch_queue.py"
PARENT_SHA = "b5bc928252722337fe220fb85067b07fa497888d7aaa70e538de464420cee1ee"
GPU_BY_VARIANT = {"h0_200": 0, "h0_300": 1, "a1_200": 2}
sys.path.insert(0, str(HERE))
import runner


def manager_source():
    source = PARENT.read_text(encoding="utf-8")
    if runner.parent.sha_text(source) != PARENT_SHA:
        raise RuntimeError("Pinned TS23 manager changed")
    for before, after in (
        ('choices=("k10p10w01", "k10p20w005")', 'choices=tuple(GPU_BY_VARIANT)'),
        ('choices=(0, 1)', 'choices=tuple(GPU_BY_VARIANT.values())'),
        ('(args.variant, args.gpu) not in (("k10p10w01", 0), ("k10p20w005", 1))',
         'GPU_BY_VARIANT[args.variant] != args.gpu'),
        ('Approved mapping: Ks10/Kp10/lambda.1 GPU0, Ks10/Kp20/lambda.05 GPU1; poll >=10 seconds',
         'Approved mapping: H0-200 GPU0, H0-300 GPU1, A1-200 GPU2; poll >=10 seconds'),
        ('epochs=300, steps_per_epoch=200, world_size=1, global_batch=32,',
         'epochs=300, steps_per_epoch=runner.settings(variant)["steps"], world_size=1, global_batch=32,'),
        ('target_physical_gpu=args.gpu, seed=4780, epochs=300, steps_per_epoch=200,',
         'target_physical_gpu=args.gpu, seed=4780, epochs=300, steps_per_epoch=runner.settings(args.variant)["steps"],'),
        ('epochs, steps = (2, 2) if smoke else (300, 200)',
         'epochs, steps = (2, 2) if smoke else (300, arm["steps"])\n    enabled = arm["hier"] == "on"'),
        ('proxy_K=arm["proxy_K"], T=50,', 'proxy_K=arm["proxy_K"], T=50 if enabled else None,'),
        ('hier_c=.1, hier_view_depth=4.6, hier_enabled=True,',
         'hier_c=.1 if enabled else None, hier_view_depth=4.6 if enabled else None, hier_enabled=enabled,'),
        ('for key in ("net", "optimizer", "proxy", "proxy_optimizer")):',
         'for key in (("net", "optimizer", "proxy", "proxy_optimizer") if enabled else ("net", "optimizer"))):'),
        ('saved["proxy_optimizer"]["state"].values()', '(saved["proxy_optimizer"] or {}).get("state", {}).values()'),
        ('if counts != {epochs * steps}:', 'if counts != ({epochs * steps} if enabled else set()):'),
        ('if smoke:\n        snapshot = torch.load', 'if smoke and enabled:\n        snapshot = torch.load'),
        ('args.variant, "--hier", "on",', 'args.variant, "--hier", runner.settings(args.variant)["hier"],'),
    ):
        source = runner.parent.replace_once(source, before, after)
    return source


def verify_first_forward(directory, variant):
    import torch
    from inter_hierarchy_MN40.hier_proxy_scratch_v8.geometry import depth

    enabled = runner.settings(variant)["hier"] == "on"
    actual = torch.load(directory / "actual_training_epoch_001.pth", map_location="cpu", weights_only=False)
    if len(actual["ids"]) != 32 or actual["rank_order"] != [0] or actual["native_c"] != .1:
        raise RuntimeError("First forward differs from the B13 single-card input contract")
    class_counts = torch.unique(actual["labels"], return_counts=True)[1]
    if class_counts.numel() != 16 or not bool((class_counts == 2).all()):
        raise RuntimeError("First forward is not 16 classes x2")
    if any(bool((depth(actual[key]) > 4.60005).any()) for key in ("native_mu", "native_nu")):
        raise RuntimeError("B13 native cap changed")
    if actual["hier_view_present"] is not enabled or actual["proxy_parameter_count"] != (512 * 256 if enabled else 0):
        raise RuntimeError("First forward constructed the wrong hierarchy state")
    initial = torch.load(directory / "initialization_e0.pth", map_location="cpu", weights_only=False)
    if enabled:
        generator = torch.Generator(device="cpu").manual_seed(4780)
        expected = torch.randn(512, 256, generator=generator) / 16 * 2.3 * .9
        if not torch.equal(initial["proxy"]["tangent_proxies"], expected):
            raise RuntimeError("A1 initial proxies differ from the historical B13 draw")
        torch.testing.assert_close(depth(actual["hier_mu"]), torch.full((32,), 4.6), atol=5e-5, rtol=0.)
    elif initial["proxy"] is not None or any(k in actual for k in ("hier_mu", "proxy_ball", "proxy_tangent")):
        raise RuntimeError("H0 has hierarchy data")


def main():
    scope = {"__name__": "b13_controls_manager", "__file__": str(PARENT), "GPU_BY_VARIANT": GPU_BY_VARIANT}
    exec(compile(manager_source(), str(PARENT) + "[B13 controls]", "exec"), scope)
    scope.update(runner=runner, RELATIVE=HERE.relative_to(REPO))
    accept_parent = scope["accept_run"]

    def accept_run(directory, variant, initial_hash, smoke):
        accepted = accept_parent(directory, variant, initial_hash, smoke)
        verify_first_forward(directory, variant)
        return accepted

    scope["accept_run"] = accept_run
    scope["main"]()


if __name__ == "__main__":
    main()
