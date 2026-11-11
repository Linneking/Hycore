"""Three fresh, single-variable ablations of the pinned TS23 training recipe."""
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import importlib.util
import math
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
PARENT = HERE.parent / "b05_cb32_a1_followup_20261026" / "runner.py"
PARENT_SHA = "b4c9b79d48890a6d0055f3efdffbe04f998c13bedcfd2f0c432e36e198ef367b"
PARENT_VARIANT = "k10p10w01"
LOSS_SOURCE = REPO / "inter_hierarchy_MN40/hier_proxy_scratch_v8/hier_loss.py"
LOSS_SHA = "e77ff498d1cd056a30c7932dfcf367379755711a4aefc7f70b1d8a88dbed651d"
VERSION = "hycore-ts23-proxy-ablations-full300-1"
ARMS = {
    "p256": (256, .9, 2.3, "proxy count 512 to 256"),
    "init05": (512, .5, 2.3, "proxy Gaussian initialization coefficient .9 to .5"),
    "cap29": (512, .9, 2.9, "proxy forward tangent cap 2.3 to 2.9"),
}


def sha_text(source):
    return hashlib.sha256(source.replace("\r\n", "\n").encode("utf-8")).hexdigest()


if sha_text(PARENT.read_text(encoding="utf-8")) != PARENT_SHA:
    raise RuntimeError("Pinned TS23 runner changed")
spec = importlib.util.spec_from_file_location("ts23_ablation_parent", PARENT)
parent = importlib.util.module_from_spec(spec)
spec.loader.exec_module(parent)
ORIGINAL = parent.ORIGINAL
INITIAL_DIGEST = parent.INITIAL_DIGEST


def settings(variant):
    if variant not in ARMS:
        raise ValueError("Expected p256, init05 or cap29")
    count, coefficient, cap, _ = ARMS[variant]
    return dict(parent.settings(PARENT_VARIANT), P=count,
                proxy_initialization_coefficient=coefficient, proxy_tangent_cap=cap)


def recipe(variant):
    return dict(parent.recipe(PARENT_VARIANT), **settings(variant))


def initial_proxy_tensor(variant, seed=4780):
    import torch

    arm = settings(variant)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    # Draw the exact TS23 tensor before taking a prefix so shape-dependent RNG
    # implementations cannot change the retained rows.
    draws = torch.randn(512, 256, generator=generator)[:arm["P"]]
    return draws / math.sqrt(256) * 2.3 * arm["proxy_initialization_coefficient"]


def transformed_source(raw, variant):
    settings(variant)
    source = parent.transformed_source(raw, PARENT_VARIANT)
    for before, after in (
        (f'CHECKPOINT_VERSION = "{parent.VERSION}"', f'CHECKPOINT_VERSION = "{VERSION}"'),
        ('choices=("k10p10w01", "k10p20w005")', f'choices={tuple(ARMS)!r}'),
        ('from ..hier_proxy_scratch_v8.hier_loss import HIERLoss\n'
         '            proxy = HIERLoss(seed=args.seed).to(device)',
         'proxy = ABLATION_HIER_CLASS(seed=args.seed).to(device)'),
        ('"HIER_view_and_proxies": "c.1 wholedepth4.6 GS once on separate fork" if enabled else "absent"',
         '"HIER_view_and_proxies": "c.1 wholedepth4.6 GS once on separate fork" if enabled else "absent",\n'
         '            "proxy_forward_tangent_cap": config["proxy_tangent_cap"],\n'
         '            "proxy_forward_depth_cap": config["proxy_depth_cap"]'),
    ):
        source = parent.replace_once(source, before, after)
    old_count = '512 * 256 if enabled else 0'
    if source.count(old_count) != 2:
        raise RuntimeError("Expected two TS23 proxy parameter counts")
    return source.replace(old_count, 'proxy.tangent_proxies.numel() if enabled else 0')


def successor_configuration(args, original):
    arm = settings(args.variant)
    legacy_args = copy.copy(args)
    legacy_args.variant = PARENT_VARIANT
    config = parent.successor_configuration(legacy_args, original)
    name = f"HC-C01-Cap23-GS1-M500_2-CB32x200 / Hshell-A1-Ks10-Kp10-L0.1-W0 / TS23-{args.variant.upper()} (S4780)"
    config.update(
        variant=args.variant, canonical_name=name, display_name=name,
        implementation_variant=HERE.name, run_alias=f"TS23-{args.variant.upper()}",
        reference_alias="TS23", **arm,
        proxy_depth_cap=2 * arm["proxy_tangent_cap"],
        proxy_initialization_tangent_scale=2.3,
        proxy_initialization=(
            f'private seeded TS23 Gaussian prefix N(0,I)/sqrt(D)*2.3*'
            f'{arm["proxy_initialization_coefficient"]}; no per-row normalization'),
        comparison_scope="TS23/B13 full300x200; only " + ARMS[args.variant][3],
    )
    config["training_source"].update(
        parent_runner_normalized_sha256=PARENT_SHA,
        hierarchy_loss_normalized_sha256=LOSS_SHA,
        successor_relative_path=HERE.relative_to(REPO).as_posix(),
        runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        transformed_normalized_sha256=sha_text(
            transformed_source(ORIGINAL.read_text(encoding="utf-8"), args.variant)),
    )
    return config


def verify_optimizer_payload(saved, epochs, schedule_epochs, variant):
    settings(variant)
    if saved["format"] != VERSION:
        raise ValueError("Not a TS23 proxy ablation checkpoint")
    report = parent.verify_optimizer_payload(
        dict(saved, format=parent.VERSION), epochs, schedule_epochs, PARENT_VARIANT)
    report["variant"] = variant
    return report


def proxy_class(variant):
    from inter_hierarchy_MN40.hier_proxy_scratch_v8 import geometry, hier_loss

    arm = settings(variant)
    source = LOSS_SOURCE.read_text(encoding="utf-8")
    if sha_text(source) != LOSS_SHA:
        raise RuntimeError("Pinned TS23 HIER loss changed")
    for before, after in (
        ('def __init__(self, num_proxies=512, dim=256, seed=22):', 'def __init__(self, seed=22):'),
        ('self.num_proxies, self.dim, self.c = num_proxies, dim, C',
         f'self.num_proxies, self.dim, self.c = {arm["P"]}, 256, C'),
        ('generator = torch.Generator(device="cpu").manual_seed(seed)\n'
         '        values = torch.randn(num_proxies, dim, generator=generator) / math.sqrt(dim) * CLIP_R * .9',
         'values = INITIAL_PROXY_TENSOR(seed)'),
    ):
        source = parent.replace_once(source, before, after)

    if variant == "cap29":
        for before, after in (
            ('self.margin, self.tau, self.clip_r = .1, .1, CLIP_R',
             'self.margin, self.tau, self.clip_r = .1, .1, 2.9'),
            ('return to_ball(self.tangent_proxies)',
             'return to_ball(self.tangent_proxies, clip_r=self.clip_r)'),
            ('assert_output_geometry(self.proxies())',
             'assert_proxy_geometry(self.proxies(), self.clip_r)'),
            ('norms > CLIP_R', 'norms > self.clip_r'),
            ('geometry_stats(proxies.detach())',
             'proxy_geometry_stats(proxies.detach(), self.clip_r)'),
        ):
            source = parent.replace_once(source, before, after)

    def assert_proxy_geometry(points, tangent_cap):
        if not bool(points.isfinite().all()) or not bool((geometry.depth(points) <= 2 * tangent_cap + 5e-5).all()):
            raise RuntimeError("Proxy output exceeds its finite depth cap")

    def proxy_geometry_stats(points, tangent_cap):
        report = geometry.geometry_stats(points)
        report["cap_fraction"] = float((geometry.depth(points) >= 2 * tangent_cap - 1e-4).float().mean())
        return report

    tree = ast.parse(source)
    definition, = (node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "HIERLoss")
    namespace = dict(vars(hier_loss), INITIAL_PROXY_TENSOR=lambda seed: initial_proxy_tensor(variant, seed),
                     assert_proxy_geometry=assert_proxy_geometry, proxy_geometry_stats=proxy_geometry_stats)
    exec(compile(ast.Module(body=[definition], type_ignores=[]), str(LOSS_SOURCE), "exec"), namespace)
    return namespace["HIERLoss"]


def require_fresh_args(args, variant):
    if args.variant != variant or args.resume is not None:
        raise RuntimeError("Variant must match runner; each arm starts fresh without resume")
    return args


def cap29_proxy_update_stats(before, after):
    import torch
    from inter_hierarchy_MN40.hier_proxy_scratch_v6.structure_monitor import proxy_update_stats
    from inter_hierarchy_MN40.hier_proxy_scratch_v8.geometry import C, to_ball

    before, after = before.detach(), after.detach()
    report = proxy_update_stats(before, after)
    sqrt_c = math.sqrt(C)
    report["raw_ball_radius_change_mean"] = float(
        (torch.tanh(sqrt_c * after.norm(dim=-1)) - torch.tanh(sqrt_c * before.norm(dim=-1))).mean() / sqrt_c)
    report["mapped_ball_radius_change_mean"] = float(
        (to_ball(after, clip_r=2.9, hook=False).norm(dim=-1)
         - to_ball(before, clip_r=2.9, hook=False).norm(dim=-1)).mean())
    report["semantics"] = "Actual AdamW tangent displacement; mapped radius includes forward cap 2.9, c=.1"
    return report


def build_namespace(variant, source=None):
    settings(variant)
    raw = ORIGINAL.read_text(encoding="utf-8") if source is None else source
    namespace = parent.build_namespace(PARENT_VARIANT, raw)
    tree = ast.parse(transformed_source(raw, variant))
    definitions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                   and node.name in ("main", "arguments")]
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(ORIGINAL), "exec"), namespace)
    original_arguments = namespace["arguments"]

    def fresh_arguments(argv=None):
        return require_fresh_args(original_arguments(argv), variant)

    namespace.update(
        arguments=fresh_arguments, CHECKPOINT_VERSION=VERSION,
        ABLATION_HIER_CLASS=proxy_class(variant),
        fixed_configuration=lambda args: successor_configuration(args, namespace["parent_configuration"](args)),
    )
    if variant == "cap29":
        namespace["proxy_update_stats"] = cap29_proxy_update_stats
    return namespace


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--variant", choices=ARMS, required=True)
    known, _ = parser.parse_known_args()
    build_namespace(known.variant)["main"]()
