"""Fresh B05 CB32 A1: independent proxy-K and HIER-weight ablations.

Historical training files are immutable. This successor pins the reviewed
continuous-stream parent and makes the single-rank/CB32/K10+20 changes explicit.
"""
from __future__ import annotations

import ast
import hashlib
import math
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
ORIGINAL = REPO / "inter_hierarchy_MN40/hier_proxy_cycle_v9c/train.py"
EXPECTED_SOURCE = "4754913a64bb355080e55853e9a502d211d6153fbb6a23cd0c182dad437c7e56"
AUDIT_SOURCE = REPO / "inter_hierarchy_MN40/hier_proxy_cycle_v9c/parameter_audit.py"
EXPECTED_AUDIT_SOURCE = "a7473905e3183926705ad7789a5c773b8dbfc658441800797347e719c5f4cecc"
VERSION = "hycore-b05-cb32-a1-k-lambda-testselect-1"
INITIAL_DIGEST = "8926765a4842652b751873fb89aa37d24d9b0f4f2d831d853b9cc0c1cd46472d"
RECIPES = {"k5p10": {"lr": .005, "eps": 1e-4}, "w01": {"lr": .005, "eps": 1e-4}}


def settings(variant):
    recipe(variant)
    return {"sample_K": 5 if variant == "k5p10" else 10,
            "proxy_K": 10 if variant == "k5p10" else 20,
            "lambda_hier": .1 if variant == "w01" else .5}


def sha_text(source):
    return hashlib.sha256(source.replace("\r\n", "\n").encode("utf-8")).hexdigest()


def recipe(variant):
    if variant not in RECIPES:
        raise ValueError("Expected approved ablation k5p10 or w01")
    return dict(RECIPES[variant])


def replace_once(source, before, after):
    if source.count(before) != 1:
        raise RuntimeError("Successor boundary must occur exactly once: " + before)
    return source.replace(before, after, 1)


def transformed_source(source, variant="k5p10"):
    selected = recipe(variant)
    arm = settings(variant)
    source = source.replace("\r\n", "\n")
    if sha_text(source) != EXPECTED_SOURCE:
        raise RuntimeError("Reviewed continuous-stream source SHA differs")
    substitutions = (
        ('CHECKPOINT_VERSION = "hycore-cycle-v9c-c01-cap23-gs1-m5002-testselect-1"', f'CHECKPOINT_VERSION = "{VERSION}"'),
        ('parser.add_argument("--hier", choices=("off", "on"), required=True)',
         'parser.add_argument("--variant", choices=("k5p10", "w01"), required=True)\n    parser.add_argument("--hier", choices=("on",), required=True)'),
        ('parser.add_argument("--steps-per-epoch", type=int, default=300)', 'parser.add_argument("--steps-per-epoch", type=int, default=200)'),
        ('!= (300, 300, 8, 16)', '!= (300, 200, 8, 16)'),
        ('Production requires300x300, workers8, eval16', 'Production requires300x200, workers8, eval16'),
        ('def fixed_configuration(args):', 'def parent_configuration(args):'),
        ('from .sampler import ClassCycleBatchSampler', 'from ..hier_proxy_scratch_v5.sampler import SourceClassBatchSampler'),
        ('ClassCycleBatchSampler(labels[train_ids], rank=rank,', 'SourceClassBatchSampler(labels[train_ids], world_size=1, classes_per_rank=16, instances_per_class=2, rank=rank,'),
        ('if int(os.environ["WORLD_SIZE"]) != 2:', 'if int(os.environ["WORLD_SIZE"]) != 1:'),
        ('Cycle-B64 requires exactly two ranks', 'Single-card B05 requires exactly one rank'),
        ('len(saved["rank_states"]) != 2', 'len(saved["rank_states"]) != 1'),
        ('devices = [None, None]', 'devices = [None]'),
        ('rank_peaks = [None, None]', 'rank_peaks = [None]'),
        ('rank_states = [None, None]', 'rank_states = [None]'),
        ('"rank_order": [0, 1]', '"rank_order": [0]'),
        ('whole_count, child_count, topk=20, seed=', f'whole_count, child_count, topk={arm["sample_K"]}, seed='),
        ('proxy(hier_mu, values["gold"], topk=20, seed=', f'proxy(hier_mu, values["gold"], topk={arm["sample_K"]}, proxy_topk={arm["proxy_K"]}, seed='),
        ('proxy_optimizer = torch.optim.AdamW(proxy.parameters(), lr=.01, betas=(.9, .999), eps=1e-8, weight_decay=.01)',
         f'proxy_optimizer = torch.optim.AdamW(proxy.parameters(), lr={selected["lr"]!r}, betas=(.9, .999), eps={selected["eps"]!r}, weight_decay=.01)'),
        ('"V9C_CYCLE_B64_" + args.hier.upper() + "_COMPLETED"', '"B05_SINGLECARD_" + args.variant.upper() + "_COMPLETED"'),
    )
    for before, after in substitutions:
        source = replace_once(source, before, after)
    if variant == "w01":
        for before, after in (
            ('"lambda_hier_after_warmup": .5 if on else 0.', '"lambda_hier_after_warmup": .1 if on else 0.'),
            ('lambda_hier=.5)', 'lambda_hier=.1)'),
            ('loss = values["base"] + .5 * hier_loss', 'loss = values["base"] + .1 * hier_loss'),
        ):
            source = replace_once(source, before, after)
        if source.count('"lambda_hier": .5 if enabled else 0.') != 2:
            raise RuntimeError("Expected exactly two HIER weight telemetry sites")
        source = source.replace('"lambda_hier": .5 if enabled else 0.', '"lambda_hier": .1 if enabled else 0.')
    ast.parse(source)
    return source


def transformed_audit_source(source, variant="k5p10"):
    arm = settings(variant)
    source = source.replace("\r\n", "\n")
    if sha_text(source) != EXPECTED_AUDIT_SOURCE:
        raise RuntimeError("Reviewed gradient-audit source SHA differs")
    source = replace_once(source, 'values["gold"], topk=topk, seed=seed,',
                          f'values["gold"], topk=topk, proxy_topk={arm["proxy_K"]}, seed=seed,')
    source = replace_once(source, '"lambda_hier": lambda_hier, "topk": topk,',
                          f'"lambda_hier": lambda_hier, "topk": topk, "sample_K": topk, "proxy_K": {arm["proxy_K"]},')
    ast.parse(source)
    return source


def successor_configuration(args, original):
    selected = recipe(args.variant)
    arm = settings(args.variant)
    name = ("B05-C0.1-CAP2.3-GSbase1-M500_2-CB32x200-HIERshell4.6-C0.1-"
            f"GShier1-Ks{arm['sample_K']}-Kp{arm['proxy_K']}-L{arm['lambda_hier']}-W0-A1-{args.variant.upper()}-LR{selected['lr']}-EPS{selected['eps']}-TS-S4780")
    config = dict(original)
    config.update(
        arm="b05", variant=args.variant, canonical_name=name, display_name=name,
        implementation_variant="b05_cb32_a1_ablation_20261025",
        world_size=1, global_batch=32, classes_per_rank=16, instances_per_class=2,
        sample_K=arm["sample_K"], proxy_K=arm["proxy_K"], T=50,
        lambda_hier_after_warmup=arm["lambda_hier"],
        sampler="uniform 16 distinct classes; 2 instances/class with replacement; SourceClassBatchSampler v5",
        sampler_epoch_reset=True,
        sampler_state_semantics="stateless deterministic seed/epoch/step; no cyclic cursors or training RNG reseeding",
        BN="ordinary local32; part then whole; single-rank DDP; no SyncBN",
        proxy_optimizer=dict(original["proxy_optimizer"], **selected),
        ce_temperature=1., source_checkpoint=None, teacher=None,
        random_streams="continuous main RNG; deterministic class plans; persistent private loader/crop/Gumbel",
        comparison_scope="relative to 10/25 B05 CB32 A1: k5p10 halves sample/proxy K10/20 to5/10; w01 changes only HIER weight .5 to .1",
        budget="60000 updates; 1920000 sample presentations; half CB64 presentations",
        training_source={
            "original_relative_path": ORIGINAL.relative_to(REPO).as_posix(),
            "original_normalized_sha256": EXPECTED_SOURCE,
            "audit_normalized_sha256": EXPECTED_AUDIT_SOURCE,
            "successor_relative_path": HERE.relative_to(REPO).as_posix(),
            "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "transformed_normalized_sha256": sha_text(transformed_source(ORIGINAL.read_text(encoding="utf-8"), args.variant)),
        })
    return config


def verify_optimizer_payload(checkpoint, completed_epochs, schedule_epochs, variant="k5p10"):
    selected = recipe(variant)
    group, = checkpoint["proxy_optimizer"]["param_groups"]
    scheduler = checkpoint["proxy_scheduler"]
    expected_lr = .0005 + (selected["lr"] - .0005) * (1 + math.cos(math.pi * completed_epochs / schedule_epochs)) / 2
    if not (checkpoint["format"] == VERSION and checkpoint["completed_epochs"] == completed_epochs
            and group["initial_lr"] == selected["lr"] and group["eps"] == selected["eps"]
            and tuple(group["betas"]) == (.9, .999) and group["weight_decay"] == .01
            and math.isclose(group["lr"], expected_lr, rel_tol=1e-12, abs_tol=1e-15)
            and scheduler["T_max"] == schedule_epochs and scheduler["eta_min"] == .0005
            and scheduler["base_lrs"] == [selected["lr"]] and scheduler["last_epoch"] == completed_epochs):
        raise RuntimeError("Saved proxy optimizer/schedule violates approved " + variant.upper())
    model_group, = checkpoint["optimizer"]["param_groups"]
    model_scheduler = checkpoint["scheduler"]
    if not (model_group["initial_lr"] == .1 and model_group["momentum"] == .9
            and model_group["weight_decay"] == 2e-4 and model_scheduler["T_max"] == schedule_epochs
            and model_scheduler["eta_min"] == .005):
        raise RuntimeError("Saved RSGD/schedule differs from B05")
    return {"variant": variant, "initial_lr": group["initial_lr"], "eps": group["eps"],
            "lr": group["lr"], "T_max": schedule_epochs, "completed_epochs": completed_epochs}


def build_namespace(variant="k5p10", source=None):
    recipe(variant)
    for path in (REPO, REPO / "inter_hierarchy_MN40"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    source = ORIGINAL.read_text(encoding="utf-8") if source is None else source
    namespace = {"__name__": "b05_singlecard_successor", "__file__": str(ORIGINAL),
                 "__package__": "inter_hierarchy_MN40.hier_proxy_cycle_v9c"}
    exec(compile(transformed_source(source, variant), str(ORIGINAL) + "[singlecard]", "exec"), namespace)
    from inter_hierarchy_MN40.hier_proxy_cycle_v9c import base_protocol

    def base_losses(outputs, labels):
        return base_protocol.global_base_losses(outputs, labels, expected_global_batch=32)

    audit_namespace = {"__name__": "b05_singlecard_audit", "__file__": str(AUDIT_SOURCE),
                       "__package__": "inter_hierarchy_MN40.hier_proxy_cycle_v9c"}
    exec(compile(transformed_audit_source(AUDIT_SOURCE.read_text(encoding="utf-8"), variant),
                 str(AUDIT_SOURCE) + "[singlecard]", "exec"), audit_namespace)
    audit_namespace["global_base_losses"] = base_losses
    original_arguments = namespace["arguments"]

    def fresh_arguments(argv=None):
        args = original_arguments(argv)
        if args.variant != variant or args.resume is not None:
            raise RuntimeError("Variant must match runner; each arm starts fresh without resume")
        return args

    namespace.update(
        arguments=fresh_arguments,
        fixed_configuration=lambda args: successor_configuration(args, namespace["parent_configuration"](args)),
        global_base_losses=base_losses,
        audit_parameter_gradients=audit_namespace["audit_parameter_gradients"],
        EXPECTED_ORIGINAL_INITIAL_DIGEST=INITIAL_DIGEST,
    )
    return namespace


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--variant", choices=("k5p10", "w01"), required=True)
    known, _ = parser.parse_known_args()
    build_namespace(known.variant)["main"]()
