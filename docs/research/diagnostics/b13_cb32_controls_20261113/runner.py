"""B13 controls using the pinned TS19/TS20 single-card training loop."""
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import importlib.util
import math
from pathlib import Path


HERE = Path(__file__).resolve().parent
PARENT = HERE.parent / "b05_singlecard_a0_a1_20261025" / "runner.py"
PARENT_SHA = "501038857131c775b4f1701aae94606619884d9ce11528666062ab258f16ff20"
VERSION = "hycore-b13-cb32-controls-20261113-v1"
STEPS = {"h0_200": 200, "h0_300": 300, "a1_200": 200}

if hashlib.sha256(PARENT.read_text(encoding="utf-8").encode()).hexdigest() != PARENT_SHA:
    raise RuntimeError("Pinned TS19/TS20 runner changed")
spec = importlib.util.spec_from_file_location("b13_control_parent", PARENT)
parent = importlib.util.module_from_spec(spec)
spec.loader.exec_module(parent)
INITIAL_DIGEST = parent.INITIAL_DIGEST


def settings(variant):
    enabled = variant == "a1_200"
    return dict(steps=STEPS[variant], hier="on" if enabled else "off",
                sample_K=20 if enabled else None, proxy_K=20 if enabled else None,
                lambda_hier=.5 if enabled else 0.)


def recipe(variant):
    return parent.recipe("a1") if settings(variant)["hier"] == "on" else None


def transformed_source(raw, variant):
    arm = settings(variant)
    source = parent.transformed_source(raw, "a1")
    for before, after in (
        (f'CHECKPOINT_VERSION = "{parent.VERSION}"', f'CHECKPOINT_VERSION = "{VERSION}"'),
        ('choices=("a0", "a1")', f'choices=({variant!r},)'),
        ('choices=("on",)', f'choices=({arm["hier"]!r},)'),
        ('--steps-per-epoch", type=int, default=200', f'--steps-per-epoch", type=int, default={arm["steps"]}'),
        ('!= (300, 200, 8, 16)', f'!= (300, {arm["steps"]}, 8, 16)'),
        ('Production requires300x200', f'Production requires300x{arm["steps"]}'),
        ('whole_count, child_count, topk=10, seed=', 'whole_count, child_count, topk=20, seed='),
        ('proxy(hier_mu, values["gold"], topk=10, proxy_topk=20, seed=',
         'proxy(hier_mu, values["gold"], topk=20, proxy_topk=20, seed='),
    ):
        source = parent.replace_once(source, before, after)
    return source


def configuration(args, original):
    arm = settings(args.variant)
    template_args = copy.copy(args)
    template_args.variant = "a1"
    config = parent.successor_configuration(
        template_args, dict(original, proxy_optimizer=original["proxy_optimizer"] or {}))
    if arm["hier"] == "off":
        for key in ("sample_K", "proxy_K", "T", "proxy_optimizer"):
            config[key] = original[key]
        config["random_streams"] = "continuous main RNG; deterministic class plans; persistent private loader/crop"
    else:
        config["sample_K"] = arm["sample_K"]
    component = "Hshell-A1-Ks20-Kp20-L0.5-W0" if arm["hier"] == "on" else "H0"
    name = f"HC-C01-Cap23-GS1-M500_2-CB32x{arm['steps']} / {component} (S4780)"
    config.update(variant=args.variant, canonical_name=name, display_name=name,
                  implementation_variant=HERE.name,
                  comparison_scope="B13 CB32 HIER-off or fixed-shell A1 K20/K20",
                  budget=f"{300 * arm['steps']} updates; 300-epoch cosine")
    config["training_source"].update(
        successor_relative_path=HERE.relative_to(parent.REPO).as_posix(),
        parent_runner_normalized_sha256=PARENT_SHA,
        runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        transformed_normalized_sha256=parent.sha_text(
            transformed_source(parent.ORIGINAL.read_text(encoding="utf-8"), args.variant)),
    )
    return config


def verify_optimizer_payload(saved, epochs, schedule_epochs, variant):
    if saved["format"] != VERSION:
        raise RuntimeError("Checkpoint version differs from B13 control")
    if settings(variant)["hier"] == "on":
        actual = parent.verify_optimizer_payload(
            dict(saved, format=parent.VERSION), epochs, schedule_epochs, "a1")
        return dict(actual, variant=variant)
    if (any(saved.get(key) is not None for key in ("proxy", "proxy_optimizer", "proxy_scheduler", "best_proxy"))
            or saved["rank_states"][0]["private_generators"]["gumbel"] is not None):
        raise RuntimeError("H0 allocated a HIER training state")
    group, = saved["optimizer"]["param_groups"]
    schedule = saved["scheduler"]
    if not (saved["completed_epochs"] == epochs and group["initial_lr"] == .1
            and group["momentum"] == .9 and group["weight_decay"] == 2e-4
            and schedule["T_max"] == schedule_epochs and schedule["last_epoch"] == epochs
            and schedule["eta_min"] == .005 and math.isclose(group["lr"], .005, abs_tol=1e-12)):
        raise RuntimeError("H0 changed the common RSGD schedule")
    return None


def build_namespace(variant):
    namespace = parent.build_namespace("a1")
    tree = ast.parse(transformed_source(parent.ORIGINAL.read_text(encoding="utf-8"), variant))
    definitions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                   and node.name in ("main", "arguments")]
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(parent.ORIGINAL), "exec"), namespace)
    original_arguments = namespace["arguments"]

    def fresh_arguments(argv=None):
        args = original_arguments(argv)
        if args.resume is not None:
            raise RuntimeError("B13 controls require a fresh start")
        return args

    namespace.update(arguments=fresh_arguments, CHECKPOINT_VERSION=VERSION,
                     fixed_configuration=lambda args: configuration(args, namespace["parent_configuration"](args)))
    return namespace


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--variant", choices=STEPS, required=True)
    known, _ = parser.parse_known_args()
    build_namespace(known.variant)["main"]()
