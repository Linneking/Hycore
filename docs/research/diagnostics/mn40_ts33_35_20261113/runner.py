"""TS33/TS34: change only the proxy recipe or HIER weight from TS32."""
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent / "b05_singlecard_a0_a1_20261025" / "runner.py"
PARENT_SHA = "501038857131c775b4f1701aae94606619884d9ce11528666062ab258f16ff20"
VERSION = "hycore-mn40-ts33-ts34-v1"
ARMS = {"ts33": ("a0", .5), "ts34": ("a1", 1.)}

if hashlib.sha256(PARENT.read_text(encoding="utf-8").encode()).hexdigest() != PARENT_SHA:
    raise RuntimeError("Pinned B13 training parent changed")
spec = importlib.util.spec_from_file_location("ts33_ts34_parent", PARENT)
parent = importlib.util.module_from_spec(spec)
spec.loader.exec_module(parent)
INITIAL_DIGEST = parent.INITIAL_DIGEST


def settings(variant):
    return dict(sample_K=20, proxy_K=20, lambda_hier=ARMS[variant][1])


def recipe(variant):
    return parent.recipe(ARMS[variant][0])


def transformed_source(raw, variant):
    optimizer, weight = ARMS[variant]
    source = parent.transformed_source(raw, optimizer)
    for before, after in (
        ('choices=("a0", "a1")', f'choices=({variant!r},)'),
        ('whole_count, child_count, topk=10, seed=', 'whole_count, child_count, topk=20, seed='),
        ('proxy(hier_mu, values["gold"], topk=10, proxy_topk=20, seed=',
         'proxy(hier_mu, values["gold"], topk=20, proxy_topk=20, seed='),
        ('lambda_hier=.5)', f'lambda_hier={weight!r})'),
        ('values["base"] + .5 * hier_loss', f'values["base"] + {weight!r} * hier_loss'),
        ('"lambda_hier": .5 if enabled else 0., "intra_contrastive"',
         f'"lambda_hier": {weight!r} if enabled else 0., "intra_contrastive"'),
        ('"base_only", "lambda_hier": .5 if enabled else 0.,',
         f'"base_only", "lambda_hier": {weight!r} if enabled else 0.,'),
    ):
        source = parent.replace_once(source, before, after)
    return source


def configuration(args, original):
    optimizer, weight = ARMS[args.variant]
    template = copy.copy(args)
    template.variant = optimizer
    config = parent.successor_configuration(template, original)
    name = (f"HC-C01-Cap23-GS1-M500_2-CB32x200 / "
            f"Hshell-{optimizer.upper()}-Ks20-Kp20-L{weight:.1f}-W0 (S4780)")
    config.update(variant=args.variant, canonical_name=name, display_name=name,
                  implementation_variant=HERE.name, sample_K=20,
                  lambda_hier_after_warmup=weight, baseline="B13",
                  comparison_scope="TS32 with proxy recipe A0" if args.variant == "ts33"
                  else "TS32 with HIER coefficient 1.0")
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
        raise RuntimeError("Incorrect TS33/TS34 checkpoint version")
    actual = parent.verify_optimizer_payload(dict(saved, format=parent.VERSION),
                                             epochs, schedule_epochs, ARMS[variant][0])
    return dict(actual, arm=variant)


def build_namespace(variant):
    namespace = parent.build_namespace(ARMS[variant][0])
    source = transformed_source(parent.ORIGINAL.read_text(encoding="utf-8"), variant)
    definitions = [node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef)
                   and node.name in ("main", "arguments")]
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(parent.ORIGINAL), "exec"), namespace)
    original_arguments = namespace["arguments"]

    def fresh_arguments(argv=None):
        args = original_arguments(argv)
        if args.resume is not None:
            raise RuntimeError("TS33/TS34 require a fresh start")
        return args

    namespace.update(arguments=fresh_arguments, CHECKPOINT_VERSION=VERSION,
                     fixed_configuration=lambda args: configuration(args, namespace["parent_configuration"](args)))
    return namespace


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--variant", choices=ARMS, required=True)
    known, _ = parser.parse_known_args()
    build_namespace(known.variant)["main"]()
