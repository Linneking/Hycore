"""Fresh CB32 A0 early rerun: observe ten epochs of the original 300-epoch recipe.

The historical single-card source and its cycle parent are immutable and pinned.
Formal evaluation, clean-eval cadence, mining, loss and all random streams remain
the original A0 protocol. Only the loop stop and checkpoint retention change.
"""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import math
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
PARENT = REPO / "docs/research/diagnostics/b05_singlecard_a0_a1_20261025/runner.py"
EXPECTED_PARENT = "501038857131c775b4f1701aae94606619884d9ce11528666062ab258f16ff20"
VERSION = "b05-cb32-a0-early10-full-protocol-cosine300-1"
ORIGINAL_AUDIT_EPOCHS = (1, 5, 20, 40, 60, 80, 99, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300)


def sha_text(source):
    return hashlib.sha256(source.replace("\r\n", "\n").encode("utf-8")).hexdigest()


def load_parent():
    if sha_text(PARENT.read_text(encoding="utf-8")) != EXPECTED_PARENT:
        raise RuntimeError("Reviewed historical single-card runner SHA differs")
    spec = importlib.util.spec_from_file_location("b05_cb32_a0_historical_runner", PARENT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


parent = load_parent()
ORIGINAL = parent.ORIGINAL
AUDIT_SOURCE = parent.AUDIT_SOURCE
INITIAL_DIGEST = parent.INITIAL_DIGEST


def transformed_source(source):
    source = parent.transformed_source(source, "a0")
    changes = (
        (f'CHECKPOINT_VERSION = "{parent.VERSION}"', f'CHECKPOINT_VERSION = "{VERSION}"'),
        ('parser.add_argument("--epochs", type=int, default=300)',
         'parser.add_argument("--epochs", type=int, default=300)\n    parser.add_argument("--observed-epochs", "--stop-after-epochs", dest="observed_epochs", type=int, default=10)'),
        ('parser.add_argument("--save-every", type=int, default=20)',
         'parser.add_argument("--save-every", type=int, default=1)'),
        ('1 <= args.epochs <= 2 and 1 <= args.steps_per_epoch <= 2 and',
         'args.epochs == 300 and 1 <= args.observed_epochs <= 2 and 1 <= args.steps_per_epoch <= 2 and'),
        ('Smoke: <=2 epochs x2 train steps, train-only, skip-final-test',
         'Smoke: <=2 observed epochs x2 train steps, 300-epoch schedules, train-only, skip-final-test'),
        ('    return args\n\n\ndef parent_configuration(args):',
         '    if args.variant != "a0" or args.resume is not None:\n        parser.error("Early rerun requires fresh A0; no resume")\n    if args.save_every != 1 or args.train_eval_every != 10:\n        parser.error("Save every observed epoch; preserve original clean-eval interval10")\n    if not args.smoke and (args.observed_epochs != 10 or args.audit_epochs != ORIGINAL_AUDIT_EPOCHS):\n        parser.error("Formal rerun observes exactly10 epochs and preserves the historical audit epochs")\n    return args\n\n\ndef parent_configuration(args):'),
        ('"main_training": not args.smoke, "diagnostic_only": args.smoke,',
         '"main_training": False, "diagnostic_only": True, "early_rerun": True, "observed_epochs": args.observed_epochs, "schedule_epochs": 300,'),
        ('        for epoch_index in range(start_epoch, args.epochs):',
         '        for epoch_index in range(start_epoch, args.observed_epochs):'),
        ('manifest.update(status="completed", completed_epochs=args.epochs, optimizer_steps=args.epochs * args.steps_per_epoch, completed_optimizer_steps=args.epochs * args.steps_per_epoch,',
         'manifest.update(status="completed", completed_epochs=args.observed_epochs, optimizer_steps=args.observed_epochs * args.steps_per_epoch, completed_optimizer_steps=args.observed_epochs * args.steps_per_epoch,'),
        ('{"status": "completed", "epoch": args.epochs,',
         '{"status": "completed", "epoch": args.observed_epochs,'),
        ('print("B05_SINGLECARD_" + args.variant.upper() + "_COMPLETED", flush=True)',
         'print("B05_CB32_A0_EARLY_RERUN_COMPLETED", flush=True)'),
        ('"format": CHECKPOINT_VERSION, "model_selection_only": True, "initialization_only": True,',
         '"format": CHECKPOINT_VERSION, "model_selection_only": False, "initialization_only": True,\n                "completed_epochs": 0, "completed_optimizer_steps": 0, "epoch": 0,\n                "optimizer": optimizer.state_dict(), "proxy_optimizer": _optional_state(proxy_optimizer),\n                "scheduler": scheduler.state_dict(), "proxy_scheduler": _optional_state(proxy_scheduler),\n                "rank_states": [checkpoint_rank_state(backbone, sampler, rank, generators)],\n                "split_sha256": split_sha, "train_ids": train_ids, "selection_ids": selection_ids,\n                "selection_domain": selection_prefix, "validation_size": 0,\n                "train_points_sha256": manifest["train_points_sha256"],\n                "best": None, "best_net": None, "best_proxy": None, "best_selection_metrics": None,'),
    )
    for old, new in changes:
        source = parent.replace_once(source, old, new)
    ast.parse(source)
    return source


def successor_configuration(args, original):
    config = parent.successor_configuration(args, original)
    original_name = config["canonical_name"]
    config.update(
        canonical_name=original_name + "-EARLY10-COS300",
        display_name=original_name + "-EARLY10-COS300",
        implementation_variant="b05_cb32_a0_early_audit_20261025",
        observed_epochs=args.observed_epochs,
        stop_after_epochs=args.observed_epochs,
        schedule_epochs=300,
        historical_training_epochs=300,
        diagnostic_only=True,
        early_rerun=True,
        original_recipe_canonical_name=original_name,
        checkpoint_retention="initialization_e0 plus full checkpoint_epoch_001..010; smoke uses its actual observed endpoints",
        audit_epochs_executed=[epoch for epoch in args.audit_epochs if epoch <= args.observed_epochs],
        schedule="Original300-epoch cosine for both optimizers; fresh early stop after requested observed epochs; no restart",
        budget=f"{args.observed_epochs * args.steps_per_epoch} updates observed from original300-epoch recipe; full-test evaluation retained outside smoke",
        comparison_scope="Fresh same-recipe A0 rerun; does not reconstruct historical unretained checkpoint tensors",
        early_rerun_source={
            "historical_singlecard_relative_path": PARENT.relative_to(REPO).as_posix(),
            "historical_singlecard_normalized_sha256": EXPECTED_PARENT,
            "successor_relative_path": HERE.relative_to(REPO).as_posix(),
            "successor_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "transformed_normalized_sha256": sha_text(transformed_source(ORIGINAL.read_text(encoding="utf-8"))),
        },
    )
    return config


def verify_optimizer_payload(checkpoint, completed_epochs):
    if checkpoint.get("format") != VERSION:
        raise RuntimeError("Not a versioned CB32 A0 early rerun checkpoint")
    # Adapt only the wrapper's format tag to reuse historical real-state checks.
    inspected = dict(checkpoint, format=parent.VERSION)
    result = parent.verify_optimizer_payload(inspected, completed_epochs, 300, "a0")
    model_group, = checkpoint["optimizer"]["param_groups"]
    model_scheduler = checkpoint["scheduler"]
    model_lr = .005 + (.1 - .005) * (1 + math.cos(math.pi * completed_epochs / 300)) / 2
    if not (model_scheduler["last_epoch"] == completed_epochs
            and math.isclose(model_group["lr"], model_lr, rel_tol=1e-12, abs_tol=1e-15)):
        raise RuntimeError("Actual RSGD schedule differs from the original300-epoch horizon")
    result.update(observed_epochs=completed_epochs, schedule_epochs=300)
    return result


def build_namespace(source=None):
    for path in (REPO, REPO / "inter_hierarchy_MN40"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    raw = ORIGINAL.read_text(encoding="utf-8") if source is None else source
    namespace = {
        "__name__": "b05_cb32_a0_early_rerun",
        "__file__": str(ORIGINAL),
        "__package__": "inter_hierarchy_MN40.hier_proxy_cycle_v9c",
        "ORIGINAL_AUDIT_EPOCHS": ORIGINAL_AUDIT_EPOCHS,
    }
    exec(compile(transformed_source(raw), str(ORIGINAL) + "[CB32-A0-early10]", "exec"), namespace)
    from inter_hierarchy_MN40.hier_proxy_cycle_v9c import base_protocol

    def base_losses(outputs, labels):
        return base_protocol.global_base_losses(outputs, labels, expected_global_batch=32)

    audit_namespace = {
        "__name__": "b05_cb32_a0_early_audit",
        "__file__": str(AUDIT_SOURCE),
        "__package__": "inter_hierarchy_MN40.hier_proxy_cycle_v9c",
    }
    exec(compile(parent.transformed_audit_source(AUDIT_SOURCE.read_text(encoding="utf-8")),
                 str(AUDIT_SOURCE) + "[CB32-A0-early10]", "exec"), audit_namespace)
    audit_namespace["global_base_losses"] = base_losses
    namespace.update(
        fixed_configuration=lambda args: successor_configuration(args, namespace["parent_configuration"](args)),
        global_base_losses=base_losses,
        audit_parameter_gradients=audit_namespace["audit_parameter_gradients"],
        EXPECTED_ORIGINAL_INITIAL_DIGEST=INITIAL_DIGEST,
    )
    return namespace


if __name__ == "__main__":
    build_namespace()["main"]()
