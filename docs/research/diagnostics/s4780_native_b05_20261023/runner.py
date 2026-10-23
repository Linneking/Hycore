"""Fresh seed4780 CB64 comparison: native HyCoRe or strict B05 + shell HIER A1.

Reuse the SHA-pinned continuous-stream V9C training/checkpoint loop. Select
already reviewed native or B05 operators together, including their gradient
audit and telemetry, so neither arm accidentally inherits the other's GS.
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
VERSION = "hycore-s4780-cb64-native-b05-shell-a1-testselect-1"
INITIAL_DIGESTS = {
    "native": "a207a450feb3c70c34d7ae567b923130b9ff6d88a1a10f4d40755b18d9fb2baf",
    "b05": "8926765a4842652b751873fb89aa37d24d9b0f4f2d831d853b9cc0c1cd46472d",
}
INITIAL_HASHES = {
    "native": "459a93c9589ee369e1473cd74236c066d11278aeaea5c5309d67b461e480f297",
    "b05": None,  # Independently reconstructed by the queue and compared to smoke.
}
OPTIMIZER_OLD = "proxy_optimizer = torch.optim.AdamW(proxy.parameters(), lr=.01, betas=(.9, .999), eps=1e-8, weight_decay=.01)"
OPTIMIZER_NEW = "proxy_optimizer = torch.optim.AdamW(proxy.parameters(), lr=.005, betas=(.9, .999), eps=1e-4, weight_decay=.01)"
REPLACEMENTS = (
    (OPTIMIZER_OLD, OPTIMIZER_NEW),
    ('CHECKPOINT_VERSION = "hycore-cycle-v9c-c01-cap23-gs1-m5002-testselect-1"', f'CHECKPOINT_VERSION = "{VERSION}"'),
    ('    parser.add_argument("--hier", choices=("off", "on"), required=True)',
     '    parser.add_argument("--arm", choices=("native", "b05"), required=True)\n    parser.add_argument("--hier", choices=("on",), required=True)'),
    ('parser.add_argument("--steps-per-epoch", type=int, default=300)',
     'parser.add_argument("--steps-per-epoch", type=int, default=200)'),
    ('!= (300, 300, 8, 16)', '!= (300, 200, 8, 16)'),
    ('Production requires300x300, workers8, eval16', 'Production requires300x200, workers8, eval16'),
    ('def fixed_configuration(args):', 'def base_configuration(args):'),
    ('ClassCycleBatchSampler(labels[train_ids],', 'SourceClassBatchSampler(labels[train_ids],'),
    ('"Cycle-B64 requires exactly two ranks"', '"S4780 CB64 requires exactly two ranks"'),
    ('"whole_part": "c.1 cap2.3 baseGS once"', '"whole_part": config["base_description"]'),
    ('"Initial state differs from original seed4780 c.1/cap/GS: "', '"Initial state differs from independently checked seed4780 arm: "'),
    ('"child_count": child_count, "native_c": .1,', '"child_count": child_count, "native_c": config["native_c"],'),
    ('"V9C_CYCLE_B64_" + args.hier.upper() + "_COMPLETED"', '"S4780_CB64_" + args.arm.upper() + "_COMPLETED"'),
)


def sha_text(source):
    return hashlib.sha256(source.replace("\r\n", "\n").encode("utf-8")).hexdigest()


def transformed_source(source):
    source = source.replace("\r\n", "\n")
    if sha_text(source) != EXPECTED_SOURCE:
        raise RuntimeError("Continuous-stream parent source differs from reviewed SHA")
    for before, after in REPLACEMENTS:
        if source.count(before) != 1:
            raise RuntimeError("Successor edit boundary is not unique: " + before)
        source = source.replace(before, after, 1)
    ast.parse(source)
    return source


def successor_configuration(args, original):
    native = args.arm == "native"
    c = 1.0 if native else .1
    name = ("NATIVE-C1-CAPnone-GSbase0-M1000_4" if native else
            "B05-C0.1-CAP2.3-GSbase1-M500_2")
    name += "-CB64x200-HIERshell4.6-C0.1-GShier1-K20-W0-A1-LR0.005-EPS1e-4-TS-S4780"
    config = dict(original)
    config.update(
        arm=args.arm, canonical_name=name, display_name=name,
        implementation_variant="s4780_native_b05_20261023",
        sampler="uniform 32 distinct classes per global batch; 2 instances/class with replacement; SourceClassBatchSampler v5",
        sampler_epoch_reset=True,
        sampler_state_semantics="stateless deterministic seed/epoch/step plans; no cyclic cursors; does not reseed training RNG",
        c=c, base_c=c, native_c=c, native_extra_cap=None if native else 2.3,
        native_GS=not native, contrastive_margin=4. if native else 2.,
        radial_gamma=1000. if native else 500.,
        radial_margin="1000 / child_count" if native else "500 / child_count",
        geometry_domains={"native_whole_part": c, "hier_view_proxy": .1},
        base_description="native c1; no added cap/GS" if native else "c.1 cap2.3 baseGS once",
        hier_view="direction-preserving fixed depth from native whole" if native else
                  "direction-preserving fixed depth from cap-after/pre-GS whole",
        hook_scope="no base GS; one HIER-view GS; one proxy GS" if native else
                   "one base GS per whole/part; one pre-base-GS HIER-view GS; one proxy GS",
        proxy_optimizer=dict(original["proxy_optimizer"], lr=.005, eps=1e-4),
        ce_temperature=1.,
        random_streams="continuous main RNG; deterministic per-step class plans; persistent private loader/crop/Gumbel; no training per-step reseed",
        schedule="300-epoch cosine; no restart; HIER active update1 (smoke uses separate 2-epoch schedule)",
        comparison_scope="two complete HIER-on recipes; curvature/cap/baseGS/margins differ jointly; no matched H0 arm",
        source_checkpoint=None, teacher=None,
        training_source={
            "original_relative_path": str(ORIGINAL.relative_to(REPO)),
            "original_normalized_sha256": EXPECTED_SOURCE,
            "successor_relative_path": str(HERE.relative_to(REPO)),
            "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "transformed_normalized_sha256": sha_text(transformed_source(ORIGINAL.read_text(encoding="utf-8"))),
        })
    return config


def verify_optimizer_payload(checkpoint, completed_epochs, schedule_epochs):
    """Verify actual saved optimizer and scheduler, rather than declarations."""
    group, = checkpoint["proxy_optimizer"]["param_groups"]
    scheduler = checkpoint["proxy_scheduler"]
    expected_lr = .0005 + (.005 - .0005) * (1 + math.cos(math.pi * completed_epochs / schedule_epochs)) / 2
    if not (checkpoint["format"] == VERSION and checkpoint["completed_epochs"] == completed_epochs
            and group["initial_lr"] == .005 and group["eps"] == 1e-4
            and tuple(group["betas"]) == (.9, .999) and group["weight_decay"] == .01
            and math.isclose(group["lr"], expected_lr, rel_tol=1e-12, abs_tol=1e-15)
            and scheduler["T_max"] == schedule_epochs and scheduler["eta_min"] == .0005
            and scheduler["last_epoch"] == completed_epochs):
        raise RuntimeError("Saved AdamW/scheduler violates A1 contract")
    model_group, = checkpoint["optimizer"]["param_groups"]
    model_scheduler = checkpoint["scheduler"]
    if not (model_group["initial_lr"] == .1 and model_group["momentum"] == .9
            and model_group["weight_decay"] == 2e-4
            and model_scheduler["T_max"] == schedule_epochs and model_scheduler["eta_min"] == .005):
        raise RuntimeError("Saved RSGD/scheduler differs from original HyCoRe recipe")
    return {"initial_lr": group["initial_lr"], "eps": group["eps"], "lr": group["lr"],
            "T_max": scheduler["T_max"], "completed_epochs": completed_epochs}


def build_namespace(arm="native", source=None):
    if arm not in ("native", "b05"):
        raise ValueError("Unknown main-pipe arm")
    for path in (REPO, REPO / "inter_hierarchy_MN40"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    source = ORIGINAL.read_text(encoding="utf-8") if source is None else source
    namespace = {"__name__": "s4780_native_b05_successor", "__file__": str(ORIGINAL),
                 "__package__": "inter_hierarchy_MN40.hier_proxy_cycle_v9c"}
    exec(compile(transformed_source(source), str(ORIGINAL) + "[s4780-cb64-a1]", "exec"), namespace)
    from inter_hierarchy_MN40.hier_proxy_scratch_v5.sampler import SourceClassBatchSampler
    if arm == "native":
        from inter_hierarchy_MN40.hier_proxy_scratch_v9 import geometry, base_protocol, parameter_audit, telemetry

        class TrainingForward(base_protocol.HyCoReTrainingForward):
            def __init__(self, backbone, hier_enabled=True):
                if not hier_enabled:
                    raise ValueError("This launch contains two HIER-on arms only")
                super().__init__(backbone)

        def base_losses(*args, **kwargs):
            values = base_protocol.global_base_losses(*args, **kwargs)
            values["hier_source_mu"] = values["mu"]
            return values

        class Telemetry(telemetry.EpochTelemetry):
            def __init__(self, train_ids, train_labels, num_classes=40, hier_enabled=True):
                super().__init__(train_ids, train_labels, num_classes=num_classes)

    else:
        from inter_hierarchy_MN40.hier_proxy_cycle_v9c import geometry, base_protocol, parameter_audit, telemetry
        TrainingForward = base_protocol.HyCoReTrainingForward
        base_losses = base_protocol.global_base_losses
        Telemetry = telemetry.EpochTelemetry

    original_arguments = namespace["arguments"]

    def fresh_arguments(argv=None):
        args = original_arguments(argv)
        if args.arm != arm or args.resume is not None:
            raise RuntimeError("Arm must match runner; both experiments start fresh, without resume")
        return args

    namespace.update(
        arguments=fresh_arguments,
        fixed_configuration=lambda args: successor_configuration(args, namespace["base_configuration"](args)),
        SourceClassBatchSampler=SourceClassBatchSampler,
        configure_backbone=geometry.configure_backbone,
        freeze_curvature=geometry.freeze_curvature,
        whole_to_hier=geometry.whole_to_hier,
        assert_hier_shell=geometry.assert_hier_shell,
        HyCoReTrainingForward=TrainingForward,
        global_base_losses=base_losses,
        audit_parameter_gradients=parameter_audit.audit_parameter_gradients,
        EpochTelemetry=Telemetry,
        EXPECTED_ORIGINAL_INITIAL_DIGEST=INITIAL_DIGESTS[arm],
    )
    return namespace


if __name__ == "__main__":
    # Read only the arm here; the pinned parser performs full strict validation.
    import argparse
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--arm", choices=("native", "b05"), required=True)
    known, _ = parser.parse_known_args()
    build_namespace(known.arm)["main"]()
