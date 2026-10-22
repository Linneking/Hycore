"""Fresh full V9-HIER with proxy AdamW lr=.005 and eps=1e-4.

Compile the SHA-pinned production entry point with two optimizer changes. Keep
its complete training, test-selection, monitoring and checkpoint code intact.
The bounded report73 observer is deliberately not loaded by this entry point.
"""
from __future__ import annotations

import ast
import hashlib
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
ORIGINAL = REPO / "inter_hierarchy_MN40/hier_proxy_scratch_v9/train.py"
EXPECTED_SOURCE = "b7d06467e4f6a8650fb4317655ea5c3fcfef567126198b325e885a0b2f28a040"
VERSION = "hycore-hier-v9-nativec1-shellc01-adamw005-eps1e4-testselect-1"
DISPLAY_NAME = "V9-HIER64-K20-W0-Cbase1-Chier0.1-H4.6-GShier-M1000_4-PLR0.005-EPS1e-4-TS-S22"
OPTIMIZER_OLD = "proxy_optimizer = torch.optim.AdamW(proxy.parameters(), lr=.01, betas=(.9, .999), eps=1e-8, weight_decay=.01)"
OPTIMIZER_NEW = "proxy_optimizer = torch.optim.AdamW(proxy.parameters(), lr=.005, betas=(.9, .999), eps=1e-4, weight_decay=.01)"
REPLACEMENTS = (
    (OPTIMIZER_OLD, OPTIMIZER_NEW),
    ('CHECKPOINT_VERSION = "hycore-hier-v9-nativec1-shellc01-testselect-1"', f'CHECKPOINT_VERSION = "{VERSION}"'),
    ('    args = arguments()\n', '    args = fresh_arguments()\n'),
    ('    config = fixed_configuration(args)\n', '    config = successor_configuration(args, fixed_configuration(args))\n'),
)


def sha_text(source):
    return hashlib.sha256(source.replace("\r\n", "\n").encode("utf-8")).hexdigest()


def transformed_source(source):
    source = source.replace("\r\n", "\n")
    if sha_text(source) != EXPECTED_SOURCE:
        raise RuntimeError("Original production V9 source differs from the reviewed SHA")
    for before, after in REPLACEMENTS:
        if source.count(before) != 1:
            raise RuntimeError("V9 optimizer successor edit boundary is no longer unique")
        source = source.replace(before, after, 1)
    ast.parse(source)
    return source


def successor_configuration(args, original):
    config = dict(original)
    config["proxy_optimizer"] = dict(original["proxy_optimizer"], lr=.005, eps=1e-4)
    config.update(display_name=DISPLAY_NAME,
        implementation_variant="v9_adamw_full_20261022",
        optimizer_changes_from_v9={"proxy_initial_lr": [.01, .005], "proxy_eps": [1e-8, 1e-4]},
        comparison_reference="original V9-HIER full run and report73 bounded joint optimizer candidate",
        training_source={"original_relative_path": str(ORIGINAL.relative_to(REPO)),
            "original_normalized_sha256": EXPECTED_SOURCE,
            "successor_relative_path": str(HERE.relative_to(REPO)),
            "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "transformed_normalized_sha256": sha_text(transformed_source(ORIGINAL.read_text(encoding="utf-8")))})
    return config


def verify_optimizer_payload(checkpoint, completed_epochs, schedule_epochs):
    """Inspect saved actual optimizer state, not its configuration declaration."""
    import math
    groups = checkpoint["proxy_optimizer"]["param_groups"]
    scheduler = checkpoint["proxy_scheduler"]
    expected_lr = .0005 + (.005 - .0005) * (1 + math.cos(math.pi * completed_epochs / schedule_epochs)) / 2
    if len(groups) != 1:
        raise RuntimeError("Unexpected proxy optimizer group count")
    group = groups[0]
    if not (checkpoint["format"] == VERSION and checkpoint["completed_epochs"] == completed_epochs
        and group["initial_lr"] == .005 and group["eps"] == 1e-4
        and group["betas"] == (.9, .999) and group["weight_decay"] == .01
        and math.isclose(group["lr"], expected_lr, rel_tol=1e-12, abs_tol=1e-15)
        and scheduler["T_max"] == schedule_epochs and scheduler["eta_min"] == .0005
        and scheduler["last_epoch"] == completed_epochs):
        raise RuntimeError("Saved actual AdamW/scheduler differs from the full-run contract")
    return {"initial_lr": group["initial_lr"], "eps": group["eps"], "lr": group["lr"],
        "T_max": scheduler["T_max"], "eta_min": scheduler["eta_min"], "completed_epochs": completed_epochs}


def verify_saved_optimizer(run_dir, completed_epochs, schedule_epochs):
    import torch
    checkpoint = torch.load(Path(run_dir) / "last.pth", map_location="cpu", weights_only=False)
    return verify_optimizer_payload(checkpoint, completed_epochs, schedule_epochs)


def build_namespace(source=None):
    source = ORIGINAL.read_text(encoding="utf-8") if source is None else source
    transformed = transformed_source(source)
    for path in (REPO, REPO / "inter_hierarchy_MN40"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    namespace = {"__name__": "v9_adamw_full_successor", "__file__": str(ORIGINAL),
        "__package__": "inter_hierarchy_MN40.hier_proxy_scratch_v9"}
    exec(compile(transformed, str(ORIGINAL) + "[adamw-full-successor]", "exec"), namespace)
    original_arguments = namespace["arguments"]

    def fresh_arguments():
        args = original_arguments()
        if args.resume is not None:
            raise RuntimeError("This full V9 optimizer comparison must start fresh; resume is forbidden")
        return args

    namespace.update(fresh_arguments=fresh_arguments, successor_configuration=successor_configuration)
    return namespace


if __name__ == "__main__":
    build_namespace()["main"]()
