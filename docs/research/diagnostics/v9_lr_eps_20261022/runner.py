"""Ten-epoch V9 diagnostic with proxy AdamW lr=.005 and eps=1e-4.

Reuse the reviewed train-only V9 observer without changing its source. The
generated training program changes only those two optimizer values and its
checkpoint version. The original 300-epoch cosine horizon is retained.
"""
from __future__ import annotations

import ast
import hashlib
import importlib.util
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
BASE_DIR = HERE.parent / "v9_proxy_cap_cause_20261021"
BASE_PATH = BASE_DIR / "runner.py"
SPEC = importlib.util.spec_from_file_location("v9_lr_eps_legacy_observer", BASE_PATH)
base = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(base)
ORIGINAL = base.ORIGINAL
EXPECTED_INITIAL_MODEL = base.EXPECTED_INITIAL_MODEL
SELECTED_STEPS = base.SELECTED_STEPS
BASE_TRANSFORM = base.transformed_source
BASE_CONFIGURATION = base.diagnostic_configuration
VERSION = "hycore-hier-v9-proxy-lr005-eps1e4-train-only-diagnostic-1"
OPTIMIZER_OLD = "proxy_optimizer = torch.optim.AdamW(proxy.parameters(), lr=.01, betas=(.9, .999), eps=1e-8, weight_decay=.01)"
OPTIMIZER_NEW = "proxy_optimizer = torch.optim.AdamW(proxy.parameters(), lr=.005, betas=(.9, .999), eps=1e-4, weight_decay=.01)"
VERSION_OLD = 'CHECKPOINT_VERSION = "hycore-hier-v9-proxy-cap-train-only-diagnostic-1"'
VERSION_NEW = f'CHECKPOINT_VERSION = "{VERSION}"'


def transformed_source(source):
    transformed = BASE_TRANSFORM(source)
    transformed = base.replace_once(transformed, OPTIMIZER_OLD, OPTIMIZER_NEW)
    transformed = base.replace_once(transformed, VERSION_OLD, VERSION_NEW)
    ast.parse(transformed)
    return transformed


diagnostic_arguments = base.diagnostic_arguments


def diagnostic_configuration(args, original):
    config = BASE_CONFIGURATION(args, original)
    config["proxy_optimizer"] = dict(config["proxy_optimizer"], lr=.005, eps=1e-4)
    config.update(
        display_name="V9-HIER64 LR0.005 EPS1e-4 train-only 10-epoch diagnostic S22",
        diagnostic_variant="v9_proxy_lr005_eps1e4_20261022",
        optimization_changes_from_report71={"proxy_initial_lr": [.01, .005], "proxy_eps": [1e-8, 1e-4]},
        comparison_reference="report71 independent fresh 10x200 diagnostic; no claim of historical bitwise replay",
    )
    return config


def build_namespace(source=None):
    # Base build_namespace resolves these callbacks from its module globals.
    base.transformed_source = transformed_source
    base.diagnostic_configuration = diagnostic_configuration
    if str(BASE_DIR) not in sys.path:
        sys.path.insert(0, str(BASE_DIR))
    namespace = base.build_namespace(source)
    previous_identity = namespace["diagnostic_source_identity"]()
    identity = dict(previous_identity,
        successor_relative_path=str(HERE.relative_to(REPO)),
        successor_runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        inherited_observer_relative_path=str(BASE_PATH.relative_to(REPO)),
        optimizer_changes={"initial_lr": .005, "eps": 1e-4},
        unchanged_proxy_schedule={"epochs": 300, "min_lr": .0005},
    )
    namespace["diagnostic_source_identity"] = lambda: identity
    return namespace


if __name__ == "__main__":
    build_namespace()["main"]()
