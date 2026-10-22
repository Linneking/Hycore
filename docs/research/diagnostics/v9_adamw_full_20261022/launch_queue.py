"""One full V9 AdamW candidate: published source, CPU gates, idle two-GPU smoke, fresh300.

Reuse the reviewed V9 production queue's immutable-source, GPU-lock and timeout
behavior. The successor entry point receives a new source identity and verifies
the optimizer saved by its independent smoke before production can start.
"""
from __future__ import annotations

import ast
import hashlib
import importlib
import json
from pathlib import Path
import shutil
import sys

import runner

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
RELATIVE = Path("docs/research/diagnostics/v9_adamw_full_20261022")
ORIGINAL = REPO / "inter_hierarchy_MN40/hier_proxy_scratch_v9/queue.py"
EXPECTED_QUEUE = "d8dd0a2dc810fc4f838f9d7ae6b00abea956b07c4f82208a825e5e7ff7fc764f"


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise RuntimeError("V9 queue successor edit boundary changed")
    return source.replace(old, new, 1)


def transformed_queue_source(source):
    source = source.replace("\r\n", "\n")
    if runner.sha_text(source) != EXPECTED_QUEUE:
        raise RuntimeError("Original V9 queue differs from the reviewed source SHA")
    replacements = (
        ("        snapshot = freeze_source(args.repo, args.root, args.expected_commit)",
         "        snapshot = freeze_successor_source(args.repo, args.root, args.expected_commit)"),
        ("'display_name': 'V9-HIER64-Cbase1-Chier0.1-H4.6-GShier-M1000_4-TS-S22', 'epochs': 300, 'steps_per_epoch': 200,",
         "'display_name': SUCCESSOR_DISPLAY_NAME, 'proxy_initial_lr': .005, 'proxy_eps': 1e-4, 'epochs': 300, 'steps_per_epoch': 200,"),
        ("        cpu_commands = [", "        cpu_commands = [\n            [sys.executable, '-m', 'unittest', 'discover', '-s', str(snapshot / SUCCESSOR_RELATIVE), '-p', 'test_contracts.py', '-v'],"),
        ("'--module', 'inter_hierarchy_MN40.hier_proxy_scratch_v9.train',", "str(snapshot / SUCCESSOR_RELATIVE / 'runner.py'),"),
        ("                state.update(smoke_passed=True, smoke_commit=manifest['commit'],",
         "                actual_optimizer = verify_successor_smoke(smoke, manifest)\n                state.update(smoke_actual_optimizer=actual_optimizer, smoke_passed=True, smoke_commit=manifest['commit'],"),
        ("                state.update(status='completed' if code == 0 else 'production_failed',",
         "                if code == 0:\n                    state['completed_actual_optimizer'] = verify_successor_completion(production)\n                state.update(status='completed' if code == 0 else 'production_failed',"),
    )
    for old, new in replacements:
        source = replace_once(source, old, new)
    ast.parse(source)
    return source


def freeze_successor_source(repo, root, expected):
    parent = importlib.import_module("inter_hierarchy_MN40.hier_proxy_scratch_v9.queue")
    snapshot = parent.freeze_source(repo, root, expected)
    target = snapshot / RELATIVE
    target.mkdir(parents=True, exist_ok=False)
    identity = json.loads((root / "source_identity.json").read_text())
    for source in (repo / RELATIVE).glob("*.py"):
        destination = target / source.name
        shutil.copy2(source, destination)
        identity["files"][str(RELATIVE / source.name)] = hashlib.sha256(destination.read_bytes()).hexdigest()
    identity["successor"] = {"relative_path": str(RELATIVE), "version": runner.VERSION,
        "proxy_initial_lr": .005, "proxy_eps": 1e-4, "proxy_min_lr": .0005,
        "parent_train_normalized_sha256": runner.EXPECTED_SOURCE,
        "parent_queue_normalized_sha256": EXPECTED_QUEUE}
    parent.write(root / "source_identity.json", identity)
    return snapshot


def verify_successor_smoke(smoke, manifest):
    config = manifest["training_config"]
    if not (config["implementation_variant"] == "v9_adamw_full_20261022"
        and config["display_name"] == runner.DISPLAY_NAME
        and config["proxy_optimizer"]["lr"] == .005 and config["proxy_optimizer"]["eps"] == 1e-4
        and config["proxy_optimizer"]["min_lr"] == .0005
        and config["epochs"] == 2 and config["steps_per_epoch"] == 2
        and manifest["resume_identity"] is None):
        raise RuntimeError("Full-run successor smoke metadata differs from reviewed configuration")
    # The original short smoke has a two-epoch schedule. Production is a new
    # process whose enforced non-smoke arguments set both schedules to 300.
    return runner.verify_saved_optimizer(smoke, 2, 2)


def verify_successor_completion(production):
    manifest = json.loads((production / "manifest.json").read_text())
    config = manifest["training_config"]
    if not (manifest["status"] == "completed" and manifest["completed_epochs"] == 300
        and manifest["official_test_read"] and manifest["selection_domain"] == "test"
        and manifest["selection_size"] == 2468 and manifest["train_size"] == 9840
        and manifest["resume_identity"] is None and config["epochs"] == 300
        and config["steps_per_epoch"] == 200 and config["proxy_optimizer"]["lr"] == .005
        and config["proxy_optimizer"]["eps"] == 1e-4):
        raise RuntimeError("Full-run completion manifest differs from its production contract")
    return runner.verify_saved_optimizer(production, 300, 300)


def main():
    for path in (REPO, REPO / "inter_hierarchy_MN40"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    source = ORIGINAL.read_text(encoding="utf-8")
    namespace = {"__name__": "v9_adamw_full_queue", "__file__": str(ORIGINAL),
        "__package__": "inter_hierarchy_MN40.hier_proxy_scratch_v9"}
    exec(compile(transformed_queue_source(source), str(ORIGINAL) + "[adamw-full-queue]", "exec"), namespace)
    namespace.update(freeze_successor_source=freeze_successor_source,
        SUCCESSOR_RELATIVE=RELATIVE, SUCCESSOR_DISPLAY_NAME=runner.DISPLAY_NAME,
        verify_successor_smoke=verify_successor_smoke,
        verify_successor_completion=verify_successor_completion)
    namespace["main"]()


if __name__ == "__main__":
    main()
