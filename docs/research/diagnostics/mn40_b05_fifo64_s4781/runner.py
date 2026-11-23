"""Run the seed 4781 B05 H0 / TS44 FIFO1 pair on the frozen TS44 source."""
from __future__ import annotations

import argparse
import ast
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(REPO))

from docs.research.diagnostics.mn40_ts33_35_20261113 import shuffle_runner as ts44

ts06 = ts44.ts06
SEED = 4781
TS44_COMMIT = "78e772209dbb263bde844f748867b118c644d9ed"
ARM_NAMES = {"h0": "MN40-B05-S4781-H0", "hier": "MN40-B05-S4781-HIER-FIFO1"}
PINNED = {
    "docs/research/diagnostics/mn40_ts33_35_20261113/shuffle_runner.py":
        "87ab50ab91c1895e9c5b3b6d718b532d17e622f658c9f4d582f9f63e69dd7bc1",
    "inter_hierarchy_MN40/hier_proxy_scratch_v8/hier_loss.py":
        "6f3a0566926862b0da78f15db11c0fb9b3cee2d83230106028c2d9e6b6930b49",
    "inter_hierarchy_MN40/hier_proxy_scratch_v5/relations.py":
        "8f74a1cfaea601613ea7b4894ff3f3154e4737386dd3c0fa2c64ea1aa942bab4",
    "inter_hierarchy_MN40/hier_proxy_cycle_v9c/geometry.py":
        "3fc8fe66fbfe41fa62fe511abecf956ebd5cb8f3a84bfc5b01c2f50c7e259d3b",
    "inter_hierarchy_MN40/hier_proxy_scratch_v6/hier_loss.py":
        "e2cd382dfc3be20605c3f50b01f8be3fed11e11f7c8b0f3b4f869997e521dcfa",
    "inter_hierarchy_MN40/hier_proxy_scratch_v8/geometry.py":
        "16d5a7879d8d23924c6fe542bb1d47973448044604602f4ab44c9c72f8c871dc",
}


def patched_adapter(source, filename):
    """Reuse TS06, parameterizing just its seed, model label, and shard order."""
    tree = ast.parse(source, filename)
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
    seed_nodes = [node for node in ast.walk(main)
                  if isinstance(node, ast.Constant) and node.value in (4780, "4780")]
    if sum(isinstance(node.value, int) for node in seed_nodes) != 6 or len(seed_nodes) != 7:
        raise RuntimeError("Unexpected TS06 random-seed boundaries")
    for node in seed_nodes:
        node.value = type(node.value)(SEED)
    for node in ast.walk(main):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value.endswith("-S4780"):
            node.value = node.value.removesuffix("-S4780") + f"-S{SEED}"
    loader = next(node for node in tree.body
                  if isinstance(node, ast.FunctionDef) and node.name == "install_data_root")
    load_text = ast.get_source_segment(source, loader)
    load_text = ts44.replace_once(
        load_text, 'os.environ.get("HYCORE_TRAIN_SHARD_ORDER") if partition == "train" else None',
        'os.environ.get(f"HYCORE_{partition.upper()}_SHARD_ORDER")',
    )
    module = ast.Module(body=[main], type_ignores=[])
    return compile(module, filename, "exec"), compile(load_text, filename, "exec")


class Hierarchy(ts44.Hierarchy):
    def start(self, torch):
        import random

        import numpy as np

        before = ts06.rng_state(torch, np)
        super().start(torch)
        after = ts06.rng_state(torch, np)
        if before["python"] != after["python"]:
            raise RuntimeError("Proxy initialization consumed Python's training RNG")
        np.testing.assert_equal(before["numpy"], after["numpy"])
        if not torch.equal(before["torch_cpu"], after["torch_cpu"]) or any(
            not torch.equal(a, b) for a, b in zip(before["torch_cuda"], after["torch_cuda"])
        ):
            raise RuntimeError("Proxy initialization consumed Torch's training RNG")

    def checkpoint(self, net, epoch, optimizer, scheduler, metrics, best, torch, np, config):
        state = super().checkpoint(net, epoch, optimizer, scheduler, metrics, best, torch, np, config)
        if self.enabled:
            metrics["hierarchy"]["proxy_updates"] = self.updates
            if metrics["hierarchy"].get("cached_pool_updates") != metrics["epoch_optimizer_steps"] - 1:
                raise RuntimeError("TS44 must use history on every step after the first in each epoch")
        return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=(*ARM_NAMES, "reference"), required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--expected-data", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--source-branch", required=True)
    parser.add_argument("--expected-initial-model")
    parser.add_argument("--smoke", action="store_true")
    options = parser.parse_args()
    if options.arm == "reference" and not options.smoke:
        parser.error("The native TS06 reference is used only for the paired smoke check")
    if not options.smoke and not options.expected_initial_model:
        parser.error("Production requires the shared preflight model digest")
    if os.environ.get("PYTHONHASHSEED") != str(SEED):
        parser.error(f"Start the process with PYTHONHASHSEED={SEED}")

    expected = json.loads(options.expected_data.read_text(encoding="utf-8"))
    for split in ("train", "test"):
        os.environ[f"HYCORE_{split.upper()}_SHARD_ORDER"] = json.dumps(expected[split]["file_order"])

    for path, digest in PINNED.items():
        ts44.pinned_text(REPO / path, digest)
    source = ts44.pinned_text(ts44.ORIGINAL, ts44.TRAIN_SHA)
    adapter = ts44.pinned_text(ts06.__file__, ts44.ADAPTER_SHA)
    main_code, loader_code = patched_adapter(adapter, ts06.__file__)
    legacy_args = ["--variant", "c01-cap", "--data-dir", str(options.data_dir),
                   "--run-dir", str(options.run_dir), "--source-commit", options.source_commit,
                   "--source-branch", options.source_branch]
    if options.smoke:
        legacy_args += ["--smoke", "--epochs", "2", "--steps-per-epoch", "2"]
    args = ts06.argument_parser().parse_args(legacy_args)
    ts44.SEED = SEED
    mode = {"reference": "reference", "h0": "off", "hier": "enabled"}[options.arm]
    hier = Hierarchy(mode, args, ts44.CACHED_POOL_ARM)
    original_import = ts06.import_original
    original_install = ts06.install_loss_curvature
    original_identity = ts06.source_identity
    original_write = ts06.atomic_json
    hier.original_configure = ts06.configure_model
    hier.original_checkpoint = ts06.checkpoint_state

    def import_original(repo):
        original = original_import(repo)
        original.hier = hier
        exec(compile(ts44.train_source(source, hier.reference), str(ts44.ORIGINAL), "exec"), vars(original))
        return original

    def install(hutil):
        original_install(hutil)
        if hier.enabled:
            import torch

            hier.start(torch)

    def identity(repo):
        paths = (*PINNED, str(Path(__file__).relative_to(repo)))
        return {**original_identity(repo), **{path: ts06.digest(repo / path) for path in paths}}

    def write(path, value):
        if path.name in ("manifest.json", "config.json"):
            config = value.get("training_config", value)
            config.update(sampler="original DataLoader shuffle without replacement", world_size=1,
                          global_batch=32, baseline="TS06/B05", hier_enabled=hier.enabled,
                          comparison_mode=mode, test_shard_order=expected["test"]["file_order"])
            if hier.enabled:
                config.update(sample_K=20, proxy_K=20, T=50, hier_c=.1, hier_view_depth=4.6,
                              proxy_count=512, proxy_init_scale=.9, proxy_cap=2.3, warmup_epochs=0,
                              lambda_hier_after_warmup=.5, lambda_sample=.5, lambda_proxy=.5,
                              proxy_optimizer=dict(name="AdamW", lr=.005, eps=1e-4, betas=[.9, .999],
                                                   weight_decay=.01, min_lr=.0005),
                              proxy_seed=SEED, gumbel_seed=SEED + 300001, mining_seed_stride=10007,
                              sample_max_pool=64, sample_anchors=32, sample_history_batches=1,
                              sample_history_detached=True, sample_history_reset="each epoch",
                              gradient_hook_scope="native whole/part GS plus independent pre-GS fixed-shell branch")
            if path.name == "manifest.json":
                value.update(experiment_id=ARM_NAMES.get(options.arm, "MN40-B05-S4781-REFERENCE"),
                             canonical_name=ARM_NAMES.get(options.arm, "MN40-B05-S4781-REFERENCE"),
                             variant=options.arm, ts44_source_commit=TS44_COMMIT)
                if hier.enabled:
                    value["changed_from_baseline"] = ["fixed-shell HIER with detached one-step FIFO candidates"]
                if value["status"] == "running" and value["completed_epochs"] == 0:
                    loaded = value["dataset_identity"]
                    splits = ("train",) if args.smoke else ("train", "test")
                    if loaded != {split: expected[split] for split in splits}:
                        raise RuntimeError("The actual loaded ModelNet40 data identity differs from the paired reference")
                    if not args.smoke and value["initialized_model_sha256"] != options.expected_initial_model:
                        raise RuntimeError("Production model initialization differs from the paired preflight")
        original_write(path, value)

    ts06.NAMES = {key: value.replace("S4780", f"S{SEED}") for key, value in ts06.NAMES.items()}
    ts06.import_original = import_original
    ts06.install_loss_curvature = install
    ts06.configure_model = hier.configure_model
    ts06.checkpoint_state = hier.checkpoint
    ts06.source_identity = identity
    ts06.atomic_json = write
    exec(main_code, vars(ts06))
    exec(loader_code, vars(ts06))
    ts06.main(legacy_args)


if __name__ == "__main__":
    main()
