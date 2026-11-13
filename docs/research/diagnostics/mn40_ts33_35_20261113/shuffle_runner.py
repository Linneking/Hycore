"""TS35: fixed-shell HIER on the original TS06 shuffle32 training path."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(REPO))
from inter_hierarchy_MN40.hycore_c01_gs_halfmargin_20261020 import train as ts06

NAME = "HC-C01-Cap23-GS1-M500_2-SH32x307 / Hshell-A1-Ks20-Kp20-L0.5-W0 (S4780)"
ORIGINAL = REPO / "classification_ModelNet40/main_pointmlp_hycore.py"
TRAIN_SHA = "3f7bc6437d6424d1197508e41c7340198277f05d0fef6961af08c110bedcc29e"
ADAPTER_SHA = "9dc3d81bfcf13ddf3621d2b9be34758842a91c23d3e373c1eaf7a00a13f0c18e"
SEED = 4780


def pinned_text(path, expected):
    source = Path(path).read_text(encoding="utf-8")
    if hashlib.sha256(source.encode()).hexdigest() != expected:
        raise RuntimeError(f"TS06 source changed: {path.name}")
    return source


def replace_once(source, before, after):
    if source.count(before) != 1:
        raise RuntimeError(f"Missing unique TS06 boundary: {before}")
    return source.replace(before, after, 1)


def train_source(source, reference):
    import ast

    train = next(node for node in ast.parse(source).body
                 if isinstance(node, ast.FunctionDef) and node.name == "train")
    source = ast.get_source_segment(source, train)
    for before, after in (
        ("        data, label = data.to(device)", "        hier.observe_input(data, label)\n        data, label = data.to(device)"),
        ("        pos_mu, _ = net(pos_data, emb=True)", "        hier.observe_crops(data, pos_data)\n        pos_mu, _ = net(pos_data, emb=True)"),
        ("        loss.backward()", "        hier.observe_base(std_loss, t_loss, h_loss)\n        loss.backward()"),
    ):
        source = replace_once(source, before, after)
    if not reference:
        for before, after in (
            ("        parent_mu, logits = net(data)", "        parent_mu, logits = hier.whole(net, data)"),
            ("        loss.backward()", "        loss = hier.add_loss(loss, label)\n        loss.backward()"),
            ("        optimizer.step()", "        optimizer.step()\n        hier.step()"),
        ):
            source = replace_once(source, before, after)
    return source


class Hierarchy:
    def __init__(self, mode, args):
        self.enabled = mode == "enabled"
        self.reference = mode == "reference"
        self.mode = mode
        self.output = args.run_dir
        self.smoke = args.smoke
        self.epochs = args.epochs
        self.geometry = None
        self.proxy = None
        self.pre_gs = None
        self.updates = 0
        self.counts = Counter()
        self.observation = {}

    def configure_model(self, model, capped):
        if not capped:
            raise RuntimeError("TS35 requires the capped TS06 backbone")
        model = self.original_configure(model, self.reference)
        if not self.reference:
            from inter_hierarchy_MN40.hier_proxy_cycle_v9c.geometry import FinalGeometry

            self.geometry = FinalGeometry()
            model.emb.register_forward_hook(self.geometry)
        return model

    def start(self, torch):
        from inter_hierarchy_MN40.hier_proxy_scratch_v8.hier_loss import HIERLoss

        self.proxy = HIERLoss(seed=SEED).to("cuda")
        self.proxy.assert_geometry()
        self.optimizer = torch.optim.AdamW(self.proxy.parameters(), lr=.005, eps=1e-4,
                                           betas=(.9, .999), weight_decay=.01)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, self.epochs, eta_min=.0005)
        self.gumbel = torch.Generator(device="cuda").manual_seed(SEED + 300001)
        torch.save(self.proxy.state_dict(), self.output / "initial_proxies.pth")

    def observe_input(self, cloud, labels):
        if self.smoke or self.updates == 0:
            self.observation = dict(cloud=tensor_sha(cloud), labels=tensor_sha(labels))

    def observe_crops(self, whole, part):
        if self.observation:
            self.observation.update(whole=tensor_sha(whole), part=tensor_sha(part),
                                    whole_count=whole.shape[-1], part_count=part.shape[-1])

    def observe_base(self, ce, contrastive, radial):
        if self.observation:
            self.observation["base_losses"] = [float(value.detach()) for value in (ce, contrastive, radial)]
            with (self.output / "input_trace.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(self.observation, allow_nan=False) + "\n")
            self.observation = {}

    def whole(self, net, data):
        if not self.enabled:
            return net(data)
        self.geometry.capture_enabled = True
        try:
            output = net(data)
            self.pre_gs = self.geometry.pre_gs
            if self.pre_gs is None:
                raise RuntimeError("Original whole forward did not provide its pre-GS embedding")
            return output
        finally:
            self.geometry.clear()

    def add_loss(self, loss, labels):
        if not self.enabled:
            return loss
        import torch
        from inter_hierarchy_MN40.hier_proxy_cycle_v9c.geometry import assert_hier_shell, whole_to_hier

        self.optimizer.zero_grad(set_to_none=True)
        pre_gs, self.pre_gs = self.pre_gs, None
        shell = whole_to_hier(pre_gs)
        assert_hier_shell(shell)
        # The original sampler draws each row at most once per epoch, so positions
        # and object IDs give identical within-batch identity for mining.
        regularizer, stats = self.proxy(shell, labels, topk=20, proxy_topk=20,
                                        seed=SEED + self.updates * 10007, generator=self.gumbel,
                                        monitor_structure=False)
        if not bool(torch.isfinite(regularizer)):
            raise RuntimeError("Nonfinite TS35 HIER loss")
        for population in ("sample", "proxy"):
            row = stats[population]
            if row["self_k_triplets"]:
                raise RuntimeError("TS35 mined a position self-negative")
            for field in ("active_triplets", "noncollision_triplets", "triplets",
                          "active_i_hinges", "active_j_hinges", "active_k_hinges"):
                self.counts[f"{population}_{field}"] += row[field]
        self.counts["same_class_flip_positions"] += int((labels == labels.flip(0)).sum())
        self.counts["sample_loss_sum"] += stats["sample_loss"]
        self.counts["proxy_loss_sum"] += stats["proxy_loss"]
        if self.updates == 0:
            grad, = torch.autograd.grad(regularizer, pre_gs, retain_graph=True)
            if not bool(torch.isfinite(grad).all()) or not bool((grad != 0).any()):
                raise RuntimeError("HIER has no finite nonzero gradient to the TS06 whole path")
            torch.save({"labels": labels.detach().cpu(), "hier_source_mu": pre_gs.detach().cpu(),
                        "hier_mu": shell.detach().cpu(), "proxy_tangent": self.proxy.tangent_proxies.detach().cpu(),
                        "proxy_ball": self.proxy.proxies().detach().cpu(), "sample_K": 20, "proxy_K": 20,
                        "lambda_hier": .5, "whole_gradient_norm": float(grad.norm())},
                       self.output / "actual_training_epoch_001.pth")
        return loss + .5 * regularizer

    def step(self):
        if self.enabled:
            import torch

            grad = self.proxy.tangent_proxies.grad
            if grad is None or not bool(torch.isfinite(grad).all()):
                raise RuntimeError("Missing or nonfinite TS35 proxy gradient")
            self.optimizer.step()
            cap = self.proxy.project_after_step(self.optimizer)
            self.counts["proxy_forward_clip_count_last"] = cap["forward_clip_count"]
        self.updates += 1

    def checkpoint(self, net, epoch, optimizer, scheduler, metrics, best, torch, np, config):
        if self.enabled:
            metrics["hierarchy"] = dict(self.counts, proxy_lr=self.optimizer.param_groups[0]["lr"])
            self.counts.clear()
            self.scheduler.step()
        state = self.original_checkpoint(net, epoch, optimizer, scheduler, metrics, best, torch, np, config)
        if self.enabled:
            state.update(proxy=self.proxy.state_dict(), proxy_optimizer=self.optimizer.state_dict(),
                         proxy_scheduler=self.scheduler.state_dict(), gumbel_rng=self.gumbel.get_state())
        return state


def tensor_sha(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--mode", choices=("reference", "off", "enabled"), required=True)
    known, remaining = parser.parse_known_args()
    args = ts06.argument_parser().parse_args(remaining)
    if known.mode != "enabled" and not args.smoke:
        parser.error("Only TS35 HIER-enabled can run a full experiment")
    source = pinned_text(ORIGINAL, TRAIN_SHA)
    pinned_text(ts06.__file__, ADAPTER_SHA)
    hier = Hierarchy(known.mode, args)
    original_import = ts06.import_original
    original_identity = ts06.source_identity
    original_install = ts06.install_loss_curvature
    hier.original_configure = ts06.configure_model
    hier.original_checkpoint = ts06.checkpoint_state

    def import_original(repo):
        original = original_import(repo)
        original.hier = hier
        exec(compile(train_source(source, hier.reference), str(ORIGINAL), "exec"), vars(original))
        return original

    def install(hutil):
        original_install(hutil)
        if hier.enabled:
            import torch

            hier.start(torch)

    def identity(repo):
        return dict(original_identity(repo), **{str(Path(__file__).relative_to(repo)): ts06.digest(__file__)})

    original_write = ts06.atomic_json

    def write(path, value):
        if path.name in ("manifest.json", "config.json"):
            config = value.get("training_config", value)
            config.update(sampler="original DataLoader shuffle without replacement",
                          world_size=1, global_batch=32, baseline="TS06/B05", hier_enabled=hier.enabled,
                          comparison_mode=known.mode)
            if hier.enabled:
                config.update(sample_K=20, proxy_K=20, T=50, hier_c=.1, hier_view_depth=4.6,
                              proxy_count=512, proxy_init_scale=.9, proxy_cap=2.3, warmup_epochs=0,
                              lambda_hier_after_warmup=.5, proxy_optimizer=dict(name="AdamW", lr=.005,
                              eps=1e-4, betas=[.9, .999], weight_decay=.01, min_lr=.0005),
                              gradient_hook_scope="native whole/part GS plus independent pre-GS fixed-shell branch")
                if path.name == "manifest.json":
                    value.update(experiment_id="TS35", variant="ts35", canonical_name=NAME,
                                 changed_from_baseline=["add independent fixed-shell HIER to original whole forward"])
        original_write(path, value)

    ts06.import_original = import_original
    ts06.install_loss_curvature = install
    ts06.configure_model = hier.configure_model
    ts06.checkpoint_state = hier.checkpoint
    ts06.source_identity = identity
    ts06.atomic_json = write
    ts06.main(remaining)


if __name__ == "__main__":
    main()
