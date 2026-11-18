"""Fixed-shell HIER on the original TS06 shuffle32 training path."""
from __future__ import annotations

import argparse
from collections import Counter, deque
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(REPO))
from inter_hierarchy_MN40.hycore_c01_gs_halfmargin_20261020 import train as ts06
from history_candidates import SELECTED_HISTORY_ARMS, candidate_statistics, select_history

ARMS = {
    "TS35": (.5, 20, 20),
    "TS36": (.1, 20, 20),
    "TS37": (1., 20, 20),
    "TS38": (.5, 10, 10),
    "TS42": (.5, 10, 20),
    "TS43": (.5, 20, 10),
    "TS44": (.5, 20, 20),
    "TS45": (.5, 20, 20),
    "TS46": (.5, 20, 20),
    "TS47": (.5, 20, 20),
    "TS48": (.5, 20, 20),
    "TS49": (.5, 20, 20),
    "TS50": (.1, 20, 20),
}
CACHE_BATCHES = {"TS44": 1, "TS47": 2, "TS48": 3, "TS49": 3, "TS50": 1}
RAMP_ARM = "TS45"
SAMPLE_ONLY_ARM = "TS46"
SPLIT_WEIGHT_ARMS = (RAMP_ARM, SAMPLE_ONLY_ARM)
RAMP_END_EPOCH = 51
ORIGINAL = REPO / "classification_ModelNet40/main_pointmlp_hycore.py"
TRAIN_SHA = "3f7bc6437d6424d1197508e41c7340198277f05d0fef6961af08c110bedcc29e"
ADAPTER_SHA = "e34d5efda05a7ee90aeda1535f0f810958de0c2348a3b50714c43d3ccb081bc2"
SEED = 4780


def cache_config(arm):
    if arm not in CACHE_BATCHES:
        return {}
    history = CACHE_BATCHES[arm]
    config = dict(sample_max_pool=64 if arm in SELECTED_HISTORY_ARMS else 32 * (history + 1),
                  sample_anchors=32, sample_history_batches=history, sample_history_detached=True,
                  sample_history_reset="each epoch")
    if arm in SELECTED_HISTORY_ARMS:
        config.update(sample_history_selection=SELECTED_HISTORY_ARMS[arm],
                      sample_selected_history_max=32, sample_history_selection_seed=SEED + 700001)
    return config


def canonical_name(arm):
    weight, sample_k, proxy_k = ARMS[arm]
    weights = f"L{weight:.1f}-W0"
    if arm in CACHE_BATCHES:
        history = CACHE_BATCHES[arm]
        pool = cache_config(arm)["sample_max_pool"]
        weights += f"-Hist{history}-SP{pool}"
        if arm in SELECTED_HISTORY_ARMS:
            weights += "-" + ("CP16x2" if arm == "TS48" else "Random32")
    elif arm == RAMP_ARM:
        weights = "Ls0to0.5e51-Lp0.5"
    elif arm == SAMPLE_ONLY_ARM:
        weights = "Ls0.5-Lp0.0-W0"
    return (f"HC-C01-Cap23-GS1-M500_2-SH32x307 / "
            f"Hshell-A1-Ks{sample_k}-Kp{proxy_k}-{weights} (S4780)")


def loss_weights(arm, epoch):
    """Return sample and proxy weights for a one-based training epoch."""
    weight, _, _ = ARMS[arm]
    if arm == RAMP_ARM:
        return weight * min((epoch - 1) / (RAMP_END_EPOCH - 1), 1.0), weight
    return weight, 0.0 if arm == SAMPLE_ONLY_ARM else weight


def split_weight_config(arm):
    initial, proxy = loss_weights(arm, 1)
    terminal_epoch = RAMP_END_EPOCH if arm == RAMP_ARM else 1
    return dict(lambda_sample_initial=initial, lambda_sample_final=loss_weights(arm, terminal_epoch)[0],
                lambda_sample_final_epoch=terminal_epoch, lambda_proxy=proxy)


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
    def __init__(self, mode, args, arm):
        self.enabled = mode == "enabled"
        self.reference = mode == "reference"
        self.weight, self.sample_k, self.proxy_k = ARMS[arm]
        self.arm = arm
        self.split_weights = arm in SPLIT_WEIGHT_ARMS
        self.current_epoch = 1
        self.history_batches = CACHE_BATCHES.get(arm, 0)
        self.use_cache = self.history_batches > 0
        self.previous = deque(maxlen=self.history_batches)
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
        points, pool_labels = shell, labels
        if self.use_cache:
            if self.previous:
                if self.arm in SELECTED_HISTORY_ARMS:
                    historical_points = torch.cat([p for p, _ in self.previous])
                    historical_labels = torch.cat([y for _, y in self.previous])
                    history_labels_cpu = historical_labels.detach().cpu().reshape(-1).tolist()
                    indices = select_history(history_labels_cpu, SELECTED_HISTORY_ARMS[self.arm], self.updates)
                    index_tensor = torch.tensor(indices, dtype=torch.long, device=shell.device)
                    points = torch.cat((shell, historical_points[index_tensor]))
                    pool_labels = torch.cat((labels, historical_labels[index_tensor]))
                    ages = [age for age, (p, _) in enumerate(self.previous, 1) for _ in range(len(p))]
                    candidate_row = candidate_statistics(labels.detach().cpu().reshape(-1).tolist(),
                                                         history_labels_cpu, indices, ages)
                    candidate_row.update(update=self.updates + 1, indices=indices)
                    with (self.output / "candidate_trace.jsonl").open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(candidate_row, allow_nan=False) + "\n")
                    self.counts[f"history_reservoir_{len(historical_points)}_updates"] += 1
                    for key in ("pool_classes", "pool_paired_classes", "current_anchors_with_peer",
                                "current_anchors_with_two_peers", "history_eligible_classes"):
                        self.counts[key + "_sum"] += candidate_row[key]
                    for age, count in candidate_row["selected_age_counts"].items():
                        self.counts[f"selected_history_age_{age}_count"] += count
                else:
                    points = torch.cat((shell, *(p for p, _ in self.previous)))
                    pool_labels = torch.cat((labels, *(y for _, y in self.previous)))
                self.counts["cached_pool_updates"] += 1
            self.previous.appendleft((shell.detach(), labels.detach()))
        regularizer, stats = self.proxy(points, pool_labels, topk=self.sample_k, proxy_topk=self.proxy_k,
                                        seed=SEED + self.updates * 10007, generator=self.gumbel,
                                        monitor_structure=False, return_mining=self.use_cache,
                                        return_components=self.split_weights,
                                        sample_anchor_count=len(shell) if self.use_cache else None)
        if self.split_weights:
            sample_weight, proxy_weight = loss_weights(self.arm, self.current_epoch)
            components = stats.pop("_components")
            sample_loss, proxy_loss = components["sample_loss"], components["proxy_loss"]
            if sample_weight == 0:
                weighted = proxy_weight * proxy_loss
            elif proxy_weight == 0:
                weighted = sample_weight * sample_loss
            else:
                weighted = sample_weight * sample_loss + proxy_weight * proxy_loss
        else:
            weighted = self.weight * regularizer
        if self.use_cache:
            triplets = stats["_sample_mining"]["triplets"]
            if len(triplets) > 50 * len(shell) or bool((triplets[:, 0] >= len(shell)).any()):
                raise RuntimeError("Cached positions must only serve as candidates")
            self.counts["cached_sample_triplets"] += int((triplets[:, 1:] >= len(shell)).any(dim=1).sum())
            self.counts["max_sample_pool"] = max(self.counts["max_sample_pool"], len(points))
            self.counts[f"sample_pool_{len(points)}_updates"] += 1
        if not bool(torch.isfinite(regularizer)) or not bool(torch.isfinite(weighted)):
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
            snapshot = {"labels": labels.detach().cpu(), "hier_source_mu": pre_gs.detach().cpu(),
                        "hier_mu": shell.detach().cpu(), "proxy_tangent": self.proxy.tangent_proxies.detach().cpu(),
                        "proxy_ball": self.proxy.proxies().detach().cpu(),
                        "sample_K": stats["sample"]["topk_including_self"],
                        "proxy_K": stats["proxy"]["topk_including_self"],
                        "whole_gradient_norm": float(grad.norm())}
            if self.split_weights:
                weighted_grad, = torch.autograd.grad(weighted, pre_gs, retain_graph=True, allow_unused=True)
                snapshot.update(lambda_sample=sample_weight, lambda_proxy=proxy_weight,
                                weighted_whole_connected=weighted_grad is not None,
                                weighted_whole_gradient_norm=float(weighted_grad.norm()) if weighted_grad is not None else 0.)
            else:
                snapshot["lambda_hier"] = self.weight
            torch.save(snapshot, self.output / "actual_training_epoch_001.pth")
        if self.split_weights and sample_weight == 0:
            # Backpropagate the proxy-only term separately so the backbone
            # follows the original backward graph during the zero-weight epoch.
            weighted.backward()
            return loss + weighted.detach()
        return loss + weighted

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
            if self.split_weights:
                if self.current_epoch != epoch + 1:
                    raise RuntimeError("HIER weight schedule and training epoch differ")
                sample, proxy = loss_weights(self.arm, self.current_epoch)
                metrics["hierarchy"].update(lambda_sample=sample, lambda_proxy=proxy)
                self.current_epoch += 1
            self.counts.clear()
            self.previous.clear()
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
    parser.add_argument("--arm", choices=ARMS, default="TS35")
    known, remaining = parser.parse_known_args()
    args = ts06.argument_parser().parse_args(remaining)
    if known.mode != "enabled" and not args.smoke:
        parser.error("Only the HIER-enabled path can run a full experiment")
    source = pinned_text(ORIGINAL, TRAIN_SHA)
    pinned_text(ts06.__file__, ADAPTER_SHA)
    hier = Hierarchy(known.mode, args, known.arm)
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
        helper = Path(__file__).with_name("history_candidates.py")
        return dict(original_identity(repo), **{str(Path(__file__).relative_to(repo)): ts06.digest(__file__),
                    str(helper.relative_to(repo)): ts06.digest(helper)})

    original_write = ts06.atomic_json

    def write(path, value):
        if path.name in ("manifest.json", "config.json"):
            config = value.get("training_config", value)
            config.update(sampler="original DataLoader shuffle without replacement",
                          world_size=1, global_batch=32, baseline="TS06/B05", hier_enabled=hier.enabled,
                          comparison_mode=known.mode)
            if hier.enabled:
                config.update(sample_K=hier.sample_k, proxy_K=hier.proxy_k, T=50, hier_c=.1, hier_view_depth=4.6,
                              proxy_count=512, proxy_init_scale=.9, proxy_cap=2.3, warmup_epochs=0,
                              lambda_hier_after_warmup=hier.weight, proxy_optimizer=dict(name="AdamW", lr=.005,
                              eps=1e-4, betas=[.9, .999], weight_decay=.01, min_lr=.0005),
                              gradient_hook_scope="native whole/part GS plus independent pre-GS fixed-shell branch")
                if hier.split_weights:
                    config.pop("lambda_hier_after_warmup", None)
                    config.update(split_weight_config(known.arm))
                if hier.use_cache:
                    config.update(cache_config(known.arm))
                if path.name == "manifest.json":
                    value.update(experiment_id=known.arm, variant=known.arm.lower(), canonical_name=canonical_name(known.arm),
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
