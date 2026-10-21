"""Bounded, train-only, from-scratch V9 proxy-cap cause observation.

The original V9 training entry point is compiled after asserted textual
transformations. Its sampler, forwards, objectives, gradient synchronization,
optimizers and 300-epoch cosine horizon remain unchanged. This diagnostic stops
after 1..10 epochs and performs no evaluation, model selection or test file read.
Private output tensors and full logs belong in a new server run directory.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
ORIGINAL = REPO / "inter_hierarchy_MN40/hier_proxy_scratch_v9/train.py"
EXPECTED_INITIAL_MODEL = "fd7251e0eb6b397ecaf3862e9cc00a6bc2be9b684329fdc31dc05915b388d29d"
EXPECTED_SOURCE = "b7d06467e4f6a8650fb4317655ea5c3fcfef567126198b325e885a0b2f28a040"
SELECTED_STEPS = (1, 2, 5, 10, 20, 50, 100, 200)


def sha_text(source):
    return hashlib.sha256(source.replace("\r\n", "\n").encode("utf-8")).hexdigest()


def replace_once(source, old, new):
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"V9 source contract changed: expected one occurrence, found {count}: {old[:100]!r}")
    return source.replace(old, new, 1)


def replace_region(source, start, end, replacement):
    if source.count(start) != 1 or source.count(end) != 1:
        raise RuntimeError("V9 source region anchors are not unique")
    left, right = source.index(start), source.index(end)
    if right <= left:
        raise RuntimeError("V9 source region anchors are out of order")
    return source[:left] + replacement + source[right:]


def transformed_source(source):
    """Fail closed when the frozen V9 entry point differs at any edit boundary."""
    source = source.replace("\r\n", "\n")
    if sha_text(source) != EXPECTED_SOURCE:
        raise RuntimeError("Original V9 source SHA differs from the reviewed diagnostic source")
    source = replace_once(source,
        'CHECKPOINT_VERSION = "hycore-hier-v9-nativec1-shellc01-testselect-1"',
        'CHECKPOINT_VERSION = "hycore-hier-v9-proxy-cap-train-only-diagnostic-1"')
    source = replace_once(source, '    args = arguments()\n', '    args = diagnostic_arguments()\n')
    source = replace_once(source, '    config = fixed_configuration(args)\n',
                          '    config = diagnostic_configuration(args, fixed_configuration(args))\n')
    source = replace_once(source, '"status": "starting", "main_training": not args.smoke,\n        "diagnostic_only": args.smoke,',
                          '"status": "starting", "main_training": False,\n        "diagnostic_only": True,')
    source = replace_region(source,
        '        if args.smoke:\n            selection_ids = train_ids[:args.eval_batch_size * args.smoke_eval_batches]',
        '        split_sha = split_identity(train_ids, selection_ids)',
        '''        # Diagnostic: train shards only; no evaluation or selection data.
        selection_ids = np.empty(0, dtype=np.int64)
        selection_points, selection_labels = points[:0], labels[:0]
        selection_shards, selection_prefix = [], "none_diagnostic_train_only"
''')
    source = replace_once(source, '        manifest["initialized_model_sha256"] = initial_hash\n',
        '''        manifest["initialized_model_sha256"] = initial_hash
        if initial_hash != DIAGNOSTIC_EXPECTED_INITIAL_MODEL:
            raise RuntimeError("Diagnostic initialization differs from historical V9 e0")
        manifest["diagnostic_source"] = diagnostic_source_identity()
''')
    source = replace_once(source, '        broadcast_parameters(proxy)\n',
        '''        broadcast_parameters(proxy)
        observer.bind(proxy, args.run_dir, rank, manifest, config)
''')
    source = replace_once(source, '        for epoch_index in range(start_epoch, args.epochs):\n',
                          '        for epoch_index in range(start_epoch, args.diagnostic_epochs):\n')
    source = replace_once(source, '                whole_count, child_count = shared_crop_counts(args.seed, epoch_index, step)\n',
        '''                whole_count, child_count = shared_crop_counts(args.seed, epoch_index, step)
                observer.begin(epoch, step + 1)
''')
    source = replace_once(source, '                loss = values["base"] + lambda_hier * hier_loss\n',
        '''                loss = values["base"] + lambda_hier * hier_loss
                observer.loss_ready(hier_stats, proxy, hier_mu, values, ids,
                                    relation_seed, whole_count, child_count, lambda_hier)
''')
    source = replace_once(source, '                optimizer.step()\n                before_proxy = proxy.tangent_proxies.detach().clone()\n',
        '''                optimizer.step()
                proxy_observation = observer.before_step(proxy, proxy_optimizer)
                before_proxy = proxy.tangent_proxies.detach().clone()
''')
    source = replace_once(source, '                proxy_cap_stats = proxy.project_after_step(proxy_optimizer)\n',
        '''                proxy_cap_stats = proxy.project_after_step(proxy_optimizer)
                observer.after_step(proxy_observation, proxy, proxy_optimizer)
''')
    source = replace_region(source,
        '            selection = rank0_evaluation(backbone, selection_points, selection_labels, args, device, selection_prefix, rank)\n',
        '            rank_states = [None, None]\n',
        '''            # No extra model forward, model selection, or score is computed.
            selection, clean_train, improved, best = None, None, False, None
''')
    source = replace_once(source,
        '''                    print(json.dumps({"epoch": epoch, "selection_domain": selection_prefix,
                        "selection_oa": selection[selection_prefix + "_oa"],
                        "selection_ce": selection[selection_prefix + "_ce"], "best": best},
                        allow_nan=False), flush=True)
''',
        '''                    print(json.dumps({"epoch": epoch, "diagnostic_train_only": True,
                        "selection": None}, allow_nan=False), flush=True)
''')
    source = replace_region(source,
        '        test_metrics, final_error = None, None\n',
        '        dist.destroy_process_group()\n        if rank == 0:\n',
        '''        test_metrics = None
        dist.barrier()
''')
    source = replace_once(source, 'manifest.update(status="completed", completed_epochs=args.epochs,',
                          'manifest.update(status="completed", completed_epochs=args.diagnostic_epochs,')
    source = replace_once(source, 'final_test=test_metrics, final_summary_skipped=args.smoke, test_used_for_selection=not args.smoke)',
                          'final_test=None, final_summary_skipped=True, test_used_for_selection=False)')
    source = replace_once(source, '                "epoch": args.epochs, "updated_utc": manifest["finished_utc"], "best": best})',
                          '                "epoch": args.diagnostic_epochs, "updated_utc": manifest["finished_utc"], "best": None})')
    source = replace_once(source, '            print("V9_FULL9840_TESTSELECT_TRAINING_COMPLETED", flush=True)',
                          '            print("V9_TRAIN_ONLY_BOUNDED_DIAGNOSTIC_COMPLETED", flush=True)')
    source = replace_once(source, 'if __name__ == "__main__":\n    main()\n', '')
    ast.parse(source)
    return source


def diagnostic_arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--diagnostic-epochs", type=int, default=10)
    p.add_argument("--steps-per-epoch", type=int, default=200)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args(argv)
    if not 1 <= a.diagnostic_epochs <= 10:
        p.error("Bounded diagnostic requires 1..10 epochs")
    if not 0 <= a.workers <= 2:
        p.error("Diagnostic requires 0..2 workers")
    if a.smoke:
        if a.diagnostic_epochs > 2 or not 1 <= a.steps_per_epoch <= 3:
            p.error("Smoke requires <=2 epochs and 1..3 steps per epoch")
    elif a.steps_per_epoch != 200:
        p.error("V9 diagnostic requires the original 200 steps per epoch")
    # The original scheduler sees 300 throughout, including the separate smoke.
    a.seed, a.epochs, a.warmup_epochs, a.lambda_hier = 22, 300, 0, .5
    a.eval_batch_size, a.train_eval_every = 32, 10
    a.audit_epochs, a.save_every = (), 20
    a.skip_final_test, a.smoke_eval_batches, a.resume = True, 2, None
    return a


def diagnostic_configuration(args, original):
    original.update({
        "display_name": "V9-HIER64 proxy-cap train-only bounded diagnostic S22",
        "diagnostic_epochs": args.diagnostic_epochs,
        "diagnostic_only": True, "early_stopping": False,
        "schedule": "original 300-epoch cosine; bounded observation stops after diagnostic_epochs",
        "selection": "none", "selection_forward": "none; zero evaluation forwards",
        "validation": "none", "evaluation_batch_limit": None,
        "test_read": False, "test_used_for_selection": False,
        "parameter_audit": "disabled; proxy-only gradients observed on same training graph",
        "selected_steps_each_epoch": list(SELECTED_STEPS),
        "proxy_update_observation": "every step; detached reads only",
    })
    return original


def cpu_tree(value):
    if hasattr(value, "detach"):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(cpu_tree(item) for item in value)
    return value


class Observer:
    def __init__(self, namespace):
        self.ns = namespace
        self.sparse = False
        self.calls = []

    def bind(self, proxy, run_dir, rank, manifest, config):
        self.proxy, self.run_dir, self.rank = proxy, run_dir, rank
        self.manifest, self.config = manifest, config

    def begin(self, epoch, step):
        self.epoch, self.step = epoch, step
        self.sparse = step in SELECTED_STEPS
        self.calls, self.extra = [], {}

    def wrap_ghhc(self, original):
        def observed(distances, triplets, tau=.1, margin=.1, generator=None, return_details=False):
            state = generator.get_state().clone() if self.sparse and generator is not None else None
            result = original(distances, triplets, tau, margin, generator, return_details)
            if self.sparse:
                if not return_details:
                    raise RuntimeError("Diagnostic expects the original detail-returning GHHC call")
                self.calls.append({
                    "distances": distances.detach(), "triplets": triplets.detach(),
                    "generator_before": state, "tau": tau, "margin": margin,
                    "details": result[2], "loss": result[0].detach(), "stats": result[1],
                })
            return result
        return observed

    def loss_ready(self, stats, proxy, hier_mu, values, ids, relation_seed, whole_count, child_count, weight):
        if not self.sparse:
            return
        torch = self.ns["torch"]
        components = stats.pop("_components")
        p = proxy.tangent_proxies
        u = p.detach() / p.detach().norm(dim=-1, keepdim=True)
        component_gradients = {}
        for name in ("sample_loss", "proxy_loss"):
            gradient, = torch.autograd.grad(weight * components[name], p, retain_graph=True)
            component_gradients[name] = gradient.detach()
        if len(self.calls) != 2:
            raise RuntimeError("Expected exactly sample and proxy GHHC calls")
        self.extra = {
            "stage": "same_actual_training_forward_before_update", "epoch": self.epoch, "step": self.step,
            "relation_seed": relation_seed, "hier_weight": weight,
            "whole_count": whole_count, "child_count": child_count,
            "ids": ids, "labels": values["gold"], "hier_mu": hier_mu,
            "native_mu": values["mu"], "native_nu": values["nu"],
            "proxy_before": p, "proxy_ball_before": proxy.proxies(),
            "sample_ghhc": self.calls[0], "proxy_ghhc": self.calls[1],
            "weighted_component_gradients": component_gradients,
            "weighted_component_radial_gradients": {
                key: (gradient * u).sum(-1) for key, gradient in component_gradients.items()},
            "weighted_component_tangent_gradient_norms": {
                key: (gradient - (gradient * u).sum(-1, keepdim=True) * u).norm(dim=-1)
                for key, gradient in component_gradients.items()},
        }
        # Freeze before the actual optimizer mutates this parameter in-place.
        if self.rank == 0:
            self.extra = cpu_tree(self.extra)

    def before_step(self, proxy, optimizer):
        if self.rank != 0:
            return None
        before = self.ns["capture_before"](proxy.tangent_proxies, optimizer)
        if self.sparse:
            parts = self.extra["weighted_component_gradients"]
            direct_sum = parts["sample_loss"] + parts["proxy_loss"]
            self.extra["weighted_component_sum_vs_actual_synced_gradient_max_abs"] = float(
                (direct_sum - before["gradient"].detach().cpu()).abs().max())
        return before

    def after_step(self, before, proxy, optimizer):
        if self.rank != 0:
            self.calls, self.extra = [], {}
            return
        torch = self.ns["torch"]
        report, snapshot = self.ns["audit_after_step"](
            before, proxy.tangent_proxies, optimizer, include_snapshot=self.sparse)
        per_proxy = report.pop("_per_proxy")
        # Exact official mapping has a 1e-5 forward-cap denominator.
        mapped_before = self.ns["proxy_depth"](self.ns["proxy_to_ball"](before["parameter"], hook=False))
        mapped_after = self.ns["proxy_depth"](self.ns["proxy_to_ball"](proxy.tangent_proxies.detach(), hook=False))
        report.update(epoch=self.epoch, step=self.step, selected_snapshot=self.sparse,
                      mapped_depth_before_median=float(mapped_before.median()),
                      mapped_depth_after_median=float(mapped_after.median()))
        # All 512 rows are recorded each step, separately from compact summaries.
        rows = {key: per_proxy[key].tolist() for key in (
            "before_norm", "after_norm", "signed_radial_gradient", "gradient_norm",
            "actual_radial_step", "actual_angular_step", "exact_norm_change", "finite_step_excess")}
        rows["tangent_gradient_norm"] = (
            per_proxy["gradient_norm"].square() - per_proxy["signed_radial_gradient"].square()).clamp_min(0).sqrt().tolist()
        rows.update(epoch=self.epoch, step=self.step,
                    mapped_depth_before=mapped_before.detach().cpu().tolist(),
                    mapped_depth_after=mapped_after.detach().cpu().tolist())
        self.ns["append_jsonl"](self.run_dir / "proxy_cause_steps.jsonl", report)
        self.ns["append_jsonl"](self.run_dir / "proxy_cause_per_proxy.jsonl", rows)
        if report["all"]["prediction_error_max_coordinate"]["max"] > 2e-6:
            raise RuntimeError("AdamW accounting differs from the actual update by >2e-6")
        if report["state_invariants"]["step_after"] != report["state_invariants"]["expected_step_after"]:
            raise RuntimeError("AdamW step counter differs from accounting")
        if self.sparse:
            payload = {"epoch": self.epoch, "step": self.step,
                       "source": self.manifest["diagnostic_source"], "config": self.config,
                       "actual_optimizer_step": snapshot, "per_proxy": per_proxy,
                       "same_forward": self.extra, "step_summary": report}
            name = f"proxy_cause_e{self.epoch:03d}_s{self.step:03d}.pth"
            path = self.run_dir / name
            self.ns["atomic_torch"](path, payload)
            self.ns["append_jsonl"](self.run_dir / "proxy_cause_snapshots.jsonl", {
                "epoch": self.epoch, "step": self.step, "file": name,
                "sha256": self.ns["checkpoint_sha"](path), "bytes": path.stat().st_size})
        self.calls, self.extra = [], {}


def build_namespace(source=None):
    source = ORIGINAL.read_text(encoding="utf-8") if source is None else source
    transformed = transformed_source(source)
    # Preserve the source module's relative and historical absolute imports.
    for path in (REPO, REPO / "inter_hierarchy_MN40"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    namespace = {"__name__": "v9_proxy_cap_cause_diagnostic",
                 "__file__": str(ORIGINAL),
                 "__package__": "inter_hierarchy_MN40.hier_proxy_scratch_v9"}
    exec(compile(transformed, str(ORIGINAL) + "[bounded-diagnostic]", "exec"), namespace)
    from optimizer_step_audit import capture_before, after_step
    geometry = importlib.import_module("inter_hierarchy_MN40.hier_proxy_scratch_v8.geometry")
    loss_module = importlib.import_module("inter_hierarchy_MN40.hier_proxy_scratch_v8.hier_loss")
    observer = Observer(namespace)
    original_class = namespace["HIERLoss"]

    class ObservedHIERLoss(original_class):
        def forward(self, *args, **kwargs):
            if observer.sparse:
                kwargs["return_components"] = True
            return super().forward(*args, **kwargs)

    loss_module.ghhc_loss = observer.wrap_ghhc(loss_module.ghhc_loss)
    identity = {"original_relative_path": str(ORIGINAL.relative_to(REPO)),
                "original_normalized_sha256": sha_text(source),
                "transformed_normalized_sha256": sha_text(transformed),
                "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "optimizer_audit_sha256": hashlib.sha256((HERE / "optimizer_step_audit.py").read_bytes()).hexdigest(),
                "teacher": None, "source_checkpoint": None,
                "training_forward": "unchanged original V9; observations on same forward",
                "test_file_access": "forbidden: only load_shards(train) retained in main"}
    namespace.update(diagnostic_arguments=diagnostic_arguments,
        diagnostic_configuration=diagnostic_configuration,
        diagnostic_source_identity=lambda: identity,
        DIAGNOSTIC_EXPECTED_INITIAL_MODEL=EXPECTED_INITIAL_MODEL,
        observer=observer, HIERLoss=ObservedHIERLoss,
        capture_before=capture_before, audit_after_step=after_step,
        proxy_depth=geometry.depth, proxy_to_ball=geometry.to_ball)
    # Fail closed even if future code accidentally invokes an old evaluation path.
    def forbidden_evaluation(*_args, **_kwargs):
        raise RuntimeError("Evaluation/selection is forbidden in this bounded diagnostic")
    namespace["evaluate_clean"] = namespace["rank0_evaluation"] = forbidden_evaluation
    namespace["select_best"] = forbidden_evaluation
    original_load = namespace["load_shards"]
    def train_shards_only(data_dir, split):
        if split != "train":
            raise RuntimeError("Only training shards may be opened by this diagnostic")
        return original_load(data_dir, split)
    namespace["load_shards"] = train_shards_only
    return namespace


if __name__ == "__main__":
    build_namespace()["main"]()
