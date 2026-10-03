"""CPU gates for V6 exclusion and observations that do not alter training."""
from pathlib import Path
import sys
from contextlib import nullcontext
import random
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import torch
from torch import nn

from inter_hierarchy_MN40.hier_proxy_scratch_v6.hier_loss import HIERLoss, expmap0_c1
from inter_hierarchy_MN40.hier_proxy_scratch_v6.structure_monitor import (
    StructureAccumulator, ancestor_summary, radial_angular_gradient,
    proxy_update_stats, preserved_eval, feature_snapshot, compare_feature_snapshots,
    run_augmentation_probe,
)
from inter_hierarchy_MN40.hier_proxy_scratch_v6 import parameter_audit


def test_exclusion_monitoring_preserves_actual_loss_gradients_and_rng():
    generator = torch.Generator().manual_seed(57)
    points = expmap0_c1(torch.randn(16, 4, generator=generator) * .2).requires_grad_(True)
    labels = torch.arange(8).repeat_interleave(2)
    hierarchy = HIERLoss(num_proxies=32, dim=4, seed=22)
    before = torch.get_rng_state().clone()
    choices_a, choices_b = torch.Generator().manual_seed(41), torch.Generator().manual_seed(41)
    loss_a, stats_a = hierarchy(points, labels, topk=6, generator=choices_a, monitor_structure=False)
    gradients_a = torch.autograd.grad(loss_a, (points, hierarchy.tangent_proxies))
    loss_b, stats_b = hierarchy(points, labels, topk=6, generator=choices_b, monitor_structure=True)
    gradients_b = torch.autograd.grad(loss_b, (points, hierarchy.tangent_proxies))
    assert torch.equal(loss_a, loss_b)
    assert all(torch.equal(a, b) for a, b in zip(gradients_a, gradients_b))
    assert torch.equal(before, torch.get_rng_state())
    assert torch.equal(choices_a.get_state(), choices_b.get_state())
    assert stats_a["sample"]["triplets"] > 0
    for component in ("sample", "proxy"):
        assert stats_b[component]["self_k_triplets"] == 0
        assert stats_b["structure"][component]["self_k_count"] == 0
        assert sum(stats_b["structure"][component]["selected_proxy_counts"]["all_draws/pair"]) == stats_b[component]["triplets"]
    try:
        hierarchy(points, labels, topk=6, exclude_self_negative=False)
    except ValueError:
        pass
    else:
        raise AssertionError("V6 accepted self-negative mining")


def test_selected_ancestor_counts_domains_and_epoch_pooling():
    points = expmap0_c1(torch.tensor([[.8, 0.], [.7, .1], [.9, .1]]))
    proxies = expmap0_c1(torch.tensor([[.2, 0.], [.6, 0.]]))
    triplets = torch.tensor([[0, 1, 2], [1, 2, 0], [2, 0, 1]])
    details = {"pair_proxy_idx": torch.tensor([1, 0, 0]),
               "triple_proxy_idx": torch.tensor([0, 0, 1]),
               "per_draw_loss": torch.tensor([.2, 0., .1])}
    report = ancestor_summary(points, proxies, triplets, details)
    assert report["selected_proxy_counts"]["all_draws/pair"] == [2, 1]
    assert report["selected_proxy_counts"]["noncollision/pair"] == [1, 1]
    assert report["noncollision_count"] == report["active_noncollision_count"] == 2
    assert report["moments"]["all_draws/pair_minus_triple_depth"]["positive_count"] == 1
    pooled = StructureAccumulator()
    pooled.update({"sample": report})
    pooled.update({"sample": report})
    result = pooled.summary()["sample"]
    assert result["draw_count"] == 6
    assert result["selected_ancestor_usage"]["all_draws/pair"]["counts"] == [4, 2]
    assert result["depth_distributions"]["noncollision/pair_depth"]["count"] == 4
    assert result["depth_distributions"]["all_draws/pair_min_endpoint_gap"]["positive_fraction"] == 1
    empty = ancestor_summary(points, proxies, triplets[:0], {k: v[:0] for k, v in details.items()})
    pooled.update({"sample": empty})
    assert pooled.summary()["sample"]["draw_count"] == 6


def test_radial_angular_gradient_and_actual_displacement():
    tangent = torch.tensor([[1., 0.], [0., 2.], [0., 0.]])
    gradient = torch.tensor([[2., 3.], [4., -5.], [6., 7.]])
    report = radial_angular_gradient(tangent, gradient)
    assert abs(report["radial_norm"] ** 2 - 29) < 1e-5
    assert abs(report["angular_norm"] ** 2 - 110) < 1e-5
    assert report["nonzero_proxy_gradient_count"] == 3
    change = torch.tensor([[.1, .2], [.3, -.4], [0., 0.]])
    update = proxy_update_stats(tangent, tangent + change)
    assert abs(update["radial_update_norm"] ** 2 - .17) < 1e-6
    assert abs(update["angular_update_norm"] ** 2 - .13) < 1e-6


def test_failed_eval_probe_restores_bn_modes_and_rng():
    model = nn.Sequential(nn.BatchNorm1d(3), nn.Dropout(.2))
    model.train()
    model[1].eval()
    buffers = [buffer.clone() for buffer in model.buffers()]
    modes = [module.training for module in model.modules()]
    torch_rng, python_rng, numpy_rng = torch.get_rng_state().clone(), random.getstate(), np.random.get_state()
    try:
        with preserved_eval(model):
            assert not any(module.training for module in model.modules())
            torch.rand(10)
            random.random()
            np.random.rand(10)
            model[0].running_mean.add_(3)
            raise RuntimeError("injected failed probe")
    except RuntimeError:
        pass
    assert modes == [module.training for module in model.modules()]
    assert all(torch.equal(a, b) for a, b in zip(buffers, model.buffers()))
    assert torch.equal(torch_rng, torch.get_rng_state()) and python_rng == random.getstate()
    actual_np = np.random.get_state()
    assert numpy_rng[0] == actual_np[0] and np.array_equal(numpy_rng[1], actual_np[1]) and numpy_rng[2:] == actual_np[2:]


class _Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(3, 4)
        self.bn = nn.BatchNorm1d(4)
        self.emb = nn.Linear(4, 4)
        self.classifier = nn.Linear(4, 8)

    def forward(self, cloud):
        features = self.bn(self.proj(cloud.mean(-1)))
        mu = expmap0_c1(self.emb(features) * .1)
        return mu, self.classifier(mu)


def test_independent_two_view_probe_and_identical_feature_comparison():
    backbone, hierarchy = _Backbone(), HIERLoss(num_proxies=32, dim=4)
    rng = np.random.default_rng(81)
    points = rng.normal(size=(24, 1024, 3)).astype(np.float32)
    labels, ids = np.arange(24) // 2, np.arange(24)
    before_rng = torch.get_rng_state().clone()
    buffers = [b.clone() for b in backbone.buffers()]
    report = run_augmentation_probe(backbone, hierarchy, points, labels, ids, "cpu", 17)
    assert report["optimizer_steps"] == 0 and report["BN_RNG_training_modes_restored"]
    assert 0 <= report["nearest_proxy_retention"] <= 1
    assert 0 <= report["nearest_four_proxy_jaccard_mean"] <= 1
    assert torch.equal(before_rng, torch.get_rng_state()) and backbone.training
    assert all(torch.equal(a, b) for a, b in zip(buffers, backbone.buffers()))
    mu = expmap0_c1(torch.randn(16, 4) * .2)
    snapshot = feature_snapshot(mu, hierarchy.proxies(), torch.arange(8).repeat_interleave(2), torch.arange(16), topk=6)
    comparison = compare_feature_snapshots(snapshot, snapshot)
    assert comparison["nearest_proxy_retention"] == comparison["nearest_four_proxy_jaccard_mean"] == 1
    assert comparison["pair_ancestor_retention"] == comparison["triple_ancestor_retention"] == 1


def test_parameter_audit_restores_input_bn_rng_and_does_not_accumulate_grad():
    class _Wrapper(nn.Module):
        def __init__(self, backbone):
            super().__init__()
            self.backbone = backbone

        def forward(self, cloud, whole_count, child_count):
            cloud.add_(1.)  # Reproduce the input-mutating original crop contract.
            nu, _ = self.backbone(cloud.transpose(1, 2))
            mu, logits = self.backbone(cloud.transpose(1, 2))
            return {"mu": mu, "nu": nu, "logits": logits}

    class _DDP:
        def __init__(self, backbone):
            self.module = _Wrapper(backbone)

        def no_sync(self):
            return nullcontext()

        def __call__(self, *args):
            return self.module(*args)

    def _base(out, labels):
        ce = nn.functional.cross_entropy(out["logits"], labels)
        return {**out, "gold": labels, "ce": ce,
                "base": ce + .01 * (out["mu"].square().mean() + out["nu"].square().mean())}

    backbone, hierarchy = _Backbone(), HIERLoss(num_proxies=32, dim=4)
    cloud = torch.randn(16, 8, 3)
    original = cloud.clone()
    labels = torch.arange(8).repeat_interleave(2)
    buffers = [buffer.clone() for buffer in backbone.buffers()]
    before_rng = torch.get_rng_state().clone()
    with patch.object(parameter_audit, "global_base_losses", _base), \
         patch.object(parameter_audit, "gather_without_grad", lambda value: value), \
         patch.object(parameter_audit.dist, "all_reduce", lambda *a, **kw: None), \
         patch.object(parameter_audit.dist, "broadcast", lambda *a, **kw: None), \
         patch.object(parameter_audit.dist, "get_world_size", lambda: 1):
        report = parameter_audit.audit_parameter_gradients(_DDP(backbone), hierarchy,
            cloud, labels, torch.arange(16), 8, 4, topk=6, seed=47)
    assert torch.equal(original, cloud) and torch.equal(before_rng, torch.get_rng_state())
    assert all(torch.equal(a, b) for a, b in zip(buffers, backbone.buffers()))
    assert all(p.grad is None for p in list(backbone.parameters()) + list(hierarchy.parameters()))
    for component in ("sample", "proxy"):
        assert report["proxy_parameter_components"][component]["self_k"]["actual_gradient_norm"] == 0
    assert report["groups"]["shared_encoder"]["weighted_hier_norm"] >= 0


if __name__ == "__main__":
    torch.set_num_threads(1)
    for name, test in list(globals().items()):
        if name.startswith("test_"):
            test()
            print(f"PASS {name}")
