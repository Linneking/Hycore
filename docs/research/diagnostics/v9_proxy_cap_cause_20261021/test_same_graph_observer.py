"""A real HIER graph verifies that observation preserves the actual update."""
from __future__ import annotations

import sys
import math
import unittest

import torch

import runner

sys.path.insert(0, str(runner.REPO))
from inter_hierarchy_MN40.hier_proxy_scratch_v8 import hier_loss as loss_module
from inter_hierarchy_MN40.hier_proxy_scratch_v8.geometry import InverseMetric
from optimizer_step_audit import capture_before


class SameGraphObservation(unittest.TestCase):
    def test_same_forward_component_probes_preserve_gradients_adam_and_rng(self):
        torch.set_num_threads(2)
        fixture = torch.Generator().manual_seed(1234)
        whole_source = torch.randn(16, 8, generator=fixture)
        labels = torch.arange(16) // 4
        outputs = []
        original_ghhc = loss_module.ghhc_loss
        original_rng = torch.random.get_rng_state().clone()
        for observe in (False, True):
            whole = whole_source.clone().requires_grad_(True)
            shell_radius = math.tanh(math.sqrt(.1) * 4.6 / 2) / math.sqrt(.1)
            mu = InverseMetric.apply(whole / whole.norm(dim=-1, keepdim=True) * shell_radius, .1)
            proxy = loss_module.HIERLoss(num_proxies=24, dim=8, seed=22)
            optimizer = torch.optim.AdamW(proxy.parameters(), lr=.01, eps=1e-8, weight_decay=.01)
            generator = torch.Generator().manual_seed(5678)
            observer = runner.Observer({"torch": torch, "capture_before": capture_before})
            observer.bind(proxy, None, 0, {}, {})
            observer.begin(1, 1)
            try:
                if observe:
                    loss_module.ghhc_loss = observer.wrap_ghhc(original_ghhc)
                loss, stats = proxy(mu, labels, topk=6, proxy_topk=6, seed=99,
                    t_per_anchor=3, generator=generator, return_components=observe,
                    return_mining=True)
                if observe:
                    observer.loss_ready(stats, proxy, mu, {"gold": labels, "mu": mu, "nu": mu},
                        torch.arange(16), 99, 900, 400, .5)
                    self.assertIsNone(proxy.tangent_proxies.grad)
                    self.assertIsNone(whole.grad)
                (.5 * loss + .01 * whole.square().sum()).backward()
                gradient = proxy.tangent_proxies.grad.clone()
                if observe:
                    observer.before_step(proxy, optimizer)
                    self.assertLess(observer.extra[
                        "weighted_component_sum_vs_actual_synced_gradient_max_abs"], 2e-7)
                optimizer.step()
                outputs.append((proxy.tangent_proxies.detach().clone(), gradient, whole.grad.clone(),
                                generator.get_state().clone(), torch.random.get_rng_state().clone()))
            finally:
                loss_module.ghhc_loss = original_ghhc
        for unchanged, observed in zip(*outputs):
            torch.testing.assert_close(observed, unchanged, rtol=0, atol=0)
        torch.testing.assert_close(torch.random.get_rng_state(), original_rng, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
