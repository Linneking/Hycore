"""CPU checks for transition tracking, cohort weighting, and fixed-choice VJPs."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

import analyze
from optimizer_step_audit import capture_before, after_step
from inter_hierarchy_MN40.hier_proxy_scratch_v6.hier_loss import ghhc_loss


class AnalyzerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_first_passages_returns_and_weighted_dynamic_cohorts(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as name:
            root = Path(name)
            self.assertTrue(root.resolve().is_relative_to(Path(__file__).resolve().parent))
            states = [[2.2, 2.2, 2.4], [2.4, 2.4, 2.2], [2.1, 2.2, 2.5], [2.4, 2.1, 2.4]]
            summaries, rows = [], []
            for s in range(3):
                before, after = np.asarray(states[s]), np.asarray(states[s + 1])
                delta = after - before
                row = {"epoch": 1, "step": s + 1, "before_norm": before.tolist(),
                    "after_norm": after.tolist(), "signed_radial_gradient": [-.1, .1, 0.],
                    "gradient_norm": [.2] * 3, "tangent_gradient_norm": [.1] * 3,
                    "actual_radial_step": (delta - .001).tolist(),
                    "actual_angular_step": [.01] * 3, "exact_norm_change": delta.tolist(),
                    "finite_step_excess": [.001] * 3,
                    "mapped_depth_before": np.minimum(2 * before, 4.6).tolist(),
                    "mapped_depth_after": np.minimum(2 * after, 4.6).tolist()}
                summary = {"epoch": 1, "step": s + 1, "all": {}, "by_before_cap": {}}
                for cohort, mask in (("all", np.ones(3, dtype=bool)),
                                     ("inside", before <= 2.3), ("outside", before > 2.3)):
                    item = {"count": int(mask.sum())}
                    for key in analyze.STEP_COMPONENTS:
                        item[key] = {"mean": [1., 9., 1.][s]}
                    item["prediction_error_max_coordinate"] = {"max": 0.}
                    if cohort == "all":
                        summary["all"] = item
                    else:
                        summary["by_before_cap"][cohort] = item
                summaries.append(summary)
                rows.append(row)
            for filename, data in (("proxy_cause_steps.jsonl", summaries), ("proxy_cause_per_proxy.jsonl", rows)):
                (root / filename).write_text("".join(json.dumps(r) + "\n" for r in data), encoding="utf-8")
            result = analyze.aggregate_steps(root)
            self.assertEqual((result["first_passages"], result["crossing_events"], result["return_events"]), (2, 4, 3))
            self.assertEqual((result["initial_over_cap"], result["ever_over_cap"], result["ever_returned"]), (1, 3, 3))
            pooled = result["pooled_optimizer_components"]["inside"]
            self.assertEqual(pooled["proxy_update_observations"], 5)
            self.assertAlmostEqual(pooled["means"]["current_tangent_g_step"], 2.6)
            self.assertEqual(result["epochs"][0]["unique_returned_proxies"], 3)
            # All exported trajectory data are aggregate; no lineage arrays remain.
            serialized = json.dumps(result, allow_nan=False)
            self.assertNotIn("before_norm", serialized)

    def test_sparse_actual_graph_and_fixed_choice_reconstruction(self):
        gen = torch.Generator().manual_seed(224)
        p = torch.nn.Parameter(torch.randn(8, 4, generator=gen) * .8)
        mu = analyze.to_ball(torch.randn(6, 4, generator=gen), hook=False).detach()
        sample_triplets = torch.tensor([[i, (i + 1) % 6, (i + 3) % 6] for i in range(6)] * 8)
        proxy_triplets = torch.tensor([[i, (i + 1) % 8, (i + 4) % 8] for i in range(8)] * 8)
        calls, gradients, losses = {}, {}, []
        for name, triplets in (("sample", sample_triplets), ("proxy", proxy_triplets)):
            mapped = analyze.to_ball(p)
            d = analyze.official_distance(mu if name == "sample" else mapped, mapped)
            loss, stats, detail = ghhc_loss(d, triplets, generator=gen, return_details=True)
            g, = torch.autograd.grad(.5 * loss, p, retain_graph=True)
            gradients[name + "_loss"] = g.detach().clone()
            calls[name + "_ghhc"] = {"distances": d.detach().clone(), "triplets": triplets,
                "margin": .1, "tau": .1, "loss": loss.detach().clone(), "stats": stats,
                "details": {k: v.detach().clone() for k, v in detail.items()}}
            losses.append(.5 * loss)
        sum(losses).backward()
        optimizer = torch.optim.AdamW([p], lr=.01, weight_decay=.01)
        before = capture_before(p, optimizer)
        raw = p.detach().clone()
        optimizer.step()
        report, snapshot = after_step(before, p, optimizer, include_snapshot=True)
        payload = {"epoch": 1, "step": 1, "actual_optimizer_step": snapshot,
            "per_proxy": report.pop("_per_proxy"), "same_forward": {"proxy_before": raw,
                "hier_mu": mu, "hier_weight": .5, "weighted_component_gradients": gradients,
                "weighted_component_sum_vs_actual_synced_gradient_max_abs": 0., **calls}}
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as name:
            self.assertTrue(Path(name).resolve().is_relative_to(Path(__file__).resolve().parent))
            path = Path(name) / "snapshot.pth"
            torch.save(payload, path)
            result = analyze.analyze_snapshot(path)
        for component in result["components"].values():
            self.assertEqual(component["distances"]["max_abs_error"], 0.)
            self.assertAlmostEqual(component["unweighted_recorded_loss"],
                                   component["unweighted_fixed_active_scalar"], places=6)
            self.assertEqual(component["hinge_activity_disagreements_due_to_recomputation"], 0)
            self.assertLess(component["gradient_decomposition_residual_max_abs"], 1e-7)
            self.assertEqual(component["gradients"]["full_st"]["all"]["n"], 8)
        # Deliberately exercise recorded activity when a recomputed hinge crosses zero.
        distances = torch.tensor([[.1, .2], [.1, .2], [.4, .1]], requires_grad=True)
        fixed, scalar, disagreements = analyze.fixed_choice_loss(distances, {
            "triplets": torch.tensor([[0, 1, 2]]), "margin": .1,
            "details": {"pair_proxy_idx": torch.tensor([0]), "triple_proxy_idx": torch.tensor([1]),
                        "masked_hinges": torch.tensor([[1., 0., 0.]])}})
        self.assertGreaterEqual(disagreements, 0)
        g, = torch.autograd.grad(fixed, distances)
        torch.testing.assert_close(g[0], torch.tensor([1., -1.]))


if __name__ == "__main__":
    unittest.main()
