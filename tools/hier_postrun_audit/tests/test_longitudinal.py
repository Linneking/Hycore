"""Audit invariants: missing time, identities and optimized versus monitored use."""
import json
from pathlib import Path
import tempfile
import unittest
from tools.hier_postrun_audit.longitudinal import analyze_longitudinal, concentration


def run(records, epochs=None):
    if epochs is None:
        epochs = [{"epoch": r[0], "proxy_steps": 2, "metrics": {"lambda_hier": .1}} for r in records]
    return {"run_id": "H20", "audit_run_key": "V6-HIER64-qualified", "display_name": "V6-HIER64",
            "epochs": epochs, "proxy_usage": [{"epoch": epoch, "component": "sample", "domain": "noncollision",
            "role": "combined", "proxy_ids": ids, "counts": counts} for epoch, ids, counts in records]}


class LongitudinalContracts(unittest.TestCase):
    def view(self, analysis, name="all_observed"):
        return analyze_longitudinal([analysis], make_plots=False)["runs"][0]["groups"][0]["views"][name]

    def test_dormancy_is_not_bridged_over_missing_epochs(self):
        observed = self.view(run([(1, [10, 20], [1, 0]), (2, [10, 20], [0, 0]),
                                  (5, [10, 20], [0, 0]), (6, [10, 20], [1, 0])]))
        proxy = observed["proxies"][0]
        self.assertEqual(proxy["max_contiguous_observed_zero_epochs"], 1)
        self.assertEqual(proxy["observed_reactivation_count"], 0)
        self.assertEqual(len(observed["segments"]), 2)
        self.assertIsNone(observed["per_epoch"][2]["newly_active_count"])
        self.assertTrue(proxy["zero_spans"][0]["right_censored"])
        self.assertTrue(proxy["zero_spans"][1]["left_censored"])

    def test_contiguous_reactivation_requires_prior_observed_use(self):
        observed = self.view(run([(1, [10, 20], [0, 0]), (2, [10, 20], [2, 0]),
                                  (3, [10, 20], [0, 0]), (4, [10, 20], [0, 0]),
                                  (5, [10, 20], [1, 0])]))
        proxy = observed["proxies"][0]
        self.assertEqual(proxy["observed_reactivation_count"], 1)
        self.assertEqual(proxy["first_observed_use_epoch"], 2)
        self.assertEqual(proxy["last_observed_use_epoch"], 5)
        self.assertEqual(observed["never_used_in_observed_window_count"], 1)

    def test_monitored_warmup_does_not_establish_optimized_lifetime(self):
        records = [(1, [10, 20], [8, 0]), (2, [10, 20], [0, 1])]
        epochs = [{"epoch": 1, "proxy_steps": 0, "metrics": {"lambda_hier": 0}},
                  {"epoch": 2, "proxy_steps": 2, "metrics": {"lambda_hier": .1}}]
        analysis = run(records, epochs)
        all_observed = self.view(analysis)
        optimized = self.view(analysis, "optimization_observed")
        self.assertEqual(all_observed["cumulative_concentration"]["used_proxy_count"], 2)
        self.assertEqual(optimized["cumulative_concentration"]["used_proxy_count"], 1)
        self.assertIsNone(optimized["proxies"][0]["first_observed_use_epoch"])

    def test_proxy_id_order_is_aligned_and_not_position_assumed(self):
        observed = self.view(run([(1, [10, 20], [2, 0]), (2, [20, 10], [0, 3])]))
        self.assertEqual(observed["proxies"][0]["cumulative_observed_selections"], 5)
        self.assertEqual(observed["proxies"][1]["cumulative_observed_selections"], 0)

    def test_invalid_identity_changes_are_unavailable_not_fake_stability(self):
        result = analyze_longitudinal([run([(1, [10, 20], [2, 0]), (2, [10, 30], [1, 3])])], make_plots=False)
        self.assertFalse(result["runs"][0]["groups"][0]["available"])
        self.assertTrue(result["warnings"])

    def test_zero_selections_have_undefined_concentration(self):
        self.assertIsNone(concentration([0, 0])["gini"])
        self.assertIsNone(concentration([0, 0])["top1_share"])
        self.assertEqual(concentration([0, 0])["effective_proxy_count"], 0)
        self.assertAlmostEqual(concentration([1, 1])["gini"], 0)
        self.assertAlmostEqual(concentration([2, 0])["gini"], .5)

    def test_no_proxy_baseline_is_unavailable_and_outputs_cannot_be_overwritten(self):
        analysis = {"audit_run_key": "V6-B64-qualified", "epochs": [], "proxy_usage": []}
        with tempfile.TemporaryDirectory() as root:
            result = analyze_longitudinal([analysis], root, make_plots=False)
            self.assertFalse(result["runs"][0]["available"])
            self.assertNotEqual(result["runs"][0]["reason"], "")
            source = Path(root) / "longitudinal_summary.json"
            before = source.read_bytes()
            with self.assertRaises(FileExistsError):
                analyze_longitudinal([analysis], root, make_plots=False)
            self.assertEqual(source.read_bytes(), before)

    def test_unknown_optimizer_state_is_not_guessed_from_joint_phase(self):
        analysis = run([(1, [10, 20], [2, 0])], [{"epoch": 1, "phase": "joint", "metrics": {"lambda_hier": .1}}])
        self.assertFalse(self.view(analysis, "optimization_observed")["available"])


    def test_graphs_render_for_valid_and_invalid_identity_groups(self):
        valid = run([(1, [10, 20], [2, 0]), (2, [10, 20], [1, 2])])
        invalid = run([(1, [10, 20], [2, 0]), (2, [10, 30], [1, 3])])
        invalid["audit_run_key"] = "invalid_group"
        with tempfile.TemporaryDirectory() as root:
            result = analyze_longitudinal([valid, invalid], root)
            self.assertEqual(len(result["figures"]), 1)
            png = Path(root) / result["figures"][0]["png"]
            self.assertGreater(png.stat().st_size, 15000)
            self.assertTrue((Path(root) / result["figures"][0]["svg"]).is_file())


if __name__ == "__main__":
    unittest.main()
