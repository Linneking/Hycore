"""Checks actual-count pooling and separation of reported baseline identities."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location("v9_lr_eps_comparison", Path(__file__).with_name("compare.py"))
compare = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(compare)


class ComparisonAggregationTests(unittest.TestCase):
    def test_hinge_pooling_excludes_collisions_and_weights_actual_counts(self):
        rows = []
        for step, n, valid, active in ((1, 10, 5, 3), (2, 30, 30, 0)):
            stats = dict(triplets=n, noncollision_triplets=valid, collisions=n-valid,
                active_triplets=active, active_i_hinges=active, active_j_hinges=0, active_k_hinges=0)
            rows.append(dict(epoch=1, step=step, ce=float(step), hier=dict(sample_loss=float(step),
                proxy_loss=float(step)*2, sample=stats, proxy=stats)))
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as name:
            path = Path(name) / "steps.jsonl"
            path.write_text("".join(json.dumps(r)+"\n" for r in rows)+"{", encoding="utf-8")
            result = compare.aggregate_learning(path)[0]
        self.assertEqual(result["steps"], 2)
        self.assertEqual(result["scalar_means"]["ce"], 1.5)
        data = result["components"]["sample"]
        self.assertAlmostEqual(data["active_triplet_fraction_noncollision"], 3/35)
        self.assertAlmostEqual(data["active_hinge_fraction_noncollision"], 3/105)
        self.assertEqual(data["loss_mean_over_updates"], 1.5)

    def test_historical_cap_counts_are_rejected_as_reference(self):
        data = dict(proxy_count=512, paired_steps=2000,
            epochs=[dict(epoch=e, end_over_cap=n) for e,n in ((1,196),(5,378),(10,502))])
        with self.assertRaisesRegex(ValueError, "report71 fresh baseline"):
            compare.validate_baseline(data)
        data["epochs"] = [dict(epoch=e, end_over_cap=n) for e,n in ((1,229),(5,444),(10,502))]
        compare.validate_baseline(data)


if __name__ == "__main__":
    unittest.main()
