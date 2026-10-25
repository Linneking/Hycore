"""Meaningful CPU gates for paired eps/depth-distribution reporting."""
from __future__ import annotations
import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location("b05_eps_analysis_under_test",HERE/"analyze.py")
analysis=importlib.util.module_from_spec(spec);spec.loader.exec_module(analysis)


def fixture(directory,eps,epochs=2,steps=2):
    directory.mkdir()
    start=np.linspace(3.5,4.1,512)
    raw_start=start/2
    for epoch in range(epochs+1):
        # Large-eps fixture moves inward; small-eps has equal bidirectional
        # movement. Distribution means and sign counts are known independently.
        delta=np.full(512,-.02*epoch) if eps==1e-4 else np.r_[np.full(256,-.01*epoch),np.full(256,.01*epoch)]
        depth=start+delta;raw=raw_start+delta/2
        record=dict(epoch=epoch,eps=eps,proxy_depth=depth.tolist(),raw_tangent_norm=raw.tolist(),
                    cap_count=0,ever_cap_count=0,returned_ever_count=0,first_crossings_epoch=0,returns_epoch=0,
                    depth_delta_epoch=(delta/epoch if epoch else np.zeros(512)).tolist(),depth_delta_init=delta.tolist(),
                    raw_norm_delta_epoch=(delta/(2*epoch) if epoch else np.zeros(512)).tolist(),raw_norm_delta_init=(delta/2).tolist())
        (directory/("observer_epoch_%03d.json"%epoch)).write_text(json.dumps(record))
        if epoch:
            group=dict(weighted_hier_norm=.02,base_norm=.1,hier_vs_base=dict(norm_ratio=.2,cosine=-.1))
            (directory/("parameter_audit_epoch_%03d.json"%epoch)).write_text(json.dumps(dict(groups=dict(shared_encoder=group))))
    rows=[dict(epoch=epoch,step=step,lr=.005,eps=eps,proxy_gradient_norm=1e-4*step,
               actual_update_norm=.02 if eps==1e-8 else .002,sqrt_vhat_median=1e-5,
               eps_dominance_fraction=0. if eps==1e-8 else 1.,update_attenuation_median=.999 if eps==1e-8 else .091,
               fixed_moment_eps1e4_vs_eps1e8_update_norm_ratio=.091)
          for epoch in range(1,epochs+1) for step in range(1,steps+1)]
    (directory/"proxy_steps.jsonl").write_text("\n".join(json.dumps(r) for r in rows)+"\n")


class DistributionContracts(unittest.TestCase):
    def test_cap_does_not_hide_raw_motion_but_mapped_motion_is_zero(self):
        mapped=np.full(512,4.6)
        raw0=np.full(512,2.4);raw1=np.full(512,3.)
        self.assertEqual(analysis.movement(mapped-mapped)["stationary_percent"],100.)
        self.assertEqual(analysis.movement(raw1-raw0)["outward_percent"],100.)
        self.assertAlmostEqual(analysis.movement(raw1-raw0)["median"],.6)

    def test_percent_denominator_and_signed_tolerance(self):
        values=np.array([-.1,-1e-6,0,1e-6,.1])
        result=analysis.movement(values)
        self.assertEqual((result["inward_percent"],result["stationary_percent"],result["outward_percent"]),(20.,60.,20.))
        edges=analysis.centered_edges(values,.05,signed=True)
        share=analysis.histogram_percent(values,edges)
        self.assertAlmostEqual(float(share.sum()),100.)
        with self.assertRaises(ValueError):analysis.histogram_percent(values,np.array([-.01,.01]))

    def test_paired_fixture_exports_sign_and_exact_mean_without_private_paths(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);small=root/"small";large=root/"large"
            fixture(small,1e-8);fixture(large,1e-4)
            out=root/"public"
            analysis.main(["--small-run",str(small),"--large-run",str(large),"--out",str(out),"--expected-epochs","2","--expected-steps","2","--no-plot"])
            summary=json.loads((out/"summary.json").read_text())
            last=[r for r in summary["epochs"] if r["arm"]=="eps1e-4" and r["epoch"]==2][0]
            self.assertEqual(last["depth_delta_epoch_inward_percent"],100.)
            self.assertAlmostEqual(last["depth_delta_init_median"],-.04)
            self.assertAlmostEqual(last["step_proxy_gradient_norm_mean"],1.5e-4)
            self.assertAlmostEqual(last["backbone_hier_base_ratio"],.2)
            self.assertNotIn(str(root), (out/"summary.json").read_text())
            with (out/"distribution_histograms.csv").open(encoding="utf-8-sig",newline="") as stream:
                totals={}
                for row in csv.DictReader(stream):
                    key=(row["arm"],row["epoch"],row["domain"]);totals[key]=totals.get(key,0)+float(row["percent"])
            self.assertTrue(all(abs(v-100)<1e-8 for v in totals.values()))

    def test_missing_steps_wrong_cap_or_changed_initialization_fail(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);small=root/"small";large=root/"large"
            fixture(small,1e-8);fixture(large,1e-4)
            path=small/"proxy_steps.jsonl";original=path.read_text();path.write_text(original.splitlines()[0]+"\n")
            with self.assertRaises(ValueError):analysis.read_arm(small,"eps1e-8",2,2)
            path.write_text(original)
            epoch=large/"observer_epoch_000.json";value=json.loads(epoch.read_text());value["proxy_depth"][0]+=.01;epoch.write_text(json.dumps(value))
            with self.assertRaises(ValueError):analysis.main(["--small-run",str(small),"--large-run",str(large),"--out",str(root/"bad"),"--expected-epochs","2","--expected-steps","2","--no-plot"])

    def test_public_report_and_png_svg_are_generated(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);small=root/"small";large=root/"large"
            fixture(small,1e-8);fixture(large,1e-4)
            out=root/"public"
            analysis.main(["--small-run",str(small),"--large-run",str(large),"--out",str(out),"--expected-epochs","2","--expected-steps","2"])
            self.assertTrue((out/"report.md").is_file())
            self.assertEqual(len(list((out/"figures").glob("*.png"))),9)
            self.assertEqual(len(list((out/"figures").glob("*.svg"))),9)
            manifest=json.loads((out/"figures_manifest.json").read_text())
            for entry in manifest["figures"]:self.assertEqual(analysis.digest(out/entry["file"]),entry["sha256"])


if __name__=="__main__":unittest.main(verbosity=2)
