"""Small mathematical invariants, independent of model/cache implementation."""
import importlib.util
from pathlib import Path
import unittest
import json
import tempfile
import csv
import numpy as np

spec=importlib.util.spec_from_file_location("audit85_analysis",Path(__file__).with_name("analyze.py"))
a=importlib.util.module_from_spec(spec); spec.loader.exec_module(a)


class GeometryContracts(unittest.TestCase):
    def test_depth_matches_expmap_inverse_for_two_curvatures(self):
        direction=np.array([[1.,0.,0.],[0.,1.,0.]])
        target=np.array([2.2,4.6])
        for c in (1.,.1):
            points=direction*(np.tanh(np.sqrt(c)*target/2)/np.sqrt(c))[:,None]
            np.testing.assert_allclose(a.origin_depth(points,c),target,atol=1e-12)

    def test_angles_rotation_and_individual_radius_invariant(self):
        rng=np.random.RandomState(6); x=rng.randn(12,7)
        q,_=np.linalg.qr(rng.randn(7,7))
        base=a.unit_directions(x); moved=a.unit_directions((x@q)*np.exp(rng.randn(12,1)))
        np.testing.assert_allclose(a.angle_degrees(base,base),a.angle_degrees(moved,moved),atol=2e-6)

    def test_centroid_does_not_reintroduce_radius(self):
        x=np.array([[1.,0.],[0.,1.],[1.,1.],[2.,1.]])
        labels=np.array([0,0,1,1])
        first=a.class_centers(a.unit_directions(x),labels)[1]
        second=a.class_centers(a.unit_directions(x*np.array([100,.01,3,8])[:,None]),labels)[1]
        np.testing.assert_allclose(first,second,atol=1e-14)

    def test_directional_cone_radii_and_gap_have_known_units(self):
        theta=np.radians([-2.,2.,38.,42.])
        unit=np.stack((np.cos(theta),np.sin(theta)),axis=1)
        labels=np.array([0,0,1,1])
        cones=a.class_cones(unit,labels)
        self.assertAlmostEqual(cones[0]["p90"],2.,places=10)
        self.assertAlmostEqual(cones[1]["p95"],2.,places=10)
        meta={"observation_id":"fixture__best","stage":"best","run_id":"fixture"}
        summary,tables,_=a.analyze_angles(unit,labels,["a","b"],meta,{(0,1):"test"})
        pair,=tables[1]
        self.assertAlmostEqual(pair["centroid_angle"],40.,places=10)
        self.assertAlmostEqual(pair["cone_gap_p90"],36.,places=10)
        self.assertAlmostEqual(pair["cone_gap_p95"],36.,places=10)
        self.assertEqual(summary["class_pairs_with_negative_cone_gap_p90"],0)

    def test_overlapping_percentile_cones_do_not_imply_classifier_failure(self):
        theta=np.radians([-20.,20.,-10.,30.])
        unit=np.stack((np.cos(theta),np.sin(theta)),axis=1)
        labels=np.array([0,0,1,1])
        summary,tables,_=a.analyze_angles(unit,labels,["a","b"],{}, {})
        self.assertLess(tables[1][0]["cone_gap_p90"],0.)
        self.assertEqual(summary["class_pairs_with_negative_cone_gap_p90"],1)
        self.assertIn("not a classifier boundary",summary["class_cone_limit"])

    def test_flat_shell_noise_is_not_a_strict_hierarchy(self):
        depths=np.array([4.6,4.6-1e-7,4.6+1e-7])
        triples=np.array([[0,1,2]])
        result=a.order_statistics(depths,depths,triples,np.array([1]),np.array([0]),proxy_endpoints=True)
        for item in result["domains"]["all"]["tolerance"].values():
            self.assertEqual(item["pair_shallower_than_both_endpoints"],0.)
            self.assertEqual(item["pair_triple_depth_tied"],1.)

    def test_nonendpoint_orders_use_their_explicit_filtered_denominator(self):
        depths=np.array([5.,5.,5.,3.,2.])
        triples=np.tile(np.array([[0,1,2]]),(4,1))
        pair=np.array([0,3,3,3])
        triple=np.array([4,2,4,3])
        result=a.order_statistics(depths,depths,triples,pair,triple,proxy_endpoints=True)
        self.assertEqual(result["domains"]["all"]["draws"],4)
        self.assertEqual(result["domains"]["noncollision"]["draws"],3)
        filtered=result["domains"]["noncollision_nonendpoint"]
        self.assertEqual(filtered["draws"],1)
        self.assertEqual(filtered["tolerance"]["0.001"]["all_three_orders_strict"],1.)
        self.assertEqual(result["pair_equals_an_endpoint_count"],1)
        self.assertEqual(result["triple_equals_an_endpoint_count"],1)
        separate=a.order_statistics(depths,depths,triples,pair,triple,proxy_endpoints=False)
        self.assertEqual(separate["domains"]["noncollision_nonendpoint"]["draws"],3)

    def test_distances_zero_diagonal_and_known_origin_radius(self):
        c=.1; x=np.array([[0.,0.],[.8,0.],[0.,.7]])
        d=a.poincare_distances(x,x,c,same=True)
        np.testing.assert_allclose(np.diag(d),0.,atol=0)
        np.testing.assert_allclose(d[0],a.origin_depth(x,c),atol=1e-12)
        np.testing.assert_allclose(d,d.T,atol=1e-12)

    def test_minimax_is_consistent_with_bruteforce_cost(self):
        distance=np.array([[1.,4.,2.],[3.,2.,5.],[2.,6.,1.]])
        rows=np.array([[0,1,2]])
        pair,triple=a.minimax_selection(distance,rows)
        self.assertEqual(int(pair[0]),0)
        self.assertEqual(int(triple[0]),0)

    def test_missing_usage_remains_unavailable(self):
        self.assertEqual(a.usage_summary(None),{"available":False})
        self.assertEqual(a.usage_summary(np.array([2,2,0]))["used_proxy_count"],2)

    def test_depth_breadth_shuffle_keeps_flat_caps_unavailable(self):
        depth=np.linspace(2.,4.,30)
        entropy=np.linspace(3.,0.,30)
        result=a.depth_breadth_association(depth,entropy)
        self.assertTrue(result["available"])
        self.assertAlmostEqual(result["spearman_depth_vs_nn32_entropy"],-1.)
        self.assertEqual(result["depth_shuffle_null"]["count"],64)
        self.assertGreater(result["depth_shuffle_null"]["p975"],result["depth_shuffle_null"]["p025"])
        self.assertFalse(a.depth_breadth_association(np.full(30,4.6),entropy)["available"])

    def test_depth_breadth_ranks_ignore_cap_jitter_and_keep_supported_two_levels(self):
        depths=np.r_[np.full(12,3.2),np.full(18,4.6)]
        entropy=np.linspace(3.,0.,30)
        first=a.depth_breadth_association(depths,entropy)
        jittered=depths.copy()
        jittered[12:]+=np.linspace(-2e-7,2e-7,18)
        second=a.depth_breadth_association(jittered,entropy)
        self.assertTrue(first["available"])
        self.assertTrue(second["available"])
        self.assertEqual(first["robust_depth_levels"],2)
        self.assertEqual(second["dominant_depth_group_count"],18)
        self.assertEqual(first["spearman_depth_vs_nn32_entropy"],second["spearman_depth_vs_nn32_entropy"])
        self.assertEqual(first["depth_shuffle_null"],second["depth_shuffle_null"])
        self.assertEqual(first["depth_rank_resolution"],.001)

    def test_one_off_cap_proxy_does_not_turn_shell_jitter_into_a_depth_pattern(self):
        depths=np.r_[3.8,4.6+np.linspace(-2e-7,2e-7,511)]
        entropy=np.arange(512,dtype=float)
        result=a.depth_breadth_association(depths,entropy)
        self.assertFalse(result["available"])
        self.assertEqual(result["robust_depth_levels"],2)
        self.assertEqual(result["dominant_depth_group_count"],511)
        self.assertEqual(result["non_dominant_proxy_count"],1)
        self.assertIn("insufficient depth replication",result["reason"])
        self.assertAlmostEqual(result["raw_spearman_descriptive_legacy"],1.)

    def test_fractional_or_private_identity_labels_are_rejected(self):
        with self.assertRaises(ValueError): a.integer_vector(np.array([1.,2.5]),"IDs")
        with self.assertRaises(ValueError):
            a.observation_meta({"key":"ok","run_id":"/private/path","stage":"best","epoch":1})

    def test_small_complete_manifest_keeps_ids_private_and_h0_proxy_absent(self):
        rng=np.random.RandomState(12)
        labels=np.repeat(np.arange(40),2)
        mu=a.unit_directions(rng.randn(80,6))*.8
        proxy=a.unit_directions(rng.randn(24,6))*1.5
        view=(a.unit_directions(mu)*(np.tanh(np.sqrt(.1)*4.6/2)/np.sqrt(.1))).astype(np.float32)
        workspace_temp_parent=Path(__file__).resolve().parent
        with tempfile.TemporaryDirectory(prefix="_audit85_contract_",dir=workspace_temp_parent) as temporary:
            root=Path(temporary)
            self.assertEqual(root.resolve().parent,workspace_temp_parent)
            np.savez(root/"h0.npz",native_mu=mu,labels=labels,canonical_ids=np.arange(80))
            np.savez(root/"h1.npz",native_mu=mu,labels=labels,canonical_ids=np.arange(80),hier_mu=view,proxy_ball=proxy)
            manifest={"observations":[
                {"key":"ts13","run_id":"TS13","stage":"best","epoch":223,"c_native":.1,"cache":"h0.npz"},
                {"key":"new_b05","run_id":"new_b05","stage":"best","epoch":273,"c_native":.1,"c_hier":.1,"cache":"h1.npz"}]}
            (root/"manifest.json").write_text(json.dumps(manifest),encoding="utf-8")
            a.main(["--manifest",str(root/"manifest.json"),"--out",str(root/"out"),"--expected-count","80","--sample-batches","1"])
            result=json.loads((root/"out/summary.json").read_text(encoding="utf-8"))
            self.assertEqual(result["status"],"completed")
            self.assertFalse(result["observations"][0]["proxies"]["available"])
            self.assertTrue(result["observations"][1]["proxies"]["available"])
            view_domain=result["observations"][1]["domains"]["hier_whole_view"]
            self.assertGreater(view_domain["constructed_shell_max_abs_depth_error"],1e-8)
            self.assertLess(view_domain["constructed_shell_max_abs_depth_error"],1e-3)
            self.assertGreater(view_domain["depth"]["std_population"],0.)
            self.assertEqual(view_domain["depth_distribution_policy"],
                             "constructed shell snapped to exact4.6 for hist/ECDF")
            with (root/"out/depth_cdf.csv").open(encoding="utf-8-sig",newline="") as stream:
                shell_rows=[row for row in csv.DictReader(stream)
                            if row["observation_id"]=="new_b05__best" and row["domain"]=="hier_whole_view"]
            at_shell=[row for row in shell_rows if abs(float(row["depth"])-4.6)<1e-10]
            self.assertEqual(len(at_shell),1)
            self.assertEqual(float(at_shell[0]["cumulative_percent"]),100.)
            self.assertTrue(all(float(row["cumulative_percent"])==0.
                                for row in shell_rows if float(row["depth"])<4.6-1e-10))
            self.assertTrue(all(float(row["cumulative_percent"])==100.
                                for row in shell_rows if float(row["depth"])>=4.6-1e-10))
            self.assertNotIn(str(root),json.dumps(result))
            self.assertNotIn("canonical_ids",result["observations"][0])
            with (root/"out/class_pair_angles.csv").open(encoding="utf-8-sig") as stream:
                self.assertEqual(sum(1 for _ in stream),1+2*780)


if __name__=="__main__": unittest.main()
