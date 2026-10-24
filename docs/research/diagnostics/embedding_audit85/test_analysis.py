"""Small mathematical invariants, independent of model/cache implementation."""
import importlib.util
from pathlib import Path
import unittest
import json
import tempfile
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

    def test_flat_shell_noise_is_not_a_strict_hierarchy(self):
        depths=np.array([4.6,4.6-1e-7,4.6+1e-7])
        triples=np.array([[0,1,2]])
        result=a.order_statistics(depths,depths,triples,np.array([1]),np.array([0]),proxy_endpoints=True)
        for item in result["domains"]["all"]["tolerance"].values():
            self.assertEqual(item["pair_shallower_than_both_endpoints"],0.)
            self.assertEqual(item["pair_triple_depth_tied"],1.)

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

    def test_fractional_or_private_identity_labels_are_rejected(self):
        with self.assertRaises(ValueError): a.integer_vector(np.array([1.,2.5]),"IDs")
        with self.assertRaises(ValueError):
            a.observation_meta({"key":"ok","run_id":"/private/path","stage":"best","epoch":1})

    def test_small_complete_manifest_keeps_ids_private_and_h0_proxy_absent(self):
        rng=np.random.RandomState(12)
        labels=np.repeat(np.arange(40),2)
        mu=a.unit_directions(rng.randn(80,6))*.8
        proxy=a.unit_directions(rng.randn(24,6))*1.5
        view=a.unit_directions(mu)*(np.tanh(np.sqrt(.1)*4.6/2)/np.sqrt(.1))
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
            self.assertNotIn(str(root),json.dumps(result))
            self.assertNotIn("canonical_ids",result["observations"][0])
            with (root/"out/class_pair_angles.csv").open(encoding="utf-8-sig") as stream:
                self.assertEqual(sum(1 for _ in stream),1+2*780)


if __name__=="__main__": unittest.main()
