"""Geometry and read-only aggregate audit contract checks (CPU only)."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np

spec = importlib.util.spec_from_file_location('angle_audit', Path(__file__).with_name('audit.py'))
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class AuditTests(unittest.TestCase):
    def test_pair_lorentz_midpoint(self):
        theta = np.radians(np.array([0., 10., 30., 90., 180.]))
        rho = np.sqrt(.1) * 4.6
        first = np.array([np.cosh(rho), np.sinh(rho), 0.])
        second = np.column_stack((np.full(len(theta), np.cosh(rho)),
                                  np.sinh(rho)*np.cos(theta), np.sinh(rho)*np.sin(theta)))
        sums = first + second
        scale = np.sqrt(sums[:,0]**2 - np.sum(sums[:,1:]**2, axis=1))
        midpoint = sums / scale[:,None]
        expected = np.arccosh(np.maximum(midpoint[:,0], 1)) / np.sqrt(.1)
        np.testing.assert_allclose(audit.pair_depth(np.cos(theta)), expected, atol=1e-7)

    def test_triple_degenerate_origin(self):
        angles = np.radians([0, 120, 240])
        dirs = np.stack([np.cos(angles), np.sin(angles)], axis=1)[None]
        self.assertLess(float(audit.triple_depth(dirs)[0]), 1e-7)

    def test_class_balance_and_cache_immutable(self):
        rng = np.random.default_rng(1)
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            path = Path(directory) / 'cache.npz'
            np.savez(path, mu=rng.normal(size=(15, 6)), labels=np.repeat([0,1,2], [3,5,7]))
            before = path.read_bytes()
            result = audit.analyze_npz({'path':str(path),'key':'synthetic','mode':'eval','car_class':9}, 20261022)
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(result['classes'], 3)
            self.assertEqual(result['within_macro']['cosine']['count'], 34)
            self.assertEqual(result['between_class_pair_balanced']['cosine']['count'], 96)
            self.assertIsNone(result['modelnet40_car_class'])
            self.assertEqual(result['sameclass_pair_otherclass_k_reference']['count'], 192)
            json.dumps(result)

    def test_rotation_and_triple_bound(self):
        rng=np.random.default_rng(22)
        n=audit.unit(rng.normal(size=(300,3,10)))
        orth=np.linalg.qr(rng.normal(size=(10,10)))[0]
        a=audit.triple_depth(n)
        b=audit.triple_depth(n@orth)
        np.testing.assert_allclose(a,b,atol=1e-12)
        for i,j in ((0,1),(0,2),(1,2)):
            pair=audit.pair_depth(np.einsum('nd,nd->n',n[:,i],n[:,j]))
            self.assertTrue(np.all(a<=pair+1e-10))


if __name__=='__main__':
    unittest.main()
