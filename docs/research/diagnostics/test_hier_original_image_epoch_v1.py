"""Identity and source-contract checks for fresh-image HIER epoch diagnostics."""
import argparse
import json
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch

from hier_original_activation_probe_v1 import official_runtime
from hier_original_image_epoch_v1 import (
    OriginalWhole, forward_images, read_identity_map, relative_image_path,
    resolve_paths, sampler_plan, state_fingerprint, unwrapped_state,
)
from hier_original_handoff_loader_v1 import safe_checkpoint

REFERENCE = Path(os.environ.get('HIER_REFERENCE', 'D:/Hycore/.codex-local/references/HIER-CVPR23'))


class ImageEpochTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        _, cls.hypnn, _, cls.sampler_cls = official_runtime(REFERENCE)

    def test_source_global_epoch_plan(self):
        labels = np.repeat(np.arange(100), 58)
        plan = sampler_plan(self.sampler_cls, labels, 17, 10)
        self.assertEqual(plan.shape, (len(labels) // 180, 180))
        np.testing.assert_array_equal(plan, sampler_plan(self.sampler_cls, labels, 17, 10))
        self.assertFalse(np.array_equal(plan, sampler_plan(self.sampler_cls, labels, 18, 10)))
        for batch in plan:
            classes, counts = np.unique(labels[batch], return_counts=True)
            self.assertEqual(len(classes), 90)
            self.assertTrue(np.all(counts == 2))

    def test_identity_is_explicit_and_confined(self):
        self.assertEqual(relative_image_path('/old/data/cars196/car_ims/000001.jpg', 'cars'),
                         'car_ims/000001.jpg')
        test_root = Path(os.environ.get('HIER_TEST_TEMP_ROOT', Path(__file__).resolve().parent)).resolve()
        with tempfile.TemporaryDirectory(dir=test_root) as directory:
            root = Path(directory)
            self.assertEqual(root.resolve().parent, test_root)
            (root / 'car_ims').mkdir()
            image = root / 'car_ims/000089.jpg'
            Image.new('RGB', (16, 16)).save(image)
            mapping = {'car_ims/000001.jpg': str(image)}
            path = root / 'identity.json'
            path.write_text(json.dumps({'mapping': mapping, 'unresolved': []}))
            self.assertEqual(read_identity_map(path), mapping)
            actual = resolve_paths(['car_ims/000001.jpg'], 'cars', root, mapping)
            self.assertEqual(actual, [image])
            with self.assertRaises(FileNotFoundError):
                resolve_paths(['car_ims/000001.jpg'], 'cars', root, {})
            with self.assertRaises(ValueError):
                resolve_paths(['car_ims/000001.jpg'], 'cars', root,
                              {'car_ims/000001.jpg': str(root.parent / 'outside.jpg')})

    def test_strict_architecture_frozen_state_and_microbatch(self):
        margs = argparse.Namespace(model='resnet50', bn_freeze=True, use_lastnorm=True,
                                   emb=512, hyp_c=.1, clip_r=2.3)
        torch.manual_seed(23)
        model = OriginalWhole(self.hypnn, margs).requires_grad_(False).train(True)
        self.assertEqual(len(model.state_dict()), 632)
        self.assertTrue(all(not m.training for m in model.modules()
                            if isinstance(m, torch.nn.BatchNorm2d)))
        self.assertTrue(model.training)
        fingerprint = state_fingerprint(model)
        images = torch.randn(2, 3, 32, 32)
        first = forward_images(model, images, torch.device('cpu'), 1)
        second = forward_images(model, images, torch.device('cpu'), 2)
        torch.testing.assert_close(first, second, atol=2e-6, rtol=2e-6)
        self.assertEqual(first.shape, (2, 512))
        self.assertEqual(state_fingerprint(model), fingerprint)
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))

    def test_supplied_best_strict_state_keys(self):
        supplied = Path(os.environ.get('HIER_BEST_CHECKPOINT',
            'D:/Hycore/原版HIER交接文档/logs/run_cub_r50/checkpoint_best.pth'))
        if not supplied.is_file():
            self.skipTest('Run this identity check with HIER_BEST_CHECKPOINT pointing to supplied best.')
        checkpoint = safe_checkpoint(supplied)
        model = OriginalWhole(self.hypnn, checkpoint['args'])
        result = model.load_state_dict(unwrapped_state(checkpoint['stduent']), strict=True)
        self.assertEqual(result.missing_keys, [])
        self.assertEqual(result.unexpected_keys, [])
        self.assertEqual(len(model.state_dict()), 632)


if __name__ == '__main__':
    unittest.main()
