"""Checkpoint-selection and split/mapping contracts; never initialize CUDA."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

from tools.hier_postrun_audit.extract import (checkpoint_epoch, choose_checkpoints,
                                            reset_inference_rng, resolve_proxy_mapping, validate_training_ids)


class ExtractContractTests(unittest.TestCase):
    def _run(self, root, manifest, epochs, aliases):
        run = root / "run"
        run.mkdir()
        (run / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        for epoch in epochs:
            (run / ("checkpoint_epoch_%03d.pth" % epoch)).touch()
        for name in aliases:
            (run / name).touch()
        def load(path):
            return aliases[path.name]
        return run, load

    def test_standard_uses_saved_milestones_and_real_best_and_last_aliases(self):
        with tempfile.TemporaryDirectory() as folder:
            aliases = {"initialization_e0.pth": {"initialization_only": True},
                       "best.pth": {"epoch": 187, "completed_epochs": 187},
                       "last.pth": {"completed_epochs": 300, "best": {"epoch": 187}}}
            run, loader = self._run(Path(folder), {"status": "completed", "completed_epochs": 300,
                "best": {"epoch": 187}, "training_config": {"warmup_epochs": 0}},
                [1, 5, 20, 40, 60, 80, 99, 100, 120, 140, 160, 180, 200, 240, 300], aliases)
            chosen, manifest = choose_checkpoints(run, checkpoint_loader=loader)
            self.assertEqual([epoch for epoch, _ in chosen], [0, 1, 5, 20, 40, 99, 100, 160, 187, 200, 300])
            self.assertEqual(dict(chosen)[187].name, "best.pth")
            self.assertEqual(dict(chosen)[300].name, "last.pth")
            self.assertTrue(manifest["manifest_epoch_aliases_agree"])

    def test_stale_manifest_cannot_relabel_the_actual_best_or_last_models(self):
        with tempfile.TemporaryDirectory() as folder:
            aliases = {"best.pth": {"model_selection_only": True, "best": {"epoch": 99}},
                       "last.pth": {"epoch": 200, "completed_epochs": 200, "best": {"epoch": 99}}}
            run, loader = self._run(Path(folder), {"completed_epochs": 300, "best": {"epoch": 231},
                "training_config": {"warmup_epochs": 20}}, [20, 40, 100, 200], aliases)
            chosen, manifest = choose_checkpoints(run, checkpoint_loader=loader)
            self.assertEqual([epoch for epoch, _ in chosen], [20, 40, 99, 100, 200])
            self.assertFalse(manifest["manifest_epoch_aliases_agree"])
            self.assertEqual(manifest["resolved_checkpoint_aliases"], {"last": 200, "best": 99})

    def test_all_checkpoint_policy_obeys_explicit_budget(self):
        with tempfile.TemporaryDirectory() as folder:
            run, loader = self._run(Path(folder), {}, [20, 40, 60], {})
            with self.assertRaisesRegex(ValueError, "budget"):
                choose_checkpoints(run, policy="all", maximum=2, checkpoint_loader=loader)

    def test_epoch_aliases_reject_ambiguous_or_conflicting_state(self):
        self.assertEqual(checkpoint_epoch({"initialization_only": True}), 0)
        self.assertEqual(checkpoint_epoch({"epoch": 187, "best": {"epoch": 100}}, "best"), 187)
        with self.assertRaisesRegex(ValueError, "disagree"):
            checkpoint_epoch({"epoch": 200, "completed_epochs": 300})
        with self.assertRaisesRegex(ValueError, "unambiguous"):
            checkpoint_epoch({"best": {"epoch": 99}}, "last")

    def test_inspected_live_mapping_does_not_enable_shadow_clip(self):
        for version in ("hycore-hier-v5-h20-1", "hycore-hier-v6-h20-selfk300-1", "hycore-hier-v7-w0-projected-proxy-1"):
            saved = {"format": version, "proxy": {"tangent_proxies": object()},
                     "training_config": {"c": 1., "extra_HIER_tangent_cap": False, "clip_r": 2.3}}
            self.assertEqual(resolve_proxy_mapping(saved), {"numeric_radius_fraction": .999})

    def test_unknown_mapping_and_enabled_implicit_cap_are_rejected(self):
        saved = {"format": "unknown", "proxy": {"tangent_proxies": object()},
                 "training_config": {"c": 1., "extra_HIER_tangent_cap": False}}
        with self.assertRaisesRegex(ValueError, "Unknown"):
            resolve_proxy_mapping(saved)
        saved["format"] = "hycore-hier-v5-h20-1"
        saved["training_config"]["extra_HIER_tangent_cap"] = True
        saved["training_config"]["clip_r"] = 2.3
        with self.assertRaisesRegex(ValueError, "explicit"):
            resolve_proxy_mapping(saved)
        mapping = {"numeric_radius_fraction": .999, "source_tangent_cap": 2.3, "source_cap_epsilon": 1e-5}
        saved["training_config"]["proxy_mapping"] = mapping
        self.assertEqual(resolve_proxy_mapping(saved), mapping)
        self.assertIsNone(resolve_proxy_mapping({"format": "baseline", "net": {}}))

    def test_existing_training_ids_must_match_exact_saved_split(self):
        validate_training_ids({"train_ids": [1, 3, 5]}, [1, 3, 5])
        validate_training_ids({"initialization_only": True}, [1, 3, 5])
        with self.assertRaisesRegex(ValueError, "fixed source split"):
            validate_training_ids({"train_ids": [1, 3, 7]}, [1, 3, 5])
        with self.assertRaisesRegex(ValueError, "fixed source split"):
            validate_training_ids({"train_ids": [5, 3, 1]}, [1, 3, 5])

    def test_each_snapshot_inference_rng_is_reset_on_cpu_cuda_and_numpy(self):
        fake_torch, fake_numpy = Mock(), Mock()
        reset_inference_rng(fake_torch, fake_numpy, 22)
        fake_torch.manual_seed.assert_called_once_with(22)
        fake_torch.cuda.manual_seed_all.assert_called_once_with(22)
        fake_numpy.random.seed.assert_called_once_with(22)


    def test_export_cannot_create_files_inside_the_source_even_when_called_directly(self):
        from tools.hier_postrun_audit.extract import extract_features
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder) / "source"
            run.mkdir()
            (run / "manifest.json").write_text('{"status":"completed"}')
            (run / "metrics.jsonl").touch()
            with self.assertRaisesRegex(ValueError, "outside"):
                extract_features(run, run / "audit", Path(folder), 0)
            self.assertFalse((run / "audit").exists())


if __name__ == "__main__":
    unittest.main()
