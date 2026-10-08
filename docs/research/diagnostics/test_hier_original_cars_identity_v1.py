"""Identity tests use synthetic MATs and tiny arbitrary image-byte fixtures."""
import argparse
import os
import tempfile
import unittest
from pathlib import Path

from scipy.io import savemat

import hier_original_cars_identity_v1 as m

TEMP_ROOT = os.environ.get("HIER_TEST_TMP")


def row(name, cls=1, bbox=(1, 2, 3, 4)):
    return {"relative_path": name, "class_original_1based": cls, "class_zero_based": cls-1,
            "bbox_x1": bbox[0], "bbox_y1": bbox[1], "bbox_x2": bbox[2], "bbox_y2": bbox[3],
            "key": (cls, *bbox)}


def mat_record(name, fname=False, cls=1, bbox=(1, 2, 3, 4)):
    return {"fname" if fname else "relative_im_path": name, "class": cls,
            "bbox_x1": bbox[0], "bbox_y1": bbox[1], "bbox_x2": bbox[2], "bbox_y2": bbox[3]}


class CarsIdentityTests(unittest.TestCase):
    def test_reordered_filenames_follow_class_and_bbox(self):
        with tempfile.TemporaryDirectory(prefix="hier_cars_identity_", dir=TEMP_ROOT) as d:
            a, b = Path(d)/"old002.jpg", Path(d)/"old001.jpg"
            a.write_bytes(b"a"); b.write_bytes(b"b")
            targets = [row("car_ims/000001.jpg", bbox=(2,3,4,5)), row("car_ims/000002.jpg")]
            sources = [dict(row("old001.jpg"), source_path=str(b)),
                       dict(row("old002.jpg", bbox=(2,3,4,5)), source_path=str(a))]
            rows, mapping, unresolved = m.build_join(targets, sources)
            self.assertEqual(mapping["car_ims/000001.jpg"], str(a))
            self.assertEqual(mapping["car_ims/000002.jpg"], str(b))
            self.assertFalse(unresolved)

    def test_equal_filename_never_overrides_wrong_identity(self):
        target = row("car_ims/000001.jpg", cls=1)
        source = dict(row("car_ims/000001.jpg", cls=2), source_path="unused")
        rows, mapping, unresolved = m.build_join([target], [source])
        self.assertFalse(mapping)
        self.assertEqual(rows[0]["status"], "no_annotation_match")

    def test_ambiguity_requires_byte_equivalence(self):
        with tempfile.TemporaryDirectory(prefix="hier_cars_identity_", dir=TEMP_ROOT) as d:
            a, b = Path(d)/"a.jpg", Path(d)/"b.jpg"
            a.write_bytes(b"different a"); b.write_bytes(b"different b")
            sources = [dict(row("a.jpg"), source_path=str(a)), dict(row("b.jpg"), source_path=str(b))]
            rows, mapping, unresolved = m.build_join([row("car_ims/000001.jpg")], sources)
            self.assertFalse(mapping)
            self.assertEqual(rows[0]["status"], "ambiguous_forward_match")
            self.assertEqual(len(unresolved), 1)

    def test_identical_candidate_bytes_record_equivalence(self):
        with tempfile.TemporaryDirectory(prefix="hier_cars_identity_", dir=TEMP_ROOT) as d:
            a, b = Path(d)/"a.jpg", Path(d)/"b.jpg"
            a.write_bytes(b"same image bytes"); b.write_bytes(b"same image bytes")
            sources = [dict(row("a.jpg"), source_path=str(a)), dict(row("b.jpg"), source_path=str(b))]
            rows, mapping, unresolved = m.build_join([row("car_ims/000001.jpg")], sources)
            self.assertEqual(rows[0]["status"], "byte_equivalent")
            self.assertTrue(rows[0]["byte_equivalent"])
            self.assertEqual(len(mapping), 1)
            self.assertFalse(unresolved)

    def test_missing_files_are_excluded_even_in_planning_mode(self):
        with tempfile.TemporaryDirectory(prefix="hier_cars_identity_", dir=TEMP_ROOT) as d:
            source = dict(row("absent.jpg"), source_path=str(Path(d)/"absent.jpg"))
            rows, mapping, unresolved = m.build_join([row("car_ims/000001.jpg")], [source], allow_missing=True)
            self.assertFalse(mapping)
            self.assertEqual(rows[0]["status"], "annotation_only_missing_file")
            self.assertEqual(len(unresolved), 1)

    def test_path_escape_rejected(self):
        for name in ("../outside.jpg", "C:/secret.jpg", "/etc/passwd"):
            with self.assertRaises(ValueError):
                m.safe_relative(name)

    def test_original_and_split_mat_fields(self):
        with tempfile.TemporaryDirectory(prefix="hier_cars_identity_", dir=TEMP_ROOT) as d:
            canonical, split = Path(d)/"canonical.mat", Path(d)/"split.mat"
            savemat(canonical, {"annotations": [mat_record("car_ims/000001.jpg")]})
            savemat(split, {"annotations": [mat_record("00001.jpg", fname=True)]})
            self.assertEqual(m.mat_rows(canonical)[0]["key"], m.mat_rows(split)[0]["key"])
            self.assertEqual(m.mat_rows(split)[0]["relative_path"], "00001.jpg")

    def test_end_to_end_canonical_cli_contract(self):
        with tempfile.TemporaryDirectory(prefix="hier_cars_identity_", dir=TEMP_ROOT) as d:
            d = Path(d)
            image_dir = d/"car_ims"; image_dir.mkdir()
            (image_dir/"000002.jpg").write_bytes(b"original bytes")
            target, canonical = d/"provided.mat", d/"canonical.mat"
            savemat(target, {"annotations": [mat_record("car_ims/000001.jpg")]})
            savemat(canonical, {"annotations": [mat_record("car_ims/000002.jpg")]})
            args = argparse.Namespace(provided_mat=str(target), canonical_mat=str(canonical), images_root=str(d),
                                      fastai_root=None, train_mat=None, test_mat=None, allow_missing_images=False, output=str(d/"output"))
            payload = m.run(args)
            self.assertEqual(payload["mapping"]["car_ims/000001.jpg"], str(image_dir/"000002.jpg"))
            self.assertEqual(payload["summary"]["resolved_existing_image_rows"], 1)
            with self.assertRaises(FileExistsError):
                m.run(args)


if __name__ == "__main__":
    unittest.main()
