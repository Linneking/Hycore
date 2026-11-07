"""Contract tests for portable content verification; no GPU or full dataset needed."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np

SPEC = importlib.util.spec_from_file_location("seg_data_identity", Path(__file__).parents[1] / "data_identity.py")
identity = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(identity)


class ContentVerificationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.path = self.root / "test.h5"
        self.write_fixture("lzf")
        self.expected = identity.content_identity(self.path)
        self.profile = {"original_manifest_sha256": "historical_manifest",
                        "splits": {"test": self.expected}}
        self.write_manifest()

    def tearDown(self):
        self.temp.cleanup()

    def write_fixture(self, compression, dtype="<f4"):
        with h5py.File(self.path, "w") as file:
            file.create_dataset("data", data=np.arange(2 * 2048 * 3).reshape(2, 2048, 3).astype(dtype),
                                compression=compression)
            file.create_dataset("pid", data=np.zeros((2, 2048), dtype="<i2"), compression=compression)
            file.create_dataset("label", data=np.arange(2, dtype="<i8"))
            file.create_dataset("ids", data=np.array(["shape/a", "shape/b"], dtype=object),
                                dtype=h5py.string_dtype("utf-8"))

    def write_manifest(self):
        (self.root / "manifest.json").write_text(json.dumps({
            "complete": True, "created": "new timestamp", "splits": {
                "test": {"objects": 2, "file_sha256": identity.file_sha256(self.path)}}}))

    def test_serialization_and_metadata_changes_are_accepted(self):
        old_file_hash = identity.file_sha256(self.path)
        self.write_fixture("gzip")
        with h5py.File(self.path, "a") as file:
            file.attrs["created"] = "different metadata"
        self.write_manifest()
        self.assertNotEqual(old_file_hash, identity.file_sha256(self.path))
        result = identity.verify_prepared(self.root, self.profile, ["test"])
        self.assertEqual(result["splits"]["test"], self.expected["content_sha256"])

    def test_coordinate_part_category_and_id_changes_are_rejected(self):
        for field in ("data", "pid", "label", "ids"):
            with self.subTest(field=field):
                self.write_fixture("lzf")
                with h5py.File(self.path, "a") as file:
                    if field == "data":
                        file[field][0, 0, 0] += 1
                    elif field == "pid":
                        file[field][0, 0] += 1
                    elif field == "label":
                        file[field][0] += 1
                    else:
                        file[field][0] = "shape/changed"
                self.write_manifest()
                with self.assertRaises(ValueError):
                    identity.verify_prepared(self.root, self.profile, ["test"])

    def test_point_and_object_order_changes_are_rejected(self):
        for scope in ("point", "object"):
            self.write_fixture("lzf")
            with h5py.File(self.path, "a") as file:
                values = file["data"][:]
                file["data"][:] = values[:, ::-1] if scope == "point" else values[::-1]
            self.write_manifest()
            with self.assertRaises(ValueError):
                identity.verify_prepared(self.root, self.profile, ["test"])

    def test_unrecorded_file_change_is_rejected(self):
        with h5py.File(self.path, "a") as file:
            file["data"][0, 0, 0] += 1
        with self.assertRaises(ValueError):
            identity.verify_prepared(self.root, self.profile, ["test"])

    def test_wrong_coordinate_dtype_is_rejected(self):
        for dtype in ("<f8", ">f4"):
            with self.subTest(dtype=dtype):
                self.write_fixture("lzf", dtype=dtype)
                with self.assertRaises(ValueError):
                    identity.content_identity(self.path)

    def test_unknown_checkpoint_profile_is_rejected(self):
        with self.assertRaises(ValueError):
            identity.reference_for("unknown_original_manifest")


if __name__ == "__main__":
    unittest.main()

