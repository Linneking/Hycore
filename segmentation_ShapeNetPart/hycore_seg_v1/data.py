"""Read-only, ID-preserving ShapeNetPart HDF5 adapter."""

from pathlib import Path

import h5py
import numpy as np
from torch.utils.data import Dataset


class ShapeNetPart(Dataset):
    def __init__(self, root, split, num_points=2048, limit=None):
        if split not in {"train", "val", "test"}:
            raise ValueError(split)
        self.path = Path(root) / (split + ".h5")
        self.split, self.num_points = split, num_points
        self._handle = None
        with h5py.File(self.path, "r") as f:
            self.categories = f["label"][:].reshape(-1).astype(np.int64)
            self.ids = f["ids"].asstr()[:].tolist()
            if f["data"].shape[1] < num_points:
                raise ValueError("Prepared point count below requested input count")
        self.length = min(len(self.ids), limit) if limit else len(self.ids)

    def __len__(self):
        return self.length

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_handle"] = None
        return state

    def close(self):
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def __del__(self):
        handle = getattr(self, "_handle", None)
        if handle is not None:
            handle.close()

    def __getitem__(self, index):
        if self._handle is None:
            self._handle = h5py.File(self.path, "r")
        points = self._handle["data"][index, :self.num_points].copy()
        parts = self._handle["pid"][index, :self.num_points].astype(np.int64)
        # Pinned DGCNN training augmentation: point-order permutation only.
        if self.split == "train":
            permutation = np.random.permutation(self.num_points)
            points, parts = points[permutation], parts[permutation]
        return points, int(self.categories[index]), parts, self.ids[index]
