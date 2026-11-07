"""Run one original preparation stage with larger lossless HDF5 chunk caches."""
import runpy
import sys
import h5py

MODULES = {
    "segmentation_ShapeNetPart.hycore_seg_v1.prepare_data",
    "segmentation_ShapeNetPart.hycore_seg_v1.prepare_antao_protocol",
}


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in MODULES:
        raise SystemExit("Choose one of the two original preparation modules")
    module = sys.argv[1]
    sys.argv = [module, *sys.argv[2:]]
    original_file = h5py.File

    class CachedFile(original_file):
        def __init__(self, *args, **kwargs):
            self._read_datasets = {}
            kwargs.setdefault("rdcc_nbytes", 64 * 1024 * 1024)
            kwargs.setdefault("rdcc_nslots", 10007)
            super().__init__(*args, **kwargs)

        def __getitem__(self, name):
            if name in self._read_datasets:
                return self._read_datasets[name]
            result = super().__getitem__(name)
            if self.mode == "r" and isinstance(result, h5py.Dataset):
                self._read_datasets[name] = result
            return result

        def close(self):
            self._read_datasets.clear()
            super().close()

    h5py.File = CachedFile
    try:
        runpy.run_module(module, run_name="__main__")
    finally:
        h5py.File = original_file


if __name__ == "__main__":
    main()

