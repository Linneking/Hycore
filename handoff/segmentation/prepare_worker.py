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

    def cached_file(*args, **kwargs):
        kwargs.setdefault("rdcc_nbytes", 64 * 1024 * 1024)
        kwargs.setdefault("rdcc_nslots", 10007)
        return original_file(*args, **kwargs)

    h5py.File = cached_file
    try:
        runpy.run_module(module, run_name="__main__")
    finally:
        h5py.File = original_file


if __name__ == "__main__":
    main()
