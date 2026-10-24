"""Original PointNet2 FPS, compiled with NVRTC and launched on a Torch stream.

This adapter replaces only the extension loading boundary. The CUDA device
source is copied verbatim from the repository kernel, including its origin
filter, first index, loop ordering and thread reduction tie behavior. It is
intended for frozen evaluation; other pointnet2 operations are not provided.

The private runtime directory receives PTX, compile logs and identity JSON.
No DLL/PTX is stored beside this source, and no CUDA device is selected until
the caller explicitly constructs ``OriginalFPS`` or installs the adapter.
"""
from __future__ import annotations

import ctypes as C
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import types


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
ORIGINAL = REPO / "pointnet2_ops_lib/pointnet2_ops/_ext-src/src/sampling_gpu.cu"
ORIGINAL_SHA = "a1ec92aabdb6e4ab3ad0b982ebb28fa79b1b1fc95e3accbfaa511ef9bc317fe4"
KERNEL_SHA = "0aefe583b1548ea099f458b7039f27484278d1a84c09a88887ce87aee703da0b"
BLOCK_SIZES = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)


def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def verified_source():
    original = ORIGINAL.read_text(encoding="utf-8").replace("\r\n", "\n")
    if sha_bytes(original.encode()) != ORIGINAL_SHA:
        raise RuntimeError("Repository FPS source differs from the reviewed identity")
    start = original.index("__device__ void __update")
    end = original.index("\nvoid furthest_point_sampling_kernel_wrapper", start)
    extracted = original[start:end].rstrip() + "\n"
    source = (HERE / "fps_kernel.cu").read_text(encoding="utf-8").replace("\r\n", "\n")
    if source != extracted or sha_bytes(source.encode()) != KERNEL_SHA:
        raise RuntimeError("NVRTC device kernel is not a verbatim copy of original FPS")
    return source


def inspect_idle_gpu(index=0):
    """Guard the start of a local evaluation job, before CUDA allocation."""
    output = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid,name,utilization.gpu,memory.used,memory.free",
         "--format=csv,noheader,nounits"], text=True, timeout=15)
    rows = []
    for line in output.splitlines():
        fields = [part.strip() for part in line.split(",")]
        if len(fields) != 6:
            continue
        i, uuid, name, utilization, used, free = fields
        rows.append(dict(index=int(i), uuid=uuid, name=name, utilization=int(utilization),
                         used_MiB=int(used), free_MiB=int(free)))
    row = next((item for item in rows if item["index"] == index), None)
    if row is None or row["utilization"] != 0 or row["used_MiB"] > 128 or row["free_MiB"] < 2048:
        raise RuntimeError("Requested local GPU is not idle; no allocation made: " + repr(row))
    applications = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"],
        text=True, timeout=15)
    if any(line.split(",")[0].strip() == row["uuid"] for line in applications.splitlines()):
        raise RuntimeError("Requested local GPU has a compute process; no allocation made")
    return row


class OriginalFPS:
    """A callable implementing ``furthest_point_sample(xyz, npoint)``."""

    def __init__(self, runtime_dir, *, device_index=0, require_idle=True):
        import torch

        self.runtime_dir = Path(runtime_dir).resolve()
        self.runtime_dir.mkdir(parents=True, exist_ok=True)
        self.device_index = int(device_index)
        self.observed_gpu = inspect_idle_gpu(self.device_index) if require_idle else None
        if torch.cuda.is_initialized() and require_idle:
            raise RuntimeError("Install FPS before any CUDA allocation, so idle status can be verified")
        source = verified_source()
        torch.cuda.set_device(self.device_index)
        torch.cuda.init()
        self.stream = torch.cuda.current_stream(self.device_index)
        major, minor = torch.cuda.get_device_capability(self.device_index)
        self.arch = f"compute_{major}{minor}"
        torch_lib = Path(torch.__file__).resolve().parent / "lib"
        self._dll_dir = os.add_dll_directory(str(torch_lib))
        self.nvrtc_path = torch_lib / "nvrtc64_120_0.dll"
        if not self.nvrtc_path.is_file():
            raise FileNotFoundError("Torch's matching NVRTC DLL is unavailable")
        self.nvrtc = C.WinDLL(str(self.nvrtc_path))
        self.driver = C.WinDLL("nvcuda.dll")
        self._declare()
        version_major, version_minor = C.c_int(), C.c_int()
        self._rtc_check(self.nvrtc.nvrtcVersion(C.byref(version_major), C.byref(version_minor)))
        self.nvrtc_version = [version_major.value, version_minor.value]
        self.options = [f"--gpu-architecture={self.arch}", "--std=c++14", "--fmad=true"]
        self.expressions = {block: f"&furthest_point_sampling_kernel<{block}>" for block in BLOCK_SIZES}
        identity_key = sha_bytes(json.dumps(dict(kernel=KERNEL_SHA, options=self.options,
                                                 nvrtc=self.nvrtc_version), sort_keys=True).encode())
        self.ptx_path = self.runtime_dir / ("original_fps_" + identity_key[:20] + ".ptx")
        self.names_path = self.ptx_path.with_suffix(".names.json")
        if self.ptx_path.is_file() and self.names_path.is_file():
            self.names = {int(key): value for key, value in json.loads(
                self.names_path.read_text(encoding="utf-8")).items()}
        else:
            self._compile(source)
        self._module = C.c_void_p()
        ptx = C.create_string_buffer(self.ptx_path.read_bytes())
        self._driver_check(self.driver.cuModuleLoadData(C.byref(self._module), ptx))
        self.functions = {}
        for block in BLOCK_SIZES:
            function = C.c_void_p()
            self._driver_check(self.driver.cuModuleGetFunction(C.byref(function), self._module,
                                                              self.names[block].encode()))
            self.functions[block] = function
        self.identity = dict(
            format="hycore85-original-fps-nvrtc-1", original_source_relative=str(ORIGINAL.relative_to(REPO)),
            original_source_normalized_sha256=ORIGINAL_SHA, kernel_normalized_sha256=KERNEL_SHA,
            adapter_sha256=sha_bytes(Path(__file__).read_bytes()), source_verbatim=True,
            compiler="NVRTC from installed Torch; no C++ host compiler or Torch extension ABI",
            nvrtc_version=self.nvrtc_version, compile_options=self.options,
            ptx_sha256=sha_bytes(self.ptx_path.read_bytes()), torch_version=torch.__version__,
            device_name=torch.cuda.get_device_name(self.device_index),
            compute_capability=[major, minor], idle_observation=self.observed_gpu,
            operation="original evaluation FPS only, no backward", no_server_access=True)
        (self.runtime_dir / "fps_runtime_identity.json").write_text(
            json.dumps(self.identity, indent=2), encoding="utf-8")

    def _declare(self):
        r = self.nvrtc
        r.nvrtcGetErrorString.argtypes = [C.c_int]
        r.nvrtcGetErrorString.restype = C.c_char_p
        r.nvrtcVersion.argtypes = [C.POINTER(C.c_int), C.POINTER(C.c_int)]
        r.nvrtcCreateProgram.argtypes = [C.POINTER(C.c_void_p), C.c_char_p, C.c_char_p,
                                       C.c_int, C.c_void_p, C.c_void_p]
        r.nvrtcAddNameExpression.argtypes = [C.c_void_p, C.c_char_p]
        r.nvrtcCompileProgram.argtypes = [C.c_void_p, C.c_int, C.POINTER(C.c_char_p)]
        r.nvrtcGetProgramLogSize.argtypes = [C.c_void_p, C.POINTER(C.c_size_t)]
        r.nvrtcGetProgramLog.argtypes = [C.c_void_p, C.c_void_p]
        r.nvrtcGetPTXSize.argtypes = [C.c_void_p, C.POINTER(C.c_size_t)]
        r.nvrtcGetPTX.argtypes = [C.c_void_p, C.c_void_p]
        r.nvrtcGetLoweredName.argtypes = [C.c_void_p, C.c_char_p, C.POINTER(C.c_char_p)]
        r.nvrtcDestroyProgram.argtypes = [C.POINTER(C.c_void_p)]
        d = self.driver
        d.cuGetErrorName.argtypes = [C.c_int, C.POINTER(C.c_char_p)]
        d.cuModuleLoadData.argtypes = [C.POINTER(C.c_void_p), C.c_void_p]
        d.cuModuleGetFunction.argtypes = [C.POINTER(C.c_void_p), C.c_void_p, C.c_char_p]
        d.cuLaunchKernel.argtypes = [C.c_void_p, C.c_uint, C.c_uint, C.c_uint,
                                    C.c_uint, C.c_uint, C.c_uint, C.c_uint, C.c_void_p,
                                    C.POINTER(C.c_void_p), C.c_void_p]
        for function in (r.nvrtcVersion, r.nvrtcCreateProgram, r.nvrtcAddNameExpression,
                         r.nvrtcCompileProgram, r.nvrtcGetProgramLogSize, r.nvrtcGetProgramLog,
                         r.nvrtcGetPTXSize, r.nvrtcGetPTX, r.nvrtcGetLoweredName,
                         r.nvrtcDestroyProgram, d.cuGetErrorName, d.cuModuleLoadData,
                         d.cuModuleGetFunction, d.cuLaunchKernel):
            function.restype = C.c_int

    def _rtc_check(self, result):
        if result:
            raise RuntimeError("NVRTC: " + self.nvrtc.nvrtcGetErrorString(result).decode())

    def _driver_check(self, result):
        if result:
            value = C.c_char_p()
            self.driver.cuGetErrorName(result, C.byref(value))
            raise RuntimeError("CUDA Driver: " + (value.value.decode() if value.value else str(result)))

    def _compile(self, source):
        program = C.c_void_p()
        self._rtc_check(self.nvrtc.nvrtcCreateProgram(C.byref(program), source.encode(),
                                                    b"fps_kernel.cu", 0, None, None))
        try:
            for expression in self.expressions.values():
                self._rtc_check(self.nvrtc.nvrtcAddNameExpression(program, expression.encode()))
            options = (C.c_char_p * len(self.options))(*(option.encode() for option in self.options))
            result = self.nvrtc.nvrtcCompileProgram(program, len(options), options)
            log_size = C.c_size_t()
            self._rtc_check(self.nvrtc.nvrtcGetProgramLogSize(program, C.byref(log_size)))
            log = C.create_string_buffer(log_size.value)
            self._rtc_check(self.nvrtc.nvrtcGetProgramLog(program, log))
            (self.runtime_dir / "fps_compile.log").write_text(log.value.decode(errors="replace"),
                                                            encoding="utf-8")
            self._rtc_check(result)
            self.names = {}
            for block, expression in self.expressions.items():
                lowered = C.c_char_p()
                self._rtc_check(self.nvrtc.nvrtcGetLoweredName(program, expression.encode(), C.byref(lowered)))
                self.names[block] = lowered.value.decode()
            size = C.c_size_t()
            self._rtc_check(self.nvrtc.nvrtcGetPTXSize(program, C.byref(size)))
            ptx = C.create_string_buffer(size.value)
            self._rtc_check(self.nvrtc.nvrtcGetPTX(program, ptx))
            self.ptx_path.write_bytes(ptx.raw)
            self.names_path.write_text(json.dumps(self.names, indent=2), encoding="utf-8")
        finally:
            self.nvrtc.nvrtcDestroyProgram(C.byref(program))

    def __call__(self, xyz, npoint):
        import torch

        if not isinstance(xyz, torch.Tensor) or xyz.ndim != 3 or xyz.shape[-1] != 3:
            raise ValueError("FPS expects [B,N,3]")
        if xyz.device.type != "cuda" or xyz.dtype != torch.float32 or not xyz.is_contiguous():
            raise ValueError("Original FPS requires contiguous CUDA float32 coordinates")
        if xyz.device.index != self.device_index:
            raise ValueError("FPS input belongs to another device")
        if not isinstance(npoint, int) or isinstance(npoint, bool) or npoint < 0:
            raise ValueError("npoint must be a nonnegative integer")
        batch, count, _ = xyz.shape
        if min(batch, count) < 1:
            raise ValueError("FPS expects nonempty batches and point sets")
        output = torch.zeros((batch, npoint), dtype=torch.int32, device=xyz.device)
        if npoint == 0:
            return output
        temporary = torch.full((batch, count), 1e10, dtype=torch.float32, device=xyz.device)
        block = min(512, 1 << (int(count).bit_length() - 1))
        stream = torch.cuda.current_stream(xyz.device)
        values = (C.c_int(batch), C.c_int(count), C.c_int(npoint),
                  C.c_void_p(xyz.data_ptr()), C.c_void_p(temporary.data_ptr()), C.c_void_p(output.data_ptr()))
        parameters = (C.c_void_p * len(values))(*(C.cast(C.pointer(value), C.c_void_p) for value in values))
        self._driver_check(self.driver.cuLaunchKernel(self.functions[block], batch, 1, 1, block, 1, 1,
                                                      0, C.c_void_p(stream.cuda_stream), parameters, None))
        # The operation is external to Torch's dispatcher; preserve allocator
        # lifetime semantics on the actual stream used for the kernel launch.
        xyz.record_stream(stream)
        temporary.record_stream(stream)
        output.record_stream(stream)
        return output


def install_pointnet2_fps(runtime_dir, *, device_index=0, require_idle=True):
    """Install the FPS-only module before importing the frozen PointMLP model."""
    if "pointnet2_ops" in sys.modules or "pointnet2_ops.pointnet2_utils" in sys.modules:
        raise RuntimeError("Install the explicit FPS adapter before importing pointnet2_ops")
    runtime = OriginalFPS(runtime_dir, device_index=device_index, require_idle=require_idle)
    package = types.ModuleType("pointnet2_ops")
    module = types.ModuleType("pointnet2_ops.pointnet2_utils")
    module.furthest_point_sample = runtime
    module.runtime_identity = runtime.identity
    package.pointnet2_utils = module
    sys.modules["pointnet2_ops"] = package
    sys.modules["pointnet2_ops.pointnet2_utils"] = module
    return runtime
