# Original PointNet2 FPS for local frozen evaluation

This adapter makes the repository's existing CUDA farthest-point sampler usable
on the local Windows Torch runtime without a Visual Studio toolchain or a Torch
C++ extension build. It does not replace the sampling algorithm with a Torch
`argmax` approximation.

`fps_kernel.cu` is a **verbatim device-source extraction** from
`pointnet2_ops_lib/pointnet2_ops/_ext-src/src/sampling_gpu.cu`, beginning at
`__update` and ending before the original host wrapper. Source equality and
normalized SHA256 are checked before use. It retains index0 initialization,
`mag <= 1e-3` filtering, float32 temporary distances initialized to1e10,
per-thread first-maximum behavior and the original reduction tie ordering.
The original host launch block-size rule (power of two up to512 threads) is
reproduced. NVRTC receives `--fmad=true`, consistent with the original default
CUDA compilation rather than a fast-math substitution.

The adapter compiles that device source to PTX with NVRTC bundled in the
installed Torch package. It then uses the CUDA Driver API to launch the kernel
on the **current Torch CUDA stream**, passing tensor pointers directly. The
temporary tensor and input/output lifetimes are recorded on that stream.

Usage, before importing the PointMLP model or `pointnet2_ops`:

```python
from pathlib import Path
import sys

sys.path.insert(0, str(Path(repo) / "docs/research/diagnostics/embedding_audit85"))
from local_fps import install_pointnet2_fps

fps_runtime = install_pointnet2_fps(private_runtime_directory)
# Now import the unchanged model through its usual source context.
```

`private_runtime_directory` must be outside versioned sources. It receives PTX,
name expressions, compilation log, runtime identity and optional test report.
No binaries are stored or committed beside the adapter. The installed module
provides **only** `pointnet2_utils.furthest_point_sample`; it does not pretend to
implement grouping, interpolation or backward operations. This is sufficient
for the frozen PointMLP forward currently used by the embedding audit.

The default constructor checks local `nvidia-smi` for an idle GPU **before CUDA
allocation** and rejects a preinitialized CUDA context. No server is accessed.
The audit parent must not run concurrent local CUDA jobs during extraction.

Validation:

```text
python test_runtime.py --runtime-dir <private-runtime-directory>
```

The bounded test checks exact output indices against an independent float32 CPU
reference retaining the original per-thread reduction rules: random sizes from
1 through1024, sample counts exceeding input counts, duplicate/symmetric ties,
near-origin filtering, all-origin input and empty output. It also verifies
nondefault-stream consistency and unchanged CUDA RNG state. On the local
RTX4060 Laptop / Torch2.6.0+cu126, all11 cases passed; the B8/N1024/M512 kernel
averaged about0.716ms over50 calls. Exact private provenance is recorded in
`fps_validation.json` and `fps_runtime_identity.json`.

**This operator gate does not replace the final frozen-model/cache check.**
Before producing full-population comparisons, use an existing trusted clean
embedding cache to verify the complete model's outputs and input/order/BN
identity under this local runtime. Cross-device and Torch-version differences
must be measured and disclosed rather than assumed bitwise identical.
