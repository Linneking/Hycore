# Frozen whole/proxy geometry audit85

Current phase: plan and preparation only. Wait for the user to report that
server access is restored before further collection or full model extraction.
The reviewed scope and resource/acceptance rules are in
`docs/research/85_EMBEDDING_AUDIT_2026-10-24/plan.md`.

- `local_fps/`: exact original CUDA FPS device source via NVRTC/Driver for the
  local idle GPU. Eleven operator gates passed; complete frozen-model/cache
  regression remains required before population export.
- `analyze.py`: CPU-only NPZ geometry analysis, no Torch/CUDA/model construction.
- `test_analysis.py`: nine synthetic mathematical and format contracts.

`analyze.py --manifest PRIVATE.json --out PUBLIC_DIR` expects clean/eval caches
with common canonical IDs/labels. The production default is9840 objects/40classes.
HIER cache fields are native_mu, hier_mu, proxy_ball, optional proxy_tangent and
existing usage arrays. H0 has no proxy. See methods.md for provenance and units.

Do not commit raw caches, point clouds, checkpoint copies, private paths or PTX.
Never use a busy GPU or change BN/weights. The synthetic tests and FPS gate are
preparation evidence, not results from the four experiments. The planned class
cone-radius/separation diagnostic is not yet implemented in the current analyzer.
