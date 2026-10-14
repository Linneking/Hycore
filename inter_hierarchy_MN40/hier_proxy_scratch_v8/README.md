# V8-HIER64-K20-W0

Approved 2026-10-15: one fresh 300x200-step ModelNet40 training, seed22,
two ranks, each16 distinct classes x2 instances, disjoint rank class sets.

- Shared final whole/part mu and classifier/intra/HIER: curvature magnitude .1.
- Final mu logmap -> official tangent cap2.3 -> expmap -> one inverse-metric hook.
- Proxy initialization: official Gaussian scale2.3*.9/sqrt256; forward cap/hook;
  no V7 calibration or post-AdamW parameter projection.
- HyCoRe: CE + .01 Rcontr(delta2) + .01 Rhier(gamma500/N).
- HIER: .5*(Rsample+Rproxy), K20/T50, margin=tau=.1, active from update1.
- HIER pairwise distances reproduce released Mobius denominator epsilon and
  custom Artanh derivative; block32 activation checkpointing reduces memory.
- Keep self-k exclusion; preserve original crops/alias, local32 BN and FPS.
- Model RSGD .1 -> .005; proxies AdamW .01 -> .0005; both300-epoch cosine.
- Keep model norm1 clipping and existing proxy optimizer protocol; these are
  explicit HyCoRe adaptations, not a claim to copy the original image optimizer.
- Validation selects; official test once after training. Smoke never reads test.

`train.py` is versioned from the V7 lifecycle, with V8-only geometry, base
losses and physical-depth telemetry. Earlier production entry points are intact.
Full-state checkpoints retain optimizer/scheduler/RNG/sampler/BN on both ranks.

`queue.py` freezes tracked Python source and the existing pointnet2 binary,
runs CPU official-equivalence/DDP contracts, waits for two GPUs with no compute
apps and zero utilization, then runs a separate2x2 smoke. Only after successful
smoke does it start a newly initialized production directory. All source,
logs, images, data and checkpoints stay private in that unique run root.

Tests: `python -m unittest inter_hierarchy_MN40.hier_proxy_scratch_v8.tests.test_contract -v`;
set `V8_OFFICIAL_HYPTORCH` to a private copy of official `hyptorch` for independent
forward/backward comparison. Gloo contract:
`python -m torch.distributed.run --standalone --nproc_per_node=2 --module inter_hierarchy_MN40.hier_proxy_scratch_v8.tests.ddp_contract`.
