# V5-HIER64-K20-W20 main training and V5-B0-32-CAP200 stability diagnostic — 2026-10-03

命名提示：本文展示名称按[统一实验命名规范](33_EXPERIMENT_NAMING_CONVENTION_2026-10-06.md)更新；原名称可查映射。文件名、运行目录、代码字段与既有结果身份保留。

## Authorization and status

The user approved V5-HIER64-K20-W20 on two idle GPUs and a V5-B0-32-CAP200 stability diagnostic on
the remaining idle GPU. The user also approved all proposed monitoring.
Implementation and bounded startup/recovery checks have passed. Both
production jobs are detached and running; each completed its first epoch,
validation and checkpoint before the launch was reported as successful.

Code branch: `codex/hier-v5-main-training`. All production code is developed
in the authoritative local working copy and transported by GitHub, then
server `fetch` / `pull --ff-only`. Existing runs are preserved.

## Reviewed experiment settings

| Setting | V5-HIER64-K20-W20 | V5-B0-32-CAP200 stability diagnostic |
|---|---|---|
| Budget | 200 epochs ×200 steps, includes20 base-only epochs | 200 epochs ×200 steps |
| Initialization | New random HyCoRe PointMLP, seed22 | Same seed22 random model |
| Split | Existing seed22 stratified8856 train /984 val | Same |
| Batch | Two local32; global32 distinct classes ×2 | One local/global32 |
| Sample selection | Uniform distinct classes; replacement within class | Epoch permutation without replacement, first6400 samples |
| Batch loss scope | CE/intra/inter all64 | CE/intra all32 |
| Intra negative | Global part.flip(0), different-class assertion | Local part.flip(0), original same-class possibility monitored |
| Input | Original whole800–1024 /part200–600 alias overwrite | Same operator |
| BN | Child then whole update local32 BN; rank0 buffer broadcast | Child then whole update local32 BN |
| FPS | Fixed512 first-stage centers, including repeated indices for small parts | Same |
| Space | Fixedc1, D256, FP32, original numerical ball projection | Same |
| Base loss | CE smoothing.2 +.01Rcontr +.01Rhier; margins4 and1000/Npart | Same |
| Model optimizer | RiemannianSGD LR.1→.005 cosine200, momentum.9, WD2e−4, norm1 clipping | Same |
| HIER | K20 inclself, ≥2 mutual nonself positives, T50 replacement, source self-k retained | Absent |
| Proxies | Random tangent P512/D256; no class/part binding | Absent |
| Inter loss | λ=.5 fromepoch21; Rsample+Rproxy, source hinges/Gumbel/collision masking | Absent |
| Proxy optimizer | AdamW LR.01→.0005, shared relative cosine, no step first20 | Absent |
| Extra cap/backward hook | Disabled, shadow monitoring | Disabled, shadow monitoring on whole/part |
| Selection | Highest validationOA, tie lower validationCE | Same |
| Official test | Once at completion on validation-selected best | Same |

An epoch permutation would supply276 complete batch32 batches on this split.
The200step cap is retained; single-epoch unique coverage is6400/8856=72.27%.
This is a source-operator stability diagnostic with an adapted training
budget/split, not a complete original-protocol reproduction or matched
global64 zero-HIER control. No virtual64 gradient-cache run is launched.

V5-HIER64-K20-W20 will preserve its new epoch20 checkpoint/prefix. A later strict V5-B64
requires separate authorization; it is not queued as part of this launch.

## Monitoring approved by the user

Every optimizer step:

- Finite losses, model/proxy gradients, and clipping norm; fail visibly on
  nonfinite values rather than silently skipping an update.
- Model gradient norm before clipping, norm after clipping, threshold1
  reach/exceed counts; proxy gradient norm separately, with no proxy mixing
  into model clipping.
- Global input sample IDs/classes, unique-ID coverage, repeat counts and
  per-class draw counts. For V5-B0-32-CAP200 also same-class negative-pair counts.
- V5-HIER64-K20-W20 eligible anchors, zero/one-positive cases, positions and uniqueIDs in
  i/j/k roles, per-class/ID anchor participation, candidate triple/pair
  coverage, repeated draws, proxy collisions, active hinges and self-k.
  Sample and proxy graphs have distinct denominators. Warmup relation
  monitoring does not imply an active HIER gradient.
- Whole/part depth and radius, proportion near the native numerical ball
  boundary, shadow tangent2.3 cap and inverse-metric factors. Native output
  boundary contact is not a measurement of preprojection trigger count.
- V5-HIER64-K20-W20 proxy radial statistics, numerical ball projection count and maximum
  replica difference after synchronized optimizer updates.
- Child/whole BN update counts, original alias and FPS/crop observations.
- Heartbeat, step timing, memory and resolved learning rates.

Every epoch:

- Training confusion matrix /OA/AA and loss summaries; validation OA/AA,
  CE, per-class accuracy and confusion matrix on fixed clean inputs.
- Coverage and gradient/geometric summaries with explicit numerators and
  denominators, epoch time and peak memory.
- Atomic last checkpoint, updated best when selected, and checkpoint
  identity/epoch/size records. Archive every20 epochs (including epoch20
  prefix), plus final200. Model, optimizers, schedule, RNG per rank,
  sampler/split/config and BN buffers are preserved for epoch-boundary
  recovery into a new run directory.

Every10 epochs: full fixed clean training-set evaluation, compared with
training-mode accuracy and validation; diagnostic evaluations preserve RNG
and do not update BN.

First batch of epochs21/40/100/160/200: actual CE/intra/HIER parameter
gradient component norms and cosines, divided into shared encoder,
Euclidean feature layers, Mobius embedding, and classifier. V5-B0-32-CAP200 reports
CE/intra only. V5-HIER64-K20-W20's discarded audit restores RNG and BN buffers.

## Acceptance before dispatch

1. Existing operator/sampler/distributed correctness gates plus telemetry
   aggregation checks.
2. Bounded two-epoch smoke for each entry point with HIER active in the
   second V5-HIER64-K20-W20 smoke epoch; finite updates and complete monitoring.
3. Validation and best/last checkpoint writing; an epoch-boundary restore
   into a new diagnostic directory.
4. Inspect live `nvidia-smi` again before smoke and main allocation; only
   completely idle assigned devices are used.
5. Detached startup confirmed using process ownership, fresh heartbeats,
   completed optimizer steps and first epoch/checkpoint, rather than PID
   existence alone.

Smoke uses truncated evaluation, is labelled diagnostic-only, and is not
used for model selection or production initialization. Production starts
again from random seed22 with its own new directory.

## Verified dispatch record

Production code: `6e13ff9f59ad3bec8f7c3db857e6558e37473c03`.
Run directory basename: `hier_proxy_v5_20261003_1347` (private absolute
server location remains in the local server map / run manifests).
Dispatch time: **2026-10-03 13:50:35 Asia/Shanghai**.

| Job | Physical GPUs | Detached parent PID | Verified progress |
|---|---|---|---|
| V5-HIER64-K20-W20 | 1,3 | 660352 (torchrun; two training workers) | epoch1 checkpoint/validation completed; epoch2 step17 |
| V5-B0-32-CAP200 | 2 | 660357 | epoch1 checkpoint/validation completed; epoch2 step33 |

Both are configured for200epochs ×200steps. V5-HIER64-K20-W20 starts with20base-only
epochs; the main HIER gradient activates at21. Monitoring is active from
epoch1. GPU0's existing compute job was not touched.

Before production: sampler5, restored-operator/distributed6 and telemetry4
checks passed. Each entry point completed a2epoch ×3step smoke, including
HIER activation and component-gradient audit in the V5-HIER64-K20-W20 smoke's second
epoch. Both restored from their epoch1 archive into new diagnostic
directories and completed the remaining3steps. Restored first-step losses
matched continuous training; later steps were not bitwise identical.
Native point grouping/gather backward uses floating atomicAdd; deterministic
200epoch replay is not promised. No smoke weights initialize production.

Both initial-model hashes match:
`fd7251e0eb6b397ecaf3862e9cc00a6bc2be9b684329fdc31dc05915b388d29d`.
Smoke losses/gradients were finite, BN updated twice per real step, and
proxy replica difference was0. Early gradient clipping did trigger and was
recorded. This establishes startup correctness, not final accuracy or
long-term numerical stability.

Each job retains `manifest.json`, `heartbeat.json`, `steps.jsonl` and
`metrics_epoch_NNN.json`. V5-HIER64-K20-W20 checkpoints are `last.pth` / `best.pth`;
V5-B0-32-CAP200 checkpoints are `last_checkpoint.pth` / `best_checkpoint.pth`.
Both archive `checkpoint_epoch_NNN.pth` every20epochs and record per-epoch
checkpoint identity. Actual progress after this snapshot comes from server
manifests; no additional complete V5-B64 job or automatic follow-up was queued.
