# V5 main training and B32 stability diagnostic — 2026-10-03

## Authorization and status

The user approved V5-H20 on two idle GPUs and a B32 stability diagnostic on
the remaining idle GPU. The user also approved all proposed monitoring.
At this document's preparation, implementation/startup checks are in
progress. Approval is not a record of successful dispatch; verified startup
details will be appended after both jobs are running.

Code branch: `codex/hier-v5-main-training`. All production code is developed
in the authoritative local working copy and transported by GitHub, then
server `fetch` / `pull --ff-only`. Existing runs are preserved.

## Reviewed experiment settings

| Setting | V5-H20 | B32 stability diagnostic |
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

An epoch permutation would supply276 complete B32 batches on this split.
The200step cap is retained; single-epoch unique coverage is6400/8856=72.27%.
This is a source-operator stability diagnostic with an adapted training
budget/split, not a complete original-protocol reproduction or matched
global64 zero-HIER control. No virtual64 gradient-cache run is launched.

H20 will preserve its new epoch20 checkpoint/prefix. A later strict V5-B0
requires separate authorization; it is not queued as part of this launch.

## Monitoring approved by the user

Every optimizer step:

- Finite losses, model/proxy gradients, and clipping norm; fail visibly on
  nonfinite values rather than silently skipping an update.
- Model gradient norm before clipping, norm after clipping, threshold1
  reach/exceed counts; proxy gradient norm separately, with no proxy mixing
  into model clipping.
- Global input sample IDs/classes, unique-ID coverage, repeat counts and
  per-class draw counts. For B32 also same-class negative-pair counts.
- H20 eligible anchors, zero/one-positive cases, positions and uniqueIDs in
  i/j/k roles, per-class/ID anchor participation, candidate triple/pair
  coverage, repeated draws, proxy collisions, active hinges and self-k.
  Sample and proxy graphs have distinct denominators. Warmup relation
  monitoring does not imply an active HIER gradient.
- Whole/part depth and radius, proportion near the native numerical ball
  boundary, shadow tangent2.3 cap and inverse-metric factors. Native output
  boundary contact is not a measurement of preprojection trigger count.
- H20 proxy radial statistics, numerical ball projection count and maximum
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
Euclidean feature layers, Mobius embedding, and classifier. B32 reports
CE/intra only. H20's discarded audit restores RNG and BN buffers.

## Acceptance before dispatch

1. Existing operator/sampler/distributed correctness gates plus telemetry
   aggregation checks.
2. Bounded two-epoch smoke for each entry point with HIER active in the
   second H20 smoke epoch; finite updates and complete monitoring.
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

Pending startup checks; no production launch is claimed yet.
