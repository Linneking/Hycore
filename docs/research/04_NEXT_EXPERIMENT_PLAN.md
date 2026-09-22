# Next experiment plan — draft for user review

Status: **do not execute until reviewed by the user.**

## Revision after user review

- Temporarily preserve the original HyCoRe train/test evaluation protocol while debugging optimization stability. These runs are engineering diagnostics, not final paper evidence. A validation split is deferred until the method is stable.
- Audit checkpoint identity before any new run. The original HyCoRe reproduction and the derived A3 checkpoint must never be mixed or labelled interchangeably.
- Build matched feature caches from both checkpoints at two locations: Euclidean/tangent features before `expmap0`, and hyperbolic `mu` after the Mobius embedding layer. The same deterministic cache is reused across matched experiments.
- Freeze the point backbone first. Diagnose `head_only`, then `hyperbolic_and_head`, and only then consider full unfreezing.
- Do not use a fixed 0.3 percentage-point OA gate before baseline variance is known. Early runs use catastrophic-failure and persistent-drift criteria; a statistical non-inferiority margin is set only after stable repeated runs exist.
- Random seeds are deferred until one protocol is stable.
- The intended hierarchy has two distinct targets: a directed radial specificity signal and a symmetric angular kinship signal. Equal-radius LCA ranking addresses only the latter and cannot by itself realize the proposed ordinary-to-special ordering.

## Decision to make

Determine first which checkpoint/representation supplies meaningful and stable instance relations, then whether an angular inter-ranking loss can be added without destructive fine-tuning drift.

The immediate goal is not to obtain a higher headline OA. It is to establish a stable protocol where the zero-increment control preserves the pretrained checkpoint, then isolate the incremental effect of the inter objective.

## Phase 0 — checkpoint and representation audit

1. Verify and hash every candidate checkpoint, beginning with the original HyCoRe reproduction and derived A3.
2. Export four primary deterministic caches: original/pre-`expmap0`, original/post-Mobius `mu`, A3/pre-`expmap0`, and A3/post-Mobius `mu`.
3. For each cache, store source commit/checkpoint hash, sample IDs, labels, feature location, geometry, number of views, augmentation configuration, and checksum.
4. Measure dynamic range, augmentation-stable mutual-kNN, and correlations against candidate independent geometry. Spearman here means rank correlation between a declared teacher relation and a declared student/geometry relation; it must never be reported without naming both variables.
5. Add `--teacher_cache` so matched runs consume the identical artifact.
6. Add `--trainable_scope` with at least:
   - `all`;
   - `hyperbolic_and_head` after inspecting exact parameter names;
   - `head_only` as the first drift diagnostic.
7. Emit a run manifest containing commit, command, teacher, seed, environment, GPU, timing, and diagnostic/non-final status.

Acceptance gate: tests pass, `beta_inter=0` does not construct/load an unnecessary teacher, and one-batch forward/backward is finite for every trainable scope.

## Phase 1 — learning-rate and drift screen

Single seed (`22`), maximum 8 epochs, original HyCoRe evaluation protocol retained for comparability, and an identical teacher cache within each matched pair:

| ID | LR | Scope | beta_inter | Purpose |
|---|---:|---|---:|---|
| P0 | 5e-4 | head_only | 0 | zero-increment retention |
| P1 | 5e-4 | head_only | 0.05 | incremental inter effect |
| P2 | 1e-4 | head_only | 0 | lower-LR retention |
| P3 | 1e-4 | head_only | 0.05 | lower-LR inter effect |

Run P0/P1 in parallel on two confirmed-idle GPUs, then P2/P3. Reuse the same teacher cache.

Diagnostic gate relative to the initialization and matched zero-increment control:

- no NaN/Inf and stable triplet coverage;
- stop obvious failures such as a persistent OA loss above roughly 1 percentage point after the initial transient, but do not treat 0.3 as a justified universal threshold;
- declared structural metrics do not collapse as in the first run;
- inter run improves at least two of three matched structural metrics (Spearman, satisfaction, ranking loss) across multiple epochs rather than one isolated point.

If neither learning rate preserves the original representation, stop method comparison and move to Phase 2. Do not increase `beta_inter` to compensate for an unstable optimizer.

## Phase 2 — trainable-scope diagnosis

Using the better learning rate from Phase 1, expand unfreezing in stages:

| ID | Scope | beta_inter |
|---|---|---:|
| S0 | hyperbolic_and_head | 0 |
| S1 | hyperbolic_and_head | 0.05 |
| S2 | all | 0 |
| S3 | all | 0.05 |

This phase identifies whether drift originates mainly in the Euclidean point backbone or the hyperbolic/classification layers. Add a frozen-teacher embedding-anchor loss only if restricted unfreezing still fails to retain structure; treat that anchor as a protocol control, not as the proposed hierarchy contribution.

## Phase 3 — confirmation

Only after one matched pair is stable, first introduce a deterministic validation split and repeat the selected configuration. Then:

- run seeds 22, 42, and 2026;
- report mean, standard deviation, paired differences, and confidence intervals;
- evaluate official ModelNet40 test data once for the selected checkpoint per seed;
- retain runtime, memory, triplet coverage, and failure cases.

## Phase 4 — two-axis hierarchy target

Before making a morphology-hierarchy claim, separate the target into:

- radial specificity `s_i`: a directed ordinary/prototypical-to-special ordering;
- angular kinship `k_ij`: a symmetric measure of shared attributes or branch membership.

The raw hyperbolic embedding may receive the radial loss. A same-direction equal-radius copy may be used only for the angular/LCA loss so radius cannot trivially solve kinship. Initial independent-signal candidates include:

- normalized Chamfer distance on consistently normalized point clouds, treated only as symmetric geometric closeness and not as a specificity direction;
- global spectral or shape descriptors;
- part-count/proportion descriptors where available;
- a fused graph whose edges must be stable under augmentation.

Teacher acceptance criteria should include within-class dynamic range, cross-augmentation mutual-kNN Jaccard, neighbourhood stability across seeds, and correlation with independently defined geometry. Frozen HyCoRe distance remains a self-distillation baseline.

## Questions for user review

1. Is the immediate paper claim intended to be “structure preservation/regularization” or “discovery of a new morphology hierarchy”? The latter requires the independent teacher in Phase 4.
2. Which observable signal should define radial specificity: attribute/part inclusion, prototype-to-outlier ordering, or another explicit rule? This choice materially changes the scientific claim.
3. Are ShapeNetPart/PartNet semantic parts acceptable as a later source of specificity supervision, even though the initial classifier is validated on ModelNet40?
