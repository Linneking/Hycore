# Next experiment plan — draft for user review

Status: **do not execute until reviewed by the user.**

## Decision to make

Determine whether the weak benefit of exact-LCA inter ranking survives after removing fine-tuning drift and test-set leakage.

The immediate goal is not to obtain a higher headline OA. It is to establish a stable protocol where the zero-increment control preserves the pretrained checkpoint, then isolate the incremental effect of the inter objective.

## Phase 0 — code/protocol changes

1. Create and persist a deterministic stratified split from the ModelNet40 training set:
   - train subset for optimization;
   - validation subset for OA/AA and structural checkpoint selection;
   - official test set evaluated only once per selected checkpoint.
2. Add `--teacher_cache` so all matched runs reuse the same frozen teacher instead of recomputing slightly different views.
3. Store teacher metadata: source commit/checkpoint hash, sample IDs, labels, mode, number of views, augmentation configuration, and feature checksum.
4. Add `--trainable_scope` with at least:
   - `all`;
   - `hyperbolic_and_head` after inspecting exact parameter names;
   - optionally `head_only` as a drift diagnostic.
5. Run deterministic structural validation each epoch on fixed validation batches.
6. Select and save best fine-tuning checkpoints by a declared validation rule; never use test OA for early stopping.
7. Emit a run manifest containing commit, command, split, teacher, seed, environment, GPU, timing, and status.

Acceptance gate: tests pass, `beta_inter=0` does not construct/load an unnecessary teacher, and one-batch forward/backward is finite for every trainable scope.

## Phase 1 — learning-rate and drift screen

Single seed (`22`), maximum 8 epochs, identical split and teacher:

| ID | LR | Scope | beta_inter | Purpose |
|---|---:|---|---:|---|
| P0 | 5e-4 | all | 0 | zero-increment retention |
| P1 | 5e-4 | all | 0.05 | incremental inter effect |
| P2 | 1e-4 | all | 0 | lower-LR retention |
| P3 | 1e-4 | all | 0.05 | lower-LR inter effect |

Run P0/P1 in parallel on two confirmed-idle GPUs, then P2/P3. Reuse the same teacher cache.

Primary gate relative to the original checkpoint and matched zero-increment control:

- no NaN/Inf and stable triplet coverage;
- validation OA does not fall persistently by more than 0.3 percentage points;
- fixed structural Spearman does not collapse as in the first run;
- inter run improves at least two of three matched structural metrics (Spearman, satisfaction, ranking loss) across multiple epochs rather than one isolated point.

If neither learning rate preserves the original representation, stop method comparison and move to Phase 2. Do not increase `beta_inter` to compensate for an unstable optimizer.

## Phase 2 — trainable-scope diagnosis

Using the better learning rate from Phase 1:

| ID | Scope | beta_inter |
|---|---|---:|
| S0 | hyperbolic_and_head | 0 |
| S1 | hyperbolic_and_head | 0.05 |
| S2 | head_only | 0 |

This phase identifies whether drift originates mainly in the Euclidean point backbone or the hyperbolic/classification layers. Add a frozen-teacher embedding-anchor loss only if restricted unfreezing still fails to retain structure; treat that anchor as a protocol control, not as the proposed hierarchy contribution.

## Phase 3 — confirmation

Only after one matched pair passes the gates:

- run seeds 22, 42, and 2026;
- report mean, standard deviation, paired differences, and confidence intervals;
- evaluate official ModelNet40 test data once for the selected checkpoint per seed;
- retain runtime, memory, triplet coverage, and failure cases.

## Phase 4 — independent morphology teacher

Before making a morphology-hierarchy claim, build and diagnose an independent teacher graph. Initial candidates:

- normalized Chamfer distance on consistently normalized point clouds;
- global spectral or shape descriptors;
- part-count/proportion descriptors where available;
- a fused graph whose edges must be stable under augmentation.

Teacher acceptance criteria should include within-class dynamic range, cross-augmentation mutual-kNN Jaccard, neighbourhood stability across seeds, and correlation with independently defined geometry. Frozen HyCoRe distance remains a self-distillation baseline.

## Questions for user review

1. Is the immediate paper claim intended to be “structure preservation/regularization” or “discovery of a new morphology hierarchy”? The latter requires the independent teacher in Phase 4.
2. Is a deterministic held-out subset of ModelNet40 training data acceptable for validation, or is there an existing project split that must be preserved?
3. Should the next main comparison prioritize full fine-tuning at lower LR or restricted unfreezing?
4. Which independent morphology signal best matches the intended scientific narrative: Chamfer geometry, handcrafted shape descriptors, semantic parts, or a fusion?

