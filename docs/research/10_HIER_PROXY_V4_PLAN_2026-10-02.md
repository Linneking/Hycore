# Shared-whole HIER v4: authorized diagnostic and 200-epoch matrix

The user approved this successor on 2026-10-02 and explicitly authorized
bounded foundational diagnostics followed by three comparable 200-epoch
trajectories on completely idle GPUs. This supersedes the execution scope
of the older teacher-based draft plan for this experiment only.

## Objective and comparison

All arms share the exact random-start first 20 epochs, checkpoint, optimizer,
learning-rate schedule, seed 22, augmentation and stratified validation split.
The shared 20 epochs count toward each trajectory's total 200 epochs.

| Arm | Objective after epoch 20 | Sample K |
|---|---|---|
| B0 | CE + 0.01 (contrastive + radial intra) | none |
| H3 | B0 + ramp * (lambda_inter inter + lambda_proxy proxy) | 3 |
| H5 | same HIER objective and coefficients as H3 | 5 |

H3/H5 differ only in sample K. B0 isolates the incremental HIER objective
under the corrected shared protocol. There is no early stopping. Maximum
validation OA selects the checkpoint, and official test is evaluated once
after all 200 epochs. Neither diagnostics nor model selection read test.

## Correctness and protocol changes shared by all arms

- New code lives exclusively in `inter_hierarchy_MN40/hier_proxy_scratch_v4`.
- Random-center parent crop remains 800--1024 points; child 200--600 points.
  Both use independent gathered storage; child no longer overwrites whole.
- Part BN retains batch-stat normalization and parameter gradients, while
  running statistics update only during whole forward. This addresses a
  concrete train/eval distribution concern; it is not proof that BN caused
  the prior collapse.
- Four distinct class blocks of eight instances preserve flip-negative
  class separation. Per-class shuffled queues cover every train instance;
  minimal fixed-shape padding is logged explicitly.
- Main-model and proxy gradient norms are clipped separately at 1, avoiding
  proxy-capacity-dependent backbone clipping. Proxy weight decay is zero so
  radial movement cannot be mistaken for hierarchy learning from decay.

## Geometry

- Whole, part and proxy curvature fixed at c=1; native d1 distances.
- Source-unit comparison margin/tau are both 0.1*sqrt(0.1)=0.0316227766.
  This conversion is exact only for corresponding scaled geometries and is
  not a claim of equivalent base networks or optimization trajectories.
- No additional final whole output cap. The source-equivalent c1 cap bounds
  radial depth at 1.45465, less than the smallest intra margin 1000/600.
- Explicit ordinary versus HIER-only metric-preconditioned backward is
  diagnosed with matched Gumbel draws. CE/intra do not receive a HIER hook.

## Unified sample and proxy relations

- Self excluded before class-priority nearest-neighbor ranking.
- Positive set is mutual K-nearest neighbors; anchor needs at least two.
- Negative set is all batch indices outside self and the mutual set.
- Each draw independently resamples both j and k with replacement; indices
  inside each triple are distinct. Same-class and cross-class triples enter
  the same all-triplet mean, with their frequencies reported separately.
- Global 256 unlabeled proxies; proxy K20 independent of sample K.
- Preserve max pair/triple costs, independent hard Gumbel selections, three
  hinges, collision masking and mean over all draws including collisions.
- Sample budgets 16/32/50/64/96 and proxy budgets 8/16/32/50 are screened for
  distinct-relation coverage, collisions, active constraints and memory.

## Fixed optimization defaults pending bounded diagnostic

Main RSGD: LR0.1 ->0.005 cosine200, momentum0.9, weight decay2e-4.
Proxy ordinary-tangent RSGD: LR0.005 ->0.0005 cosine180, momentum0.9,
weight decay0. HIER ramp20 epochs after the shared20; initial shared
lambda_inter=lambda_proxy=0.03. Any final diagnostic-driven choice is
recorded before full dispatch; H3/H5 retain identical weights and budgets.

## Required gates and records

CPU tests cover geometry units, live/detached gradients, mutual candidate
sets, full epoch coverage, unchanged inputs, BN isolation and finite
backward. GPU preflight uses a new output and idle GPU only. Each arm records
commit, full resolved config, split/sample identities, seed, physical GPU,
start time, shared checkpoint/features hashes, per-epoch metrics and final
test selection. Results/checkpoints remain server-local and never committed.
