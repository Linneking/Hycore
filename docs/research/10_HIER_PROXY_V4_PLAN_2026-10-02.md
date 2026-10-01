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
lambda_inter=lambda_proxy=0.03. H3/H5 retain identical weights and budgets.

## Bounded diagnostic decision

Four training batches were inspected at random initialization and at the
previous v3 epoch-20 checkpoint (reference only; not the new initialization).
At the reference, K3/K5 eligible-anchor coverage was 61.72%/85.16%.
For T64, negative-pair coverage was 89.80%/91.24%, versus 83.34%/84.81%
at T50. T64 ancestor collision rates were 2.89%/2.48%; active-triple rates
were 23.95%/22.84%. Full candidate-triple coverage was 57.83%/41.46%; these
denominators differ from negative-pair and eligible-anchor coverage.

Selected sample T64 for both arms. T96 increased negative-pair coverage to
approximately97% with 50% more draws, so was not selected. Proxy T32 gave
7104 draws/1051 active triples at the reference, collision2.03%; its raw
loss0.02957 was close to T50's0.02970. Selected proxy T32, K20.

The reference median whole radius was0.99574. Matched ordinary versus
metric-preconditioned whole-gradient norms were49.791 and0.001207
(ratio2.424e-5), with identical forward loss. Selected explicit ordinary
backward for both whole and proxy; this differs from HIER's output hook.
There is no additional final whole cap. Units/margin/tau stay as above.

Base gates passed: input unchanged; part running statistics unchanged;
whole statistics update once; full base backward finite. Training split:
8856 train/984 validation,281 batches/epoch,8992 draws and136 necessary
padding repeats (1.51%),100% planned unique coverage. GPU preflight peak
allocated memory was28601.7MiB; full HIER short-run memory is a launch gate.

## Required gates and records

CPU tests cover geometry units, live/detached gradients, mutual candidate
sets, full epoch coverage, unchanged inputs, BN isolation and finite
backward. GPU preflight uses a new output and idle GPU only. Each arm records
commit, full resolved config, split/sample identities, seed, physical GPU,
start time, shared checkpoint/features hashes, per-epoch metrics and final
test selection. Results/checkpoints remain server-local and never committed.

Operational startup: all three assigned processes reserve a small CUDA
context only after an idle check. H3/H5 then wait for the common prefix;
foreign compute owners are rechecked before model allocation. The existing
PyTorch/CUDA package bytes may be copied into an SSD cache to avoid repeated
shared-HDD cold loading; runtime module paths are recorded in manifests.
This changes storage location, not framework versions or objective settings.

## Launch gate result

All14 CPU numerical/protocol checks passed. A3-epoch smoke trajectory
(shared1-epoch prefix,2batches per epoch) completed for B0/H3/H5. Actual
HIER sample/proxy gradients were finite and nonzero; no extra whole
projection occurred. Peak allocated memory in the last smoke epoch was
28390.8/28394.2/28393.2MiB for B0/H3/H5. These are runtime gates, not
classification-performance measurements. The full run removes the batch
limit, uses20 shared prefix epochs and200 total epochs, workers4.

Runtime versions verified: PyTorch2.8.0+cu128, Geoopt0.5.1, NumPy1.26.4,
SciPy1.13.1, h5py3.14.0. Physical assignment: B0 GPU1, H3 GPU2, H5 GPU3;
GPU2 is RTX5090D and the other two RTX5090, so runtime is not a hardware
matched comparison. Source labels/checkpoint identity remain explicit.
