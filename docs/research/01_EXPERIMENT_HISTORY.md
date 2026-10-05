# Experiment history

## V6 balanced64 B0 authorized, smoke passed and detached — 2026-10-06

User approved the matched B0 first, sustained mechanism experiments afterward
until usage limits, and no warmup for future newly initialized HIER runs.
Versioned v6_balanced_b0 preserves historical e20 complete model/momentum/
300-period scheduler/two-rank BN/RNG/sampler and continues e21–300; only
HIER/proxy updates are disabled. Six local/server contract tests and a separate
two-update dual-GPU restore smoke pass, with exact state checks and zero
gradient/parameter replica difference. Smoke test data were partial and
nonreportable, no official test was read, and its checkpoint cannot initialize
production. Code f649fcc on codex/v6-balanced-b0-mechanisms.

Production detached at 02:26:20 Asia/Shanghai after fresh idle-GPU checks,
using the independently hashed original e20. First full epoch acceptance is
pending, not implied by PID survival. See [29](29_V6_BALANCED_B0_AND_MECHANISM_LAUNCH_2026-10-06.md).

## Actual-use subset audit and pending next decisions — 2026-10-06

Read-only NumPy analysis joined matching-epoch sample pair/triple counts to
V6 best99/e200/e300 clean-eval caches by proxy ID. At each checkpoint, the
all-draw and active-draw masks coincide with its eligible nonboundary subset
(188 IDs); this does not assert unchanged identity across checkpoints.
Their maximum repeated raw top4 counts are 94/56/92; raw ID coverage 30/39/27,
direction ID coverage 328/173/163. The e300 used subset retains a 90.16%
bottom-radius 1% slot share. Its 297 extra eligible boundary proxies were
unused by sample at epoch300; 281 share the same dresser top4, explaining
373 = 92 + 281. This corrects the denominator interpretation, not the raw
retrieval result or proof of training descendants.

No inference, GPU, backpropagation or optimizer update. Sanitized subset
statistics and explanations are in [28](28_HIER_EVIDENCE_AND_NEXT_DECISIONS_2026-10-06.md).
[04](04_NEXT_EXPERIMENT_PLAN.md) now records pending fixed-condition mechanism
checks and a matched V6 B0 fork with resume-fidelity conditions; no launch.

## H20 whole / radial proxy retrieval diagnosis — 2026-10-06

Existing V5e200,V6best99/e200/e300 and six B32/B64 best/last caches were
analyzed on CPU, ten models aligned to the same8856H20training objects.
Sorted HDF5 versus glob-order ID mismatch was remapped by shard name/row;
all labels and final input-cache SHA256 checks pass. Code6e77c59,
branchcodex/hier-whole-cache-diagnostic. No GPU, inference, optimizer update
or new main training. Five numerical/identity/rotation checks and interactive
desktop/320px layout/data-update checks pass.

V6e300 raw top4 covers28IDs, whole equal-radius190, proxy equal-radius30.
373/485proxies share the largest raw set; whole equal-radius reduces this
to53, proxy equal-radius retains367. Deleting89lowest-radius objects yields
51IDs and281repetitions of a new dresser set;87.68%of slots come from the
remaining bottom1%. Borrowing sourceB32 radii retains concentration but shifts
identities;99.95%of slots now come from the borrowed bottom1%. Equal whole
radius and direction rankings are exactly equivalent, not independent evidence.

V6e300 dresser median depth/angle3.558/18.76degrees versus B64last4.568/2.97;
overall depth medians4.617/4.684. Current controls diagnose frozen retrieval
radial bias, not which loss caused the class-specific change. Existing baseline
protocol differences still preclude isolating the HIER objective. See[27](27_HIER_WHOLE_RADIAL_RETRIEVAL_DIAGNOSIS_2026-10-06.md).

## Original HyCoRe/B64 matched class geometry and no-update batch probes — 2026-10-05

Six best/last checkpoints of historical seed4780 B32, source seed22 B32 and
B64 were evaluated on full9840train and2468test, identical clean1024whole
and ID-stable nested200/400/600point patches. Alltest OA reproduce original
logs;12splits/1920class rows and three4batch no-update probes passed matched
input/checkpoint hashes, finite, unchanged/restored BN/RNG/parameter checks.
See[26](26_ORIGINAL_HYCORE_B64_CLASS_GEOMETRY_2026-10-05.md).

B64 best clean-test whole depth4.139 is shallower than historical4.364 and
source4.390; final is deeper than both. Original flower_pot was1/20 correct,
source2/20, B64 3/20. Historical last clean train/test gap6.725pp and B64
6.583pp show this ordinary gap is not unique to B64. Historical logs also
contain an early resume/repeatedepoch5 and independent-bestAA metadata trap.

Fixed-input trainBN increases whole–part400 gap to about3.0 from eval below.9
in allthree models. Regrouping local32 with fixed negative IDs yields mean
whole hyperbolic movement5.7–6.0 while class prediction changes at most.391%
in these256probe anchors. This confirms conditional geometry sensitivity,
not a full training cause or unique B64 failure. No optimizer updates or
new main training occurred; source paths/checkpoint files remain untouched.

## V6 B64 completed and two-document handoff — 2026-10-05

B64 finished all300epochs at09:24:55Asia/Shanghai in8h35m08s. Best-test
OA94.1653%/AA91.7116%@267; final300 OA93.1524%/AA90.7872%. Every epoch
used153updates/9792distinct IDs/drop48 and all2468test objects. Actual
best/last SHA256 matched their save records;600first/last-step replica
checks were zero, all finite checks passed,15periodic archives exist.
Full9840 cumulative training coverage was reached by epoch2.

Source B32 best-test is93.841%@216 and final93.233%; B64's single-seed
best advantage is about.324pp, not a robust batch-size gain. Final B64
clean-train OA99.7358% exceeds same-epoch test by6.5834pp; flower_pot
is3/20 correct at best and final. The base model also has high whole
boundary proximity, so this observation is not specific to HIER.
Different H20 data/sampling/update/selection protocols preclude treating
this B64 as a matched zero-HIER control.

See[23: completed B64](23_V6_B64_SHUFFLE_START_2026-10-05.md),
[25: technical handoff](25_HYCORE_HIER_HPCS_TECHNICAL_HANDOFF_2026-10-05.md),
and the ignored local
[24: private engineering handoff](/D:/Hycore/.codex-local/handoff/24_PROJECT_ENGINEERING_HANDOFF_2026-10-05.md).
Only reviewed derived curves and sanitized documentation enter Git; no
GPU analysis, new training, checkpoint alteration or result deletion occurred.

## V6 B64 shuffle baseline authorized and detached — 2026-10-05

The user selected full9840training and original every-epoch official test/
best-test selection. One new300epoch baseline uses153shuffled no-replacement
global64batches per epoch (9792distinct IDs,48dropped), two local32 ranks,
original c1/D256 CE+intra, alias/BN/FPS and SGD/cosine/norm1; no HIER/proxy.
Production code commit `56b40a2b11d3274ff7a9832b8e4cced3b366ec5c`, branch
`codex/v6-b64-shuffle-baseline`; detached at00:48:45Asia/Shanghai on two idle GPUs.
Eight sampler, two real source-criterion and six distributed/operator CPU
checks passed. A fresh2epoch x2step GPU smoke passed with zero gradient and
parameter replica differences, finite/clipped updates and reloadable full
checkpoints; partial smoke test results are not experiment results.
See[23](23_V6_B64_SHUFFLE_START_2026-10-05.md) for production acceptance.
This changes several H20 protocol factors and is not the pending matched
balancedglobal64 B0. No additional matrix was launched.

## V6 completed and V5/V6 structure audit — 2026-10-04

V6-H20 completed300x200 updates at12:01:52Asia/Shanghai; original-source
B32 completed300x307 at09:53:24. H20 bestval94.3089%@99 yields final
testOA91.2480%/AA88.9169%, versusV5 test92.1394%/AA89.1581%.
Original B32 best-test93.841%@216 is an engineering source reproduction,
not a matchedglobal64 or validation-selected control.

Read-only V6 best99/e200/e300 visualizations each evaluated all8856clean
training instances. At e300,485proxy anchors are eligible, but373 share
one dresser top-four set; sample-to-nearest-proxy entropy-effective count
is15.59, versusV5 25.43. Late actual sample ancestors are shallower than
endpoints and pair is deeper than triple98.17%; proxy ancestor ordering
and triple usage concentration remain concerns. Single-epoch coverage stays
about66%, both runs finite and proxy replicas identical. A fixed64batch
cache control confirms radial sensitivity of late sample reciprocal graphs.

Self-k exclusion, extended cosine/budget and diagnostic input-clone fix are
reported separately. See[22](22_V5_V6_RESULTS_AND_STRUCTURE_2026-10-04.md)
for metrics, units, limits and proposed diagnostics. No new main training.

## V6 authorized — 2026-10-04

The user approved dual-GPU H20 with V5 operators/hyperparameters, sample
and proxy index self-k excluded, and a300epoch cosine/budget. The spare
single GPU is assigned to original-source HyCoRe default B32, full9840train
and307steps/epoch for300epochs. Original test-every-epoch selection is
retained for this requested engineering reproduction; H20 keeps validation
selection and final-test-only evaluation. A V5epoch200 read-only proxy
top-four point-cloud script is requested first. See[21](21_V6_TRAINING_START_2026-10-04.md)
for actual implementation, checks and launch status. Both jobs launched at
01:33Asia/Shanghai with commitb3542cb, completed their first full epoch
checkpoint and enteredepoch2. H20 usesGPUs1/2 and originalB32 usesGPU3.
The startup gates and all8856-instance V5 visualization have passed; no final
V6 training result is reported at launch.

## B32 source-fidelity audit — 2026-10-03

Read-only training/source audit confirms that V5-B32 changed more than the
200-step cap: it also uses a held-out split, 200-epoch cosine period, rewritten
RNG flow, workers2, and validation selection. The model directories match,
and original part overwriting, two BN updates, CE/intra margins and weights,
RiemannianSGD and norm1 clipping are retained. Source default seed22 matches
B32, whereas the saved94.044% historical run used seed4780/300epochs/workers8.
Strict hinge-zero and floating-point grouping differences are also disclosed.
See [20: complete audit](20_B32_HYCORE_FIDELITY_AUDIT_2026-10-03.md).
No new training or training-code change was performed for this confirmation.

## V5 completed — 2026-10-03

Both jobs completed200epochs ×200steps with finite telemetry and one final
official-test evaluation on the validation-selected best checkpoint.
H20 bestval94.0041%@179, testOA92.1394%/AA89.1581%; B32 bestval94.5122%@172,
testOA92.5851%/AA89.3285%. Both epoch200 validationOA is93.0894%; cleantrain
OA is99.6161%/99.4354%. V4's severe late inference decline did not recur.
H20 has substantial whole/proxy radial saturation and only about32% proxy
anchor eligibility late in training. B32 differs in batch and sampling,
so its advantage is not an isolated estimate of the HIER effect.
See [19: complete result and monitoring summary](19_V5_FINAL_RESULTS_2026-10-03.md).

## V5 production launch authorized — 2026-10-03

The user approved200×200 V5-H20 (two local32 ranks, global64) and a spare
single-card B32 operator-stability diagnostic, including all recommended
monitoring. B32 uses an epoch permutation capped at6400 distinct instances;
it is not a matchedglobal64 B0 or a full original-protocol reproduction.
Implementation and bounded startup checks are recorded in
[17](17_V5_TRAINING_START_2026-10-03.md); actual running status is recorded
after detached jobs and completed updates/checkpoints have been verified.

## Environment and resource audit

- Server environment: Python 3.9, PyTorch 2.8.0 + CUDA 12.8, Geoopt 0.5.1.
- Hardware: four RTX 5090 GPUs. The first run used only GPUs 2 and 3 after confirming they were idle.
- ModelNet40 data and historical checkpoints remain server-local.
- Original HyCoRe reproduction (`...hycore_var-4780/best_checkpoint.pth`): epoch 229, OA 94.044%, AA 91.545% as recorded in the checkpoint.
- Derived A3 checkpoint independently evaluated at OA 93.9222% and AA 91.5907%. A3 starts from the original HyCoRe reproduction and continues with `alpha=0`, so it is not the original HyCoRe weight.

## Correctness findings in the legacy path

1. Legacy `gromov_product()` treated equal-shaped tensors elementwise rather than producing an `[N,N]` matrix.
2. Cached teacher similarities were sorted by global ID while student embeddings retained random batch order.
3. Class-balanced arguments existed but were not connected to the DataLoader.
4. A single random augmented view was used as a teacher cache.
5. No same-protocol `beta_inter=0` control existed.
6. Student-derived target distances risked circular supervision.
7. Large-learning-rate fine-tuning confounded inter-objective effects with protocol drift.

## Teacher gate

### Cosine teacher — stopped after one epoch

- Reliable triplets: 229 per epoch.
- Initial training-batch satisfaction: 88.65%.
- Interpretation: class-discriminative cosine features have too little within-class dynamic range and too little optimization headroom.

### Frozen hyperbolic-distance teacher — passed signal gate

- Average reliable triplets: approximately 17,050 per epoch.
- First-epoch satisfaction: 59.11%.
- Finite, non-zero inter gradient.
- Interpretation: adequate coverage and headroom; selected as the first main teacher.

## Twenty-epoch seed-22 comparison

Shared setup: learning rate `5e-3`, class-balanced batches, original HyCoRe losses, identical seed and schedule.

| Setting | Best OA | Best AA | Final OA | Final AA |
|---|---:|---:|---:|---:|
| Derived A3 initialization | 93.9222 | 91.5907 | — | — |
| Same protocol, `beta_inter=0` | 93.4360 | 92.3192 | 92.0583 | 91.3448 |
| Hyperbolic teacher + equal radius + exact LCA | 93.4765 | 91.8157 | 92.1394 | 91.3029 |

The main method exceeds the control by only 0.0405 OA at each run's best epoch and by 0.0810 OA at the final epoch. These are not meaningful single-seed classification gains.

## Fixed no-augmentation structural evaluation

Same 50 class-balanced batches, 7,000 within-class pairs, and 3,440 reliable triplets:

| Checkpoint | Spearman | Satisfaction | Ranking loss |
|---|---:|---:|---:|
| Derived A3, evaluated against an A3-derived teacher | 0.8050 | 72.238% | 0.03974 |
| `beta_inter=0`, final | 0.4576 | 59.535% | 0.05951 |
| Main method, final | 0.4708 | 60.320% | 0.05257 |

Relative to the zero-increment control, the main method improves Spearman by 0.0133, satisfaction by 0.785 percentage points, and ranking loss by about 11.65%. Both fine-tuned models remain far below their A3 initialization. Because teacher and reference are both derived from A3, the high A3 Spearman is a self-consistency measure, not evidence that original HyCoRe already contains a valid morphology hierarchy. The defensible conclusion is only that inter ranking weakly mitigates drift relative to A3 self-distillation.

## Known protocol flaw to fix

The first v2 run evaluated the ModelNet40 test set after every epoch. The next protocol must create a deterministic stratified train/validation split, select checkpoints using validation metrics, and use the official test set only for final reporting.

## V4 completed and V5 review — 2026-10-03

The shared-whole HIER V4 matrix completed200epochs at seed22 on the fixed
8856/984 train/validation split. Validation-selected final test OA was
88.3712% for B0,89.5462% for H3 and88.1686% for H5. The shared protocol
also showed substantial late inference decline. B0 changed part storage,
BN updates and batch construction, so it is not an original-protocol
HyCoRe reproduction. See [V4 lessons](12_V4_LESSONS_2026-10-03.md) for
evidence and limits of causal conclusions.

The user requested lessons and a training-change catalogue before the next
dual-GPU tests. [V5 review](13_DUAL_GPU_V5_CHANGE_INDEX_2026-10-03.md)
records coverage calculations, source-fidelity choices, script interfaces
and acceptance gates. No V5 GPU run has been launched; full training awaits
review of the current plan.

## V5 bounded diagnostics completed — 2026-10-03

The user authorized schemeA/200step,c1,D256,random proxies,numeric
margin/tau=.1,no additional HIER cap/backward hook,and global64 for all
three losses. Source sampling/relations, restored HyCoRe operators and
distributed gradient checks passed18 CPU checks. A16-step dual-rank joint
diagnostic completed with finite values, BN updated twice and identical
proxy replicas. Fixed-feature K/LR checks followed; no main training or
test/validation model evaluation was launched.

See [V5 diagnostic results](15_V5_DIAGNOSTIC_RESULTS_2026-10-03.md).
K10 and proxy LR.01 are coverage/numerical candidates only. Mature-reference
anchor coverage remains low and shadow cap rates increase, so short-run
success cannot be extrapolated to200epochs.

## V5 expanded K diagnostics and main-test recommendation — 2026-10-03

Additional detection authorized by the user completed64global64 plans with
two augmented training views each. K10/12/16/20 anchor coverage was
37.30/42.38/51.94/60.39%; all40classes had eligible anchors in the observed
window. Paired cross-class reciprocal graph stability remained low and
radius-dependent; this is not independent morphology evidence.

An8-step K20 joint diagnostic from a read-only V4epoch20 model copy passed.
Actual shared-encoder HIER/base parameter-gradient norm ratio at lambda.5
was.327 then.503, correcting the much larger partial-gradient ratio at mu.
Peak memory was28.73GiB/card; whole output frequently approached the native
numerical ball boundary, so finite short-run output is not a long-run guarantee.

See [16: expanded results](16_V5_TOPK_FOLLOWUP_2026-10-03.md) and the
current [main-test settings](04_NEXT_EXPERIMENT_PLAN.md). K20/lambda.5,
randomP512/D256,c1,proxyLR.01,and20base-only epochs within200total epochs
are recommended for review. No main training was launched; GPU jobs exited.
