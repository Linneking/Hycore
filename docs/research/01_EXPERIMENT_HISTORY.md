# Experiment history

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
