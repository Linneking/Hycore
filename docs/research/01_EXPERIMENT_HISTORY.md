# Experiment history

## Environment and resource audit

- Server environment: Python 3.9, PyTorch 2.8.0 + CUDA 12.8, Geoopt 0.5.1.
- Hardware: four RTX 5090 GPUs. The first run used only GPUs 2 and 3 after confirming they were idle.
- ModelNet40 data and historical checkpoints remain server-local.
- Original A3 checkpoint independently reproduced OA 93.9222% and AA 91.5907%.

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
| Original A3 checkpoint | 93.9222 | 91.5907 | — | — |
| Same protocol, `beta_inter=0` | 93.4360 | 92.3192 | 92.0583 | 91.3448 |
| Hyperbolic teacher + equal radius + exact LCA | 93.4765 | 91.8157 | 92.1394 | 91.3029 |

The main method exceeds the control by only 0.0405 OA at each run's best epoch and by 0.0810 OA at the final epoch. These are not meaningful single-seed classification gains.

## Fixed no-augmentation structural evaluation

Same 50 class-balanced batches, 7,000 within-class pairs, and 3,440 reliable triplets:

| Checkpoint | Spearman | Satisfaction | Ranking loss |
|---|---:|---:|---:|
| Original A3 | 0.8050 | 72.238% | 0.03974 |
| `beta_inter=0`, final | 0.4576 | 59.535% | 0.05951 |
| Main method, final | 0.4708 | 60.320% | 0.05257 |

Relative to the zero-increment control, the main method improves Spearman by 0.0133, satisfaction by 0.785 percentage points, and ranking loss by about 11.65%. Both fine-tuned models remain far below the original checkpoint. The defensible conclusion is that inter ranking weakly mitigates structural drift; it has not learned a better hierarchy.

## Known protocol flaw to fix

The first v2 run evaluated the ModelNet40 test set after every epoch. The next protocol must create a deterministic stratified train/validation split, select checkpoints using validation metrics, and use the official test set only for final reporting.

