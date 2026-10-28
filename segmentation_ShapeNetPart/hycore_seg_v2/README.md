# Native HyCoRe-seg v2

Three user-approved architectures train from scratch with seed 4780:

- `p0`: official HyCoRe DGCNN classification encoder, pointwise 1024→512,
  official expmap0 and complete Möbius 512→256→50; no category input.
- `pglobal`: same encoder, concatenate pointwise 1024 with broadcast global-max
  1024, then pointwise 2048→512 and the same head; no category input.
- `pseg3`: pinned AnTao DGCNN segmentation encoder including STN, category
  one-hot16→64, global/category broadcast and local fusion1280→256, BN8,
  LeakyReLU and dp1; expmap0 then complete Möbius256→256→128→50 replaces the
  final three affine maps. The original tail BN9/BN10, activations and dp2 are
  removed, following the original HyCoRe head style.

All use the two HyCoRe-seg feature regularizers on the same pre-exp pointwise
features: each object's whole/part means are computed in Euclidean space before
the official expmap0. The hierarchy margin is `1000/part_points`, contrastive
margin 4 and both coefficients .01. The original batch-flip negative rule is
retained without filtering same-category pairs. Batch-size32 plus drop_last
prevents self pairs, but does not guarantee different-category pairs; diagnostics
report both fractions. The regularizer returns only its two weighted terms;
cross entropy is added once at the training entry point.

## Data and training

Reuse the read-only v1 prepared data adapter, IoU implementation, checkpoint and
experiment utilities. Model/geometry/loss implementation lives in this isolated
v2 package; existing classification and v1 training are unchanged. This is a
paper-guided reconstruction, not the unpublished authors' segmentation code.

`configs/native_seg.json` fixes batch32, 2048 XYZ points, k40, seed4780, workers8,
FP32 single GPU, RiemannianSGD `.01→.0001` cosine200, momentum .9, weight decay
1e-4 and total gradient clipping at 1. The CE call directly reuses the original
`hutil.cal_loss(..., smoothing=False)` function: mean `F.cross_entropy` on flattened
per-point predictions and labels, without label smoothing or temperature.

A batch contains 32 objects, each with 2048 points and 2048 segmentation labels.
Training object order is shuffled, with `drop_last=True`; the existing data
adapter also shuffles points together with their labels inside each object.
Graph neighborhoods never mix objects. Evaluation preserves all objects and
uses `shuffle=False`, `drop_last=False`. Parts have a separate CPU random
generator, checkpointed independently of loader/model randomness.

Production explicitly follows the user-selected AnTao benchmark protocol:
trainval14007/test2874, fixed200 epochs, each-epoch **unmasked test Instance
mIoU** selects best (`>=`, later epoch wins ties), no early stopping. This is
test-assisted model selection and does not estimate performance on an untouched
test set. The prepared normal-bearing raw release is XYZ-adapted, not a bytewise
copy of the original Stanford HDF5; preserve its original membership including
known duplicate identities and record their overlap in each run manifest.

Before every production run, a separate clean train/val no-test smoke uses the
full batch32/N2048/k40 shape, 2 epochs × 2 steps and the 200-epoch LR schedule.
It checks BN update counts, finite model/optimizer state, checkpoints, fresh
initialization and first-step gradients from both objectives into the encoder.
The independent head must receive no regularizer gradient. Production starts
fresh and checks `--expected-initial-sha256` before its first update.

Run directories are new and immutable. Each contains source identity, resolved
config/command, data hashes, GPU/environment, no-teacher/fresh initialization,
e0/best/last and periodic complete checkpoints, sampler/RNG/optimizer/scheduler
state, per-epoch metrics, sampled step diagnostics and status/failure markers.

Example production command (set a verified idle physical GPU first):

```bash
CUDA_VISIBLE_DEVICES=0 python -m segmentation_ShapeNetPart.hycore_seg_v2.train \
  --config segmentation_ShapeNetPart/hycore_seg_v2/configs/native_seg.json \
  --architecture p0 --physical-gpu 0 --protocol antao_testsel \
  --data /path/to/prepared_benchmark --out /path/to/new_run
```

The companion launcher gates a fresh production run on its clean-data smoke.
Different architectures have different capacities/backbones; their differences
must not be attributed solely to the regularizer or the number of Möbius layers.
