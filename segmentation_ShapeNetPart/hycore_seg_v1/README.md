# DGCNN / HyCoRe-seg v1

User-authorized implementation, 2026-10-28. First train and validate the pure
DGCNN baseline on server 8000. HyCoRe-seg is implemented for subsequent checks;
its production training is not automatically chained behind the baseline.

## Sources

- Pinned community DGCNN: `antao97/dgcnn.pytorch` at
  `07d534c2702905010ec9991619f552d8cacae45b`; vendored with MIT license.
- Original HyCoRe geometry: `classification_ModelNet40/models/manifolds/`
  and `models/ops/manifold_layers.py`, based on Geoopt.
- Paper: https://doi.org/10.1109/ICASSP49357.2023.10095253
- Detailed reconstruction choices: `docs/research/98_HYCORE_SEG_REPRODUCTION_2026-10-27/`.

## Explicit protocol

The original Stanford HDF5 download was unavailable. The pinned public normal-
bearing ShapeNetPart release is prepared as XYZ only: center using all input
points, unit maximum radius, first 2048 points, deterministic cyclic padding if
needed, no normals or extra augmentation. Train shuffles point order and labels
together. This adapter is not asserted bitwise equal to the original HDF5 pack.

Its official split lists contain 9 train/val, 23 train/test and 3 val/test shared
object IDs. Preprocessing keeps test intact, then val, and removes overlapping
IDs from lower-priority splits. All original lists, overlap IDs, removed IDs,
hashes and effective counts are retained in the private data manifest. This is
an explicit leakage correction, not an exact original-protocol reproduction.

Main metric: unmasked 50-way predictions; shape IoU over the object's legal
parts, empty union=1. Report instance and 16-object-class macro mIoU. Legal-part-
masked predictions are a separately named secondary metric. Select checkpoints
on validation only; run official test once after the fixed 200 epochs, on the
validation-selected checkpoint. No early stopping or voting.

Baseline: unchanged upstream network, SGD .1, momentum .9, WD1e-4, cosine200 to
.001, batch32/eval16, k40, dropout.5, seed4780, ordinary unsmoothed CE, FP32.
The wrapper exposes 128-d features but preserves upstream outputs, gradients,
BN and state names, tested against the vendored model.

HyCoRe-seg: replace only the final bias-free128->50 map with a shared Möbius
map, initialized with the exact same weight. Ball coordinates are logits; this
head/logit interpretation is a declared assumption. Whole/part representations
are Euclidean means of one shared feature forward, then expmap0 at K=-1.
One feature-neighborhood part/object, one uniform integer size200..600/batch,
stable nearest indices including center, uniform different-object-class negative.
CE + .01 Rcontr + .01 Rhier; margins4 and1000/N respectively. Use Geoopt0.5.1
RSGD. Disabled regularizers perform no part sampling or extra encoding. Separate
CPU RNG for part selection preserves the main input/dropout stream.

Geoopt0.5.1 RSGD's first momentum buffer differs from PyTorch SGD, even for
ordinary Euclidean parameters. Baseline retains upstream SGD; head-only and
full HyCoRe-seg both use the same RSGD. Their comparison isolates regularizers;
baseline-vs-head also changes optimizer. An additional Euclidean RSGD control
is needed before attributing that difference solely to the head. Do not silently
patch the published optimizer behavior.

## Commands (from repository root)

```bash
CUDA_VISIBLE_DEVICES='' python -m unittest segmentation_ShapeNetPart.hycore_seg_v1.test_contracts -v
python -m segmentation_ShapeNetPart.hycore_seg_v1.prepare_data --archive <zip> --output <new-data-root> --expected-archive-sha256 <sha>
CUDA_VISIBLE_DEVICES=0 python -m segmentation_ShapeNetPart.hycore_seg_v1.train --config segmentation_ShapeNetPart/hycore_seg_v1/configs/dgcnn_baseline.json --data <prepared-root> --out <new-run-root> --physical-gpu 0
```

Smoke: add `--epochs 2 --max-steps 2 --limit-train 64 --limit-val 32 --workers 0
--no-final-test`. Keep the 200-epoch cosine; restart production from seed, not
smoke weights. Resume is an explicit complete-state fork into a new directory,
with matching config, data, RNG, optimizer, scheduler and prior best checkpoint.
All data, checkpoints, logs and private manifests remain outside Git.
