# DGCNN / HyCoRe-seg v1 implementation and launch record

## 2026-10-28 — user authorization and local acceptance

The user explicitly authorized implementation, dataset download, environment
creation and training on server8000. The first production experiment is pure
DGCNN part segmentation; HyCoRe-seg is implemented and checked separately, with
no automatic full-matrix launch or automatic fine-tuning from baseline weights.

Source path: `segmentation_ShapeNetPart/hycore_seg_v1/`. Upstream community DGCNN
is pinned at `07d534c2702905010ec9991619f552d8cacae45b`, MIT license included.
Only its hardcoded CUDA device was changed to the input device. The wrapper's
eval/train outputs, parameter gradients and BN/state updates match that vendor
exactly in CPU tests. Original classification/HIER/PartHier code is unchanged.

Baseline: seed4780,200epochs,SGD LR.1/momentum.9/WD1e-4,cosine200 to.001,
ordinary CE,FP32,k40,emb1024,dropout.5,train batch32/eval16,2048XYZ,
point-order shuffle only,object category one-hot16. Full passes over training,
drop incomplete training tail. Validation unmasked Instance mIoU selects best;
ties lower CE then earlier epoch. Official test is evaluated once at the end.
Masked mIoU is separately named, not mixed into checkpoint selection.

HyCoRe-seg: shared128-d pre-final features, one forward, feature-neighborhood
part, Euclidean mean then expmap0,whole shallow/part deep,1000/N′ hierarchy
margin and4 contrastive margin,coefficients.01 each. A bias-free shared Möbius
128->50 head preserves trainable parameter count and initial weight. Ball
coordinates as logits are a declared reconstruction assumption. No teacher,
hierarchy proxy, extra cap/GS, queue or subcloud re-encoding.

### Data identity and leakage correction

The upstream Stanford HDF5 download failed. The public normal-bearing archive
was downloaded from `wangps/shapenet_segmentation` at dataset revision
`dbde146b974e1fc8628b47b1b1c4e50d8bc1a2ef`.
Archive SHA256: `0e26411700bae2da38ee8ecc719ba4db2e6e0133486e258665952ad5dfced0fe`;
706988960bytes; the server copy matches. Normals are discarded, XYZ is centered
using all original points and scaled by the maximum radius, then first2048
points retained (deterministic cyclic padding if necessary). This adapter is
not claimed byte-identical to the original unavailable HDF5 dataset.

The official JSON lists contain12137/1870/2874 entries. Cross-list intersections
are9train-val,23train-test,3val-test object IDs. Keep test unchanged, then val,
and remove cross-split duplicate IDs from lower-priority splits. Retain original
split files and all removed IDs in the private manifest. Effective counts and
per-split input/file hashes will be recorded after preparation. This explicit
protocol correction is applied to every future matched arm; it must not be
attributed to HyCoRe-seg.

### Scientific and engineering checks

16local CPU contracts pass: exact vendor eval/train/gradients/BN parity, head
parameter and initialization parity, analytic FP64 geometry, independent-leaf
radial gradient signs, feature selection/gather gradients, no input mutation,
different-category negatives and invalid-negative denominator, user margin,
disabled sampling, Euclidean pooling before exp, shape/class IoU and empty
union, dataset point-label synchronization and repeat eval, complete state
next-step restoration, and optimizer semantics.

The initial failed optimizer equivalence check was useful: Geoopt0.5.1 initializes
the momentum buffer from the current gradient and updates it immediately,
whereas PyTorch SGD uses that initial gradient directly on step1. With zero
momentum they match; with.9,weight decay0,one scalar initially.2 becomes.16 under
SGD and.124 underRSGD. The published optimizer is not patched. Baseline retains
upstream SGD; head-only and full HyCoRe-seg share RSGD. Baseline-vs-head is
therefore a joint head/optimizer comparison; an extra Euclidean RSGD control
is necessary before attributing that contrast solely to the head.

The other two initial failures were an FP64 test initialized through a rounded
FP32 curvature, and an HDF5 handle not closed before Windows temporary-file
cleanup. Exact-FP64 construction and explicit dataset close fixed those tests.
They did not change the baseline network or training recipe.

### Server acceptance and production

Server environment and data preparation are in progress. Before production:
run server CPU contracts, an independent full-size2epoch×2step smoke, verify
finite actual optimizer/checkpoints,BN4,200epoch scheduler,zero official test,
then start fresh200 from seed in a new directory. Every GPU allocation rechecks
nvidia-smi and refuses occupied devices. The detached launcher only chains the
baseline smoke and baseline production; it does not launch HyCoRe-seg training.

No new segmentation accuracy is reported at this preparation stage.
