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
split files and all removed IDs in the private manifest. Effective counts are
12105/1867/2874, with per-split input/file hashes in the complete manifest. This explicit
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

The independent8000 environment is ready: Python3.9,PyTorch2.8.0+cu128,
Geoopt0.5.1,h5py3.14.0,NumPy1.26.4. Dependencies were copied into a separate
runtime and the cloned environment's path changed only there; old environments
and runtime files were not modified.16server CPU contracts pass. A first server
test exposed one FP32 ULP of SGD/retraction rounding; the mathematical equivalence
test now uses FP64 analytic-scale tolerance, with the real momentum difference
still tested explicitly. No optimizer or baseline update was changed to pass it.

The full-size batch32/2048-point2epoch×2step GPU smoke passed all10acceptance
checks:4updates,BN4,finite model/optimizer,complete state,200epoch cosine,
no regularizer,zero official test. Its weights are not used for production.
The CPU end-to-end runner also passed2×2 for both baseline and HyCoRe-seg on
clearly labelled synthetic small fixtures; this is engineering evidence, not
segmentation performance or a real-data HyCoRe-seg GPU acceptance.

At **2026-10-28 09:41:32 Asia/Shanghai**, the baseline started fresh200epochs
on8000 physicalGPU0, runID `dgcnn_seg_v1_s4780_20261028_A/main`, source
`c15371536bc5dacef90825cc8b886b94563b0a98`. It is detached from the chat.
The source checkout is separate from existing server runs and remains frozen.

Independent startup readout confirms first epoch378updates/12096training
objects seen (drop9),1867validation objects,BN378,regularizer0,best/last saved,
exactly matching smoke initialization and first input. First epoch took82.17s,
peak allocated18,622,219,776bytes. At09:44:06 it was epoch2/step350/728updates.
The official test has not been evaluated yet. See [sanitized startup evidence](startup_acceptance.json).

Only baseline production is running. HyCoRe-seg is implemented, CPU-checked
and deployed but its real-data GPU validation and full training remain subsequent
work after the baseline gate. No final200epoch result is claimed here; no
automatic extra experiment matrix, checkpoint fine-tuning or monitoring task
was created. Do not dispatch this baseline a second time.
