# Local experiment and implementation audit

Research only, 2026-11-01. No training/model/config changes or GPU operations were performed by this audit. Results below come from the preserved local report source, with live-log observations supplied by the coordinating agent separately identified. Percentages use best test Instance mIoU and the Class mIoU from that same epoch.

## 1. What the completed experiments establish

Source of the complete score table and layer definitions: [report builder, score_rows](../124_HYCORE_SEG_ALL_RESULTS_2026-10-31/build_report.py:170); earlier independently recorded A–E results: [115 comparison](../115_PSEG3_ABCDE_RESULTS_2026-10-30/comparison.md:35).

| Arm | Best epoch | Instance % | Same-epoch Class % | Tail after common Conv8/BN8/Leaky output T256 |
|---|---:|---:|---:|---|
| DGCNN | 186 | 85.383 | 81.777 | dp1 → Conv9/BN9/Leaky → dp2 → Conv10/BN10/Leaky → Conv11 |
| v1 H0 | 186 | 83.431 | 68.073 | Original tail up to 128, then EXP → bare Möbius matvec128→50, R off |
| v1 HC | 172 | 83.179 | 67.569 | Same v1 H0 tail, R on |
| v2 P0 | 183 | 78.866 | 67.644 | Different classification frontend, pointwise H512 → EXP → M512→256 → M256→50 |
| v2 Pglobal | 197 | 82.081 | 65.400 | P0 with global-max broadcast concatenated before H512 |
| v2 Pseg3 | 147 | 75.201 | 55.473 | dp1 → EXP → M256→256 → M256→128 → M128→50 |
| A | 166 | 81.992 | 62.764 | EXP → M256→256 → M256→128 → M128→50; R(T) |
| B | 166 | 83.122 | 67.568 | EXP → M256→128 → M128→50; R(T) |
| C | 158 | 78.112 | 58.454 | A with dp1; stopped after 180 complete epochs |
| D | 179 | 82.119 | 64.162 | dp1 → L9 → EXP → M256→128 → M128→50; R after L9 |
| E | 162 | 82.897 | 64.797 | Same D, R off |
| F | 185 | 84.297 | 71.278 | L9 → EXP → M256→128 → M128→50; R before L9 |
| G | 199 | 82.385 | 64.491 | F with dp1; R before L9 |
| H | 197 | 84.618 | 74.821 | Same F main path; R after L9 |
| F0 | 173 | 84.521 | 76.093 | F with R off |
| G0 | 162 | 82.897 | 64.797 | G with R off |
| H0 | 173 | 84.521 | 76.093 | H with R off |

L9 means the original Conv9(256→256,bias=False)/BN9/Leaky. R before I means one whole forward, select neighbors in its feature tensor, mean whole/selected part features → EXP → two relation losses. Every arm above has plain CE and 50-way unmasked primary predictions. P0/Pglobal do not share the community segmentation frontend or category input. See [complete layer table](../124_HYCORE_SEG_ALL_RESULTS_2026-10-31/build_report.py:197).

The most useful R-on minus R-off comparisons, in percentage points:

| Pair | Instance difference | Class difference | Interpretation |
|---|---:|---:|---|
| v1 HC − v1 H0 | −0.252 | −0.504 | Earlier simplified-head implementation; not official full M |
| D − E | −0.778 | −0.635 | Same architecture, initialization and protocol, both R terms jointly enabled |
| F − F0 | −0.224 | −4.815 | Main path same; R from Conv9 input |
| G − G0 | −0.512 | −0.306 | Main path same with dp1 |
| H − H0 | +0.097 | −1.272 | A small positive Instance difference but negative Class difference |

This is stronger evidence than “we have never run an R-off comparison”: several such controls already exist, and they do not show a consistent two-metric gain. It is also weaker than “R can never help”: all are seed4780, best-of-200 test selection, and regularizer-on F/G/H began on RTX4090 before moving to RTX5090 while zero arms used RTX5090. H’s +0.097pp Instance does not establish a robust gain. [Hardware limitation](../121_PSEG3_FGH_ZERO_REG_2026-10-30/execution.md:11).

F0 and H0 have the identical active training graph once R is off; G0 and E likewise match. They are useful engineering reproducibility checks, not independent architectural ablations or additional random seeds. The zero wrappers retain detached geometry and independent RNG streams; they do not add another model forward or BN update. [Zero loss wrapper](/D:/Hycore/segmentation_ShapeNetPart/hycore_seg_v7/losses.py:1), [zero-control acceptance](../121_PSEG3_FGH_ZERO_REG_2026-10-30/execution.md:5).

Compared with H0, H still trails the Euclidean DGCNN reference by 0.765pp Instance and 6.956pp Class; H0 trails it by 0.862pp and 5.684pp. Their shared low-Class performance matters more than treating H’s 0.097pp Instance lead as success. The DGCNN-to-H0 comparison includes tail normalization/nonlinearity/dropout and optimization changes, so it is not a pure geometry comparison.

## 2. The current head contains a stronger structural restriction than its layer count suggests

The reused official layer is exactly:

```text
weight = transp0(bias, _weight)
bias   = expmap0(_weight * _bias)
forward(x) = mobius_matvec(weight, x)
```

There is no additive bias operation, BN, LeakyReLU, or dropout inside its forward. `_bias` is a trainable parameter affecting the effective weight through transport; it is not an ordinary output bias. [Copied official source with provenance](../../../segmentation_ShapeNetPart/hycore_seg_v2/native.py:79), [M forward](../../../segmentation_ShapeNetPart/hycore_seg_v2/native.py:95).

For negative curvature, origin Möbius matrix-vector multiplication produces a positive scalar times `W x` for a nonzero transformed vector. EXP at the origin and numerical ball projection also rescale the complete vector by a positive scalar. Thus for a fixed set of model parameters, an uninterrupted stack of these actual M layers satisfies

```text
argmax(M_last(...M_first(EXP(H))))
  = argmax(W_eff_last ... W_eff_first H).
```

This directional equivalence survives intermediate radial projections; the exact coordinate/norm composition does not. Zero vectors and floating-point degeneracies are exceptions. It does not make the learning trajectory equivalent to ordinary linear training: derivatives, parameterization, radial scales and projection can differ substantially. It does mean adding another such M does not restore the removed pointwise LeakyReLU decision expressivity. Local Geoopt implementation: `math._mobius_matvec`, lines 1224–1239 in the installed 0.5.1 package, computes `tan_k(...)*(mx/mx_norm)`; manifold `mobius_matvec`, lines 289–296, applies radial `project`.

This explains why “two or three M layers” should not be presented as preserving the expressive role of Conv9/BN/Leaky plus Conv10/BN/Leaky. It motivates a targeted nonlinear-head or readout diagnosis while retaining H’s good trunk, rather than more M-depth toggles. It is an architectural hypothesis, not proof of the observed performance cause.

## 3. CE around 3 is largely explained by bounded logits

The forward returns the final ball coordinates directly as logits: [v5 forward_head](../../../segmentation_ShapeNetPart/hycore_seg_v5/model.py:59). The trainer then applies ordinary CE to flattened [B*N,50] predictions: [I plain_ce](../../../segmentation_ShapeNetPart/hycore_seg_v8/train.py:102), [training call](../../../segmentation_ShapeNetPart/hycore_seg_v8/train.py:322). The original Riemannian layer implementation is reused; this is not an accidental extra output projection added solely in segmentation.

For FP32 c=1 ball radius at most r=.996 and K=50 logits, the minimum attainable single-label CE over every vector in that ball is

`Lmin = log(1 + (K−1) exp(−r sqrt(K/(K−1)))) = 2.94002157`.

The corresponding maximum possible correct-class softmax probability is only 0.05286459. This follows by symmetry/Jensen over the other K−1 logits and the L2 norm constraint; equality has target logit `r sqrt((K−1)/K)` and every other logit `−r/sqrt(K(K−1))`.

Consequences:

- CE≈2.97 is near this readout’s mathematical lower bound and cannot be compared directly with an unbounded Euclidean head’s CE≈0.1.
- The probability bound is not an accuracy bound: a target can rank first for every point with a low softmax probability.
- A small scalar regularizer compared with CE does not establish weak training influence. Compare weighted gradients on the same shared parameters and their angles.
- Multiplying fixed-checkpoint logits by a positive scalar changes CE and confidence but never its argmax/mIoU. A temperature/readout proposal changes optimization only if used during training.
- For already correctly ranked points, CE often encourages expanding the radial logit scale. Seeing M3 near the numerical boundary is therefore compatible with the classification objective; it is not by itself proof of a bug.

Coordinating agent’s live-log extraction on 2026-11-01 found late train CE H≈2.96776 and H0≈2.96524. At a sampled I update weighted regularizer parameter-group norms were roughly 0.028/0.034 on shared parameters versus CE≈0.060, despite scalar R≈0.011 versus CE≈2.986. These are different concepts and should not be collapsed into a loss-value ratio.

## 4. The chosen radial margins often cannot be satisfied at the observed whole radius

The actual radial loss is `mean relu(d0(whole) − d0(part) + 1000/Npart)`, and contrast loss is `mean relu(d(whole,part) − d(whole,negative_part) + 4)`. Negative partners are uniformly selected from different-category samples with replacement. [Loss implementation](../../../segmentation_ShapeNetPart/hycore_seg_v3/losses.py:47).

FP32 `rmax=.996` gives maximum depth `dmax=log((1+rmax)/(1−rmax))=6.212606`. A necessary condition for a zero radial loss at a given whole embedding is `dmax − d0(whole) ≥ gamma`.

| Part points | gamma | Largest feasible whole depth | Largest feasible whole radius |
|---:|---:|---:|---:|
| 200 | 5.000 | 1.212606 | 0.541519 |
| 400 | 2.500 | 3.712606 | 0.952336 |
| 600 | 1.666667 | 4.545939 | 0.979002 |

For a whole radius of .98, available depth gap is only 1.617486, below even the smallest margin used here. No possible part radius can then satisfy any of these margins unless whole moves inward. This is a local feasibility statement, not a declaration that all gradients are dead. Its usefulness is to interpret persistent hinge loss, projection incidence and inward radial gradients together.

A fixed modest pre-EXP scale could reduce initial saturation, but it changes geometry and the interaction with the fixed margins. It therefore needs both R-off and R-on arms at the same scale. It is an engineering hypothesis, not a documented paper correction. Adding a hard tangent cap can itself eliminate radial derivatives above that cap and should not be sold as an automatic cure.

## 5. I is a useful extension, but it changes more than the regularizer tap

I has the same main architecture and initialization as H. It selects feature-neighbor indices from the whole post-Conv9 features, gathers raw XYZ, reencodes the part through the whole shared trunk, pools its post-Conv9 feature, applies EXP and the first shared M, then evaluates R in 128 dimensions. Whole and part execute in train mode in that order; all 14 BN modules update twice per batch. [I model](../../../segmentation_ShapeNetPart/hycore_seg_v8/model.py:31), [I training order](../../../segmentation_ShapeNetPart/hycore_seg_v8/train.py:313), [confirmed I plan](../125_PSEG3_I_REENCODE_2026-11-01/plan.md:15).

For BN momentum=.1, running means are `.81 old + .09 whole + .10 part`. The official PointMLP ordering is part then whole; it uses `.81 old + .09 part + .10 whole`. Both training forwards use their own batch statistics. I therefore combines branch-location change, part reencoding, representation dimensionality change, and BN distribution change. Its current training curve cannot establish the isolated benefit of passing R through the first M.

The coordinating agent’s read-only snapshot during this audit had I around epoch65, best epoch47 at 77.519%/63.988%, with last10 mean Instance≈72.946%. A recent 20-record geometry window had hierarchy satisfaction≈42.8%, part boundary≈83.3%, whole boundary≈17.3%, depth gap≈2.379. H’s final comparable window was ≈2.81%, 71.25%, 67.5%, and .338 respectively. I can achieve a clearer radial ordering while having worse segmentation; geometry improvement does not establish useful task regularization. I was incomplete at that snapshot, so no final ranking is claimed.

For I follow-up, evaluate stored best checkpoint with whole-only BN recalibration on training data as a diagnostic before a full retrain; this must use a copy and cannot change test-selection data. If planning I0, retain both whole and part forwards/BN updates with zero regularizer to isolate R from BN; ordinary H0 does not provide that control.

## 6. Sampling and class imbalance: what is and is not supported

All clouds have N2048 points. CE averaged over B*N gives each cloud equal total weight, just as a regularizer averaged over B gives one object term. Thus the mere phrase “pointwise CE versus objectwise R” does not imply bigger clouds get larger weight here. CE does weight semantic parts according to their point counts; shuffled objects follow the imbalanced training category distribution. Class mIoU averages the 16 categories equally, whereas Instance mIoU averages all test objects. Neither term currently class-balances this discrepancy. [Part selector](../../../segmentation_ShapeNetPart/hycore_seg_v2/losses.py:14), [CE reduction](../../../segmentation_ShapeNetPart/hycore_seg_v2/native.py:130).

Repeated small-category scores (bag44.819%, rocket22.810%, mug48.610%) across architectures are a concrete reason to inspect per-object prediction histograms and per-part recall. Equal scalar IoU alone cannot prove constant-label predictions. Such a diagnosis should precede adding class balancing or replacing the head. [Previous per-class evidence](../115_PSEG3_ABCDE_RESULTS_2026-10-30/comparison.md:76).

The coordinator also checked category-masked secondary scores at best checkpoints: H and I were identical to unmasked primary mIoU; F0 masking added only about .0184pp Instance. Cross-category invalid predictions therefore do not explain most of the performance deficit at those checkpoints. Training a category-restricted CE would be a new loss protocol, not a correction to the reported metric.

## 7. Most informative next steps

1. Keep H’s established main path while analyzing stored best/late checkpoints: prediction histograms/per-part recall, logit norms/gaps/confidence, gradient norms and CE–R angles on shared groups, radial feasibility by Npart, and mean features’ local/global/category contribution. These are more discriminating than another dp1/M-depth toggle.
2. Split the regularizer: use same hardware/init/BN/data/selection protocol for Roff, contrast-only, radial-only, and combined where needed. Existing combined-vs-off evidence does not identify the responsible component. A pilot should record per-class accuracy and gradient direction, not stop solely on hinge satisfaction.
3. If projection/feasibility is the bottleneck, compare a fixed pre-EXP scale in a paired on/off design; do not conclude gain is from R unless scaled-Ron beats scaled-Roff.
4. If readout/removed nonlinear capacity is the bottleneck, test a clearly marked extension preserving the useful Euclidean nonlinear block and using a hyperbolic classifier or trained logit scale with a matched Roff control. Scaling logits only at test time cannot improve mIoU.
5. Treat I as an extension separate from segmentation-paper reproduction. Diagnose its BN and reencoding effects before generalizing its poor or good result to shared-M regularization.

No new training is authorized by this research note. Existing test-selected single-seed results remain diagnostic; a final method claim needs validation-based selection, independent test reporting, repeated seeds and a disclosed data split without cross-split duplicate IDs.
