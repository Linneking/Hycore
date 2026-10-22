# Existing-feature angular audit, 2026-10-22

This is a CPU analysis of existing feature arrays and saved actual forwards.
There were no encoder forwards, optimizer updates, GPU use, dataset reads,
test forwards or modifications to existing results. Both machines executed
published commit `2ed430f965f03559ae7034ae0fc970dcc1a1e877` with2 CPU threads.
V9 analysis ran15:51:28–15:51:35.538; Cars15:51:55–15:52:09.263, Asia/Shanghai.
End times are the completed aggregate file modification times; source SHA256
was checked before and after each observation. V9 runtime was Python3.9.23,
NumPy1.26.4, Torch2.8.0+cu128, with CUDA visibility disabled.

Local inputs/remote locations remain in the private specs and handoff;
anonymous outputs are archived in the report72 `tables/angles_compact.json`
and `tables/angles.csv`. The larger local `cars_summary.json`/`v9_summary.json`
include redundant per-class CDF grids and remain in the private artifact
folder outside the repository.

## Scope and interpretation

Original HIER Cars: best epoch45,8054 train images,98 seen fine-grained car
classes,512-dimensional features, frozen ImageNet BN/eval, c=.1. This does
not use the98 unseen retrieval-evaluation classes. Source checkpoint SHA256
is `1a7df567acba0fdec00cc6ee921304c42a62659383ee2d964bae4f8115233080`.

V9 clean/eval caches: same9840 ModelNet40 training objects and40 classes;
HIER best204/e300 and matched no-HIER B64 e300,256-dimensional features.
Whole directions are normalized for a common comparison shell c=.1,H=4.6.
V9 native c1 depth variation is deliberately removed in these geometric
references. ModelNet40 car is one coarse category (class7), unlike Cars'98
fine-grained vehicle categories.

The71 diagnostic repeat contributes six saved actual augmented training
forwards: e1s1/e1s5/e1s20/e1s100/e5s200/e10s200, each global64=32classes×2.
They are not bitwise snapshots of historical original V9 production.

Within macro statistics include all unordered nonself pairs per class and
give each class equal total weight. Between statistics sample32 object
pairs per unordered class pair; class centroid angles are separate.
Supervised triple references sample64 sameclass nonself pairs per class
and a uniformly selected different class k, yielding6272 Cars and2560 V9
triples. Actual reciprocal draws, including replacement multiplicities,
remain a separate conditional population.

Pair midpoint and triple minimax depth are ideal absolute geometry
references. HIER is a relative hinge objective with finite stochastic proxy
selection, so these are neither its learned proxy optimum nor proof of an
inward gradient. Triple centers use the exact three-point hyperbolic
minimax1center, not the average of pair midpoint depths.

## Main clean/eval observations

| Observation | Within angle q50 | Within midpoint q50 | Within midpoint >4.14 | Between object angle q50 | Sameclass pair+other k triple center q50 |
|---|---:|---:|---:|---:|---:|
| Cars original HIER |42.503°|3.81540|11.385%|70.652°|2.80667|
| V9 HIER best204 |2.971°|4.59515|98.097%|91.899°|2.30783|
| V9 HIER e300 |1.993°|4.59781|97.833%|91.908°|2.30797|
| V9 B64 e300 |1.805°|4.59821|98.703%|91.734°|2.31289|

Macro within cosine means are Cars .73421, V9 HIER best .98751/e300 .98850,
and V9 B64 e300 .99239. Thus user intuition about strong classification
direction clustering is supported descriptively in late clean/eval caches.
No-HIER B64 is also extremely concentrated, so HIER is not required to
produce this geometry. It does not establish which loss caused it.

ModelNet40 car alone: HIER best within q50=1.5925°, midpoint4.598604;
e300=.9900°, midpoint4.599460; B64 e300=.8721°, midpoint4.599581.
These are useful vehicle examples, not a task-matched Cars comparison.

Class centroid between q50 is Cars63.171° vs V9 HIER e30091.926°.
Strong within clustering and wide between-class separation imply very
different pair and triple absolute centers: e300 supervised pair depth
mean4.55682 vs triple center mean2.32257. This is an increased geometric
separation opportunity, not an intrinsic need for all proxies to be on-shell.

## Actual training forwards change the causal interpretation

|71 snapshot|Within mean angle|Within midpoint mean|P(within midpoint>4.14)|Actual ij sameclass|Actual pair midpoint mean|Actual triple center mean|Selected pair/triple depth mean|
|---|---:|---:|---:|---:|---:|---:|---:|
|e1s1|69.275°|2.99201|0%|14.281%|3.23836|2.57498|3.89487/3.88517|
|e1s5|71.573°|2.92182|0%|13.311%|3.26934|2.55428|3.65779/3.65487|
|e1s20|66.907°|3.06631|3.125%|13.639%|3.50593|2.54630|3.97958/3.84906|
|e1s100|71.930°|2.90728|9.375%|5.781%|3.66468|1.84869|4.37008/3.98247|
|e5s200|42.674°|3.76993|31.25%|7.406%|3.08179|1.83398|4.59349/4.30287|
|e10s200|33.290°|4.00339|50%|6.844%|2.93687|1.82967|4.59887/4.45389|

The CSV preserves unrounded values.
Early training directions are not yet the very tight late clean/eval
clusters. Therefore late class clustering can explain a pressure toward
deep class-pair centers, but cannot alone explain the earliest cap crossings.
At e10 actual reciprocal pair midpoint q50=2.75981 and triple center q50=1.80796,
while selected pair/triple proxy mean depths=4.59887/4.45389. Most mined pair
roles are crossclass, not the tight sameclass pairs in the clean summary.

Weighted quantiles use the cumulative weighted empirical distribution with
linear interpolation. In small32-pair batches, angle q50 and depth q50 can
refer to opposite adjacent middle observations under a decreasing transform.
Do not describe e10 angleq50=25.9848° as mapping to depthq50=4.08663 by one
formula evaluation. They are independent marginal quantiles. Means or the
direct P(depth>4.14) make the actual-forward table easier to interpret.

## Why most j are crossclass despite a label boost

V9 imports V8 HIERLoss (`hier_proxy_scratch_v9/train.py:29`).
`hier_proxy_scratch_v8/hier_loss.py:52` defaults topk20; lines62–64 use
`exp(-distance)+sameclass`, then the shared reciprocal miner. In
`hier_proxy_scratch_v5/relations.py`, line64 chooses topk including self;
lines67–68 intersect directed neighbor sets and remove self; line72 keeps
anchors with at least two nonself mutual positives. Lines79–82 draw j
uniformly with replacement from these mutual positives and independently
draw k from the negative pool.

With32 classes×2 there is only one possible sameclass nonself mate. Conditional
on an eligible anchor with mutual degree d_i, its sameclass-j probability is
1/d_i when that mate is in the pool, and0 otherwise; d_i≤19. The label boost
secures the mate's high rank, but does not restrict all j to the same class.
Observed ~6.8–14.3% sameclass j is therefore compatible with this protocol,
not evidence that the reciprocal miner ignores class labels.

## Checks

Four local audit tests pass: independent Lorentz pair midpoint, degenerate
triple at origin, class balancing/source immutability/JSON serialization,
rotation invariance/triple depth bound. The independent geometry module has
additional KKT/projection certificate tests. Every analyzed triple checks
the projection optimality residual≤1e-7; every source hash remains unchanged.
