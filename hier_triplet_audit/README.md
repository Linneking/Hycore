# HIER triplet audit for HyCoRe / ModelNet40

This is an independent, **no-training** diagnostic. It does not modify the
existing `inter_hierarchy_MN40` implementation or checkpoints.

The script extracts deterministic, unaugmented **training-set** whole-object
embeddings (`parent_mu`, 256-D, Poincaré curvature magnitude `c=1`) from the
verified original HyCoRe best checkpoint. All nine settings reuse the same
feature cache and fixed 5-class × 8-instance batches. A3 is not used.

Run from the repository root in the existing server `hycore` environment:

```bash
python -m unittest discover -s hier_triplet_audit -p 'test_*.py'
CUDA_VISIBLE_DEVICES=<completely-idle-GPU> python hier_triplet_audit/run.py \
  --checkpoint /absolute/path/to/original/best_checkpoint.pth \
  --data-dir /absolute/path/to/modelnet40_ply_hdf5_2048 \
  --output /absolute/new/path/hier_triplet_audit_<timestamp> \
  --device cuda:0
```

The output directory must not exist. Keep it outside the Git repository. The
script records the Git commit, checkpoint hash, source HDF5 ordering, seeds,
versions, timing, feature cache, per-batch counts and a compact summary.

## Grid

| IDs | Mining | Sample K | Label score bonus | Proxy K |
|---|---|---:|---:|---:|
| H01–H05 | Released HIER-style mutual top-K | 4, 8, 12, 20, 30 | +1 | same as sample K |
| H06–H07 | Same, without label score bonus | 8, 20 | 0 | same as sample K |
| C01–C02 | Same-class mutual top-K vs same-class non-mutual | 2, 4 | n/a | 20 |

The HIER-style branch intentionally retains the released miner's self-in-top-K
and self-in-negative-pool behavior so that its actual impact can be measured.
The class-conditioned branch excludes self from both pools. Both retain the
released `>1` mutual-positive anchor guard and sample 50 triples per eligible
anchor with replacement.

The proxy probe samples `p_ij` and `p_ijk` from the same random proxy bank using
Gumbel choices on negative maximum hyperbolic distances. It reports proxy
collision, hinge activity and post-mask effective-loss fractions. **These
proxies are not trained**. Their values test whether the objective has usable
hinge activity at initialization; they do not establish nonzero encoder
gradients, a learned hierarchy, or future classification accuracy. The
descriptive batch-resampling intervals are conditional on one fixed checkpoint,
not confidence intervals for method performance. Raw loss magnitudes across
K settings are not an
achievement ranking because K changes the candidate distribution.

The experiment does not use ModelNet40 test examples and does not claim that
embedding proximity is equivalent to morphological similarity. Training any
HIER component requires a separate reviewed implementation/experiment plan.
