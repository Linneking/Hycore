# B05 CB32 A0: early rerun plus unchanged audit85 geometry

User approved2026-10-25: only the October25 CB32 old-HIER A0 run. Original
early retained weights are e0/e1/e5; new e0..e10 are explicitly a fresh same
recipe rerun, not recovered historical weights. PartHier is outside scope.

`runner.py --variant a0 --hier on --data-dir DATA --run-dir NEW`
uses global32=16classesx2 replacement, Ks10/Kp20/T50, seed4780, strictB05,
fixed4.6 HIER, AdamW .01/eps1e-8, unchanged300-epoch cosine horizons, actual
10x200 updates. Every-epoch test, clean-eval e10 and original audits e1/e5
are preserved. Full recovery checkpoints e0..e10 are retained. The old runner
and original cycle parent are SHA-pinned and immutable.

`launch.py --settings PRIVATE.json` freezes committed dependencies, checks
CPU contracts/data identity, reserves one idle GPU, runs an independent2x2
train-only smoke then fresh10. It checks actual optimizers/schedulers, every
saved epoch, BN4000, historical initialization/first-batch input identity and
reports e1/e5 parameter differences without claiming bitwise replay.

After the training process actually exits and CUDA context is released,
unchanged audit85 gate/collector use savedBN/eval, canonical sorted-shard
batch32 contexts, clean first1024 of the same9840train objects. Both original
e1/e5 are auxiliary; TS13-best is the fixed representative-class reference.
No audit optimizer/backward/test embedding/BN calibration is performed.

`postprocess.py --manifest PRIVATE.json --out NEW`
calls the original audit85 analyzer unchanged; outputs full geometry CSV/JSON,
11PNG/SVG plot pairs, per-epoch/paired-proxy movement tables and Chinese report.
Common frozen ancestor readout remains audit85 CB64/K20/8batches/minimax and
is explicitly separate from actual CB32/K10-K20/Gumbel training usage.
Initialization usage is N/A. IDs across original/replay are not semantic pairs.

All weights, coordinates, object IDs, complete logs, input archives and host
details remain private. Public output contains only reviewed derived statistics.
CPU contracts: test_runner.py (8), test_postprocess.py (4), plus unchanged
audit85 server contracts. Rendered synthetic fixtures are not measured results.
