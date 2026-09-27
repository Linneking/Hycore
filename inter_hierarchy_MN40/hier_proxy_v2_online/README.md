# Online HIER proxy diagnostic (v2)

This directory is a versioned successor to `hier_proxy_v1`; it does not alter
the frozen-cache results. It starts from the verified epoch-229 original HyCoRe
checkpoint, keeps the Euclidean backbone and original part–whole loss intact,
and mines B1/B2 sample triplets from the current student whole embedding.
The detached embedding is used only for discrete triplet selection; the live
embedding and trainable proxies receive the HIER loss gradients.
The HIER-only path projects the live whole embedding into the open ball before
distance calculation: the first full run encountered raw embeddings outside
the strict interior check; the precise upstream numerical cause remains to be
diagnosed. The original classification and part–whole losses still receive
the unmodified embedding. Each epoch logs how many samples needed projection
and the maximum raw radius; non-finite values or raw radius above 1.10 abort
training. This guard is deliberately logged and does not establish why a few
raw HyCoRe outputs exceeded radius one in the first full-run attempt.

`--initialization-cache` is checked against the source checkpoint and dataset.
It is used once to set the initial proxy scale, never to mine training samples.
For a clean B1/B2 comparison, both arms use the same calibrated `beta_in` and
`beta_p`; B2 alone enables `beta_out`. `--hier-scale` multiplies every enabled
HIER term while leaving HyCoRe classification and intra losses unchanged.

The `calibrate` mode writes `calibration.json`. Each `train` run must use a new
directory. It saves a manifest, an epoch-zero baseline, one `metrics.csv` row
per epoch, deterministic diagnostic test predictions per epoch, fixed-probe
relations, head/proxy states and proxy geometry snapshots. Per-epoch test
curves are **engineering diagnostics**, never used for early stopping or paper
model selection. A later confirmatory protocol requires a proper validation
split and one final official test evaluation.

Example (run from `inter_hierarchy_MN40`, with absolute paths substituted):

```bash
python -m hier_proxy_v2_online.train --mode calibrate --arm B1 \
  --pretrained CHECKPOINT --data-dir DATA --run-dir NEW_CALIBRATION_DIR \
  --initialization-cache INITIAL_MU_PT --device cuda:0

python -m hier_proxy_v2_online.train --mode train --arm B1 \
  --pretrained CHECKPOINT --data-dir DATA --run-dir NEW_B1_DIR \
  --initialization-cache INITIAL_MU_PT \
  --calibration-json NEW_CALIBRATION_DIR/calibration.json \
  --hier-scale 1 --epochs 8 --device cuda:0
```

Unit checks are in `tests/`. Run them in the server's `hycore` environment
before training. Keep datasets, caches, checkpoints, and complete logs off Git.

`launch_screen.py` is the server-side launcher for the fixed 8-epoch screen:
`B1_scale1`, `B2_scale1`, and `B1_scale2` start simultaneously on three
distinct GPUs that are idle at launch. After `B1_scale1` succeeds, `B2_scale2`
starts on its GPU only if that GPU is still idle. A newly occupied card causes
the affected run to be marked skipped, never preempted. The launcher writes
`launcher_state.json` and separate console logs under the run parent; each
training process also writes its own manifest, metrics, and checkpoints.
