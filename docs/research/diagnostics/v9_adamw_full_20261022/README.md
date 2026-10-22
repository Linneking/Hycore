# Full V9-HIER proxy AdamW candidate

This versioned successor performs one fresh 300×200-update V9-HIER experiment.
It changes only the proxy AdamW initial learning rate (`0.01 → 0.005`) and
epsilon (`1e-8 → 1e-4`). The proxy cosine minimum remains `0.0005`.

The source-pinned production V9 main loop remains intact: seed22, two ranks,
global64 class-balanced sampling, all9840 train objects, native c1 HyCoRe,
c0.1 HIER depth4.6 whole view, proxy cap2.3/GS/random initialization, lambda.5,
margin/tau.1,300 epochs,200 updates per epoch, and test2468 every epoch with
strict unrounded OA best selection. The established test-selection protocol is
explicitly labelled TS. No early stopping or resume is enabled.

`runner.py` compiles the original V9 source after four asserted replacements:
the actual AdamW statement, checkpoint version, argument hook (fresh-only),
and configuration hook (the two values plus source/version identity).
The original source is never edited. There is no report73 observer, additional
per-step model forward, per-proxy diagnostic JSON, or 80-snapshot-per-ten-epoch
extension. Existing metrics, structure telemetry, parameter audits and ordinary
checkpoints remain. Original e0/e1/e5/e99/every20 checkpoints and real training
pre-optimizer snapshots at configured audit epochs are retained.

`launch_queue.py` compiles the original V9 queue with source-pinned successor
entry, frozen wrapper files, an additional CPU suite and saved actual optimizer
acceptance. It runs the parent CPU and dual-rank Gloo contracts, waits for two
completely idle GPUs, locks and rechecks both, performs a separate train-only
2×2 smoke, then launches production in a new process and directory. The smoke
retains the original two-epoch short schedule; production retains the enforced
300-epoch cosine. No smoke weights initialize production. Query failures remain
unknown and wait safely. Existing results are never overwritten.

Local CPU check: `python -m unittest test_contracts -v` from this directory.
The suite checks exact source deltas, preserved evaluation/training paths,
actual AdamW construction and CPU steps, cosine state and saved-state rejection,
GPU queue protections and source-change rejection. Runtime/source/data and
physical GPU details belong in private run manifests, not this repository.

This is a joint optimizer candidate. It does not isolate LR from epsilon, and
the ten-epoch reduction in cap saturation does not establish full-run accuracy
or semantic hierarchy benefits.
