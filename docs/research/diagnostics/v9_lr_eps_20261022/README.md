# V9 proxy LR/eps ten-epoch diagnostic

User-authorized fresh V9-HIER64 observation, seed22, 10 epochs × 200 updates.
Only the proxy AdamW initial learning rate and eps differ from report71:
`lr=.01 -> .005`, `eps=1e-8 -> 1e-4`. Proxy minimum LR stays `.0005` and both
cosine schedulers retain their original 300-epoch horizon. The backbone optimizer,
sampler, initialization, native-c1/HIER-c.1 geometry, HIER shell4.6, cap2.3,
branch GS, lambda.5, margin.1 and tau.1 are preserved.

This is a combined stability candidate, not a factorial isolation of LR vs eps.
It is train-only: no validation/test forwards, checkpoint selection, early stop,
teacher or input checkpoint. Existing results and production source are untouched.

`runner.py` loads the reviewed report71 observer and asserts the frozen original
V9 training source SHA. Its generated source differs only at the optimizer line
and diagnostic checkpoint version. Every-step per-proxy radial/angle/Adam
accounting and eight same-forward sparse snapshots per epoch are inherited.
`diagnostic_source` records both the inherited observer and successor SHA.

Run CPU source tests in this directory:

```text
python -m unittest test_runner_contracts -v
```

The existing observer tests must also pass in the sibling
`v9_proxy_cap_cause_20261021` directory. The queue runs those, this suite, and the
existing V9 geometry/base suites before using any GPU. It freezes source, records
commit/config/device/time identities, requires two completely idle GPUs with
process locks and a second idle check, runs a separate fresh 2×2 smoke, then
starts the fresh ten-epoch observation in a new directory.

Launch with deployment-specific arguments; no private server paths are embedded:

```text
PYTHONPATH=<repo> <python> <repo>/docs/research/diagnostics/v9_lr_eps_20261022/launch_queue.py --repo <repo> --root <new-immutable-run-root> --data-dir <existing-modelnet40-h5-dir> --expected-commit <commit> --source-branch <branch> --diagnostic-epochs 10
```

Server choice is made before submission: prefer the user-designated primary
server if two GPUs are completely idle, otherwise use the designated fallback.
Submit only one queue. Dataset loading and crop/sampling identities remain those
of the original report71/V9 entry point; smoke and main initialize independently.

Primary observations are cap count, first crossing/return, raw norm/depth
quantiles, radial/tangential gradients, actual step size, Adam history/decay
accounting, hinge progress/collisions and proxy coverage. Compare the completed
10×200 trajectory with report71 and retain any numerical/runtime differences.
Do not interpret fewer capped proxies alone as proof of semantic hierarchy or
classification improvement.
