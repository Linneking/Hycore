# CPU comparison with the fresh report71 baseline

Run only after the scripts have been published and deployed through Git.
This reads anonymous source aggregates and private raw telemetry; it never loads
an encoder/checkpoint or changes training. Full per-step arrays remain private.

```text
python docs/research/diagnostics/v9_lr_eps_20261022/analysis/compare.py \
  --run-dir NEW_RUN/main \
  --baseline-summary docs/research/71_V9_PROXY_CAP_CAUSE_2026-10-22/summary.json \
  --baseline-run-dir REPORT71_RUN/main \
  --out NEW_RUN/comparison_new_timestamp
```

The report71 reference is validated as 512 proxies, 2000 updates and cap counts
229/444/502 at epochs 1/5/10. These are not the earlier historical V9 counts.
If report71 private telemetry is not accessible, omit `--baseline-run-dir`;
trajectory and optimizer comparisons still work, while baseline loss/hinge data
are explicitly marked unavailable. They must not be inferred from sparse states.
For different servers, export `aggregate_learning(old_main / "steps.jsonl", (10, 200))`
as an anonymous JSON epoch list on the old server, then pass that small file with
`--baseline-learning-json FILE` instead of `--baseline-run-dir`. The imported
file SHA is recorded. No full step telemetry needs to move between servers.

Outputs are aggregate JSON, three CSVs and one PNG/SVG figure. They include cap
first crossings/returns, raw radius and mapped depth, actual AdamW contributions,
sample/proxy losses and hinge activation. Hinge fractions pool actual counts and
clearly state whether collision draws are excluded. Loss means use optimizer
updates on changing augmented training batches, not fixed evaluation batches.
Lower cap or loss alone does not establish a morphology or classification gain.

Run the independent aggregation checks with:

```text
python -m unittest discover -s docs/research/diagnostics/v9_lr_eps_20261022/analysis -p test_comparison_aggregation.py
```
