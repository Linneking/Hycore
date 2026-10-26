# Simplified PartHier embedding audit94

Implements the user-approved [plan](../../94_PARTHIER_V1_EMBEDDING_AUDIT_2026-10-26/plan.md).
Four frozen checkpoints, identical full9840 training inputs and deterministic
single-center nested crops. No training, optimizer, backward, BN updates or test
forward. Original PointMLP model files are unchanged from the TS02 source commit.

Run from an immutable repository snapshot on the server:

```text
python -m docs.research.diagnostics.parthier_embedding_audit94_v1.pipeline --settings PRIVATE_SETTINGS.json
```

Private settings contain input paths and identities, four approved observation
labels/epochs, a new output root, and two completely idle physical GPU indices.
The pipeline stages exact net/queue state, checks GPU availability, exports all
views with two independent workers, verifies shared input/crop identities and
unchanged model state, runs metrics, then renders the five figure groups.
Workers use advisory locks and refuse occupied GPUs. Existing output files are
not overwritten. Failed gates stop downstream analysis and leave explicit status.

For a statistics-only repair, a new private root can set `reuse_cache` to the
previous validated cache and invoke module `readout` instead of `pipeline`.
It rechecks all four file hashes, native gates, unchanged state and shared crop
identity before analysis. Neither cache files nor earlier results are changed.
The duplicate native-intra scalar check permits recorded FP32 rounding at
rtol/atol1e-5; scientific margins and the1e-3 near-threshold flag stay unchanged.

The saved Q2048 is checkpoint-specific, static and includes historical train-mode
features. This is a saved-state audit, not exact training replay. A=B collisions
are excluded from success rates and retained in native loss denominators.
Basic order and designed margin success are reported separately. All9840 clean
embeddings define the class-center directions; there is no reference split.

Raw checkpoints, embeddings, private manifests/settings and full logs stay out
of Git. Only the concise report, aggregate tables and five PNG/SVG figure groups
are publishable. The report exporter uses a whitelist for public metadata.

Local CPU checks:

```text
python -m unittest docs.research.diagnostics.parthier_embedding_audit94_v1.test_collect docs.research.diagnostics.parthier_embedding_audit94_v1.test_metrics docs.research.diagnostics.parthier_embedding_audit94_v1.test_report
```
