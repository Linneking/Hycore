# Reviewed V5 / V6 aggregates — 2026-10-04

These are derived summaries reviewed for the accompanying
[report](../../22_V5_V6_RESULTS_AND_STRUCTURE_2026-10-04.md).
Raw monitoring JSONs, checkpoints, embeddings, point clouds and server
identity are deliberately retained outside Git.

## Files and units

- `epoch_curves.csv`: 500 rows (V5 e1–200, V6 e1–300). OA columns use
  percent; fractions use 0–1; LR is the actual update learning rate.
  `sample_distinct_active_fraction` and `proxy_distinct_active_fraction`
  divide active, noncollision, i!=k draws by all noncollision, i!=k draws.
  Empty clean-train entries mean not measured, not zero.
- `training_monitoring.png` / `.pdf`: produced by the tracked
  `inter_hierarchy_MN40/plot_v5_v6_monitoring.py`, using the above CSV.
  Both runs' active curves have the same distinct-index denominator.
  Boundary observations come from augmented, part-overwritten training
  forwards, not the clean geometry cache.
- `geometry_comparison.csv`: four frozen clean-eval checkpoints,
  full8856training instances and512proxies each. c=1 hyperbolic distances;
  angles in degrees; radii are Euclidean ball norms; fractions use0–1.
  `eligible_*` retrieval covers each checkpoint's reciprocal-qualified
  proxies; `all_*` covers all512. `sample_nearest_*` is the reverse
  sample-to-proxy assignment, not Gumbel ancestor usage.

## Audit conditions

Seed22; matching train IDs/labels; no test split used for the feature caches.
All geometry calculations use NumPy FP64 and canonical ID tie-breaking.
Within-class geometry covers all1,552,308unordered distinct-ID pairs.
Cross-class geometry samples10,000pairs per anchor class, with replacement,
using the same pair IDs for all four models; foreign examples are uniform
over other-class rows, not uniform over other classes.

The separate frozen64batch radius-sensitivity probe uses NumPy FP32 and
stable position tie ordering. It is described in section6.4 of the report;
its raw batch plans and outputs are not included in these aggregates.

Production commits: V5 `6e13ff9f59ad3bec8f7c3db857e6558e37473c03`;
V6 `b3542cb7372f25758a44d2fa7bd048829947c924`.
The extended V6 cosine period is a protocol difference, not only a longer
tail of the V5 trajectory. These are one-seed results.

Current display names are V5-HIER64-K20-W20 and V6-HIER64-K20-W20. The original CSV/run IDs remain unchanged; the same curves with standardized labels are in [the canonical figure](../v5_v6_canonical_2026-10-06/training_monitoring.png). See [the naming convention](../../33_EXPERIMENT_NAMING_CONVENTION_2026-10-06.md).
