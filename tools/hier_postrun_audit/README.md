# HIER post-training audit v1

Read-only reports for HyCoRe V5/V6/V7 and classification baselines. See [the research plan](../../docs/research/38_HIER_POSTRUN_AUDIT_PLAN_2026-10-07.md) for definitions and interpretation limits.

Run from the repository root:

```bash
python -m tools.hier_postrun_audit inventory --run-dir <completed-run-arm>
python -m tools.hier_postrun_audit run --run-dir <completed-run-arm> --out-dir <new-report-directory>
python -m tools.hier_postrun_audit run --run-dir <completed-run-arm> --compare-run <other-arm> --out-dir <new-report-directory>
```

The default reads epoch summaries on CPU. It does not load models, scan the large step log, forward test data, train, or modify the source. Reports contain PNG/SVG figures, offline HTML with run/epoch table filters, CSV and explicit missing-data tables. Existing output directories are rejected. Reports must be outside source runs. Incomplete sources require `--allow-incomplete`.

Dependencies: Python 3.9+, NumPy and Matplotlib for figures/geometry. Log inventory and numeric normalization use the standard library. `--no-plots` is an explicit numerical-only fallback; missing Matplotlib otherwise fails clearly.

Existing frozen caches:

```bash
python -m tools.hier_postrun_audit run --run-dir <run-arm> --snapshot-spec <spec.json> --out-dir <new-report-directory>
```

Spec is a list or `{"snapshots":[...]}`; relative files resolve against the spec directory. Each NPZ requires `mu`, `sample_ids`, `labels`; optional `proxy_tangent`, `proxy_ball`, explicit `proxy_ids`. State `c`, `epoch`, `run_key`, `input_mode`, `input_sha256`, and an explicit `proxy_mapping` when mapping saved tangents. For own unclipped V5/V6/V7 proxy mapping use `{"numeric_radius_fraction":0.999}`. If a source forward tangent cap exists, specify it and its epsilon. Never guess a version's mapping. CPU snapshot analysis can load trusted own checkpoint proxy parameters, but safe Torch weights-only loading is the default; historical pickle checkpoints require explicit `trusted_checkpoint:true`.

Example:
```json
{"snapshots":[
 {"epoch":20,"run_key":"one-source-run","cache":"e020_whole_cache.npz",
  "c":1.0,"input_mode":"clean first1024/eval BN","input_sha256":"actual-input-bytes-sha256",
  "proxy_mapping":{"numeric_radius_fraction":0.999},"label":"e20"}
]}
```

Add `--pointcloud-pool <candidate_clouds.npz>` to render real object galleries from an existing cache; the NPZ needs clouds/sample_ids/labels.

Temporal comparisons require the same source run, object ID set, labels, actual input SHA, input mode, curvature, and explicit stable proxy IDs. Missing/unequal identities are rejected with reasons. Nearest-neighbour top4 retention differs from training ancestor activation. Class purity is an auxiliary measure, not independent morphology evidence.

Optional clean export from own completed balanced-protocol checkpoints:

```bash
python -m tools.hier_postrun_audit run --run-dir <completed-run-arm> --out-dir <new-report-directory> \
  --extract --gpu <physical-idle-gpu-index> --data-dir <ModelNet40-hdf5-dir> \
  --population panel --per-class 32 --checkpoint-policy standard --max-checkpoints 12 \
  --max-seconds 3600 --batch-size 32 --seed 22 --with-pointclouds
```

GPU export is **opt-in**. It rechecks an idle GPU before allocation, uses saved train IDs and frozen clean first1024/eval BN, and verifies checkpoint/state/input identity. It forwards training objects only. No optimizer, validation, test, or process termination. HDF5/PyTorch, the existing PointMLP/pointnet2 runtime, and the own recognized checkpoint adapter are required. Missing train IDs or unknown proxy mapping requires an explicit adapter rather than split guessing. Standard policy chooses only existing milestones plus saved best/last; file aliases do not invent epochs. All-checkpoint/full-pool export requires explicit budgets. Cache/point-cloud/source files stay private and outside Git.

Implemented: epoch geometry and usage trajectories; hard/noncollision/live-hinge activation; proxy ID usage heatmaps and JSD/rank/Jaccard; general-c depth; saved tangent vs mapped position; paired frozen snapshot displacement; raw/direction/equal-radius retrieval coverage, concentration and top4 retention; optional real point-cloud galleries.

Planned extensions remain explicitly unavailable until measured: fixed-query/Gumbel noise controls, exact per-proxy gradient activation, full dormancy history, independent shape hierarchy validation, class-conditioned failure cards. No report invents those fields.

Tests (CPU, no model/GPU):
```bash
python -m unittest discover -s tools/hier_postrun_audit/tests -v
```
