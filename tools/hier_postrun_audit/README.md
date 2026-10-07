# HIER post-training audit v2

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

Extended system audit joins completed exports without rerunning production training:

```bash
python -m tools.hier_postrun_audit system \
  --artifact-dir <v5-audit> --artifact-dir <v6-audit> \
  --artifact-dir <v7-audit> --artifact-dir <matched-b64-audit> \
  --out-dir <new-system-audit> \
  --mechanism-epochs 100,200,best,last --structure-epochs 100,200,best,last \
  --query-count 64 --noise-repeats 8 --shape-points 64 --shape-per-class 8 \
  --bootstrap 200 --mechanism-seconds 1800 --structure-seconds 1800 --seed 22
```

Artifacts contain `normalized_runs.json`, `feature_exports/snapshot_spec.json`, actual clouds and saved snapshot analysis. `system` writes a fresh directory, verifies input/checkpoint/cache identity and qualifies duplicate storage names (e.g. two H20 arms) before any temporal join. Retrieval uses the exported candidate pool; independent shape queries use a bounded stratified subset. Use `--population full` during export for full training-pool coverage. `--epochs 20,100,200,best,last` explicitly chooses only existing checkpoint identities; aliases are deduplicated by actual epoch.

Modules: `longitudinal.py` (censored dormancy, reactivation, cumulative usage), `mechanisms.py` (fixed balanced64 mining, eight Gumbel repeats, component partial gradients and saved Adam state), `structure.py` (independent Chamfer, object/class bootstrap, chance-adjusted shape neighbours and clean training failure slices), `comparison.py` (same-epoch/update evidence, recorded validation-selected final results, separate benefit ledger), and `system.py` (bounded orchestration and scientific report). Missing capabilities remain explicit. Cached HIER gradients do not imply shared encoder or optimizer update direction; the separate frozen backbone probe measures genuine encoder gradients and performs zero optimizer updates. Reference distance operators are labelled numerical counterfactuals, never silently substituted for the production loss.

Shape queries use both original coordinates and centered unit RMS, keep rotations, and use deterministic FPS64. Class purity is auxiliary, neither geometric validation nor a genuine hierarchy proof. Generalization conclusions use existing saved final test only. This single-seed historical comparison changes several parameters, so benefits are joint observations rather than isolated causal effects.

Tests (CPU; no production checkpoint or GPU required):
```bash
python -m unittest discover -s tools/hier_postrun_audit/tests -v
```

Frozen shared-encoder probes are separate, explicit GPU actions. Use one own
trusted checkpoint spec and one entirely idle physical GPU per process:

```bash
python -m tools.hier_postrun_audit backbone \
  --snapshot-spec <completed-export>/feature_exports/snapshot_spec.json \
  --epoch 200 --out-dir <new-backbone-probe> \
  --data-dir <ModelNet40-hdf5-dir> --gpu <idle-physical-index> \
  --microbatch-size 16 --max-seconds 600 --seed 22
```

The global objective uses the same ordered64 training objects and the actual
HyCoRe crop alias, CE/intra and versioned HIER operators. Microbatches limit
encoder memory under eval BN; they do not change the global64 mining/loss.
The default stages features without retaining encoder graphs, calculates global
feature adjoints, then replays one child/whole microbatch at a time and accumulates
the actual parameter VJP. Saved forward RNG and feature equality are checked
before success. It avoids activation-checkpoint hooks across scripted geometry.
The probe preserves parameters, BN buffers, existing gradients, inputs and RNG,
records input/crop identity, and performs no optimizer step. This condition is
different from both clean first1024 export and augmented training BN. Failed
attempts retain failed manifests in their own folders and are never successful
evidence. Parameter-gradient ratios are before clipping/Adam; they do not imply
the actual next update displacement.

Bundle a completed CPU system report and successful probes into a fresh public
offline delivery, preserving the source reports:

```bash
python -m tools.hier_postrun_audit delivery \
  --system-dir <completed-system-audit> \
  --backbone-dir <v5-backbone-probe> --backbone-dir <v6-backbone-probe> \
  --backbone-dir <v7-backbone-probe> --out-dir <new-public-delivery>
```

Delivery copies only public HTML/JSON/CSV/PNG/SVG. Weights, NPZ, raw logs,
working specs and normalized raw runs stay private. It verifies source hashes
and every offline file link, and gates joint encoder interpretation on identical
ordered64 IDs/labels, actual cloud bytes, crop centers, seed, model source and
inference conditions. It compares each run's own scalar HIER/base gradient
ratio and cosine; model parameter rows are never paired across runs.

The real V5/V6/V7 plus matched B64 execution and findings are documented in
[40: system audit results](../../docs/research/40_HIER_SYSTEM_AUDIT_RESULTS_2026-10-07.md).
