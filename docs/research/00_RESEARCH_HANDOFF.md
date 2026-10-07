# Research handoff: HyCoRe inter-sample hierarchy

实验展示名统一遵循[33：实验命名规范](33_EXPERIMENT_NAMING_CONVENTION_2026-10-06.md)。旧文件名、运行ID及代码字段保留用于溯源；本次只规范称呼。

## Post-training audit method — 2026-10-07

The user approved [38](38_HIER_POSTRUN_AUDIT_PLAN_2026-10-07.md) and authorized
implementation/debugging and a real V5/V6/V7 audit with matched B64.
See [40: system audit results](40_HIER_SYSTEM_AUDIT_RESULTS_2026-10-07.md),
[39: execution](39_HIER_SYSTEM_AUDIT_EXECUTION_2026-10-07.md), and
`tools/hier_postrun_audit/`. The audit joins31 frozen checkpoints, identical8856
training inputs,11 controlled mechanism panels and15 independent shape panels.
V6 fixes invalid mining; V7 eliminates proxy numerical saturation and broadens
sample ancestor usage. Raw top4 coverage remains concentrated, and independent
shape gains over B64 are mixed. Training/eval geometry, retrieval/actual ancestors,
cached partials/shared encoder gradients/Adam state have separate meanings.
No source results were changed, no optimizer update or new test forward was made.
The following completion/status entries are historical; geometry findings in40
supersede their pending top4/shape statements.

## V7 completed — 2026-10-07 16:00 check

V7-HIER64-K20-W0 completed300 at14:43:33 Shanghai,60000 updates; process exited,
GPU1/2 released. Validation selected187 (OA93.8008%); its one official test is
OA92.2204%, AA89.7204%,2468 objects. Every epoch has200 updates and984 validation
objects, finite scalar summaries, lambda0.1 and proxy replica difference0.
Proxy numerical projection count remains0; post-step parameter cap holds within
FP32 tolerance. Actual best/last hashes match saved identities. e300 validation
OA91.0569% shows late decline. Classification changes are a single-seed multi-factor
historical comparison; whole depth still near its boundary, top4/shape checks pending.
See[37](37_V7_COMPLETED_2026-10-07.md). No new experiment launched; below are historical snapshots.

## V7 launched under delegated decision authority — 2026-10-07 02:52

The latest user explicitly delegated final V7 decisions and requested launch then exit.
The unique V7-HIER64-K20-W0 e300 run detached on GPU1/2 at02:48:51 Shanghai,
code165981f, run v7_hier64_20261007_0250. Five CPU tests and separate four-update
dual-GPU smoke passed. W0/lambda0.1, calibrated initial depth2.240048 from16
zero-update batches (whole median2.488942), post-AdamW proxy parameter cap depth6,
original balanced64/base protocol and validation selection. At02:52:07 e1step59
is finite with replica difference0, official test unread; no completed epoch yet.
The bounded128-update CPU comparison completed. Background parent is PID1;
interactive work exits while main continues. Final decision and handoff are
[36](36_V7_PROXY_SAFETY_DECISION_AND_LAUNCH_2026-10-07.md); this supersedes pending-review statements below.

## V7 single-run design requested — 2026-10-07

The user chose V7: HIER from the first update with lower weight, and proxy
initialization inside the whole median; budget is one complete e300 run,
with other cards reserved for bounded diagnostics. The concrete proposal is
[34](34_V7_SINGLE_RUN_PLAN_2026-10-07.md), pending review through
[04](04_NEXT_EXPERIMENT_PLAN.md): dual-GPU global64, W0, lambda0.1,
one-time exact proxy depth min(5,0.9*m0), retaining the V6 base protocol.
Depth5 is not guaranteed inside an untrained whole distribution. Initial
geometry calibration has zero optimizer updates and restores production
BN/RNG. AdamW proxy movement need not scale proportionally with lambda.
No new production entry point or training has been started. Historical
first-epoch lambda0.5 recommendations below do not override this new design.

[35](35_HIER_MARGIN_BOUNDARY_AND_PROXY_GRADIENT_AUDIT_2026-10-07.md) adds a CPU audit:
no missing margin unit/factor, but released numerical distances differ
substantially from the current exact c1 formula near the boundary. A same-ID
join of existing gradients finds negligible late sample gradients for297
boundary proxies and almost purely angular proxy parameter gradients;
many tangent parameters already exceed the projection saturation threshold.
These results do not isolate the cause of initial outward movement. V7
preflight should check fixed-relation numerics and actual AdamW displacement.

## V6-B64 completed — 2026-10-06 18:40

[32: V6-B64 completed result](32_V6_BALANCED_B0_COMPLETED_2026-10-06.md) supersedes
earlier running snapshots. V6-B64 finished300 at12:13:15 Asia/Shanghai and evaluated
the validation-selected e231 model once on2468 official test objects.
Best validation OA94.3089%, test OA91.8558%, AA89.9994%; historical V6-HIER64-K20-W20 has
the same best validation OA at99 and test OA91.2480%, AA88.9169%.
V6-B64 is+0.6078 test OA points (15 more correct objects), a single-seed historical
comparison without a bitwise replay claim. All280 continuation epochs have
200 updates,984 validation objects, finite telemetry and HIER0; actual best
and final archive hashes match saved identities. Process exited, GPUs released.
e300 validation OA91.2602% also shows late decline without HIER. Same-epoch
geometry and independent shape comparisons remain pending; no new run launched.

## Latest mechanism findings and V6-B64 status — 2026-10-06 08:41

[31: mechanism findings and V6-B64 status](31_HIER_MECHANISM_FINDINGS_AND_B0_STATUS_2026-10-06.md)
records the completed natural16, shape4x64, matched e40 and conditional Gumbel8
one-step evidence. HIER reduces outward movement on the four fixed e300 hotspots
under all eight noise repetitions, while their total update remains outward;
the natural population direction is not uniform. Matched e40 V6-HIER64-K20-W20 is deeper than
V6-B64 on the four inspected classes, without established classification benefit.
V6-B64 has completed198 epochs and is training199; best validation OA94.1057%
at191, no official test read. The bounded diagnostic queue completed10/10.
The proxy100-step strict endpoint gate failed after600 actual updates; its
recovery and independent matched V6-B64 morphology comparison remain pending.

## Authorized V6-B64 continuation and sustained diagnostics — 2026-10-06

The user approved V6-B64 and continuing small mechanism
experiments until usage limits, with the main training taking priority.
Future newly initialized HIER runs must enable weight .5 from epoch1 without
20-epoch warmup. Historical prefixes remain accurately labelled.

[29: launch and diagnostic record](29_V6_BALANCED_B0_AND_MECHANISM_LAUNCH_2026-10-06.md)
records a strict complete-state historical V6-HIER64-K20-W20 e20 fork for V6-B64. Code f649fcc,
branch codex/v6-balanced-b0-mechanisms. Local/server six contract checks and
two-update dual-GPU restore smoke pass, including exact optimizer/scheduler/
BN/RNG/sampler restoration and zero gradient/parameter replica difference.
Production was detached at 02:26:20 Asia/Shanghai, continuing e21–300 with
original 300-epoch cosine, data/sampling and CE/intra; source checkpoint SHA
is recorded in29. Full e21 acceptance now passes: 200 updates, 984 validation
objects and a complete checkpoint reloaded on CPU through the strict V6-B64
contract; scheduler continues at21. Training proceeded into e22.
No smoke checkpoint initializes production. Validation selects the model;
official test is read only once at the end. Diagnostics use separate resources.

## Current evidence and pending decisions — 2026-10-06

[28: contradictions, actual-use subsets and next decisions](28_HIER_EVIDENCE_AND_NEXT_DECISIONS_2026-10-06.md)
adds a read-only join by proxy ID between V6 clean-eval caches and matching
epoch sample ancestor counts. At e300, the 373 repeated top4 sets split into
92 of 188 sample-used proxies and 281 of 297 eligible boundary proxies unused
by sample that epoch. At best99 the used subset repeats 94 of 188. Thus the
expanded eligible denominator exaggerates apparent end-stage sample-use
retrieval deterioration. The e300 used subset still draws 90.16% of top4 slots
from the bottom-radius 1%; retrieval bias remains.

Local sample depth ordering is supported; independent within-class morphology
and the causal HIER increment remain unproven. The current pending proposal
in [04](04_NEXT_EXPERIMENT_PLAN.md) prioritizes fixed-condition loss/selection
diagnostics, independent relation validation and a fidelity-checked matched
balanced64 V6-B64 fork. No new experiment was launched.

## Latest V6-HIER64-K20-W20 whole / radial retrieval diagnosis — 2026-10-06

[27: matched whole geometry and frozen proxy retrieval controls](27_HIER_WHOLE_RADIAL_RETRIEVAL_DIAGNOSIS_2026-10-06.md)
uses ten existing caches, all aligned to the same8856training objects by HDF5
shard name and row. V6-HIER64-K20-W20's sorted-shard IDs differ from26's glob-order IDs;
canonical remapping passes every label check. No new model forward, GPU,
optimizer update or main training occurred; all source cache hashes remain unchanged.

V6e300 eligible proxy top4 covers28IDs with raw distance,190with whole
equal-radius copies,30with proxy equal-radius copies. Removing the lowest
radius1% only replaces the hotspot:281/485proxies still share one new set,
and87.68%of slots come from the remaining pool's lowest1%. Borrowing same-ID
V6-B0-32 radii with V6-HIER64-K20-W20 directions/proxies unchanged shifts hotspots but
retains concentration. This locates a retrieval radial bias, not an isolated
HIER training effect or proof that proxies coincide.

Dresser whole median depth is3.558and within-class angle18.76degrees at
V6e300 versus4.568/2.97degrees at V6-B0-64 last. Overall whole median depths are
4.617/4.684, so the change is class-dependent. Distinguish proxy-to-whole
retrieval from actual pair/triple ancestor use. The next question is loss-wise
radial/angular gradients and proxy objective behavior; no next matrix is launched.

## Latest matched representation audit — 2026-10-05

[26: original HyCoRe versus V6-B0-64, full class geometry and batch probes](26_ORIGINAL_HYCORE_B64_CLASS_GEOMETRY_2026-10-05.md)
compares ORIG-B0-32-S4780, V6-B0-32 and V6-B0-64 best/last under
identical clean train/test inputs and fixed200/400/600-point patches. All12full
splits and three no-update probes passed identity, finite and buffer checks.
Best clean-test whole mean depth is4.364/4.390/4.139 respectively; V6-B0-64 is
not uniformly more outward. Original flower_pot is1/20, V6-B0-32 2/20,
V6-B0-64 3/20 at best, so this weak class predates the new split.

A more important finding is BN-mode dependence: fixed overwritten whole and
patch400 input gives mean whole-minus-part depth gap.767/.831/.575 in eval
versus3.005/3.117/3.060 in trainBN. Changing local32 membership moves the
geometry while mostly preserving class predictions. These conditional probes
do not prove a batch training causal effect; relation stability across BN modes
and groupings should be checked before interpreting HIER's soft hierarchy.
No new main training was launched. Full40-class tables and sanitized derived
statistics/plots are linked from26.

## Current handoff — 2026-10-05

V6-B0-64 has completed all300epochs, finishing at09:24:55Asia/Shanghai.
Best-test OA is94.1653%/AA91.7116% at267; final300 OA is93.1524%/AA90.7872%.
All300epochs have153updates,9792distinct training IDs,48dropped tail objects
and complete2468-object test evaluation.600first/last-step replica checks
show zero parameter/gradient difference; actual best/last file hashes match
the save records. No new training was launched during this results/handoff audit.

Start with the two new handoffs:

1. [Private engineering handoff: server, code, configs, weights and operations](/D:/Hycore/.codex-local/handoff/24_PROJECT_ENGINEERING_HANDOFF_2026-10-05.md).
   This ignored local file contains server access/path information and is not in Git.
2. [Technical handoff: HyCoRe + inter, HIER and Onghena/HPCS](25_HYCORE_HIER_HPCS_TECHNICAL_HANDOFF_2026-10-05.md).
   This shareable document covers formulas, paper/code differences, evidence and next questions.

The completed V6-B0-64 configuration, monitoring and comparisons are in
[23: V6-B0-64 launch and final results](23_V6_B64_SHUFFLE_START_2026-10-05.md);
V5-HIER64-K20-W20 and V6-HIER64-K20-W20 morphology/proxy evidence remains in
[22: results and structure](22_V5_V6_RESULTS_AND_STRUCTURE_2026-10-04.md).
Reviewed public curves are in
[the V6-B0-64 summary plot](artifacts/b64_v6_2026-10-05/monitoring_canonical.png) and
[300epoch CSV](artifacts/b64_v6_2026-10-05/epoch_curves.csv).

**Current method:** original HyCoRe CE/intra and online HIER sample/proxy
regularization share the same whole embedding, fixedc1/D256. Proxies are
abstract ancestors, not physical parts; there is no proxy–part loss or teacher
in V5/V6. Original part overwrite, two BN updates and FPS behavior are retained.
The frozen-teacher/equal-radius v2 route below is historical, not current.

V6-B0-64 establishes that the source-style global64 base training can work. It has
different data, sampling, update budget and test-based selection from V6-HIER64-K20-W20,
so it does not isolate HIER's causal effect. A matched control, frozen-proxy
gradient diagnostics and independent morphology checks remain proposed for
review in[04](04_NEXT_EXPERIMENT_PLAN.md); no next matrix is authorized here.

## Historical navigation snapshots

The dated entries below record what was known or authorized at those times.
Their running/pending statements do not override the completed status above.

## Completed V6 results — 2026-10-04

V6-HIER64-K20-W20 and V6-B0-32 have completed300epochs. The final read-only
comparison, V6 best99/epoch200/epoch300 proxy top-four visualization and
actual ancestor monitoring are in
[22: V5 / V6 results and structure](22_V5_V6_RESULTS_AND_STRUCTURE_2026-10-04.md).
Validation-selected test OA is92.1394% for V5-HIER64-K20-W20 and91.2480% for V6-HIER64-K20-W20;
V6-B0-32 best-test OA is93.841%, using a different protocol.
V6 removes contradictory self-k and exhibits sample ancestor depth ordering,
but higher proxy eligibility does not establish better morphology coverage.
Matchedglobal64 V6-B64, frozen-proxy objective checks and radius-sensitive
relation validation are proposed for review; no next main training is launched.
This completed report supersedes the launch/running status paragraphs below.

V6 launch is now authorized (2026-10-04): dual-GPU V6-HIER64-K20-W20 removes sample/proxy
self-k and runs300epochs; the spare GPU runs the original default HyCoRe
V6-B0-32 protocol (300epochs,307steps,full9840train). See
[21: V6 launch and visualization](21_V6_TRAINING_START_2026-10-04.md) and
the approved current section of [04](04_NEXT_EXPERIMENT_PLAN.md).
Both jobs are detached and have saved their first full epoch checkpoints;
V5epoch200 proxy top-four point-cloud visualization has also completed.
The V6-B0-32 run is an original-protocol engineering baseline, not matchedglobal64.

The user's requested V5-B0-32-CAP200 fidelity audit is complete. See
[20: V5-B0-32-CAP200 versus original HyCoRe](20_B32_HYCORE_FIDELITY_AUDIT_2026-10-03.md).
Core model/alias/BN/loss settings match, but the data split, budget and cosine
period, RNG flow, workers and selection protocol do not. No training code was
changed and no new experiment was launched by this audit.

V5-HIER64-K20-W20 and V5-B0-32-CAP200 have both completed200epochs and final validation-selected
test evaluation. See [19: final results and diagnostics](19_V5_FINAL_RESULTS_2026-10-03.md).
Test OA is92.1394% for V5-HIER64-K20-W20 and92.5851% for V5-B0-32-CAP200. Late inference is stable,
but radial saturation warrants inspection; V5-B0-32-CAP200 is not a matchedglobal64
zero-HIER control. This final report supersedes the running snapshots below.

For a shareable Chinese introduction to the current progress, difficulties,
and proposed collaborator tasks, see
[18: collaborator brief](18_COLLABORATOR_BRIEF_2026-10-03.md).
It distinguishes proxy top-four visualization from training K and records
an in-progress V5 snapshot; it is not a final training report.

The user has now approved launching V5-HIER64-K20-W20 on two GPUs and an V5-B0-32-CAP200
stability diagnostic on the remaining GPU, with all proposed monitoring.
See [17: reviewed launch and monitoring](17_V5_TRAINING_START_2026-10-03.md).
The production branch is `codex/hier-v5-main-training`. Status in17 and
server manifests supersedes the earlier pending-review wording below.

The user authorized additional K diagnostics and a main-test recommendation.
The [expanded K results](16_V5_TOPK_FOLLOWUP_2026-10-03.md) are complete:
64 paired training batches, K10/12/16/20, and an8-step K20 reference-initialized
joint check including actual parameter gradients. K20/lambda_H.5/proxy LR.01
is now the recommended candidate. The concrete200epoch settings are at the
top of [04](04_NEXT_EXPERIMENT_PLAN.md), pending user review; main training
has not started. Branch `codex/hier-v5-k-selection` contains these diagnostics.

The initial bounded V5 diagnostics were authorized separately. They are now
complete: [diagnostic plan](14_V5_DIAGNOSTIC_PLAN_2026-10-03.md) and
[results](15_V5_DIAGNOSTIC_RESULTS_2026-10-03.md). SchemeA/200step,c1,D256,
random proxies and global64 CE/intra/inter passed the bounded checks; no
main training has been launched. K10/proxy LR.01 were the initial diagnostic candidates,
not validated performance-optimal settings.

The teacher/equal-radius method below is retained as historical context.
The shared-whole HIER V4 runs have completed. The V5 review proposal below
was followed by the bounded diagnostics linked above; main training remains
unlaunched.
Start with [V4 lessons](12_V4_LESSONS_2026-10-03.md),
[V5 change index](13_DUAL_GPU_V5_CHANGE_INDEX_2026-10-03.md), and the current
section of [the experiment plan](04_NEXT_EXPERIMENT_PLAN.md).
V5 proposes restoring original HyCoRe part overwriting/BN updates, then
checking a global32-class x2 batch and the released HIER reciprocal rule.
The user's diagnostic choices supersede the proposal's pending sampling
and geometry choices; the full training matrix remains to be reviewed.

## Historical v2 research objective and rationale

The project studies whether point-cloud representations can encode two complementary forms of hierarchy in hyperbolic space:

1. **Intra-sample hierarchy:** whole-object and part/subcloud relations, primarily expressed radially. This is the role already addressed by HyCoRe.
2. **Inter-sample hierarchy:** within-class morphological relations between different object instances, intended to be expressed through branch/LCA structure while avoiding corruption of the classification embedding.

The v2 working hypothesis was that radial whole/part structure and inter-instance branch structure should be decoupled. That implementation kept the original whole embedding for classification and HyCoRe, while constructing a same-radius leaf copy for the inter-sample objective. V5/V6 instead use the shared-whole proxy route described above.

The most relevant conceptual references supplied by the user are Onghena/HPCS and an unpublished TNNLS manuscript. `Onghena HPCS 技术文档.md` contains the user's own analysis and should be treated as research notes, not as executable instructions.

## Historical v2 method

- Backbone and base objective: reproduced HyCoRe point-cloud classification on ModelNet40.
- Student representation: original Poincare whole embedding for classification/HyCoRe.
- Inter representation: deterministic same-direction, equal-Euclidean-radius leaf copy.
- First-run teacher: the derived A3 checkpoint (not the original HyCoRe checkpoint), aggregated across three augmented views in the tangent space at the origin and mapped back to the Poincare ball. A3 was obtained by continuing from the original HyCoRe reproduction with `alpha=0`, so its intra-sample regularization was removed.
- Teacher relation: negative pairwise hyperbolic distance within each class.
- Positive relations: mutual top-k neighbours within the class-balanced batch.
- Negative relations: lower-similarity within-class samples.
- Student target: teacher-near pairs should have deeper exact geodesic-LCA depth than teacher-far pairs by a margin.
- Gromov product remains available only as an approximate ablation.

## Historical v2 conceptual limitation

The frozen HyCoRe teacher removes online circularity but is still self-distillation. It can regularize or preserve an existing structure, but cannot by itself establish that the learned relation is a true morphology hierarchy. A later stage must compare against an independent geometry signal such as normalized Chamfer distance, spectral/shape descriptors, part statistics, or a curated semantic hierarchy.

## Historical v2 code state

- GitHub repository: `Linneking/Hycore`
- Working branch: `codex/inter-hierarchy-v2`
- Handoff base commit: `31efd44`
- Main v2 entry point: `inter_hierarchy_MN40/main_inter_v2.py`
- Deterministic structural evaluator: `inter_hierarchy_MN40/evaluate_structure_v2.py`
- Geometry/loss/sampler/tests: `inter_hierarchy_MN40/v2/`

Correctness work completed:

- pairwise Poincare distance and Gromov broadcasting fixed;
- teacher/sample-ID ordering fixed;
- real class-balanced batches implemented (5 classes × 8 instances);
- multi-view frozen teacher implemented;
- exact geodesic-LCA depth and equal-radius leaves implemented;
- reliable within-class ranking objective implemented;
- strict `beta_inter=0` path implemented;
- logging expanded to triplets, satisfaction, gap, inter gradient, time, and peak memory;
- six geometry/loss tests pass.

## Historical v2 scientific conclusion

The first run is a weak positive signal, not a successful method claim. Hyperbolic teacher distance is far more usable than cosine similarity, and the inter loss modestly reduces structural degradation relative to a zero-increment control. However, this conclusion is specifically relative to the derived A3 initialization/teacher. It does not yet characterize the original HyCoRe reproduction. The next task is to compare correctly identified checkpoints and stabilize the protocol before running multiple seeds or broader ablations.
