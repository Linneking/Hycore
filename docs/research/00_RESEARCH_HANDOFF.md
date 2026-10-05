# Research handoff: HyCoRe inter-sample hierarchy

## Latest H20 whole / radial retrieval diagnosis — 2026-10-06

[27: matched whole geometry and frozen proxy retrieval controls](27_HIER_WHOLE_RADIAL_RETRIEVAL_DIAGNOSIS_2026-10-06.md)
uses ten existing caches, all aligned to the same8856training objects by HDF5
shard name and row. H20's sorted-shard IDs differ from26's glob-order IDs;
canonical remapping passes every label check. No new model forward, GPU,
optimizer update or main training occurred; all source cache hashes remain unchanged.

V6e300 eligible proxy top4 covers28IDs with raw distance,190with whole
equal-radius copies,30with proxy equal-radius copies. Removing the lowest
radius1% only replaces the hotspot:281/485proxies still share one new set,
and87.68%of slots come from the remaining pool's lowest1%. Borrowing same-ID
sourceB32 radii with H20 directions/proxies unchanged shifts hotspots but
retains concentration. This locates a retrieval radial bias, not an isolated
HIER training effect or proof that proxies coincide.

Dresser whole median depth is3.558and within-class angle18.76degrees at
V6e300 versus4.568/2.97degrees at B64last. Overall whole median depths are
4.617/4.684, so the change is class-dependent. Distinguish proxy-to-whole
retrieval from actual pair/triple ancestor use. The next question is loss-wise
radial/angular gradients and proxy objective behavior; no next matrix is launched.

## Latest matched representation audit — 2026-10-05

[26: original HyCoRe versus B64, full class geometry and batch probes](26_ORIGINAL_HYCORE_B64_CLASS_GEOMETRY_2026-10-05.md)
compares historical seed4780 B32, source seed22 B32 and B64 best/last under
identical clean train/test inputs and fixed200/400/600-point patches. All12full
splits and three no-update probes passed identity, finite and buffer checks.
Best clean-test whole mean depth is4.364/4.390/4.139 respectively; B64 is
not uniformly more outward. Original flower_pot is1/20, sourceB32 2/20,
B64 3/20 at best, so this weak class predates the new split.

A more important finding is BN-mode dependence: fixed overwritten whole and
patch400 input gives mean whole-minus-part depth gap.767/.831/.575 in eval
versus3.005/3.117/3.060 in trainBN. Changing local32 membership moves the
geometry while mostly preserving class predictions. These conditional probes
do not prove a batch training causal effect; relation stability across BN modes
and groupings should be checked before interpreting HIER's soft hierarchy.
No new main training was launched. Full40-class tables and sanitized derived
statistics/plots are linked from26.

## Current handoff — 2026-10-05

V6 B64 shuffle has completed all300epochs, finishing at09:24:55Asia/Shanghai.
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

The completed B64 configuration, monitoring and comparisons are in
[23: B64 launch and final results](23_V6_B64_SHUFFLE_START_2026-10-05.md);
V5/V6 H20 morphology/proxy evidence remains in
[22: results and structure](22_V5_V6_RESULTS_AND_STRUCTURE_2026-10-04.md).
Reviewed public curves are in
[the B64 summary plot](artifacts/b64_v6_2026-10-05/monitoring.png) and
[300epoch CSV](artifacts/b64_v6_2026-10-05/epoch_curves.csv).

**Current method:** original HyCoRe CE/intra and online HIER sample/proxy
regularization share the same whole embedding, fixedc1/D256. Proxies are
abstract ancestors, not physical parts; there is no proxy–part loss or teacher
in V5/V6. Original part overwrite, two BN updates and FPS behavior are retained.
The frozen-teacher/equal-radius v2 route below is historical, not current.

B64 establishes that the source-style global64 base training can work. It has
different data, sampling, update budget and test-based selection from H20,
so it does not isolate HIER's causal effect. A matched control, frozen-proxy
gradient diagnostics and independent morphology checks remain proposed for
review in[04](04_NEXT_EXPERIMENT_PLAN.md); no next matrix is authorized here.

## Historical navigation snapshots

The dated entries below record what was known or authorized at those times.
Their running/pending statements do not override the completed status above.

## Completed V6 results — 2026-10-04

V6 H20 and original-source B32 have completed300epochs. The final read-only
comparison, V6 best99/epoch200/epoch300 proxy top-four visualization and
actual ancestor monitoring are in
[22: V5 / V6 results and structure](22_V5_V6_RESULTS_AND_STRUCTURE_2026-10-04.md).
Validation-selected test OA is92.1394% forV5-H20 and91.2480% forV6-H20;
original-source B32 best-test OA is93.841%, using a different protocol.
V6 removes contradictory self-k and exhibits sample ancestor depth ordering,
but higher proxy eligibility does not establish better morphology coverage.
Matchedglobal64 B0, frozen-proxy objective checks and radius-sensitive
relation validation are proposed for review; no next main training is launched.
This completed report supersedes the launch/running status paragraphs below.

V6 launch is now authorized (2026-10-04): dual-GPU H20 removes sample/proxy
self-k and runs300epochs; the spare GPU runs the original default HyCoRe
B32 protocol (300epochs,307steps,full9840train). See
[21: V6 launch and visualization](21_V6_TRAINING_START_2026-10-04.md) and
the approved current section of [04](04_NEXT_EXPERIMENT_PLAN.md).
Both jobs are detached and have saved their first full epoch checkpoints;
V5epoch200 proxy top-four point-cloud visualization has also completed.
The B32 run is an original-protocol engineering baseline, not matchedglobal64.

The user's requested B32 fidelity audit is complete. See
[20: B32 versus original HyCoRe](20_B32_HYCORE_FIDELITY_AUDIT_2026-10-03.md).
Core model/alias/BN/loss settings match, but the data split, budget and cosine
period, RNG flow, workers and selection protocol do not. No training code was
changed and no new experiment was launched by this audit.

V5-H20 and B32 have both completed200epochs and final validation-selected
test evaluation. See [19: final results and diagnostics](19_V5_FINAL_RESULTS_2026-10-03.md).
Test OA is92.1394% forH20 and92.5851% forB32. Late inference is stable,
but radial saturation warrants inspection; B32 is not a matchedglobal64
zero-HIER control. This final report supersedes the running snapshots below.

For a shareable Chinese introduction to the current progress, difficulties,
and proposed collaborator tasks, see
[18: collaborator brief](18_COLLABORATOR_BRIEF_2026-10-03.md).
It distinguishes proxy top-four visualization from training K and records
an in-progress V5 snapshot; it is not a final training report.

The user has now approved launching V5-H20 on two GPUs and an ordinaryB32
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
