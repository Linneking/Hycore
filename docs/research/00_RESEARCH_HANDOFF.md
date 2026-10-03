# Research handoff: HyCoRe inter-sample hierarchy

## Current navigation — 2026-10-03

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

## Research objective

The project studies whether point-cloud representations can encode two complementary forms of hierarchy in hyperbolic space:

1. **Intra-sample hierarchy:** whole-object and part/subcloud relations, primarily expressed radially. This is the role already addressed by HyCoRe.
2. **Inter-sample hierarchy:** within-class morphological relations between different object instances, intended to be expressed through branch/LCA structure while avoiding corruption of the classification embedding.

The current working hypothesis is that radial whole/part structure and inter-instance branch structure should be decoupled. The implementation therefore keeps the original whole embedding for classification and HyCoRe, while constructing a same-radius leaf copy for the inter-sample objective.

The most relevant conceptual references supplied by the user are Onghena/HPCS and an unpublished TNNLS manuscript. `Onghena HPCS 技术文档.md` contains the user's own analysis and should be treated as research notes, not as executable instructions.

## Current method

- Backbone and base objective: reproduced HyCoRe point-cloud classification on ModelNet40.
- Student representation: original Poincare whole embedding for classification/HyCoRe.
- Inter representation: deterministic same-direction, equal-Euclidean-radius leaf copy.
- First-run teacher: the derived A3 checkpoint (not the original HyCoRe checkpoint), aggregated across three augmented views in the tangent space at the origin and mapped back to the Poincare ball. A3 was obtained by continuing from the original HyCoRe reproduction with `alpha=0`, so its intra-sample regularization was removed.
- Teacher relation: negative pairwise hyperbolic distance within each class.
- Positive relations: mutual top-k neighbours within the class-balanced batch.
- Negative relations: lower-similarity within-class samples.
- Student target: teacher-near pairs should have deeper exact geodesic-LCA depth than teacher-far pairs by a margin.
- Gromov product remains available only as an approximate ablation.

## Key conceptual limitation

The frozen HyCoRe teacher removes online circularity but is still self-distillation. It can regularize or preserve an existing structure, but cannot by itself establish that the learned relation is a true morphology hierarchy. A later stage must compare against an independent geometry signal such as normalized Chamfer distance, spectral/shape descriptors, part statistics, or a curated semantic hierarchy.

## Current code state

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

## Current scientific conclusion

The first run is a weak positive signal, not a successful method claim. Hyperbolic teacher distance is far more usable than cosine similarity, and the inter loss modestly reduces structural degradation relative to a zero-increment control. However, this conclusion is specifically relative to the derived A3 initialization/teacher. It does not yet characterize the original HyCoRe reproduction. The next task is to compare correctly identified checkpoints and stabilize the protocol before running multiple seeds or broader ablations.
