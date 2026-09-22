# Original HyCoRe multi-class radius audit

## Scope and implementation

The deterministic radius-audit script now accepts `--class-label` and `--class-name`, while retaining chair as its backward-compatible default. Three additional ModelNet40 categories were selected for visibly different within-class structures:

- lamp (label 19): desk, floor, curved-arm, and shade/base variants;
- table (label 33): four-leg, trestle/pedestal, long/round, and cabinet-like variants;
- sofa (label 30): armchair-like, straight, curved, and different arm/back structures.

All results use the original HyCoRe best checkpoint at epoch 229 and deterministic, unaugmented ModelNet40 test point clouds. No training or A3 comparison was performed.

## Summary

| Class | Test samples | Accuracy | Radius min | Q1 | Median | Q3 | Max | Radius std | Mean prototype distance | Mean prototype angle |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| lamp | 20 | 85% | 1.0855 | 3.6279 | 4.5586 | 4.9407 | 5.2224 | 1.2498 | 2.8714 | 15.30 deg |
| table | 100 | 78% | 0.5538 | 2.5239 | 3.1346 | 3.7474 | 5.1396 | 0.8398 | 2.4544 | 20.12 deg |
| sofa | 100 | 100% | 1.9037 | 4.2452 | 4.3388 | 4.4065 | 4.6864 | 0.3783 | 1.2226 | 2.62 deg |

The prototype is the tangent-space mean mapped back to the Poincare ball. Angles are measured between `logmap0` directions.

## Preliminary interpretation

### Lamp

Lamp provides the clearest relation between radius and class prototypicality. Large-radius examples are predominantly canonical upright lamps with a shade, straight support, and base. Low-radius examples include desk/curved-arm lamps and unusual cylindrical or bulbous silhouettes. All three misclassified lamp samples fall within the lowest 26.3% of the radius ordering. This supports an interpretation of radius as recognizability or alignment with the learned class direction more strongly than an abstract-to-specific semantic order.

### Table

Table has a broad and continuous radial distribution and substantial morphology variation. Some outer samples are cabinet/box-like and receive very low target-class scores. The class has 22 misclassifications spread from the minimum to maximum radius percentile, so radius is heavily confounded by the table/desk/other furniture decision boundary. Table is informative for studying boundary ambiguity but is not clean evidence for a semantic specificity axis.

### Sofa

All sofa samples are classified correctly, but most lie in a narrow outer radial band: Q1 to maximum spans only 4.2452--4.6864. The innermost outliers include armchair-like or unusual curved silhouettes, while ordinary sofa variants dominate the dense outer band. Radius therefore offers limited resolution for within-sofa morphology in this checkpoint.

## Current cross-class observation

Across chair, lamp, table, and sofa, the recurring pattern is not `larger radius = more attributes or greater semantic specificity`. A more consistent provisional interpretation is:

- large radius often accompanies a confident, canonical member of the predicted class;
- small radius often accompanies an atypical silhouette, class-boundary ambiguity, or misclassification;
- the strength of this pattern varies substantially by category.

This is an observational result only. It argues against using the current raw HyCoRe radius as an automatic specificity label without additional supervision or normalization.

## Artifacts

Server directories:

- `/mnt/BBB5090/jhr/HyCoRe_runs/radius_lamp_original_20260922_v1/`
- `/mnt/BBB5090/jhr/HyCoRe_runs/radius_table_original_20260922_v1/`
- `/mnt/BBB5090/jhr/HyCoRe_runs/radius_sofa_original_20260922_v1/`

Local Git-ignored copies:

- `D:\Hycore\.codex-local\results\radius_lamp_original_20260922_v1\`
- `D:\Hycore\.codex-local\results\radius_table_original_20260922_v1\`
- `D:\Hycore\.codex-local\results\radius_sofa_original_20260922_v1\`

Each directory contains a five-band montage, a three-view inner/outer montage, a statistics plot, a sample-level CSV, extracted embeddings, selected point clouds, JSON summary, and run log.
