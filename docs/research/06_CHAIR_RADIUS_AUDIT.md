# Original HyCoRe chair-radius audit

## Scope

This diagnostic implements only the approved first four steps. It does not compare A3, compute LCA/Gromov quantities, or train a model.

- Checkpoint: `classification_ModelNet40/checkpoints/Hype_PointNet-Offv_pointmlp_hycore_var-4780/best_checkpoint.pth`
- Checkpoint epoch/OA/AA metadata: 229 / 94.044 / 91.545
- Data: all 100 chair samples in the unaugmented ModelNet40 test split
- Points per object: 1,024
- Model prediction accuracy on these chair samples: 97/100
- Script: `inter_hierarchy_MN40/diagnostics/chair_radius_audit.py`
- Server artifacts: `/mnt/BBB5090/jhr/HyCoRe_runs/chair_radius_original_20260922_v1/`
- Local ignored copy: `D:\Hycore\.codex-local\results\chair_radius_original_20260922_v1\`

## Quantitative result

Hyperbolic radius is `d(o,z)` for the original whole-object HyCoRe embedding.

| Statistic | Radius |
|---|---:|
| minimum | 1.6517 |
| Q1 | 4.0356 |
| median | 4.2765 |
| Q3 | 4.4045 |
| maximum | 4.7382 |
| mean | 4.1070 |
| standard deviation | 0.5135 |

The class has substantial radial range, but most samples form a dense outer band around 4.0--4.6 and a small number of low-radius outliers account for much of the range.

The chair prototype is defined by averaging `logmap0(z)` and mapping that mean back with `expmap0`. Mean hyperbolic distance to this prototype is 1.5782 (standard deviation 0.9117). Mean tangent-direction angle to the prototype is 5.217 degrees (standard deviation 10.667 degrees).

## Classification-related observation

All three misclassified chairs occur in the lowest 10% of the radius distribution:

| Sample ID | Radius percentile | Predicted label | Prototype distance | Prototype angle |
|---:|---:|---:|---:|---:|
| 1239 | 2.02% | 32 (`stool`) | 4.8775 | 51.25 deg |
| 2262 | 5.05% | 32 (`stool`) | 6.0780 | 72.79 deg |
| 1505 | 9.09% | 10 (`cup`) | 6.1434 | 60.42 deg |

This is evidence that radius is strongly entangled with class recognizability or class-direction alignment. It is not yet evidence for an abstract-to-specific semantic axis.

The classifier outputs are points produced by the repository's hyperbolic classifier. Their ordinary softmax values are recorded only as a relative diagnostic and must not be interpreted as calibrated probabilities.

## Visual inspection note

The selected montage contains four samples at each of the minimum, Q1, median, Q3, and maximum regions. A separate figure renders the four innermost and four outermost samples from three views.

Preliminary inspection does not show a monotonic `ordinary -> attribute-rich` progression. Several innermost samples have unusual silhouettes (lounger/pedestal/stool-like), while many outer samples are readily recognizable conventional chairs. This preliminary interpretation must be checked by the user against the saved images before deciding what radius represents.

## Artifacts

- `chair_radius_montage.png`: five radial bands for manual morphology inspection.
- `chair_radius_extremes_multiview.png`: inner/outer extremes from three views.
- `chair_radius_statistics.png`: radius histogram and scatter plots.
- `chair_radius_samples.csv`: all 100 sample-level measurements.
- `chair_embeddings.pt`: extracted embeddings and tangent features (server only).
- `selected_chair_pointclouds.npz`: selected raw point clouds (server only).
- `summary.json` and run log.
