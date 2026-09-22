# Original HyCoRe shared disk projection

## Purpose

Visualize the spatial organization of original HyCoRe whole-object embeddings using one shared projection, so categories can be compared in the same coordinate system. The selected furniture-oriented set is:

`chair, lamp, table, sofa, stool, desk, bed, bookshelf`

An additional overview includes all 40 ModelNet40 categories. All 2,468 unaugmented test objects were embedded with the original epoch-229 HyCoRe checkpoint. No training was performed.

## Projection definitions

### Radial-preserving direction PCA

1. Map each 256-D Poincare embedding to the origin tangent space.
2. Normalize its tangent vector to obtain direction only.
3. Fit a shared uncentered two-component PCA to directions from all 2,468 samples.
4. Project and renormalize the two-dimensional direction.
5. Restore the sample's original Euclidean Poincare-ball radius.

This view preserves exact radial placement but distorts angular relations during 256-D to 2-D projection.

### Tangent PCA plus 2-D expmap

1. Fit shared uncentered PCA directly to all tangent vectors.
2. Project them to two tangent dimensions.
3. Map the projected vectors into a two-dimensional Poincare disk.

This view shows the leading tangent variation but does not preserve original radii.

Neither projection is a hyperbolic isometry. The first two direction components explain 16.96% and 6.67% of direction energy (23.63% total). The first two tangent components explain 14.46% and 6.29% (20.75% total). Consequently, all disk figures are qualitative.

## Selected-class centroids in the radial-preserving view

| Class | Test count | Centroid x | Centroid y | Mean hyperbolic radius | Accuracy |
|---|---:|---:|---:|---:|---:|
| chair | 100 | -0.009 | -0.953 | 4.107 | 97.0% |
| stool | 20 | -0.302 | -0.824 | 4.429 | 90.0% |
| table | 100 | 0.805 | -0.337 | 3.111 | 78.0% |
| desk | 86 | 0.929 | -0.071 | 3.922 | 91.9% |
| sofa | 100 | 0.485 | -0.836 | 4.240 | 100% |
| bed | 100 | 0.558 | -0.777 | 4.194 | 99.0% |
| bookshelf | 100 | 0.602 | 0.756 | 4.266 | 99.0% |
| lamp | 20 | -0.884 | 0.100 | 3.965 | 85.0% |

## Preliminary observations

- Chair and stool occupy adjacent lower-disk directions.
- Table and desk overlap strongly on the right, consistent with both geometric similarity and their observed classification ambiguity.
- Bed and sofa occupy adjacent lower-right sectors.
- Bookshelf forms a compact upper-right sector, separated from the other selected furniture categories.
- Lamp lies on the left side, nearly opposite table/desk in the displayed plane.
- Most correctly recognized samples lie near the disk boundary; low-radius atypical or confused samples extend inward.

The all-class overview resembles angular class sectors around the boundary more than a visible nested tree. This is consistent with a classification representation in which directions separate categories and radius is entangled with class alignment or confidence. The plot alone does not establish genuine inter- or intra-class hierarchy.

Representative point-cloud panels associate left/right/top/bottom/central projected locations with actual object shape. They are useful for manual inspection, but directional extremes in two-dimensional PCA need not be extremes in the original 256-D geometry.

## Artifacts

Script:

- `inter_hierarchy_MN40/diagnostics/embedding_disk_projection.py`

Server:

- `/mnt/BBB5090/jhr/HyCoRe_runs/embedding_disk_original_20260922_v1/`

Local Git-ignored copy:

- `D:\Hycore\.codex-local\results\embedding_disk_original_20260922_v1\`

Main outputs:

- `selected8_disk_projection_comparison.png`
- `selected8_disk_small_multiples.png`
- `selected8_projection_representative_shapes.png`
- `all40_disk_overview.png`
- `embedding_disk_coordinates.csv`
- `class_centroids.csv`
- `projection_data.pt`
- `summary.json`
