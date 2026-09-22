# Server checkpoint inventory

Inventory scope: `/mnt/BBB5090/jhr/HyCoRe/`, searched recursively for `*.pth`, `*.pt`, and `*.ckpt` on 2026-09-22.

## ModelNet40 / inter-hierarchy checkpoints

### Original HyCoRe reproduction

Directory:

```text
classification_ModelNet40/checkpoints/Hype_PointNet-Offv_pointmlp_hycore_var-4780/
```

| File | Epoch | Current/checkpoint OA | Best OA | Best AA | Identity |
|---|---:|---:|---:|---:|---|
| `best_checkpoint.pth` | 229 | 94.044 | 94.044 | 91.545 | Original HyCoRe reproduction used as source for A1/A2/A3 |
| `last_checkpoint.pth` | 299 | 92.828 | 94.044 | 91.545 | Final epoch, not the best model |

The original training code uses label-smoothed cross-entropy plus `0.01 * triplet + 0.01 * hierarchy`; it is not pure InfoNCE.

### Derived A-series

All three start from the original HyCoRe `best_checkpoint.pth`, use LR 0.1, run 30 epochs, and keep `warmup_epochs=30`, so their inter-class objective is not active during this run.

| Directory | alpha | Best epoch/OA | Last epoch/OA | Interpretation |
|---|---:|---:|---:|---|
| `inter_hierarchy_MN40/checkpoints/InterHierarchy-A1_baseline_warmup-20260626192840/` | 0.01 | 18 / 93.882 | 29 / 93.760 | Continue with original-strength intra loss |
| `inter_hierarchy_MN40/checkpoints/InterHierarchy-A2_relaxed_intra-20260626192840/` | 0.001 | 26 / 93.639 | 29 / 93.476 | Continue with weaker intra loss |
| `inter_hierarchy_MN40/checkpoints/InterHierarchy-A3_no_rhier-20260626192840/` | 0 | 20 / 93.922 | 29 / 93.517 | Continue without intra loss; label-smoothed CE remains |

### B-series / legacy inter experiments

| Directory | Source | Best epoch/OA | Last epoch/OA | Notes |
|---|---|---:|---:|---|
| `inter_hierarchy_MN40/checkpoints/InterHierarchy-B1_intra_class_a0-20260627004052/` | Original HyCoRe best | 9 / 93.314 | 16 / 90.762 | `alpha=0`, `beta_ic=0.05`; stopped before planned 200 epochs |
| `inter_hierarchy_MN40/experiments/B1_v1/checkpoints/B1-B1_v1-20260627014412/` | A3 best | 167 / 94.003 | 199 / 93.558 | `alpha=0.01`, `beta_ic=0.05`, `beta_inter=0`, warmup 0 |

Several timestamped directories contain only `args.txt`, `log.txt`, and `out.txt`; they contain no weight file and are not listed as checkpoints.

### First corrected v2 run

| Directory/file | Source | Epoch | Notes |
|---|---|---:|---|
| `inter_hierarchy_MN40/runs_v2/control_beta0_s22_e20/last.pth` | A3 best | 19 | Matched v2 zero-increment control |
| `inter_hierarchy_MN40/runs_v2/main_exact_lca_s22_e20/last.pth` | A3 best | 1 | Cosine-teacher run stopped by signal gate |
| `inter_hierarchy_MN40/runs_v2/main_hypdist_exact_lca_s22_e20/last.pth` | A3 best | 19 | Hyperbolic-distance teacher main run |
| `.../main_exact_lca_s22_e20/teacher.pt` | A3 best | — | Feature cache, not model weights |
| `.../main_hypdist_exact_lca_s22_e20/teacher.pt` | A3 best | — | Feature cache, not model weights |

The first version of the v2 saver did not save a separate fine-tuning `best.pth` unless it exceeded the initialization; that bug was fixed after these runs.

## HPCS reference checkpoints

These are unrelated to the HyCoRe ModelNet40 checkpoint family but also exist in the repository:

```text
资料库/HPCS/checkpoints/vndgcnn_backbone/best_model.pth
资料库/HPCS/checkpoints/shapenet/model.ckpt
资料库/HPCS/checkpoints/partnet/<category>/model.ckpt
```

PartNet categories present: Bed, Bottle, Chair, Clock, Dishwasher, Display, Door, Earphone, Faucet, Knife, Lamp, Microwave, Refrigerator, StorageFurniture, Table, TrashCan, and Vase.

## Missing expected paper weights

No separate PointMLP+HyCoRe weight with recorded OA 94.3%, and no separate voting weight with OA 94.5%, was found under the repository. Voting is normally an evaluation procedure over an existing checkpoint rather than a separately trained weight. The available local reproduction peaks at 94.044%. If official paper weights were downloaded elsewhere on the server, they are outside this inventory scope and need a separately specified search location.

## Representation caches to create next

One model checkpoint can provide multiple representations; these are feature caches, not different model weights:

1. `pre_expmap`: the 512-D output of `self.proj` before `expmap0` (Euclidean/tangent representation);
2. `post_mobius`: the 256-D output `mu = self.emb(expmap0(x))` (hyperbolic representation used by the classifier).

For a fair teacher comparison, extract both representations from the same identified checkpoint, same sample IDs, same deterministic views, and same split. Also compare at least the original HyCoRe best and derived A3 best checkpoints.
