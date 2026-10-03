# V5 epoch200 proxy neighbourhood visualisation

Read-only inspection of the complete V5-H20 `checkpoint_epoch_200.pth`.
The contemporaneous `net` and `proxy` states are loaded together; `best_net`
is not used. No training step, augmentation or test-set evaluation occurs.

Before using a GPU, inspect `nvidia-smi` and choose a completely idle card.
Use the existing V5 environment with PyTorch, the native PointNet2 FPS
extension, Geoopt, NumPy, h5py and Pillow. The HTML needs no web connection,
Plotly, downloaded JavaScript or browser extension.

From the repository root:

```bash
python inter_hierarchy_MN40/visualize_v5_proxy_neighbors.py \
  --checkpoint /path/to/H20/checkpoint_epoch_200.pth \
  --data-dir /path/to/modelnet40_ply_hdf5_2048 \
  --output-dir /path/to/NEW_proxy_visualisation_directory \
  --device cuda:0 --batch-size 32 --topk 4 --num-proxies 10 \
  --split train_ids --seed 22
```

The output directory must not exist. With `CUDA_VISIBLE_DEVICES`, `cuda:0`
means the first visible physical GPU. Paths in the example are placeholders;
private server locations are not part of this document.

## What is selected

- The default retrieval set is exactly the checkpoint's 8856 unique
  `train_ids`, from sorted official training HDF5 shards. `validation_ids`
  is an explicit alternate set; official test shards are never opened.
- Each sample uses its raw first1024 points and the model's clean `eval()`
  whole embedding in the original 256-dimensional c1 Poincare ball.
- All512 hierarchy proxies use V5's actual `expmap0_c1` mapping and native
  numerical ball projection. The proxy graph uses the source FP32 K20 rule,
  including self, then removes the diagonal. Eligibility requires at least
  two nonself reciprocal neighbours and a nonempty negative pool.
- Ten eligible proxies are chosen uniformly without replacement using
  NumPy `default_rng(seed)`. The selection does not inspect class purity,
  neighbour appearance, radius or degree magnitude.
- All512 proxies are queried against the entire retrieval set using raw
  high-dimensional hyperbolic distance in FP64, without label boosts or
  category filters. Each displayed row contains four distinct sample IDs.
  Ties resolve by sample ID, then position. **Display top4 is not training K20.**

Here “eligible” means the current proxy can anchor the training relation
mining rule. It does not certify a useful learned ancestor. V5 telemetry did
not retain selected pair/triple proxy IDs, so historical per-proxy ancestor
usage is explicitly `unavailable`. The report includes only the saved
epoch200 aggregate activity counts. Replaying Gumbel selection on clean
features would not recover actual training usage.

## Output

- `proxy_neighbors.png`: ten rows by four columns, with a common camera,
  class, global training-shard sample ID and distance in each cell.
- `proxy_neighbors.html`: standalone offline page. Drag any point cloud to
  rotate it, scroll to zoom, double-click to reset, or synchronise views.
- `neighbors.csv`: the selected rows and exact retrieval distances.
- `selected_clouds.npz`: raw FP32 model-input coordinates, IDs, labels,
  proxy IDs and distances for all displayed samples. Arrays are ordered
  `[selected_proxy, neighbour_rank, point, xyz]`.
- `feature_cache.npz`: all split sample IDs/labels and whole features,
  all512 ball proxies, reciprocal mask, eligibility/degrees, and nearest
  four IDs/distances for every proxy.
- `summary.json`: checkpoint hash, current/source commit, split identity,
  device, timing, geometry, selection rule, all proxy degrees, selected rows,
  ancestor-usage limitation and output identity.

Display only centers the bounding box and applies one common scale to each
cloud, preserving its aspect ratio. It does not change features or retrieval.
The NPZ retains the exact input coordinates; embedded HTML coordinates round
to six decimal places. Keep all outputs outside version control.

CPU utility checks (no checkpoint, dataset or GPU required):

```bash
python inter_hierarchy_MN40/test_visualize_v5_proxy_neighbors.py
```

The actual source proxy-graph test additionally runs when PyTorch is present.
