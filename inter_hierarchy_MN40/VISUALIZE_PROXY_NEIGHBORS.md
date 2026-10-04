# Proxy top-four point-cloud visualization

Both entry points use the saved checkpoint's matching `net` and `proxy` states,
clean first-1024-point features from the checkpoint's 8,856 training IDs,
the FP32 training proxy graph for eligibility, and FP64 raw c=1 Poincare
distance for nearest-four retrieval. Ten eligible proxies are drawn uniformly
with seed 22. Each run requires a new output directory and leaves the source
checkpoint and training results untouched.

Run from the repository root in the training Python environment, with a CUDA
device available for PointMLP's native farthest-point sampling:

```bash
python inter_hierarchy_MN40/visualize_v5_proxy_neighbors.py \
  --checkpoint /path/to/v5/H20/checkpoint_epoch_200.pth \
  --data-dir /path/to/modelnet40_ply_hdf5_2048 \
  --output-dir /new/output/v5_epoch200 --device cuda:0

python inter_hierarchy_MN40/visualize_v6_proxy_neighbors.py \
  --checkpoint /path/to/v6/H20/checkpoint_epoch_300.pth --expected-epoch 300 \
  --data-dir /path/to/modelnet40_ply_hdf5_2048 \
  --output-dir /new/output/v6_epoch300 --device cuda:0
```

For V6 epoch 200, use `checkpoint_epoch_200.pth --expected-epoch 200`.
For a V6 validation-best checkpoint, read its recorded epoch first and pass
`best.pth --expected-epoch <that epoch>` (for example, 99). The script rejects
selection-only `best.pth` files because they lack a matching proxy state; it
never combines `best_net` with a proxy from another epoch. V5 continues to
require its complete epoch-200 checkpoint and has no new argument.

The V5 proxy graph permits index self as a negative; V6 excludes it. Both
remove the diagonal from mutual positives and require at least two nonself
reciprocal positives. This eligibility is a graph sampling condition, not
evidence that a proxy is a true morphology ancestor. V6's summary reports
per-proxy hard-Gumbel selection counts only when the checkpoint contains the
actual epoch structure telemetry. Display top-four is independent of training
K20 and does not select by label or class purity.

Outputs include an offline rotatable `proxy_neighbors.html`, static
`proxy_neighbors.png`, `neighbors.csv`, `summary.json`, and compressed clean
feature/point-cloud arrays. Use separate output directories for V5 and each
V6 epoch so identities and weights remain unambiguous.
