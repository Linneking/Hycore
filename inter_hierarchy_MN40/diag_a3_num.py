import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'pointnet2_ops_lib'))
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch, numpy as np
from torch.utils.data import DataLoader
from scipy.stats import spearmanr

from models.pointmlp import Hype_pointMLP
from data import ModelNet40
from embedding_cache import build_index, compute_full_w_matrix
from hutil import gromov_product
from models.manifolds import PoincareBall

device = 'cuda'
cp_path = 'checkpoints/InterHierarchy-A3_no_rhier-20260626192840/best_checkpoint.pth'

print("Loading model...")
net = Hype_pointMLP().to(device)
ckpt = torch.load(cp_path, map_location='cpu')
sd = ckpt['net']
new_sd = {k[7:] if k.startswith('module.') else k: v for k, v in sd.items()}
net.load_state_dict(new_sd, strict=True)
net.eval()
print(f"Loaded epoch {ckpt['epoch']}, best_test_acc={ckpt.get('best_test_acc')}%")

train_set = ModelNet40(partition='train', num_points=1024)
loader = DataLoader(train_set, num_workers=4, batch_size=32, shuffle=False, drop_last=True)

# Collect embeddings
all_mu, all_labels_l = [], []
with torch.no_grad():
    for data, label, _ in loader:
        data = data.to(device).permute(0, 2, 1)
        parent_mu, _ = net(data)
        all_mu.append(parent_mu.detach().cpu())
        all_labels_l.append(label.cpu())

Z_all = torch.cat(all_mu, dim=0)
labels_np = torch.cat(all_labels_l, dim=0).squeeze().numpy()
print(f"Collected {Z_all.shape[0]} embeddings (drop_last=True)")

Z_norm, class_to_indices = build_index(Z_all, torch.from_numpy(labels_np), verbose=True)
W = compute_full_w_matrix(Z_norm)
N = W.shape[0]
ball = PoincareBall(c=1.0, dim=256)

# ── 1. w_ij distribution ──
intra_vals, inter_vals = [], []
for i in range(N):
    for j in range(i+1, N):
        if labels_np[i] == labels_np[j]:
            intra_vals.append(W[i,j])
        else:
            inter_vals.append(W[i,j])
intra_vals = np.array(intra_vals)
inter_vals = np.array(inter_vals)
print("\n" + "="*60)
print("1. w_ij Intra vs Inter Class")
print("="*60)
print(f"  Intra: mean={intra_vals.mean():.4f}, std={intra_vals.std():.4f}, min={intra_vals.min():.4f}, max={intra_vals.max():.4f}")
print(f"  Inter: mean={inter_vals.mean():.4f}, std={inter_vals.std():.4f}, min={inter_vals.min():.4f}, max={inter_vals.max():.4f}")
print(f"  Separation: {intra_vals.mean()-inter_vals.mean():.4f}")

# ── 2. Spearman ──
G_all = gromov_product(Z_all, Z_all, ball).numpy()
sample_size = min(5000, N*(N-1)//2)
idx_i = np.random.randint(0, N, sample_size)
idx_j = np.random.randint(0, N, sample_size)
w_sampled = W[idx_i, idx_j]
g_sampled = G_all[idx_i, idx_j]
rho, pval = spearmanr(w_sampled, g_sampled)
print(f"\n  Global Spearman rho = {rho:.4f} (p={pval:.6e})")

# Intra-only Spearman
intra_idx = np.where(idx_i != idx_j)[0]
intra_mask = labels_np[idx_i[intra_idx]] == labels_np[idx_j[intra_idx]]
if intra_mask.sum() > 10:
    rho_i, _ = spearmanr(w_sampled[intra_idx][intra_mask], g_sampled[intra_idx][intra_mask])
    print(f"  Intra-only Spearman rho = {rho_i:.4f} (n={intra_mask.sum()})")

# ── 3. Per-class w_ij ──
print("\n" + "="*60)
print("2. Per-class Intra w_ij (Top-5 & Bottom-5)")
print("="*60)
unique_c = sorted(class_to_indices.keys())
class_means = {}
for c in unique_c:
    idx = class_to_indices[c]
    if len(idx) < 2: continue
    vals = [W[i,j] for ii, i in enumerate(idx) for j in idx[ii+1:]]
    class_means[c] = (np.mean(vals), np.std(vals), len(idx))
sorted_c = sorted(class_means.items(), key=lambda x: x[1][0], reverse=True)
for c, (m, s, n_c) in sorted_c[:5]:
    print(f"  Class {c:2d} (n={n_c:3d}): mean={m:.4f}, std={s:.4f}")
print("  ...")
for c, (m, s, n_c) in sorted_c[-5:]:
    print(f"  Class {c:2d} (n={n_c:3d}): mean={m:.4f}, std={s:.4f}")

# ── 4. Per-class Norm ──
print("\n" + "="*60)
print("3. Per-class Whole Embedding Norm (ball.dist0)")
print("="*60)
norms_all = ball.dist0(Z_all).numpy()
print(f"  Global: mean={norms_all.mean():.4f}, std={norms_all.std():.4f}, min={norms_all.min():.4f}, max={norms_all.max():.4f}")
class_norm_stats = {}
for c in unique_c:
    idx = class_to_indices[c]
    ns = norms_all[idx]
    class_norm_stats[c] = (ns.mean(), ns.std(), ns.min(), ns.max())
class_norm_sorted = sorted(class_norm_stats.items(), key=lambda x: x[1][0], reverse=True)
print("\n  Top-5 largest radius:")
for c, (m, s, mn, mx) in class_norm_sorted[:5]:
    print(f"    Class {c:2d}: mean={m:.4f}, std={s:.4f}, min={mn:.4f}, max={mx:.4f}")
print("  Bottom-5 smallest radius:")
for c, (m, s, mn, mx) in class_norm_sorted[-5:]:
    print(f"    Class {c:2d}: mean={m:.4f}, std={s:.4f}, min={mn:.4f}, max={mx:.4f}")
max_mean = max(v[0] for v in class_norm_stats.values())
min_mean = min(v[0] for v in class_norm_stats.values())
print(f"\n  Max class mean - Min class mean = {max_mean - min_mean:.4f}")
print(f"  Ratio max/min = {max_mean/max(min_mean,1e-8):.2f}x")
print("\n  => If ratio > 2x, Gromov product is radius-dominated, angle meaningless.")
