"""快速诊断 A1 权重的 wij 质量 + Spearman ρ + 范数"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'pointnet2_ops_lib'))
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import torch, numpy as np
from torch.utils.data import DataLoader
from scipy.stats import spearmanr
from models.pointmlp import Hype_pointMLP
from data import ModelNet40
from embedding_cache import build_index, compute_full_w_matrix
from hutil import gromov_product
from models.manifolds import PoincareBall

device = 'cuda'
cp = 'checkpoints/InterHierarchy-A1_baseline_warmup-20260626192840/best_checkpoint.pth'

print("Loading A1...")
net = Hype_pointMLP().to(device)
sd = torch.load(cp, map_location='cpu')['net']
new_sd = {k[7:] if k.startswith('module.') else k: v for k, v in sd.items()}
net.load_state_dict(new_sd, strict=True)
net.eval()
print("Done.")

train_set = ModelNet40(partition='train', num_points=1024)
loader = DataLoader(train_set, num_workers=4, batch_size=32, shuffle=False, drop_last=True)

Z_all, labels_l = [], []
with torch.no_grad():
    for data, label, _ in loader:
        data = data.to(device).permute(0, 2, 1)
        parent_mu, _ = net(data)
        Z_all.append(parent_mu.detach().cpu())
        labels_l.append(label.cpu())
Z_all = torch.cat(Z_all, dim=0)
labels_np = torch.cat(labels_l, dim=0).squeeze().numpy()
print(f"Collected {Z_all.shape[0]} embeddings")

Z_norm, class_to_indices = build_index(Z_all, torch.from_numpy(labels_np), verbose=True)
W = compute_full_w_matrix(Z_norm)
N = W.shape[0]
ball = PoincareBall(c=1.0, dim=256)
unique_c = sorted(class_to_indices.keys())

# ── 1. wij 分布 ──
intra_v, inter_v = [], []
for i in range(N):
    for j in range(i+1, N):
        if labels_np[i]==labels_np[j]: intra_v.append(W[i,j])
        else: inter_v.append(W[i,j])
intra_v=np.array(intra_v); inter_v=np.array(inter_v)
print("\n"+"="*60)
print("1. w_ij Intra vs Inter")
print("="*60)
print(f"  Intra: mean={intra_v.mean():.4f} std={intra_v.std():.4f} min={intra_v.min():.4f} max={intra_v.max():.4f}")
print(f"  Inter: mean={inter_v.mean():.4f} std={inter_v.std():.4f} min={inter_v.min():.4f} max={inter_v.max():.4f}")
print(f"  Separation: {intra_v.mean()-inter_v.mean():.4f}")

# ── 2. Spearman ──
G_all = gromov_product(Z_all, Z_all, ball).numpy()
sample_size = min(5000, N*(N-1)//2)
idx_i = np.random.randint(0, N, sample_size)
idx_j = np.random.randint(0, N, sample_size)
rho, pv = spearmanr(W[idx_i,idx_j], G_all[idx_i,idx_j])
print(f"\n  Global Spearman rho = {rho:.4f} (p={pv:.6e})")

# ── 3. Per-class w_ij mean/std ──
class_wij = {}
for c in unique_c:
    idx = class_to_indices[c]
    if len(idx)<2: continue
    vals = [W[i,j] for ii,i in enumerate(idx) for j in idx[ii+1:]]
    class_wij[c] = (np.mean(vals), np.std(vals), len(idx))
sorted_w = sorted(class_wij.items(), key=lambda x:x[1][0], reverse=True)
print("\n"+"="*60)
print("2. Per-class intra w_ij (Top-5 / Bottom-5)")
print("="*60)
for c,(m,s,n_c) in sorted_w[:5]:
    print(f"  C{c:2d} n={n_c:3d} mean={m:.4f} std={s:.4f}")
print("  ...")
for c,(m,s,n_c) in sorted_w[-5:]:
    print(f"  C{c:2d} n={n_c:3d} mean={m:.4f} std={s:.4f}")
all_std = [v[1] for v in class_wij.values()]
print(f"  Intra w_ij std stats: mean={np.mean(all_std):.4f} max={np.max(all_std):.4f} min={np.min(all_std):.4f}")

# ── 4. Per-class norm ──
norms_all = ball.dist0(Z_all).numpy()
class_norm = {}
for c in unique_c:
    ns = norms_all[labels_np==c]
    class_norm[c] = (ns.mean(), ns.std(), ns.min(), ns.max())
sorted_n = sorted(class_norm.items(), key=lambda x:x[1][0], reverse=True)
print("\n"+"="*60)
print("3. Per-class Norm (ball.dist0)")
print("="*60)
print(f"  Global: mean={norms_all.mean():.4f} std={norms_all.std():.4f} min={norms_all.min():.4f} max={norms_all.max():.4f}")
print(f"  Inter-class range: {sorted_n[0][1][0]-sorted_n[-1][1][0]:.4f} (ratio {sorted_n[0][1][0]/max(sorted_n[-1][1][0],1e-8):.2f}x)")
print(f"  Intra-class std: mean={np.mean([v[1] for v in class_norm.values()]):.4f} max={np.max([v[1] for v in class_norm.values()]):.4f}")
print("  Top-5 radius:")
for c,(m,s,mn,mx) in sorted_n[:5]:
    print(f"    C{c:2d}: mean={m:.4f} std={s:.4f} [{mn:.4f},{mx:.4f}]")
print("  Bottom-5 radius:")
for c,(m,s,mn,mx) in sorted_n[-5:]:
    print(f"    C{c:2d}: mean={m:.4f} std={s:.4f} [{mn:.4f},{mx:.4f}]")

# Top-3 intra-class variance
sorted_by_std = sorted(class_norm.items(), key=lambda x:x[1][1], reverse=True)
print("  Top-3 intra-class norm variance:")
for c,(m,s,mn,mx) in sorted_by_std[:3]:
    print(f"    C{c:2d}: mean={m:.4f} std={s:.4f} [{mn:.4f},{mx:.4f}]")

print("\nDone.")
