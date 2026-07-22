"""全局 PCA 可视化：所有6类嵌入在同一 Poincaré盘上展示"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'pointnet2_ops_lib'))
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import torch, numpy as np
from torch.utils.data import DataLoader
from models.pointmlp import Hype_pointMLP
from data import ModelNet40
from models.manifolds import PoincareBall
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ball = PoincareBall(c=1.0, dim=256)
device = 'cuda'
cp = 'checkpoints/InterHierarchy-A1_baseline_warmup-20260626192840/best_checkpoint.pth'

print("Loading A1...")
net = Hype_pointMLP().to(device)
sd = torch.load(cp, map_location='cpu')['net']
new_sd = {k[7:] if k.startswith('module.') else k: v for k, v in sd.items()}
net.load_state_dict(new_sd, strict=True)
net.eval()

loader = DataLoader(ModelNet40(partition='train', num_points=1024),
                    num_workers=0, batch_size=2, shuffle=False, drop_last=True)
targets = [0, 17, 22, 35, 15, 21]
Z_dict = {c: [] for c in targets}
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

print(f"Collecting...")
with torch.no_grad():
    for data, label, _ in loader:
        data = data.to(device).permute(0, 2, 1)
        parent_mu, _ = net(data)
        for i in range(len(label)):
            c = label[i].item()
            if c in targets:
                Z_dict[c].append(parent_mu[i].cpu())
        if all(len(Z_dict[c]) >= (100 if c == 15 else 200) for c in targets):
            break

# Concatenate all classes
all_z = []
all_labels_arr = []
for c in targets:
    zc = torch.stack(Z_dict[c]).numpy()
    all_z.append(zc)
    all_labels_arr.extend([c] * zc.shape[0])
all_z = np.concatenate(all_z, axis=0)  # [N_total, 256]
print(f"Total: {all_z.shape[0]} embeddings")

# Global PCA 256→2
pca = PCA(n_components=2)
Z_2d = pca.fit_transform(all_z)
var_ratio = pca.explained_variance_ratio_.sum()

# Compute centroids in PCA space
centroids_2d = {}
for c in targets:
    mask = np.array(all_labels_arr) == c
    centroids_2d[c] = Z_2d[mask].mean(axis=0)

# Origin in PCA space
origin_2d = pca.transform(np.zeros((1, 256)))[0]

# Global scaling
max_abs = max(np.abs(Z_2d).max(), np.abs(list(centroids_2d.values())).max(), np.abs(origin_2d).max(), 1e-8)
scale_factor = 0.85 / max_abs
Z_2d *= scale_factor
origin_2d *= scale_factor
for c in targets:
    centroids_2d[c] *= scale_factor

# ── Plot ──
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Left: all classes together
ax = axes[0]
circle = plt.Circle((0, 0), 0.9, fill=False, color='gray', linestyle='--', linewidth=0.8)
ax.add_patch(circle)
for i, c in enumerate(targets):
    mask = np.array(all_labels_arr) == c
    ax.scatter(Z_2d[mask, 0], Z_2d[mask, 1], s=1, alpha=0.4, c=colors[i], label=f'C{c}')
    ax.scatter(centroids_2d[c][0], centroids_2d[c][1], s=60, c=colors[i], marker='*', 
               edgecolors='black', linewidths=1, zorder=5)
ax.scatter(origin_2d[0], origin_2d[1], s=80, c='black', marker='+', linewidths=3, label='Origin', zorder=6)
ax.set_title(f'All 6 Classes (PC var={var_ratio:.1%})', fontsize=12)
ax.set_xlim(-1.05, 1.05); ax.set_ylim(-1.05, 1.05); ax.set_aspect('equal')
ax.legend(fontsize=6, ncol=2)

# Right: per-class subplots (zoom on each class's cluster)
axes_right = axes[1].inset_axes([0, 0, 1, 1]) if False else None
# Actually, let's use a cleaner approach: 6 small subplots on the right
import matplotlib.gridspec as gridspec
fig2, axes2 = plt.subplots(2, 3, figsize=(14, 9))
axes2 = axes2.flatten()

for idx, c in enumerate(targets):
    mask = np.array(all_labels_arr) == c
    zc_2d = Z_2d[mask]
    n_c = zc_2d.shape[0]
    
    ax = axes2[idx]
    circle = plt.Circle((0, 0), 0.9, fill=False, color='gray', linestyle='--', linewidth=0.8)
    ax.add_patch(circle)
    ax.scatter(zc_2d[:, 0], zc_2d[:, 1], s=1, alpha=0.4, c=colors[idx])
    ax.scatter(centroids_2d[c][0], centroids_2d[c][1], s=80, c='red', marker='*', edgecolors='darkred', linewidths=2)
    ax.scatter(origin_2d[0], origin_2d[1], s=60, c='black', marker='+', linewidths=2)
    # O→C arrow
    ax.arrow(origin_2d[0], origin_2d[1],
             centroids_2d[c][0] - origin_2d[0], centroids_2d[c][1] - origin_2d[1],
             head_width=0.01, head_length=0.015, fc='red', ec='red', alpha=0.5)
    
    oc_dist = np.sqrt((centroids_2d[c][0]-origin_2d[0])**2 + (centroids_2d[c][1]-origin_2d[1])**2)
    oc_angle = np.arctan2(centroids_2d[c][1]-origin_2d[1], centroids_2d[c][0]-origin_2d[0]) * 180 / np.pi
    ax.set_title(f'Class {c} (n={n_c})  |OC|={oc_dist:.3f}  ∠={oc_angle:.0f}°', fontsize=9)
    ax.set_xlim(-1.05, 1.05); ax.set_ylim(-1.05, 1.05); ax.set_aspect('equal')
    
    print(f'Class {c}: n={n_c}  |O→C|={oc_dist:.3f}  ∠={oc_angle:.1f}°')

fig2.suptitle('Poincare Disk PCA: Per-Class Zoom (Global PCA 256→2)\nA1 weights, ModelNet40', fontsize=13)
fig2.tight_layout()
out_path2 = os.path.join(os.path.dirname(__file__), 'data_vis', 'pca_poincare_perclass.png')
fig2.savefig(out_path2, dpi=150)
plt.close(fig2)

# Save overview
fig.suptitle('Poincare Disk Global PCA: All 6 Classes Together\nA1 weights, ModelNet40', fontsize=13)
fig.tight_layout()
out_path1 = os.path.join(os.path.dirname(__file__), 'data_vis', 'pca_poincare_all.png')
fig.savefig(out_path1, dpi=150)
plt.close(fig)

print(f'PC var total: {var_ratio:.1%}')
print(f'Saved: {out_path1}')
print(f'Saved: {out_path2}')
