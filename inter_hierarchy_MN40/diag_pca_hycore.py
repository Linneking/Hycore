"""HyCoRe 原始权重 PCA 可视化"""
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

device = 'cuda'
cp = '../classification_ModelNet40/checkpoints/Hype_PointNet-Offv_pointmlp_hycore_var-4780/best_checkpoint.pth'

print("Loading HyCoRe original...")
net = Hype_pointMLP()
sd = torch.load(cp, map_location='cpu')['net']
# 这个 checkpoint 保存时包了 DataParallel，需要 strip module. 前缀
new_sd = {}
for k, v in sd.items():
    name = k[7:] if k.startswith('module.') else k
    new_sd[name] = v
net.load_state_dict(new_sd, strict=True)
net = net.to(device)
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

all_z, all_labels_arr = [], []
for c in targets:
    zc = torch.stack(Z_dict[c]).numpy()
    all_z.append(zc)
    all_labels_arr.extend([c] * zc.shape[0])
all_z = np.concatenate(all_z, axis=0)
print(f"Total: {all_z.shape[0]} embeddings")

# 全局 PCA
pca = PCA(n_components=2)
Z_2d = pca.fit_transform(all_z)
var_ratio = pca.explained_variance_ratio_.sum()

centroids_2d = {}
for c in targets:
    mask = np.array(all_labels_arr) == c
    centroids_2d[c] = Z_2d[mask].mean(axis=0)
origin_2d = pca.transform(np.zeros((1, 256)))[0]

max_abs = max(np.abs(Z_2d).max(), np.abs(list(centroids_2d.values())).max(), np.abs(origin_2d).max(), 1e-8)
scale_factor = 0.85 / max_abs
Z_2d *= scale_factor; origin_2d *= scale_factor
for c in targets: centroids_2d[c] *= scale_factor

# 逐类放大图
fig, axes = plt.subplots(2, 3, figsize=(14, 9))
axes = axes.flatten()
for idx, c in enumerate(targets):
    mask = np.array(all_labels_arr) == c
    zc_2d = Z_2d[mask]; n_c = zc_2d.shape[0]
    ax = axes[idx]
    circle = plt.Circle((0,0), 0.9, fill=False, color='gray', linestyle='--', linewidth=0.8)
    ax.add_patch(circle)
    ax.scatter(zc_2d[:,0], zc_2d[:,1], s=1, alpha=0.4, c=colors[idx])
    ax.scatter(centroids_2d[c][0], centroids_2d[c][1], s=80, c='red', marker='*', edgecolors='darkred', linewidths=2)
    ax.scatter(origin_2d[0], origin_2d[1], s=60, c='black', marker='+', linewidths=2)
    ax.arrow(origin_2d[0], origin_2d[1], centroids_2d[c][0]-origin_2d[0], centroids_2d[c][1]-origin_2d[1],
             head_width=0.01, head_length=0.015, fc='red', ec='red', alpha=0.5)
    oc_d = np.sqrt((centroids_2d[c][0]-origin_2d[0])**2 + (centroids_2d[c][1]-origin_2d[1])**2)
    oc_a = np.arctan2(centroids_2d[c][1]-origin_2d[1], centroids_2d[c][0]-origin_2d[0])*180/np.pi
    ax.set_title(f'C{c} n={n_c} |OC|={oc_d:.3f} ∠={oc_a:.0f}°', fontsize=9)
    ax.set_xlim(-1.05,1.05); ax.set_ylim(-1.05,1.05); ax.set_aspect('equal')
    print(f'C{c}: n={n_c} |O→C|={oc_d:.3f} ∠={oc_a:.1f}°')

fig.suptitle(f'HyCoRe Original Weights PCA\nPC var={var_ratio:.1%}', fontsize=13)
fig.tight_layout()
out = os.path.join(os.path.dirname(__file__), 'data_vis', 'pca_hycore_perclass.png')
fig.savefig(out, dpi=150); plt.close()
print(f'Saved: {out}')
