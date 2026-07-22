"""6类 Poincaré盘 PCA 可视化：样本点 + 质心 + 原点"""
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

print(f"Collecting classes {targets}...")
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

print("Samples:", {c: len(v) for c, v in Z_dict.items()})

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()

for idx, c in enumerate(targets):
    Z_c = torch.stack(Z_dict[c]).numpy()  # [n_c, 256]
    n_c = Z_c.shape[0]
    
    # PCA 256→2
    pca = PCA(n_components=2)
    Z_2d = pca.fit_transform(Z_c)  # [n_c, 2]
    var_ratio = pca.explained_variance_ratio_.sum()
    
    # 欧氏质心
    centroid_euc = Z_c.mean(axis=0, keepdims=True)
    centroid_2d = pca.transform(centroid_euc)[0]
    
    # 原点（全零向量在 PCA 空间中的投影）
    origin_2d = pca.transform(np.zeros((1, 256)))[0]
    
    # 缩放拟合到单位圆内
    max_abs = max(np.abs(Z_2d).max(), np.abs(centroid_2d).max(), np.abs(origin_2d).max(), 1e-8)
    scale = 0.85 / max_abs
    Z_2d *= scale
    centroid_2d *= scale
    origin_2d *= scale
    
    ax = axes[idx]
    # Poincaré 盘边界
    circle = plt.Circle((0, 0), 0.9, fill=False, color='gray', linestyle='--', linewidth=0.8)
    ax.add_patch(circle)
    
    # 样本点
    ax.scatter(Z_2d[:, 0], Z_2d[:, 1], s=2, alpha=0.5, c='steelblue', label=f'Samples (n={n_c})')
    # 质心
    ax.scatter(centroid_2d[0], centroid_2d[1], s=80, c='red', marker='*', edgecolors='darkred', 
               linewidths=1.5, label='Centroid', zorder=5)
    # 原点
    ax.scatter(origin_2d[0], origin_2d[1], s=60, c='black', marker='+', linewidths=2, label='Origin', zorder=5)
    # O→C 箭头
    ax.arrow(origin_2d[0], origin_2d[1], 
             centroid_2d[0] - origin_2d[0], centroid_2d[1] - origin_2d[1],
             head_width=0.03, head_length=0.04, fc='red', ec='red', alpha=0.6, zorder=4)
    
    # 标注
    o_dist = np.sqrt(centroid_2d[0]**2 + centroid_2d[1]**2)
    o_angle = np.arctan2(centroid_2d[1], centroid_2d[0]) * 180 / np.pi
    ax.set_title(f'Class {c} (n={n_c})\nPC var={var_ratio:.1%}  |C|={o_dist:.2f}  ∠={o_angle:.0f}°', fontsize=10)
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_aspect('equal')
    ax.legend(fontsize=6, loc='lower right')
    
    print(f'Class {c}: PC_var={var_ratio:.1%}  centroid_dist={o_dist:.3f}  centroid_angle={o_angle:.1f}°')

plt.suptitle('Poincare Disk PCA Projection: Sample Embeddings + Class Centroid + Origin\nA1 (α=0.01) weights on ModelNet40', fontsize=13)
plt.tight_layout()
out_path = os.path.join(os.path.dirname(__file__), 'data_vis', 'pca_poincare_disk.png')
plt.savefig(out_path, dpi=150)
plt.close()
print(f'\nSaved to: {out_path}')
