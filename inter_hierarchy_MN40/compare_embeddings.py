"""
三组实验嵌入对比可视化脚本
A1 (alpha=0.01), A2 (alpha=0.001), A3 (alpha=0.0)
输出 5 张对比图表到 data_vis/compare/
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'pointnet2_ops_lib'))
os.environ["CUDA_VISIBLE_DEVICES"] = "2"  # 用 GPU 2

import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from scipy.stats import spearmanr

from models.pointmlp import Hype_pointMLP
from data import ModelNet40
from embedding_cache import recompute_embeddings, build_index, compute_full_w_matrix
from hutil import gromov_product
from models.manifolds import PoincareBall


def load_weights(checkpoint_path):
    """加载实验保存的权重"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    sd = checkpoint['net']
    new_sd = {}
    for k, v in sd.items():
        name = k[7:] if k.startswith('module.') else k
        new_sd[name] = v
    return new_sd, checkpoint


def collect_embeddings(weight_path, device):
    """加载权重，全量前向（不用 DataParallel，避免最后一个 batch 的 BN 问题）"""
    net = Hype_pointMLP()
    state_dict, ckpt = load_weights(weight_path)
    net.load_state_dict(state_dict, strict=True)
    net = net.to(device)
    net.eval()

    train_set = ModelNet40(partition='train', num_points=1024)
    loader = DataLoader(train_set, num_workers=4, batch_size=32, shuffle=False, drop_last=True)

    # 直接裸模型全量前向，不做 DataParallel（避免最后 batch size=1 时 BN 出错）
    all_mu = []
    all_labels = []
    with torch.no_grad():
        for data, label, _ in loader:
            data = data.to(device).permute(0, 2, 1)
            parent_mu, _ = net(data)
            all_mu.append(parent_mu.detach().cpu())
            all_labels.append(label.cpu())

    Z_all = torch.cat(all_mu, dim=0)
    labels = torch.cat(all_labels, dim=0).squeeze()
    print(f"  Collected {Z_all.shape[0]} embeddings")
    return Z_all, labels.numpy(), ckpt


def plot_embedding_tsne(ax, Z_all, labels, title, ball):
    """庞加莱球上的 2D t-SNE 投影"""
    Z_np = Z_all.numpy()
    if Z_np.shape[0] > 2000:
        idx = np.random.choice(Z_np.shape[0], 2000, replace=False)
        Z_sub = Z_np[idx]
        labels_sub = labels[idx]
    else:
        Z_sub = Z_np
        labels_sub = labels

    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    Z_2d = tsne.fit_transform(Z_sub)

    # 画庞加莱圆盘边界
    circle = plt.Circle((0, 0), 1, fill=False, color='gray', linestyle='--', linewidth=0.5)
    ax.add_patch(circle)

    scatter = ax.scatter(Z_2d[:, 0], Z_2d[:, 1], c=labels_sub, cmap='tab20', s=3, alpha=0.6)
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])


def plot_norm_distribution(axes, name, Z_all, color, ball):
    """绘制 whole 嵌入范数分布直方图"""
    norms = ball.dist0(Z_all).numpy()
    axes.hist(norms, bins=50, alpha=0.6, color=color, label=f'{name} (mean={norms.mean():.3f})')
    axes.set_xlabel('Poincare norm (dist from origin)')
    axes.set_ylabel('Count')
    axes.set_title('Whole Embedding Norm Distribution', fontsize=11)
    axes.legend(fontsize=8)


def plot_wij_intra_matrix(ax, W, labels, class_to_indices, title):
    """绘制各类的类内 w_ij 均值排序"""
    unique_classes = sorted(class_to_indices.keys())
    class_means = {}
    for c in unique_classes:
        indices = class_to_indices[c]
        if len(indices) < 2:
            continue
        vals = []
        for ii, i in enumerate(indices):
            for j in indices[ii + 1:]:
                vals.append(W[i, j])
        class_means[c] = np.mean(vals)

    sorted_c = sorted(class_means.items(), key=lambda x: x[1], reverse=False)
    classes = [c for c, _ in sorted_c]
    means = [m for _, m in sorted_c]

    colors = ['green' if m > 0.98 else 'orange' if m > 0.90 else 'red' for m in means]
    ax.barh(range(len(classes)), means, color=colors, height=0.7)
    ax.set_yticks(range(len(classes)))
    ax.set_yticklabels(classes, fontsize=6)
    ax.set_xlabel('Intra-class w_ij mean')
    ax.set_title(title, fontsize=10)
    ax.axvline(x=0.98, color='green', linestyle='--', alpha=0.5)
    ax.axvline(x=0.90, color='red', linestyle='--', alpha=0.5)


def main():
    device = 'cuda'
    output_dir = os.path.join(os.path.dirname(__file__), 'data_vis', 'compare')
    os.makedirs(output_dir, exist_ok=True)

    experiments = {
        'A1 (α=0.01)': 'checkpoints/InterHierarchy-A1_baseline_warmup-20260626192840/best_checkpoint.pth',
        'A2 (α=0.001)': 'checkpoints/InterHierarchy-A2_relaxed_intra-20260626192840/best_checkpoint.pth',
        'A3 (α=0.0)': 'checkpoints/InterHierarchy-A3_no_rhier-20260626192840/best_checkpoint.pth',
    }

    ball = PoincareBall(c=1.0, dim=256)
    data = {}

    # ── 收集所有嵌入 ──
    print("=" * 60)
    print("Collecting embeddings for 3 experiments...")
    print("=" * 60)
    for name, path in experiments.items():
        if not os.path.exists(path):
            print(f"  [SKIP] {name}: checkpoint not found at {path}")
            continue
        print(f"\n--- {name} ---")
        print(f"  Loading: {path}")
        Z_all, labels, ckpt = collect_embeddings(path, device)
        Z_norm, class_to_indices = build_index(Z_all, torch.from_numpy(labels), verbose=True)
        W = compute_full_w_matrix(Z_norm)

        data[name] = {
            'Z_all': Z_all,
            'labels': labels,
            'Z_norm': Z_norm,
            'class_to_indices': class_to_indices,
            'W': W,
            'ckpt': ckpt,
        }
        print(f"  Best test acc: {ckpt.get('best_test_acc', 'N/A')}%")

    if len(data) < 2:
        print("ERROR: Need at least 2 experiments to compare. Exiting.")
        return

    # ── 图1：庞加莱球 2D t-SNE 投影（三组并排）──
    print("\n[1/4] Generating t-SNE projections...")
    fig, axes = plt.subplots(1, len(data), figsize=(6 * len(data), 5))
    if len(data) == 1:
        axes = [axes]
    for ax, (name, d) in zip(axes, data.items()):
        plot_embedding_tsne(ax, d['Z_all'], d['labels'], name, ball)
    fig.suptitle('Poincare Ball 2D t-SNE (2000 sampled points)', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'tsne_comparison.png'), dpi=150)
    plt.close()
    print(f"  Saved: {output_dir}/tsne_comparison.png")

    # ── 图2：Whole 范数分布直方图 ──
    print("\n[2/4] Generating norm distribution histograms...")
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ['blue', 'orange', 'red']
    for (name, d), color in zip(data.items(), colors):
        plot_norm_distribution(ax, name, d['Z_all'], color, ball)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'norm_distribution.png'), dpi=150)
    plt.close()
    print(f"  Saved: {output_dir}/norm_distribution.png")

    # ── 图3：类内 w_ij 均值排序（三组并排）──
    print("\n[3/4] Generating intra-class w_ij ranking...")
    fig, axes = plt.subplots(1, len(data), figsize=(5 * len(data), 8))
    if len(data) == 1:
        axes = [axes]
    for ax, (name, d) in zip(axes, data.items()):
        plot_wij_intra_matrix(ax, d['W'], d['labels'], d['class_to_indices'], name)
    fig.suptitle('Per-Class Intra w_ij Mean (sorted)', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'class_wij_ranking.png'), dpi=150)
    plt.close()
    print(f"  Saved: {output_dir}/class_wij_ranking.png")

    # ── 图4：Gromov 积 vs w_ij 散点图（三组并排）──
    print("\n[4/4] Generating Gromov vs w_ij scatter plots...")
    fig, axes = plt.subplots(1, len(data), figsize=(6 * len(data), 5))
    if len(data) == 1:
        axes = [axes]
    for ax, (name, d) in zip(axes, data.items()):
        N = d['Z_all'].shape[0]
        G_all = gromov_product(d['Z_all'], d['Z_all'], ball).numpy()
        W = d['W']
        sample_size = 3000
        idx_i = np.random.randint(0, N, sample_size)
        idx_j = np.random.randint(0, N, sample_size)
        w_sampled = W[idx_i, idx_j]
        g_sampled = G_all[idx_i, idx_j]
        rho, pval = spearmanr(w_sampled, g_sampled)
        ax.scatter(w_sampled, g_sampled, alpha=0.3, s=2, c='steelblue')
        ax.set_xlabel('w_ij (Cosine Similarity)')
        ax.set_ylabel('Gromov Product (LCA depth)')
        ax.set_title(f'{name}\nSpearman ρ = {rho:.4f}', fontsize=10)
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.3)
    fig.suptitle('Gromov Product vs w_ij', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'gromov_vs_wij_comparison.png'), dpi=150)
    plt.close()
    print(f"  Saved: {output_dir}/gromov_vs_wij_comparison.png")

    # ── 图5：总结表 ──
    print("\n" + "=" * 60)
    print("SUMMARY: Embedding Comparison")
    print("=" * 60)
    for name, d in data.items():
        norms = ball.dist0(d['Z_all']).numpy()
        intra_vals = []
        inter_vals = []
        N = d['W'].shape[0]
        labels = d['labels']
        W = d['W']
        for i in range(min(N, 5000)):
            for j in range(i + 1, min(N, 5000)):
                if labels[i] == labels[j]:
                    intra_vals.append(W[i, j])
                else:
                    inter_vals.append(W[i, j])
        intra_vals = np.array(intra_vals)
        inter_vals = np.array(inter_vals)

        print(f"\n{name}:")
        print(f"  Best test acc: {d['ckpt'].get('best_test_acc', 'N/A')}%")
        print(f"  Whole norm: mean={norms.mean():.4f}, std={norms.std():.4f}, "
              f"min={norms.min():.4f}, max={norms.max():.4f}")
        print(f"  Intra w_ij: mean={intra_vals.mean():.4f}, std={intra_vals.std():.4f}")
        print(f"  Inter w_ij: mean={inter_vals.mean():.4f}, std={inter_vals.std():.4f}")
        print(f"  Separation: {intra_vals.mean() - inter_vals.mean():.4f}")

    print(f"\nAll visualizations saved to: {output_dir}/")


if __name__ == '__main__':
    main()