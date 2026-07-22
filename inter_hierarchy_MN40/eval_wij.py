"""
w_ij 效果评估脚本
运行方式: python eval_wij.py

功能:
  1. 加载 HyCoRe 预训练权重
  2. 全量计算训练集的 w_ij 矩阵
  3. 输出诊断报告：
     - 同类 vs 跨类 w_ij 分布统计
     - 各类的类内相似度排序
     - w_ij 热力图（选 10 类展示）
     - Gromov 积 vs w_ij 相关性
  4. 将可视化图表保存到 data_vis/ 目录
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'pointnet2_ops_lib'))
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

# 本地导入
from models.pointmlp import Hype_pointMLP
from data import ModelNet40
from embedding_cache import recompute_embeddings, build_index, compute_full_w_matrix
from hutil import gromov_product
from models.manifolds import PoincareBall


def load_pretrained(net, path):
    """加载 HyCoRe 预训练权重（兼容 DataParallel 的 module. 前缀）"""
    checkpoint = torch.load(path, map_location='cpu')
    state_dict = checkpoint['net']
    # 剥离 module. 前缀（checkpoint 是 DataParallel 保存的）
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
    # 加载到裸模型（还未包装 DataParallel）
    net.load_state_dict(new_state_dict, strict=True)
    print(f"Loaded pretrained from epoch {checkpoint['epoch']}, acc={checkpoint.get('best_test_acc')}%")


def analyze_wij(W, labels, class_to_indices, Z_norm, Z_all, output_dir):
    """★ w_ij 全面诊断分析"""
    os.makedirs(output_dir, exist_ok=True)
    N = W.shape[0]
    unique_classes = sorted(class_to_indices.keys())

    # ══════════════════════════════════════════
    # 1. 同类 vs 跨类 w_ij 分布
    # ══════════════════════════════════════════
    intra_vals = []
    inter_vals = []
    for i in range(N):
        for j in range(i + 1, N):
            if labels[i] == labels[j]:
                intra_vals.append(W[i, j])
            else:
                inter_vals.append(W[i, j])

    intra_vals = np.array(intra_vals)
    inter_vals = np.array(inter_vals)

    print("\n" + "=" * 60)
    print("★ 1. 同类 vs 跨类 w_ij 分布")
    print("=" * 60)
    print(f"  同类 (intra-class): mean={intra_vals.mean():.4f}, std={intra_vals.std():.4f}, "
          f"min={intra_vals.min():.4f}, max={intra_vals.max():.4f}")
    print(f"  跨类 (inter-class): mean={inter_vals.mean():.4f}, std={inter_vals.std():.4f}, "
          f"min={inter_vals.min():.4f}, max={inter_vals.max():.4f}")
    print(f"  分离度 (intra_mean - inter_mean): {intra_vals.mean() - inter_vals.mean():.4f}")
    print(f"  ★ 解读: 同类 w_ij 应显著 > 跨类 w_ij，差值越大说明预训练嵌入的方向聚类越好")

    # 画分布直方图
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].hist(intra_vals, bins=40, alpha=0.6, label='Intra-class', color='green')
    axes[0].hist(inter_vals, bins=40, alpha=0.6, label='Inter-class', color='red')
    axes[0].set_xlabel('Cosine Similarity (w_ij)')
    axes[0].set_ylabel('Count')
    axes[0].set_title('w_ij Distribution: Intra vs Inter Class')
    axes[0].legend()
    axes[0].axvline(x=intra_vals.mean(), color='darkgreen', linestyle='--')
    axes[0].axvline(x=inter_vals.mean(), color='darkred', linestyle='--')

    axes[1].boxplot([intra_vals, inter_vals], labels=['Intra-class', 'Inter-class'])
    axes[1].set_ylabel('Cosine Similarity (w_ij)')
    axes[1].set_title('w_ij Box Plot')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '1_wij_distribution.png'), dpi=150)
    plt.close()
    print(f"  图表已保存: {os.path.join(output_dir, '1_wij_distribution.png')}")

    # ══════════════════════════════════════════
    # 2. 各类类内 w_ij 均值排序
    # ══════════════════════════════════════════
    print("\n" + "=" * 60)
    print("★ 2. 各类类内 w_ij 均值（Top-10 & Bottom-10）")
    print("=" * 60)
    class_intra_means = {}
    for c in unique_classes:
        indices = class_to_indices[c]
        if len(indices) < 2:
            continue
        vals = []
        for ii, i in enumerate(indices):
            for j in indices[ii + 1:]:
                vals.append(W[i, j])
        class_intra_means[c] = np.mean(vals)

    sorted_classes = sorted(class_intra_means.items(), key=lambda x: x[1], reverse=True)
    print("  Top-10 类内最紧凑的类别:")
    for c, mean_w in sorted_classes[:10]:
        print(f"    Class {c:2d} ({len(class_to_indices[c]):3d} samples): intra_w = {mean_w:.4f}")
    print("  Bottom-10 类内最发散的类别:")
    for c, mean_w in sorted_classes[-10:]:
        print(f"    Class {c:2d} ({len(class_to_indices[c]):3d} samples): intra_w = {mean_w:.4f}")

    # ══════════════════════════════════════════
    # 3. w_ij 热力图
    # ══════════════════════════════════════════
    print("\n" + "=" * 60)
    print("★ 3. w_ij 热力图（10 类 × 每类 8 样本）")
    print("=" * 60)

    display_classes = sorted_classes[:10]
    sample_indices = []
    class_boundaries = [0]
    for c, _ in display_classes:
        idx = class_to_indices[c][:8]
        sample_indices.extend(idx)
        class_boundaries.append(class_boundaries[-1] + len(idx))

    W_sub = W[np.ix_(sample_indices, sample_indices)]

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(W_sub, cmap='RdYlGn', vmin=0, vmax=1)
    for b in class_boundaries[1:-1]:
        ax.axhline(y=b - 0.5, color='black', linewidth=1.5)
        ax.axvline(x=b - 0.5, color='black', linewidth=1.5)
    ax.set_title('w_ij Heatmap (10 classes, 8 samples each)\n'
                 'Expected: block-diagonal pattern (green blocks on diagonal)')
    plt.colorbar(im, ax=ax, label='Cosine Similarity')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '2_wij_heatmap.png'), dpi=150)
    plt.close()
    print(f"  图表已保存: {os.path.join(output_dir, '2_wij_heatmap.png')}")

    # 块内/块外均值比
    block_sum, off_block_sum = 0, 0
    block_count, off_block_count = 0, 0
    row_offset = 0
    for ic, (c, _) in enumerate(display_classes):
        n_c = min(8, len(class_to_indices[c]))
        for ii in range(n_c):
            for jj in range(n_c):
                if ii == jj:
                    continue
                block_sum += W_sub[row_offset + ii, row_offset + jj]
                block_count += 1
        # 与其他类的块外求和
        for jc, (c2, _) in enumerate(display_classes):
            if ic == jc:
                continue
            n_c2 = min(8, len(class_to_indices[c2]))
            off_start = sum(min(8, len(class_to_indices[display_classes[k][0]])) for k in range(jc))
            for ii in range(n_c):
                for jj in range(n_c2):
                    off_block_sum += W_sub[row_offset + ii, off_start + jj]
                    off_block_count += 1
        row_offset += n_c

    if block_count > 0 and off_block_count > 0:
        block_avg = block_sum / block_count
        off_avg = off_block_sum / off_block_count
        print(f"  块内均值: {block_avg:.4f}, 块外均值: {off_avg:.4f}")
        print(f"  块内/块外比值: {block_avg / max(off_avg, 1e-8):.2f}x")

    # ══════════════════════════════════════════
    # 4. Gromov 积 vs w_ij 相关性
    # ══════════════════════════════════════════
    print("\n" + "=" * 60)
    print("★ 4. Gromov 积 vs w_ij 相关性分析")
    print("=" * 60)

    ball = PoincareBall(c=1.0, dim=256)
    G_all = gromov_product(Z_all, Z_all, ball).numpy()

    sample_size = 5000
    idx_i = np.random.randint(0, N, sample_size)
    idx_j = np.random.randint(0, N, sample_size)
    w_sampled = W[idx_i, idx_j]
    g_sampled = G_all[idx_i, idx_j]
    rho, pval = spearmanr(w_sampled, g_sampled)
    print(f"  Spearman ρ(Gromov积, w_ij) = {rho:.4f} (p={pval:.6f})")
    print(f"  ★ 解读: 正相关表明 w_ij 大的对，其 LCA 也更深（Gromov 积更大）")
    print(f"           该值应随 inter-sample 训练逐步增大")

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(w_sampled, g_sampled, alpha=0.3, s=2)
    ax.set_xlabel('w_ij (Cosine Similarity)')
    ax.set_ylabel('Gromov Product (LCA depth)')
    ax.set_title(f'Gromov Product vs w_ij\nSpearman ρ = {rho:.4f}')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '3_gromov_vs_wij.png'), dpi=150)
    plt.close()
    print(f"  图表已保存: {os.path.join(output_dir, '3_gromov_vs_wij.png')}")

    # ══════════════════════════════════════════
    # 5. 总结
    # ══════════════════════════════════════════
    print("\n" + "=" * 60)
    print("★ 5. 总结")
    print("=" * 60)
    separation = intra_vals.mean() - inter_vals.mean()
    print(f"  w_ij 分离度: {separation:.4f}  {'✓ 良好' if separation > 0.1 else '✗ 不足' if separation > 0 else '✗ 反向'}")
    print(f"  Gromov-w 相关性: {rho:.4f}  {'✓ 显著' if rho > 0.1 else '✗ 弱'}")
    if separation > 0.1 and rho > 0.1:
        print(f"    → w_ij 质量良好，可直接进入 inter-sample 训练")
    elif separation > 0:
        print(f"    → w_ij 方向正确但不够强，建议延长 warmup")
    else:
        print(f"    → w_ij 存在问题，请确认预训练权重是否正常加载")


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pretrained_path = 'checkpoints/InterHierarchy-A3_no_rhier-20260626192840/best_checkpoint.pth'

    print("=" * 60)
    print("w_ij 效果评估")
    print("=" * 60)

    print("\n[1/4] Loading model...")
    net = Hype_pointMLP()
    net = net.to(device)
    load_pretrained(net, pretrained_path)   # 加载到裸模型（剥离 module. 前缀）
    net = torch.nn.DataParallel(net)        # 再包装 DataParallel

    print("\n[2/4] Loading data...")
    train_set = ModelNet40(partition='train', num_points=1024)
    full_loader = DataLoader(train_set, num_workers=4, batch_size=32, shuffle=False, drop_last=False)

    print("\n[3/4] Computing embeddings & w_ij...")
    Z_all, all_labels = recompute_embeddings(net, full_loader, device, verbose=True)
    Z_norm, class_to_indices = build_index(Z_all, all_labels, verbose=True)
    W = compute_full_w_matrix(Z_norm)
    print(f"  w_ij matrix shape: {W.shape}, memory: {W.nbytes / 1024 / 1024:.1f} MB")

    print("\n[4/4] Analyzing w_ij quality...")
    output_dir = os.path.join(os.path.dirname(__file__), 'data_vis')
    analyze_wij(W, all_labels.numpy(), class_to_indices, Z_norm, Z_all, output_dir)

    print(f"\n所有图表已保存到: {output_dir}/")
    print("请查看 PNG 文件以获取可视化结果。")


if __name__ == '__main__':
    main()