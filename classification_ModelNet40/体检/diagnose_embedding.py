"""
HyCoRe 预训练权重嵌入诊断脚本
检查项：
  1. 同类样本 whole 嵌入的 w_ij（欧氏余弦相似度，等价于原点切空间余弦相似度）
  2. whole 嵌入在 Poincaré 球中的深度 dist0（双曲测地线距离）
  3. 单一样本 KNN 子采样得到的 part 嵌入的深度（仅采样 10% 样本）

使用方式：
  cd classification_ModelNet40
  python 体检/diagnose_embedding.py
"""

import os
import sys

# 将 classification_ModelNet40 和 pointnet2_ops_lib 加入搜索路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))  # classification_ModelNet40
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'pointnet2_ops_lib'))

os.environ["CUDA_VISIBLE_DEVICES"] = "2"  # 使用 GPU 2
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from collections import defaultdict

from models.pointmlp import Hype_pointMLP
from models.manifolds import PoincareBall
from data import ModelNet40
from utils import progress_bar

# 复用 hutil 中的 get_children_np（KNN 子采样生成 part 点云）
from hutil import get_children_np


# ══════════════════════════════════════════════════════════
#  配置
# ══════════════════════════════════════════════════════════
CHECKPOINT_PATH = "checkpoints/Hype_PointNet-Offv_pointmlp_hycore_var-4780/best_checkpoint.pth"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BALL = PoincareBall(c=1.0, dim=256)
NUM_POINTS = 1024
BATCH_SIZE = 32

# 类别名称映射（ModelNet40）
CLASS_NAMES = [
    'airplane', 'bathtub', 'bed', 'bench', 'bookshelf',
    'bottle', 'bowl', 'car', 'chair', 'cone',
    'cup', 'curtain', 'desk', 'door', 'dresser',
    'flower_pot', 'glass_box', 'guitar', 'keyboard', 'lamp',
    'laptop', 'mantel', 'monitor', 'night_stand', 'person',
    'piano', 'plant', 'radio', 'range_hood', 'sink',
    'sofa', 'stairs', 'stool', 'table', 'tent',
    'toilet', 'tv_stand', 'vase', 'wardrobe', 'xbox'
]


def load_model(checkpoint_path):
    """加载 HyCoRe 预训练模型"""
    print(f"[Loading] {checkpoint_path}")
    net = Hype_pointMLP().to(DEVICE)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint['net']
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
    net.load_state_dict(new_state_dict, strict=True)
    net.eval()
    epoch = checkpoint.get('epoch', 'unknown')
    acc = checkpoint.get('best_test_acc', 'N/A')
    print(f"  Loaded epoch={epoch}, best_test_acc={acc}%")
    return net


def collect_whole_embeddings(net):
    """遍历训练集，收集所有样本的 whole 嵌入"""
    print("\n[Collect] Whole embeddings from training set...")
    train_set = ModelNet40(partition='train', num_points=NUM_POINTS)
    loader = DataLoader(train_set, num_workers=0, batch_size=BATCH_SIZE,
                        shuffle=False, drop_last=True)

    Z_all, labels_all = [], []
    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(loader):
            data = data.to(DEVICE).permute(0, 2, 1)
            mu, _ = net(data)
            Z_all.append(mu.detach().cpu())
            labels_all.append(label)
            if (batch_idx + 1) % 20 == 0:
                progress_bar(batch_idx, len(loader),
                             f'Whole emb ({batch_idx+1}/{len(loader)})')

    Z_all = torch.cat(Z_all, dim=0)
    labels = torch.cat(labels_all, dim=0).squeeze().numpy()
    print(f"\n  Total: {Z_all.shape[0]} samples")
    return Z_all, labels


def collect_part_embeddings_subsample(net, max_samples_per_class=5):
    """
    对每类最多采 max_samples_per_class 个样本，为其生成 part 嵌入。
    加快速度：不做逐样本 KNN，而是在 batch 内随机子采样。
    """
    print(f"\n[Collect] Part embeddings (subsample max {max_samples_per_class}/class)...")
    train_set = ModelNet40(partition='train', num_points=NUM_POINTS)

    # 建立类到样本索引的映射
    class_to_indices = defaultdict(list)
    for idx in range(len(train_set)):
        _, label = train_set[idx]
        class_to_indices[int(label)].append(idx)

    Z_part_all, labels_part_all = [], []
    for c, indices in sorted(class_to_indices.items()):
        np.random.seed(42)
        chosen = np.random.choice(indices, min(max_samples_per_class, len(indices)), replace=False)
        for idx in chosen:
            pointcloud, _ = train_set[idx]
            data = torch.tensor(pointcloud, dtype=torch.float32).unsqueeze(0).to(DEVICE).permute(0, 2, 1)
            try:
                mar, part_data, n_points = get_children_np(
                    data, starting=NUM_POINTS-1, kmin=200, kmax=600)
                part_mu, _ = net(part_data, emb=True)
                Z_part_all.append(part_mu.detach().cpu())
                labels_part_all.append(c)
            except Exception as e:
                print(f"  Warning: class {c} sample failed: {e}")
                continue

    Z_part_all = torch.cat(Z_part_all, dim=0)
    labels_part = np.array(labels_part_all)
    print(f"  Total part samples: {Z_part_all.shape[0]}")
    return Z_part_all, labels_part


# ══════════════════════════════════════════════════════════
#  Task 1: w_ij 分析（CPU 上做，避免 GPU 内存）
# ══════════════════════════════════════════════════════════
def analyze_wij(Z_all_cpu, labels):
    """按类计算同类样本 whole 嵌入之间的 w_ij"""
    print("\n" + "=" * 70)
    print("  TASK 1: 同类样本 w_ij 分析（原点切空间余弦相似度）")
    print("=" * 70)

    unique_classes = sorted(np.unique(labels))
    per_class_detail = []

    for c in unique_classes:
        mask = labels == c
        idx = np.where(mask)[0]
        n_c = len(idx)
        if n_c < 3:
            continue

        Z_c = Z_all_cpu[idx]  # [n_c, 256]

        Z_norm = F.normalize(Z_c, p=2, dim=1)
        W = (Z_norm @ Z_norm.T).numpy()

        triu_idx = np.triu_indices(n_c, k=1)
        w_vals = W[triu_idx]

        mean_w = float(np.mean(w_vals))
        std_w = float(np.std(w_vals))
        min_w = float(np.min(w_vals))
        max_w = float(np.max(w_vals))

        c_name = CLASS_NAMES[c] if c < len(CLASS_NAMES) else f"class_{c}"
        per_class_detail.append((c, c_name, n_c, mean_w, std_w, min_w, max_w, w_vals))

    # 全局加权统计
    total_n = sum(r[2] for r in per_class_detail)
    global_mean = sum(r[3] * r[2] for r in per_class_detail) / total_n
    global_std = np.sqrt(sum(r[4]**2 * r[2] for r in per_class_detail) / total_n)

    print(f"\n  {'='*55}")
    print(f"  Global w_ij (weighted by class size):")
    print(f"    mean={global_mean:.4f}  std={global_std:.4f}")
    print(f"  {'='*55}")

    per_class_detail.sort(key=lambda x: x[3], reverse=True)
    print(f"\n  Top-10 classes by w_ij mean:")
    print(f"  {'Class':>4s}  {'Name':<15s}  {'N':>4s}  {'mean':>8s}  {'std':>8s}  {'min':>8s}  {'max':>8s}")
    print(f"  {'-'*60}")
    for c, name, n_c, mu, std, mn, mx, _ in per_class_detail[:10]:
        print(f"  {c:4d}  {name:<15s}  {n_c:4d}  {mu:8.4f}  {std:8.4f}  {mn:8.4f}  {mx:8.4f}")

    print(f"\n  Bottom-5 classes by w_ij mean:")
    for c, name, n_c, mu, std, mn, mx, _ in per_class_detail[-5:]:
        print(f"  {c:4d}  {name:<15s}  {n_c:4d}  {mu:8.4f}  {std:8.4f}  {mn:8.4f}  {mx:8.4f}")

    per_class_detail.sort(key=lambda x: x[4], reverse=True)
    print(f"\n  Top-5 classes by w_ij std (highest variance):")
    for c, name, n_c, mu, std, mn, mx, _ in per_class_detail[:5]:
        print(f"  {c:4d}  {name:<15s}  {n_c:4d}  mean={mu:.4f}  std={std:.4f}  [{mn:.4f}, {mx:.4f}]")

    low_sim = sum(1 for r in per_class_detail if r[3] < 0.5)
    high_sim = sum(1 for r in per_class_detail if r[3] > 0.8)
    print(f"\n  Classes with w_ij mean < 0.5: {low_sim}/{len(per_class_detail)}")
    print(f"  Classes with w_ij mean > 0.8: {high_sim}/{len(per_class_detail)}")

    # 全局值分布直方图
    all_w = np.concatenate([r[7] for r in per_class_detail])
    print(f"\n  Global w_ij value distribution (all pairs):")
    print(f"    samples: {len(all_w)}")
    for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
        print(f"    P{p:2d}={np.percentile(all_w, p):.4f}")
    print(f"    mean={all_w.mean():.4f}  std={all_w.std():.4f}  min={all_w.min():.4f}  max={all_w.max():.4f}")

    return per_class_detail


# ══════════════════════════════════════════════════════════
#  Task 2: whole 嵌入深度
# ══════════════════════════════════════════════════════════
def analyze_whole_depth(Z_all, labels, unique_classes):
    """计算 whole 嵌入的 dist0"""
    print("\n" + "=" * 70)
    print("  TASK 2: Whole 嵌入深度分析（Poincaré dist0）")
    print("=" * 70)

    depths = BALL.dist0(Z_all.to(DEVICE)).cpu().numpy()

    print(f"\n  Global depth statistics:")
    print(f"    mean={depths.mean():.4f}  std={depths.std():.4f}")
    print(f"    min={depths.min():.4f}  max={depths.max():.4f}")
    print(f"    median={np.median(depths):.4f}")
    for p in [5, 25, 75, 95]:
        print(f"    P{p}={np.percentile(depths, p):.4f}")

    class_stats = {}
    for c in unique_classes:
        mask = labels == c
        ds = depths[mask]
        if len(ds) < 3:
            continue
        class_stats[c] = (float(ds.mean()), float(ds.std()), float(ds.min()), float(ds.max()), len(ds))

    sorted_by_mean = sorted(class_stats.items(), key=lambda x: x[1][0], reverse=True)
    max_mean_cls = sorted_by_mean[0]
    min_mean_cls = sorted_by_mean[-1]

    print(f"\n  Per-class depth summary:")
    print(f"    Inter-class range: {max_mean_cls[1][0] - min_mean_cls[1][0]:.4f}")
    print(f"    ratio (deepest/shallowest): {max_mean_cls[1][0]/max(min_mean_cls[1][0],1e-8):.2f}x")

    intra_stds = [v[1] for v in class_stats.values()]
    print(f"    Intra-class std: mean={np.mean(intra_stds):.4f}  "
          f"max={np.max(intra_stds):.4f}  min={np.min(intra_stds):.4f}")

    print(f"\n  Top-5 deepest classes:")
    print(f"  {'Class':>4s}  {'Name':<15s}  {'N':>4s}  {'mean':>8s}  {'std':>8s}  {'min':>8s}  {'max':>8s}")
    print(f"  {'-'*60}")
    for c, (mu, std, mn, mx, n) in sorted_by_mean[:5]:
        name = CLASS_NAMES[c] if c < len(CLASS_NAMES) else f"class_{c}"
        print(f"  {c:4d}  {name:<15s}  {n:4d}  {mu:8.4f}  {std:8.4f}  {mn:8.4f}  {mx:8.4f}")

    print(f"\n  Bottom-5 shallowest classes:")
    for c, (mu, std, mn, mx, n) in sorted_by_mean[-5:]:
        name = CLASS_NAMES[c] if c < len(CLASS_NAMES) else f"class_{c}"
        print(f"  {c:4d}  {name:<15s}  {n:4d}  {mu:8.4f}  {std:8.4f}  {mn:8.4f}  {mx:8.4f}")

    sorted_by_std = sorted(class_stats.items(), key=lambda x: x[1][1], reverse=True)
    print(f"\n  Top-5 highest intra-class depth variance:")
    for c, (mu, std, mn, mx, n) in sorted_by_std[:5]:
        name = CLASS_NAMES[c] if c < len(CLASS_NAMES) else f"class_{c}"
        print(f"  {c:4d}  {name:<15s}  {n:4d}  mean={mu:.4f}  std={std:.4f}  [{mn:.4f}, {mx:.4f}]")

    return class_stats, depths


# ══════════════════════════════════════════════════════════
#  Task 3: part 嵌入深度
# ══════════════════════════════════════════════════════════
def analyze_part_depth(Z_part_cpu, labels_part, unique_classes, whole_depths, labels_whole):
    """计算 part 嵌入的 dist0，与 whole 对比"""
    print("\n" + "=" * 70)
    print("  TASK 3: Part 嵌入深度分析（KNN 子采样 → Poincaré dist0）")
    print("=" * 70)

    part_depths = BALL.dist0(Z_part_cpu.to(DEVICE)).cpu().numpy()

    print(f"\n  Global part depth statistics:")
    print(f"    mean={part_depths.mean():.4f}  std={part_depths.std():.4f}")
    print(f"    min={part_depths.min():.4f}  max={part_depths.max():.4f}")
    print(f"    median={np.median(part_depths):.4f}")
    for p in [5, 25, 75, 95]:
        print(f"    P{p}={np.percentile(part_depths, p):.4f}")

    class_stats_part = {}
    for c in unique_classes:
        mask = labels_part == c
        ds = part_depths[mask]
        if len(ds) < 1:
            continue
        class_stats_part[c] = (float(ds.mean()), float(ds.std()), float(ds.min()), float(ds.max()), len(ds))

    # Whole vs Part 对比
    print(f"\n  {'='*55}")
    print(f"  Whole vs Part depth comparison:")
    print(f"  {'='*55}")

    whole_mean = whole_depths.mean()
    part_mean = part_depths.mean()
    print(f"  Whole depth: mean={whole_mean:.4f}  std={whole_depths.std():.4f}")
    print(f"  Part depth:  mean={part_mean:.4f}  std={part_depths.std():.4f}")
    print(f"  Δ (Whole - Part): {whole_mean - part_mean:.4f}")

    class_stats_whole = {}
    for c in unique_classes:
        mask_w = labels_whole == c
        ds_w = whole_depths[mask_w]
        if len(ds_w) >= 3:
            class_stats_whole[c] = (float(ds_w.mean()), float(ds_w.std()), len(ds_w))

    print(f"\n  Per-class Whole vs Part depth:")
    print(f"  {'Class':>4s}  {'Name':<15s}  {'nW':>4s}  {'nP':>4s}  {'WholeMean':>10s}  {'PartMean':>10s}  {'Δ(W-P)':>10s}")
    print(f"  {'-'*78}")

    delta_list = []
    for c in sorted(unique_classes):
        if c in class_stats_whole and c in class_stats_part:
            w_mu = class_stats_whole[c][0]
            p_mu = class_stats_part[c][0]
            nW = class_stats_whole[c][2]
            nP = class_stats_part[c][4]
            delta = w_mu - p_mu
            delta_list.append(delta)
            name = CLASS_NAMES[c] if c < len(CLASS_NAMES) else f"class_{c}"
            sign = "+" if delta >= 0 else "-"
            print(f"  {c:4d}  {name:<15s}  {nW:4d}  {nP:4d}  {w_mu:10.4f}  {p_mu:10.4f}  {sign}{abs(delta):9.4f}")

    pos_delta = sum(1 for d in delta_list if d > 0)
    neg_delta = sum(1 for d in delta_list if d < 0)
    print(f"\n  Summary Δ(Whole-Part):")
    print(f"    Mean Δ: {np.mean(delta_list):.4f}")
    print(f"    Classes Part closer to origin (Δ>0): {pos_delta}/{len(delta_list)}")
    print(f"    Classes Part farther from origin (Δ<0): {neg_delta}/{len(delta_list)}")

    return class_stats_part, part_depths


# ══════════════════════════════════════════════════════════
#  保存
# ══════════════════════════════════════════════════════════
def save_report(per_class_wij, whole_class_stats, part_class_stats, whole_depths, part_depths, labels):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    np.save(os.path.join(OUTPUT_DIR, "whole_depths.npy"), whole_depths)
    np.save(os.path.join(OUTPUT_DIR, "part_depths.npy"), part_depths)
    np.save(os.path.join(OUTPUT_DIR, "labels.npy"), labels)

    with open(os.path.join(OUTPUT_DIR, "report.txt"), 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("HyCoRe 预训练权重嵌入诊断报告\n")
        f.write(f"Checkpoint: {CHECKPOINT_PATH}\n")
        f.write("=" * 70 + "\n\n")

        f.write("[TASK 1] 同类样本 w_ij（原点切空间余弦相似度）\n")
        f.write("-" * 50 + "\n")
        f.write(f"{'Class':>4s}  {'Name':<15s}  {'N':>4s}  {'mean':>8s}  {'std':>8s}  {'min':>8s}  {'max':>8s}\n")
        for c, name, n_c, mu, std, mn, mx, _ in sorted(per_class_wij, key=lambda x: x[0]):
            f.write(f"{c:4d}  {name:<15s}  {n_c:4d}  {mu:8.4f}  {std:8.4f}  {mn:8.4f}  {mx:8.4f}\n")

        f.write("\n\n[TASK 2] Whole 嵌入深度（Poincaré dist0）\n")
        f.write("-" * 50 + "\n")
        f.write(f"Global: mean={whole_depths.mean():.4f} std={whole_depths.std():.4f} "
                f"min={whole_depths.min():.4f} max={whole_depths.max():.4f}\n")
        f.write(f"{'Class':>4s}  {'Name':<15s}  {'N':>4s}  {'mean':>8s}  {'std':>8s}  {'min':>8s}  {'max':>8s}\n")
        for c, (mu, std, mn, mx, n) in sorted(whole_class_stats.items()):
            name = CLASS_NAMES[c] if c < len(CLASS_NAMES) else f"class_{c}"
            f.write(f"{c:4d}  {name:<15s}  {n:4d}  {mu:8.4f}  {std:8.4f}  {mn:8.4f}  {mx:8.4f}\n")

        f.write("\n\n[TASK 3] Part 嵌入深度（KNN 子采样 → Poincaré dist0）\n")
        f.write("-" * 50 + "\n")
        f.write(f"Global: mean={part_depths.mean():.4f} std={part_depths.std():.4f} "
                f"min={part_depths.min():.4f} max={part_depths.max():.4f}\n")
        if part_class_stats:
            f.write(f"{'Class':>4s}  {'Name':<15s}  {'N':>4s}  {'mean':>8s}  {'std':>8s}  {'min':>8s}  {'max':>8s}\n")
            for c, (mu, std, mn, mx, n) in sorted(part_class_stats.items()):
                name = CLASS_NAMES[c] if c < len(CLASS_NAMES) else f"class_{c}"
                f.write(f"{c:4d}  {name:<15s}  {n:4d}  {mu:8.4f}  {std:8.4f}  {mn:8.4f}  {mx:8.4f}\n")

    print(f"\n[Save] 结果已保存到 {OUTPUT_DIR}/")


# ══════════════════════════════════════════════════════════
#  main
# ══════════════════════════════════════════════════════════
def main():
    print("=" * 70)
    print("  HyCoRe 预训练权重嵌入诊断")
    print(f"  Device: {DEVICE}")
    print("=" * 70)

    net = load_model(CHECKPOINT_PATH)

    # Step 1: Whole 嵌入
    Z_all, labels = collect_whole_embeddings(net)
    unique_classes = sorted(np.unique(labels))
    print(f"  Classes: {len(unique_classes)}")

    # Task 1: w_ij（在 CPU 上做）
    per_class_wij = analyze_wij(Z_all, labels)

    # Task 2: whole 深度
    whole_class_stats, whole_depths = analyze_whole_depth(Z_all, labels, unique_classes)

    # Step 2: Part 嵌入（子采样加速）
    Z_part, labels_part = collect_part_embeddings_subsample(net, max_samples_per_class=5)

    # Task 3: part 深度
    part_class_stats, part_depths = analyze_part_depth(
        Z_part, labels_part, unique_classes, whole_depths, labels)

    # Save
    save_report(per_class_wij, whole_class_stats, part_class_stats,
                whole_depths, part_depths, labels)

    print("\n" + "=" * 70)
    print("  诊断完成！")
    print("=" * 70)


if __name__ == '__main__':
    main()