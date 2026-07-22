"""
diagnose_wij_v2.py
补充分析：
  Task 1: 用双曲距离代替 w_ij，三种单调变换（-d, 1/(1+d), exp(-d)），
          逐类 + 全局统计均值/方差/上下界/分布
  Task 2: 分析当前 w_ij（余弦相似度，即 L2 归一化内积）的影响因子
          ——w_ij 高的 pair 是角度相近还是半径相近？

使用方式：
  cd classification_ModelNet40
  python 体检/diagnose_wij_v2.py
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'pointnet2_ops_lib'))

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader

from models.pointmlp import Hype_pointMLP
from models.manifolds import PoincareBall
from data import ModelNet40
from utils import progress_bar

# ══════════════════════════════════════════════════════════
#  配置
# ══════════════════════════════════════════════════════════
CHECKPOINT_PATH = "checkpoints/Hype_PointNet-Offv_pointmlp_hycore_var-4780/best_checkpoint.pth"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BALL = PoincareBall(c=1.0, dim=256)
NUM_POINTS = 1024
BATCH_SIZE = 32

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


def load_model_and_embeddings():
    """加载模型 + 收集 whole 嵌入，缓存到磁盘"""
    cache_path = os.path.join(OUTPUT_DIR, "whole_embeddings.npy")
    labels_cache = os.path.join(OUTPUT_DIR, "embeddings_labels.npy")

    if os.path.exists(cache_path) and os.path.exists(labels_cache):
        print("[Cache] Loading cached embeddings...")
        Z_all = torch.tensor(np.load(cache_path))
        labels = np.load(labels_cache)
        print(f"  Loaded {Z_all.shape[0]} embeddings, {len(np.unique(labels))} classes")
        return Z_all, labels

    print(f"[Loading] {CHECKPOINT_PATH}")
    net = Hype_pointMLP().to(DEVICE)
    ckpt = torch.load(CHECKPOINT_PATH, map_location='cpu')
    sd = {k[7:] if k.startswith('module.') else k: v for k, v in ckpt['net'].items()}
    net.load_state_dict(sd, strict=True)
    net.eval()
    print(f"  Loaded epoch={ckpt.get('epoch','?')}, best_acc={ckpt.get('best_test_acc','?')}%")

    print("\n[Collect] Whole embeddings...")
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
            if (batch_idx + 1) % 30 == 0:
                progress_bar(batch_idx, len(loader), f'Emb ({batch_idx+1}/{len(loader)})')
    Z_all = torch.cat(Z_all, dim=0)
    labels = torch.cat(labels_all, dim=0).squeeze().numpy()
    print(f"\n  Total: {Z_all.shape[0]} samples")

    np.save(cache_path, Z_all.numpy())
    np.save(labels_cache, labels)
    print(f"  Saved to {cache_path}")
    return Z_all, labels


# ══════════════════════════════════════════════════════════
#  Task 1: 双曲距离作为 w_ij
# ══════════════════════════════════════════════════════════
def task1_hyperbolic_wij(Z_all_cpu, labels):
    """
    对每类计算 pairwise 双曲距离，然后三种单调变换 → w_ij
    返回 per-class 统计
    """
    print("\n" + "=" * 70)
    print("  TASK 1: 双曲距离作为 w_ij")
    print("=" * 70)

    unique_classes = sorted(np.unique(labels))
    per_class = []

    for c in unique_classes:
        mask = labels == c
        idx = np.where(mask)[0]
        n_c = len(idx)
        if n_c < 3:
            continue

        Z_c = Z_all_cpu[idx].to(DEVICE)  # [n_c, 256]
        # 双曲距离矩阵 [n_c, n_c]
        D = BALL.dist(Z_c.unsqueeze(1), Z_c.unsqueeze(0)).cpu().numpy()
        triu_idx = np.triu_indices(n_c, k=1)
        d_vals = D[triu_idx]

        # 三种变换
        w_neg = -d_vals
        w_recip = 1.0 / (1.0 + d_vals)
        w_exp = np.exp(-d_vals)

        name = CLASS_NAMES[c] if c < len(CLASS_NAMES) else f"class_{c}"
        per_class.append({
            'c': c, 'name': name, 'n': n_c,
            'd_mean': float(np.mean(d_vals)), 'd_std': float(np.std(d_vals)),
            'd_min': float(np.min(d_vals)), 'd_max': float(np.max(d_vals)),
            'neg_mean': float(np.mean(w_neg)), 'neg_std': float(np.std(w_neg)),
            'recip_mean': float(np.mean(w_recip)), 'recip_std': float(np.std(w_recip)),
            'exp_mean': float(np.mean(w_exp)), 'exp_std': float(np.std(w_exp)),
        })

    # 全局汇总：collect all raw values
    all_w_neg, all_w_recip, all_w_exp = [], [], []
    for r in per_class:
        mask = labels == r['c']
        Z_c = Z_all_cpu[mask].to(DEVICE)
        n_c = r['n']
        D = BALL.dist(Z_c.unsqueeze(1), Z_c.unsqueeze(0)).cpu().numpy()
        triu_idx = np.triu_indices(n_c, k=1)
        d = D[triu_idx]
        all_w_neg.append(-d)
        all_w_recip.append(1/(1+d))
        all_w_exp.append(np.exp(-d))

    all_neg = np.concatenate(all_w_neg)
    all_recip = np.concatenate(all_w_recip)
    all_exp = np.concatenate(all_w_exp)

    print(f"\n  {'='*55}")
    print(f"  Global w_ij via hyperbolic distance (all ~1.9M pairs):")
    print(f"  {'='*55}")
    print(f"  {'Transform':>12s}  {'mean':>9s}  {'std':>9s}  {'P5':>9s}  {'P25':>9s}  {'P50':>9s}  {'P75':>9s}  {'P95':>9s}  {'min':>9s}  {'max':>9s}")
    print(f"  {'-'*95}")
    for name, vals in [('w=-d', all_neg), ('w=1/(1+d)', all_recip), ('w=exp(-d)', all_exp)]:
        print(f"  {name:>12s}  {vals.mean():9.4f}  {vals.std():9.4f}  "
              f"{np.percentile(vals,5):9.4f}  {np.percentile(vals,25):9.4f}  "
              f"{np.percentile(vals,50):9.4f}  {np.percentile(vals,75):9.4f}  "
              f"{np.percentile(vals,95):9.4f}  {vals.min():9.4f}  {vals.max():9.4f}")

    # 逐类 Top/Bottom
    per_class.sort(key=lambda x: x['recip_mean'], reverse=True)
    print(f"\n  Top-5 classes by w=1/(1+d) mean (most similar):")
    for r in per_class[:5]:
        print(f"  C{r['c']:2d} {r['name']:<15s} n={r['n']:4d}  "
              f"d={r['d_mean']:.4f}±{r['d_std']:.4f}  "
              f"recip={r['recip_mean']:.4f}  exp={r['exp_mean']:.4f}")

    print(f"\n  Bottom-5 classes by w=1/(1+d) mean (least similar):")
    for r in per_class[-5:]:
        print(f"  C{r['c']:2d} {r['name']:<15s} n={r['n']:4d}  "
              f"d={r['d_mean']:.4f}±{r['d_std']:.4f}  "
              f"recip={r['recip_mean']:.4f}  exp={r['exp_mean']:.4f}")

    # 与余弦相似度的对比
    print(f"\n  {'='*55}")
    print(f"  Resolution comparison (cosine vs hyperbolic transforms):")
    print(f"  {'='*55}")
    # Cosine from first diagnosis: mean≈0.984, std≈0.062, P5≈0.979, P95≈0.9996
    print(f"  Cosine w_ij:       mean=0.984  std=0.062  range=[0.062, 0.999]")
    print(f"  w=1/(1+d):         mean={all_recip.mean():.4f}  std={all_recip.std():.4f}  range=[{all_recip.min():.4f}, {all_recip.max():.4f}]")
    print(f"  w=exp(-d):         mean={all_exp.mean():.4f}  std={all_exp.std():.4f}  range=[{all_exp.min():.4f}, {all_exp.max():.4f}]")
    print(f"  Coefficient of variation: cos={0.062/0.984:.4f}  recip={all_recip.std()/all_recip.mean():.4f}  exp={all_exp.std()/all_exp.mean():.4f}")
    print(f"  ← Larger CoV = higher discriminative power for HypHC softmax")

    return per_class


# ══════════════════════════════════════════════════════════
#  Task 2: w_ij（余弦相似度）的影响因子分析
# ══════════════════════════════════════════════════════════
def task2_wij_factors(Z_all_cpu, labels):
    """
    分析 w_ij（余弦相似度）与角度/半径的关系。
    对每个 pair (i,j) 记录:
      w:   余弦相似度
      dr:  半径差 |r_i - r_j|
      D:   双曲距离
      cos: 真正的余弦（等于 w）
    """
    print("\n" + "=" * 70)
    print("  TASK 2: w_ij（余弦相似度）影响因子分析")
    print("=" * 70)

    # 计算所有嵌入的欧氏范数
    norms = Z_all_cpu.norm(p=2, dim=1).numpy()  # [N]

    unique_classes = sorted(np.unique(labels))
    results_per_class = {}

    # 全局采样（避免 O(N²) 全量计算）
    all_w, all_dr, all_D = [], [], []

    for c in unique_classes:
        mask = labels == c
        idx = np.where(mask)[0]
        n_c = len(idx)
        if n_c < 3:
            continue

        Z_c = Z_all_cpu[idx]
        r_c = norms[idx]

        # w_ij 矩阵
        Z_norm = F.normalize(Z_c, p=2, dim=1)
        W = (Z_norm @ Z_norm.T).numpy()

        # 半径差矩阵
        R_diff = np.abs(r_c[:, None] - r_c[None, :])

        # 双曲距离矩阵（在 GPU 上算）
        D = BALL.dist(Z_c.to(DEVICE).unsqueeze(1), Z_c.to(DEVICE).unsqueeze(0)).cpu().numpy()

        triu_idx = np.triu_indices(n_c, k=1)
        w_vals = W[triu_idx]
        dr_vals = R_diff[triu_idx]
        D_vals = D[triu_idx]

        # 保存到全局
        all_w.append(w_vals)
        all_dr.append(dr_vals)
        all_D.append(D_vals)

        # 按 w_ij 分桶分析半径差
        bins = [(0.999, 1.001), (0.99, 0.999), (0.95, 0.99), (0.8, 0.95), (-1, 0.8)]
        bin_stats = []
        for low, high in bins:
            b_mask = (w_vals >= low) & (w_vals < high)
            if b_mask.sum() > 0:
                bin_stats.append({
                    'range': f'[{low:.3f}, {high:.3f})',
                    'n': int(b_mask.sum()),
                    'dr_mean': float(dr_vals[b_mask].mean()),
                    'dr_std': float(dr_vals[b_mask].std()),
                    'D_mean': float(D_vals[b_mask].mean()),
                    'D_std': float(D_vals[b_mask].std()),
                })

        results_per_class[c] = {
            'n': n_c,
            'w_mean': float(w_vals.mean()),
            'dr_mean': float(dr_vals.mean()),
            'D_mean': float(D_vals.mean()),
            'corr_w_dr': float(np.corrcoef(w_vals, dr_vals)[0, 1]) if len(w_vals) > 2 else 0,
            'corr_w_D': float(np.corrcoef(w_vals, -D_vals)[0, 1]) if len(w_vals) > 2 else 0,
            'bin_stats': bin_stats,
        }

    # 全局拼接
    all_w_global = np.concatenate(all_w)
    all_dr_global = np.concatenate(all_dr)
    all_D_global = np.concatenate(all_D)

    # ── 全局相关分析 ──
    print(f"\n  Global pairwise analysis (sample of {len(all_w_global):,} pairs):")
    print(f"    w_ij (cosine):      mean={all_w_global.mean():.4f}  std={all_w_global.std():.4f}")
    print(f"    |r_i - r_j|:        mean={all_dr_global.mean():.4f}  std={all_dr_global.std():.4f}")
    print(f"    D (hyperbolic dist): mean={all_D_global.mean():.4f}  std={all_D_global.std():.4f}")
    print(f"    corr(w_ij, |r_i-r_j|): {np.corrcoef(all_w_global, all_dr_global)[0,1]:.4f}")
    print(f"    corr(w_ij, -D):        {np.corrcoef(all_w_global, -all_D_global)[0,1]:.4f}")

    # ── 按 w_ij 分桶 ──
    print(f"\n  {'='*55}")
    print(f"  w_ij bucketed analysis — how w_ij relates to radius diff & hyperbolic dist")
    print(f"  {'='*55}")

    bins_global = [(0.999, 1.001), (0.99, 0.999), (0.95, 0.99), (0.9, 0.95), (0.7, 0.9), (-1, 0.7)]
    print(f"  {'w_ij range':>16s}  {'N pairs':>10s}  {'|r_i-r_j| mean':>14s}  {'D mean':>10s}  {'D std':>10s}")
    print(f"  {'-'*65}")
    for low, high in bins_global:
        b_mask = (all_w_global >= low) & (all_w_global < high)
        if b_mask.sum() > 0:
            print(f"  [{low:.3f}, {high:.3f})  {b_mask.sum():10d}  "
                  f"{all_dr_global[b_mask].mean():14.4f}  "
                  f"{all_D_global[b_mask].mean():10.4f}  {all_D_global[b_mask].std():10.4f}")

    # ── 逐类分析：高 w_ij 是否对应小半径差 ──
    print(f"\n  {'='*55}")
    print(f"  Per-class: corr(w_ij, |r_i-r_j|) — negative means high w → small radius diff")
    print(f"  {'='*55}")
    sorted_cls = sorted(results_per_class.items(), key=lambda x: x[1]['corr_w_dr'])
    print(f"  {'Class':>4s}  {'Name':<15s}  {'n':>4s}  {'corr(w,dr)':>10s}  {'corr(w,-D)':>10s}")
    print(f"  {'-'*53}")
    for c, info in sorted_cls[:5]:
        name = CLASS_NAMES[c] if c < len(CLASS_NAMES) else f"class_{c}"
        print(f"  {c:4d}  {name:<15s}  {info['n']:4d}  {info['corr_w_dr']:10.4f}  {info['corr_w_D']:10.4f}")
    print(f"  ...")
    for c, info in sorted_cls[-5:]:
        name = CLASS_NAMES[c] if c < len(CLASS_NAMES) else f"class_{c}"
        print(f"  {c:4d}  {name:<15s}  {info['n']:4d}  {info['corr_w_dr']:10.4f}  {info['corr_w_D']:10.4f}")

    # ── 重点关注 flower_pot ──
    fp = results_per_class.get(15)
    if fp:
        print(f"\n  {'='*55}")
        print(f"  Focus: flower_pot (class 15) — lowest w_ij mean, highest std")
        print(f"  {'='*55}")
        print(f"  w_ij mean={fp['w_mean']:.4f}  |r_i-r_j| mean={fp['dr_mean']:.4f}  D mean={fp['D_mean']:.4f}")
        print(f"  corr(w, dr)={fp['corr_w_dr']:.4f}  corr(w, -D)={fp['corr_w_D']:.4f}")
        print(f"  w_ij buckets:")
        for b in fp['bin_stats']:
            print(f"    {b['range']:>16s}  n={b['n']:6d}  "
                  f"dr={b['dr_mean']:.4f}±{b['dr_std']:.4f}  D={b['D_mean']:.4f}±{b['D_std']:.4f}")

    # ── 保存原始数据 ──
    save_path = os.path.join(OUTPUT_DIR, "wij_factor_data.npz")
    np.savez_compressed(save_path,
                        w=all_w_global, dr=all_dr_global, D=all_D_global)
    print(f"\n  Saved pairwise data to {save_path}")

    return results_per_class


# ══════════════════════════════════════════════════════════
#  main
# ══════════════════════════════════════════════════════════
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=" * 70)
    print("  diagnose_wij_v2: Hyperbolic wij & Cosine factor analysis")
    print(f"  Device: {DEVICE}")
    print("=" * 70)

    Z_all, labels = load_model_and_embeddings()

    # Task 1
    task1_hyperbolic_wij(Z_all, labels)

    # Task 2
    task2_wij_factors(Z_all, labels)

    print("\n" + "=" * 70)
    print("  Done!")
    print("=" * 70)


if __name__ == '__main__':
    main()