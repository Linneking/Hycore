"""快速诊断 A1 vs A3（采样版）"""
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
ball = PoincareBall(c=1.0, dim=256)

def diagnose(name, cp_path):
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    net = Hype_pointMLP().to(device)
    sd = torch.load(cp_path, map_location='cpu')['net']
    new_sd = {k[7:] if k.startswith('module.') else k: v for k, v in sd.items()}
    net.load_state_dict(new_sd, strict=True)
    net.eval()

    loader = DataLoader(ModelNet40(partition='train', num_points=1024),
                        num_workers=4, batch_size=32, shuffle=False, drop_last=True)
    Z_all, labels_l = [], []
    with torch.no_grad():
        for data, label, _ in loader:
            data = data.to(device).permute(0, 2, 1)
            parent_mu, _ = net(data)
            Z_all.append(parent_mu.detach().cpu())
            labels_l.append(label.cpu())
    Z_all = torch.cat(Z_all, dim=0)
    labels_np = torch.cat(labels_l, dim=0).squeeze().numpy()
    N = Z_all.shape[0]
    print(f"  {N} embeddings")

    Z_norm, class_to_indices = build_index(Z_all, torch.from_numpy(labels_np), verbose=False)
    W = compute_full_w_matrix(Z_norm)

    # ── wij分布 (采样 100K 对) ──
    n_sample = 100000
    idx_i = np.random.randint(0, N, n_sample)
    idx_j = np.random.randint(0, N, n_sample)
    mask_same = labels_np[idx_i] == labels_np[idx_j]
    intra_s = W[idx_i[mask_same], idx_j[mask_same]]
    inter_s = W[idx_i[~mask_same], idx_j[~mask_same]]
    print(f"  w_ij Intra: mean={intra_s.mean():.4f} std={intra_s.std():.4f} n={len(intra_s)}")
    print(f"  w_ij Inter: mean={inter_s.mean():.4f} std={inter_s.std():.4f} n={len(inter_s)}")

    # ── Spearman (采样5K对) ──
    G_all = gromov_product(Z_all, Z_all, ball).numpy()
    n_s = min(5000, N)
    ii = np.random.randint(0, N, n_s)
    jj = np.random.randint(0, N, n_s)
    rho, pv = spearmanr(W[ii,jj], G_all[ii,jj])
    print(f"  Spearman rho = {rho:.4f} (p={pv:.6e})")

    # ── Per-class w_ij std (采样) ──
    unique_c = sorted(class_to_indices.keys())
    class_std = []
    for c in unique_c:
        idx = class_to_indices[c]
        if len(idx) < 5: continue
        # 每个类随机采样 200 对
        n_pairs = min(200, len(idx)*(len(idx)-1)//2)
        pairs = set()
        while len(pairs) < n_pairs:
            a, b = np.random.choice(idx, 2, replace=False)
            pairs.add((min(a,b), max(a,b)))
        vals = [W[a,b] for a,b in pairs]
        class_std.append((c, np.mean(vals), np.std(vals), len(idx)))
    sorted_w = sorted(class_std, key=lambda x: x[1], reverse=True)
    print(f"  Per-class w_ij (Top-3/Bottom-3):")
    for c,m,s,n_c in sorted_w[:3]:
        print(f"    C{c:2d} n={n_c:3d} mean={m:.4f} std={s:.4f}")
    print(f"    ...")
    for c,m,s,n_c in sorted_w[-3:]:
        print(f"    C{c:2d} n={n_c:3d} mean={m:.4f} std={s:.4f}")
    all_s = [v[2] for v in class_std]
    print(f"  Intra w_ij std stats: mean={np.mean(all_s):.4f} max={np.max(all_s):.4f} min={np.min(all_s):.4f}")

    # ── Norm ──
    norms = ball.dist0(Z_all).numpy()
    class_norms = {}
    for c in unique_c:
        ns = norms[labels_np==c]
        class_norms[c] = (ns.mean(), ns.std())
    sorted_n = sorted(class_norms.items(), key=lambda x: x[1][0], reverse=True)
    rng = sorted_n[0][1][0] - sorted_n[-1][1][0]
    ratio = sorted_n[0][1][0] / max(sorted_n[-1][1][0], 1e-8)
    intra_std_mean = np.mean([v[1] for v in class_norms.values()])
    intra_std_max = np.max([v[1] for v in class_norms.values()])
    print(f"  Norm: global_mean={norms.mean():.4f} inter_range={rng:.4f} ({ratio:.2f}x) intra_std={intra_std_mean:.4f}/{intra_std_max:.4f}")

    del net, Z_all, W, G_all
    torch.cuda.empty_cache()
    return dict(intra_mean=intra_s.mean(), intra_std=intra_s.std(),
                inter_mean=inter_s.mean(), rho=rho, intra_wij_std=all_s,
                norm_range=rng, intra_norm_std=intra_std_mean, intra_norm_std_max=intra_std_max)

r1 = diagnose("A1 (α=0.01)", "checkpoints/InterHierarchy-A1_baseline_warmup-20260626192840/best_checkpoint.pth")
r3 = diagnose("A3 (α=0.0)",  "checkpoints/InterHierarchy-A3_no_rhier-20260626192840/best_checkpoint.pth")

print("\n"+"="*60)
print("  A1 vs A3 对比总结")
print("="*60)
for k in ['intra_mean','intra_std','inter_mean','rho','norm_range','intra_norm_std','intra_norm_std_max']:
    print(f"  {k:20s}: A1={r1[k]:.4f}  A3={r3[k]:.4f}")
print(f"  intra_wij_std_mean: A1={np.mean(r1['intra_wij_std']):.4f}  A3={np.mean(r3['intra_wij_std']):.4f}")
print("\nDone.")
