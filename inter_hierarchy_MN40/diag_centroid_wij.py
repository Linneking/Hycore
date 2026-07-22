"""验证方案2：类质心切平面的 wij vs 原点切平面的 wij"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'pointnet2_ops_lib'))
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import torch, numpy as np
from torch.utils.data import DataLoader
from scipy.stats import spearmanr
from models.pointmlp import Hype_pointMLP
from data import ModelNet40
from models.manifolds import PoincareBall

ball = PoincareBall(c=1.0, dim=256)
device = 'cuda'

def logmap_tangent(ball, base_point, points):
    """将 points 映射到 base_point 处的切空间
    geoopt logmap(base_point, points) 返回切向量
    """
    v = ball.logmap(base_point, points)  # [N, dim] 切向量
    # L2 归一化切向量以计算余弦相似度
    v_norm = torch.nn.functional.normalize(v, p=2, dim=1)
    return v_norm

def diagnose_centroid(name, cp_path):
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
    labels = torch.cat(labels_l, dim=0).squeeze().numpy()
    N = Z_all.shape[0]
    print(f"  {N} embeddings")

    # ── 按类别计算两种 wij ──
    unique_c = sorted(np.unique(labels))
    results = []  # (class, n, w_origin_std, w_centroid_std, rho_oc)
    
    for c in unique_c:
        mask = labels == c
        idx = np.where(mask)[0]
        n_c = len(idx)
        if n_c < 5:
            continue
        
        Z_c = Z_all[idx].to(device)  # [n_c, dim]
        
        # 方法1：原点切平面 (当前方法) = L2 归一化后内积
        Z_origin_norm = torch.nn.functional.normalize(Z_c.clone(), p=2, dim=1)
        W_origin = (Z_origin_norm @ Z_origin_norm.T).cpu().numpy()  # [n_c, n_c]
        
        # 方法2：类质心切平面
        # Step 1: 欧氏质心
        centroid_euc = Z_c.mean(dim=0, keepdim=True)  # [1, dim]
        # Step 2: 投影回庞加莱球 (expmap0 近似)
        centroid_hyp = ball.expmap0(centroid_euc)  # [1, dim]
        # Step 3: logmap 得到切向量
        V_centroid = logmap_tangent(ball, centroid_hyp, Z_c)  # [n_c, dim]
        W_centroid = (V_centroid @ V_centroid.T).cpu().numpy()  # [n_c, n_c]
        
        # 提取上三角（不含对角线）
        triu_idx = np.triu_indices(n_c, k=1)
        w_orig = W_origin[triu_idx]
        w_cent = W_centroid[triu_idx]
        
        # Spearman
        rho_oc, _ = spearmanr(w_orig, w_cent)
        results.append((c, n_c, w_orig.std(), w_cent.std(), rho_oc))
    
    # 汇总
    stds_orig = [r[2] for r in results]
    stds_cent = [r[3] for r in results]
    rhos = [r[4] for r in results]
    
    print(f"\n  {'='*55}")
    print(f"  Per-class summary (n={len(results)} classes)")
    print(f"  {'='*55}")
    print(f"  w_origin std:  mean={np.mean(stds_orig):.4f}  max={np.max(stds_orig):.4f}  min={np.min(stds_orig):.4f}")
    print(f"  w_centroid std: mean={np.mean(stds_cent):.4f}  max={np.max(stds_cent):.4f}  min={np.min(stds_cent):.4f}")
    print(f"  std ratio (cent/orig): {np.mean(stds_cent)/max(np.mean(stds_orig),1e-8):.2f}x")
    print(f"  Spearman ρ(origin, centroid): mean={np.mean(rhos):.4f}  median={np.median(rhos):.4f}")
    
    # 详细列出 std 变化最大的类
    ratios = [(r[0], r[1], r[3]/max(r[2],1e-8), r[4]) for r in results]
    ratios.sort(key=lambda x: x[2], reverse=True)
    print(f"\n  Top-5 classes with largest std amplification:")
    for c, n_c, ratio, rho in ratios[:5]:
        print(f"    C{c:2d} n={n_c:3d}  cent/orig={ratio:.2f}x  ρ={rho:.4f}")
    print(f"  Top-5 classes with smallest std amplification:")
    for c, n_c, ratio, rho in ratios[-5:]:
        print(f"    C{c:2d} n={n_c:3d}  cent/orig={ratio:.2f}x  ρ={rho:.4f}")
    
    # 高 ρ 类的比例
    n_good = sum(1 for r in rhos if r > 0.5)
    print(f"\n  Classes with ρ > 0.5: {n_good}/{len(rhos)} ({100*n_good/len(rhos):.1f}%)")
    
    del net, Z_all
    torch.cuda.empty_cache()
    return dict(stds_orig=stds_orig, stds_cent=stds_cent, rhos=rhos)

r1 = diagnose_centroid("A1 (α=0.01)", "checkpoints/InterHierarchy-A1_baseline_warmup-20260626192840/best_checkpoint.pth")
print("\n\nDone.")
