"""快速计算 A1/A2/A3 的 Gromov-w Spearman ρ"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'pointnet2_ops_lib'))
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import torch
import numpy as np
from torch.utils.data import DataLoader
from scipy.stats import spearmanr
from models.pointmlp import Hype_pointMLP
from data import ModelNet40
from embedding_cache import build_index, compute_full_w_matrix
from hutil import gromov_product
from models.manifolds import PoincareBall

experiments = {
    'A1 (α=0.01)': '../inter_hierarchy_MN40/checkpoints/InterHierarchy-A1_baseline_warmup-20260626192840/best_checkpoint.pth',
    'A2 (α=0.001)': '../inter_hierarchy_MN40/checkpoints/InterHierarchy-A2_relaxed_intra-20260626192840/best_checkpoint.pth',
    'A3 (α=0.0)': '../inter_hierarchy_MN40/checkpoints/InterHierarchy-A3_no_rhier-20260626192840/best_checkpoint.pth',
}

device = 'cuda'
ball = PoincareBall(c=1.0, dim=256)

for name, path in experiments.items():
    print(f"\n--- {name} ---")
    ckpt = torch.load(path, map_location='cpu')
    sd = {k[7:] if k.startswith('module.') else k: v for k, v in ckpt['net'].items()}
    net = Hype_pointMLP().to(device)
    net.load_state_dict(sd, strict=True)
    net.eval()

    loader = DataLoader(ModelNet40(partition='train', num_points=1024), num_workers=4, batch_size=32, shuffle=False, drop_last=True)
    all_mu, all_labels = [], []
    with torch.no_grad():
        for data, label, _ in loader:
            data = data.to(device).permute(0, 2, 1)
            parent_mu, _ = net(data)
            all_mu.append(parent_mu.detach().cpu())
            all_labels.append(label.cpu())

    Z_all = torch.cat(all_mu, dim=0)
    labels = torch.cat(all_labels, dim=0).squeeze().numpy()
    print(f"  Collected {Z_all.shape[0]} embeddings")

    Z_norm, _ = build_index(Z_all, torch.from_numpy(labels), verbose=False)
    W = compute_full_w_matrix(Z_norm)
    G_all = gromov_product(Z_all, Z_all, ball).numpy()

    N = Z_all.shape[0]
    idx_i = np.random.randint(0, N, 5000)
    idx_j = np.random.randint(0, N, 5000)
    rho, pval = spearmanr(W[idx_i, idx_j], G_all[idx_i, idx_j])
    print(f"  Gromov-w Spearman ρ = {rho:.4f} (p={pval:.6f})")

print("\nDone.")