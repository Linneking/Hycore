"""Quick Gromov-w ρ for B1's best checkpoint"""
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

# Find latest checkpoint
import glob
ckpt_dirs = sorted(glob.glob("experiments/B1_v1/checkpoints/*/"))
latest = ckpt_dirs[-1] + "best_checkpoint.pth"
print(f"Loading: {latest}")

ckpt = torch.load(latest, map_location='cpu')
sd = {k[7:] if k.startswith('module.') else k: v for k, v in ckpt['net'].items()}
net = Hype_pointMLP().to('cuda')
net.load_state_dict(sd, strict=True)
net.eval()

loader = DataLoader(ModelNet40(partition='train', num_points=1024), num_workers=4, batch_size=32, shuffle=False, drop_last=True)
ball = PoincareBall(c=1.0, dim=256)
all_mu, all_labels = [], []
with torch.no_grad():
    for data, label, _ in loader:
        data = data.to('cuda').permute(0, 2, 1)
        parent_mu, _ = net(data)
        all_mu.append(parent_mu.detach().cpu())
        all_labels.append(label.cpu())

Z_all = torch.cat(all_mu, dim=0)
labels = torch.cat(all_labels, dim=0).squeeze().numpy()
print(f"Collected {Z_all.shape[0]} embeddings")

Z_norm, _ = build_index(Z_all, torch.from_numpy(labels), verbose=False)
W = compute_full_w_matrix(Z_norm)
G_all = gromov_product(Z_all, Z_all, ball).numpy()

N = Z_all.shape[0]
idx_i = np.random.randint(0, N, 5000)
idx_j = np.random.randint(0, N, 5000)
rho, pval = spearmanr(W[idx_i, idx_j], G_all[idx_i, idx_j])
print(f"B1 Gromov-w Spearman ρ = {rho:.4f} (p={pval:.6f})")
print(f"Epoch: {ckpt.get('epoch', 'N/A')}, Best test acc: {ckpt.get('best_test_acc', 'N/A')}%")