"""质心切平面 wij 直方图对比：收集6个代表性类"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'pointnet2_ops_lib'))
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import torch, numpy as np
from torch.utils.data import DataLoader
from scipy.stats import spearmanr
from models.pointmlp import Hype_pointMLP
from data import ModelNet40
from models.manifolds import PoincareBall
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
                    num_workers=0, batch_size=4, shuffle=False, drop_last=True)

# 收集6个类：0(compact), 17(compact), 22(compact), 35(compact), 15(divergent), 21(medium)
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
                Z_dict[c].append(parent_mu[i].detach().cpu())
        # 每个类收集至少 300 个（class 15 只收集 150）
        enough = all(len(Z_dict[c]) >= (150 if c == 15 else 300) for c in targets)
        if enough:
            break

for c in targets:
    print(f"  Class {c}: {len(Z_dict[c])} samples")

print("\nComputing wij and plotting...")
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()

for idx, c in enumerate(targets):
    Z_c = torch.stack(Z_dict[c]).to(device)
    n_c = Z_c.shape[0]
    
    # Origin method
    Z_o = torch.nn.functional.normalize(Z_c, p=2, dim=1)
    W_o = (Z_o @ Z_o.T).cpu().numpy()
    
    # Centroid method
    mu_euc = Z_c.mean(dim=0, keepdim=True)
    mu_hyp = ball.expmap0(mu_euc)
    V = ball.logmap(mu_hyp, Z_c)
    V_n = torch.nn.functional.normalize(V, p=2, dim=1)
    W_c = (V_n @ V_n.T).cpu().numpy()
    
    triu = np.triu_indices(n_c, k=1)
    wo = W_o[triu]
    wc = W_c[triu]
    rho, _ = spearmanr(wo, wc)
    
    ax = axes[idx]
    ax.hist(wo, bins=40, alpha=0.5, label=f'Origin (std={wo.std():.4f})', color='blue')
    ax.hist(wc, bins=40, alpha=0.5, label=f'Centroid (std={wc.std():.4f})', color='red')
    ax.axvline(x=wo.mean(), color='blue', linestyle='--', alpha=0.5)
    ax.axvline(x=wc.mean(), color='red', linestyle='--', alpha=0.5)
    ax.set_title(f'Class {c} (n={n_c}, ρ={rho:.3f})', fontsize=11)
    ax.set_xlabel('Cosine similarity')
    ax.set_ylabel('Frequency')
    ax.legend(fontsize=7)
    
    del Z_c; torch.cuda.empty_cache()

plt.suptitle('w_ij Distribution: Origin Tangent Plane vs Class-Centroid Tangent Plane\nA1 (α=0.01) weights on ModelNet40', fontsize=13)
plt.tight_layout()
out_path = os.path.join(os.path.dirname(__file__), 'data_vis', 'centroid_wij_hist.png')
plt.savefig(out_path, dpi=150)
plt.close()
print(f"\nSaved to: {out_path}")

# Also print summary
print("\n" + "="*60)
print("Summary Table")
print("="*60)
print(f"{'Class':>6} {'n':>5} {'orig_std':>10} {'cent_std':>10} {'ratio':>8} {'rho':>8}")
print("-"*52)
for idx, c in enumerate(targets):
    Z_c = torch.stack(Z_dict[c]).to(device)
    n_c = Z_c.shape[0]
    Z_o = torch.nn.functional.normalize(Z_c, p=2, dim=1)
    W_o = (Z_o @ Z_o.T).cpu().numpy()
    mu_euc = Z_c.mean(dim=0, keepdim=True)
    mu_hyp = ball.expmap0(mu_euc)
    V = ball.logmap(mu_hyp, Z_c)
    V_n = torch.nn.functional.normalize(V, p=2, dim=1)
    W_c = (V_n @ V_n.T).cpu().numpy()
    triu = np.triu_indices(n_c, k=1)
    wo = W_o[triu]; wc = W_c[triu]
    rho, _ = spearmanr(wo, wc)
    print(f"{c:>6} {n_c:>5} {wo.std():>10.4f} {wc.std():>10.4f} {wc.std()/max(wo.std(),1e-8):>7.1f}x {rho:>8.4f}")

print("\nDone.")
