"""Ultra-mini centroid wij test"""
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
cp = 'checkpoints/InterHierarchy-A1_baseline_warmup-20260626192840/best_checkpoint.pth'

net = Hype_pointMLP().to(device)
sd = torch.load(cp, map_location='cpu')['net']
new_sd = {k[7:] if k.startswith('module.') else k: v for k, v in sd.items()}
net.load_state_dict(new_sd, strict=True)
net.eval()

loader = DataLoader(ModelNet40(partition='train', num_points=1024),
                    num_workers=0, batch_size=4, shuffle=False, drop_last=True)
Z_dict = {0: [], 15: [], 22: []}
label_dict = {0: [], 15: [], 22: []}

print("Collecting A1 class 0/15/22...")
with torch.no_grad():
    for data, label, _ in loader:
        data = data.to(device).permute(0, 2, 1)
        parent_mu, _ = net(data)
        for i in range(len(label)):
            c = label[i].item()
            if c in (0, 15, 22):
                Z_dict[c].append(parent_mu[i].detach().cpu())
                label_dict[c].append(c)
        if all(len(Z_dict[c]) > 0 for c in [0, 15, 22]):
            pass  # keep collecting for more data
        if len(Z_dict[0]) >= 300 and len(Z_dict[15]) >= 100 and len(Z_dict[22]) >= 300:
            break

print(f"  Class 0: {len(Z_dict[0])}, Class 15: {len(Z_dict[15])}, Class 22: {len(Z_dict[22])}")

for c in [0, 22, 15]:
    Z_c = torch.stack(Z_dict[c]).to(device)
    n_c = Z_c.shape[0]
    # Origin
    Z_o = torch.nn.functional.normalize(Z_c, p=2, dim=1)
    W_o = (Z_o @ Z_o.T).cpu().numpy()
    # Centroid
    mu_euc = Z_c.mean(dim=0, keepdim=True)
    mu_hyp = ball.expmap0(mu_euc)
    V = ball.logmap(mu_hyp, Z_c)
    V_n = torch.nn.functional.normalize(V, p=2, dim=1)
    W_c = (V_n @ V_n.T).cpu().numpy()
    
    triu = np.triu_indices(n_c, k=1)
    wo, wc = W_o[triu], W_c[triu]
    rho, _ = spearmanr(wo, wc)
    print(f"\n  Class {c}: n={n_c}")
    print(f"    w_origin:  mean={wo.mean():.4f}  std={wo.std():.4f}")
    print(f"    w_centroid: mean={wc.mean():.4f}  std={wc.std():.4f}")
    print(f"    std ratio cent/orig = {wc.std()/max(wo.std(),1e-8):.2f}x")
    print(f"    Spearman ρ = {rho:.4f}")
    del Z_c; torch.cuda.empty_cache()

print("\nDone.")
