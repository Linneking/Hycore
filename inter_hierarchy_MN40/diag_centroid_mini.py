"""Mini版：只对A1的3个类做质心切平面验证"""
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

print("Loading A1...")
net = Hype_pointMLP().to(device)
sd = torch.load(cp, map_location='cpu')['net']
new_sd = {k[7:] if k.startswith('module.') else k: v for k, v in sd.items()}
net.load_state_dict(new_sd, strict=True)
net.eval()

loader = DataLoader(ModelNet40(partition='train', num_points=1024),
                    num_workers=0, batch_size=16, shuffle=False, drop_last=True)
Z_all, labels_l = [], []
print("Collecting...")
with torch.no_grad():
    for i, (data, label, _) in enumerate(loader):
        data = data.to(device).permute(0, 2, 1)
        parent_mu, _ = net(data)
        Z_all.append(parent_mu.detach().cpu())
        labels_l.append(label.cpu())
        if (i+1) % 50 == 0:
            print(f"  batch {i+1}", flush=True)
Z_all = torch.cat(Z_all, dim=0)
labels = torch.cat(labels_l, dim=0).squeeze().numpy()
print(f"  {Z_all.shape[0]} embeddings")

# Pick 3 classes: 0 (largest), 15 (most divergent), 22 (compact)
target_classes = [0, 15, 22]
for c in target_classes:
    mask = labels == c
    idx = np.where(mask)[0]
    n_c = len(idx)
    print(f"\n{'='*50}")
    print(f"Class {c} (n={n_c})")
    
    Z_c = Z_all[idx].to(device)
    
    # Origin method
    Z_origin_norm = torch.nn.functional.normalize(Z_c, p=2, dim=1)
    W_origin = (Z_origin_norm @ Z_origin_norm.T).cpu().numpy()
    
    # Centroid method
    centroid_euc = Z_c.mean(dim=0, keepdim=True)
    centroid_hyp = ball.expmap0(centroid_euc)
    V = ball.logmap(centroid_hyp, Z_c)
    V_norm = torch.nn.functional.normalize(V, p=2, dim=1)
    W_centroid = (V_norm @ V_norm.T).cpu().numpy()
    
    triu = np.triu_indices(n_c, k=1)
    w_orig = W_origin[triu]
    w_cent = W_centroid[triu]
    rho, _ = spearmanr(w_orig, w_cent)
    
    print(f"  w_origin:  mean={w_orig.mean():.4f}  std={w_orig.std():.4f}")
    print(f"  w_centroid: mean={w_cent.mean():.4f}  std={w_cent.std():.4f}")
    print(f"  std ratio:  {w_cent.std()/max(w_orig.std(),1e-8):.2f}x")
    print(f"  Spearman ρ(origin, centroid) = {rho:.4f}")
    
    del Z_c
    torch.cuda.empty_cache()

print("\nDone.")
