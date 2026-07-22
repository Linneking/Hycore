"""CPU 诊断 A1/A2/A3 逐类半径（避开 GPU expmap0 问题）"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'pointnet2_ops_lib'))
import torch, numpy as np
torch.set_num_threads(4)
from torch.utils.data import DataLoader
from models.pointmlp import Hype_pointMLP
from data import ModelNet40
from models.manifolds import PoincareBall

EXPS = {
    'A1 (a=0.01)':  'checkpoints/InterHierarchy-A1_baseline_warmup-20260626192840/best_checkpoint.pth',
    'A2 (a=0.001)': 'checkpoints/InterHierarchy-A2_relaxed_intra-20260626192840/best_checkpoint.pth',
    'A3 (a=0.0)':   'checkpoints/InterHierarchy-A3_no_rhier-20260626192840/best_checkpoint.pth',
}

ball = PoincareBall(c=1.0, dim=256)
device = 'cpu'

def load_net(path):
    net = Hype_pointMLP().to(device)
    sd = torch.load(path, map_location='cpu')['net']
    new_sd = {k[7:] if k.startswith('module.') else k: v for k, v in sd.items()}
    net.load_state_dict(new_sd, strict=True)
    net.eval()
    return net

def collect_norms(net):
    train_set = ModelNet40(partition='train', num_points=1024)
    loader = DataLoader(train_set, num_workers=0, batch_size=8, shuffle=False, drop_last=True)
    Z_all, labels_l = [], []
    with torch.no_grad():
        for batch_idx, (data, label, _) in enumerate(loader):
            data = data.to(device).permute(0, 2, 1)
            parent_mu, _ = net(data)
            Z_all.append(parent_mu.detach().cpu() if device=='cuda' else parent_mu.clone())
            labels_l.append(label)
            if (batch_idx + 1) % 50 == 0:
                print(f"    Processed {batch_idx+1} batches...", flush=True)
    Z_all = torch.cat(Z_all, dim=0)
    labels = torch.cat(labels_l, dim=0).squeeze()
    norms = ball.dist0(Z_all).numpy()
    return norms, labels.numpy()

for name, path in EXPS.items():
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    if not os.path.exists(path):
        print(f"  [SKIP]")
        continue
    net = load_net(path)
    print("  Collecting norms (CPU)...", flush=True)
    norms, labels = collect_norms(net)
    print(f"  Done. {len(norms)} embeddings")

    unique_c = sorted(np.unique(labels))
    class_stats = {}
    for c in unique_c:
        mask = labels == c
        ns = norms[mask]
        class_stats[c] = (ns.mean(), ns.std(), ns.min(), ns.max(), mask.sum())

    sorted_c = sorted(class_stats.items(), key=lambda x: x[1][0], reverse=True)
    max_mean, min_mean = sorted_c[0][1][0], sorted_c[-1][1][0]

    print(f"  Global norm: mean={norms.mean():.4f} std={norms.std():.4f} min={norms.min():.4f} max={norms.max():.4f}")
    print(f"  Inter-class range: {max_mean-min_mean:.4f} (ratio {max_mean/max(min_mean,1e-8):.2f}x)")
    
    intra_stds = [v[1] for v in class_stats.values()]
    print(f"  Intra-class std: mean={np.mean(intra_stds):.4f} max={np.max(intra_stds):.4f} min={np.min(intra_stds):.4f}")
    
    print(f"  Top-5 radius:", flush=True)
    for c, (m, s, mn, mx, n_c) in sorted_c[:5]:
        print(f"    C{c:2d} n={n_c:3d} mean={m:.4f} std={s:.4f} [{mn:.4f},{mx:.4f}]")
    print(f"  Bottom-5 radius:", flush=True)
    for c, (m, s, mn, mx, n_c) in sorted_c[-5:]:
        print(f"    C{c:2d} n={n_c:3d} mean={m:.4f} std={s:.4f} [{mn:.4f},{mx:.4f}]")

    sorted_by_std = sorted(class_stats.items(), key=lambda x: x[1][1], reverse=True)
    print(f"  Top-3 intra-class variance:", flush=True)
    for c, (m, s, mn, mx, n_c) in sorted_by_std[:3]:
        print(f"    C{c:2d} n={n_c:3d} mean={m:.4f} std={s:.4f} [{mn:.4f},{mx:.4f}]")

    del net

print("\nDone.")
