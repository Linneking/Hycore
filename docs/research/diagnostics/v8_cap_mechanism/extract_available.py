"""Build a bounded training-only panel using checkpoints already on this server."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from inter_hierarchy_MN40.hier_proxy_scratch_v5.diagnose import Hype_pointMLP, load_shards, seed_all
from inter_hierarchy_MN40.hier_proxy_scratch_v8.geometry import configure_backbone


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--epochs", default="0,1,5,20,40")
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    epochs = list(map(int, a.epochs.split(",")))
    points, labels, _ = load_shards(a.data_dir, "train")
    assert len(labels) == 9840
    rng = np.random.default_rng(22)
    ids = np.concatenate([rng.choice(np.flatnonzero(labels == c), 8, replace=False) for c in range(40)])
    records = []
    for epoch in epochs:
        path = a.run / ("initialization_e0.pth" if epoch == 0 else f"checkpoint_epoch_{epoch:03d}.pth")
        before = hashlib.sha256(path.read_bytes()).hexdigest()
        cp = torch.load(path, map_location="cpu", weights_only=False)
        seed_all(22)
        model = configure_backbone(Hype_pointMLP())
        model.load_state_dict(cp["net"], strict=True)
        model.cuda().eval()
        embeddings = []
        with torch.no_grad():
            for start in range(0, len(ids), 32):
                x = torch.as_tensor(points[ids[start:start+32], :1024].copy(), device="cuda")
                mu, _ = model(x.transpose(1, 2).contiguous())
                embeddings.append(mu.cpu().numpy())
        mu = np.concatenate(embeddings)
        np.savez_compressed(a.out / f"e{epoch:03d}_whole_cache.npz",
            mu=mu, labels=labels[ids], sample_ids=ids,
            proxy_tangent=cp["proxy"]["tangent_proxies"].cpu().numpy())
        compact = {k: cp[k] for k in ("proxy", "proxy_optimizer", "training_config", "train_ids",
                                      "initialized_model_sha256") if k in cp}
        compact["epoch"] = epoch
        torch.save(compact, a.out / f"proxy_e{epoch:03d}.pt")
        if epoch in (5, max(epochs)):
            torch.save({k: cp[k] for k in ("net", "proxy", "training_config", "train_ids",
                                          "initialized_model_sha256") if k in cp},
                       a.out / f"model_e{epoch:03d}.pt")
        assert hashlib.sha256(path.read_bytes()).hexdigest() == before
        records.append(dict(epoch=epoch, source_sha256=before,
            proxy_overcap=int((cp["proxy"]["tangent_proxies"].norm(dim=-1) > 2.3).sum()),
            panel_objects=len(ids), source_train_pool=9840))
        (a.out / "input_manifest.json").write_text(json.dumps(records, indent=2))
        print(json.dumps(records[-1]), flush=True)
        del model, cp
        torch.cuda.empty_cache()
    print("CURRENT_SERVER_PANEL_COMPLETED", flush=True)


if __name__ == "__main__":
    main()

