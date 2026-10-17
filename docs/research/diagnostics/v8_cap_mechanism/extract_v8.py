"""CPU-only compact extraction from completed V8; never modifies sources."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import numpy as np
import torch


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--features", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    records = []
    for epoch in (0, 1, 5, 20, 100, 300):
        name = "initialization_e0.pth" if epoch == 0 else f"checkpoint_epoch_{epoch:03d}.pth"
        source = a.run / name
        before = digest(source)
        saved = torch.load(source, map_location="cpu", weights_only=False)
        compact = {k: saved[k] for k in ("proxy", "proxy_optimizer", "training_config",
                   "train_ids", "initialized_model_sha256") if k in saved}
        compact["epoch"], compact["source_sha256"] = epoch, before
        torch.save(compact, a.out / f"proxy_e{epoch:03d}.pt")
        v = saved["proxy"]["tangent_proxies"].float().norm(dim=-1)
        records.append(dict(epoch=epoch, source_sha256=before, cap_count=int((v > 2.3).sum()),
                            norm_min=float(v.min()), norm_median=float(v.median()), norm_max=float(v.max())))
        if epoch in (5, 300):
            torch.save({k: saved[k] for k in ("net", "proxy", "training_config", "train_ids",
                       "initialized_model_sha256")}, a.out / f"model_e{epoch:03d}.pt")
        cache = a.features / f"e{epoch:03d}_whole_cache.npz"
        shutil.copyfile(cache, a.out / cache.name)
        assert digest(source) == before, "Source checkpoint changed"
        del saved, compact
    fields = ["epoch", "step", "proxy_gradient_norm", "proxy_cap_forward_clip_count",
              "proxy_cap_parameter_tangent_max", "proxy_update_norm", "proxy_radial_update_norm",
              "proxy_angular_update_norm", "proxy_signed_radial_update_mean",
              "proxy_tangent_norm_change_mean", "proxy_outward_update_fraction",
              "whole_depth", "part_depth"]
    with (a.run / "steps.jsonl").open() as src, (a.out / "trajectory.csv").open("w", newline="") as dst:
        writer = csv.DictWriter(dst, fieldnames=fields)
        writer.writeheader()
        for line in src:
            row = json.loads(line)
            writer.writerow({k: row[k] for k in fields})
    (a.out / "snapshot_summary.json").write_text(json.dumps(records, indent=2))
    archive = a.out.with_suffix(".tar.gz")
    with tarfile.open(archive, "w:gz") as tar:
        for path in sorted(a.out.iterdir()):
            tar.add(path, arcname=path.name)
    print(json.dumps(dict(output=str(a.out), archive=str(archive), bytes=archive.stat().st_size,
                          snapshots=records)))


if __name__ == "__main__":
    main()
