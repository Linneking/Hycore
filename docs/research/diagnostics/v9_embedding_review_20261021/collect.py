"""Read-only, explicit-domain V8/V9 whole/proxy audit.

Private spec selects user-owned checkpoints and training shards. `stage` makes
diagnostic net-only copies; `export` runs clean/eval on a freshly idle GPU.
No optimizer, backward, test input, BN recalibration, or source write is used.
"""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

sys.dont_write_bytecode = True


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False), encoding="utf-8")


def load(path):
    import torch
    try:
        return torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except RuntimeError as exc:
        if "mmap" not in str(exc):
            raise
        return torch.load(path, map_location="cpu", weights_only=False)


def epoch_of(saved):
    # Original adapters store raw zero-based epoch alongside completed_epochs.
    if saved.get("completed_epochs") is not None:
        return int(saved["completed_epochs"])
    return int(saved.get("epoch", -1))


def usage(metric, component, domain="all_draws"):
    import numpy as np
    report = metric.get("structure", {}).get(component, {})
    if not isinstance(report, dict):
        return None
    report = report.get("selected_ancestor_usage", {})
    parts = [report.get(domain + "/" + role, {}).get("counts") for role in ("pair", "triple")]
    if any(x is None for x in parts):
        return None
    return np.asarray(parts, dtype=np.int64).sum(0)


def metric_at(run, epoch):
    p = Path(run) / ("metrics_epoch_%03d.json" % epoch)
    return json.loads(p.read_text()) if p.exists() else {}


def make_context(repo):
    for p in (repo, repo / "classification_ModelNet40", repo / "pointnet2_ops_lib",
              repo / "inter_hierarchy_MN40"):
        sys.path.insert(0, str(p))


def stage(spec, out):
    """CPU-only copying of selected net states and existing train snapshots."""
    import torch
    torch.set_num_threads(2)
    manifest = {"observations": [], "source_files_unchanged": True,
                "optimizer_updates": 0, "test_forwarded": False}
    for row in spec["observations"]:
        cp = Path(row["checkpoint"])
        before = sha(cp)
        saved = load(cp)
        epoch = epoch_of(saved)
        if "expected_epoch" in row and epoch != row["expected_epoch"]:
            raise ValueError("Selected checkpoint epoch differs: " + row["key"])
        pack = {k: saved[k] for k in ("net", "proxy", "format", "epoch", "completed_epochs",
                "training_config", "train_ids", "validation_ids", "selection_ids", "split_sha256") if k in saved}
        target = out / (row["key"] + "_" + row["stage"] + ".pth")
        torch.save(pack, target)
        if sha(cp) != before:
            raise RuntimeError("Source changed during diagnostic copy")
        entry = dict(row, checkpoint=str(target), source_checkpoint=str(cp),
                     checkpoint_sha256=before, staged_checkpoint_sha256=sha(target), epoch=epoch)
        metric = metric_at(row["run_dir"], epoch) if row.get("run_dir") else {}
        for name in ("sample", "proxy"):
            for dom, suffix in (("all_draws", "usage"), ("active_noncollision", "effective_usage")):
                counts = usage(metric, name, dom)
                if counts is not None:
                    entry[name + "_" + suffix] = counts.tolist()
        manifest["observations"].append(entry)
        del saved, pack
    write(out / "staged_manifest.json", manifest)


def idle_uuid(index):
    text = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,memory.used,utilization.gpu",
                                    "--format=csv,noheader,nounits"], text=True)
    active = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid",
                                      "--format=csv,noheader,nounits"], text=True)
    for line in text.splitlines():
        i, uuid, memory, util = [p.strip() for p in line.split(",")]
        if int(i) == index:
            if float(memory) > 64 or float(util) != 0 or any(uuid in x for x in active.splitlines()):
                raise RuntimeError("Requested GPU is occupied; nothing interrupted")
            return uuid
    raise ValueError("GPU not found")


def state_hash(model):
    h = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        arr = tensor.detach().cpu().contiguous().numpy()
        h.update(name.encode()); h.update(str(arr.dtype).encode()); h.update(str(arr.shape).encode()); h.update(arr.tobytes())
    return h.hexdigest()


def export(spec, out, gpu):
    started = time.monotonic()
    gpu_uuid = idle_uuid(gpu)
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_uuid
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    import h5py
    import numpy as np
    import torch
    if torch.cuda.is_initialized():
        raise RuntimeError("CUDA initialized before idle check")
    torch.set_num_threads(2)
    make_context(Path(spec["repo"]))
    from inter_hierarchy_MN40.models.pointmlp import Hype_pointMLP
    from inter_hierarchy_MN40.hier_proxy_scratch_v8.geometry import (
        configure_backbone as configure_v8, to_ball, official_distance)
    from inter_hierarchy_MN40.hier_proxy_scratch_v9.geometry import whole_to_hier
    from inter_hierarchy_MN40.hier_proxy_cycle_v9c.geometry import whole_to_hier as cycle_view
    shards = sorted(Path(spec["data_dir"]).glob("ply_data_train*.h5"))
    clouds, labels, shard_proof = [], [], []
    for p in shards:
        with h5py.File(p, "r") as f:
            clouds.append(f["data"][:, :1024].astype(np.float32))
            labels.append(f["label"][:].reshape(-1).astype(np.int64))
        shard_proof.append({"name": p.name, "sha256": sha(p), "count": len(labels[-1])})
    clouds, labels = np.concatenate(clouds), np.concatenate(labels)
    if clouds.shape != (9840, 1024, 3) or labels.shape != (9840,):
        raise ValueError("Expected all 9840 canonical training objects")
    input_sha = hashlib.sha256(clouds.tobytes()).hexdigest()
    manifest = {"format": "v9-embedding-readonly-export-1", "status": "running",
                "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                "observations": [], "training_shards": shard_proof,
                "input_sha256": input_sha, "label_sha256": hashlib.sha256(labels.tobytes()).hexdigest(),
                "sample_count": len(labels), "gpu_uuid": gpu_uuid,
                "mode": "clean first1024 / eval historical BN / same sorted shard+row order",
                "optimizer_updates": 0, "backward_calls": 0, "test_forwarded": False,
                "BN_recalibrated": False, "script_sha256": sha(__file__)}
    write(out / "export_manifest.json", manifest)
    # Recheck immediately before first allocation, after CPU data loading.
    if idle_uuid(gpu) != gpu_uuid:
        raise RuntimeError("GPU identity changed")
    device = torch.device("cuda:0")
    for row in spec["observations"]:
        cp = Path(row["checkpoint"]); before = sha(cp); saved = load(cp)
        if row.get("staged_checkpoint_sha256") and before != row["staged_checkpoint_sha256"]:
            raise RuntimeError("Staged checkpoint identity mismatch")
        epoch = epoch_of(saved)
        if "expected_epoch" in row and epoch != row["expected_epoch"]:
            raise ValueError("Wrong fixed epoch")
        kind = row["kind"]
        torch.manual_seed(22); np.random.seed(22)
        model = Hype_pointMLP()
        if kind in ("orig_c01_gs", "v8_hier", "cycle_off", "cycle_hier"):
            model = configure_v8(model)
        net = {k.removeprefix("module."): v for k, v in saved["net"].items()}
        model.load_state_dict(net, strict=True)
        model.requires_grad_(False); model.eval(); model.to(device)
        actual_c = float(model.manifold.c.detach().cpu())
        if not math.isclose(actual_c, float(row["c_native"]), rel_tol=1e-5):
            raise ValueError("Loaded curvature does not match the declared native domain")
        state_before = state_hash(model)
        torch.manual_seed(22); torch.cuda.manual_seed_all(22); np.random.seed(22)
        mus = []
        with torch.no_grad():
            for start in range(0, len(clouds), 32):
                batch = torch.from_numpy(clouds[start:start+32].copy()).to(device)
                mu, _ = model(batch.transpose(1, 2).contiguous())
                if not bool(torch.isfinite(mu).all()):
                    raise RuntimeError("Nonfinite whole")
                mus.append(mu.cpu())
        mu = torch.cat(mus)
        cache = {"native_mu": mu.numpy(), "labels": labels,
                 "canonical_ids": np.arange(len(labels), dtype=np.int64)}
        if saved.get("proxy"):
            proxy_values = [v for k, v in saved["proxy"].items() if k.split(".")[-1] == "tangent_proxies"]
            if len(proxy_values) != 1:
                raise ValueError("Ambiguous hierarchy proxy state")
            tangent = proxy_values[0].float().to(device)
            with torch.no_grad():
                ball = to_ball(tangent, c=.1, clip_r=2.3, hook=False)
                if kind == "v9_hier":
                    view = whole_to_hier(mu.to(device))
                elif kind == "cycle_hier":
                    view = cycle_view(mu.to(device))
                else:
                    view = mu.to(device)
                if not math.isclose(float(row.get("c_hier", -1)), .1, rel_tol=1e-7):
                    raise ValueError("Proxy/view domain must explicitly be c=.1")
                cache.update(hier_mu=view.cpu().numpy(), proxy_ball=ball.cpu().numpy(),
                             proxy_tangent=tangent.cpu().numpy())
                equal = view / view.norm(dim=-1, keepdim=True) * (math.tanh(math.sqrt(.1)*2.3)/math.sqrt(.1))
                unit = view / view.norm(dim=-1, keepdim=True)
                punit = ball / ball.norm(dim=-1, keepdim=True)
                for name, candidates in (("nn4", view), ("nn4_equal", equal)):
                    picked = []
                    for offset in range(0, len(ball), 8):
                        distances = official_distance(ball[offset:offset+8], candidates, c=.1, block=8)
                        picked.append(distances.topk(4, largest=False).indices.cpu())
                    cache[name] = torch.cat(picked).numpy()
                cache["nn4_angle"] = (punit @ unit.T).topk(4).indices.cpu().numpy()
        metric = metric_at(row["run_dir"], epoch) if row.get("run_dir") else {}
        for name in ("sample", "proxy"):
            for dom, suffix in (("all_draws", "usage"), ("active_noncollision", "effective_usage")):
                counts = usage(metric, name, dom)
                if counts is None and row.get(name + "_" + suffix) is not None:
                    counts = np.asarray(row[name + "_" + suffix], dtype=np.int64)
                if counts is not None:
                    cache[name + "_" + suffix] = counts
        if state_hash(model) != state_before or any(p.grad is not None for p in model.parameters()):
            raise RuntimeError("Frozen inference changed model/BN or created gradients")
        if sha(cp) != before:
            raise RuntimeError("Checkpoint changed")
        filename = row["key"] + "_" + row["stage"] + ".npz"
        np.savez_compressed(out / filename, **cache)
        entry = {k: v for k, v in row.items() if k not in ("checkpoint", "run_dir", "sample_usage", "proxy_usage", "sample_effective_usage", "proxy_effective_usage")}
        entry.update(cache=filename, epoch=epoch, checkpoint_sha256=row.get("checkpoint_sha256", before),
                     measured_checkpoint_sha256=before, native_cap_tangent=2.3 if actual_c < .2 else None,
                     hier_cap_tangent=2.3 if "proxy_ball" in cache else None,
                     usage_source_epoch=epoch, native_c_observed=actual_c,
                     input_sha256=input_sha, model_BN_unchanged=True, source_unchanged=True)
        manifest["observations"].append(entry)
        write(out / "export_manifest.json", manifest)
        print(json.dumps({"key": row["key"], "stage": row["stage"], "epoch": epoch,
                          "whole_count": len(mu), "has_proxy": "proxy_ball" in cache,
                          "wall_seconds": round(time.monotonic()-started, 2)}), flush=True)
        del model, saved, net, mu, mus
        torch.cuda.empty_cache()
    if hashlib.sha256(clouds.tobytes()).hexdigest() != input_sha:
        raise RuntimeError("Clean input changed")
    manifest.update(status="completed", finished_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                    wall_seconds=time.monotonic()-started)
    write(out / "export_manifest.json", manifest)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("stage", "export"))
    p.add_argument("--spec", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--gpu", type=int)
    a = p.parse_args()
    spec = json.loads(a.spec.read_text(encoding="utf-8-sig"))
    if a.out.exists():
        raise FileExistsError("Diagnostic output must be a new directory")
    for row in spec["observations"]:
        source = Path(row["checkpoint"]).resolve()
        target = a.out.resolve()
        if source.parent == target or source.parent in target.parents:
            raise ValueError("Output cannot be inside an existing source run")
    a.out.mkdir(parents=True)
    if a.mode == "stage":
        stage(spec, a.out)
    else:
        if a.gpu is None:
            p.error("export requires an explicit idle GPU")
        export(spec, a.out, a.gpu)


if __name__ == "__main__":
    main()
