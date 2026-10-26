"""Stage fixed checkpoints and export five frozen views for simplified audit94."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def clean_state(state):
    result = {}
    for key, value in state.items():
        name = key[7:] if key.startswith("module.") else key
        if name in result:
            raise ValueError("Ambiguous DataParallel parameter: " + name)
        result[name] = value
    return result


def epoch_of(state):
    if "completed_epochs" not in state:
        raise ValueError("Completed-epoch metadata is required")
    return int(state["completed_epochs"])


def centers(ids):
    return ((ids * 15485863 + 4780) % 1024,
            (ids * 32452843 + 4780 + 97) % 1024)


def torch_load(path):
    import torch
    return torch.load(path, map_location="cpu", weights_only=False, mmap=True)


def prepare(settings):
    import torch
    torch.set_num_threads(2)
    root = Path(settings["root"])
    cache = root / "cache"
    cache.mkdir(exist_ok=False)
    records = {}
    for row in settings["observations"]:
        label = row["label"]
        source = Path(row["checkpoint"])
        before = sha(source)
        if row.get("expected_sha256") and before != row["expected_sha256"]:
            raise ValueError("Source checkpoint SHA mismatch: " + label)
        saved = torch_load(source)
        epoch = epoch_of(saved)
        if epoch != row["epoch"]:
            raise ValueError("Wrong source epoch: " + label)
        net = clean_state(saved["net"])
        if any(not bool(torch.isfinite(x).all()) for x in net.values() if x.is_floating_point()):
            raise ValueError("Nonfinite checkpoint: " + label)
        target = cache / (label + "_state.pt")
        with target.open("xb") as out:
            torch.save({"net": net, "completed_epochs": epoch}, out)
        if label.startswith("TS18"):
            queue = saved["queue"]
            if queue["features"].shape != (2048, 256):
                raise ValueError("Expected saved Q2048: " + label)
            if not bool(((queue["object_ids"] >= 0) & (queue["object_ids"] < 9840)).all()):
                raise ValueError("Queue IDs are outside canonical training population")
            if saved["training_config"]["queue"]["capacity"] != 2048:
                raise ValueError("Queue metadata differs")
            with (cache / (label + "_queue.pt")).open("xb") as out:
                torch.save({"queue": queue, "global_step": saved["global_step"],
                            "completed_epochs": epoch, "source_sha256": before}, out)
            # Production sorted-shard identity is checked against the source manifest.
            if settings["part_train_shards"] != sorted(settings["part_train_shards"]):
                raise ValueError("PartHier queue object IDs need a noncanonical remap")
        if sha(source) != before:
            raise RuntimeError("Source checkpoint changed during read")
        records[label] = {"epoch": epoch, "source_sha256": before,
                          "staged_sha256": sha(target), "c": 1.0,
                          "source_commit": row["source_commit"]}
        del saved, net
    write(cache / "checkpoints.json", records)
    return records


def idle_gpu(index):
    text = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,memory.used,utilization.gpu",
                                    "--format=csv,noheader,nounits"], text=True)
    active = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid",
                                      "--format=csv,noheader,nounits"], text=True)
    for line in text.splitlines():
        number, uuid, memory, utilization = [x.strip() for x in line.split(",")]
        if int(number) == index:
            if float(memory) > 64 or float(utilization) != 0 or uuid in active:
                raise RuntimeError("Requested GPU is occupied; nothing interrupted")
            return uuid
    raise ValueError("Unknown GPU")


def model_hash(model):
    h = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        arr = tensor.detach().cpu().contiguous().numpy()
        h.update(name.encode()); h.update(str(arr.dtype).encode()); h.update(str(arr.shape).encode())
        h.update(arr.tobytes())
    return h.hexdigest()


def load_data(settings):
    import h5py
    import numpy as np
    clouds, labels, proof = [], [], []
    full_points_hash = hashlib.sha256()
    for path in sorted(Path(settings["data_dir"]).glob("ply_data_train*.h5")):
        digest = sha(path)
        if digest != settings["expected_shard_sha256"][path.name]:
            raise ValueError("Training shard identity mismatch: " + path.name)
        with h5py.File(path, "r") as handle:
            full = handle["data"][:].astype(np.float32)
            lab = handle["label"][:].reshape(-1).astype(np.int64)
        full_points_hash.update(full.tobytes())
        clouds.append(full[:, :1024].copy()); labels.append(lab)
        proof.append({"name": path.name, "sha256": digest, "count": len(lab)})
    clouds, labels = np.concatenate(clouds), np.concatenate(labels)
    if clouds.shape != (9840, 1024, 3) or labels.shape != (9840,):
        raise ValueError("Expected complete canonical 9840 population")
    label_hash = hashlib.sha256(labels.tobytes()).hexdigest()
    if label_hash != settings["part_data_identity"]["train_labels"]:
        raise ValueError("PartHier canonical queue labels differ")
    if full_points_hash.hexdigest() != settings["part_data_identity"]["train_points"]:
        raise ValueError("PartHier full source point bytes differ")
    return clouds, labels, {"count": 9840, "points_sha256": hashlib.sha256(clouds.tobytes()).hexdigest(),
                            "labels_sha256": label_hash, "shards": proof,
                            "crop_seed": 4780, "whole_count": 1024,
                            "canonical_ids": "sorted shard filename then row",
                            "crop_centers": "(id*15485863+4780)%1024; (id*32452843+4877)%1024"}


def compare(a, b):
    import numpy as np
    a, b = a.astype(np.float64), b.astype(np.float64)
    difference = np.abs(a - b)
    ua, ub = a / np.linalg.norm(a, axis=-1, keepdims=True), b / np.linalg.norm(b, axis=-1, keepdims=True)
    angle = np.degrees(np.arccos(np.clip(np.sum(ua * ub, axis=-1), -1, 1)))
    depth_error = np.abs(2 * np.arctanh(np.linalg.norm(a, axis=-1)) - 2 * np.arctanh(np.linalg.norm(b, axis=-1)))
    record = {"max_abs": float(difference.max()), "mean_abs": float(difference.mean()),
              "angle_p99": float(np.quantile(angle, .99)), "depth_p99": float(np.quantile(depth_error, .99))}
    record["passed"] = record["max_abs"] <= 2e-4 and record["mean_abs"] <= 2e-6 and record["angle_p99"] <= .05 and record["depth_p99"] <= .005
    return record


def collect(settings, labels_to_run, gpu):
    import fcntl
    lock = open("/tmp/hycore_%d_gpu_%d.lock" % (os.getuid(), gpu), "a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    uuid = idle_gpu(gpu)
    os.environ["CUDA_VISIBLE_DEVICES"] = uuid
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    import numpy as np
    import torch
    if torch.cuda.is_initialized():
        raise RuntimeError("CUDA was initialized before resource check")
    torch.set_num_threads(2)
    repo = Path(settings["source"])
    for path in (repo, repo / "pointnet2_ops_lib", repo / "inter_hierarchy_MN40"):
        sys.path.insert(0, str(path))
    from models.pointmlp import Hype_pointMLP
    from inter_hierarchy_MN40.parthier_v1.base_protocol import sample_nested_parts
    from inter_hierarchy_MN40.hier_proxy_scratch_v9.geometry import configure_backbone
    from inter_hierarchy_MN40.hier_proxy_scratch_v9.base_protocol import get_children_alias
    clouds, labs, input_proof = load_data(settings)
    cache = Path(settings["root"]) / "cache"
    identity_path = cache / ("input_identity_gpu%d.json" % gpu)
    write(identity_path, input_proof)
    if idle_gpu(gpu) != uuid:
        raise RuntimeError("GPU became occupied before allocation")
    device = torch.device("cuda:0")
    torch.manual_seed(4780); torch.cuda.manual_seed_all(4780)
    # Blackwell TF32 convolution kernels can change small-part output with
    # microbatch size. Keep the audit in strict FP32 and retain the gate.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    records = json.loads((cache / "checkpoints.json").read_text())
    ids = np.arange(len(labs), dtype=np.int64)
    wc, pc = centers(ids)
    input_crop_hash = None
    for label in labels_to_run:
        started = time.monotonic()
        target = cache / (label + ".npz")
        if target.exists():
            raise FileExistsError("Existing export must not be overwritten")
        cp = cache / (label + "_state.pt")
        if sha(cp) != records[label]["staged_sha256"]:
            raise ValueError("Staged state identity changed")
        saved = torch_load(cp)
        model = configure_backbone(Hype_pointMLP())
        model.load_state_dict(saved["net"], strict=True)
        model.requires_grad_(False); model.eval(); model.to(device)
        del saved
        before = model_hash(model)
        chunks = {name: [] for name in ("clean", "whole", "parts", "clean_prediction", "whole_prediction")}
        crop_hash = hashlib.sha256()
        status_path = Path(settings["root"]) / (label + "_status.json")
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            # Verify actual source alias cropping against the published nested adapter.
            probe = torch.from_numpy(clouds[:4].copy()).to(device)
            w, parts, meta = sample_nested_parts(probe, 1024, whole_centers=wc[:4], centers=pc[:4])
            raw = probe.transpose(1, 2).contiguous()
            _, ow, _ = get_children_alias(raw, 1024, centers=wc[:4])
            _, op, _ = get_children_alias(ow, 600, centers=pc[:4])
            if not torch.equal(w, ow) or not torch.equal(parts[2], op):
                raise RuntimeError("Native crop/source identity differs")
            probe_inputs = [probe.transpose(1, 2).contiguous(), *parts, w]
            gates = []
            for x in probe_inputs:
                full = model(x)[0].cpu().numpy()
                split = torch.cat([model(v)[0] for v in x.split(2)]).cpu().numpy()
                gates.append(compare(full, split))
            if not all(g["passed"] for g in gates):
                raise RuntimeError("Frozen batch-size consistency gate failed: " + repr(gates))
            if model_hash(model) != before:
                raise RuntimeError("Gate changed model/BN")
            for start in range(0, len(labs), 32):
                stop = min(start + 32, len(labs))
                x = torch.from_numpy(clouds[start:stop].copy()).to(device)
                whole, parts, meta = sample_nested_parts(x, 1024, whole_centers=wc[start:stop], centers=pc[start:stop])
                if not all(meta[k] for k in ("parts_nested", "part600_matches_whole_prefix", "whole_suffix_unchanged")):
                    raise RuntimeError("Nested crop contract failed")
                for key in ("whole_indices", "part_source_indices"):
                    crop_hash.update(meta[key].cpu().numpy().tobytes())
                clean, clean_logits = model(x.transpose(1, 2).contiguous())
                part_mu = torch.stack([model(part, emb=True)[0] for part in parts], dim=1)
                whole_mu, whole_logits = model(whole)
                for value in (clean, part_mu, whole_mu):
                    if not bool(torch.isfinite(value).all()) or not bool((value.square().sum(-1) < 1).all()):
                        raise RuntimeError("Invalid c1 embedding")
                chunks["clean"].append(clean.cpu().numpy())
                chunks["whole"].append(whole_mu.cpu().numpy())
                chunks["parts"].append(part_mu.cpu().numpy())
                chunks["clean_prediction"].append(clean_logits.argmax(-1).cpu().numpy())
                chunks["whole_prediction"].append(whole_logits.argmax(-1).cpu().numpy())
                if start % 320 == 0:
                    write(status_path, {"status": "collecting", "label": label, "objects": stop,
                                        "total": len(labs), "updated_utc": now(), "pid": os.getpid()})
        if model_hash(model) != before or any(p.grad is not None for p in model.parameters()):
            raise RuntimeError("Frozen export changed model/BN or created gradients")
        crop_digest = crop_hash.hexdigest()
        if input_crop_hash is not None and crop_digest != input_crop_hash:
            raise RuntimeError("Crops differ across checkpoints")
        input_crop_hash = crop_digest
        payload = {k: np.concatenate(v) for k, v in chunks.items()}
        with target.open("xb") as out:
            np.savez(out, ids=ids, labels=labs, **payload)
        manifest = {"status": "completed", "label": label, "epoch": records[label]["epoch"],
                    "source_sha256": records[label]["source_sha256"], "cache_sha256": sha(target),
                    "input": input_proof, "crop_sha256": crop_digest, "model_state_unchanged": True,
                    "state_sha256": before, "gate": gates, "objects": 9840, "gpu_uuid": uuid,
                    "wall_seconds": time.monotonic() - started, "peak_allocated_MiB": torch.cuda.max_memory_allocated()/2**20,
                    "optimizer_updates": 0, "backward_calls": 0, "test_forwarded": False,
                    "tf32": False, "cudnn_benchmark": False, "updated_utc": now()}
        write(cache / (label + "_manifest.json"), manifest)
        write(status_path, manifest)
        print(json.dumps({"label": label, "status": "completed", "wall_seconds": manifest["wall_seconds"]}), flush=True)
        del model, chunks, payload
        torch.cuda.empty_cache()
    lock.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("prepare", "collect"))
    p.add_argument("--settings", type=Path, required=True)
    p.add_argument("--labels", nargs="+")
    p.add_argument("--gpu", type=int)
    args = p.parse_args()
    settings = json.loads(args.settings.read_text(encoding="utf-8-sig"))
    if args.mode == "prepare":
        prepare(settings)
    else:
        collect(settings, args.labels, args.gpu)


if __name__ == "__main__":
    main()
