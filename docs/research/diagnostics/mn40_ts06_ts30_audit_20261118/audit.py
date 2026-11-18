"""Read-only sampling replay and frozen evaluation; private paths are CLI inputs."""
from __future__ import annotations
import argparse
from collections import defaultdict
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
from inter_hierarchy_MN40.hycore_c01_gs_halfmargin_20261020 import train as source
from inter_hierarchy_MN40.hier_proxy_scratch_v5.sampler import SourceClassBatchSampler

RUNS = {
    "TS06": "orig_c01_cap_gs_halfmargin_20261020_160111/gs_half_margin",
    "TS30": "b13_cb32_controls_20261113_B/b13control_20261113_B_h0_200/main",
    "TS31": "b13_cb32_controls_20261113_B/b13control_20261113_B_h0_300/main",
}
LATE = (200, 220, 240, 260, 280, 300)
NAMES = "airplane bathtub bed bench bookshelf bottle bowl car chair cone cup curtain desk door dresser flower_pot glass_box guitar keyboard lamp laptop mantel monitor night_stand person piano plant radio range_hood sink sofa stairs stool table tent toilet tv_stand vase wardrobe xbox".split()


def write(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")


def load_json(path):
    return json.loads(Path(path).read_text())


def model_class():
    return source.import_original(REPO).Hype_pointMLP


def data_arrays(data, meta):
    import h5py
    result = {}
    for split in ("train", "test"):
        points, labels, ids = [], [], []
        for name in meta[split]["file_order"]:
            assert source.digest(Path(data) / name) == meta[split]["file_sha256"][name], name
            with h5py.File(Path(data) / name) as f:
                x = f["data"][:, :1024].astype(np.float32)
                y = f["label"][:].reshape(-1).astype(np.int64)
            points.append(x); labels.append(y)
            ids.extend(f"{split}/{name}/{row:04d}" for row in range(len(y)))
        result[split] = (np.concatenate(points), np.concatenate(labels), np.asarray(ids))
    return result


class IndexData(Dataset):
    def __init__(self, size): self.size = size
    def __len__(self): return self.size
    def __getitem__(self, index): return index


def replay_sh(run, labels, initial_sha):
    torch.manual_seed(4780); np.random.seed(4780); random.seed(4780)
    model = source.configure_model(model_class()(), True)
    assert source.state_digest(model) == initial_sha, "CPU initialization differs"
    del model
    train = DataLoader(IndexData(len(labels)), batch_size=32, shuffle=True, drop_last=True, num_workers=0)
    test = DataLoader(IndexData(2468), batch_size=16, shuffle=False, num_workers=0)
    plans, checks = [], []
    for epoch in range(1, 301):
        batch = np.stack([b.numpy() for b in train])
        assert batch.shape == (307, 32) and len(np.unique(batch)) == 9824
        for _ in test: pass
        plans.append(batch)
        p = run / f"checkpoint_epoch_{epoch:03d}.pth"
        if p.exists():
            state = torch.load(p, map_location="cpu", weights_only=False)
            matched = torch.equal(state["rng"]["torch_cpu"], torch.get_rng_state())
            checks.append(dict(epoch=epoch, cpu_rng_exact=matched))
            if not matched:
                raise RuntimeError(f"SH CPU RNG replay differs at e{epoch}")
            del state
    return np.asarray(plans, dtype=np.int32), checks


def replay_cb(run, labels, steps):
    sampler = SourceClassBatchSampler(labels, world_size=1, classes_per_rank=16,
                                     instances_per_class=2, steps=steps, seed=4780)
    plans, checks = [], []
    for epoch in range(1, 301):
        sampler.set_epoch(epoch - 1)
        batch = np.asarray(list(sampler), dtype=np.int32)
        actual = load_json(run / f"metrics_epoch_{epoch:03d}.json")["telemetry"]["sample_coverage"]
        stored = np.zeros(len(labels), dtype=np.int64)
        for index, count in actual["id_draw_counts"].items(): stored[int(index)] = count
        rebuilt = np.bincount(batch.ravel(), minlength=len(labels))
        assert np.array_equal(stored, rebuilt), f"CB per-ID mismatch e{epoch}"
        assert np.array_equal(np.bincount(labels, weights=rebuilt, minlength=40), actual["per_class_draws"])
        checks.append(dict(epoch=epoch, all_id_counts_match=True, unique=actual["unique_ids"]))
        plans.append(batch)
    return np.asarray(plans), checks


def sampling_stats(plans, labels, best, out, arm):
    counts = np.stack([np.bincount(p.ravel(), minlength=len(labels)) for p in plans]).astype(np.int16)
    peers = [set() for _ in labels]
    last = np.zeros(len(labels), np.int32); gaps = last.copy(); duplicate_positions = 0
    for epoch, batches in enumerate(plans, 1):
        seen = np.flatnonzero(counts[epoch - 1])
        gaps[seen] = np.maximum(gaps[seen], epoch - last[seen] - 1); last[seen] = epoch
        for batch in batches:
            duplicate_positions += len(batch) - len(set(batch.tolist()))
            groups = defaultdict(set)
            for index in batch: groups[int(labels[index])].add(int(index))
            for group in groups.values():
                for index in group: peers[index].update(group - {index})
    gaps = np.maximum(gaps, 300 - last)
    np.savez_compressed(out / f"{arm}_sampling.npz", epoch_counts=counts,
        best_counts=counts[:best].sum(0), final_counts=counts.sum(0), seen_epochs=(counts > 0).sum(0),
        longest_absence=gaps, distinct_same_class_peers=np.array([len(s) for s in peers]),
        labels=labels, batch_plans=plans)
    return dict(total_presentations=int(counts.sum()), best_presentations=int(counts[:best].sum()),
                count_min=int(counts.sum(0).min()), count_max=int(counts.sum(0).max()),
                count_mean=float(counts.sum(0).mean()), never_seen=int((counts.sum(0) == 0).sum()),
                duplicate_within_batch_positions=duplicate_positions,
                first_epoch_unique=int((counts[0] > 0).sum()))


def prepare(args):
    out = args.out; out.mkdir(parents=True, exist_ok=False)
    (out / "weights").mkdir()
    runs = {arm: args.runs_root / rel for arm, rel in RUNS.items()}
    sh_meta = load_json(runs["TS06"] / "manifest.json")
    data_meta = sh_meta["dataset_identity"]
    data = data_arrays(args.data, data_meta)
    np.savez_compressed(out / "identity.npz", train_ids=data["train"][2], train_labels=data["train"][1],
                        test_ids=data["test"][2], test_labels=data["test"][1])
    manifest = dict(commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
                    data=data_meta, models=[], sampling={}, no_optimizer_updates=True)
    for arm, run in runs.items():
        best_file = run / ("best_checkpoint.pth" if arm == "TS06" else "best.pth")
        best_state = torch.load(best_file, map_location="cpu", weights_only=False)
        best_epoch = int(best_state["completed_epochs"])
        if arm != "TS06":
            assert np.array_equal(best_state["train_ids"], np.arange(9840))
            assert load_json(run / "manifest.json")["training_shards"] == data_meta["train"]["file_order"]
        del best_state
        for tag in ("best", *[f"e{e}" for e in LATE]):
            path = best_file if tag == "best" else run / f"checkpoint_epoch_{int(tag[1:]):03d}.pth"
            state = torch.load(path, map_location="cpu", weights_only=False)
            net = {k.removeprefix("module."): v.cpu() for k, v in state["net"].items()}
            epoch = int(state["completed_epochs"])
            if arm == "TS06": expected = state["metrics"]["test"]
            else: expected = dict(acc=state["metrics"]["test"]["test_oa"], acc_avg=state["metrics"]["test"]["test_aa"])
            filename = f"{arm}_{tag}.pth"
            torch.save(dict(net=net, epoch=epoch), out / "weights" / filename)
            manifest["models"].append(dict(arm=arm, tag=tag, epoch=epoch, filename=filename,
                expected_test=expected, source_sha256=source.digest(path), compact_sha256=source.digest(out / "weights" / filename)))
            del state, net
        labels = data["train"][1]
        if arm == "TS06":
            plans, checks = replay_sh(run, labels, sh_meta["initialized_model_sha256"])
            provenance = "exact sampler replay; initial model hash and every available archived CPU RNG verified"
        else:
            plans, checks = replay_cb(run, labels, 200 if arm == "TS30" else 300)
            provenance = "all 300 epochs recorded per-ID counts exactly match deterministic sampler replay"
        manifest["sampling"][arm] = dict(best_epoch=best_epoch, checks=checks, provenance=provenance,
            **sampling_stats(plans, labels, best_epoch, out, arm))
        write(out / "manifest.json", manifest)
        print(arm, manifest["sampling"][arm]["provenance"], flush=True)
    write(out / "manifest.json", manifest)


def idle(gpu):
    from inter_hierarchy_MN40.hier_proxy_scratch_v8.queue import query_gpus, idle_gpus
    return next((x for x in idle_gpus(query_gpus()) if x["index"] == gpu), None)


def infer(args):
    args.out.mkdir(parents=True, exist_ok=False)
    meta = load_json(args.inputs / "manifest.json")
    data = data_arrays(args.data, meta["data"])
    identity = np.load(args.inputs / "identity.npz")
    for split in data: assert np.array_equal(data[split][2], identity[f"{split}_ids"])
    with Path(f"/tmp/hycore_{os.getuid()}_gpu_{args.gpu}.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        chosen = idle(args.gpu)
        assert chosen is not None, "Target GPU is busy"
        time.sleep(10)
        assert idle(args.gpu)["uuid"] == chosen["uuid"]
        os.environ["CUDA_VISIBLE_DEVICES"] = chosen["uuid"]
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True
        klass = model_class()
        results = []
        for item in meta["models"]:
            arm, tag = item["arm"], item["tag"]
            path = args.inputs / "weights" / item["filename"]
            assert source.digest(path) == item["compact_sha256"]
            state = torch.load(path, map_location="cpu", weights_only=False)
            model = klass()
            if arm == "TS06": model = source.configure_model(model, True)
            else:
                from inter_hierarchy_MN40.hier_proxy_cycle_v9c.geometry import configure_backbone
                model = configure_backbone(model)
            model.load_state_dict(state["net"], strict=True); model = model.cuda().eval()
            before = source.state_digest(model)
            for split in (("test", "train") if tag in ("best", "e300") else ("test",)):
                points, labels, ids = data[split]
                logits = []
                with torch.inference_mode():
                    for offset in range(0, len(points), 16):
                        cloud = torch.from_numpy(points[offset:offset+16]).transpose(1, 2).contiguous().cuda()
                        _, scores = model(cloud)
                        logits.append(scores.cpu().numpy())
                logits = np.concatenate(logits); pred = logits.argmax(1)
                correct = pred == labels
                oa = float(correct.mean() * 100)
                aa = float(np.mean([correct[labels == c].mean() for c in range(40)]) * 100)
                if split == "test":
                    expected = item["expected_test"]
                    assert abs(oa - expected["acc"]) < .00051, (arm, tag, oa, expected)
                    assert abs(aa - expected["acc_avg"]) < .00051, (arm, tag, aa, expected)
                assert before == source.state_digest(model), "Evaluation mutated model/BN"
                np.savez_compressed(args.out / f"{arm}_{tag}_{split}.npz", ids=ids, labels=labels,
                                    predictions=pred, logits=logits)
                row = dict(arm=arm, tag=tag, epoch=item["epoch"], split=split, oa=oa, aa=aa,
                           correct=int(correct.sum()), source_sha256=item["source_sha256"], frozen_state_unchanged=True)
                results.append(row); print(json.dumps(row), flush=True)
                write(args.out / "manifest.json", dict(status="running", results=results, gpu=args.gpu))
            del model, state; torch.cuda.empty_cache()
        write(args.out / "manifest.json", dict(status="completed", results=results, gpu=args.gpu, no_optimizer_updates=True))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=("prepare", "infer"))
    p.add_argument("--runs-root", type=Path)
    p.add_argument("--inputs", type=Path)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--gpu", type=int, default=0)
    args = p.parse_args()
    torch.set_num_threads(2)
    (prepare if args.mode == "prepare" else infer)(args)


if __name__ == "__main__": main()
