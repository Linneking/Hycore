"""Evaluate the three reference checkpoints with the existing unmasked metric."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=("DGCNN", "TAN-R0", "TAN-R"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--data-check", choices=("exact", "content"), default="exact")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists")
    import torch
    from segmentation_ShapeNetPart.hycore_seg_v1.utils import sha256, state_hash, seed_all
    from segmentation_ShapeNetPart.hycore_seg_v1.train import loader
    from segmentation_ShapeNetPart.hycore_seg_v1.data import ShapeNetPart
    from segmentation_ShapeNetPart.hycore_seg_v13.metrics import evaluate
    from segmentation_ShapeNetPart.hycore_seg_v1.model import DGCNNPartSeg
    from segmentation_ShapeNetPart.hycore_seg_v10.model import NativeTail2
    torch.set_num_threads(4)
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if args.data_check == "exact":
        if sha256(args.data / "manifest.json") != ck["data_manifest_sha256"]:
            raise ValueError("Data manifest differs from the saved experiment; for a verified rebuild use --data-check content")
        manifest = json.loads((args.data / "manifest.json").read_text(encoding="utf-8"))
        if sha256(args.data / "test.h5") != manifest["splits"]["test"]["file_sha256"]:
            raise ValueError("Test file differs from its manifest")
        data_verification = {"mode": "exact", "manifest_sha256": ck["data_manifest_sha256"]}
    else:
        from handoff.segmentation.data_identity import reference_for, verify_prepared
        profile = reference_for(ck["data_manifest_sha256"])
        data_verification = verify_prepared(args.data, profile, ("test",))
    cfg = ck["config"]
    seed_all(cfg["seed"])
    if args.experiment == "DGCNN":
        model = DGCNNPartSeg("euclidean", k=cfg["k"], dropout=cfg["dropout"])
    else:
        model = NativeTail2("h", k=cfg["k"], dropout=cfg["dropout"])
    model.load_state_dict(ck["model"], strict=True)
    device = torch.device(args.device)
    model.to(device)
    before = state_hash(model.state_dict())
    dataset = ShapeNetPart(args.data, "test", cfg["num_points"])
    data_loader = loader(dataset, cfg["eval_batch_size"], 0,
                         torch.Generator().manual_seed(cfg["seed"] + 2003), False)
    scores = evaluate(model, data_loader, device)
    if before != state_hash(model.state_dict()) or scores["objects"] != 2874:
        raise AssertionError("Evaluation changed model state or missed test objects")
    result = {"experiment": args.experiment, "epoch": ck["epoch"],
              "checkpoint_sha256": sha256(args.checkpoint),
              "data_verification": data_verification, "metrics": scores}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: scores[k] for k in ("instance_miou", "class_miou", "objects")}))


if __name__ == "__main__":
    main()
