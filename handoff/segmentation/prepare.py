"""Build and verify historical ShapeNetPart inputs from the public archive on CPU."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from handoff.segmentation.data_identity import verify_prepared


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    archive = args.archive.resolve()
    output = args.output.resolve()
    source = output.with_name(output.name + "_disjoint_source")
    if not archive.is_file():
        parser.error("Download the pinned public archive first")
    if output.exists() or source.exists():
        parser.error("Use new output paths; preparation never replaces existing data")
    profiles = json.loads(Path(__file__).with_name("data_reference.json").read_text(encoding="utf-8"))
    if len(profiles["profiles"]) != 1:
        parser.error("Expected one historical preparation profile")
    profile = profiles["profiles"][0]
    environment = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
    subprocess.run([sys.executable, "-m", "handoff.segmentation.prepare_worker",
                    "segmentation_ShapeNetPart.hycore_seg_v1.prepare_data",
                    "--archive", str(archive), "--output", str(source),
                    "--expected-archive-sha256", profile["archive_sha256"]],
                   cwd=ROOT, env=environment, check=True)
    subprocess.run([sys.executable, "-m", "handoff.segmentation.prepare_worker",
                    "segmentation_ShapeNetPart.hycore_seg_v1.prepare_antao_protocol",
                    "--source", str(source), "--output", str(output)],
                   cwd=ROOT, env=environment, check=True)
    verified = verify_prepared(output, profile, ("trainval", "test"))
    (output / "rebuild_verification.json").write_text(
        json.dumps(verified, indent=2) + "\n", encoding="utf-8")
    print("Verified historical array contents: " + str(output))


if __name__ == "__main__":
    main()
