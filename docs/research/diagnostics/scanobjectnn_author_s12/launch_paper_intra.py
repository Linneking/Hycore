"""Run the paper's native intra-hierarchy settings on the author entry."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import subprocess

from dispatch import reserve, run_arm
from entry import ARM_NAMES, PAPER_INTRA_ARM, POINTMLP_ARGS, SEED, support

BASELINE = ARM_NAMES[0]
DESKTOP_PROGRAMS = {"Xorg", "gnome-shell"}


def desktop_only(gpu):
    output = subprocess.check_output(
        ["nvidia-smi", "pmon", "-i", str(gpu["index"]), "-c", "1"],
        text=True, timeout=20,
    )
    rows = [line.split() for line in output.splitlines() if line.strip() and not line.lstrip().startswith("#")]
    return bool(rows) and all(
        len(row) >= 4 and row[0] == str(gpu["index"])
        and (row[1] == "-" or (row[2] == "G" and row[-1] in DESKTOP_PROGRAMS))
        for row in rows
    )


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def match_inputs(first, second):
    keys = ("cloud", "labels", "whole", "part")
    return (
        first["initial_model_sha256"] == second["initial_model_sha256"]
        and len(first["batches"]) == len(second["batches"]) == 4
        and all(a[key] == b[key] for a, b in zip(first["batches"], second["batches"]) for key in keys)
    )


def accept(preflight, reference, reference_smoke=None):
    baseline, paper = (read_json(preflight / arm / "main" / "smoke_trace.json")
                       for arm in (BASELINE, PAPER_INTRA_ARM))
    base_manifest, paper_manifest = (read_json(preflight / arm / "main" / "manifest.json")
                                     for arm in (BASELINE, PAPER_INTRA_ARM))
    identity = ("author_commit", "author_source_sha256", "extension_sha256", "data_sha256",
                "libraries", "seed", "batch_size")
    first_base, first_paper = (row["batches"][0]["base_losses"] for row in (baseline, paper))
    checks = {
        "historical_author_h0": reference["arm"] == BASELINE and reference["status"] == "completed"
        and reference["hier_enabled"] is False and reference["smoke"] is False
        and reference["argv"] == POINTMLP_ARGS and reference["seed"] == SEED
        and reference["steps_per_epoch"] == 356 and reference["test_rows"] == 2882,
        "runtime_and_data_match": all(reference[key] == base_manifest[key] == paper_manifest[key]
                                      for key in identity),
        "paired_initial_state_and_four_inputs": match_inputs(baseline, paper),
        "paired_rng": baseline["rng"] == paper["rng"],
        "class_and_contrastive_loss_unchanged": all(
            math.isclose(a, b, rel_tol=1e-5, abs_tol=1e-5) for a, b in zip(first_base[:2], first_paper[:2])
        ),
        "native_hierarchy_loss_changes": math.isfinite(first_base[2]) and math.isfinite(first_paper[2])
        and first_paper[2] > first_base[2],
        "inter_object_hierarchy_disabled": not base_manifest["hier_enabled"] and not paper_manifest["hier_enabled"],
    }
    if reference_smoke is not None:
        checks["historical_initial_state_and_four_inputs"] = match_inputs(reference_smoke, baseline)
    passed = all(checks.values())
    support.write(preflight / "acceptance.json", {
        "passed": passed, "checks": checks, "observed_at": support.now(),
        "initial_model_sha256": baseline["initial_model_sha256"],
        "first_base_losses": {BASELINE: first_base, PAPER_INTRA_ARM: first_paper},
        "batch_configuration": {"batch_size": 32, "steps_per_epoch": 356, "epochs": 300},
        "python": {"reference": reference["python"], "current": paper_manifest["python"]},
        "historical_code": reference["source_commit"],
    })
    if not passed:
        raise RuntimeError(f"Paper intra preflight failed: {checks}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True, choices=(0, 1))
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--reference-manifest", type=Path, required=True)
    parser.add_argument("--reference-smoke", type=Path)
    args = parser.parse_args()
    reference = read_json(args.reference_manifest)
    reference_smoke = read_json(args.reference_smoke) if args.reference_smoke else None
    root = args.run_root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    preflight = root / "preflight"
    preflight.mkdir()
    state = {"arm": PAPER_INTRA_ARM, "gpu": args.gpu, "manager_pid": os.getpid(), "seed": SEED,
             "source_commit": args.source_commit, "reference_manifest_sha256": support.sha_file(args.reference_manifest),
             "queued_at": support.now()}

    def status(value, **details):
        state.update(state=value, updated_at=support.now(), **details)
        support.write(root / "queue.json", state)

    status("waiting_for_gpu_for_preflight")
    try:
        with reserve(args.gpu, extra_check=desktop_only) as gpu:
            status("preflight", gpu_uuid=gpu["uuid"])
            for arm in (BASELINE, PAPER_INTRA_ARM):
                folder = preflight / arm
                folder.mkdir()
                code = run_arm(folder, arm, gpu, args.source_commit, smoke=True)
                if code:
                    raise RuntimeError(f"{arm} preflight exited with status {code}; see {folder / 'main.log'}")
        accept(preflight, reference, reference_smoke)
        status("waiting_for_idle_gpu")
        with reserve(args.gpu, extra_check=desktop_only) as gpu:
            code = run_arm(root, PAPER_INTRA_ARM, gpu, args.source_commit, state=state)
        if code:
            raise RuntimeError(f"Production exited with status {code}")
        status("completed", exit_code=0)
    except BaseException as error:
        status("failed", error=repr(error))
        raise


if __name__ == "__main__":
    main()
