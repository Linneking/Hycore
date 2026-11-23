"""Launch one fresh seed 4781 arm after the shared preflight and idle GPU checks."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import time

from preflight import check_checkout, command, environment, idle_device, read_json
from runner import ARM_NAMES, REPO, SEED, TS44_COMMIT, ts06


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--arm", choices=ARM_NAMES, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--source-branch", required=True)
    args = parser.parse_args()
    from inter_hierarchy_MN40.hier_proxy_scratch_v8.queue import now, write

    folder = args.root / args.arm
    folder.mkdir(parents=True, exist_ok=False)
    destination = folder / "queue.json"
    output = folder / "main"
    record = dict(arm=args.arm, experiment_id=ARM_NAMES[args.arm], gpu=args.gpu, manager_pid=os.getpid(),
                  source_commit=args.source_commit, seed=SEED)

    def status(state, **details):
        record.update(state=state, updated_at=now(), **details)
        write(destination, record)

    try:
        status("waiting_for_preflight")
        while not args.gate.is_file():
            time.sleep(10)
        gate = read_json(args.gate)
        if (gate.get("passed"), gate.get("source_commit"), gate.get("ts44_source_commit"), gate.get("seed")) != (
            True, args.source_commit, TS44_COMMIT, SEED
        ):
            raise RuntimeError("The shared preflight gate rejected this deployment")
        identity = args.gate.with_name("dataset_identity.json")
        if ts06.digest(identity) != gate["dataset_identity_sha256"]:
            raise RuntimeError("The paired data identity changed since preflight")

        status("waiting_for_idle_gpu")
        check_checkout(args.source_commit)
        with idle_device(args.gpu) as gpu, (folder / "main.log").open("x", encoding="utf-8") as log:
            argv = command(args, args.arm, output, identity, expected_initial_model=gate["initialized_model_sha256"])
            process = subprocess.Popen(argv, cwd=REPO,
                                       env=environment(gpu), stdin=subprocess.DEVNULL,
                                       stdout=log, stderr=subprocess.STDOUT)
            status("running", pid=process.pid, gpu_uuid=gpu["uuid"], started_at=now())
            code = process.wait()
        if code:
            raise RuntimeError(f"Training exited with status {code}")
        manifest = read_json(output / "manifest.json")
        if (manifest["status"], manifest["completed_epochs"], manifest["optimizer_steps"],
            manifest["initialized_model_sha256"], manifest["dataset_identity"]) != (
            "completed", 300, 92100, gate["initialized_model_sha256"], read_json(identity)
        ):
            raise RuntimeError("Production did not finish with the approved budget and preflight identities")
        status("completed", exit_code=0)
    except BaseException as error:
        status("failed", error=repr(error))
        raise


if __name__ == "__main__":
    main()
