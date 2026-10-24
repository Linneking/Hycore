"""Detached single-phase runner. Rechecks idle GPUs and holds per-user locks.

All host-specific paths are supplied in a private identity JSON, not in source.
Each bounded phase requires an agent-side quota check before the next phase.
"""
import argparse
import datetime as dt
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

if __package__:
    from .gate import validate_full_gate
else:
    from gate import validate_full_gate


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--identity", type=Path, required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--mode", choices=("smoke", "diagnostic", "full"), required=True)
    p.add_argument("--epochs", type=int, required=True)
    p.add_argument("--steps", type=int, required=True)
    p.add_argument("--queue-size", type=int, choices=(0, 2048), default=2048)
    p.add_argument("--save-on-cpu", action="store_true")
    p.add_argument("--activation-checkpoint", action="store_true")
    p.add_argument("--checkpoint-policy", choices=("all", "small_medium"), default="all")
    p.add_argument("--diagnostic-max-whole", action="store_true")
    a = p.parse_args()
    if a.save_on_cpu and a.activation_checkpoint:
        p.error("choose one activation-memory strategy")
    if a.diagnostic_max_whole and a.mode != "diagnostic":
        p.error("maximum-whole capacity probe is diagnostic-only")
    s = json.loads(a.identity.read_text())
    root = Path(s["root"])
    if Path(a.name).name != a.name or (root / a.name).exists():
        raise RuntimeError("phase name must identify a new child directory")
    cpu_gate_path = root / "cpu_contracts_passed.json"
    if not cpu_gate_path.is_file():
        raise RuntimeError("CPU/Gloo acceptance missing")
    if json.loads(cpu_gate_path.read_text()).get("commit") != s["commit"]:
        raise RuntimeError("CPU/Gloo acceptance belongs to another source commit")
    if a.mode == "full":
        if not a.activation_checkpoint or a.save_on_cpu:
            raise RuntimeError("full run requires the validated BN-preserving checkpoint path")
        full_gate_path = root / "full_gate.json"
        if not full_gate_path.is_file():
            raise RuntimeError("full-run GPU acceptance gate missing")
        full_gate = json.loads(full_gate_path.read_text())
        validate_full_gate(full_gate, s["commit"])
        if a.checkpoint_policy != full_gate["smoke"]["config"].get("checkpoint_policy", "all"):
            raise RuntimeError("full checkpoint policy differs from CUDA training acceptance")
    status_path = root / (a.name + "_dispatch.json")
    if status_path.exists():
        raise RuntimeError("phase has already been submitted")
    def save(**values):
        values.update(commit=s["commit"], mode=a.mode, name=a.name,
                      updated_utc=dt.datetime.now(dt.timezone.utc).isoformat(), parent_pid=os.getpid())
        temporary = status_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(values, indent=2))
        temporary.replace(status_path)
    save(status="waiting_for_idle_gpus")
    handles, chosen = [], []
    for attempt in range(2880):
        try:
            active = set(subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid", "--format=csv,noheader"], text=True, timeout=20).splitlines())
            rows = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,utilization.gpu,memory.used", "--format=csv,noheader,nounits"], text=True, timeout=20).splitlines()
            for row in rows:
                index, uuid, util, memory = [x.strip() for x in row.split(",")]
                if uuid in active or int(util) != 0 or int(memory) > 100:
                    continue
                handle = open("/tmp/hycore_gpu_" + index + ".lock", "a+")
                try:
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    handle.close()
                    continue
                handles.append(handle)
                chosen.append((index, uuid))
                if len(chosen) == 2:
                    break
            if len(chosen) == 2:
                # Recheck compute owners after acquiring the locks.
                active2 = set(subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid", "--format=csv,noheader"], text=True, timeout=20).splitlines())
                if not any(uuid in active2 for _, uuid in chosen):
                    break
        except (subprocess.SubprocessError, ValueError):
            pass
        for handle in handles:
            handle.close()
        handles, chosen = [], []
        time.sleep(30)
    if len(chosen) != 2:
        save(status="failed", error="no idle GPU pair within waiting limit")
        raise RuntimeError("no idle GPU pair")
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=",".join(x[0] for x in chosen),
               HYCORE_PHYSICAL_GPU_UUIDS=",".join(x[1] for x in chosen),
               PARTHIER_SOURCE_COMMIT=s["commit"], PARTHIER_SOURCE_BRANCH=s["branch"],
               PYTHONPATH=s["source"] + "/pointnet2_ops_lib:" + s["source"] + "/inter_hierarchy_MN40:" + s["source"],
               OMP_NUM_THREADS="2", PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1")
    cmd = [s["python"], "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=2",
           "-m", "inter_hierarchy_MN40.parthier_v1.train", "--data-dir", s["data"],
           "--run-dir", str(root / a.name), "--mode", a.mode, "--epochs", str(a.epochs),
           "--steps-per-epoch", str(a.steps), "--queue-size", str(a.queue_size)]
    if a.save_on_cpu:
        cmd.append("--save-on-cpu")
    if a.activation_checkpoint:
        cmd.append("--activation-checkpoint")
        cmd.extend(["--checkpoint-policy", a.checkpoint_policy])
    if a.diagnostic_max_whole:
        cmd.append("--diagnostic-max-whole")
    with (root / (a.name + ".log")).open("x") as output:
        child = subprocess.Popen(cmd, cwd=s["source"], env=env, stdout=output, stderr=subprocess.STDOUT)
        save(status="running", child_pid=child.pid, gpus=chosen, command=cmd)
        code = child.wait()
    save(status="completed" if code == 0 else "failed", exit_code=code, child_pid=child.pid, gpus=chosen)
    for handle in handles:
        handle.close()
    raise SystemExit(code)


if __name__ == "__main__":
    main()
