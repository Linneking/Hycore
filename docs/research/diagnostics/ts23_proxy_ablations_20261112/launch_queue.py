"""Run one approved TS23 proxy ablation on its assigned physical GPU."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
RELATIVE = HERE.relative_to(REPO)
PARENT_DIRECTORY = Path("docs/research/diagnostics/b05_cb32_a1_followup_20261026")
PARENT = REPO / PARENT_DIRECTORY / "launch_queue.py"
PARENT_SHA = "b5bc928252722337fe220fb85067b07fa497888d7aaa70e538de464420cee1ee"
GPU_BY_VARIANT = {"p256": 0, "init05": 1, "cap29": 2}

sys.path.insert(0, str(HERE))
import runner


def identity_fields(variant):
    arm = runner.settings(variant)
    return {
        "reference_alias": "TS23",
        "P": arm["P"],
        "proxy_initialization_coefficient": arm["proxy_initialization_coefficient"],
        "proxy_tangent_cap": arm["proxy_tangent_cap"],
    }


def manager_source():
    source = PARENT.read_text(encoding="utf-8").replace("\r\n", "\n")
    if hashlib.sha256(source.encode()).hexdigest() != PARENT_SHA:
        raise RuntimeError("Pinned TS23 manager changed")
    substitutions = (
        ('choices=("k10p10w01", "k10p20w005")', 'choices=tuple(GPU_BY_VARIANT)'),
        ('choices=(0, 1)', 'choices=tuple(GPU_BY_VARIANT.values())'),
        ('(args.variant, args.gpu) not in (("k10p10w01", 0), ("k10p20w005", 1))',
         'GPU_BY_VARIANT[args.variant] != args.gpu'),
        ('Approved mapping: Ks10/Kp10/lambda.1 GPU0, Ks10/Kp20/lambda.05 GPU1; poll >=10 seconds',
         'Approved mapping: p256 GPU0, init05 GPU1, cap29 GPU2; poll >=10 seconds'),
        ('snapshot["proxy_tangent"].shape != (512, 256)',
         'snapshot["proxy_tangent"].shape != (arm["P"], 256)'),
        ('("proxy_graph", arm["proxy_K"], 512)',
         '("proxy_graph", arm["proxy_K"], arm["P"])'),
    )
    for before, after in substitutions:
        if source.count(before) != 1:
            raise RuntimeError("TS23 manager boundary changed: " + before)
        source = source.replace(before, after)
    return source


def verify_initial_and_first_forward(directory, variant):
    import torch
    from inter_hierarchy_MN40.hier_proxy_scratch_v8.geometry import depth, to_ball

    fields = identity_fields(variant)
    cap = fields["proxy_tangent_cap"]
    coefficient = fields["proxy_initialization_coefficient"]
    generator = torch.Generator(device="cpu").manual_seed(4780)
    reference = torch.randn(512, 256, generator=generator) / math.sqrt(256) * 2.3 * coefficient
    expected = reference[:fields["P"]]
    saved = torch.load(directory / "initialization_e0.pth", map_location="cpu", weights_only=False)
    if any(saved["training_config"].get(key) != value for key, value in fields.items()):
        raise RuntimeError("Saved e0 configuration does not describe the approved ablation")
    if not torch.equal(saved["proxy"]["tangent_proxies"], expected):
        raise RuntimeError("Saved proxy e0 differs from the exact expected TS23 Gaussian draw")

    first = torch.load(directory / "actual_training_epoch_001.pth", map_location="cpu", weights_only=False)
    mapped = to_ball(expected, clip_r=cap, hook=False)
    if first["proxy_parameter_count"] != expected.numel() or not torch.equal(first["proxy_tangent"], expected):
        raise RuntimeError("First real forward did not use the approved count and saved proxy e0")
    torch.testing.assert_close(first["proxy_ball"], mapped, atol=1e-6, rtol=1e-6)
    h = depth(first["proxy_ball"])
    if (bool((h > 2 * cap + 5e-5).any())
            or any(bool((depth(first[key]) > 4.60005).any()) for key in ("native_mu", "native_nu"))):
        raise RuntimeError("First real forward exceeded its native or proxy depth limit")
    torch.testing.assert_close(depth(first["hier_mu"]), torch.full((32,), 4.6), atol=5e-5, rtol=0.)
    return dict(fields, initial_proxy_tensor_sha256=hashlib.sha256(expected.numpy().tobytes()).hexdigest(),
                initial_proxy_depth_median=float(h.median()), initial_proxy_depth_max=float(h.max()))


def main():
    scope = {"__name__": "ts23_ablation_manager_base", "__file__": str(PARENT),
             "GPU_BY_VARIANT": GPU_BY_VARIANT}
    exec(compile(manager_source(), str(PARENT) + "[TS23 proxy ablations]", "exec"), scope)
    scope.update(runner=runner, RELATIVE=RELATIVE)
    freeze_parent = scope["freeze_source"]
    accept_parent = scope["accept_run"]
    write_parent = scope["write"]

    def freeze_source(repo, root, commit, variant):
        snapshot = freeze_parent(repo, root, commit, variant)
        relative = PARENT_DIRECTORY / "runner.py"
        subprocess.run(["git", "-C", str(repo), "ls-files", "--error-unmatch", "--", relative.as_posix()],
                       check=True, stdout=subprocess.DEVNULL)
        target = snapshot / relative
        target.parent.mkdir(parents=True, exist_ok=False)
        shutil.copy2(repo / relative, target)
        path = root / "source_identity.json"
        identity = json.loads(path.read_text(encoding="utf-8"))
        identity["files"][relative.as_posix()] = hashlib.sha256(target.read_bytes()).hexdigest()
        identity["successor"].update(identity_fields(variant))
        write_parent(path, identity)
        return snapshot

    def accept_run(directory, variant, initial_hash, smoke):
        accepted = accept_parent(directory, variant, initial_hash, smoke)
        accepted.update(verify_initial_and_first_forward(directory, variant))
        return accepted

    def write(path, value):
        if path.name == "queue_status.json":
            value.update(identity_fields(value["variant"]))
        write_parent(path, value)

    scope.update(freeze_source=freeze_source, accept_run=accept_run, write=write)
    scope["main"]()


if __name__ == "__main__":
    main()
