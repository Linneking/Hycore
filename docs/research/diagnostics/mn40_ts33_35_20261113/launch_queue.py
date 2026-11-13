"""Launch TS33/TS34 through the existing B13 single-GPU gates."""
from pathlib import Path

import runner

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
PARENT = HERE.parent / "b05_cb32_a1_followup_20261026" / "launch_queue.py"
PARENT_SHA = "b5bc928252722337fe220fb85067b07fa497888d7aaa70e538de464420cee1ee"
GPU_BY_VARIANT = {"ts33": 0, "ts34": 1}


def manager_source():
    source = PARENT.read_text(encoding="utf-8")
    if runner.parent.sha_text(source) != PARENT_SHA:
        raise RuntimeError("Pinned B13 manager changed")
    for before, after in (
        ('choices=("k10p10w01", "k10p20w005")', 'choices=tuple(GPU_BY_VARIANT)'),
        ('(args.variant, args.gpu) not in (("k10p10w01", 0), ("k10p20w005", 1))',
         'GPU_BY_VARIANT[args.variant] != args.gpu'),
        ('Approved mapping: Ks10/Kp10/lambda.1 GPU0, Ks10/Kp20/lambda.05 GPU1; poll >=10 seconds',
         'Approved mapping: TS33 GPU0, TS34 GPU1; poll >=10 seconds'),
    ):
        source = runner.parent.replace_once(source, before, after)
    return source


if __name__ == "__main__":
    scope = {"__name__": "ts33_ts34_queue", "__file__": str(PARENT), "GPU_BY_VARIANT": GPU_BY_VARIANT}
    exec(compile(manager_source(), str(PARENT), "exec"), scope)
    scope.update(runner=runner, RELATIVE=HERE.relative_to(REPO))
    scope["main"]()
