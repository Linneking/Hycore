# Local–GitHub–server workflow

## Experiment-end reporting — user requirement, 2026-11-02

Every completed experiment must be explained in the conversation, not only by a score table, log, or artifact link. Report each logical experiment with:

1. **Reason and hypothesis:** which observation motivated it, what remained unknown, and what the experiment was intended to distinguish.
2. **Control:** what changed, what was matched, and which comparison identifies the effect.
3. **Result:** the necessary numbers, including relevant failure cases and tradeoffs.
4. **Conclusion and limits:** whether the hypothesis is supported, weakened, or unresolved; distinguish local diagnostic findings from full-task gains and avoid treating absence of evidence as exclusion.
5. **Consequence:** how the result changes the next step, or why it does not justify a change yet.

Progress updates must explicitly remain interim. Do not present a pending full-training run as completed, and do not substitute a local probe score for the DGCNN Class mIoU target. Future plans and reports must carry these explanations forward so the user does not have to reconstruct the reasoning from previous messages.

## Sources of truth

- **Code source of truth:** local `D:\Hycore` working copy.
- **Version exchange and backup:** `git@github.com:Linneking/Hycore.git`.
- **Compute and large artifacts:** the private GPU server described in `.codex-local/SERVER_PATHS.md`.

Datasets, checkpoints, teacher caches, complete logs, TensorBoard/W&B data, host details, credentials, and keys must never be pushed to GitHub.

## Normal development cycle

```bash
# Local PowerShell / VS Code terminal
cd D:\Hycore
git status --short --branch
git switch -c experiment/<descriptive-name>
# edit, test, review
git add <explicit-files>
git commit -m "..."
git push -u origin experiment/<descriptive-name>
```

```bash
# Server
cd <server-repo>
git status --short --branch
git fetch Hycore
git switch experiment/<descriptive-name>
git pull --ff-only Hycore experiment/<descriptive-name>
```

Never pull into a dirty server worktree. Do not use `git reset --hard` to resolve drift. If a server hotfix is unavoidable, commit it on the experiment branch and push it before continuing locally.

## Experiment identity

2026-10-23最新执行授权：用户已选择严格保留B05基础GS，要求两条HIER-on均部署8001并明确“启动”；断电期间仅设计/不连接约束已由该指令覆盖。仅执行[79](79_S4780_NATIVE_B05_LAUNCH_2026-10-23.md)所列两条fresh实验，不自动增加匹配H0或重启旧任务。以下“current request is design only”为此前历史状态。

Use [75: TS baseline/component names](75_TS_EXPERIMENT_BASELINE_ATLAS_2026-10-22/report.md) and [TS registry v3](experiment_registry_ts_v3.json) for current TS display aliases. Keep [63](63_EXPERIMENT_NAMING_AND_GRADIENT_SCOPE_2026-10-20.md), registry v2, [33](33_EXPERIMENT_NAMING_CONVENTION_2026-10-06.md) and the legacy registry for historical identities. Latest user-authorized future defaults (2026-10-23): full9840 training/test2468 each-epoch best OA, no early stopping; **seed4780 for every lineage**, **class-balanced64 with within-class replacement; abandon cyclic64**. Historical validation-selected runs retain VS984 and all historical runs retain their actual seeds. Server power loss was reported; the current request is design only, with no server connection, implementation, dispatch or automatic restart authorized. See [76](76_S4780_FIXED_SHELL_BASELINE_DESIGN_2026-10-23.md) for unresolved B05 GS scope and suggested matched controls. B0-32/B0-64 historically mean shuffle; B32/B64 mean class-balanced with HIER disabled. Record base cap/GS separately from HIER view/GS and optimizer. Every new manifest records canonical display name and immutable run ID; an instantaneous diagnostic base arm is not a trained baseline.

Every run directory must contain or log:

- Git commit SHA and branch;
- complete command and resolved configuration;
- seed and data split identity;
- source checkpoint and teacher-cache identity/hash;
- physical GPU index and environment versions;
- start/end timestamps, wall time, and peak memory;
- metrics CSV/JSON and failure status.

Run directories are immutable. A rerun receives a new directory.

## Result retrieval

The agent reads full artifacts through SSH using `.codex-local/SERVER_PATHS.md`. Only concise, reviewed summaries may be committed under `docs/research/`; raw logs and weights remain server-local or are downloaded to a non-repository delivery directory.

## Branch roles

- `main`: stable user branch.
- `codex/inter-hierarchy-v2`: first corrected implementation and handoff base.
- `experiment/*`: one focused protocol or method question per branch.
- Merge only after tests pass and the experiment manifest is recorded.

