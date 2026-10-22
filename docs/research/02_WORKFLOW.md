# Local–GitHub–server workflow

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

