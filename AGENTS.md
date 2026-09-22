# HyCoRe research-agent instructions

Before planning or editing, read these files in order:

1. `docs/research/00_RESEARCH_HANDOFF.md`
2. `docs/research/01_EXPERIMENT_HISTORY.md`
3. `docs/research/02_WORKFLOW.md`
4. `docs/research/04_NEXT_EXPERIMENT_PLAN.md`
5. `.codex-local/SERVER_PATHS.md` when it exists locally

Operational rules:

- Treat `D:\Hycore` as the authoritative code working copy. Edit and test here, commit to a dedicated branch, push to GitHub, then update the server with `git fetch` and `git pull --ff-only`.
- Do not commit datasets, checkpoints, teacher caches, full logs, TensorBoard/W&B data, credentials, host details, or private keys.
- Do not modify or delete existing server results. Every run must use a new directory and record commit, config, seed, GPU, start time, and teacher/checkpoint identity.
- Before using a GPU, inspect `nvidia-smi`; use only completely idle GPUs and never interrupt another user's process.
- Keep the original HyCoRe and legacy inter-hierarchy paths reproducible. New work belongs in the v2 path or a clearly versioned successor.
- Do not launch the next experiment matrix until the user has reviewed `04_NEXT_EXPERIMENT_PLAN.md`.
- Separate three effects in analysis: correctness fixes, training-protocol changes, and the inter-sample hierarchy objective itself.
- A frozen HyCoRe teacher is self-distillation. Do not describe it as evidence of a genuine morphology hierarchy without an independent geometric or semantic validation signal.
- Prefer validation data for model selection. The ModelNet40 test set must not be used every epoch for early stopping in the next protocol.

