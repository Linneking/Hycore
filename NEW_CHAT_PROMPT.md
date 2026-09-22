# Prompt for the next Codex project conversation

We are continuing a research project on inter-sample hierarchy learning for HyCoRe point-cloud classification. The local repository `D:\Hycore` is the authoritative code working copy and is synchronized through `git@github.com:Linneking/Hycore.git`; the GPU server is used only for execution and large artifacts.

Before proposing changes, read `AGENTS.md` and every file it requires, including the private local `.codex-local/SERVER_PATHS.md`. Then inspect Git status, branch, and the v2 code. Do not launch experiments yet.

Your first task is to review `docs/research/04_NEXT_EXPERIMENT_PLAN.md` critically. Check whether its controls, train/validation/test separation, teacher-cache reuse, trainable-scope comparison, metrics, gates, and scientific interpretation are sufficient. Identify confounds or unnecessary experiments, propose a minimal revised plan with estimated GPU hours, and ask only questions whose answers would materially change that plan.

Important state: the original A3 checkpoint reproduces OA 93.9222%/AA 91.5907%. The first 20-epoch v2 run found that cosine teacher similarity was unusable, while frozen hyperbolic distance supplied dense triplets. Exact-LCA ranking modestly improved structural metrics relative to a matched zero-increment control, but both fine-tuned models substantially degraded from the original checkpoint. Therefore stabilize the protocol before multi-seed or broad ablation work, and do not claim discovery of a genuine morphology hierarchy from HyCoRe self-distillation alone.

