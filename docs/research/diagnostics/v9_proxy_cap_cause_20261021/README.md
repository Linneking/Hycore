# V9 proxy-cap cause diagnostic

User-authorized scope: one fresh, train-only 5–10 epoch observation of V9 on the specified compute server. The executed run used 10×200 updates and the original 300-epoch cosine horizon. No evaluation/model selection or official test read occurs. This is a mechanism reproduction, not a bitwise replay of the historical full-DDP training.

Public findings: [71 report](../../71_V9_PROXY_CAP_CAUSE_2026-10-22/report.md). Private locations and commands are in the ignored engineering handoff71; do not put them here.

## Files

- `runner.py` asserts the normalized original V9 source SHA and each transformation boundary. It keeps the source sampler, forwards, objectives, synchronization, optimizers and schedule while bounding the loop, excluding all selection/test code, and inserting the observer. `--diagnostic-epochs 10` means ten observed epochs with scheduler horizon300. Only `--smoke` permits fewer than200 steps per epoch.
- `launch_queue.py` freezes published source, runs CPU gates, checks two fully idle GPUs twice under existing per-GPU locks, executes a separate2×2 train-only smoke, then starts a fresh bounded diagnostic. Its name deliberately avoids shadowing Python's standard `queue` module.
- `optimizer_step_audit.py` records real raw gradients, m/v and the actual AdamW displacement every update; the additive attribution shares the actual updated second-moment denominator. It does not change production state or RNG.
- `analyze.py` pools the2000 updates and reconstructs recorded-choice/recorded-active-hinge gradients onCPU. Cohorts are defined before each update and weighted by proxy-update count. The CPU full-minus-fixed residual is an approximation to routing; a small forward distance error alone does not bound derivative error.
- `counterfactuals.py` restores a saved real state into privateCPU AdamW clones for one step: native, eps1e−4 only, historical m radial removed, or historical m zero with v/step retained. These are conditional one-step interventions, not fresh optimizers or training trajectories.
- `replay_routing.py` restores the saved CUDA generator state on one verified idle GPU. It requires identical hard choices/hinge activity and bounds replayed full gradients against observed gradients before separating full/fixed on the same GPU distance graph. The executed14 components matched bitwise, eliminating the CPU reconstruction uncertainty for those saved steps. This has no encoder forward or optimizer update.
- `acceptance.py` checks completion, length, test exclusion, source/model identities, replica equality, shell depth and snapshot/checkpointSHA. The public acceptance also records subsequent frozen replay and observed GPU release.
- `plot.py` renders three static PNG/SVG figures from anonymous aggregates only; validates that inputs remain unchanged.
- `loss_force_audit.md`, `optimizer_cap_audit.md`, `review.md` record source-level derivations and reporting boundaries. `cpu_mapping_probe.py/json` quantify cap/GS derivatives.

## Offline use

Run in the existing verified project runtime. Output directories/files must be new; source snapshots remain private and read-only.

```text
python analyze.py --run-dir PRIVATE_MAIN --out NEW_ANONYMOUS_DIRECTORY
python counterfactuals.py --run-dir PRIVATE_MAIN --out NEW_COUNTERFACTUAL_JSON
python replay_routing.py --run-dir PRIVATE_MAIN --out NEW_REPLAY_JSON
python acceptance.py --run-dir PRIVATE_MAIN --out NEW_ACCEPTANCE_JSON
python plot.py PUBLIC_REPORT_DIRECTORY
```

For GPU replay, the launcher must check occupancy and hold the corresponding GPU lock before exposing exactly one device. Never infer availability from historical records. For training, use the published queue rather than directly bypassing its checks.

## Validation

12 observer/source/Adam tests,19 original V9 geometry/base tests,5 offline analyzer/counterfactual tests, and4 frozen routing tests passed in the server runtime; the2×2 GPU smoke passed before fresh training. All2000 actual optimizer updates were reconstructed, maximum coordinate error2.58e−7. The native CPU one-step clones matched recordedGPU parameters within5.96e−8. All80 sparse snapshots and the last checkpoint passed hash checks. The7-step×2-component frozenGPU replay matched distance matrices, hard choices, hinge values, losses and full gradients bitwise.

No dataset, full log, parameter array, sample ID table, credential or host detail belongs in the public outputs. Anonymous public statistics preserve the distinction among raw norm, mapped depth, signed gradient and actual optimizer displacement. Lower cap counts or a smaller hierarchy loss do not alone establish morphology or classification gains.
