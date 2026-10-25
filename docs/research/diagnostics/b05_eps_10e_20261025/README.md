# B05 epsilon-only paired ten-epoch diagnostics

Approved2026-10-25 on server8000. Both fresh seed4780, strictB05 c0.1/cap2.3/
baseGS/500_N and2, pre-base-GS fixed4.6 HIER branch, CB64=32classesx2withreplacement,
two local32 ranks, K20sample/proxy, P512/D256,T50,lambda.5,margin/tau.1.
Both proxyLR.005, minLR.0005, betas.9/.999,WD.01; ONLY epsilon differs1e-8/1e-4.
The small-epsilon arm is not historical A0 (which hadLR.01).

`runner.py --eps small|large --data-dir ... --run-dir ...` runs10epochsx200steps,
with both ORIGINAL300-epoch cosines. No test/eval forward, BN recalibration,
model selection, teacher, source checkpoint, resume or warmup. A separate
`--smoke --diagnostic-epochs2 --steps-per-epoch2` also keeps300-epoch schedules.
The reviewed parent is SHA-pinned, historical files are never edited.

`observer.py` reads actual synchronized proxy.grad afterlambda.5/cap/GS and
beforeAdamW, then the actual parameter update. Eachstep records per-coordinate
sqrt(vhat), epsilon dominance, detached step reconstruction and fixed-moment
epsilon counterfactual updates. These counterfactuals are algebraic diagnostics,
not additional training arms. Epsilon does not directly transform raw gradients.
Independent weight decay is separated from the adaptive update.

Each epoch saves all512 actual mapped depths, rawtangentnorms, signed changes
from the previous epoch and initialization, cap counts, per-step firstcrossings
and returns, plus inward/outward/stationary counts. Cap can conceal raw motion,
so mapped and raw quantities remain separate. Everyepoch's firstbatch backbone
gradient audit clones the actual current private Gumbelstate/relationseed and
restoresBN/RNG. This is a firstbatch probe, not an epochmean backbone gradient.

`launch_queue.py` freezes all published dependencies, checks CPU contracts and
data identity, reserves idle physicalGPU0/1, runs both independent2x2smokes, then
freshsmall10 andfreshlarge10 sequentially. Actual optimizer/scheduler,2000step
records,10epochdistributions,BN4000 and replica checks must pass. Both initial
models/proxies and firstbatch input identity must match; later features and
Gumbel draw lengths may diverge along the different trajectories.

Afterboth finish, `analyze.py --small-run ... --large-run ... --out NEW_DIR`
automatically producesCSV/JSON,9PNG/SVG plots and aChinese report. No chat polling
or monitoring automation is needed. All weights, raw logs and instance IDs stay
in private server directories. Only reviewed aggregates enterGit.
