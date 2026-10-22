# Independent review: V9 LR/eps ten-epoch diagnostic

Reviewed 2026-10-22 against the report71 source observer, report71/72 evidence,
the current successor runner/queue and the original V9 training source. This is
a read-only design/source review, not a GPU run or a review of training outcomes.

## Scope and source verdict

The successor is an authorized combined stability experiment: proxy AdamW
initial LR `.01 -> .005` and eps `1e-8 -> 1e-4`. It does not isolate the causal
contribution of either individual factor. The generated training source recovers
the complete report71 generated source when those optimizer values and the
checkpoint version are restored. The backbone optimizer, margin/tau `.1`,
lambda `.5`, native-c1/HIER-c.1 geometry, HIER whole depth4.6, cap2.3, GS,
sampling, initialization, synchronization and observations therefore remain
unchanged at the reviewed transformation boundaries. The two schedulers keep
their original300-epoch horizon; proxy eta_min remains `.0005`. Only ten epochs
of200 updates are run, with no evaluation/test read or model selection.

Nine successor source/CLI contracts passed independently on CPU. These include
the actual optimizer AST, metadata matching, exact source delta, train-only
loading, original cosine horizon, fresh smoke and fail-closed source identity.
The inherited numerical/observer contracts must still pass in the deployment
runtime before GPU use. The GPU queue must use unknown occupancy as unavailable,
keep the existing process locks, recheck idleness before smoke/main, and submit
only one instance after choosing the user-prioritized primary or fallback host.

## Identity acceptance

- Match the original seed22 network e0 SHA and proxy e0 SHA to report71, with
  an independent fresh main initialization after the separate2x2 smoke.
- `load_shards` is inherited from v2_online and explicitly sorts shard filenames.
  Preserve this row order; verify sorted train-shard names/byte hashes, training
  labels and first-batch IDs/crop counts. A split-ID hash alone does not validate
  point coordinates. No additional reordering is needed.
- Inspect saved optimizer groups as well as configuration: eps `.0001`,
  initial_lr `.005`, weight decay `.01`, betas `.9/.999`, T_max300 and eta_min
  `.0005`. The proxy LR should remain close to `.005` over these ten epochs,
  rather than annealing to its minimum at epoch10.
- Record code/commit/runtime, physical devices, seed, start/end, no teacher or
  input checkpoint, full2000 updates and80 same-forward sparse snapshots.
- Source code equality is not a promise of a bitwise trajectory across devices
  or runtimes. Compare with report71's independent fresh diagnostic, not by
  replacing the historical production V9 trajectory in report69.

## Required interpretation of the observations

Primary: raw norm quantiles, mapped-depth distribution, raw `norm>2.3` count,
mapped `depth>=4.5999` count, first crossings, crossing/return events and unique
ever-crossed/ever-returned proxies. Keep event counts and unique-proxy counts
separate. Lower raw-cap count and lower forward depth are related but distinct.

Mechanism: current signed radial gradient, tangential gradient norm, actual
radial/angular displacement, finite-step norm increase, and historical-m versus
current-gradient/weight-decay attribution using the actual denominator. The
inherited observer captures these without changing the actual training graph.

Progress controls: retain sample and proxy HIER losses, noncollision counts,
per-hinge activation and role/proxy coverage. Fewer capped proxies can result
from smaller effective learning; cap reduction alone cannot establish a better
learned hierarchy, useful directions or improved classification. Report the
base CE/intra progress alongside the geometry, without introducing test-based
selection into this bounded diagnostic.

Reference report71 is ten epochs with end-cap counts229/444/502 at epochs1/5/10,
502 unique first crossings and zero returns. These are a mechanism reproduction,
not historical V9's196/378 early counts. No change to margin is included in this
experiment; its suitability is a separate question requiring noncollision gap
and activation statistics.

No source-level blocker was found in the reviewed successor runner. Deployment
runtime checks, smoke acceptance and actual run completion remain runtime gates.
