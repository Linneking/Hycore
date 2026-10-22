# Cars / V9 angular geometry and AdamW update-scale audit

This is a read-only follow-up to report 71, requested on 2026-10-22. It uses
existing feature caches and saved optimizer-step states. It does not start a
training run, encode additional test objects, alter model buffers, or change
the production optimizer.

The audit separates three questions:

1. How close to the depth-4.6 shell would pair and triple **absolute minimax
   reference centers** lie, given each model's directions? These reference
   centers are not claimed to be HIER's loss optimum or learned ancestors.
2. How large are actual saved AdamW updates compared with the remaining raw
   radius budget? How sensitive is one fixed-state step to epsilon, learning
   rate, and common gradient scaling?
3. Which differences in task, sampling, dimension, precision, schedules and
   feature distributions remain confounded in a Cars versus ModelNet40
   comparison?

`angles/`, `adam_scale/`, and `math/` contain independent audit components.
Public outputs contain aggregate statistics only. Full features, checkpoint
states, source paths and sample identities remain private. Every remote
execution uses a new output directory; old caches and reports are read-only.

Code is edited and checked in the authoritative local working copy, committed
and pushed on a dedicated branch, then fetched and fast-forwarded on the
server before execution. CPU analysis is preferred; GPU use requires an idle
device check and is unnecessary when existing caches suffice.

The report must distinguish observed late clean/eval geometry, actual early
training-forward geometry from the separate report-71 diagnostic repeat,
mathematical reference geometry, and fixed-state one-step interventions.
None alone establishes a long-run causal training improvement.
