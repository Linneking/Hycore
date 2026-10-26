# PartHier frozen five-question follow-up

Implements the approved94/followup_plan.md, deployed to8000 by the user's latest
instruction. Reuses four complete9840 caches and each PartHier checkpoint's own
staticQ2048. No encoder, optimizer, network backward or training is constructed.
The only autograd calls use independent output leaves for native intra probes.

`pipeline --settings PRIVATE.json` acquires an idleGPU advisory lock, validates
transferred SHA identities, replays the exact200 fixed relation panels, checks
audit94 counts/rates, computes the five readouts, then releases the GPU before
CPU plotting. New roots only; no retry overwrites. Full per-object/per-relation
arrays remain under private output, outside Git.

The replay gate requires exactdraw/valid/collision counts and rate agreement
within1e-5 across GPU models, recording whether rates are exactly equal. Angle
calculations and detached gradient probes useFP64; selection retains production
FP32, explicit Gumbel generators, mining rules and numerical distance operator.

The equal-depth readout modifies only an independent candidate copy. It keeps
all directions andijk, does not replace mainA/B or success rates, and compares
all-draw softmax probability mass. Hotspot expected counts are not conditioned
onA!=B; valid-draw hotspot descriptions are separate. Queue/current exposures
and repeated object/scale identities are distinguished.

The gradient probe retains sharedflip-negative contributions and both moving
endpoints. It describes output-coordinate Euclidean descent, not actual network
parameter/optimizer updates. Finite-pool feasibility exhausts only six displayed
HIER counterexamples; it does not estimate a population feasibility rate.

Synthetic CPU contracts:
`python -m unittest docs.research.diagnostics.parthier_followup95_v1.test_core -v`
