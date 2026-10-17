# V8-HIER64-FULL9840-TESTSEL

User-authorized 2026-10-17 protocol successor to V8; original V8 entry unchanged.
Train on the full official9840 objects. No independent validation set.
Evaluate official2468 test objects after every epoch in eval/no_grad; strictly
higher full-precision test OA selects best, retaining the earlier checkpoint
on ties. Test never updates gradients/BN, the optimizer or LR, and no early
stopping is performed. Report test-selected results separately from heldout
validation-selected V8.

Keep V8 c=.1, cap2.3, inverse-metric hook, gamma500/N, delta2, HIER weight.5,
K20/T50, 512proxies and256dimensions; seed22;300epochs x200steps.
Two ranks use16disjoint classes x2/local32/global64; class-internal replacement.
All9840 objects are eligible, not guaranteed to be visited each epoch.
Both optimizers/schedulers and initialization match the original V8.

The standalone lifecycle retains original V8 training code with explicit
train/test domains and a distinct checkpoint format. Geometry/loss/telemetry
and the guarded GPU allocator/source freezer are imported from unchanged V8.
Fresh smoke runs use only training objects for scoring; they never read test
and are never used to initialize production. Final_test.json is the selected
checkpoint's recorded full test evaluation, not an extra final forward.

Queue freezes tracked source and binary identities, checks protocol and
geometry/DDP contracts, waits for two completely idle UUID-pinned GPUs,
runs2x2 smoke, and launches a fresh full training in a unique directory.

Fulltrain controls: controls.py provides B64 (200steps/tworanks) and B0-32-C01
(307shuffle32steps/singlerank;9824distinctslots/drop16). Both have zero HIER
calls and no proxies. controls_queue.py prepares once and independently
schedules each arm: two idle devices permit B64; one permits B0-32. It never
requires three simultaneous idle devices, and each arm has its own2x2smoke.
