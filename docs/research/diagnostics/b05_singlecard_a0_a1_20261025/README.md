# B05 single-card fixed-shell HIER: A1 versus A0

Approved pair, 2026-10-25: physical GPU0 uses new A1 AdamW LR0.005/eps1e-4;
physical GPU1 uses old A0 LR0.01/eps1e-8. Both keep B05 c0.1/cap2.3/base GS/
500_N and2, the pre-base-GS depth4.6 HIER branch, P512/D256, T50, lambda0.5,
margin/tau0.1, FP32, seed4780, non-cyclic replacement-balanced16 classes x2,
sample K10/proxy K20, fresh300 epochs x200 updates. No H0 arm or resume.

The original dual-card entry points remain unchanged. `runner.py` pins and
transforms the reviewed continuous-stream parent, including every single-rank
collective, global32 base loss, actual split-K HIER forward and gradient audit.
The only A0/A1 difference is proxy initial LR and epsilon; both end at0.0005.

`launch_queue.py` freezes the complete committed source and extension, validates
CPU contracts, initialization and private data identity, then waits for its
specified GPU with no compute processes, utilization0 and free memory>=30000MiB.
It takes the shared per-user/GPU lock and checks idle again before allocation.
Only after that does a separate train-only2x2 smoke execute. Actual saved
optimizer/scheduler, single-rank state, BN8, finite tensors and split K must
pass before freshly initialized production. Smoke weights never initialize it.
Each arm has a new result root. Failure records a state file and stops that arm;
no other process is interrupted, no original result is modified or deleted.

Selection remains the user-authorized full9840/each-epoch test2468 protocol,
strict best OA without early stopping. CB32x200 has60000 optimizer updates and
1920000 sample presentations; it is not equal data exposure to CB64x200.

The queues survive chat exit. User asked to stop after submission, so real CUDA
smoke is deferred until idle; a queued task must not be described as CUDA-tested
or already training. No separate monitoring automation is created.
