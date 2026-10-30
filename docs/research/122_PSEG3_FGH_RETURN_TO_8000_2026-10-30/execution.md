# F/G/H return migration and resume queue — 2026-10-30

The user requested that the regularizer-enabled F/G/H jobs release their previous compute resources and continue on the original three-GPU node. Existing F0/G0/H0 jobs were not interrupted.

The source jobs were stopped after preserving complete checkpoints. The compact transfer contains `last.pth`, `best.pth`, checkpoint-selection metadata, complete metric history through the checkpoint epoch, and geometry history. Its SHA256 is `963a06f77b87d163a897252999e68b22e0f01095bdb670e7b768d0c0508b16d9`. Original source run directories remain unchanged.

The destination uses the same frozen training commit `c8b93dc66e2dd5588f77c1490ec085010b855a01`, the same production data manifest `452943480f7fbdf82507a8306e3f535963ec3a4ae93f57807b3b6c36957ebfb8`, and the existing Python 3.9/Torch 2.8/Geoopt 0.5.1 runtime. All 42 v6 CPU contracts passed before queue submission.

Checkpoint preparation changed only the copied checkpoint's `physical_gpu` field. Model parameters, optimizer, scheduler, global RNG, loader RNG, part sampler RNG, and negative sampler RNG were compared recursively and remained equal. The source checkpoint copies and all old results were preserved.

| Arm | Complete source epoch | Updates | Destination GPU | Queue state at 22:07 Beijing |
|---|---:|---:|---:|---|
| F | 128 | 55,936 | 0 | waiting for F0 to release a completely idle GPU |
| G | 131 | 57,247 | 1 | waiting for G0 to release a completely idle GPU |
| H | 130 | 56,810 | 2 | waiting for H0 to release a completely idle GPU |

Each detached queue requires three consecutive idle observations: zero compute processes, 0% utilization, and less than 500 MiB used. The training entry point repeats its own idle-GPU check immediately before loading the resume state. Each job keeps the original total budget of 200 epochs; the resumed output is a new run directory.

At the independent reconnect check, all three queue processes had parent PID 1 and were waiting without GPU allocation. F0/G0/H0 remained healthy at epochs 170/170/165 and were expected to release GPU0/1/2 around 22:58/22:58/23:10 Beijing. Exact paths and process identities remain in the ignored local handoff.
