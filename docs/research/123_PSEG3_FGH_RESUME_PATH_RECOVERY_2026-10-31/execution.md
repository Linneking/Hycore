# F/G/H resume data-path recovery — 2026-10-31

The previous migration queues passed the idle-GPU gate but all three trainers rejected their resume configuration before restoring a checkpoint or performing an update. The copied checkpoints still contained the source node's absolute data directory. The target data manifest matched; the absolute `data` field was the only remaining configuration difference.

Recovery used new directories and the same frozen training commit `c8b93dc66e2dd5588f77c1490ec085010b855a01`. The copied checkpoint configuration now adapts both the verified target data directory and physical GPU. No model, optimizer, scheduler, RNG, regularizer coefficient or epoch budget changed. Original checkpoints, failed attempts and historical metrics were preserved.

`resume_relocated.py` validates the adapted configuration using the original entry point's argument parser and exact resume comparison. It then checks that every non-configuration checkpoint field is equal to its source. Actual GPU restoration was independently verified by comparing the new `e0.pth` to the resume input: model, optimizer, scheduler, global Python/NumPy/Torch/CUDA RNG, train/eval loader RNG, part/negative RNG, epoch/update counters and best-selection state all matched exactly.

At 00:21 Beijing, each arm had completed its first resumed epoch, evaluated all 2,874 test objects, saved a full checkpoint, and entered the following epoch. The first resumed epochs each had 437 updates and finite model/optimizer state. Detached workers had parent PID 1. F/G/H remained on GPU0/1/2, total budget 200 epochs.

| Arm | Resume after | First complete resumed epoch | Instance mIoU | Class mIoU | Current epoch at 00:21 |
|---|---:|---:|---:|---:|---:|
| F | 128 | 129 | 83.512% | 70.422% | 130 |
| G | 131 | 132 | 82.013% | 64.286% | 133 |
| H | 130 | 131 | 83.657% | 73.355% | 132 |

Recent epoch durations were 107.7/107.7/110.7 seconds, giving approximate completion times 02:28/02:22/02:27 Beijing on October 31. The existing read-only query script was updated to the successful recovery directory and combines preserved source metrics with newly completed epochs. Private deployment paths and PIDs are excluded from this report.

F0/G0/H0 had already completed 200 epochs normally, with 87,400 updates each and no failure file. Their best Instance/same-epoch Class results were F0 84.521/76.093 at e173, G0 82.897/64.797 at e162, H0 84.521/76.093 at e173 (percent). F0/H0 also had matching last-epoch metrics, consistent with their identical active main paths when regularizers are disabled.
