# Experiment I execution record

## Local source and checks

Source branch `codex/pseg3-i-reencode-20261101`, latest local commit `0cdcd2b`. The new implementation is isolated in `segmentation_ShapeNetPart/hycore_seg_v8/`; frozen H and earlier paths were not edited.

The six focused CPU contracts pass. A fresh synthetic full-entry run completed two epochs and four optimizer updates, wrote best/last/full-state checkpoints, and kept model and optimizer finite. The 14 BN counters each reached eight, matching two train-mode forwards per update. The sampled CE gradient reached the final Möbius classifier; each weighted regularizer reached shared Conv1/Conv8, Conv9/BN9 and the first Möbius map, while the final classifier gradient from either regularizer was `None`.

The launcher's acceptance parser was exercised against that CPU run: all applicable routing, BN, state and checkpoint checks passed; the expected real-data/full-size gates correctly failed on synthetic small data. The real-data GPU smoke remains required before production.

## Deployment gate

The project workflow calls for pushing the dedicated branch and fetching/pulling the exact commit on the training server. Automatic approval review rejected the GitHub push of non-public Experiment I source because the exact destination's ownership and authorization to receive this code had not been confirmed. No alternate transfer or GPU run was attempted. Deployment and queueing remain pending explicit confirmation for the named project remote.
