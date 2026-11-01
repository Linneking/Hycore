# Experiment I execution record

## Local source and checks

Source branch `codex/pseg3-i-reencode-20261101`, latest local commit `0cdcd2b`. The new implementation is isolated in `segmentation_ShapeNetPart/hycore_seg_v8/`; frozen H and earlier paths were not edited.

The six focused CPU contracts pass. A fresh synthetic full-entry run completed two epochs and four optimizer updates, wrote best/last/full-state checkpoints, and kept model and optimizer finite. The 14 BN counters each reached eight, matching two train-mode forwards per update. The sampled CE gradient reached the final Möbius classifier; each weighted regularizer reached shared Conv1/Conv8, Conv9/BN9 and the first Möbius map, while the final classifier gradient from either regularizer was `None`.

The launcher's acceptance parser was exercised against that CPU run: all applicable routing, BN, state and checkpoint checks passed; the expected real-data/full-size gates correctly failed on synthetic small data. The real-data GPU smoke remains required before production.

## Deployment gate

The project workflow calls for pushing the dedicated branch and fetching/pulling the exact commit on the training server. Automatic approval review initially rejected the GitHub push of non-public Experiment I source because the exact destination and payload had not been explicitly authorized. The user first chose to keep the code local, then explicitly authorized pushing commit `cdadb55` to the named project remote for deployment. No alternate transfer was used.

## Remote acceptance and training start — 2026-11-01

The dedicated branch was pushed to the project remote, and an isolated clean server checkout fetched and pulled `cdadb5536ed09eb9556719bd59d5bbcb7bea2df4` with `--ff-only`. Existing source trees and results were not modified. The six CPU contracts passed again in the server environment.

The launcher acquired a fully idle RTX 5090 GPU after repeated checks. Real ShapeNetPart validation smoke used batch 32, 2048 points, part 600, two epochs and four updates, with no test evaluation. All ten acceptance gates passed: finite model/optimizer state, 14 BN counters each advancing twice per update, H-matched initial SHA `17c51802fac0a58caf86f4267e5dc8370f35c6f15b9eb74c11a224ddb184f940`, and expected CE/regularizer gradient routing. The launcher waited for the smoke process to release the GPU, reacquired an idle GPU, and then began a fresh 200-epoch run from that same initial SHA. No H checkpoint or teacher was loaded. Production data-manifest SHA is `452943480f7fbdf82507a8306e3f535963ec3a4ae93f57807b3b6c36957ebfb8`.

The first production epoch completed 437 updates and evaluated all 2,874 test objects. Instance mIoU was 65.717%, class mIoU 47.766%; these are startup metrics, not a final comparison. Model and optimizer were finite, the 14 BN counters were all 874, and both best and last full-state checkpoints were saved. The run entered epoch 2. The existing H-matched test-every-epoch selection protocol is retained for comparability and is not a validation-selected final estimate.
