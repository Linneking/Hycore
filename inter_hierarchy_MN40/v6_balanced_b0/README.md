# V6 balanced64 B0 complete-state fork

Production: python -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=2 --module inter_hierarchy_MN40.v6_balanced_b0.train --data-dir DATA --run-dir NEW --source-checkpoint H20_E20

Restore smoke adds --fork-smoke --smoke-steps 2 --smoke-epochs 1 --smoke-eval-batches 1 --skip-final-test. Smoke remains a fresh diagnostic directory, is not resumable and never replaces production. --source-sha256 HASH optionally enforces an independently recorded source digest.

This is the historical V6 matched control: inherited epochs1-20 plus new epochs21-300. It never repeats warmup. The source configuration's warmup_epochs, P/K/T and proxy initialization entries are retained as source protocol metadata; the explicit target configuration disables HIER and proxy optimization. Future new HIER runs follow the user's zero-warmup instruction through their separate entry.

--resume B0_COMPLETE continues only this format, requires the same --source-checkpoint, and writes a new directory. Selection-only, partial, old HIER post20, wrong data/config, incomplete ranks, reset scheduler or missing momentum states are rejected. Full model state, RiemannianSGD momentum, cosine state, each rank's BN/RNG/sampler and selected model/history are restored and audited before any forward. There is no proxy allocation, relation mining or extra forward. Source HIER proxy initialization used a private generator, so omitting it consumes no base RNG.

Only CE + .01 Rcontrastive + .01 Rradial updates the backbone. The exact V6 global64 loss operators, balanced replacement sampler, local32 ordinary BN, original part alias/FPS and RNG formulas remain. First/last-step gradients and parameters are compared across ranks; BN must update exactly twice. Every step records IDs/loss/geometry/heartbeat; every epoch validates and saves a complete last checkpoint. Archives include regular20-epoch intervals and99/200/300. Validation selects the model over the inherited prefix plus continuation; official test is read once after completion.

The old H20 curve is a same-protocol historical reference, not a promised bitwise paired continuation. Clean geometry caches/independent morphology comparisons are separate jobs and must not stall this main run.
