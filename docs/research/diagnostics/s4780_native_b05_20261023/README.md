# Seed4780 native/strict-B05 shell-HIER comparison

User-authorized pair only: native c1 without added base cap/GS, versus strict
B05 c.1/cap2.3/base GS/half margins. Both fresh seed4780, non-cyclic balanced
global64, 300x200, full9840 / each-epoch test2468 best OA, no early stopping.

`runner.py` compiles the SHA-pinned V9C continuous-stream loop and selects the
existing arm-specific base operators, geometry, gradient audit and telemetry.
It does not edit legacy production files. HIER is c.1/depth4.6/one branch GS;
B05 forks after cap and before base GS. AdamW is lr.005, eps1e-4, min.0005.

`launch_queue.py` freezes all tracked code dependencies plus this successor,
runs CPU/Gloo gates, verifies data bytes, and locks two compute-idle GPUs by
UUID. Each arm has an independent train-only 2x2 smoke; production is freshly
initialized afterward. Actual saved optimizers and schedules are checked.

Run the contracts with `python -m unittest discover -s <this-directory> -v`.
Launch the runner with `--arm native|b05 --hier on --data-dir ... --run-dir ...`;
the queue also requires repository, immutable commit and source-branch identity.
Do not reuse a result directory, resume the old interrupted run, add H0 arms,
or interpret this two-recipe comparison as an isolated curvature/HIER effect.
