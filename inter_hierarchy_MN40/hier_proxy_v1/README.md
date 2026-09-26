# HyCoRe + HIER proxy pilot (isolated v1)

This directory contains the first frozen-backbone ModelNet40 pilot. It does not
modify the historical HyCoRe `intra` objective or earlier `inter` experiments.

- **B0:** original smoothed classification loss + original part–whole intra loss.
- **B1:** B0 + within-class HIER-style sample/proxy loss + proxy–proxy loss.
- **B2:** B1 + cross-class HIER-style sample/proxy loss.

All arms start from the **original** epoch-229 HyCoRe checkpoint (OA 94.044),
not the later A3 checkpoint. The PointMLP Euclidean backbone, including
BatchNorm running statistics, is frozen. The hyperbolic embedding and
classification layers are trainable; B1/B2 also train global unlabeled proxies.
The ModelNet40 training split stays intact. Runs have a predetermined epoch
count and test only the final model; there is no test-driven checkpoint choice.

The neighbor graph for B1/B2 comes from one shared, deterministic multi-view
cache of the original checkpoint's whole-object hyperbolic embeddings. Within
each balanced batch, a positive is a *mutual* same-class top-3 neighbor; the
within-class negative is outside the anchor's directed top-3. The cross-class
negative is half nearest-other-class and half random-other-class. Mining is
detached, while the HIER loss is applied to current student embeddings, so its
gradient reaches the trainable hyperbolic layers and proxies.

The proxy objective follows the **released HIER code's** independent hard
straight-through Gumbel selection and collision masking, not the paper's
conditional triple-proxy description. The 128 proxies, `K_proxy=8`, c=1,
and 5×8 batches are adaptations for this pilot, not HIER paper defaults.
Collision, effective-triplet, gradient, time, and memory statistics should be
read alongside OA/AA. Loss reduction alone is not proof of a meaningful
within-class soft hierarchy.

Run output (logs, caches, checkpoints) belongs in a separate server-local
directory outside Git. The fixed reference cache and calibration JSON are
created before B0/B1/B2 are launched. Test and smoke commands are recorded in
the server-local experiment manifest so that no data, weights, or private host
information enters the repository.
