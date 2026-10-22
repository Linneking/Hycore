# Frozen feature angular audit

`audit.py` consumes a private JSON spec with an `observations` list. Cache rows
have `key,path,mode,feature_key,source_curvature` and optional `car_class`;
saved-forward rows have `kind="snapshot",key,path` and optional `car_class`.
Paths and raw arrays stay server-local. It creates a fresh output directory.

The common comparison geometry uses c=.1 and whole depth4.6. Pair midpoint
depth and the exact absolute triple minimax center are ideal references, not
solutions of HIER's relative hinge objective or predictions of learned proxies.
Class-balanced within statistics use every nonself unordered pair with equal
mass per class. Between statistics use32 sample pairs per unordered class
pair, separately from normalized class-centroid comparisons. A supervised
triple reference uses64 distinct sameclass pairs per class and a uniformly
sampled different-class third object. Actual saved reciprocal triplet draws
are analyzed separately, preserving their sampling multiplicities.

ModelNet40 car class7 must be reported separately from the macro40 result.
Cars has98 fine-grained training classes; their within-class statistics do not
measure the same label granularity as ModelNet40's one coarse car class.
Frozen clean/eval caches also differ from actual augmented training forwards.
No causal data-distribution effect can be identified from this comparison alone.

Local verification: `python -m unittest discover -s angles -p test_audit.py -v`.
The mathematical geometry module has independent tests one directory above.
No model construction, encoder forward, optimizer, GPU, dataset or test read
is performed by this program. Source SHA256 is checked before and after reads.
