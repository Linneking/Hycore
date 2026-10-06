"""CPU comparison of current HIER distances, the released implementation, and exact geometry.

Loads only inspected mathematical definitions from the supplied official source.
Existing training caches are read-only; no model forward, backward or updates.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
from pathlib import Path
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--official-pmath", type=Path, required=True)
    parser.add_argument("--cache", type=Path, action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--runtime", type=Path)
    args = parser.parse_args()
    if args.runtime:
        sys.path.insert(0, str(args.runtime))
    sys.path.insert(0, str(args.repo))
    import numpy as np
    import torch
    from inter_hierarchy_MN40.hier_proxy_scratch_v5.relations import poincare_distance

    torch.set_num_threads(2)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    source = args.official_pmath.read_text(encoding="utf-8")
    names = {"Artanh", "artanh", "_tensor_dot", "_mobius_addition_batch", "_dist_matrix", "dist_matrix"}
    definitions = [node for node in ast.parse(source).body
                   if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
    if {node.name for node in definitions} != names:
        raise ValueError("Official mathematical definitions missing")
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(args.official_pmath), "exec"), namespace)
    released_distance = namespace["dist_matrix"]

    def exact_c(x, y, c):
        numerator = math.sqrt(c) * torch.cdist(x, y, compute_mode="donot_use_mm_for_euclid_dist")
        denominator = torch.sqrt((1-c*x.square().sum(-1))[:, None] * (1-c*y.square().sum(-1))[None, :])
        return 2/math.sqrt(c) * torch.asinh(numerator/denominator)

    def compare(x, y, c=1., square=False):
        current, released = [], []
        for start in range(0, len(x), 32):
            batch = x[start:start+32]
            current.append(poincare_distance(batch, y) if c == 1 else exact_c(batch, y, c))
            released.append(released_distance(batch, y, c=c))
        current, released = torch.cat(current), torch.cat(released)
        denominator = (1-c*x.square().sum(-1))[:, None] * (1-c*y.square().sum(-1))[None, :]
        squared = torch.cdist(x, y, compute_mode="donot_use_mm_for_euclid_dist").square()
        reference = torch.acosh((1+2*c*squared/denominator).clamp_min(1))/math.sqrt(c)
        mask = torch.ones_like(current, dtype=torch.bool)
        if square:
            mask.fill_diagonal_(False)
        error = (current-released).abs()[mask].double()
        result = {
            "pairs": int(mask.sum()), "c": c,
            "current_reference_max_abs_error": float((current-reference).abs().max()),
            "current_max": float(current[mask].max()), "released_max": float(released[mask].max()),
            "absolute_difference_quantiles": dict(zip(("min", "median", "p90", "p99", "max"),
                torch.quantile(error, torch.tensor([0., .5, .9, .99, 1.], dtype=torch.float64)).tolist())),
            "fraction_difference_gt_margin_0_1": float((error > .1).double().mean()),
        }
        if not bool(torch.isfinite(current).all() and torch.isfinite(released).all()):
            raise ValueError("Nonfinite distance")
        return result

    points = torch.tensor([[.999, 0.], [-.999, 0.], [.996, 0.]], dtype=torch.float64)
    current, released = poincare_distance(points), released_distance(points, points, c=1.)
    report = {
        "format": "hier-margin-distance-scale-cpu-audit-v1",
        "scope": "CPU geometry only; existing train caches; no GPU/model/gradient/optimizer/test access",
        "script_sha256": digest(Path(__file__)), "official_pmath_sha256": digest(args.official_pmath),
        "current_relations_sha256": digest(args.repo / "inter_hierarchy_MN40/hier_proxy_scratch_v5/relations.py"),
        "toy_boundary": {"opposite_current": float(current[0, 1]), "opposite_released": float(released[0, 1]),
                         "same_ray_current": float(current[0, 2]), "same_ray_released": float(released[0, 2])},
        "toy_hinges": {"sample_origin_depths": [1.0986122886681098, .9694005571881035, 1.0986122886681098],
                       "proxy_origin_depths": [2.9444389791664407, 2.1972245773362196]},
        "fixtures": {}, "caches": {},
    }
    samples = torch.tensor([[.5, 0.], [.45, 0.], [-.5, 0.]], dtype=torch.float64)
    proxies = torch.tensor([[.9, 0.], [-.8, 0.]], dtype=torch.float64)
    distances = poincare_distance(samples, proxies)
    hinges = torch.stack((distances[0, 0]-distances[0, 1]+.1,
                          distances[1, 0]-distances[1, 1]+.1,
                          distances[2, 1]-distances[2, 0]+.1)).relu()
    report["toy_hinges"].update(distance_rows=distances.tolist(), hinges=hinges.tolist(),
        pair_minimax_index=int(distances[:2].max(0).values.argmin()),
        triple_minimax_index=int(distances.max(0).values.argmin()))
    if bool((hinges != 0).any()):
        raise ValueError("Counterexample does not satisfy hinges")
    generator = torch.Generator().manual_seed(117)
    directions = torch.randn(32, 16, generator=generator, dtype=torch.float64)
    directions /= directions.norm(dim=-1, keepdim=True)
    for c in (.1, 1.):
        tangent_norm = torch.linspace(.02, 2.3, 32, dtype=torch.float64)
        ball = directions * (torch.tanh(math.sqrt(c)*tangent_norm)/math.sqrt(c))[:, None]
        report["fixtures"][str(c)] = compare(ball, ball, c=c, square=True)

    for path in args.cache:
        before = digest(path)
        with np.load(path) as cache:
            proxy_key = "proxy_ball"
            whole_key = next((key for key in ("mu", "whole", "whole_ball", "features", "embedding", "embeddings")
                              if key in cache), None)
            proxies = torch.from_numpy(cache[proxy_key].copy())
            whole = torch.from_numpy(cache[whole_key][:64].copy()) if whole_key else None
        record = {"source_name": path.parent.name+"/"+path.name, "sha256": before,
                  "proxy_count": len(proxies), "whole_key": whole_key, "distances": {}}
        for dtype in (torch.float32, torch.float64):
            record["distances"][str(dtype)+"/proxy_proxy"] = compare(proxies.to(dtype), proxies.to(dtype), square=True)
            if whole is not None:
                record["distances"][str(dtype)+"/whole64_proxy"] = compare(whole.to(dtype), proxies.to(dtype))
        record["source_unchanged"] = before == digest(path)
        if not record["source_unchanged"]:
            raise ValueError("Source cache changed")
        report["caches"][path.parent.name] = record
        print(json.dumps({"phase": "cache_complete", "label": path.parent.name}), flush=True)
    (args.output_dir / "aggregate.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
