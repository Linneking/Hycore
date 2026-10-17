"""Seconds-only frozen symptom check; exit1 flags the reported geometry."""
import argparse
import json
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache")
    args = parser.parse_args()
    with np.load(args.cache, allow_pickle=False) as data:
        tangent = data["proxy_tangent"].astype(np.float64)
        whole = data["mu"].astype(np.float64)
    norms = np.linalg.norm(tangent, axis=1)
    depths = 2 / np.sqrt(.1) * np.arctanh(np.sqrt(.1) * np.linalg.norm(whole, axis=1))
    report = dict(proxy_over_cap=int((norms > 2.3).sum()), proxy_count=len(norms),
                  raw_tangent_median=float(np.median(norms)), whole_depth_std=float(depths.std()),
                  whole_depth_median=float(np.median(depths)))
    report["symptom_present"] = bool((norms > 2.3).mean() > .9 and depths.std() > .02)
    print(json.dumps(report))
    raise SystemExit(1 if report["symptom_present"] else 0)


if __name__ == "__main__":
    main()
