"""CPU-only mapping Jacobian probe against the exact V9-imported source."""
import importlib.util
import json
from pathlib import Path

import torch


def main():
    torch.set_num_threads(2)
    here = Path(__file__).resolve().parent
    repo = here.parents[3]
    source = repo / "inter_hierarchy_MN40/hier_proxy_scratch_v8/geometry.py"
    spec = importlib.util.spec_from_file_location("actual_v9_proxy_geometry", source)
    geometry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(geometry)
    generator = torch.Generator().manual_seed(22)
    raw = torch.randn(512, 256, generator=generator) / (256 ** .5) * 2.3 * .9
    norms = raw.norm(dim=-1)
    result = {"device": "cpu", "source": "inter_hierarchy_MN40/hier_proxy_scratch_v8/geometry.py",
              "production_updates": 0, "encoder_forwards": 0, "test_forwards": 0,
              "seed22_initialization": {"norm_min": float(norms.min()),
                                        "norm_median": float(norms.median()),
                                        "norm_max": float(norms.max()),
                                        "depth_median": float(geometry.depth(geometry.to_ball(raw)).median()),
                                        "cap_count": int((norms > 2.3).sum())}, "jacobian": []}
    for radius in (2.07, 2.29998, 2.30002, 2.31, 6.885):
        for hook in (False, True):
            w = torch.tensor([[radius, 0., 0.]], dtype=torch.float64, requires_grad=True)
            output = geometry.to_ball(w, hook=hook)
            radial, = torch.autograd.grad(output[0, 0], w, retain_graph=True)
            angular, = torch.autograd.grad(output[0, 1], w)
            result["jacobian"].append({"raw_norm": radius, "GS": hook,
                                       "mapped_depth": float(geometry.depth(output.detach())),
                                       "radial_gain": float(radial[0, 0]),
                                       "angular_gain": float(angular[0, 1]),
                                       "radial_angular_ratio": float(radial[0, 0] / angular[0, 1])})
    target = here / "cpu_mapping_probe.json"
    target.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
