"""Keep prior sampling/role telemetry, using correct c=.1 physical units."""
import math
import numpy as np
from ..hier_proxy_scratch_v5.telemetry import EpochTelemetry as PreviousTelemetry, _array, _distribution
from .geometry import C


class EpochTelemetry(PreviousTelemetry):
    def _observe_geometry(self, part, embedding):
        points = _array(embedding, np.float64)
        radius = np.linalg.norm(points, axis=-1)
        q = math.sqrt(C) * radius
        if not np.isfinite(points).all() or np.any(q >= 1):
            raise ValueError("Invalid c=.1 ball coordinates")
        h = 2 / math.sqrt(C) * np.arctanh(q)
        if np.any(h > 4.60005):
            raise ValueError("Final shared representation violates HIER cap")
        self.geometry_values[part]["radius"].extend(radius.tolist())
        self.geometry_values[part]["depth"].extend(h.tolist())
        self.geometry_values[part]["inverse_metric_factor"].extend(((1 - C * radius**2)**2 / 4).tolist())
        self.geometry_counts[part]["positions"] += len(radius)
        self.geometry_counts[part]["at_forward_depth_cap"] += int((h >= 4.5999).sum())

    def summary(self):
        result = super().summary()
        for part, observations in self.geometry_values.items():
            counts = self.geometry_counts[part]
            result["geometry"][part] = {key: _distribution(value) for key, value in observations.items()}
            result["geometry"][part].update(c=C, depth_unit="origin hyperbolic distance",
                positions=counts["positions"], live_tangent_cap=2.3, live_depth_cap=4.6,
                at_forward_depth_cap_count=counts["at_forward_depth_cap"],
                at_forward_depth_cap_fraction=counts["at_forward_depth_cap"] / max(1, counts["positions"]))
        return result
