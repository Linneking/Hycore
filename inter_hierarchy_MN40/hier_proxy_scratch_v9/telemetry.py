"""Native c1 telemetry; HIER view statistics remain a separate c.1 domain."""
from ..hier_proxy_scratch_v5.telemetry import EpochTelemetry as PreviousTelemetry

class EpochTelemetry(PreviousTelemetry):
    def summary(self):
        result = super().summary()
        for part in ('whole', 'part'):
            values = result['geometry'][part]
            values.update(c=1., depth_unit='native c1 origin hyperbolic distance',
                          live_tangent_cap=None, live_depth_cap=None, inverse_metric_hook_applied=False)
            values['inverse_metric_factor_is_counterfactual_not_applied'] = True
        result['geometry']['hier_view_domain'] = {
            'c': .1, 'target_whole_depth': 4.6, 'hook_applied': True,
            'source': 'hier_reported_per_batch.whole.* and proxy.*; not native whole statistics'}
        return result
