"""C.1 telemetry separates base capped embeddings and HIER fixed-shell view."""
from ..hier_proxy_scratch_v8.telemetry import EpochTelemetry as PreviousTelemetry


class EpochTelemetry(PreviousTelemetry):
    def __init__(self, train_ids, train_labels, num_classes=40, hier_enabled=False):
        super().__init__(train_ids, train_labels, num_classes=num_classes)
        self.hier_enabled = bool(hier_enabled)

    def summary(self):
        result = super().summary()
        for part in ("whole", "part"):
            result["geometry"][part].update(
                space="base_hycore_c01_capped", inverse_metric_hook_applied=True,
                gradient_scope="base objectives once; HIER uses pre-GS branch")
        result["geometry"]["hier_view_domain"] = {
            "enabled": self.hier_enabled, "c": .1,
            "target_whole_depth": 4.6 if self.hier_enabled else None,
            "hook_applied": self.hier_enabled,
            "source": "cap-after/pre-base-GS whole; separately normalized" if self.hier_enabled else None,
            "zero_construct_when_disabled": not self.hier_enabled,
        }
        return result
