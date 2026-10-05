"""Strict complete-state V6-H20 to balanced B0 transition, independent of torch."""
from __future__ import annotations
import math

SOURCE_VERSION = "hycore-hier-v6-h20-selfk300-1"
CHECKPOINT_VERSION = "hycore-v6-balanced-b0-e20-fork-1"
OBJECTIVE_CHANGES = {
    "lambda_hier_after_warmup": 0.,
    "proxy_optimizer": None,
    "schedule": "original300-epoch model cosine continued from historicale20; no proxy",
    "audit_epochs": [],
}
DERIVED_METADATA = {
    "arm": "V6 balanced64 B0",
    "HIER": False,
    "proxy_updates": False,
    "shared_historical_prefix_epochs": 20,
    "continuation_new_warmup_epochs": 0,
}

def derive_b0_config(reference):
    return {**reference, **OBJECTIVE_CHANGES, **DERIVED_METADATA}

def validate_checkpoint_contract(saved, config, reference, split_sha, train_ids, val_ids):
    if config != derive_b0_config(reference):
        raise RuntimeError("Target configuration differs from explicit B0 objective whitelist")
    if reference.get("epochs") != 300 or reference.get("steps_per_epoch") != 200 or reference.get("warmup_epochs") != 20:
        raise RuntimeError("Source protocol must retain historical300/200/20")
    if saved.get("model_selection_only") or saved.get("diagnostic_only") or saved.get("continuation_smoke_not_resumable"):
        raise RuntimeError("Selection/diagnostic checkpoint cannot resume production")
    version = saved.get("format")
    completed = int(saved.get("completed_epochs", -1))
    if version == SOURCE_VERSION:
        if saved.get("training_config") != reference or completed != 20:
            raise RuntimeError("Source must be exact complete V6-H20 epoch20 configuration")
    elif version == CHECKPOINT_VERSION:
        if saved.get("training_config") != config or not 20 <= completed <= 300:
            raise RuntimeError("B0 resume configuration or epoch mismatch")
    else:
        raise RuntimeError("Unsupported complete-checkpoint format")
    if saved.get("split_sha256") != split_sha:
        raise RuntimeError("Split checksum mismatch")
    if list(saved.get("train_ids", [])) != train_ids or list(saved.get("validation_ids", [])) != val_ids:
        raise RuntimeError("Train/validation object identity mismatch")
    required = {"net","optimizer","scheduler","rank_states","best","best_net",
                "initialization","initialized_model_sha256","commit"}
    if not required.issubset(saved):
        raise RuntimeError("Missing complete model/optimizer/scheduler/rank/selection state")
    if version == SOURCE_VERSION and not {"proxy","proxy_optimizer","proxy_scheduler"}.issubset(saved):
        raise RuntimeError("Historical source checkpoint is incomplete")
    states = saved["rank_states"]
    if len(states) != 2 or [state.get("rank") for state in states] != [0,1]:
        raise RuntimeError("Exactly ordered rank0/rank1 complete states required")
    for state in states:
        if not {"BN_buffers","rng","sampler"}.issubset(state):
            raise RuntimeError("Missing rank-specific buffer/RNG/sampler state")
        if not {"python","numpy","torch","cuda"}.issubset(state["rng"]):
            raise RuntimeError("Incomplete rank RNG state")
        if state["sampler"].get("epoch") != completed - 1:
            raise RuntimeError("Sampler completed epoch mismatch")
    scheduler = saved["scheduler"]
    if scheduler.get("last_epoch") != completed or scheduler.get("T_max") != 300 or scheduler.get("eta_min") != .005:
        raise RuntimeError("Cosine must retain completed epochs and original300-period")
    expected_lr = .005 + (.1-.005)*(1+math.cos(math.pi*completed/300))/2
    groups = saved["optimizer"].get("param_groups", [])
    if not groups or not saved["optimizer"].get("state"):
        raise RuntimeError("Complete SGD parameter groups and momentum state required")
    for group in groups:
        if abs(group.get("lr", -1)-expected_lr) > 1e-12 or group.get("momentum") != .9 or group.get("weight_decay") != 2e-4:
            raise RuntimeError("Optimizer LR/momentum/weight-decay mismatch")
    if not isinstance(saved["best"], dict) or not 1 <= saved["best"].get("epoch",0) <= completed or not saved["best_net"]:
        raise RuntimeError("Embedded validation-selected model/history missing")
    return {"source_format":version,"completed_epochs":completed,
            "allowed_configuration_changes":sorted(OBJECTIVE_CHANGES),
            "derived_metadata":sorted(DERIVED_METADATA)}
