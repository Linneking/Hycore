"""Validate the recorded CUDA evidence required before a fresh full run.

This checks facts supplied from real artifacts. It never runs a check,
creates evidence, reads remote state, or treats CPU tests as CUDA evidence.
"""
from __future__ import annotations


def _require(condition, message):
    if not condition:
        raise ValueError("PartHier full gate: " + message)


def _positive_int(value):
    return type(value) is int and value > 0


def _completed_run(gate, name, commit):
    run = gate.get(name)
    _require(isinstance(run, dict), name + " evidence is missing")
    _require(run.get("status") == "completed", name + " did not complete")
    _require(run.get("commit") == commit, name + " commit differs")
    config = run.get("config")
    _require(isinstance(config, dict), name + " config is missing")
    _require(config.get("mode") == name, name + " has the wrong mode")
    _require(config.get("activation_checkpoint") is True,
             name + " did not use activation checkpointing")
    _require(config.get("save_on_cpu") is False, name + " used CPU offload")
    _require(type(config.get("backbone_passes_per_step")) is int and
             config["backbone_passes_per_step"] == 4, name + " requires four backbone passes")
    queue = config.get("queue")
    _require(isinstance(queue, dict) and type(queue.get("capacity")) is int and
             queue["capacity"] == 2048, name + " requires Q2048")
    epochs, per_epoch = config.get("epochs"), config.get("steps_per_epoch")
    _require(_positive_int(epochs) and _positive_int(per_epoch),
             name + " update budget is invalid")
    _require(type(run.get("completed_epochs")) is int and run["completed_epochs"] == epochs,
             name + " completed-epoch evidence differs from its budget")
    _require(type(run.get("updates")) is int and run["updates"] == epochs * per_epoch,
             name + " actual update count differs from its completed budget")
    checks = run.get("checks")
    _require(isinstance(checks, dict), name + " acceptance checks are missing")
    for key in ("bn_updates_four", "finite", "replica_consistent", "queue_consistent"):
        _require(checks.get(key) is True, name + " has no passing " + key + " evidence")
    return run


def validate_full_gate(gate, commit):
    """Return None on complete same-source CUDA evidence; otherwise raise.

    ``smoke`` and ``diagnostic`` contain ``status``, ``commit``, the actual
    ``config``, ``completed_epochs``, ``updates``, and four boolean ``checks``.
    Diagnostic ``steps`` has one record per actual update, with ``update``
    (one-based), ``queue_before``, ``whole_count`` and ``finite`` fields.
    """
    _require(isinstance(commit, str) and bool(commit) and commit == commit.strip(),
             "expected source commit is missing")
    _require(isinstance(gate, dict), "evidence must be an object")
    _require(gate.get("source_commit") == commit, "source commit differs")
    identity = gate.get("identity")
    _require(isinstance(identity, dict) and identity.get("commit") == commit,
             "identity commit differs")
    contract = gate.get("cuda_checkpoint_contract")
    _require(isinstance(contract, dict), "CUDA checkpoint contract is missing")
    _require(contract.get("status") == "passed", "CUDA checkpoint contract did not pass")
    _require(contract.get("commit") == commit, "CUDA checkpoint contract commit differs")
    _require(contract.get("device") == "cuda", "checkpoint contract is not CUDA evidence")
    smoke = _completed_run(gate, "smoke", commit)
    _require((smoke["config"]["epochs"], smoke["config"]["steps_per_epoch"], smoke["updates"]) == (2, 2, 4),
             "smoke must be an actual completed 2x2 run")
    diagnostic = _completed_run(gate, "diagnostic", commit)
    _require(smoke["config"].get("checkpoint_policy", "all") == diagnostic["config"].get("checkpoint_policy", "all"),
             "smoke and capacity checkpoint policies differ")
    _require(contract.get("checkpoint_policy", "all") == smoke["config"].get("checkpoint_policy", "all"),
             "CUDA contract and training checkpoint policies differ")
    updates = diagnostic["updates"]
    _require(updates >= 14, "diagnostic needs at least 14 actual updates")
    steps = diagnostic.get("steps")
    _require(isinstance(steps, list) and len(steps) == updates,
             "diagnostic needs one actual step record per update")
    full_queue_whole1024 = False
    for update, row in enumerate(steps, 1):
        _require(isinstance(row, dict), "diagnostic step evidence must be an object")
        _require(type(row.get("update")) is int and row["update"] == update,
                 "diagnostic updates must be consecutive and one-based")
        queue_before, whole_count = row.get("queue_before"), row.get("whole_count")
        _require(type(queue_before) is int and 0 <= queue_before <= 2048,
                 "diagnostic queue-before evidence is invalid")
        _require(type(whole_count) is int and 800 <= whole_count <= 1024,
                 "diagnostic whole-count evidence is invalid")
        _require(row.get("finite") is True, "diagnostic has a nonfinite or unverified step")
        if queue_before == 2048 and whole_count == 1024:
            full_queue_whole1024 = True
    _require(all(row["queue_before"] == 2048 for row in steps[-3:]),
             "last three actual diagnostic updates did not start with Q2048")
    _require(full_queue_whole1024, "no whole1024 update actually ran with full Q2048")
