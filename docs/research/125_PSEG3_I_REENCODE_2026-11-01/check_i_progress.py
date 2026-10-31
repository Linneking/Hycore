"""Read-only Experiment I training status, best score, BN and gradient routing."""

import argparse
import json
from pathlib import Path


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def rows(path):
    if not path.exists():
        return []
    result = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            result.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    root = args.root
    queue = load(root / "queue.json") if (root / "queue.json").exists() else {}
    print("Queue:", queue.get("state", "not submitted"))
    if queue.get("state") == "failed":
        print("Failure:", queue.get("error", "see launcher log"))
    smoke = root / "smoke_validation.json"
    if smoke.exists():
        checks = load(smoke)
        print("Real-data smoke:", "passed" if checks.get("all_checks_pass") else "failed")
        print("Smoke gates:", {name: value for name, value in checks.items()
                              if name != "all_checks_pass"})
    main_dir = root / "main"
    if not (main_dir / "status.json").exists():
        return
    status = load(main_dir / "status.json")
    history = rows(main_dir / "metrics.jsonl")
    print("Training:", status.get("state"), "epoch", status.get("epoch"),
          "step", status.get("step", "-"), "updates", status.get("updates"))
    print("Updated:", status.get("at"))
    if history:
        latest = history[-1]
        test = latest.get("test", latest.get("val", {}))
        print("Latest complete e%d: Instance %.3f%% / Class %.3f%%" %
              (latest["epoch"], 100 * test["instance_miou"], 100 * test["class_miou"]))
        print("Loss: %.5f; CE: %.5f; R: %.5f" %
              (latest["train_loss"], latest["train_ce"], latest["train_reg"]))
        print("BN count range:", min(latest["bn_counts"].values()),
              max(latest["bn_counts"].values()))
        print("Peak allocated memory GiB: %.2f" % (latest["peak_memory_bytes"] / 2**30))
    best = main_dir / "best_selection.json"
    if best.exists():
        selected = load(best)
        score = selected["metrics"]
        print("Best e%d: Instance %.3f%% / same-epoch Class %.3f%%" %
              (selected["selected_epoch"], 100 * score["instance_miou"],
               100 * score["class_miou"]))
    gradients = [row for row in rows(main_dir / "geometry.jsonl") if row.get("gradients")]
    if gradients:
        groups = gradients[-1]["gradients"]
        for term in ("ce", "contr", "hier"):
            values = groups.get(term, {}).get("parameter_groups", {})
            print("%s gradients:" % term,
                  {name: values.get(name, {}).get("status")
                   for name in ("shared", "euclidean9", "m2", "m3")})
    print("Failure file:", (main_dir / "failure.json").exists())


if __name__ == "__main__":
    main()
