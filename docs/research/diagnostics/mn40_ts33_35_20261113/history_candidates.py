"""Select detached historical candidates without consuming training RNG streams."""
from collections import Counter
import random

SELECTED_HISTORY_ARMS = {"TS48": "class_pairs16x2", "TS49": "uniform32"}


def select_history(labels, mode, update, seed=4780, limit=32):
    """Indices refer to newest-first history; each object occurs once per epoch."""
    rng = random.Random(seed + 700001 + update * 10007)
    size = min(limit, len(labels))
    population = list(range(len(labels)))
    if mode == "uniform32":
        return rng.sample(population, size)
    if mode != "class_pairs16x2":
        raise ValueError(mode)
    groups = {}
    for index, label in enumerate(labels):
        groups.setdefault(int(label), []).append(index)
    eligible = sorted(label for label, indices in groups.items() if len(indices) >= 2)
    classes = rng.sample(eligible, min(limit // 2, len(eligible)))
    chosen = [index for label in classes for index in rng.sample(groups[label], 2)]
    occupied = set(chosen)
    chosen.extend(rng.sample([i for i in population if i not in occupied], size - len(chosen)))
    rng.shuffle(chosen)
    assert len(chosen) == len(set(chosen)) == size
    return chosen


def candidate_statistics(current, historical, indices, ages):
    chosen_labels = [int(historical[index]) for index in indices]
    current = [int(label) for label in current]
    counts = Counter(current + chosen_labels)
    chosen_ages = [ages[index] for index in indices]
    return dict(history_objects=len(historical), selected_history_objects=len(indices),
                history_eligible_classes=sum(n >= 2 for n in Counter(historical).values()),
                selected_history_classes=len(set(chosen_labels)),
                pool_classes=len(counts), pool_paired_classes=sum(n >= 2 for n in counts.values()),
                current_anchors_with_peer=sum(counts[label] >= 2 for label in current),
                current_anchors_with_two_peers=sum(counts[label] >= 3 for label in current),
                selected_age_counts={str(age): chosen_ages.count(age) for age in (1, 2, 3)})
