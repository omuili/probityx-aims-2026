from __future__ import annotations


def constant_half(input: list[dict], labeled: list | None = None) -> float:
    """B0: the complete example from the competition page."""
    return 0.5


def empirical_mean(input: list[dict], labeled: list | None = None) -> float:
    """B1: official empirical-mean predictor (third_party/paiec_baseline/empirical_mean/model.py).

    Mean of acquired labels whose subject dict equals the target subject dict and whose item has
    the same benchmark_id; unsmoothed; 0.5 when no matching labels.
    """
    subject, item = input
    if not labeled:
        return 0.5
    benchmark = item.get("benchmark_id")
    if not isinstance(benchmark, str) or not benchmark:
        raise ValueError("The mean predictor requires an anonymous item benchmark_id.")
    successes = count = 0
    for (obs_subject, obs_item), label in labeled:
        if obs_subject != subject:
            continue
        if obs_item.get("benchmark_id") != benchmark:
            continue
        if label not in (0, 1):
            raise ValueError("Acquired responses must be binary (0 or 1).")
        successes += int(label); count += 1
    return float(successes / count) if count else 0.5
