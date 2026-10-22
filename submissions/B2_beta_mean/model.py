"""Beta-Binomial smoothed pair success rate, prior strength M (uninformative prior mean 0.5)."""
M = 2.0


def predict(input: list[dict], labeled: list | None = None) -> float:
    subject, item = input
    benchmark = item.get("benchmark_id")
    k = n = 0
    for (obs_subject, obs_item), label in (labeled or []):
        if obs_subject != subject or obs_item.get("benchmark_id") != benchmark:
            continue
        k += int(label); n += 1
    return float((M / 2 + k) / (M + n))
