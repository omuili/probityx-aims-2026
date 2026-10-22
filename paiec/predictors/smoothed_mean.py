from __future__ import annotations


def make_beta_mean(m: float):
    def predict(input: list[dict], labeled: list | None = None) -> float:
        subject, item = input
        benchmark = item.get("benchmark_id")
        k = n = 0
        for (obs_subject, obs_item), label in (labeled or []):
            if obs_subject != subject or obs_item.get("benchmark_id") != benchmark:
                continue
            k += int(label); n += 1
        if m <= 0 and n == 0:
            return 0.5
        return float((m / 2 + k) / (m + n))
    predict.__name__ = f"beta_mean_m{m:g}"
    return predict
