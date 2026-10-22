"""Competition metrics, implemented from the published definitions.

Brier: mean squared difference between predicted probability and observed 0/1 outcome, over all
recorded responses of a pair's evaluation items. ALC: trapezoid area under Brier vs
x = log2(n+1) over budgets (0,1,3,7,15,31), normalised by the axis width 5, which gives weights
0.1, 0.2, 0.2, 0.2, 0.2, 0.1. ECE: mean absolute gap between mean predicted probability and
observed success rate over 10 equal-width bins, weighted by response count.
"""
from __future__ import annotations
import numpy as np

BUDGETS = (0, 1, 3, 7, 15, 31)
ALC_WEIGHTS = {0: 0.1, 1: 0.2, 3: 0.2, 7: 0.2, 15: 0.2, 31: 0.1}


def brier(p: np.ndarray, y: np.ndarray) -> float:
    p = np.asarray(p, float); y = np.asarray(y, float)
    if p.size == 0:
        return float("nan")
    return float(np.mean((p - y) ** 2))


def ece(p: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    p = np.asarray(p, float); y = np.asarray(y, float)
    if p.size == 0:
        return float("nan")
    idx = np.minimum((p * bins).astype(int), bins - 1)
    total = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            total += m.sum() / p.size * abs(p[m].mean() - y[m].mean())
    return float(total)


def alc(brier_by_budget: dict[int, float]) -> float:
    """Trapezoid area under the learning curve on the log2(n+1) axis, normalised to width 5.

    Equivalent to the published weights 0.1/0.2/0.2/0.2/0.2/0.1; computed from the trapezoid rule
    so the equivalence is tested rather than assumed.
    """
    xs = [np.log2(n + 1) for n in BUDGETS]
    ys = [brier_by_budget[n] for n in BUDGETS]
    area = sum((xs[k + 1] - xs[k]) * (ys[k] + ys[k + 1]) / 2 for k in range(len(BUDGETS) - 1))
    return float(area / (xs[-1] - xs[0]))


def alc_weighted(brier_by_budget: dict[int, float]) -> float:
    return float(sum(ALC_WEIGHTS[n] * brier_by_budget[n] for n in BUDGETS))
