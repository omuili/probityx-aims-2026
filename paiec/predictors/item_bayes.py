from __future__ import annotations
import numpy as np
from scipy.special import gammaln

GRID = np.linspace(-8.0, 8.0, 641)
_SIG = lambda x: 1.0 / (1.0 + np.exp(-x))


def _log_prior(m: float) -> np.ndarray:
    """log density on theta induced by p = sigma(theta) ~ Beta(m/2, m/2)."""
    a = m / 2.0; p = _SIG(GRID)
    return (a - 1) * np.log(p) + (a - 1) * np.log(1 - p) + np.log(p) + np.log(1 - p)   # Jacobian sigma'(theta)


def posterior_predictive(labels, deltas, target_delta: float, lam: float, m: float = 2.0) -> float:
    """labels: list of 0/1; deltas: predicted relative difficulty of each labelled item."""
    logp = _log_prior(m).copy()
    th = GRID + 0.0
    for y, d in zip(labels, deltas):
        x = th - lam * d
        logp += -np.logaddexp(0.0, -x) if y else -np.logaddexp(0.0, x)  
    logp -= logp.max(); w = np.exp(logp); w /= w.sum()
    return float(np.clip(np.sum(w * _SIG(th - lam * target_delta)), 1e-9, 1 - 1e-9))   # guard float round-off past 1


def _matching(labeled, subject, benchmark):
    out = []
    for (obs_subject, obs_item), label in (labeled or []):
        if obs_subject == subject and obs_item.get("benchmark_id") == benchmark:
            out.append((int(label), obs_item.get("item_content", "")))
    return out


def make_b4(dm, lam: float, m: float = 2.0, delta_fn=None):
    """dm: fitted DifficultyModel (training benchmarks only). delta is a DIFFICULTY (higher = harder), i.e.
    dm.predict_difficulty = minus the relative success logit. delta_fn overrides it for analysis-only
    ceilings and must follow the same sign convention. (An earlier version passed the success logit
    directly, flipping the sign; those results were superseded.)"""
    rel = delta_fn or dm.predict_difficulty
    def predict(input: list[dict], labeled: list | None = None) -> float:
        subject, item = input
        obs = _matching(labeled, subject, item.get("benchmark_id"))
        ys = [y for y, _ in obs]; ds = [rel(c) for _, c in obs]
        return posterior_predictive(ys, ds, rel(item.get("item_content", "")), lam, m)
    return predict


def make_b4_abs(dm, m: float = 2.0):
    """Prior centred on each item's predicted absolute logit success: theta ~ Logistic(mu_i, 1) where
    mu_i = dm.predict_abs(item). Labels update a shared offset; delta enters through mu differences."""
    def predict(input: list[dict], labeled: list | None = None) -> float:
        subject, item = input
        obs = _matching(labeled, subject, item.get("benchmark_id"))
        mu_t = dm.predict_abs(item.get("item_content", ""))
        ys = [y for y, _ in obs]; ds = [mu_t - dm.predict_abs(c) for _, c in obs]
        logp = _log_prior(m).copy(); th = GRID + mu_t
        for y, d in zip(ys, ds):
            x = th - d; logp += -np.logaddexp(0.0, -x) if y else -np.logaddexp(0.0, x)
        logp -= logp.max(); w = np.exp(logp); w /= w.sum()
        return float(np.clip(np.sum(w * _SIG(th)), 1e-9, 1 - 1e-9))
    return predict
