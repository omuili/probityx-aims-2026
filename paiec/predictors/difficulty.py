from __future__ import annotations
import numpy as np, pandas as pd
from ..features.item_text import item_features, content_hash, FEATURES


def item_table(items: pd.DataFrame, responses: pd.DataFrame, bench: str) -> pd.DataFrame:
    """One row per item with text, response counts and both targets. Uses ONLY this benchmark's data."""
    r = responses[responses["response"].isin([0, 1])].copy(); r["item_id"] = r.item_id.astype(str)
    g = r.groupby("item_id")["response"].agg(["sum", "count"]).rename(columns={"sum": "k", "count": "n"})
    it = items.copy(); it["item_id"] = it.item_id.astype(str); it = it.set_index("item_id")[["content"]]
    tab = g.join(it, how="inner"); tab["content"] = tab.content.fillna("").astype(str)
    tab["p"] = tab.k / tab.n
    tab["abs_logit"] = np.log((tab.k + 0.5) / (tab.n - tab.k + 0.5))
    tab["rel_logit"] = tab.abs_logit - tab.abs_logit.mean()
    tab["bench"] = bench
    return tab.reset_index()


def featurize(contents) -> np.ndarray:
    return np.stack([item_features(c) for c in contents])


class DifficultyModel:
    """Ridge on standardized text features. fit() takes the concatenated item tables of the TRAINING
    benchmarks; alpha is chosen by the caller (nested LOBO). Weighting: each item counts once, so a
    6,306-item benchmark would dominate; we weight items so every training benchmark has equal total
    weight, matching the equal-per-benchmark spirit of the transfer question."""

    def __init__(self, alpha: float = 10.0, equal_benchmark_weight: bool = True):
        self.alpha, self.eqw = alpha, equal_benchmark_weight
        self.mu = self.sd = None; self.w = {}; self.b = {}; self._cache = {}

    def fit(self, train_items: pd.DataFrame):
        X = featurize(train_items.content.tolist())
        w = np.ones(len(train_items))
        if self.eqw:
            counts = train_items.bench.value_counts(); w = train_items.bench.map(lambda b: 1.0 / counts[b]).to_numpy() * len(counts)
        self.mu = np.average(X, axis=0, weights=w); self.sd = np.sqrt(np.average((X - self.mu) ** 2, axis=0, weights=w)) + 1e-8
        Z = (X - self.mu) / self.sd; W = np.diag(w)
        for target in ("abs_logit", "rel_logit"):
            y = train_items[target].to_numpy(float); ybar = np.average(y, weights=w)
            A = Z.T @ W @ Z + self.alpha * np.eye(Z.shape[1]); self.w[target] = np.linalg.solve(A, Z.T @ W @ (y - ybar)); self.b[target] = ybar
        self.train_benchmarks = sorted(train_items.bench.unique().tolist())
        return self

    def _z(self, content: str) -> np.ndarray:
        h = content_hash(content)
        if h not in self._cache:
            self._cache[h] = (item_features(content) - self.mu) / self.sd
        return self._cache[h]

    BOUND = 6.0  

    def predict_abs(self, content: str) -> float: return float(np.clip(self._z(content) @ self.w["abs_logit"] + self.b["abs_logit"], -self.BOUND, self.BOUND))
    def predict_rel(self, content: str) -> float: return float(np.clip(self._z(content) @ self.w["rel_logit"] + self.b["rel_logit"], -self.BOUND, self.BOUND))
    def predict_difficulty(self, content: str) -> float:
        """Relative DIFFICULTY (higher = harder) = minus the relative success logit. This is the sign B4 expects."""
        return -self.predict_rel(content)

    def predict_many(self, contents, target="rel_logit") -> np.ndarray:
        Z = (featurize(contents) - self.mu) / self.sd
        return np.clip(Z @ self.w[target] + self.b[target], -self.BOUND, self.BOUND)

    def coefficients(self, target="rel_logit") -> pd.Series:
        return pd.Series(self.w[target], index=FEATURES).sort_values(key=np.abs, ascending=False)
