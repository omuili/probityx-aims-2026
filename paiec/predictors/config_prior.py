from __future__ import annotations
import numpy as np, pandas as pd

LEVELS = ("exact", "name", "provider", "harness", "global")


def _key(sd: dict, fields) -> tuple:
    return tuple(str(sd.get(f, "")).strip().lower() for f in fields)


def _sigmoid(x): return 1.0 / (1.0 + np.exp(-x))


class ConfigPrior:
    def __init__(self, shrink_c: float = 20.0):
        self.c = shrink_c
        self.tables = {lvl: {} for lvl in LEVELS[:-1]}   # level -> key -> [sum_n_a, sum_n, n_pairs]
        self.G = 0.0
        self.fitted_benchmarks = []

    @staticmethod
    def _subject_rows(subjects: pd.DataFrame, responses: pd.DataFrame) -> pd.DataFrame:
        r = responses[responses["response"].isin([0, 1])].copy(); r["subject_id"] = r.subject_id.astype(str)
        g = r.groupby("subject_id")["response"].agg(["sum", "count"]).rename(columns={"sum": "k", "count": "n"})
        g["logit"] = np.log((g.k + 0.5) / (g.n - g.k + 0.5))
        S = subjects.copy(); S["subject_id"] = S.subject_id.astype(str)
        return g.join(S.set_index("subject_id"), how="inner")

    def fit(self, train: list[tuple[str, pd.DataFrame, pd.DataFrame]]):
        """train: list of (benchmark_name, subjects_df, responses_df) for TRAINING benchmarks only."""
        from ..data.load import subject_dict
        mus = []
        for bench, subjects, responses in train:
            rows = self._subject_rows(subjects, responses)
            if rows.empty: continue
            mu_b = rows.logit.mean(); mus.append(mu_b); self.fitted_benchmarks.append(bench)
            for _, row in rows.iterrows():
                sd = subject_dict(row); a = row.logit - mu_b; n = float(row.n)
                for lvl, fields in [("exact", ("normalized_name", "harness", "harness_version", "reasoning_effort")),
                                    ("name", ("normalized_name",)), ("provider", ("provider",)), ("harness", ("harness",))]:
                    k = _key(sd, fields)
                    if all(x == "" for x in k): continue      
                    t = self.tables[lvl].setdefault(k, [0.0, 0.0, 0])
                    t[0] += n * a; t[1] += n; t[2] += 1
        self.G = float(np.mean(mus)) if mus else 0.0
        return self

    def prior(self, subject: dict) -> tuple[float, str, int]:
        """Return (prior mean p0, back-off level used, number of training pairs supporting it)."""
        for lvl, fields in [("exact", ("normalized_name", "harness", "harness_version", "reasoning_effort")),
                            ("name", ("normalized_name",)), ("provider", ("provider",)), ("harness", ("harness",))]:
            t = self.tables[lvl].get(_key(subject, fields))
            if t is not None:
                a_hat = t[0] / (t[1] + self.c)
                return float(_sigmoid(self.G + a_hat)), lvl, int(t[2])
        return float(_sigmoid(self.G)), "global", 0


def make_b3(prior: ConfigPrior, m_informative: float, m_backoff: float, global_only: bool = False):
    """Beta-Binomial predictor with configuration-aware prior mean. global_only=True is the ablation
    that uses sigmoid(G) for every subject (tests whether merely lowering the prior mean explains gains)."""
    def predict(input: list[dict], labeled: list | None = None) -> float:
        subject, item = input
        if global_only:
            p0, lvl = float(_sigmoid(prior.G)), "global"
        else:
            p0, lvl, _ = prior.prior(subject)
        m = m_informative if lvl in ("exact", "name") else m_backoff
        benchmark = item.get("benchmark_id"); k = n = 0
        for (obs_subject, obs_item), label in (labeled or []):
            if obs_subject != subject or obs_item.get("benchmark_id") != benchmark: continue
            k += int(label); n += 1
        return float((m * p0 + k) / (m + n))
    return predict
