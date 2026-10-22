from __future__ import annotations
import os, sys, time
import numpy as np, pandas as pd
sys.path.insert(0, os.getcwd())
from paiec.data.load import benchmark_dirs, load_pairs
from paiec.eval.protocol import run_pair, summarize, BUDGETS
from paiec.predictors.baselines import constant_half, empirical_mean

SEEDS = (0, 1, 2)
rows = []
for b in benchmark_dirs():
    pairs, meta = load_pairs(b)
    for name, fn in [("B0_constant_half", constant_half), ("B1_empirical_mean", empirical_mean)]:
        for seed in SEEDS:
            t0 = time.time()
            res = [r for r in (run_pair(p, fn, split_seed=0, order_seed=seed) for p in pairs) if r is not None]
            if not res:
                continue
            s = summarize(res); s.update(bench=b, method=name, seed=seed, secs=round(time.time() - t0, 1)); rows.append(s)
df = pd.DataFrame(rows); os.makedirs("reports", exist_ok=True); df.to_csv("reports/baselines.csv", index=False)
g = df.groupby(["bench", "method"]).agg(n_pairs=("n_pairs", "first"), alc_mean=("alc", "mean"), alc_sd=("alc", "std"),
                                       **{f"b{n}": (f"brier_{n}", "mean") for n in BUDGETS}).round(4)
print(g.to_string())
pooled = df.assign(w=df.n_pairs).groupby(["method", "seed"]).apply(lambda x: np.average(x.alc, weights=x.w), include_groups=False)
print("\npooled equal-per-pair ALC (mean over seeds):\n", pooled.groupby("method").mean().round(4).to_string())
