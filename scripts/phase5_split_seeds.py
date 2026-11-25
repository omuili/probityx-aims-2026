"""Split-seed robustness of the main result: B0, B1, B2 fixed m=2 and B2 strict LOBO under split seeds 0, 1, 2.

Identical to phase2_b2.py except that the 50/50 acquisition/evaluation split seed varies. For each split
seed, m is re-selected by LOBO on the three training benchmarks; candidate-order seeds 0, 1, 2 as before.
Split seed 0 reproduces the reported figures.
"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.getcwd())
from paiec.data.load import benchmark_dirs, load_pairs
from paiec.eval.protocol import run_pair
from paiec.predictors.smoothed_mean import make_beta_mean

SPLITS = (0, 1, 2); SEEDS = (0, 1, 2)
GRID = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0, 32.0, 64.0, float("inf")]
OUT = "reports/phase5"; os.makedirs(OUT, exist_ok=True)
B = benchmark_dirs(); P = {b: [p for p in load_pairs(b)[0] if len(p.items) >= 80] for b in B}


def predictor(m):
    return (lambda inp, labeled=None: 0.5) if m == float("inf") else make_beta_mean(m)


rows = []
for split in SPLITS:
    for m in GRID:
        fn = predictor(m)
        for b in B:
            for j, p in enumerate(P[b]):
                for seed in SEEDS:
                    rows.append({"split": split, "bench": b, "pair": f"{b}:{j}", "m": m, "seed": seed,
                                 "alc": run_pair(p, fn, split_seed=split, order_seed=seed).alc})
    print("split", split, "done", flush=True)
df = pd.DataFrame(rows); df.to_csv(f"{OUT}/split_seeds_pairs.csv", index=False)

pm = df.groupby(["split", "bench", "pair", "m"]).alc.mean().reset_index()
out, sel = [], []
for split in SPLITS:
    s = pm[pm.split == split]
    mstar = {b: float(s[s.bench != b].groupby("m").alc.mean().idxmin()) for b in B}
    sel.append({"split": split, **{f"m*_{b}": mstar[b] for b in B}})
    pa = lambda f: pd.concat([s[(s.bench == b) & (s.m == f(b))] for b in B]).set_index("pair").alc
    b0, b1, b2f, b2l = pa(lambda b: float("inf")), pa(lambda b: 0.0), pa(lambda b: 2.0), pa(lambda b: mstar[b])
    rng = np.random.default_rng(0); d = (b2l - b1).to_numpy()
    bs = np.array([rng.choice(d, len(d)).mean() for _ in range(5000)])
    out.append({"split": split, "B0": b0.mean(), "B1": b1.mean(), "B2_fixed_m2": b2f.mean(), "B2_LOBO": b2l.mean(),
                "LOBO_minus_B1": d.mean(), "ci_lo": np.percentile(bs, 2.5), "ci_hi": np.percentile(bs, 97.5),
                "pairs_improved": int((d < 0).sum())})
out = pd.DataFrame(out); sel = pd.DataFrame(sel)
L = ["# Phase 5: split-seed robustness of the main result (candidate-order seeds 0, 1, 2 within each split)\n",
     "## Pooled equal-per-pair ALC, 140 pairs\n", out.round(4).to_markdown(index=False), "",
     f"Across split seeds: B2 LOBO {out.B2_LOBO.min():.4f} to {out.B2_LOBO.max():.4f}; B1 {out.B1.min():.4f} to {out.B1.max():.4f}; "
     f"LOBO minus B1 {out.LOBO_minus_B1.min():+.4f} to {out.LOBO_minus_B1.max():+.4f}.", "",
     "## LOBO-selected m per split seed\n", sel.to_markdown(index=False), ""]
open(f"{OUT}/split_seeds.md", "w").write("\n".join(L)); print("\n".join(L))
