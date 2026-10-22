import os, sys, numpy as np, pandas as pd
sys.path.insert(0, os.getcwd())
from paiec.data.load import benchmark_dirs, load_pairs
from paiec.eval.protocol import run_pair, BUDGETS
from paiec.predictors.smoothed_mean import make_beta_mean
OUT = "reports/phase4"; os.makedirs(OUT, exist_ok=True); SEEDS = (0, 1, 2)
def uncertainty(inp, prediction=None, labeled=None, context=None):
    if context["labels_remaining"] <= 0: return False
    if context["items_remaining"] <= context["labels_remaining"]: return True
    return abs(prediction - 0.5) <= 0.15
def eager(inp, prediction=None, labeled=None, context=None):
    return context["labels_remaining"] > 0
POL = {"random": None, "uncertainty_0.15": uncertainty, "eager_first31": eager}
rows = []
for b in benchmark_dirs():
    pairs = [p for p in load_pairs(b)[0] if len(p.items) >= 80]
    for name, pol in POL.items():
        for seed in SEEDS:
            for j, p in enumerate(pairs):
                r = run_pair(p, make_beta_mean(2.0), acquire=pol, split_seed=0, order_seed=seed)
                rows.append({"bench": b, "pair": f"{b}:{j}", "seed": seed, "policy": name, "alc": r.alc, "n_acquired": len(r.acquired), **{f"brier_{n}": r.brier[n] for n in BUDGETS}})
    print(b, "done", flush=True)
df = pd.DataFrame(rows); df.to_csv(f"{OUT}/h2_pairlevel_pairs.csv", index=False)
g = df.groupby("policy").agg(alc=("alc", "mean"), labels_spent=("n_acquired", "mean"), **{f"b{n}": (f"brier_{n}", "mean") for n in BUDGETS}).round(4)
pa = df.groupby(["policy", "pair"]).alc.mean().unstack(0); rng = np.random.default_rng(0)
def ci(a, b):
    d = (pa[a] - pa[b]).to_numpy(); bs = np.array([rng.choice(d, len(d)).mean() for _ in range(5000)])
    return f"{d.mean():+.4f} [{np.percentile(bs, 2.5):+.4f}, {np.percentile(bs, 97.5):+.4f}]"
L = ["# Phase 4 (H2), step 1: acquisition policies under the pair-level predictor B2 frozen\n",
     "Expectation: no policy can beat random except by noise, because B2 treats every candidate as exchangeable. This is the empirical check.\n",
     g.to_markdown(), "",
     f"- uncertainty vs random (ALC, paired over 140 pairs): {ci('uncertainty_0.15', 'random')}",
     f"- eager vs random: {ci('eager_first31', 'random')}", "",
     "Per benchmark ALC:", df.groupby(["bench", "policy"]).alc.mean().unstack("policy").round(4).to_markdown(), ""]
open(f"{OUT}/h2_pairlevel.md", "w").write("\n".join(L)); print("\n".join(L))
