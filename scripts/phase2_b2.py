from __future__ import annotations
import os, sys, time
import numpy as np, pandas as pd
sys.path.insert(0, os.getcwd())
from paiec.data.load import benchmark_dirs, load_pairs
from paiec.eval.protocol import run_pair, BUDGETS
from paiec.predictors.smoothed_mean import make_beta_mean

SEEDS = (0, 1, 2)
GRID = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0, 32.0, 64.0, float("inf")]
OUT = "reports/phase2_b2"; os.makedirs(OUT, exist_ok=True)


def predictor(m):
    if m == float("inf"):
        return lambda inp, labeled=None: 0.5
    return make_beta_mean(m)


rows = []
t0 = time.time()
for b in benchmark_dirs():
    pairs, _ = load_pairs(b)
    pairs = [p for p in pairs if len(p.items) >= 80]
    for m in GRID:
        fn = predictor(m)
        for seed in SEEDS:
            for j, p in enumerate(pairs):
                r = run_pair(p, fn, split_seed=0, order_seed=seed)
                row = {"bench": b, "pair": f"{b}:{j}", "subject": p.subject["normalized_name"], "harness": p.subject["harness"],
                       "seed": seed, "m": m, "alc": r.alc, "n_eval_items": r.n_eval_items}
                row.update({f"brier_{n}": r.brier[n] for n in BUDGETS}); row.update({f"ece_{n}": r.ece[n] for n in BUDGETS})
                rows.append(row)
    print(f"{b}: {len(pairs)} pairs done, {time.time()-t0:.0f}s", flush=True)
df = pd.DataFrame(rows); df.to_csv(f"{OUT}/pairs.csv", index=False)


pm = df.groupby(["bench", "pair", "m"]).alc.mean().reset_index()

sel = []
for b in benchmark_dirs():
    train = pm[pm.bench != b].groupby("m").alc.mean()          
    m_star = float(train.idxmin())
    sel.append({"held_out": b, "m_star": m_star, "train_pooled_alc_at_m_star": float(train.min()),
                "n_train_pairs": int(pm[pm.bench != b].pair.nunique())})
sel = pd.DataFrame(sel); sel.to_csv(f"{OUT}/selection.csv", index=False)
mstar = dict(zip(sel.held_out, sel.m_star))


def pick(m_of_bench):
    parts = [df[(df.bench == b) & (df.m == m_of_bench(b))] for b in benchmark_dirs()]
    return pd.concat(parts)
B2 = pick(lambda b: mstar[b]).assign(method="B2_beta_mean_LOBO")
B1 = pick(lambda b: 0.0).assign(method="B1_empirical_mean")
B0 = pick(lambda b: float("inf")).assign(method="B0_constant_half")
allm = pd.concat([B0, B1, B2])

def table(d):
    g = d.groupby("method")
    out = g.agg(n_pairs=("pair", "nunique"), alc=("alc", "mean"), **{f"b{n}": (f"brier_{n}", "mean") for n in BUDGETS},
                **{f"ece{n}": (f"ece_{n}", "mean") for n in BUDGETS})
    return out

L = ["# Phase 2, B2: Beta-Binomial shrinkage (m selected by LOBO on pooled equal-per-pair ALC)\n",
     f"Seeds {SEEDS}; split seed 0; grid m = {GRID}. m = 0 is B1, m = inf is B0. All numbers are equal-weight over pairs, then mean over seeds.\n",
     "## LOBO selection of m\n", sel.to_markdown(index=False), "",
     "## In-sample m curve (pooled over all 140 pairs; for inspection only, not used for selection)\n",
     pm.groupby("m").alc.mean().round(4).to_frame("pooled_alc").T.to_markdown(), "",
     "## Pooled equal-per-pair results (selection statistic)\n", table(allm).round(4).to_markdown(), ""]
for b in benchmark_dirs():
    L += [f"## Fold: {b}  (m* = {mstar[b]:g}; {allm[allm.bench==b].pair.nunique()} pairs)\n", table(allm[allm.bench == b]).round(4).to_markdown(), ""]


p2 = B2.groupby("pair").alc.mean(); p1 = B1.groupby("pair").alc.mean(); d = (p2 - p1).rename("delta_alc").to_frame()
d["bench"] = [x.split(":")[0] for x in d.index]
L += ["## Pair-level change, B2 minus B1 (ALC; negative = B2 better)\n",
      f"- pairs improved: {(d.delta_alc < 0).sum()} of {len(d)}; worsened: {(d.delta_alc > 0).sum()}",
      f"- mean {d.delta_alc.mean():.4f}, median {d.delta_alc.median():.4f}, min {d.delta_alc.min():.4f}, max {d.delta_alc.max():.4f}",
      d.groupby("bench").delta_alc.agg(["count", "mean", "median", "min", "max", lambda x: (x < 0).mean()]).rename(columns={"<lambda_0>": "frac_improved"}).round(4).to_markdown(), ""]

rng = np.random.default_rng(0); deltas = d.delta_alc.to_numpy()
bs = np.array([rng.choice(deltas, len(deltas)).mean() for _ in range(5000)])
L += [f"- paired bootstrap over pairs (5000 resamples) of the pooled ALC gain: mean {deltas.mean():.4f}, 95% CI [{np.percentile(bs,2.5):.4f}, {np.percentile(bs,97.5):.4f}]", ""]

L += ["## Interpretation notes\n",
      "- Budget-1 Brier is the term B2 exists to fix: B1 predicts 0 or 1 from a single label.",
      "- ECE per budget is reported above for B0, B1, B2; lower is better; it is a diagnostic only.",
      "- Formative Codabench result for B1 (submission 943338, ALC 0.226339) is a runtime/sanity reference, not a target; see reports/codabench/B1_submission_943338.md.", ""]
open(f"{OUT}/summary.md", "w").write("\n".join(L)); print("\n".join(L))


import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
xs = [np.log2(n + 1) for n in BUDGETS]
fig, axes = plt.subplots(1, 5, figsize=(18, 3.6), sharey=True)
for ax, (title, d_) in zip(axes, [("pooled (140 pairs)", allm)] + [(b, allm[allm.bench == b]) for b in benchmark_dirs()]):
    t = table(d_)
    for meth, color in [("B0_constant_half", "0.5"), ("B1_empirical_mean", "C1"), ("B2_beta_mean_LOBO", "C0")]:
        ax.plot(xs, [t.loc[meth, f"b{n}"] for n in BUDGETS], marker="o", color=color, label=f"{meth} (ALC {t.loc[meth,'alc']:.3f})")
    ax.set_xticks(xs); ax.set_xticklabels(BUDGETS); ax.set_title(title, fontsize=10); ax.grid(alpha=.3); ax.set_xlabel("label budget")
    ax.legend(fontsize=7)
axes[0].set_ylabel("mean Brier over pairs")
plt.tight_layout(); plt.savefig(f"{OUT}/curves.png", dpi=130); print("saved", f"{OUT}/curves.png")
