from __future__ import annotations
import os, sys, time, itertools
import numpy as np, pandas as pd
sys.path.insert(0, os.getcwd())
from paiec.data.load import benchmark_dirs, load_pairs, load_tables
from paiec.eval.protocol import run_pair, BUDGETS
from paiec.predictors.baselines import constant_half, empirical_mean
from paiec.predictors.smoothed_mean import make_beta_mean
from paiec.predictors.config_prior import ConfigPrior, make_b3

SEEDS = (0, 1, 2); OUT = "reports/phase2_b3"; os.makedirs(OUT, exist_ok=True)
GRID = list(itertools.product([1.0, 2.0, 3.0, 4.0, 6.0], [1.0, 2.0, 3.0, 4.0, 6.0], [0.0, 20.0, 100.0]))  # (m_inf, m_back, c)
BENCH = benchmark_dirs()
T = {b: load_tables(b) for b in BENCH}
P = {b: [p for p in load_pairs(b)[0] if len(p.items) >= 80] for b in BENCH}
RATE = {f"{b}:{j}": float(np.concatenate([np.asarray(v, float) for v in p.responses.values()]).mean()) for b in BENCH for j, p in enumerate(P[b])}

def fit_prior(train_benches, c):
    return ConfigPrior(shrink_c=c).fit([(b, T[b]["subjects"], T[b]["response"]) for b in train_benches])

def evaluate(bench, fn, seeds=SEEDS, tag=None):
    rows = []
    for seed in seeds:
        for j, p in enumerate(P[bench]):
            r = run_pair(p, fn, split_seed=0, order_seed=seed)
            row = {"bench": bench, "pair": f"{bench}:{j}", "seed": seed, "alc": r.alc, "rate": RATE[f"{bench}:{j}"]}
            row.update({f"brier_{n}": r.brier[n] for n in BUDGETS}); row.update({f"ece_{n}": r.ece[n] for n in BUDGETS})
            if tag: row.update(tag)
            rows.append(row)
    return pd.DataFrame(rows)

t0 = time.time(); sel_rows, all_rows, cov_rows = [], [], []
for held in BENCH:
    train = [b for b in BENCH if b != held]
    best, best_alc = None, np.inf
    for (mi, mb, c) in GRID:
        parts = []
        for t in train:
            inner = [b for b in train if b != t]
            fn = make_b3(fit_prior(inner, c), mi, mb)
            parts.append(evaluate(t, fn, seeds=(0,)))
        alc = pd.concat(parts).alc.mean()
        if alc < best_alc: best, best_alc = (mi, mb, c), alc
    mi, mb, c = best
    sel_rows.append({"held_out": held, "m_informative": mi, "m_backoff": mb, "shrink_c": c, "nested_pooled_alc": best_alc,
                     "n_nested_pairs": sum(len(P[t]) for t in train)})
    prior = fit_prior(train, c)
    assert held not in prior.fitted_benchmarks
    for j, p in enumerate(P[held]):
        p0, lvl, support = prior.prior(p.subject)
        cov_rows.append({"bench": held, "pair": f"{held}:{j}", "level": lvl, "support_pairs": support, "prior_mean": p0, "rate": RATE[f"{held}:{j}"],
                         "subject": p.subject["normalized_name"], "harness": p.subject["harness"]})
    for name, fn in [("B0_constant_half", constant_half), ("B1_empirical_mean", empirical_mean), ("B2_frozen_m2", make_beta_mean(2.0)),
                     ("B3_config_prior", make_b3(prior, mi, mb)), ("B3_global_prior_ablation", make_b3(prior, mi, mb, global_only=True))]:
        all_rows.append(evaluate(held, fn, tag={"method": name}))
    print(f"{held}: selected m_inf={mi} m_back={mb} c={c} (nested ALC {best_alc:.4f}); {time.time()-t0:.0f}s", flush=True)

df = pd.concat(all_rows); df.to_csv(f"{OUT}/pairs.csv", index=False)
sel = pd.DataFrame(sel_rows); sel.to_csv(f"{OUT}/selection.csv", index=False)
cov = pd.DataFrame(cov_rows); cov.to_csv(f"{OUT}/coverage.csv", index=False)
ORDER = ["B0_constant_half", "B1_empirical_mean", "B2_frozen_m2", "B3_config_prior", "B3_global_prior_ablation"]

def table(d):
    g = d.groupby("method").agg(n_pairs=("pair", "nunique"), alc=("alc", "mean"), **{f"b{n}": (f"brier_{n}", "mean") for n in BUDGETS},
                                **{f"ece{n}": (f"ece_{n}", "mean") for n in BUDGETS}).reindex(ORDER)
    return g

L = ["# Phase 2, B3: configuration-aware prior mean (strict LOBO, nested hyperparameter selection)\n",
     "**Hypothesis.** B2 protects against small-sample overconfidence but systematically harms very low-success pairs because its prior mean is fixed at 0.5. A configuration-aware prior mean should preserve B2's budget-1/3 calibration gains while reducing those failures.\n",
     f"Seeds {SEEDS}; split seed 0. Grid: m_informative and m_backoff in {{1,2,3,4,6}}, shrink c in {{0,20,100}}; selected per held-out fold by nested LOBO on pooled equal-per-pair ALC (one seed inside the nest). No information from the held-out benchmark enters its prior.\n",
     "## Selected hyperparameters per fold\n", sel.round(4).to_markdown(index=False), "",
     "## Coverage: which back-off level supplied each held-out pair's prior\n",
     cov.groupby(["bench", "level"]).size().unstack(fill_value=0).reindex(columns=[l for l in ["exact", "name", "provider", "harness", "global"] if l in cov.level.unique()]).to_markdown(), "",
     f"Pooled: {cov.level.value_counts().to_dict()} of {len(cov)} pairs.", "",
     "## Pooled equal-per-pair results (selection statistic), 140 pairs\n", table(df).round(4).to_markdown(), ""]
for b in BENCH:
    L += [f"## Fold: {b} ({df[df.bench == b].pair.nunique()} pairs)\n", table(df[df.bench == b]).round(4).to_markdown(), ""]


pa = df.groupby(["method", "pair"]).alc.mean().unstack(0)
d = (pa["B3_config_prior"] - pa["B2_frozen_m2"]).rename("delta").to_frame(); d["rate"] = pd.Series(RATE); d["bench"] = [x.split(":")[0] for x in d.index]
d = d.join(cov.set_index("pair")[["level", "prior_mean"]])
rng = np.random.default_rng(0); bs = np.array([rng.choice(d.delta.to_numpy(), len(d)).mean() for _ in range(5000)])
L += ["## Pair-level change, B3 minus B2 frozen (ALC; negative = B3 better)\n",
      f"- pairs improved: {(d.delta < 0).sum()} of {len(d)}; worsened: {(d.delta > 0).sum()}; mean {d.delta.mean():+.4f}, median {d.delta.median():+.4f}, min {d.delta.min():+.4f}, max {d.delta.max():+.4f}",
      f"- paired bootstrap over pairs (5000): 95% CI [{np.percentile(bs, 2.5):+.4f}, {np.percentile(bs, 97.5):+.4f}]", "",
      d.groupby("level").delta.agg(["count", "mean", "median", lambda x: (x < 0).mean()]).rename(columns={"<lambda_0>": "frac_improved"}).round(4).to_markdown(), ""]

bins = pd.cut(d.rate, [0, .05, .1, .2, .5, 1.0], include_lowest=True)
strat = pa.join(d[["rate"]]).assign(bin=bins).groupby("bin", observed=True)[ORDER].mean().round(4)
strat.insert(0, "n_pairs", d.groupby(bins, observed=True).size())
L += ["## Stratified by true pair success rate (analysis only; the predictor never sees this)\n", strat.to_markdown(), ""]

lost = (pa["B2_frozen_m2"] - pa["B1_empirical_mean"]) > 0
L += [f"## The pairs where B2 (frozen) was worse than B1: {int(lost.sum())} pairs\n",
      f"- B3 better than B1 on {int(((pa['B3_config_prior'] - pa['B1_empirical_mean']) < 0)[lost].sum())} of them; B3 better than B2 on {int((d.delta < 0)[lost].sum())} of them.",
      f"- mean ALC on these pairs: B1 {pa['B1_empirical_mean'][lost].mean():.4f}, B2 {pa['B2_frozen_m2'][lost].mean():.4f}, B3 {pa['B3_config_prior'][lost].mean():.4f}, B3-global {pa['B3_global_prior_ablation'][lost].mean():.4f}",
      f"- and on the other {int((~lost).sum())} pairs: B1 {pa['B1_empirical_mean'][~lost].mean():.4f}, B2 {pa['B2_frozen_m2'][~lost].mean():.4f}, B3 {pa['B3_config_prior'][~lost].mean():.4f}, B3-global {pa['B3_global_prior_ablation'][~lost].mean():.4f}", ""]

cq = cov.assign(err=cov.prior_mean - cov.rate)
L += ["## Prior quality on held-out pairs (prior mean vs true success rate)\n",
      cq.groupby("level").agg(n=("pair", "size"), corr=("prior_mean", lambda x: np.corrcoef(x, cq.loc[x.index, "rate"])[0, 1] if len(x) > 2 else np.nan),
                              mean_abs_err=("err", lambda x: x.abs().mean()), mean_bias=("err", "mean")).round(3).to_markdown(), "",
      f"Baseline for comparison: a constant prior of 0.5 has mean absolute error {np.abs(0.5 - cov.rate).mean():.3f} over the same pairs.", ""]
open(f"{OUT}/summary.md", "w").write("\n".join(L)); print("\n".join(L))

import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
xs = [np.log2(n + 1) for n in BUDGETS]; cols = {"B0_constant_half": "0.5", "B1_empirical_mean": "C1", "B2_frozen_m2": "C0", "B3_config_prior": "C2", "B3_global_prior_ablation": "C4"}
fig, axes = plt.subplots(1, 5, figsize=(19, 3.8), sharey=True)
for ax, (title, dd) in zip(axes, [("pooled (140 pairs)", df)] + [(b, df[df.bench == b]) for b in BENCH]):
    t = table(dd)
    for meth in ORDER:
        ax.plot(xs, [t.loc[meth, f"b{n}"] for n in BUDGETS], marker="o", ms=4, color=cols[meth], label=f"{meth} ({t.loc[meth, 'alc']:.3f})")
    ax.set_xticks(xs); ax.set_xticklabels(BUDGETS); ax.set_title(title, fontsize=10); ax.grid(alpha=.3); ax.set_xlabel("label budget"); ax.legend(fontsize=6)
axes[0].set_ylabel("mean Brier over pairs"); plt.tight_layout(); plt.savefig(f"{OUT}/curves.png", dpi=130); print("saved curves")
