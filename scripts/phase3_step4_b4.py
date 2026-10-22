from __future__ import annotations
import os, sys, time
import numpy as np, pandas as pd
sys.path.insert(0, os.getcwd())
from paiec.data.load import benchmark_dirs, load_pairs, load_tables
from paiec.eval.protocol import run_pair, BUDGETS
from paiec.predictors.baselines import constant_half, empirical_mean
from paiec.predictors.smoothed_mean import make_beta_mean
from paiec.predictors.config_prior import ConfigPrior, make_b3
from paiec.predictors.difficulty import item_table, DifficultyModel
from paiec.predictors.item_bayes import make_b4, make_b4_abs
from paiec.features.item_text import content_hash

SEEDS = (0, 1, 2); OUT = "reports/phase3"; os.makedirs(OUT, exist_ok=True)
LAMS = [0.0, 0.25, 0.5, 1.0, 2.0]
B = benchmark_dirs(); T = {b: load_tables(b) for b in B}
IT = {b: item_table(T[b]["items"], T[b]["response"], b) for b in B}
P = {b: [p for p in load_pairs(b)[0] if len(p.items) >= 80] for b in B}
RATE = {f"{b}:{j}": float(np.concatenate([np.asarray(v, float) for v in p.responses.values()]).mean()) for b in B for j, p in enumerate(P[b])}
ALPHA = dict(pd.read_csv(f"{OUT}/step2_summary.csv").set_index("held_out").alpha)

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

t0 = time.time(); all_rows, sel_rows = [], []
for held in B:
    train = [b for b in B if b != held]; alpha = float(ALPHA[held])

    best_lam, best_alc = None, np.inf
    for lam in LAMS:
        parts = []
        for t in train:
            dm_inner = DifficultyModel(alpha=alpha).fit(pd.concat([IT[b] for b in train if b != t]))
            parts.append(evaluate(t, make_b4(dm_inner, lam), seeds=(0,)))
        alc = pd.concat(parts).alc.mean()
        if alc < best_alc: best_lam, best_alc = lam, alc
        print(f"  [{held}] nested lambda={lam}: {alc:.4f}", flush=True)
    sel_rows.append({"held_out": held, "alpha": alpha, "lambda": best_lam, "nested_pooled_alc": best_alc})
    dm = DifficultyModel(alpha=alpha).fit(pd.concat([IT[b] for b in train])); assert held not in dm.train_benchmarks
    prior = ConfigPrior(shrink_c=100).fit([(b, T[b]["subjects"], T[b]["response"]) for b in train])
    
    tab = IT[held]; rng = np.random.default_rng(0); folds = rng.integers(0, 5, len(tab)); oo = np.zeros(len(tab))
    for f in range(5):
        mm = DifficultyModel(alpha=alpha, equal_benchmark_weight=False).fit(tab[folds != f]); oo[folds == f] = mm.predict_many(tab[folds == f].content.tolist(), "rel_logit")
    cv_delta = {content_hash(c): -d for c, d in zip(tab.content, oo)}   # difficulty = minus success logit
    ceiling_fn = lambda c: cv_delta.get(content_hash(c), 0.0)
    methods = [("B0_constant_half", constant_half), ("B1_empirical_mean", empirical_mean), ("B2_frozen_m2", make_beta_mean(2.0)),
               ("B3_global_secondary", make_b3(prior, 3.0, 3.0, global_only=True)),
               ("B4_rel", make_b4(dm, best_lam)), ("B4_rel_lambda1_fixed", make_b4(dm, 1.0)), ("B4_abs", make_b4_abs(dm)),
               ("B4_rel_inbench_cv_ceiling_ANALYSIS_ONLY", make_b4(dm, 1.0, delta_fn=ceiling_fn))]
    for name, fn in methods:
        all_rows.append(evaluate(held, fn, tag={"method": name}))
    print(f"{held}: lambda*={best_lam} ({time.time()-t0:.0f}s)", flush=True)

df = pd.concat(all_rows); df.to_csv(f"{OUT}/step4_pairs.csv", index=False)
sel = pd.DataFrame(sel_rows); sel.to_csv(f"{OUT}/step4_selection.csv", index=False)
ORDER = ["B0_constant_half", "B1_empirical_mean", "B2_frozen_m2", "B3_global_secondary", "B4_rel", "B4_rel_lambda1_fixed", "B4_abs", "B4_rel_inbench_cv_ceiling_ANALYSIS_ONLY"]
def table(d):
    return d.groupby("method").agg(n_pairs=("pair", "nunique"), alc=("alc", "mean"), **{f"b{n}": (f"brier_{n}", "mean") for n in BUDGETS},
                                   **{f"ece{n}": (f"ece_{n}", "mean") for n in BUDGETS}).reindex(ORDER)
pa = df.groupby(["method", "pair"]).alc.mean().unstack(0); b0 = df.groupby(["method", "pair"]).brier_0.mean().unstack(0)
rng = np.random.default_rng(0)
def ci(a, b, mask=None, col=pa):
    d = (col[a] - col[b]); d = d[mask] if mask is not None else d; d = d.to_numpy()
    bs = np.array([rng.choice(d, len(d)).mean() for _ in range(5000)])
    return f"{d.mean():+.4f} [{np.percentile(bs, 2.5):+.4f}, {np.percentile(bs, 97.5):+.4f}], improved {int((d < 0).sum())}/{len(d)}"
nomswe = ~pa.index.str.startswith("multi_swebench")
L = ["# Phase 3, step 4: H1 test. Difficulty-aware update (B4) vs B0, B1, B2 frozen, B3-global\n",
     "lambda = 0 reproduces B2 frozen exactly; the nested selection may choose it. All difficulty models are fit on training benchmarks only.\n",
     "## Selected lambda per fold (nested LOBO)\n", sel.round(4).to_markdown(index=False), "",
     "## Pooled equal-per-pair results, 140 pairs\n", table(df).round(4).to_markdown(), "",
     "## Budget 0 isolated (Brier at budget 0, pooled; this is where item information could add what labels cannot)\n",
     b0[ORDER].mean().round(4).to_frame("brier_0_pooled").T.to_markdown(), "",
     "## Paired bootstrap over pairs, ALC (negative = first better)\n",
     f"- B4_rel vs B2_frozen: {ci('B4_rel', 'B2_frozen_m2')}",
     f"- B4_rel vs B2_frozen, excluding Multi-SWE-bench: {ci('B4_rel', 'B2_frozen_m2', nomswe)}",
     f"- B4_rel_lambda1_fixed vs B2_frozen: {ci('B4_rel_lambda1_fixed', 'B2_frozen_m2')}",
     f"- B4_abs vs B2_frozen: {ci('B4_abs', 'B2_frozen_m2')}",
     f"- B4_rel vs B3_global: {ci('B4_rel', 'B3_global_secondary')}",
     f"- ANALYSIS ONLY, ceiling (in-benchmark CV delta) vs B2_frozen: {ci('B4_rel_inbench_cv_ceiling_ANALYSIS_ONLY', 'B2_frozen_m2')}",
     f"- ANALYSIS ONLY, ceiling vs B2_frozen at budget 0 only: {ci('B4_rel_inbench_cv_ceiling_ANALYSIS_ONLY', 'B2_frozen_m2', col=b0)}", ""]
for b in B:
    L += [f"## Fold: {b} ({df[df.bench == b].pair.nunique()} pairs)\n", table(df[df.bench == b]).round(4).to_markdown(), ""]
bins = pd.cut(pd.Series(RATE), [0, .05, .1, .2, .5, 1.0], include_lowest=True)
strat = pa.join(bins.rename("bin")).groupby("bin", observed=True)[ORDER].mean().round(4); strat.insert(0, "n_pairs", bins.groupby(bins, observed=True).size())
L += ["## Stratified by true pair success rate (analysis only)\n", strat.to_markdown(), ""]
open(f"{OUT}/step4_b4.md", "w").write("\n".join(L)); print("\n".join(L))

import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
xs = [np.log2(n + 1) for n in BUDGETS]; show = ["B1_empirical_mean", "B2_frozen_m2", "B3_global_secondary", "B4_rel", "B4_abs", "B4_rel_inbench_cv_ceiling_ANALYSIS_ONLY"]
cols = dict(zip(show, ["C1", "C0", "C4", "C2", "C3", "0.4"]))
fig, axes = plt.subplots(1, 5, figsize=(20, 3.9), sharey=True)
for ax, (title, dd) in zip(axes, [("pooled (140 pairs)", df)] + [(b, df[df.bench == b]) for b in B]):
    t = table(dd)
    for meth in show:
        ax.plot(xs, [t.loc[meth, f"b{n}"] for n in BUDGETS], marker="o", ms=4, color=cols[meth], ls="--" if "ANALYSIS" in meth else "-", label=f"{meth[:22]} ({t.loc[meth, 'alc']:.3f})")
    ax.set_xticks(xs); ax.set_xticklabels(BUDGETS); ax.set_title(title, fontsize=10); ax.grid(alpha=.3); ax.set_xlabel("label budget"); ax.legend(fontsize=6)
axes[0].set_ylabel("mean Brier over pairs"); plt.tight_layout(); plt.savefig(f"{OUT}/step4_curves.png", dpi=130); print("saved curves")
