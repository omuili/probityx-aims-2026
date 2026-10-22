import os, sys, numpy as np, pandas as pd
sys.path.insert(0, os.getcwd())
from paiec.data.load import benchmark_dirs, load_pairs, load_tables
from paiec.eval.protocol import run_pair, BUDGETS
from paiec.predictors.smoothed_mean import make_beta_mean
from paiec.predictors.difficulty import item_table, DifficultyModel
from paiec.predictors.item_bayes import make_b4
from paiec.features.item_text import content_hash
OUT = "reports/phase3"; B = benchmark_dirs(); T = {b: load_tables(b) for b in B}
IT = {b: item_table(T[b]["items"], T[b]["response"], b) for b in B}
P = {b: [p for p in load_pairs(b)[0] if len(p.items) >= 80] for b in B}
ALPHA = dict(pd.read_csv(f"{OUT}/step2_summary.csv").set_index("held_out").alpha)
LAMS = [0.1, 0.25, 0.5, 1.0]
def evaluate(bench, fn):
    rows = []
    for j, p in enumerate(P[bench]):
        r = run_pair(p, fn, split_seed=0, order_seed=0); rows.append({"bench": bench, "pair": f"{bench}:{j}", "alc": r.alc, **{f"brier_{n}": r.brier[n] for n in BUDGETS}})
    return pd.DataFrame(rows)
rows = []
for held in B:
    tab = IT[held]; alpha = float(ALPHA[held])
    rng = np.random.default_rng(0); folds = rng.integers(0, 5, len(tab)); cv = np.zeros(len(tab))
    for f in range(5):
        mm = DifficultyModel(alpha=alpha, equal_benchmark_weight=False).fit(tab[folds != f]); cv[folds == f] = mm.predict_many(tab[folds == f].content.tolist(), "rel_logit")
    # deltas are DIFFICULTIES: minus the relative success logit
    src = {"cv": {content_hash(c): -d for c, d in zip(tab.content, cv)}, "true": {content_hash(c): -d for c, d in zip(tab.content, tab.rel_logit)}}
    b2 = evaluate(held, make_beta_mean(2.0)); rows.append(b2.assign(source="B2_frozen", lam=0.0))
    for name, look in src.items():
        for lam in LAMS:
            rows.append(evaluate(held, make_b4(None, lam, delta_fn=lambda c, look=look: look.get(content_hash(c), 0.0))).assign(source=name, lam=lam))
    print(held, "done", flush=True)
df = pd.concat(rows); df.to_csv(f"{OUT}/step4b_ceiling_pairs.csv", index=False)
pooled = df.groupby(["source", "lam"]).agg(alc=("alc", "mean"), brier_0=("brier_0", "mean"), brier_1=("brier_1", "mean"), brier_7=("brier_7", "mean"), brier_31=("brier_31", "mean")).round(4)
perb = df.groupby(["source", "lam", "bench"]).alc.mean().unstack("bench").round(4)
best = {s: g.alc.idxmin()[1] for s, g in pooled.groupby(level=0) if s != "B2_frozen"}
L = ["# Phase 3, step 4b: ANALYSIS-ONLY ceilings for item-difficulty information (seed 0, 140 pairs)\n",
     "Neither row is deployable: 'cv' uses the held-out benchmark's outcomes through in-benchmark cross-validation; 'true' uses each item's actual difficulty. lambda is chosen on the held-out fold. These are optimistic upper bounds on what the B4 mechanism could deliver with (a) this feature set at its best and (b) perfect item difficulty.\n",
     "## Pooled, by delta source and lambda\n", pooled.to_markdown(), "", f"Best lambda per source (optimistic): {best}", "",
     "## Per benchmark ALC\n", perb.to_markdown(), ""]
open(f"{OUT}/step4b_ceiling.md", "w").write("\n".join(L)); print("\n".join(L))
