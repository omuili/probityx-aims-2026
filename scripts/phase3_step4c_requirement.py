import os, sys, numpy as np, pandas as pd
from scipy.stats import spearmanr
sys.path.insert(0, os.getcwd())
from paiec.data.load import benchmark_dirs, load_pairs, load_tables
from paiec.eval.protocol import run_pair, BUDGETS
from paiec.predictors.smoothed_mean import make_beta_mean
from paiec.predictors.difficulty import item_table
from paiec.predictors.item_bayes import make_b4
from paiec.features.item_text import content_hash
OUT = "reports/phase3"; B = benchmark_dirs(); T = {b: load_tables(b) for b in B}
IT = {b: item_table(T[b]["items"], T[b]["response"], b) for b in B}
P = {b: [p for p in load_pairs(b)[0] if len(p.items) >= 80] for b in B}
def evaluate(bench, fn):
    rows = []
    for j, p in enumerate(P[bench]):
        r = run_pair(p, fn, split_seed=0, order_seed=0); rows.append({"bench": bench, "pair": f"{bench}:{j}", "alc": r.alc, **{f"brier_{n}": r.brier[n] for n in BUDGETS}})
    return pd.DataFrame(rows)
rng = np.random.default_rng(0); rows = []
for held in B:
    tab = IT[held]; true = -tab.rel_logit.to_numpy()          # difficulty (higher = harder)
    rows.append(evaluate(held, make_beta_mean(2.0)).assign(source="B2_frozen", target_rho=0.0, lam=0.0, rho=0.0))
    for target in [0.3, 0.5, 0.7, 0.9]:
     
        sd = true.std(); noise_sd = sd * np.sqrt(1 / target**2 - 1); noisy = true + rng.normal(0, noise_sd, len(true))
        rho = spearmanr(true, noisy).correlation; look = {content_hash(c): float(d) for c, d in zip(tab.content, noisy)}
        for lam in [0.5, 1.0]:
            rows.append(evaluate(held, make_b4(None, lam, delta_fn=lambda c, look=look: look.get(content_hash(c), 0.0))).assign(source="noisy_true", target_rho=target, lam=lam, rho=rho))
    print(held, "done", flush=True)
df = pd.concat(rows); df.to_csv(f"{OUT}/step4c_requirement_pairs.csv", index=False)
g = df.groupby(["target_rho", "lam"]).agg(alc=("alc", "mean"), brier_0=("brier_0", "mean"), brier_1=("brier_1", "mean"), brier_7=("brier_7", "mean"), rho_achieved=("rho", "mean")).round(4)
L = ["# Phase 3, step 4c: ANALYSIS-ONLY requirement curve. ALC as a function of how well item difficulty is predicted\n",
     "Item difficulty = true relative difficulty plus Gaussian noise tuned to a target Spearman. Deployable text models in step 2 reached Spearman about 0.0 to 0.2 out of benchmark (0.2 to 0.4 within benchmark). Seed 0, 140 pairs, equal weight per pair.\n",
     g.to_markdown(), "", "Row target_rho = 0 is B2 frozen (no item information)."]
open(f"{OUT}/step4c_requirement.md", "w").write("\n".join(L)); print("\n".join(L))
