from __future__ import annotations
import os, sys, itertools
import numpy as np, pandas as pd
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import roc_auc_score
sys.path.insert(0, os.getcwd())
from paiec.data.load import benchmark_dirs, load_tables
from paiec.predictors.difficulty import item_table, DifficultyModel, featurize
from paiec.features.item_text import FEATURES

OUT = "reports/phase3"; os.makedirs(OUT, exist_ok=True)
B = benchmark_dirs(); T = {b: load_tables(b) for b in B}
IT = {b: item_table(T[b]["items"], T[b]["response"], b) for b in B}
ALPHAS = [1.0, 10.0, 100.0, 1000.0]
sig = lambda x: 1 / (1 + np.exp(-x))

def corr_block(y_true, y_pred):
    sp = spearmanr(y_true, y_pred).correlation; pe = pearsonr(y_true, y_pred)[0]
    hard = (y_true < np.median(y_true)).astype(int)
    auc = roc_auc_score(hard, -y_pred) if 0 < hard.mean() < 1 else np.nan
    return sp, pe, auc

rows, preds, coefs = [], [], {}
for held in B:
    train = [b for b in B if b != held]
    best_a, best_s = None, -np.inf
    for a in ALPHAS:
        s = []
        for t in train:
            m = DifficultyModel(alpha=a).fit(pd.concat([IT[b] for b in train if b != t]))
            s.append(spearmanr(IT[t].rel_logit, m.predict_many(IT[t].content.tolist(), "rel_logit")).correlation)
        if np.nanmean(s) > best_s: best_a, best_s = a, np.nanmean(s)
    m = DifficultyModel(alpha=best_a).fit(pd.concat([IT[b] for b in train]))
    assert held not in m.train_benchmarks
    tab = IT[held]; pr = m.predict_many(tab.content.tolist(), "rel_logit"); pa = m.predict_many(tab.content.tolist(), "abs_logit")
    sp_r, pe_r, auc_r = corr_block(tab.rel_logit.to_numpy(), pr); sp_a, pe_a, auc_a = corr_block(tab.abs_logit.to_numpy(), pa)
   
    rng = np.random.default_rng(0); folds = rng.integers(0, 5, len(tab)); oo = np.zeros(len(tab))
    for f in range(5):
        mm = DifficultyModel(alpha=best_a, equal_benchmark_weight=False).fit(tab[folds != f]); oo[folds == f] = mm.predict_many(tab[folds == f].content.tolist(), "rel_logit")
    sp_o = spearmanr(tab.rel_logit, oo).correlation
  
    true_level = tab.p.mean(); pred_level = sig(pa).mean(); train_level = np.mean([IT[b].p.mean() for b in train])
    rows.append({"held_out": held, "n_items": len(tab), "alpha": best_a, "nested_spearman": round(best_s, 3),
                 "spearman_rel": sp_r, "pearson_rel": pe_r, "auroc_hard_rel": auc_r, "spearman_abs": sp_a, "auroc_hard_abs": auc_a,
                 "in_benchmark_cv_spearman_rel (analysis only)": sp_o,
                 "true_mean_item_success": true_level, "pred_level_from_text": pred_level, "train_benchmarks_mean_level": train_level, "constant_half": 0.5})
    preds.append(pd.DataFrame({"bench": held, "item_id": tab.item_id, "p_true": tab.p, "rel_true": tab.rel_logit, "rel_pred": pr, "abs_true": tab.abs_logit, "abs_pred": pa}))
    coefs[held] = m.coefficients("rel_logit")
    print(f"{held}: alpha={best_a} spearman rel={sp_r:+.3f} abs={sp_a:+.3f} | in-benchmark CV oracle {sp_o:+.3f} | level true {true_level:.3f} text {pred_level:.3f} train-mean {train_level:.3f}", flush=True)

R = pd.DataFrame(rows); pd.concat(preds).to_csv(f"{OUT}/step2_predictions.csv", index=False)
C = pd.DataFrame(coefs).reindex(FEATURES)
L = ["# Phase 3, step 2: does item text predict difficulty out of benchmark?\n",
     f"Features: {len(FEATURES)} text-only statistics of item_content (paiec/features/item_text.py). Model: ridge on standardized features, "
     "training benchmarks equally weighted, alpha by nested LOBO. Targets from training benchmarks only: abs_logit = logit item success across subjects; rel_logit = abs_logit centred within benchmark.\n",
     "## Transfer to the held-out benchmark (deployable numbers)\n",
     R[["held_out", "n_items", "alpha", "nested_spearman", "spearman_rel", "pearson_rel", "auroc_hard_rel", "spearman_abs", "auroc_hard_abs"]].round(3).to_markdown(index=False), "",
     "Spearman: rank correlation between predicted and true item difficulty within the held-out benchmark (0 = no information). AUROC: separating the harder half of items from the easier half.\n",
     "## Analysis-only reference: how much text signal exists WITHIN each benchmark\n",
     R[["held_out", "spearman_rel", "in_benchmark_cv_spearman_rel (analysis only)"]].round(3).to_markdown(index=False), "",
     "The in-benchmark number fits and tests inside the held-out benchmark (5-fold CV) and therefore uses its outcomes; it is NOT deployable and is shown only so the transfer number can be read against the ceiling of this feature set.\n",
     "## Step 3: benchmark-level information at budget 0 from aggregated item predictions\n",
     R[["held_out", "true_mean_item_success", "pred_level_from_text", "train_benchmarks_mean_level", "constant_half"]].round(3).to_markdown(index=False), "",
     "pred_level_from_text = mean over the held-out benchmark's items of sigmoid(predicted abs_logit). Compare its miss against the two references a pair-level model can use: the training benchmarks' mean level (B3-global) and 0.5 (B2).\n",
     "## Ridge coefficients for rel_logit (standardized features; positive = predicts easier), per fold\n", C.round(3).to_markdown(), ""]
R.to_csv(f"{OUT}/step2_summary.csv", index=False); open(f"{OUT}/step2_difficulty.md", "w").write("\n".join(L)); print("\n".join(L))
