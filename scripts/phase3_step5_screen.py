from __future__ import annotations
import os, sys, time
import numpy as np, pandas as pd, scipy.sparse as sp
from scipy.stats import spearmanr, pearsonr
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
sys.path.insert(0, os.getcwd())
import torch; torch.set_num_threads(8)
from paiec.data.load import benchmark_dirs, load_tables
from paiec.predictors.difficulty import item_table
from paiec.features.representations import HandFeatures, TfidfRep, EncoderRep

OUT = "reports/phase3"; os.makedirs(OUT, exist_ok=True)
B = benchmark_dirs(); T = {b: load_tables(b) for b in B}
IT = {b: item_table(T[b]["items"], T[b]["response"], b) for b in B}
ALPHAS = [1.0, 10.0, 100.0, 1000.0, 10000.0]
GATE_POOLED, GATE_MIN_FOLD = 0.50, -0.10
REPS = {"hand19": lambda: HandFeatures(), "tfidf": lambda: TfidfRep(), "minilm": lambda: EncoderRep("minilm"), "bge": lambda: EncoderRep("bge")}

def weights(tab):
    counts = tab.bench.value_counts(); return tab.bench.map(lambda b: 1.0 / counts[b]).to_numpy() * len(counts)

def fit_predict(rep, train_tab, test_tab, alpha, dense_cache_tag=None):
    """Fit rep on training contents, standardize (dense) with training stats, ridge with equal benchmark weight."""
    rep.fit(train_tab.content.tolist())
    if isinstance(rep, EncoderRep):
       
        Xtr = np.vstack([rep.transform(train_tab[train_tab.bench == b].content.tolist(), cache_tag=b) for b in sorted(train_tab.bench.unique())])
        train_tab = pd.concat([train_tab[train_tab.bench == b] for b in sorted(train_tab.bench.unique())])
        Xte = rep.transform(test_tab.content.tolist(), cache_tag=test_tab.bench.iloc[0])
    else:
        Xtr = rep.transform(train_tab.content.tolist()); Xte = rep.transform(test_tab.content.tolist())
    w = weights(train_tab); y = train_tab.rel_logit.to_numpy(); ybar = np.average(y, weights=w)
    if sp.issparse(Xtr):
        model = Ridge(alpha=alpha, fit_intercept=False, solver="sparse_cg", random_state=0).fit(Xtr, y - ybar, sample_weight=w)
        return model.predict(Xte) + ybar
    mu = np.average(Xtr, axis=0, weights=w); sd = np.sqrt(np.average((Xtr - mu) ** 2, axis=0, weights=w)) + 1e-8
    model = Ridge(alpha=alpha, fit_intercept=False).fit((Xtr - mu) / sd, y - ybar, sample_weight=w)
    return model.predict((Xte - mu) / sd) + ybar

def metrics(y, p):
    hard = (y < np.median(y)).astype(int)
    return spearmanr(y, p).correlation, pearsonr(y, p)[0], (roc_auc_score(hard, -p) if 0 < hard.mean() < 1 else np.nan)

rows, preds = [], []; t0 = time.time()
for name, make in REPS.items():
    for held in B:
        train = [b for b in B if b != held]
        best_a, best_s = None, -np.inf
        for a in ALPHAS:                               
            s = []
            for t in train:
                p = fit_predict(make(), pd.concat([IT[b] for b in train if b != t]), IT[t], a)
                s.append(spearmanr(IT[t].rel_logit, p).correlation)
            if np.nanmean(s) > best_s: best_a, best_s = a, float(np.nanmean(s))
        p = fit_predict(make(), pd.concat([IT[b] for b in train]), IT[held], best_a)
        sp_, pe_, auc_ = metrics(IT[held].rel_logit.to_numpy(), p)
        rows.append({"representation": name, "held_out": held, "n_items": len(p), "alpha": best_a, "nested_spearman": best_s, "spearman": sp_, "pearson": pe_, "auroc_hard": auc_})
        preds.append(pd.DataFrame({"representation": name, "bench": held, "item_id": IT[held].item_id, "rel_true": IT[held].rel_logit, "rel_pred": p}))
        print(f"{name:7s} {held:18s} alpha={best_a:<7g} spearman={sp_:+.3f} pearson={pe_:+.3f} auroc={auc_:.3f}  ({time.time()-t0:.0f}s)", flush=True)

R = pd.DataFrame(rows); Pdf = pd.concat(preds); Pdf.to_csv(f"{OUT}/step5_predictions.csv", index=False)
summary = []
for name in REPS:
    r = R[R.representation == name]; pp = Pdf[Pdf.representation == name]
    pooled = spearmanr(pp.rel_true, pp.rel_pred).correlation
    summary.append({"representation": name, "pooled_spearman_all_items": pooled, "mean_fold_spearman": r.spearman.mean(), "median_fold_spearman": r.spearman.median(),
                    "min_fold_spearman": r.spearman.min(), "min_fold": r.loc[r.spearman.idxmin(), "held_out"],
                    "gate": bool(pooled >= GATE_POOLED and r.spearman.min() >= GATE_MIN_FOLD)})
S = pd.DataFrame(summary); R.to_csv(f"{OUT}/step5_folds.csv", index=False); S.to_csv(f"{OUT}/step5_summary.csv", index=False)
L = ["# Phase 3, step 5: representation screening for out-of-benchmark item difficulty (strict LOBO)\n",
     "Pre-registered set: hand19 (step 2 reference), TF-IDF word 1-2 grams (vocabulary from training folds), all-MiniLM-L6-v2 (frozen), bge-base-en-v1.5 (frozen). Deterministic chunking: 1,500-char chunks, up to 32 evenly spaced, mean-pooled. Ridge on the representation, alpha by nested LOBO, training benchmarks equally weighted, target = within-benchmark centred success logit. No fine-tuning, no Codabench feedback, no B4 runs.\n",
     f"**Gate:** pooled out-of-benchmark Spearman >= {GATE_POOLED} and no held-out benchmark below {GATE_MIN_FOLD}.\n",
     "## Summary\n", S.round(3).to_markdown(index=False), "",
     "pooled_spearman_all_items: Spearman over the concatenation of every held-out item across the four folds (6,306 of the 8,877 items are SWE-rebench, so this statistic is dominated by that fold; the mean and median over folds are reported alongside).\n",
     "## Per fold\n", R.round(3).to_markdown(index=False), ""]
passing = S[S.gate].representation.tolist()
L.append(f"## Gate decision\n\n{'Representations clearing the gate: ' + ', '.join(passing) if passing else 'No representation clears the gate.'}\n")
open(f"{OUT}/step5_screening.md", "w").write("\n".join(L)); print("\n".join(L))
