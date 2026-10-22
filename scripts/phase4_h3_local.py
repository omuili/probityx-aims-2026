from __future__ import annotations
import os, sys, time, itertools, glob
import numpy as np, pandas as pd
from scipy.stats import pearsonr
sys.path.insert(0, os.getcwd())
from paiec.data.load import benchmark_dirs, load_pairs, load_tables
from paiec.eval.protocol import run_pair, BUDGETS
from paiec.predictors.smoothed_mean import make_beta_mean
from paiec.predictors.local_smoother import make_h3, EmbeddingLookup
from paiec.features.item_text import content_hash

REP = "bge"; OUT = "reports/phase4"; os.makedirs(OUT, exist_ok=True); SEEDS = (0, 1, 2)
GRID = list(itertools.product([0.05, 0.1, 0.2], [1.0, 2.0, 4.0]))      # (tau, w0)
B = benchmark_dirs()
store = {}
for f in glob.glob(f"data/cache/emb_{REP}_*.npz"):
    z = np.load(f, allow_pickle=True); store.update(dict(zip(z["hashes"].tolist(), z["emb"])))
LOOK = EmbeddingLookup(store); print(f"embeddings cached: {len(store)}", flush=True)
P = {b: [p for p in load_pairs(b)[0] if len(p.items) >= 80] for b in B}
RATE = {f"{b}:{j}": float(np.concatenate([np.asarray(v, float) for v in p.responses.values()]).mean()) for b in B for j, p in enumerate(P[b])}


def part_a(tau=0.1):
    rows = []; rng = np.random.default_rng(0)
    for b in B:
        res, loc, loc_perm = [], [], []
        for p in P[b]:
            ids = list(p.items); E = np.stack([LOOK.get(p.items[i]["item_content"]) for i in ids])
            y = np.array([np.mean(p.responses[i]) for i in ids]); r = y - y.mean()
            E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12); W = np.exp(-(1 - E @ E.T) / tau); np.fill_diagonal(W, 0.0)
            l = (W @ r) / (W.sum(1) + 1e-12); rp = rng.permutation(r); lp = (W @ rp) / (W.sum(1) + 1e-12)
            res.append(r); loc.append(l); loc_perm.append(lp)
        res, loc, loc_perm = map(np.concatenate, (res, loc, loc_perm))
        rows.append({"bench": b, "n_items": len(res), "corr_residual_vs_local": pearsonr(res, loc)[0], "corr_permutation_null": pearsonr(res, loc_perm)[0]})
    return pd.DataFrame(rows)

A = part_a(); print(A.round(3).to_string(), flush=True)


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
    train = [b for b in B if b != held]
    best, best_alc = None, np.inf
    for tau, w0 in GRID:
        alc = pd.concat([evaluate(t, make_h3(LOOK, tau, w0), seeds=(0,)) for t in train]).alc.mean()
        if alc < best_alc: best, best_alc = (tau, w0), alc
    tau, w0 = best; sel_rows.append({"held_out": held, "tau": tau, "w0": w0, "nested_pooled_alc": best_alc})
    for name, fn in [("B2_frozen_m2", make_beta_mean(2.0)), ("H3_local", make_h3(LOOK, tau, w0)), ("H3_local_PERMUTED_control", make_h3(LOOK, tau, w0, permute=True))]:
        all_rows.append(evaluate(held, fn, tag={"method": name}))
    print(f"{held}: tau={tau} w0={w0} ({time.time()-t0:.0f}s)", flush=True)

df = pd.concat(all_rows); df.to_csv(f"{OUT}/h3_pairs.csv", index=False)
sel = pd.DataFrame(sel_rows); sel.to_csv(f"{OUT}/h3_selection.csv", index=False)
ORDER = ["B2_frozen_m2", "H3_local", "H3_local_PERMUTED_control"]
def table(d):
    return d.groupby("method").agg(n_pairs=("pair", "nunique"), alc=("alc", "mean"), **{f"b{n}": (f"brier_{n}", "mean") for n in BUDGETS}, **{f"ece{n}": (f"ece_{n}", "mean") for n in BUDGETS}).reindex(ORDER)
pa = df.groupby(["method", "pair"]).alc.mean().unstack(0); rng = np.random.default_rng(0)
def ci(a, b, mask=None):
    d = (pa[a] - pa[b]); d = d[mask] if mask is not None else d; d = d.to_numpy()
    bs = np.array([rng.choice(d, len(d)).mean() for _ in range(5000)])
    return f"{d.mean():+.4f} [{np.percentile(bs, 2.5):+.4f}, {np.percentile(bs, 97.5):+.4f}], improved {int((d < 0).sum())}/{len(d)}"
nomswe = ~pa.index.str.startswith("multi_swebench")
L = ["# Phase 4 (H3): within-pair local outcome transfer, frozen bge-base embeddings\n",
     "H3: within a subject-benchmark pair, outcomes on semantically similar labelled items carry information about an unlabelled target beyond the pair-wide Beta-Binomial posterior. Smoother p_hat = (w0 p_B2 + sum_j w_ij y_j)/(w0 + sum_j w_ij), w_ij = exp(-(1 - cos)/tau). Pre-registered grid tau in {0.05, 0.1, 0.2}, w0 in {1, 2, 4}, selected per fold by nested LOBO. No cross-benchmark transfer of anything.\n",
     "## Part A: residual diagnostic (analysis only, all outcomes)\n",
     "Pearson correlation between an item's residual outcome (y - pair mean) and the similarity-weighted leave-one-out mean of the other items' residuals in the same pair, tau = 0.1; against a within-pair permutation null.\n",
     A.round(3).to_markdown(index=False), "",
     "## Part B: protocol, 140 pairs, three seeds\n", "Selected hyperparameters:", sel.round(4).to_markdown(index=False), "",
     "### Pooled equal-per-pair\n", table(df).round(4).to_markdown(), "",
     "### Paired bootstrap over pairs, ALC (negative = first better)\n",
     f"- H3 vs B2 frozen: {ci('H3_local', 'B2_frozen_m2')}",
     f"- H3 vs B2 frozen, excluding Multi-SWE-bench: {ci('H3_local', 'B2_frozen_m2', nomswe)}",
     f"- H3 permuted control vs B2 frozen: {ci('H3_local_PERMUTED_control', 'B2_frozen_m2')}",
     f"- H3 vs its permuted control: {ci('H3_local', 'H3_local_PERMUTED_control')}", ""]
for b in B:
    L += [f"### Fold: {b} ({df[df.bench == b].pair.nunique()} pairs)\n", table(df[df.bench == b]).round(4).to_markdown(), ""]
bins = pd.cut(pd.Series(RATE), [0, .05, .1, .2, .5, 1.0], include_lowest=True)
strat = pa.join(bins.rename("bin")).groupby("bin", observed=True)[ORDER].mean().round(4); strat.insert(0, "n_pairs", bins.groupby(bins, observed=True).size())
L += ["### Stratified by true pair success rate (analysis only)\n", strat.to_markdown(), ""]
open(f"{OUT}/h3_local.md", "w").write("\n".join(L)); print("\n".join(L))
