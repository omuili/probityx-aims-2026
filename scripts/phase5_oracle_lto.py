"""ANALYSIS-ONLY oracle and requirement curve with leave-target-subject-out (LTO) item difficulty.

Same protocol, predictor (B4) and lambda grid as phase3_step4b_ceiling.py / phase3_step4c_requirement.py,
but each target pair's item difficulty is computed from the responses of the OTHER subjects on the
benchmark only, so the scored outcomes of the target subject never enter its own difficulty.
SWE-rebench has a single subject, so LTO difficulty is undefined there and it is reported as n/a
(pooled figures use the remaining 139 pairs; B2 is reported on the same 139 pairs).
"""
import os, sys, numpy as np, pandas as pd
from scipy.stats import spearmanr
sys.path.insert(0, os.getcwd())
from paiec.data.load import benchmark_dirs, load_pairs, load_tables
from paiec.eval.protocol import run_pair, BUDGETS
from paiec.predictors.smoothed_mean import make_beta_mean
from paiec.predictors.difficulty import item_table
from paiec.predictors.item_bayes import make_b4
from paiec.features.item_text import content_hash

OUT = "reports/phase5"; os.makedirs(OUT, exist_ok=True)
LAMS = [0.1, 0.25, 0.5, 1.0]
TARGETS = [0.3, 0.5, 0.7, 0.9]


def score(p, fn, bench, j, **extra):
    r = run_pair(p, fn, split_seed=0, order_seed=0)
    return {"bench": bench, "pair": f"{bench}:{j}", "alc": r.alc, **{f"brier_{n}": r.brier[n] for n in BUDGETS}, **extra}


rows = []
for bi, bench in enumerate(benchmark_dirs()):
    T = load_tables(bench)
    resp = T["response"].copy(); resp["subject_id"] = resp.subject_id.astype(str)
    resp = resp[resp.response.isin([0, 1, 0.0, 1.0])]
    pairs, _ = load_pairs(bench)
    sids = [sid for sid, _ in resp.groupby("subject_id")]          # same order as load_pairs
    assert len(sids) == len(pairs)
    kept = [(sid, p) for sid, p in zip(sids, pairs) if len(p.items) >= 80]
    for j, (sid, p) in enumerate(kept):
        assert sum(len(v) for v in p.responses.values()) == int((resp.subject_id == sid).sum())
        rows.append(score(p, make_beta_mean(2.0), bench, j, source="B2_frozen", lam=0.0, target_rho=np.nan, rho=np.nan))
        others = resp[resp.subject_id != sid]
        if others.empty:
            continue                                               # SWE-rebench: no other subject
        tab = item_table(T["items"], others, bench)
        diff = -tab.rel_logit.to_numpy()                           # difficulty (higher = harder)
        look = {content_hash(c): float(d) for c, d in zip(tab.content, diff)}
        for lam in LAMS:
            fn = make_b4(None, lam, delta_fn=lambda c, look=look: look.get(content_hash(c), 0.0))
            rows.append(score(p, fn, bench, j, source="lto_true", lam=lam, target_rho=np.nan, rho=np.nan))
        for ti, target in enumerate(TARGETS):
            rng = np.random.default_rng([0, bi, j, ti])
            noisy = diff + rng.normal(0, diff.std() * np.sqrt(1 / target**2 - 1), len(diff))
            rho = spearmanr(diff, noisy).correlation
            nlook = {content_hash(c): float(d) for c, d in zip(tab.content, noisy)}
            for lam in [0.5, 1.0]:
                fn = make_b4(None, lam, delta_fn=lambda c, look=nlook: look.get(content_hash(c), 0.0))
                rows.append(score(p, fn, bench, j, source="lto_noisy", lam=lam, target_rho=target, rho=rho))
    print(bench, len(kept), "pairs done", flush=True)

df = pd.DataFrame(rows); df.to_csv(f"{OUT}/oracle_lto_pairs.csv", index=False)
lto_pairs = set(df[df.source == "lto_true"].pair)
d139 = df[df.pair.isin(lto_pairs)]
agg = dict(alc=("alc", "mean"), brier_0=("brier_0", "mean"), brier_7=("brier_7", "mean"), brier_31=("brier_31", "mean"))
oracle = d139[d139.source.isin(["B2_frozen", "lto_true"])].groupby(["source", "lam"]).agg(n_pairs=("pair", "nunique"), **agg).round(4)
perb = df[df.source.isin(["B2_frozen", "lto_true"])].groupby(["source", "lam", "bench"]).alc.mean().unstack("bench").round(4)
req = d139[d139.source == "lto_noisy"].groupby(["target_rho", "lam"]).agg(rho_achieved=("rho", "mean"), **agg).round(4)

# Same-run comparison with the original (all-subject) oracle on the same 139 pairs, from step 4b's outputs.
orig = pd.read_csv("reports/phase3/step4b_ceiling_pairs.csv")
orig = orig[(orig.source == "true") & orig.pair.isin(lto_pairs)].groupby("lam").alc.mean().round(4)

L = ["# Phase 5: ANALYSIS-ONLY oracle and requirement curve, leave-target-subject-out difficulty (seed 0)\n",
     "Item difficulty for each target pair is computed from the other subjects' responses on the benchmark only. "
     "SWE-rebench (one subject) is n/a; pooled rows use the other 139 pairs. lambda is chosen on the held-out fold (optimistic).\n",
     "## Oracle, pooled over 139 pairs\n", oracle.to_markdown(), "",
     "## Original all-subject oracle on the same 139 pairs (from reports/phase3/step4b_ceiling_pairs.csv)\n",
     orig.to_frame("alc_all_subject_true").to_markdown(), "",
     "## Per benchmark ALC\n", perb.to_markdown(), "",
     "## Requirement curve (noise added to LTO difficulty; rho = Spearman with LTO difficulty), 139 pairs\n", req.to_markdown(), ""]
open(f"{OUT}/oracle_lto.md", "w").write("\n".join(L)); print("\n".join(L))
