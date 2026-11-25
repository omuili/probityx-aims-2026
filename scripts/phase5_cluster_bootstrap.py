"""Cluster-aware uncertainty for the main comparison and the alternatives table (no new protocol runs).

Reads the pair-level outputs written by phase2_b2.py, phase2_b3.py, phase3_step4_b4.py,
phase4_h2_pairlevel.py and phase4_h3_local.py. For each comparison (per-pair ALC averaged over
candidate-order seeds 0, 1, 2) it reports:
  - the original paired bootstrap over pairs (5000 resamples, seed 0), which reproduces the report;
  - a two-way cluster (pigeonhole) bootstrap over harness and model name: harness groups and model
    names are resampled independently and each pair is weighted by the product of its draws
    (pairs without a harness form their own harness group; model names are shared across benchmarks);
  - the mean difference on each benchmark separately.
Items shared by the pairs of a benchmark are not resampled: the pair-level outputs do not keep
item-level predictions. The per-benchmark rows are the summary that does not assume independence
within a benchmark.
"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.getcwd())

OUT = "reports/phase5"; os.makedirs(OUT, exist_ok=True)
NB = 5000
BENCH = ["multi_swebench", "real_webagents", "researchcodebench", "swe_rebench"]


def per_pair(path, col, name):
    d = pd.read_csv(path); return d[d[col] == name].groupby("pair").alc.mean()


b2p = pd.read_csv("reports/phase2_b2/pairs.csv")
mstar = dict(pd.read_csv("reports/phase2_b2/selection.csv").set_index("held_out").m_star)
b2_lobo = pd.concat([b2p[(b2p.bench == b) & (b2p.m == mstar[b])] for b in BENCH]).groupby("pair").alc.mean()
b1 = b2p[b2p.m == 0.0].groupby("pair").alc.mean()
b2f = b2p[b2p.m == 2.0].groupby("pair").alc.mean()
meta = b2p[(b2p.m == 2.0) & (b2p.seed == 0)].set_index("pair")[["bench", "subject", "harness"]]
meta["hgroup"] = [h if isinstance(h, str) and h else f"pair:{p}" for p, h in zip(meta.index, meta.harness)]

S3, S4, H2, H3 = ("reports/phase2_b3/pairs.csv", "reports/phase3/step4_pairs.csv",
                  "reports/phase4/h2_pairlevel_pairs.csv", "reports/phase4/h3_pairs.csv")
ref3 = per_pair(S3, "method", "B2_frozen_m2")
COMP = [  # (label, alternative, reference)
    ("B2 strict LOBO vs B1", b2_lobo, b1),
    ("B2 fixed m=2 vs B1", b2f, b1),
    ("B3 configuration prior", per_pair(S3, "method", "B3_config_prior"), ref3),
    ("B3-global prior", per_pair(S3, "method", "B3_global_prior_ablation"), ref3),
    ("B4 text difficulty, nested lambda", per_pair(S4, "method", "B4_rel"), per_pair(S4, "method", "B2_frozen_m2")),
    ("B4 text difficulty, lambda=1", per_pair(S4, "method", "B4_rel_lambda1_fixed"), per_pair(S4, "method", "B2_frozen_m2")),
    ("B4 text level as prior mean", per_pair(S4, "method", "B4_abs"), per_pair(S4, "method", "B2_frozen_m2")),
    ("H2 uncertainty sampling", per_pair(H2, "policy", "uncertainty_0.15"), per_pair(H2, "policy", "random")),
    ("H2 eager labelling", per_pair(H2, "policy", "eager_first31"), per_pair(H2, "policy", "random")),
    ("H3 local smoother", per_pair(H3, "method", "H3_local"), per_pair(H3, "method", "B2_frozen_m2")),
    ("B4 in-benchmark CV (analysis only)", per_pair(S4, "method", "B4_rel_inbench_cv_ceiling_ANALYSIS_ONLY"), per_pair(S4, "method", "B2_frozen_m2")),
]

pairs = meta.index.to_numpy()
hcodes, hlev = pd.factorize(meta.hgroup); mcodes, mlev = pd.factorize(meta.subject)


def iid_ci(d):
    rng = np.random.default_rng(0)
    bs = np.array([rng.choice(d, len(d)).mean() for _ in range(NB)])
    return np.percentile(bs, [2.5, 97.5])


def twoway_ci(d):
    rng = np.random.default_rng(0); out = np.empty(NB)
    for k in range(NB):
        wh = np.bincount(rng.integers(0, len(hlev), len(hlev)), minlength=len(hlev))
        wm = np.bincount(rng.integers(0, len(mlev), len(mlev)), minlength=len(mlev))
        w = wh[hcodes] * wm[mcodes]
        out[k] = (w * d).sum() / w.sum() if w.sum() else np.nan
    return np.nanpercentile(out, [2.5, 97.5])


rows = []
for label, alt, ref in COMP:
    d = (alt - ref).reindex(pairs).to_numpy()
    assert not np.isnan(d).any(), label
    lo, hi = iid_ci(d); clo, chi = twoway_ci(d)
    pb = pd.Series(d, index=pairs).groupby(meta.bench).mean()
    rows.append({"comparison": label, "delta": d.mean(), "iid_lo": lo, "iid_hi": hi, "twoway_lo": clo, "twoway_hi": chi,
                 "improved": f"{(d < 0).sum()}/{len(d)}", **{f"d_{b}": pb[b] for b in BENCH},
                 "bench_better_tie_worse": f"{(pb < -1e-4).sum()}/{(pb.abs() <= 1e-4).sum()}/{(pb > 1e-4).sum()}"})
R = pd.DataFrame(rows); R.to_csv(f"{OUT}/cluster_bootstrap.csv", index=False)
L = ["# Phase 5: cluster-aware uncertainty (negative delta favours the first method)\n",
     f"{len(hlev)} harness groups, {len(mlev)} model names, {len(pairs)} pairs; {NB} resamples, seed 0. "
     "iid = original paired bootstrap over pairs; twoway = pigeonhole bootstrap over harness x model name.\n",
     R.round(4).to_markdown(index=False), ""]
open(f"{OUT}/cluster_bootstrap.md", "w").write("\n".join(L)); print("\n".join(L))
