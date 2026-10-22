"""Generate the dataset summary report. Writes reports/data_report.md.

Per eligible benchmark: subjects, items, trials per pair; pair success-rate distribution; number
of pairs with >= 80 items (the ones the platform adapts on); within-benchmark difficulty spread
from a per-benchmark Rasch fit; two Brier references (base-rate ybar(1-ybar) and the irreducible
floor E[pi(1-pi)] estimated from repeated trials); configuration overlap across benchmarks;
variation of reasoning effort and test_condition; text statistics; duplicate item text across
benchmarks by content hash.
"""
from __future__ import annotations
import os, re
import numpy as np, pandas as pd
from .load import benchmark_dirs, load_tables, load_pairs
from ..eval.protocol import MIN_ITEMS


def rasch_fit(y: pd.DataFrame, iters: int = 200, prior_sd: float = 3.0) -> tuple[pd.Series, pd.Series]:
    """Alternating Newton steps on a Rasch model P(y=1) = sigma(theta_s - delta_i), item mean 0.

    A weak Gaussian prior (sd 3 logits) on abilities and difficulties keeps items answered all-0 or
    all-1 by every subject finite; without it their difficulty diverges and the spread is meaningless.
    """
    subj = {s: k for k, s in enumerate(sorted(y.subject_id.unique()))}
    item = {i: k for k, i in enumerate(sorted(y.item_id.unique()))}
    s = y.subject_id.map(subj).to_numpy(); i = y.item_id.map(item).to_numpy(); r = y.response.to_numpy(float)
    th = np.zeros(len(subj)); de = np.zeros(len(item))
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(th[s] - de[i])))
        g = np.bincount(s, r - p, len(subj)) - th / prior_sd**2; h = np.bincount(s, p * (1 - p), len(subj)) + 1 / prior_sd**2
        th += np.clip(g / h, -1, 1)
        p = 1 / (1 + np.exp(-(th[s] - de[i])))
        g = np.bincount(i, p - r, len(item)) - de / prior_sd**2; h = np.bincount(i, p * (1 - p), len(item)) + 1 / prior_sd**2
        de += np.clip(g / h, -1, 1); de -= de.mean()
    return pd.Series(th, index=list(subj)), pd.Series(de, index=list(item))


def text_stats(content: pd.Series) -> dict:
    c = content.fillna("").astype(str)
    code = c.str.count(r"```|def |class |import |\{|\}|;") / c.str.len().clip(lower=1)
    return {"median_chars": int(c.str.len().median()), "p90_chars": int(c.str.len().quantile(0.9)),
            "code_marker_density_median": float(code.median()), "empty_fraction": float((c.str.len() == 0).mean())}


def main(out: str = "reports/data_report.md"):
    os.makedirs(os.path.dirname(out), exist_ok=True)
    L = ["# Data report (Phase 1)\n", f"Eligible benchmarks (binary, item granularity): {benchmark_dirs()}\n"]
    hashes = {}; configs = {}
    for b in benchmark_dirs():
        T = load_tables(b); pairs, meta = load_pairs(b)
        resp = T["response"]; resp = resp[resp["response"].isin([0, 1])].copy()
        resp["subject_id"] = resp.subject_id.astype(str); resp["item_id"] = resp.item_id.astype(str)
        L.append(f"\n## {meta['row']['name']} (`{b}`)\n")
        L.append(f"- domain {meta['row'].get('domain')}; modality {meta['row'].get('modality')}; released {meta['row'].get('release_date')}")
        L.append(f"- subjects {meta['n_subjects']}, items {meta['n_items']}, binary responses {meta['n_responses']}, overall success rate {resp.response.mean():.3f}")
        trials = resp.groupby(["subject_id", "item_id"]).size()
        L.append(f"- trials per (subject,item): mean {trials.mean():.2f}, max {trials.max()}, fraction with >=2 trials {(trials>=2).mean():.2f}")
        n_items_pair = resp.groupby("subject_id").item_id.nunique()
        L.append(f"- pairs with >= {MIN_ITEMS} items (platform adapts on these): {(n_items_pair>=MIN_ITEMS).sum()} of {len(n_items_pair)}; items per pair median {int(n_items_pair.median())}")
        rate = resp.groupby("subject_id").response.mean()
        L.append(f"- pair success rates: min {rate.min():.2f}, q25 {rate.quantile(.25):.2f}, median {rate.median():.2f}, q75 {rate.quantile(.75):.2f}, max {rate.max():.2f}; pairs in [0.2,0.8]: {((rate>=.2)&(rate<=.8)).mean():.2f}")
        # Brier references per pair, then equal-weight mean over pairs
        per_item = resp.groupby(["subject_id", "item_id"]).response.agg(["mean", "count"])
        base = (rate * (1 - rate)).mean()
        ok = per_item["count"] >= 2
        if ok.mean() >= 0.05:
            floor_item = (per_item["count"] / (per_item["count"] - 1) * per_item["mean"] * (1 - per_item["mean"]))[ok]
            floor = floor_item.groupby(level=0).mean().mean()
            L.append(f"- Brier references (equal weight over pairs): base-rate ybar(1-ybar) = {base:.3f}; irreducible floor E[pi(1-pi)] from repeated trials = {floor:.3f} (on {ok.mean():.0%} of pair-items with >=2 trials)")
        else:
            L.append(f"- Brier references: base-rate ybar(1-ybar) = {base:.3f}; irreducible floor not identifiable (only {ok.mean():.1%} of pair-items have >=2 trials)")
        th, de = rasch_fit(resp)
        L.append(f"- Rasch fit: item difficulty sd {de.std():.2f} (logits), IQR {de.quantile(.75)-de.quantile(.25):.2f}; subject ability sd {th.std():.2f}, range [{th.min():.2f}, {th.max():.2f}]")
        S = T["subjects"]
        for col in ["harness", "reasoning_effort", "harness_version", "provider"]:
            if col in S:
                vc = S[col].fillna("").astype(str).value_counts()
                L.append(f"- {col}: {len(vc)} distinct; top {dict(list(vc.head(4).items()))}")
        if "test_condition" in resp:
            L.append(f"- test_condition distinct: {resp.test_condition.fillna('').astype(str).nunique()}; interactors distinct: {resp.interactors.fillna('').astype(str).nunique()}")
        L.append(f"- item text: {text_stats(T['items']['content'])}")
        for _, r in S.iterrows():
            configs.setdefault(str(r.get("normalized_name", "")), set()).add(b)
        for h in T["items"]["content_hash"].dropna().astype(str):
            hashes.setdefault(h, set()).add(b)
    shared_models = {m: sorted(v) for m, v in configs.items() if len(v) > 1}
    L.append("\n## Cross-benchmark overlap\n")
    L.append(f"- model names appearing in more than one benchmark: {len(shared_models)} of {len(configs)}")
    for m, v in sorted(shared_models.items())[:40]:
        L.append(f"  - {m}: {v}")
    dup = {h: sorted(v) for h, v in hashes.items() if len(v) > 1}
    L.append(f"- item content hashes shared across benchmarks (leakage check): {len(dup)}")
    open(out, "w").write("\n".join(L) + "\n"); print("\n".join(L))


if __name__ == "__main__":
    main()
