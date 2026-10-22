"""Load measurement-db tables into platform-shaped Pair objects.

Every field value handed to predictors is a plain string; missing values are "" (competition
interface). Subject dicts carry exactly the eight platform keys. Item dicts carry item_content,
item_features, interactors and an anonymous benchmark_id. interactors is recorded per response in
measurement-db; we attach the per-item modal value.
"""
from __future__ import annotations
import glob, hashlib, os
import numpy as np, pandas as pd
from ..eval.protocol import Pair, SUBJECT_KEYS, ITEM_KEYS

RAW = "data/raw"


def _s(v) -> str:
    if v is None:
        return ""
    if isinstance(v, float) and np.isnan(v):
        return ""
    if isinstance(v, (list, np.ndarray)):
        return " ".join(str(x) for x in v)
    return str(v)


def benchmark_dirs(raw: str = RAW) -> list[str]:
    return sorted(os.path.basename(d.rstrip("/")) for d in glob.glob(os.path.join(raw, "*/"))
                  if os.path.exists(os.path.join(d, "response.parquet")))


def load_tables(bench: str, raw: str = RAW) -> dict[str, pd.DataFrame]:
    d = os.path.join(raw, bench)
    return {t: pd.read_parquet(os.path.join(d, f"{t}.parquet")) for t in ("benchmarks", "subjects", "items", "response")}


def anon_benchmark_id(bench: str) -> str:
    return "benchmark_" + hashlib.sha1(bench.encode()).hexdigest()[:6]


def subject_dict(row: pd.Series) -> dict:
    return {k: _s(row.get(k)) for k in SUBJECT_KEYS}


def load_pairs(bench: str, raw: str = RAW) -> tuple[list[Pair], dict]:
    """Return (pairs, meta). One Pair per subject with >= 1 response; items keyed by item_id."""
    T = load_tables(bench, raw)
    items = T["items"].set_index(T["items"]["item_id"].astype(str))
    resp = T["response"].copy()
    resp["item_id"] = resp["item_id"].astype(str); resp["subject_id"] = resp["subject_id"].astype(str)
    resp = resp[resp["response"].isin([0, 1, 0.0, 1.0])]
    inter = resp.groupby("item_id")["interactors"].agg(lambda x: (m.iloc[0] if len(m := x.fillna("").astype(str).mode()) else ""))
    bid = anon_benchmark_id(bench)
    item_dicts = {iid: {"item_content": _s(items.at[iid, "content"]) if iid in items.index else "",
                        "item_features": _s(items.at[iid, "item_features"]) if iid in items.index else "",
                        "interactors": _s(inter.get(iid, "")), "benchmark_id": bid} for iid in resp["item_id"].unique()}
    subjects = T["subjects"].copy(); subjects["subject_id"] = subjects["subject_id"].astype(str)
    pairs = []
    for sid, g in resp.groupby("subject_id"):
        srow = subjects[subjects["subject_id"] == sid]
        sd = subject_dict(srow.iloc[0]) if len(srow) else {k: "" for k in SUBJECT_KEYS}
        responses = {iid: [int(v) for v in gg["response"].tolist()] for iid, gg in g.groupby("item_id")}
        pairs.append(Pair(sd, bid, {iid: item_dicts[iid] for iid in responses}, responses))
    meta = {"bench": bench, "benchmark_id": bid, "n_subjects": len(pairs), "n_items": len(item_dicts),
            "n_responses": int(len(resp)), "row": T["benchmarks"].iloc[0].to_dict()}
    return pairs, meta
