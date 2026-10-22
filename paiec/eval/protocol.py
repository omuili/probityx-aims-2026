"""Offline reproduction of the competition's test-time adaptation protocol.

For each subject-benchmark pair with at least MIN_ITEMS distinct items, items are split 50/50 into
an acquisition pool (labels may be revealed; never scored) and an evaluation pool (scored; never
revealed). The split is seeded and fixed. Acquisition candidates stream in a fixed random order;
a policy decides buy/skip with the platform's context dictionary; the nested label sets at budgets
0, 1, 3, 7, 15, 31 are snapshots of one trajectory. Predictions at each budget are scored on all
recorded responses of the evaluation items. Pair ALCs are averaged with equal weight.

The interface matches the platform: predict(input, labeled) -> float with
input = [subject_dict, item_dict] and labeled = [[[subject_dict, item_dict], label], ...];
acquisition_function(input, prediction, labeled, context) -> bool.
"""
from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Callable
import numpy as np
from .metrics import BUDGETS, brier, ece, alc

MIN_ITEMS = 80
MAX_LABELS = 31
SUBJECT_KEYS = ("normalized_name", "provider", "release_date", "access_date", "harness",
                "harness_version", "reasoning_effort", "subject_features_extra")
ITEM_KEYS = ("item_content", "item_features", "interactors", "benchmark_id")

Predict = Callable[[list, list], float]
Acquire = Callable[[list, float | None, list | None, dict | None], bool] | None


@dataclass
class Pair:
    subject: dict                       # platform-shaped subject dict (all strings)
    benchmark_id: str                   # anonymous id used in item dicts
    items: dict[str, dict]              # item_id -> platform-shaped item dict
    responses: dict[str, list[int]]     # item_id -> recorded 0/1 outcomes (all trials)


@dataclass
class PairResult:
    brier: dict[int, float]
    ece: dict[int, float]
    alc: float
    n_eval_items: int
    n_eval_responses: int
    acquired: list[str] = field(default_factory=list)   # item ids in acquisition order


def split_pool(item_ids: list[str], seed: int) -> tuple[list[str], list[str]]:
    """Seeded 50/50 split of a pair's items into (acquisition, evaluation)."""
    rng = np.random.default_rng(seed)
    ids = sorted(item_ids)
    perm = rng.permutation(len(ids))
    half = len(ids) // 2
    acq = [ids[k] for k in perm[:half]]
    ev = [ids[k] for k in perm[half:]]
    return acq, ev


def random_policy(input, prediction=None, labeled=None, context=None) -> bool:
    """Platform default when labeling.py is absent: pick uniformly so the budget is filled.

    Deterministic given the candidate order: buy with probability labels_remaining/items_remaining,
    which yields a uniform random subset of size 31 over the stream.
    """
    rng = context["_rng"]
    if context["labels_remaining"] <= 0:
        return False
    if context["items_remaining"] <= context["labels_remaining"]:
        return True
    return bool(rng.random() < context["labels_remaining"] / context["items_remaining"])


def run_pair(pair: Pair, predict: Predict, acquire: Acquire = None, *, split_seed: int = 0,
             order_seed: int = 0) -> PairResult | None:
    if len(pair.items) < MIN_ITEMS:
        return None
    acq_ids, eval_ids = split_pool(list(pair.items), split_seed)
    rng = np.random.default_rng(order_seed)
    order = [acq_ids[k] for k in rng.permutation(len(acq_ids))]
    policy = acquire or random_policy
    labeled: list = []
    acquired: list[str] = []
    checkpoints: dict[int, list] = {0: []}
    # One trajectory; snapshot the labeled list when its size hits each budget.
    for pos, iid in enumerate(order):
        if len(labeled) >= MAX_LABELS:
            break
        inp = [pair.subject, pair.items[iid]]
        ctx = {"subject_id": "anon", "benchmark_id": pair.benchmark_id,
               "labels_acquired": len(labeled), "labels_remaining": MAX_LABELS - len(labeled),
               "max_labels": MAX_LABELS, "items_remaining": len(order) - pos, "_rng": rng}
        pred = None
        if acquire is not None:
            pred = float(predict(inp, list(labeled)))
        if policy(inp, pred, list(labeled), ctx):
            # reveal one recorded response for this item (the platform reveals one response)
            y = int(pair.responses[iid][rng.integers(len(pair.responses[iid]))])
            labeled.append([[pair.subject, pair.items[iid]], y]); acquired.append(iid)
            if len(labeled) in BUDGETS:
                checkpoints[len(labeled)] = list(labeled)
    # budgets not reached (fewer than 31 candidates) reuse the largest available label set
    reached = max(checkpoints)
    for n in BUDGETS:
        if n not in checkpoints:
            checkpoints[n] = checkpoints[reached] if n > reached else checkpoints[max(k for k in checkpoints if k <= n)]
    ys = np.concatenate([np.asarray(pair.responses[i], float) for i in eval_ids])
    briers, eces = {}, {}
    for n in BUDGETS:
        L = checkpoints[n]
        preds = {}
        for iid in eval_ids:
            p = float(predict([pair.subject, pair.items[iid]], list(L)))
            if not math.isfinite(p) or p < 0 or p > 1:
                raise ValueError(f"invalid probability {p!r}")
            preds[iid] = p
        ps = np.concatenate([np.full(len(pair.responses[i]), preds[i]) for i in eval_ids])
        briers[n] = brier(ps, ys); eces[n] = ece(ps, ys)
    return PairResult(briers, eces, alc(briers), len(eval_ids), int(ys.size), acquired)


def summarize(results: list[PairResult]) -> dict:
    """Equal-weight average over pairs, as the platform does.

    Selection statistic: pool the held-out pairs of all LOBO folds and call this once
    on the pooled list. Per-benchmark summaries are reported alongside but never averaged equally
    across benchmarks, because pair counts differ by two orders of magnitude.
    """
    out = {"n_pairs": len(results), "alc": float(np.mean([r.alc for r in results]))}
    for n in BUDGETS:
        out[f"brier_{n}"] = float(np.mean([r.brier[n] for r in results]))
        out[f"ece_{n}"] = float(np.mean([r.ece[n] for r in results]))
    return out
