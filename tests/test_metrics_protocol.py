import numpy as np
from paiec.eval.metrics import brier, alc, alc_weighted, ece, BUDGETS
from paiec.eval.protocol import Pair, run_pair, summarize, split_pool, MAX_LABELS
from paiec.predictors.baselines import constant_half, empirical_mean


def test_alc_trapezoid_equals_published_weights():
    rng = np.random.default_rng(0)
    for _ in range(50):
        b = {n: float(rng.uniform(0, 0.3)) for n in BUDGETS}
        assert abs(alc(b) - alc_weighted(b)) < 1e-12


def test_illustrative_curve_from_competition_page():
    # Page: Brier 0.22, 0.18, 0.145, 0.12, 0.1, 0.09 -> Brier ALC 0.140
    b = dict(zip(BUDGETS, [0.22, 0.18, 0.145, 0.12, 0.1, 0.09]))
    assert abs(alc(b) - 0.140) < 1e-9


def test_constant_half_brier_is_quarter():
    y = np.array([0, 1, 1, 0, 1])
    assert abs(brier(np.full(5, 0.5), y) - 0.25) < 1e-12


def test_ece_perfect_and_bins():
    p = np.array([0.05, 0.95, 0.95, 0.05]); y = np.array([0, 1, 1, 0])
    assert ece(p, y) < 0.06


def make_pair(n_items=100, p_true=0.7, trials=2, seed=1):
    rng = np.random.default_rng(seed)
    subj = {"normalized_name": "m", "provider": "p", "release_date": "", "access_date": "", "harness": "h",
            "harness_version": "1", "reasoning_effort": "", "subject_features_extra": ""}
    items = {f"i{k}": {"item_content": f"q{k}", "item_features": "", "interactors": "", "benchmark_id": "b1"} for k in range(n_items)}
    resp = {k: [int(rng.random() < p_true) for _ in range(trials)] for k in items}
    return Pair(subj, "b1", items, resp)


def test_split_is_seeded_and_disjoint():
    ids = [f"i{k}" for k in range(100)]
    a1, e1 = split_pool(ids, 7); a2, e2 = split_pool(ids, 7)
    assert a1 == a2 and e1 == e2 and not set(a1) & set(e1) and len(a1) + len(e1) == 100


def test_small_pairs_are_skipped():
    assert run_pair(make_pair(n_items=50), constant_half) is None


def test_constant_half_alc_quarter():
    r = run_pair(make_pair(), constant_half)
    assert abs(r.alc - 0.25) < 1e-12 and all(abs(v - 0.25) < 1e-12 for v in r.brier.values())


def test_empirical_mean_budget0_is_half_and_uses_labels_later():
    r = run_pair(make_pair(p_true=0.9), empirical_mean, order_seed=3)
    assert abs(r.brier[0] - 0.25) < 1e-12
    assert len(r.acquired) == MAX_LABELS
    assert r.brier[31] < r.brier[0]


def test_nested_labels_and_budget_snapshots():
    seen = []
    def spy(inp, labeled):
        seen.append(len(labeled)); return 0.5
    run_pair(make_pair(), spy)
    assert set(seen) == set(BUDGETS)


def test_summary_equal_weight():
    rs = [run_pair(make_pair(p_true=p, seed=s), empirical_mean) for s, p in enumerate([0.2, 0.9])]
    s = summarize(rs)
    assert s["n_pairs"] == 2 and abs(s["alc"] - np.mean([r.alc for r in rs])) < 1e-12
