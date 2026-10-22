import numpy as np
from paiec.predictors.smoothed_mean import make_beta_mean
from paiec.predictors.baselines import empirical_mean

S = {"normalized_name": "m", "provider": "p", "release_date": "", "access_date": "", "harness": "h",
     "harness_version": "1", "reasoning_effort": "", "subject_features_extra": ""}
S2 = dict(S, harness="other")
I = lambda k, b="b1": {"item_content": f"q{k}", "item_features": "", "interactors": "", "benchmark_id": b}


def labels(ys, subj=S, bench="b1"):
    return [[[subj, I(k, bench)], y] for k, y in enumerate(ys)]


def test_budget0_is_half_for_any_m():
    for m in [0, 0.5, 2, 100]:
        assert make_beta_mean(m)([S, I(9)], []) == 0.5


def test_m_zero_recovers_official_empirical_mean():
    rng = np.random.default_rng(0)
    for _ in range(50):
        ys = rng.integers(0, 2, size=rng.integers(1, 12)).tolist()
        assert make_beta_mean(0.0)([S, I(99)], labels(ys)) == empirical_mean([S, I(99)], labels(ys))


def test_one_label_never_gives_zero_or_one_when_m_positive():
    assert 0 < make_beta_mean(2.0)([S, I(1)], labels([1])) < 1
    assert 0 < make_beta_mean(2.0)([S, I(1)], labels([0])) < 1
    assert make_beta_mean(2.0)([S, I(1)], labels([1])) == (1 + 1) / (2 + 1)


def test_shrinks_toward_half_as_m_grows_and_toward_mean_as_n_grows():
    ys = [1, 1, 1, 1, 0]
    p_small, p_big = make_beta_mean(0.5)([S, I(1)], labels(ys)), make_beta_mean(50)([S, I(1)], labels(ys))
    assert abs(p_big - 0.5) < abs(p_small - 0.5)
    p_more = make_beta_mean(2.0)([S, I(1)], labels(ys * 20))
    assert abs(p_more - 0.8) < abs(make_beta_mean(2.0)([S, I(1)], labels(ys)) - 0.8)


def test_ignores_other_subjects_and_other_benchmarks():
    L = labels([1, 1, 1]) + labels([0, 0, 0], subj=S2) + labels([0, 0, 0], bench="b2")
    assert make_beta_mean(2.0)([S, I(1)], L) == (1 + 3) / (2 + 3)
