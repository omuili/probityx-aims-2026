import numpy as np
from paiec.predictors.item_bayes import posterior_predictive, make_b4, make_b4_abs
from paiec.predictors.smoothed_mean import make_beta_mean


class FakeDM:
    def __init__(self, rel=None, absl=None):
        self.rel = rel or (lambda c: 0.0); self.absl = absl or (lambda c: 0.0)
    def predict_rel(self, c): return self.rel(c)
    def predict_difficulty(self, c): return -self.rel(c)
    def predict_abs(self, c): return self.absl(c)


S = {"normalized_name": "m", "provider": "p", "release_date": "", "access_date": "", "harness": "h", "harness_version": "", "reasoning_effort": "", "subject_features_extra": ""}
I = lambda k, txt="x": {"item_content": f"{txt}{k}", "item_features": "", "interactors": "", "benchmark_id": "b"}
L = lambda ys: [[[S, I(k)], y] for k, y in enumerate(ys)]


def test_lambda_zero_reproduces_frozen_b2_exactly():
    f = make_b4(FakeDM(), lam=0.0, m=2.0); g = make_beta_mean(2.0)
    rng = np.random.default_rng(0)
    for _ in range(40):
        ys = rng.integers(0, 2, size=int(rng.integers(0, 32))).tolist()
        assert abs(f([S, I(99)], L(ys)) - g([S, I(99)], L(ys))) < 2e-3   # grid discretization only


def test_budget0_uniform_prior_gives_half_and_hard_item_below_half():
    assert abs(posterior_predictive([], [], 0.0, lam=1.0) - 0.5) < 1e-3
    assert posterior_predictive([], [], target_delta=2.0, lam=1.0) < 0.5 < posterior_predictive([], [], target_delta=-2.0, lam=1.0)


def test_failing_easy_items_lowers_ability_more_than_failing_hard_items():
    # H1 mechanism: same empirical rate (0/3), different inferred ability
    easy = posterior_predictive([0, 0, 0], [-2.0, -2.0, -2.0], target_delta=0.0, lam=1.0)
    hard = posterior_predictive([0, 0, 0], [2.0, 2.0, 2.0], target_delta=0.0, lam=1.0)
    assert easy < hard


def test_b4_abs_prior_centres_on_item_prediction_at_budget0():
    dm = FakeDM(absl=lambda c: 2.0 if "hard" not in c else -2.0)
    f = make_b4_abs(dm)
    easy, hard = f([S, I(1, "easy")], []), f([S, I(1, "hard")], [])
    assert easy > 0.7 and hard < 0.3 and abs(easy + hard - 1.0) < 1e-3


def test_sign_convention_predicted_easy_item_gets_higher_budget0_prediction():
    class DM:
        def predict_rel(self, c): return 2.0 if "easy" in c else -2.0
        def predict_difficulty(self, c): return -self.predict_rel(c)
    f = make_b4(DM(), lam=1.0)
    assert f([S, I(1, "easy")], []) > 0.5 > f([S, I(1, "hard")], [])


def test_true_difficulty_beats_b2_on_synthetic_pairs():
    rng = np.random.default_rng(1); n_items = 200
    diff = rng.normal(0, 1.5, n_items)                     # true difficulty (higher = harder)
    look = {f"x{k}": float(diff[k]) for k in range(n_items)}
    dm = type("DM", (), {"predict_difficulty": staticmethod(lambda c: look[c]), "predict_rel": staticmethod(lambda c: -look[c])})
    b4 = make_b4(dm, lam=1.0); b2 = make_beta_mean(2.0)
    sq4 = sq2 = 0.0; cnt = 0
    for theta in [-1.0, 0.5, 1.5]:
        items = [I(k) for k in range(n_items)]
        ys = (rng.random(n_items) < 1 / (1 + np.exp(-(theta - diff)))).astype(int)
        lab = [[[S, items[k]], int(ys[k])] for k in range(7)]          # 7 labels
        for k in range(7, n_items):
            for L_ in ([], lab):
                sq4 += (b4([S, items[k]], L_) - ys[k]) ** 2; sq2 += (b2([S, items[k]], L_) - ys[k]) ** 2; cnt += 1
    assert sq4 / cnt < sq2 / cnt


def test_output_always_a_valid_probability_under_extreme_inputs():
    for delta, ys in [(-50.0, [1] * 31), (50.0, [0] * 31), (-50.0, []), (0.0, [1] * 31)]:
        p = posterior_predictive(ys, [delta] * len(ys), target_delta=delta, lam=2.0)
        assert 0.0 <= p <= 1.0 and np.isfinite(p)


def test_no_nan_for_absurd_deltas_and_large_lambda():
    p = posterior_predictive([1, 0] * 15, [1e4, -1e4] * 15, target_delta=1e4, lam=2.0)
    assert np.isfinite(p) and 0.0 <= p <= 1.0
