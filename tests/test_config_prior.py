import numpy as np, pandas as pd
from paiec.predictors.config_prior import ConfigPrior, make_b3, _sigmoid


def bench(name, subjects, rates, n=100, seed=0):
    """subjects: list of dicts with normalized_name/provider/harness; rates: success prob per subject."""
    rng = np.random.default_rng(seed)
    S = pd.DataFrame([{"subject_id": f"{name}_s{i}", "display_name": s["normalized_name"], "normalized_name": s["normalized_name"],
                       "provider": s.get("provider", ""), "release_date": "", "access_date": "", "harness": s.get("harness", ""),
                       "reasoning_effort": "", "harness_version": "", "subject_features_extra": ""} for i, s in enumerate(subjects)])
    R = pd.DataFrame([{"subject_id": f"{name}_s{i}", "item_id": str(j), "response": int(rng.random() < r)}
                      for i, r in enumerate(rates) for j in range(n)])
    return name, S, R


def platform(sd):
    return {k: str(sd.get(k, "")) for k in ("normalized_name", "provider", "release_date", "access_date", "harness", "harness_version", "reasoning_effort", "subject_features_extra")}


STRONG = {"normalized_name": "StrongModel", "provider": "Acme", "harness": "H1"}
WEAK = {"normalized_name": "WeakModel", "provider": "Acme", "harness": "H1"}
OTHER = {"normalized_name": "OtherModel", "provider": "Zeta", "harness": "H2"}


def test_backoff_levels_and_ordering():
    cp = ConfigPrior(shrink_c=0).fit([bench("b1", [STRONG, WEAK], [0.8, 0.1]), bench("b2", [STRONG, OTHER], [0.7, 0.4], seed=1)])
    p_strong, lvl_s, _ = cp.prior(platform(STRONG)); p_weak, lvl_w, _ = cp.prior(platform(WEAK))
    assert lvl_s == "exact" and lvl_w == "exact" and p_strong > p_weak
    p, lvl, _ = cp.prior(platform(dict(STRONG, harness="H9"))); assert lvl == "name"
    p, lvl, _ = cp.prior(platform({"normalized_name": "NewModel", "provider": "Acme", "harness": "H9"})); assert lvl == "provider"
    p, lvl, _ = cp.prior(platform({"normalized_name": "NewModel", "provider": "Nobody", "harness": "H1"})); assert lvl == "harness"
    p, lvl, k = cp.prior(platform({"normalized_name": "X", "provider": "Y", "harness": "Z"})); assert lvl == "global" and k == 0 and abs(p - _sigmoid(cp.G)) < 1e-12


def test_shrinkage_pulls_toward_global():
    b = [bench("b1", [STRONG, WEAK], [0.9, 0.1])]
    p_sharp, _, _ = ConfigPrior(shrink_c=0).fit(b).prior(platform(STRONG))
    p_shrunk, _, _ = ConfigPrior(shrink_c=1000).fit(b).prior(platform(STRONG))
    G = _sigmoid(ConfigPrior().fit(b).G)
    assert abs(p_shrunk - G) < abs(p_sharp - G)


def test_strict_lobo_prior_ignores_held_out_benchmark():
    train = [bench("b1", [STRONG, WEAK], [0.8, 0.1])]
    cp = ConfigPrior().fit(train)
    assert cp.fitted_benchmarks == ["b1"]
    cp2 = ConfigPrior().fit(train + [bench("held", [STRONG], [0.01], seed=3)])
    assert cp.prior(platform(STRONG))[0] > cp2.prior(platform(STRONG))[0]   # held-out data would have changed it


def test_b3_predictor_reduces_to_beta_update_around_prior_mean():
    cp = ConfigPrior(shrink_c=0).fit([bench("b1", [STRONG, WEAK], [0.8, 0.1])])
    S = platform(WEAK); I = {"item_content": "", "item_features": "", "interactors": "", "benchmark_id": "new"}
    f = make_b3(cp, m_informative=2.0, m_backoff=2.0)
    p0 = cp.prior(S)[0]
    assert abs(f([S, I], []) - p0) < 1e-12
    assert abs(f([S, I], [[[S, I], 1]]) - (2 * p0 + 1) / 3) < 1e-12
    g = make_b3(cp, 2.0, 2.0, global_only=True)
    assert abs(g([S, I], []) - _sigmoid(cp.G)) < 1e-12
