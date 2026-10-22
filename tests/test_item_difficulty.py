import numpy as np, pandas as pd
from paiec.features.item_text import item_features, FEATURES, content_hash
from paiec.predictors.difficulty import DifficultyModel, item_table


def test_feature_vector_shape_and_determinism():
    v = item_features("Fix the bug in `foo.py`:\n```python\ndef f(x):\n    return x\n```\nTraceback (most recent call last)")
    assert v.shape == (len(FEATURES),) and np.all(np.isfinite(v))
    assert np.array_equal(v, item_features("Fix the bug in `foo.py`:\n```python\ndef f(x):\n    return x\n```\nTraceback (most recent call last)"))
    assert np.all(np.isfinite(item_features(""))) and content_hash("") == content_hash(None)


def test_features_respond_to_obvious_properties():
    f = dict(zip(FEATURES, item_features("short task?")))
    g = dict(zip(FEATURES, item_features("x" * 5000 + "\n```\ncode\n```\nhttp://a.b/c ```")))
    assert g["log_chars"] > f["log_chars"] and g["n_code_fences"] > f["n_code_fences"] and g["n_urls"] > f["n_urls"] and f["n_questions"] > g["n_questions"]


def make_bench(name, n_items, seed):
    rng = np.random.default_rng(seed)
    items = pd.DataFrame({"item_id": [f"{name}_{i}" for i in range(n_items)], "content": ["word " * int(rng.integers(5, 400)) for _ in range(n_items)]})
    # difficulty grows with length: longer text -> lower success
    L = items.content.str.len().to_numpy(); p = 1 / (1 + np.exp((L - L.mean()) / L.std()))
    resp = pd.DataFrame([{"subject_id": f"s{s}", "item_id": iid, "response": int(rng.random() < pi)} for iid, pi in zip(items.item_id, p) for s in range(8)])
    return item_table(items, resp, name)


def test_item_table_targets():
    t = make_bench("b", 50, 0)
    assert set(["k", "n", "p", "abs_logit", "rel_logit", "bench", "content"]) <= set(t.columns)
    assert abs(t.rel_logit.mean()) < 1e-9 and (t.n == 8).all()


def test_difficulty_model_transfers_length_signal_and_uses_only_training_stats():
    train = pd.concat([make_bench("a", 120, 1), make_bench("b", 120, 2)]); held = make_bench("c", 120, 3)
    m = DifficultyModel(alpha=10.0).fit(train)
    assert m.train_benchmarks == ["a", "b"]
    pred = m.predict_many(held.content.tolist(), "rel_logit")
    assert np.corrcoef(pred, held.rel_logit)[0, 1] > 0.5           # longer -> harder is learned and transfers
    assert abs(m.predict_rel(held.content.iloc[0]) - pred[0]) < 1e-9  # cached single-item path agrees
