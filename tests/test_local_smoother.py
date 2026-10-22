import numpy as np
from paiec.predictors.local_smoother import make_h3, EmbeddingLookup, beta_mean
from paiec.features.item_text import content_hash
from paiec.predictors.smoothed_mean import make_beta_mean

S = {"normalized_name": "m", "provider": "p", "release_date": "", "access_date": "", "harness": "h", "harness_version": "", "reasoning_effort": "", "subject_features_extra": ""}
I = lambda k: {"item_content": f"item{k}", "item_features": "", "interactors": "", "benchmark_id": "b"}


def lookup_from(vectors):
    return EmbeddingLookup({content_hash(f"item{k}"): np.asarray(v, float) for k, v in vectors.items()})


def test_budget0_and_identical_embeddings_reduce_to_b2():
    L = lookup_from({k: [1.0, 0.0] for k in range(10)})
    f = make_h3(L, tau=0.1, w0=2.0); b2 = make_beta_mean(2.0)
    assert f([S, I(9)], []) == 0.5
    lab = [[[S, I(k)], y] for k, y in enumerate([1, 0, 1, 1])]
    p = f([S, I(9)], lab); pb2 = b2([S, I(9)], lab)
    assert abs(p - (2 * pb2 + 3) / (2 + 4)) < 1e-9


def test_large_w0_recovers_b2():
    L = lookup_from({0: [1, 0], 1: [0, 1], 2: [1, 0], 9: [1, 0]})
    lab = [[[S, I(0)], 1], [[S, I(1)], 0], [[S, I(2)], 1]]
    p = make_h3(L, tau=0.05, w0=1e9)([S, I(9)], lab)
    assert abs(p - make_beta_mean(2.0)([S, I(9)], lab)) < 1e-6


def test_similar_neighbours_dominate_when_tau_small():
    L = lookup_from({0: [1, 0], 1: [0, 1], 9: [1, 0]})
    lab = [[[S, I(0)], 1], [[S, I(1)], 0]]          
    p_local = make_h3(L, tau=0.05, w0=0.5)([S, I(9)], lab)
    p_b2 = make_beta_mean(2.0)([S, I(9)], lab)
    assert p_local > p_b2 + 0.1


def test_permutation_control_preserves_counts_and_is_deterministic():
    L = lookup_from({k: np.eye(6)[k] for k in range(6)})
    lab = [[[S, I(k)], y] for k, y in enumerate([1, 1, 1, 0, 0])]
    f = make_h3(L, tau=0.05, w0=1.0, permute=True)
    assert f([S, I(5)], lab) == f([S, I(5)], lab)
    assert abs(make_h3(L, 0.05, 1e9, permute=True)([S, I(5)], lab) - beta_mean(3, 5)) < 1e-6


def test_missing_embedding_falls_back_to_b2():
    L = lookup_from({0: [1, 0]})
    lab = [[[S, I(0)], 1], [[S, I(7)], 0]]     # item7 has no embedding
    assert abs(make_h3(L, 0.1, 1.0)([S, I(9)], lab) - beta_mean(1, 2)) < 1e-9
