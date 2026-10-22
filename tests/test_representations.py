import numpy as np
from paiec.features.representations import chunks, normalise, TfidfRep, HandFeatures, CHUNK_CHARS, MAX_CHUNKS


def test_chunking_is_deterministic_and_bounded():
    long = "abc " * 100000                       
    c1, c2 = chunks(long), chunks(long)
    assert c1 == c2 and len(c1) == MAX_CHUNKS and all(len(x) <= CHUNK_CHARS for x in c1)
    assert chunks("short text") == ["short text"] and chunks("") == [""]


def test_normalise_collapses_whitespace_and_newlines():
    assert normalise("a \t b\r\nc") == "a b\nc"


def test_tfidf_vocabulary_comes_from_training_only():
    rep = TfidfRep().fit(["alpha beta gamma"] * 5 + ["delta epsilon"] * 5)
    X = rep.transform(["alpha zeta zeta zeta"])            # 'zeta' unseen -> no column
    assert "zeta" not in rep.vec.vocabulary_ and X.shape[1] == len(rep.vec.vocabulary_)


def test_hand_features_shape():
    assert HandFeatures().transform(["x", "y y"]).shape[0] == 2
