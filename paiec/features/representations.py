"""Step 5 screening: candidate text representations for item difficulty, pre-registered set.

  hand19  : the 19 text statistics of item_text.py (step 2 reference)
  tfidf   : lexical TF-IDF, word 1-2 grams, sublinear tf; vocabulary fitted on TRAINING benchmarks only
  minilm  : frozen general semantic encoder  sentence-transformers/all-MiniLM-L6-v2 (384-d)
  bge     : frozen general-purpose encoder    BAAI/bge-base-en-v1.5 (768-d), no remote code

Deterministic preprocessing: whitespace-normalised text; for encoders, documents are split into
fixed 1,500-character chunks and up to MAX_CHUNKS chunks are taken at evenly spaced positions
(covering the whole document), embedded, and mean-pooled. No fine-tuning. Embeddings are cached on
disk per (model, benchmark) keyed by content hash so every fold reuses them; caching an embedding of
an item is not leakage because the encoder is frozen and sees no outcomes.
"""
from __future__ import annotations
import os, re
import numpy as np
from .item_text import item_features, content_hash

CHUNK_CHARS = 1500
MAX_CHUNKS = 32
ENCODERS = {"minilm": "sentence-transformers/all-MiniLM-L6-v2", "bge": "BAAI/bge-base-en-v1.5"}
CACHE = "data/cache"


def normalise(text: str) -> str:
    return re.sub(r"[ \t]+", " ", re.sub(r"\r\n?", "\n", text or "")).strip()


def chunks(text: str) -> list[str]:
    t = normalise(text)
    if not t:
        return [""]
    parts = [t[i:i + CHUNK_CHARS] for i in range(0, len(t), CHUNK_CHARS)]
    if len(parts) > MAX_CHUNKS:
        idx = np.linspace(0, len(parts) - 1, MAX_CHUNKS).round().astype(int)
        parts = [parts[i] for i in idx]
    return parts


class HandFeatures:
    name = "hand19"
    def fit(self, train_contents): return self
    def transform(self, contents): return np.stack([item_features(c) for c in contents])


class TfidfRep:
    name = "tfidf"
    def __init__(self, max_features: int = 50000):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=3, max_features=max_features, dtype=np.float32)
    def fit(self, train_contents):
        self.vec.fit([normalise(c)[:200000] for c in train_contents]); return self
    def transform(self, contents):
        return self.vec.transform([normalise(c)[:200000] for c in contents])   # sparse


class EncoderRep:
    def __init__(self, key: str, batch_size: int = 64):
        self.name, self.model_name, self.bs = key, ENCODERS[key], batch_size; self._model = None
    def fit(self, train_contents): return self
    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name, device="cpu")
        return self._model
    def transform(self, contents, cache_tag: str | None = None):
        os.makedirs(CACHE, exist_ok=True)
        path = os.path.join(CACHE, f"emb_{self.name}_{cache_tag}.npz") if cache_tag else None
        hashes = [content_hash(c) for c in contents]
        if path and os.path.exists(path):
            z = np.load(path, allow_pickle=True); store = dict(zip(z["hashes"].tolist(), z["emb"]))
        else:
            store = {}
        todo = [(h, c) for h, c in zip(hashes, contents) if h not in store]
        if todo:
            model = self._load(); flat, owner = [], []
            for h, c in todo:
                cs = chunks(c); flat += cs; owner += [h] * len(cs)
            E = model.encode(flat, batch_size=self.bs, normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True)
            owner = np.array(owner)
            for h, _ in todo:
                store[h] = E[owner == h].mean(axis=0)
            if path:
                np.savez(path, hashes=np.array(list(store.keys())), emb=np.stack(list(store.values())))
        return np.stack([store[h] for h in hashes])
