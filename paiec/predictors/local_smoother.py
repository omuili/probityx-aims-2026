from __future__ import annotations
import hashlib
import numpy as np
from ..features.item_text import content_hash


def beta_mean(k: int, n: int, m: float = 2.0) -> float:
    return (m / 2 + k) / (m + n)


class EmbeddingLookup:
    def __init__(self, store: dict[str, np.ndarray]):
        self.store = store
    def get(self, content: str) -> np.ndarray | None:
        return self.store.get(content_hash(content))


def make_h3(lookup: EmbeddingLookup, tau: float, w0: float, m: float = 2.0, permute: bool = False):
    def predict(input: list[dict], labeled: list | None = None) -> float:
        subject, item = input
        benchmark = item.get("benchmark_id")
        obs = [(int(y), oi.get("item_content", "")) for (os_, oi), y in (labeled or []) if os_ == subject and oi.get("benchmark_id") == benchmark]
        n = len(obs); k = sum(y for y, _ in obs)
        p_b2 = beta_mean(k, n, m)
        if n == 0:
            return p_b2
        ys = np.array([y for y, _ in obs], float)
        if permute:
            key = hashlib.sha1((str(sorted(subject.items())) + benchmark + str(n)).encode()).digest()
            rng = np.random.default_rng(int.from_bytes(key[:8], "little")); ys = ys[rng.permutation(n)]
        e_t = lookup.get(item.get("item_content", ""))
        E = [lookup.get(c) for _, c in obs]
        if e_t is None or any(e is None for e in E):
            return p_b2
        E = np.stack(E); cos = E @ e_t / (np.linalg.norm(E, axis=1) * np.linalg.norm(e_t) + 1e-12)
        w = np.exp(-(1.0 - cos) / tau)
        return float(np.clip((w0 * p_b2 + w @ ys) / (w0 + w.sum()), 1e-9, 1 - 1e-9))
    return predict
