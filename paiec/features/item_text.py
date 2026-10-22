"""Step 1 of Phase 3: simple item-only features computed from item_content alone.

Only item_content is consistently available on every eligible benchmark: item_features is a
benchmark-specific key=value string (lang=..., website=..., paper=...) and is empty on SWE-rebench;
interactors is empty everywhere. So these features use the task text only. They are cheap, need no
vocabulary or fitted statistic, and are therefore leakage-free by construction; any standardization
happens later, inside the difficulty model, with training-fold statistics.
"""
from __future__ import annotations
import hashlib, math, re
import numpy as np

FEATURES = ["log_chars", "log_words", "log_lines", "avg_word_len", "frac_code_lines", "n_code_fences", "frac_digits",
            "frac_punct", "frac_upper", "frac_non_ascii", "n_urls", "n_questions", "n_paths", "has_error_words",
            "n_bullets", "has_latex", "has_html", "frac_blank_lines", "max_line_len_log"]

_URL = re.compile(r"https?://\S+"); _PATH = re.compile(r"(?:[\w.-]+/){1,}[\w.-]+\.\w{1,5}|\b\w+\.(?:py|js|ts|java|cpp|c|h|rs|go|rb|php|cs|sh|yaml|yml|json|toml|md)\b")
_ERR = re.compile(r"\b(Traceback|Error|Exception|error:|failed|FAILED|assert|panic|segfault|stack trace)\b")
_HTML = re.compile(r"</?[a-zA-Z][^>]*>"); _LATEX = re.compile(r"\\(documentclass|begin\{|section\{|usepackage)")
_BULLET = re.compile(r"(?m)^\s*(?:[-*+]|\d+[.)])\s")
_CODEISH = re.compile(r"^\s{4,}|^\t|[{};]\s*$|^\s*(def |class |import |from |return |if |for |while |#include|public |private )")


def _log1p(x: float) -> float: return math.log1p(max(0.0, x))


def item_features(text: str) -> np.ndarray:
    t = text or ""
    lines = t.split("\n"); n_lines = max(1, len(lines)); n_chars = len(t)
    words = t.split(); n_words = max(1, len(words))
    code_lines = sum(1 for l in lines if _CODEISH.search(l))
    fences = t.count("```")
    digits = sum(ch.isdigit() for ch in t); punct = sum((not ch.isalnum()) and (not ch.isspace()) for ch in t)
    upper = sum(ch.isupper() for ch in t); non_ascii = sum(ord(ch) > 127 for ch in t); alpha = max(1, sum(ch.isalpha() for ch in t))
    denom = max(1, n_chars)
    return np.array([
        _log1p(n_chars), _log1p(n_words), _log1p(n_lines), sum(len(w) for w in words) / n_words,
        code_lines / n_lines, fences, digits / denom, punct / denom, upper / alpha, non_ascii / denom,
        _log1p(len(_URL.findall(t))), _log1p(t.count("?")), _log1p(len(_PATH.findall(t))), float(bool(_ERR.search(t))),
        _log1p(len(_BULLET.findall(t))), float(bool(_LATEX.search(t))), float(bool(_HTML.search(t))),
        sum(1 for l in lines if not l.strip()) / n_lines, _log1p(max((len(l) for l in lines), default=0)),
    ], dtype=float)


def content_hash(text: str) -> str:
    return hashlib.sha1((text or "").encode("utf-8", "ignore")).hexdigest()
