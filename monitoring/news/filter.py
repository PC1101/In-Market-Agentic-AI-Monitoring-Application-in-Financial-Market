"""Stage 1 of the news pipeline: cheap regex/keyword risk filter.

A hand-built lexicon of market-stress language. Deliberately transparent (it goes
in the methodology write-up) and deliberately recall-oriented — stage 2 (the
quantitative aggregator) and stage 3 (triage + LLM) handle precision.
"""

from __future__ import annotations

import re

import pandas as pd

#: (label, pattern) — label is what appears in risk_terms lists / agent context.
RISK_PATTERNS: list[tuple[str, str]] = [
    ("sell-off/selloff", r"sell[- ]?off"),
    ("meltdown", r"meltdown"),
    ("liquidation", r"liquidat\w*"),
    ("margin call", r"margin call"),
    ("deleveraging", r"deleverag\w*"),
    ("unwind", r"unwind\w*"),
    ("crash", r"crash\w*"),
    ("plunge", r"plung\w*"),
    ("turmoil", r"turmoil"),
    ("contagion", r"contagion"),
    ("credit crunch", r"credit crunch"),
    ("subprime", r"subprime"),
    ("default", r"default\w*"),
    ("bailout", r"bail[- ]?out\w*"),
    ("bankruptcy", r"bankrupt\w*"),
    ("downgrade", r"downgrad\w*"),
    ("recession", r"recession\w*"),
    ("volatility spike", r"volatil\w*"),
    ("panic", r"panic\w*"),
    ("crisis", r"cris[ie]s"),
    ("quant fund stress", r"quant\w*\s+(?:fund|strateg|model|trading)"),
    ("hedge fund stress", r"hedge[- ]fund\w*\s+(?:loss|losses|collapse|redemption|blow)"),
]

_COMPILED = [(label, re.compile(pat, re.IGNORECASE)) for label, pat in RISK_PATTERNS]

#: Matches any Cyrillic character (pre-2009 FNSPID is ~26% Russian Lenta.ru content).
_CYRILLIC = re.compile(r"[\u0400-\u04FF]")


def is_non_english(text: str, threshold: float = 0.3) -> bool:
    """True if more than ``threshold`` fraction of alphabetic chars are Cyrillic.

    Used to detect pre-2009 Russian articles that were scraped into FNSPID along
    with English financial news. The English risk lexicon produces zero hits on
    Cyrillic text, so these rows are harmless to precision but bloat the corpus
    and distort article-count metrics.
    """
    if not isinstance(text, str) or not text:
        return False
    alpha = [c for c in text if c.isalpha()]
    if len(alpha) < 5:
        return False
    cyrillic = sum(1 for c in alpha if _CYRILLIC.match(c))
    return cyrillic / len(alpha) > threshold


def filter_non_english(df: pd.DataFrame,
                       text_cols: tuple[str, ...] = ("title", "summary")) -> pd.DataFrame:
    """Drop rows where the concatenated text is predominantly non-English (Cyrillic).

    Call this before ``filter_news`` to reduce noise from pre-2009 Russian scrape.
    The filter is opt-in — existing callers are unaffected.
    """
    if df.empty:
        return df
    joined = df[list(text_cols)].fillna("").agg(" ".join, axis=1)
    mask = ~joined.map(is_non_english)
    return df[mask].reset_index(drop=True)


def score_text(text: str) -> list[str]:
    """Return the distinct risk-term labels matched in ``text``."""
    if not isinstance(text, str) or not text:
        return []
    return [label for label, rx in _COMPILED if rx.search(text)]


def filter_news(df: pd.DataFrame, text_cols=("title", "summary")) -> pd.DataFrame:
    """Keep only risk-relevant articles; add risk_terms / n_risk_terms columns."""
    out = df.copy()
    if out.empty:
        out["risk_terms"] = pd.Series(dtype=object)
        out["n_risk_terms"] = pd.Series(dtype=int)
        return out
    joined = out[list(text_cols)].fillna("").agg(" ".join, axis=1)
    out["risk_terms"] = joined.map(score_text)
    out["n_risk_terms"] = out["risk_terms"].map(len)
    return out[out["n_risk_terms"] > 0].reset_index(drop=True)
