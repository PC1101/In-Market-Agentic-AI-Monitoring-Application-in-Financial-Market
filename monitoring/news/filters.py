"""Regex/keyword stress filter — first stage of the news pipeline.

The lexicon is organised by risk category; a headline's matched categories feed
directly into the News Context Agent's ``risk_flags`` vocabulary.

HINDSIGHT-BIAS NOTE (methodological, do not weaken): the lexicon contains only
*generic* financial-distress terms. It deliberately names **no institutions,
funds, or people** (e.g. no "Lehman", no "Bear Stearns") — a keyword list written
in 2026 that names the actors of 2007-2011 would leak hindsight into the
information set. Terms were cross-checked against the tone of the
Loughran-McDonald financial negative-word list rather than against the six
evaluation windows.
"""

from __future__ import annotations

import re
from functools import lru_cache

# category -> list of regex patterns (compiled case-insensitively below).
STRESS_LEXICON: dict[str, list[str]] = {
    "liquidity_credit": [
        r"liquidat\w+",
        r"margin calls?",
        r"credit (crunch|squeeze|freeze|crisis|concern|pressure|worry|worries)",
        r"defaults?\b",
        r"insolven\w+",
        r"bankrupt\w+",
        r"bail-?outs?",
        r"contagion",
        r"write-?downs?",
        r"counter-?party",
        r"frozen (credit|market|fund)",
    ],
    "market_stress": [
        r"sell-?offs?",
        r"plunges?|plummet\w*|tumbles?|routs?\b",
        r"crash\w*",
        r"collapse\w*",
        r"panic\w*",
        r"turmoil",
        r"meltdown",
        r"volatil\w+",
        r"bear market",
        r"free-?fall",
        r"stocks? (dive|sink|slide|slump)\w*",
    ],
    "fund_quant_stress": [
        r"hedge funds?",
        r"quant\w*",
        r"redemptions?",
        r"de-?leverag\w+",
        r"unwind\w*",
        r"(halt|suspend)\w* (redemption|withdrawal)s?",
        r"forced (selling|sales)",
        r"fund (losses|freeze|failure)s?",
    ],
    "macro_policy": [
        r"recessions?",
        r"downgrades?",
        r"rating (cut|action|agenc)\w*",
        r"subprime",
        r"mortgage-?backed",
        r"emergency (rate|meeting|lending|loan|measure)s?",
        r"(inject\w*|intervene\w*|intervention).{0,20}(liquidity|market|bank)",
        r"central bank\w*",
        r"sovereign debt",
        r"debt ceiling",
    ],
}

_COMPILED: dict[str, re.Pattern] = {
    cat: re.compile("|".join(f"(?:{p})" for p in pats), re.IGNORECASE)
    for cat, pats in STRESS_LEXICON.items()
}

# NYT sections/desks considered financial by default. A stress-keyword hit in
# ANY section still qualifies (crises reach the front page).
FINANCIAL_SECTIONS = {
    "business", "business day", "your money", "the markets", "financial",
    "dealbook", "economy", "small business", "mutual funds", "market place",
}


@lru_cache(maxsize=500_000)
def _classify_cached(text: str) -> tuple[str, ...]:
    return tuple(sorted(cat for cat, rx in _COMPILED.items() if rx.search(text)))


def classify_headline(text: str) -> list[str]:
    """Return the sorted list of stress categories the text matches (may be []).

    Cached: the daily triage loop re-classifies the same record for every
    decision date whose lookback window covers it.
    """
    return list(_classify_cached(text or ""))


def _searchable_text(record: dict) -> str:
    parts = [record.get("headline") or ""]
    if record.get("abstract"):
        parts.append(record["abstract"])
    return " ".join(parts)


def split_records(records: list[dict]) -> tuple[list[dict], list[dict]]:
    """Split records into (candidates, matched).

    candidates: financial-section records OR any record with a keyword hit —
                the denominator for daily article counts.
    matched:    records with >=1 stress-keyword hit; each gains ``categories``
                (list) and ``n_hit_categories`` (int) keys. Subset of candidates.
    """
    candidates: list[dict] = []
    matched: list[dict] = []
    for r in records:
        cats = classify_headline(_searchable_text(r))
        section = (r.get("section") or "").lower()
        is_financial = section in FINANCIAL_SECTIONS
        if cats:
            r = dict(r, categories=cats, n_hit_categories=len(cats))
            matched.append(r)
            candidates.append(r)
        elif is_financial or not section:
            # section-less sources (e.g. FNSPID wire titles) are all candidates
            candidates.append(r)
    return candidates, matched
