"""Sentiment scorers for the quantitative news-stress signal.

``FinBERTScorer`` mirrors ``FinBERTSentimentEngine`` in
``Stat Arb/statsArb-dev/src/fingpt_sentiment.py`` (same model, same
daily-stress definition) but is re-implemented here so ``monitoring/`` has no
import dependency on that subproject. It runs on **CPU by default**: headline
volumes per decision date are small, and this avoids VRAM contention with the
llama3.2:3b agent on the 4 GB laptop GPU.

``transformers``/``torch`` are imported lazily on first use — tests and CI use
``FakeScorer`` and never touch them.

LOOK-AHEAD BIAS WARNING: FinBERT was trained on financial corpora that may
include post-crisis text; the Week-4 with/without-events training control is the
project-level mitigation.
"""

from __future__ import annotations

import math
import re

FINBERT_MODEL = "ProsusAI/finbert"
_BATCH_SIZE = 16


class FakeScorer:
    """Deterministic keyword-based stand-in for tests (no ML dependencies).

    A headline is "net negative" iff it contains a word from a small negative
    set — enough to exercise the aggregator and triage deterministically.
    """

    _NEGATIVE = re.compile(
        r"loss|crash|plunge|panic|crisis|meltdown|default|sell-?off|margin call"
        r"|liquidat|bankrupt|turmoil|collapse|fear|slump",
        re.IGNORECASE,
    )

    def daily_stress(self, headlines: list[str]) -> float:
        if not headlines:
            return math.nan
        n_neg = sum(1 for h in headlines if self._NEGATIVE.search(h or ""))
        return n_neg / len(headlines)


class FinBERTScorer:
    """ProsusAI/finbert headline scorer (lazy-loaded, CPU by default)."""

    def __init__(self, model_name: str = FINBERT_MODEL, device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self._pipe = None

    def _ensure_loaded(self):
        if self._pipe is not None:
            return
        from transformers import (  # noqa: PLC0415 — deliberate lazy import
            BertForSequenceClassification, BertTokenizer, pipeline,
        )
        device_idx = 0 if self.device == "cuda" else -1
        tokenizer = BertTokenizer.from_pretrained(self.model_name)
        model = BertForSequenceClassification.from_pretrained(self.model_name)
        self._pipe = pipeline(
            "text-classification", model=model, tokenizer=tokenizer,
            device=device_idx, top_k=None, truncation=True, max_length=512,
        )

    def score_batch(self, headlines: list[str]) -> list[dict]:
        """Per-headline {"positive": p, "neutral": p, "negative": p}."""
        if not headlines:
            return []
        self._ensure_loaded()
        out = []
        for i in range(0, len(headlines), _BATCH_SIZE):
            for item in self._pipe(headlines[i:i + _BATCH_SIZE]):
                scores = {d["label"].lower(): d["score"] for d in item}
                out.append({
                    "positive": scores.get("positive", 0.0),
                    "neutral": scores.get("neutral", 0.0),
                    "negative": scores.get("negative", 0.0),
                })
        return out

    def daily_stress(self, headlines: list[str]) -> float:
        """Fraction of headlines that are net-negative (negative > positive).

        Same definition as FinBERTSentimentEngine.daily_stress. NaN if empty.
        """
        if not headlines:
            return math.nan
        scored = self.score_batch(headlines)
        n_neg = sum(1 for r in scored if r["negative"] > r["positive"])
        return n_neg / len(scored)
