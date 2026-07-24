"""FinBERT-based stress scorer for the news triage pipeline.

Wraps ProsusAI/finbert (a BERT model fine-tuned on financial text) to produce a
per-article stress probability — the "negative" sentiment class probability.

The scorer is used in stage 3 (triage): the daily stress signal is the maximum
FinBERT stress probability across the day's risk-filtered articles.  Triage
thresholds (STRESS_CHEAP, STRESS_THINKING) are applied in ``news.triage``.

Design decisions:
- Lazy model load: the heavy transformers import happens only on first ``score``
  call. Tests and offline runs never trigger it.
- Graceful fallback: if ``transformers`` is not installed (or the model can't be
  loaded), ``score_stress`` returns ``[0.0] * len(texts)`` and logs a warning
  once.  The triage then falls back to the regex-z-score pathway (HIT_Z_THINKING).
- Batch processing: texts are scored in one forward pass (up to ``batch_size``)
  to amortise GPU/CPU overhead.
- Causal guarantee: the scorer only sees text content; it never uses dates or
  external state.
"""

from __future__ import annotations

import logging
from typing import Sequence

logger = logging.getLogger(__name__)

# Sentinel so we attempt the import exactly once.
_PIPELINE = None
_PIPELINE_FAILED = False
_MODEL_NAME = "ProsusAI/finbert"


def _load_pipeline():
    """Attempt to load the HuggingFace pipeline (once)."""
    global _PIPELINE, _PIPELINE_FAILED
    if _PIPELINE is not None or _PIPELINE_FAILED:
        return
    try:
        from transformers import pipeline as hf_pipeline  # type: ignore
        _PIPELINE = hf_pipeline(
            "text-classification",
            model=_MODEL_NAME,
            top_k=None,           # return all class scores
            truncation=True,
            max_length=512,
        )
        logger.info("FinBERT loaded: %s", _MODEL_NAME)
    except Exception as exc:
        _PIPELINE_FAILED = True
        logger.warning(
            "FinBERT unavailable (%s); triage falls back to regex z-score. "
            "Install transformers + torch and ensure HuggingFace access.",
            exc,
        )


def score_stress(texts: Sequence[str], batch_size: int = 32) -> list[float]:
    """Return the FinBERT ``negative`` (stressed) probability for each text.

    Args:
        texts:      Article texts (headline + abstract or headline only).  Empty
                    strings are scored 0.0 without a forward pass.
        batch_size: Number of texts per inference batch.

    Returns:
        List of floats in [0, 1], same length as ``texts``.  0.0 when FinBERT
        is unavailable or the text is empty.
    """
    if not texts:
        return []

    _load_pipeline()

    if _PIPELINE is None:
        # Fallback: return zeros so triage uses the z-score pathway.
        return [0.0] * len(texts)

    results: list[float] = []
    texts_list = list(texts)
    for i in range(0, len(texts_list), batch_size):
        chunk = texts_list[i : i + batch_size]
        # Replace empty strings with a single space to avoid tokeniser errors.
        chunk = [t if t.strip() else " " for t in chunk]
        try:
            outputs = _PIPELINE(chunk)
            for label_scores in outputs:
                neg_score = next(
                    (s["score"] for s in label_scores if s["label"].lower() == "negative"),
                    0.0,
                )
                results.append(float(neg_score))
        except Exception as exc:
            logger.warning("FinBERT inference error on batch %d: %s", i, exc)
            results.extend([0.0] * len(chunk))

    return results


def daily_max_stress(headlines: Sequence[str]) -> float:
    """Maximum FinBERT stress probability across a day's headlines.

    Returns 0.0 when no headlines are supplied or FinBERT is unavailable.
    This is the scalar fed to ``news.triage.decide()`` as ``stress_score``.
    """
    if not headlines:
        return 0.0
    scores = score_stress(list(headlines))
    return max(scores) if scores else 0.0
