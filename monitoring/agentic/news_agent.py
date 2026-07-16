"""News Context Agent v1: filtered news in, validated risk flags out.

Mirror of ``runner.run_supervisor``: guardrails → prompt → local model →
schema validation (one repair retry) → JSONL log.
"""

from __future__ import annotations

from .guardrails import assert_no_lookahead, filter_news_by_timestamp, scrub_future_dated
from .model import LocalModel
from .prompts import build_news_prompt, NEWS_PROMPT_VERSION
from .schemas import validate_news_flags, NewsFlags, SchemaError, NEWS_FLAGS_JSON_SCHEMA


def run_news_agent(as_of: str, articles: list[dict], model: LocalModel,
                   logger=None, latency_s: float | None = None,
                   extra_label: str | None = None, max_articles: int = 40) -> NewsFlags:
    """Run the News Context Agent for one decision date.

    Args:
        as_of: decision date (ISO string).
        articles: dicts with keys date/ticker/title/summary; publication-date
                  filtering and future-text scrubbing are (re-)applied here,
                  fail-closed, regardless of what the caller did.
    """
    articles = filter_news_by_timestamp(articles, as_of, ts_field="date")
    articles = scrub_future_dated(articles, as_of)
    assert_no_lookahead({"as_of": as_of, "articles": articles}, as_of)

    system, user = build_news_prompt(as_of, articles, max_articles=max_articles)
    raw = model.complete(system, user, json_schema=NEWS_FLAGS_JSON_SCHEMA)
    error = None
    try:
        flags = validate_news_flags(raw)
    except SchemaError as e:
        error = str(e)
        repair = user + f"\n\nYour previous output was invalid: {e}\nReturn corrected JSON."
        raw = model.complete(system, repair, json_schema=NEWS_FLAGS_JSON_SCHEMA)
        flags = validate_news_flags(raw)

    # The model may not have been shown every article (max_articles cap) and may
    # miscount; pin the audited number.
    flags.n_articles = len(articles[:max_articles])

    if logger is not None:
        extra = {"model": model.name}
        if extra_label is not None:
            extra["label"] = extra_label
        logger.log_invocation(
            agent="news_context",
            prompt_version=NEWS_PROMPT_VERSION,
            as_of=as_of,
            raw_output=raw,
            assessment=flags.to_dict(),
            error=error,
            latency_s=latency_s,
            extra=extra,
        )
    return flags
