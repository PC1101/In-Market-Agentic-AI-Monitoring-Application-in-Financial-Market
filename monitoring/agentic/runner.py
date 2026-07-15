"""Glue: turn a causal context into a validated agent assessment.

This is the Week-2 scaffold of the assessment loop the Week-3 agents will flesh out:
build prompt → call the local model → validate against the schema (with one repair
retry) → log. Information parity is enforced upstream by ``guardrails.as_of_context``.
"""

from __future__ import annotations

from .prompts import (
    build_supervisor_prompt, SUPERVISOR_PROMPT_VERSION,
    build_news_prompt, NEWS_PROMPT_VERSION, NEWS_PROMPT_VERSION_THINKING,
)
from .schemas import (
    validate_assessment, AgentAssessment, SchemaError,
    validate_news_context, NewsContextSummary,
)
from .model import LocalModel
from .guardrails import assert_no_lookahead


def run_supervisor(context: dict, model: LocalModel, logger=None,
                   latency_s: float | None = None,
                   extra_label: str | None = None,
                   prompt_builder=build_supervisor_prompt,
                   prompt_version: str = SUPERVISOR_PROMPT_VERSION) -> AgentAssessment:
    """Run the Performance Supervisor Agent on one causal context.

    Args:
        context: causal context from ``guardrails.as_of_context`` (already lookahead-safe).
        model:   a LocalModel implementation.
        logger:  optional RunLogger to record the invocation.
        latency_s: optional measured latency to record (callers time the model call).

    Returns:
        A validated AgentAssessment.

    Raises:
        SchemaError: if the model output cannot be validated after one repair attempt.
    """
    # Defensive re-check: never send future-dated context to the model.
    assert_no_lookahead(context, context["as_of"])

    system, user = prompt_builder(context)

    raw = model.complete(system, user)
    error = None
    try:
        assessment = validate_assessment(raw)
    except SchemaError as e:
        error = str(e)
        # One repair retry: re-ask with the validation error appended.
        repair_user = user + f"\n\nYour previous output was invalid: {e}\nReturn corrected JSON."
        raw = model.complete(system, repair_user)
        assessment = validate_assessment(raw)  # propagate if still invalid

    if logger is not None:
        extra = {"model": model.name}
        if extra_label is not None:
            extra["label"] = extra_label
        logger.log_invocation(
            agent="performance_supervisor",
            prompt_version=prompt_version,
            as_of=context["as_of"],
            raw_output=raw,
            assessment=assessment.to_dict(),
            error=error,
            latency_s=latency_s,
            extra=extra,
        )
    return assessment


def run_news_agent(news_block: dict, model: LocalModel, logger=None,
                   latency_s: float | None = None,
                   prompt_variant: str = "standard",
                   extra_label: str | None = None) -> NewsContextSummary:
    """Run the News Context Agent on one causal news block.

    Args:
        news_block: causal block from ``news.pipeline.build_news_block`` (already
                    T-1-cut and lookahead-checked at construction).
        model:      a LocalModel implementation.
        logger:     optional RunLogger to record the invocation.
        latency_s:  optional measured latency to record.
        prompt_variant: "standard" (CHEAP_MODEL) or "thinking" (THINKING_MODEL).

    Returns:
        A validated NewsContextSummary.

    Raises:
        SchemaError: if the model output cannot be validated after one repair
                     attempt. Callers map this (and transport errors) to the
                     CLASSICAL_ESCALATION triage mode.
    """
    # Defensive re-check: never send future-dated news to the model.
    assert_no_lookahead(news_block, news_block["as_of"])

    system, user = build_news_prompt(news_block, variant=prompt_variant)
    version = (NEWS_PROMPT_VERSION_THINKING if prompt_variant == "thinking"
               else NEWS_PROMPT_VERSION)

    raw = model.complete(system, user)
    error = None
    try:
        summary = validate_news_context(raw)
    except SchemaError as e:
        error = str(e)
        repair_user = user + f"\n\nYour previous output was invalid: {e}\nReturn corrected JSON."
        raw = model.complete(system, repair_user)
        summary = validate_news_context(raw)  # propagate if still invalid

    if logger is not None:
        extra = {"model": model.name, "triage_mode": news_block.get("triage_mode")}
        if extra_label is not None:
            extra["label"] = extra_label
        logger.log_invocation(
            agent="news_context",
            prompt_version=version,
            as_of=news_block["as_of"],
            raw_output=raw,
            assessment=summary.to_dict(),
            error=error,
            latency_s=latency_s,
            extra=extra,
        )
    return summary
