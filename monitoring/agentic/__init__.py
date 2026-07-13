"""Agentic monitoring framework scaffold (Week 2).

Provides the pieces the Week-3 agents are built on:
  * ``schemas``   — the structured-JSON assessment contract + a validator.
  * ``prompts``   — prompt templates for the Performance Supervisor Agent.
  * ``guardrails``— information-parity guards (as-of dating, timestamp filtering).
  * ``model``     — pluggable local-model client (+ an offline stub for testing).
  * ``logging_utils`` — JSONL run logger.
  * ``runner``    — glue that turns a context into a validated assessment.
"""

from .schemas import (
    AgentAssessment, ASSESSMENT_JSON_SCHEMA, State, Action, validate_assessment,
    NewsFlags, OverallRisk, validate_news_flags,
)
from .guardrails import (
    as_of_context, assert_no_lookahead, filter_news_by_timestamp, scrub_future_dated,
)
from .model import LocalModel, OfflineStubModel, default_model
from .logging_utils import RunLogger
from .runner import run_supervisor
from .news_agent import run_news_agent

__all__ = [
    "AgentAssessment",
    "ASSESSMENT_JSON_SCHEMA",
    "State",
    "Action",
    "validate_assessment",
    "NewsFlags",
    "OverallRisk",
    "validate_news_flags",
    "as_of_context",
    "assert_no_lookahead",
    "filter_news_by_timestamp",
    "scrub_future_dated",
    "LocalModel",
    "OfflineStubModel",
    "default_model",
    "RunLogger",
    "run_supervisor",
    "run_news_agent",
]
