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
    NewsContextSummary, NEWS_CONTEXT_JSON_SCHEMA, RiskFlag, NewsIntensity,
    validate_news_context, SchemaError,
)
from .guardrails import as_of_context, assert_no_lookahead, filter_news_by_timestamp
from .model import LocalModel, OfflineStubModel, OllamaModel, make_model
from .logging_utils import RunLogger
from .runner import run_supervisor, run_news_agent

__all__ = [
    "AgentAssessment",
    "ASSESSMENT_JSON_SCHEMA",
    "State",
    "Action",
    "validate_assessment",
    "NewsContextSummary",
    "NEWS_CONTEXT_JSON_SCHEMA",
    "RiskFlag",
    "NewsIntensity",
    "validate_news_context",
    "SchemaError",
    "as_of_context",
    "assert_no_lookahead",
    "filter_news_by_timestamp",
    "LocalModel",
    "OfflineStubModel",
    "OllamaModel",
    "make_model",
    "RunLogger",
    "run_supervisor",
    "run_news_agent",
]
