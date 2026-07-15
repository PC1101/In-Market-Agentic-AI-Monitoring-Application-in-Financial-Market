"""Prompt templates for the agentic monitors.

Week 2 ships the Performance Supervisor Agent prompt: it receives strategy telemetry and
the classical-detector firings visible as of the decision date, and must return a single
JSON object matching ``ASSESSMENT_JSON_SCHEMA``. Prompts are version-tagged so Week-4
prompt iteration can be tracked (VRI: "agent prompts iterated and version-controlled").
"""

from __future__ import annotations

import json

from .schemas import State, Action, RiskFlag, NewsIntensity

# v2: raw JSON-Schema dump replaced by a concrete example — small local models
# (llama3.2:3b) echoed the schema wrapper and nested the assessment under "properties".
# v3: optional news-context block appended to the user prompt (Week 3); the output
# schema is unchanged, so v2 outputs remain valid.
SUPERVISOR_PROMPT_VERSION = "supervisor-v3"

SUPERVISOR_SYSTEM = f"""\
You are a Performance Supervisor monitoring a systematic trading strategy. Each day you
receive the strategy's recent performance telemetry, the outputs of classical
change-point detectors, and (when available) a structured summary of financial-news
context prepared from headlines published before the decision date. Your job is to
assess whether the strategy is behaving normally or is undergoing a regime break, and
to recommend an action.

You may ONLY use the information provided in the user message. It reflects what was known
as of the stated decision date. Do not speculate about future events or use knowledge of
what happened after the decision date.

Respond with a SINGLE flat JSON object and nothing else — no schema, no wrapper, no
markdown. It must have exactly these fields, like this example:

{{
  "state": "WATCH",
  "action": "INVESTIGATE",
  "root_cause": "One or two sentences naming the most likely driver.",
  "confidence": 0.7,
  "as_of": "2008-09-15",
  "detectors_cited": ["page_hinkley"]
}}

Field guidance:
- state: one of {State.ALL} — escalate as evidence of a break accumulates.
- action: one of {Action.ALL} — HOLD when normal; REDUCE/HALT as risk rises; INVESTIGATE
  when signals are ambiguous.
- root_cause: one or two sentences naming the most likely driver.
- confidence: your calibrated confidence in this assessment, 0..1.
- as_of: echo the decision date exactly.
- detectors_cited: list the detector names whose firings informed you (may be empty).
  You may also cite "news_context" if the news summary materially informed you.
"""


def build_supervisor_prompt(context: dict) -> tuple[str, str]:
    """Return (system_prompt, user_prompt) for the Performance Supervisor Agent.

    Args:
        context: causal context dict from ``guardrails.as_of_context``.

    Returns:
        (system, user) prompt strings.
    """
    news = context.get("news")
    if news is not None:
        news_part = f"News context summary:\n{json.dumps(news, indent=2)}\n\n"
    else:
        news_part = "News context summary: none available\n\n"

    user = (
        f"Decision date (as_of): {context['as_of']}\n\n"
        f"Strategy telemetry:\n{json.dumps(context['telemetry'], indent=2)}\n\n"
        f"Classical detector alarms visible so far:\n"
        f"{json.dumps(context['detector_alarms'], indent=2)}\n\n"
        f"{news_part}"
        f"Return your JSON assessment now."
    )
    return SUPERVISOR_SYSTEM, user


# --------------------------------------------------------------------------- #
# News Context Agent (Week 3)                                                  #
# --------------------------------------------------------------------------- #

# Same lesson as supervisor-v2: concrete example object, never a schema dump.
NEWS_PROMPT_VERSION = "news-context-v1"
NEWS_PROMPT_VERSION_THINKING = "news-context-v1-thinking"

NEWS_SYSTEM = f"""\
You are a News Context analyst supporting the monitoring of a systematic trading
strategy. Each day you receive an aggregate news-stress signal and a list of
filtered financial-news headlines published on or before the day BEFORE the
decision date. Your job is to summarise what the news says about market stress
relevant to systematic strategies, as structured risk flags plus a short narrative.

You may ONLY use the headlines and signal provided in the user message. They
reflect what was published as of the stated cutoff. Do not use any knowledge of
events after the decision date, and do not speculate about what happens next.

Respond with a SINGLE flat JSON object and nothing else — no schema, no wrapper,
no markdown. It must have exactly these fields, like this example:

{{
  "as_of": "2008-09-15",
  "risk_flags": ["CREDIT_EVENT", "MARKET_SELLOFF"],
  "narrative": "One to three sentences summarising the stress picture in the provided headlines.",
  "news_intensity": "HIGH",
  "confidence": 0.8,
  "headlines_cited": ["Exact text of a provided headline you relied on"]
}}

Field guidance:
- as_of: echo the decision date exactly.
- risk_flags: one or more of {RiskFlag.ALL}. Use NONE alone (and no other flag)
  when the headlines show no relevant stress. Category hints from the keyword
  filter: liquidity_credit -> LIQUIDITY_STRESS or CREDIT_EVENT;
  market_stress -> MARKET_SELLOFF; fund_quant_stress -> FUND_STRESS;
  macro_policy -> POLICY_INTERVENTION, RATING_ACTION, or MACRO_DETERIORATION.
- narrative: at most 600 characters, grounded ONLY in the provided headlines.
- news_intensity: one of {NewsIntensity.ALL} — LOW for routine coverage,
  ELEVATED for clear stress themes, HIGH for broad or acute stress.
- confidence: your calibrated confidence in this summary, 0..1.
- headlines_cited: verbatim headlines (from those provided) supporting the flags;
  may be empty when risk_flags is [\"NONE\"].
"""

_NEWS_THINKING_SUFFIX = """

Before deciding, reason step by step INSIDE the narrative construction (do not
output your reasoning separately): first identify which stress categories the
headlines support, then weigh how recent and how concentrated they are, then set
news_intensity and confidence accordingly. Output remains ONLY the single JSON
object described above.
"""


def build_news_prompt(news_block: dict, variant: str = "standard") -> tuple[str, str]:
    """Return (system_prompt, user_prompt) for the News Context Agent.

    Args:
        news_block: causal block from ``news.pipeline.build_news_block``.
        variant:    "standard" or "thinking" (extended-reasoning suffix, used by
                    the THINKING_MODEL triage mode).

    Returns:
        (system, user) prompt strings.
    """
    if variant not in ("standard", "thinking"):
        raise ValueError(f"unknown news prompt variant {variant!r}")
    system = NEWS_SYSTEM + (_NEWS_THINKING_SUFFIX if variant == "thinking" else "")

    headlines = news_block.get("headlines", [])
    lines = [
        f"- [{h['date']}] ({', '.join(h.get('categories', []))}) {h['headline']}"
        for h in headlines
    ] or ["(none matched the stress filter in the lookback window)"]

    user = (
        f"Decision date (as_of): {news_block['as_of']}\n"
        f"News cutoff (last visible publication day): {news_block['cutoff']}\n\n"
        f"Aggregate news signal:\n{json.dumps(news_block['signal'], indent=2)}\n\n"
        f"Coverage note: {news_block.get('coverage_note', 'n/a')}\n\n"
        f"Filtered headlines (most recent last):\n" + "\n".join(lines) + "\n\n"
        f"Return your JSON news-context summary now."
    )
    return system, user
