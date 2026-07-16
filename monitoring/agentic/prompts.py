"""Prompt templates for the agentic monitors.

The Performance Supervisor Agent receives strategy telemetry and the
classical-detector firings visible as of the decision date, and must return a single
JSON object matching ``ASSESSMENT_JSON_SCHEMA``. Prompts are version-tagged so Week-4
prompt iteration can be tracked (VRI: "agent prompts iterated and version-controlled").
"""

from __future__ import annotations

import json

from .schemas import State, Action, NEWS_FLAGS_JSON_SCHEMA, OverallRisk

# v2: raw JSON-Schema dump replaced by a concrete example — small local models
# (llama3.2:3b) echoed the schema wrapper and nested the assessment under "properties".
SUPERVISOR_PROMPT_VERSION = "supervisor-v2"

SUPERVISOR_SYSTEM = f"""\
You are a Performance Supervisor monitoring a systematic trading strategy. Each day you
receive the strategy's recent performance telemetry and the outputs of classical
change-point detectors. Your job is to assess whether the strategy is behaving normally
or is undergoing a regime break, and to recommend an action.

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
"""


def build_supervisor_prompt(context: dict) -> tuple[str, str]:
    """Return (system_prompt, user_prompt) for the Performance Supervisor Agent.

    Args:
        context: causal context dict from ``guardrails.as_of_context``.

    Returns:
        (system, user) prompt strings.
    """
    user = (
        f"Decision date (as_of): {context['as_of']}\n\n"
        f"Strategy telemetry:\n{json.dumps(context['telemetry'], indent=2)}\n\n"
        f"Classical detector alarms visible so far:\n"
        f"{json.dumps(context['detector_alarms'], indent=2)}\n\n"
        f"Return your JSON assessment now."
    )
    return SUPERVISOR_SYSTEM, user


# v3 (Week 3): v2 plus optional macro-market and News Context Agent blocks.
SUPERVISOR_PROMPT_VERSION_V3 = "supervisor-v3"

SUPERVISOR_SYSTEM_V3 = SUPERVISOR_SYSTEM + """
You may additionally receive:
- A macro-market block (VIX, Treasury yields, fed funds, TED spread, and the latest
  macro releases AS THEY WERE KNOWN on the decision date).
- A structured summary from a News Context Agent that has read the filtered financial
  news published up to the decision date.
Weigh telemetry and classical detectors first; use news/macro context to confirm,
explain (root_cause), or discount them. Cite detectors in detectors_cited as before.
"""


def build_supervisor_prompt_v3(context: dict) -> tuple[str, str]:
    """v3: telemetry + detector alarms + optional macro block + optional news summary."""
    parts = [
        f"Decision date (as_of): {context['as_of']}",
        f"Strategy telemetry:\n{json.dumps(context['telemetry'], indent=2)}",
        f"Classical detector alarms visible so far:\n"
        f"{json.dumps(context['detector_alarms'], indent=2)}",
    ]
    if context.get("macro"):
        parts.append(f"Macro/market context (as known on the decision date):\n"
                     f"{json.dumps(context['macro'], indent=2)}")
    if context.get("news"):
        parts.append(f"News Context Agent summary:\n{json.dumps(context['news'], indent=2)}")
    parts.append("Return your JSON assessment now.")
    return SUPERVISOR_SYSTEM_V3, "\n\n".join(parts)


NEWS_PROMPT_VERSION = "news-context-v1"

NEWS_SYSTEM = f"""\
You are a News Context Agent supporting the monitoring of a systematic trading strategy.
You receive financial news headlines/summaries published on or before a decision date.
Your job is to extract MARKET-RISK-relevant context: what stress, if any, is the news
describing, and how severe is it for systematic strategies?

You may ONLY use the information provided in the user message. It reflects what was
published as of the stated decision date. Do not use any knowledge of what happened
after that date.

Respond with a SINGLE JSON object and nothing else. It must conform to this schema:

{json.dumps(NEWS_FLAGS_JSON_SCHEMA, indent=2)}

Field guidance:
- overall_risk: {OverallRisk.ALL} — the aggregate stress level the news describes.
- risk_flags: zero or more {{flag, evidence}} items; 'flag' is a short snake_case label
  (e.g. "forced_deleveraging", "credit_stress"), 'evidence' quotes or paraphrases the
  specific headlines supporting it.
- narrative: 2-4 sentences summarising the news-implied market state.
- confidence: your calibrated confidence, 0..1.
- as_of: echo the decision date exactly.
- n_articles: how many articles you were shown.
"""


def build_news_prompt(as_of: str, articles: list[dict], max_articles: int = 40) -> tuple[str, str]:
    """Return (system, user) prompts for the News Context Agent.

    Args:
        as_of: decision date (ISO).
        articles: dicts with date/ticker/title/summary — already timestamp-filtered
                  and future-date-scrubbed by the caller.
    """
    lines = []
    for a in articles[:max_articles]:
        tick = a.get("ticker") or "MARKET"
        summ = (a.get("summary") or "")[:280]
        lines.append(f"- [{a['date']}] ({tick}) {a['title']}" + (f" — {summ}" if summ else ""))
    body = "\n".join(lines) if lines else "(no risk-relevant articles passed the filter)"
    user = (
        f"Decision date (as_of): {as_of}\n\n"
        f"Filtered news published on or before the decision date "
        f"({min(len(articles), max_articles)} of {len(articles)} shown):\n{body}\n\n"
        f"Return your JSON assessment now."
    )
    return NEWS_SYSTEM, user
