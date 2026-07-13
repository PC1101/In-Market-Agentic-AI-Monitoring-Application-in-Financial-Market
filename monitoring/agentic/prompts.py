"""Prompt templates for the agentic monitors.

Week 2 ships the Performance Supervisor Agent prompt: it receives strategy telemetry and
the classical-detector firings visible as of the decision date, and must return a single
JSON object matching ``ASSESSMENT_JSON_SCHEMA``. Prompts are version-tagged so Week-4
prompt iteration can be tracked (VRI: "agent prompts iterated and version-controlled").
"""

from __future__ import annotations

import json

from .schemas import State, Action

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
