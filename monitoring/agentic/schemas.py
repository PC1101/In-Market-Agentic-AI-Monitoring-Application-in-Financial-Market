"""Structured-JSON output contract for the agentic monitors.

Every agent (the Performance Supervisor Agent this week, the News Context Agent in
Week 3) must return JSON matching this schema. Keeping the contract in one place lets us
(a) prompt the model with the exact schema, and (b) validate/repair its output before it
enters the evaluation pipeline. Validation is pure-Python (no ``jsonschema`` dependency).

The assessment fields:
  state       — regime assessment          (NORMAL | WATCH | ALERT | CRITICAL)
  action      — recommended response        (HOLD | REDUCE | HALT | INVESTIGATE)
  root_cause  — short free-text explanation
  confidence  — model confidence in [0, 1]
  as_of       — decision date (ISO 8601); must not post-date the information given
  detectors_cited — classical detectors the agent leaned on (may be empty)
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import date


class State:
    NORMAL = "NORMAL"
    WATCH = "WATCH"
    ALERT = "ALERT"
    CRITICAL = "CRITICAL"
    ALL = (NORMAL, WATCH, ALERT, CRITICAL)


class Action:
    HOLD = "HOLD"
    REDUCE = "REDUCE"
    HALT = "HALT"
    INVESTIGATE = "INVESTIGATE"
    ALL = (HOLD, REDUCE, HALT, INVESTIGATE)


class OverallRisk:
    LOW = "LOW"
    ELEVATED = "ELEVATED"
    HIGH = "HIGH"
    SEVERE = "SEVERE"
    ALL = (LOW, ELEVATED, HIGH, SEVERE)


#: JSON-Schema description embedded verbatim into prompts so the model knows the contract.
ASSESSMENT_JSON_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["state", "action", "root_cause", "confidence", "as_of"],
    "properties": {
        "state": {"type": "string", "enum": list(State.ALL)},
        "action": {"type": "string", "enum": list(Action.ALL)},
        "root_cause": {"type": "string", "minLength": 1, "maxLength": 1000},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "as_of": {"type": "string", "description": "ISO 8601 date (YYYY-MM-DD)"},
        "detectors_cited": {"type": "array", "items": {"type": "string"}},
    },
}


class SchemaError(ValueError):
    """Raised when an agent output fails validation."""


@dataclass
class AgentAssessment:
    """A validated agent assessment."""

    state: str
    action: str
    root_cause: str
    confidence: float
    as_of: str
    detectors_cited: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def validate_assessment(obj: dict) -> AgentAssessment:
    """Validate a raw dict against the assessment schema and return an AgentAssessment.

    Raises:
        SchemaError: on any missing field, wrong type, bad enum value, or out-of-range
                     confidence / malformed date.
    """
    if not isinstance(obj, dict):
        raise SchemaError(f"assessment must be a JSON object, got {type(obj).__name__}")

    required = ASSESSMENT_JSON_SCHEMA["required"]
    missing = [k for k in required if k not in obj]
    if missing:
        raise SchemaError(f"missing required field(s): {missing}")

    if obj["state"] not in State.ALL:
        raise SchemaError(f"invalid state {obj['state']!r}; allowed {State.ALL}")
    if obj["action"] not in Action.ALL:
        raise SchemaError(f"invalid action {obj['action']!r}; allowed {Action.ALL}")

    rc = obj["root_cause"]
    if not isinstance(rc, str) or not rc.strip():
        raise SchemaError("root_cause must be a non-empty string")

    conf = obj["confidence"]
    if not isinstance(conf, (int, float)) or isinstance(conf, bool):
        raise SchemaError("confidence must be a number")
    if not (0.0 <= float(conf) <= 1.0):
        raise SchemaError(f"confidence {conf} out of range [0, 1]")

    try:
        date.fromisoformat(obj["as_of"])
    except (ValueError, TypeError):
        raise SchemaError(f"as_of must be an ISO date (YYYY-MM-DD), got {obj['as_of']!r}")

    cited = obj.get("detectors_cited", [])
    if not isinstance(cited, list) or not all(isinstance(c, str) for c in cited):
        raise SchemaError("detectors_cited must be a list of strings")

    return AgentAssessment(
        state=obj["state"],
        action=obj["action"],
        root_cause=rc.strip(),
        confidence=float(conf),
        as_of=obj["as_of"],
        detectors_cited=list(cited),
    )


#: Output contract for the News Context Agent (Week 3).
NEWS_FLAGS_JSON_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["overall_risk", "risk_flags", "narrative", "confidence", "as_of"],
    "properties": {
        "overall_risk": {"type": "string", "enum": list(OverallRisk.ALL)},
        "risk_flags": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["flag", "evidence"],
                "properties": {
                    "flag": {"type": "string", "minLength": 1, "maxLength": 80},
                    "evidence": {"type": "string", "minLength": 1, "maxLength": 500},
                },
            },
        },
        "narrative": {"type": "string", "minLength": 1, "maxLength": 1500},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "as_of": {"type": "string", "description": "ISO 8601 date (YYYY-MM-DD)"},
        "n_articles": {"type": "integer", "minimum": 0},
    },
}


@dataclass
class NewsFlags:
    """A validated News Context Agent output."""

    overall_risk: str
    risk_flags: list[dict]
    narrative: str
    confidence: float
    as_of: str
    n_articles: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


def validate_news_flags(obj: dict) -> NewsFlags:
    """Validate a raw dict against NEWS_FLAGS_JSON_SCHEMA (same style as
    ``validate_assessment``)."""
    if not isinstance(obj, dict):
        raise SchemaError(f"news flags must be a JSON object, got {type(obj).__name__}")

    missing = [k for k in NEWS_FLAGS_JSON_SCHEMA["required"] if k not in obj]
    if missing:
        raise SchemaError(f"missing required field(s): {missing}")

    if obj["overall_risk"] not in OverallRisk.ALL:
        raise SchemaError(f"invalid overall_risk {obj['overall_risk']!r}; allowed {OverallRisk.ALL}")

    flags = obj["risk_flags"]
    if not isinstance(flags, list):
        raise SchemaError("risk_flags must be a list")
    for f in flags:
        if (not isinstance(f, dict) or not isinstance(f.get("flag"), str)
                or not f["flag"].strip() or not isinstance(f.get("evidence"), str)
                or not f["evidence"].strip()):
            raise SchemaError(f"each risk flag needs non-empty 'flag' and 'evidence': {f!r}")

    narrative = obj["narrative"]
    if not isinstance(narrative, str) or not narrative.strip():
        raise SchemaError("narrative must be a non-empty string")

    conf = obj["confidence"]
    if not isinstance(conf, (int, float)) or isinstance(conf, bool):
        raise SchemaError("confidence must be a number")
    if not (0.0 <= float(conf) <= 1.0):
        raise SchemaError(f"confidence {conf} out of range [0, 1]")

    if obj["as_of"] != "XXXX-XX-XX":  # allow Condition-B masked sentinel
        try:
            date.fromisoformat(obj["as_of"])
        except (ValueError, TypeError):
            raise SchemaError(f"as_of must be an ISO date, got {obj['as_of']!r}")

    n_articles = obj.get("n_articles", 0)
    if not isinstance(n_articles, int) or isinstance(n_articles, bool) or n_articles < 0:
        raise SchemaError("n_articles must be a non-negative integer")

    return NewsFlags(
        overall_risk=obj["overall_risk"],
        risk_flags=[{"flag": f["flag"].strip(), "evidence": f["evidence"].strip()} for f in flags],
        narrative=narrative.strip(),
        confidence=float(conf),
        as_of=obj["as_of"],
        n_articles=n_articles,
    )
