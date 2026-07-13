"""Local-model integration for the agentic layer.

``LocalModel`` is the pluggable interface every agent talks to. Two implementations ship:

  * ``OfflineStubModel`` — a deterministic, rule-based stand-in that reads the context and
    emits a schema-valid assessment WITHOUT any LLM. It lets the whole agentic pipeline
    (prompting → schema validation → logging → evaluation) run and be tested offline this
    week, before the real local model is wired in. It is NOT the research model — it is
    test/CI scaffolding.

  * ``OllamaModel`` — a thin client for a locally-served model (e.g. Ollama on
    localhost:11434). It is import-safe with no server running; ``complete`` only reaches
    out when actually called. This is where the Week-1 "local AI model" plugs in.

Both return a parsed dict; ``runner.run_supervisor`` validates it against the schema.
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod

from .schemas import State, Action


class LocalModel(ABC):
    """Interface for a local text-completion model returning JSON."""

    name: str = "local-model"

    @abstractmethod
    def complete(self, system: str, user: str) -> dict:
        """Return a parsed JSON object (the agent's assessment)."""
        raise NotImplementedError

    @staticmethod
    def _extract_json(text: str) -> dict:
        """Pull the first JSON object out of a model's raw text response."""
        text = text.strip()
        # Strip markdown code fences if present.
        text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end < start:
            raise ValueError(f"no JSON object found in model output: {text[:200]!r}")
        return json.loads(text[start : end + 1])


class OfflineStubModel(LocalModel):
    """Deterministic rule-based stand-in (no LLM). For offline pipeline tests."""

    name = "offline-stub"

    def complete(self, system: str, user: str) -> dict:
        ctx = self._parse_user(user)
        tel = ctx.get("telemetry", {})
        alarms = ctx.get("detector_alarms", {})
        n_fired = sum(1 for v in alarms.values() if v)

        dd = float(tel.get("current_drawdown", 0.0))
        worst = float(tel.get("recent_worst_day", 0.0))
        mean_ret = float(tel.get("recent_mean_daily_return", 0.0))

        # Simple monotone escalation from the visible evidence.
        severity = 0
        if n_fired >= 1 or dd <= -0.05 or worst <= -0.03:
            severity = 1
        if n_fired >= 2 or dd <= -0.10 or worst <= -0.05:
            severity = 2
        if n_fired >= 3 or dd <= -0.20:
            severity = 3

        state = [State.NORMAL, State.WATCH, State.ALERT, State.CRITICAL][severity]
        action = [Action.HOLD, Action.INVESTIGATE, Action.REDUCE, Action.HALT][severity]
        confidence = min(0.95, 0.4 + 0.15 * severity + 0.1 * min(n_fired, 3))

        cited = [k for k, v in alarms.items() if v]
        cause = (
            f"{n_fired} classical detector(s) fired; recent mean daily return "
            f"{mean_ret:+.4f}, drawdown {dd:+.2%}, worst day {worst:+.2%}."
        )
        return {
            "state": state,
            "action": action,
            "root_cause": cause,
            "confidence": round(confidence, 2),
            "as_of": ctx.get("as_of", ""),
            "detectors_cited": cited,
        }

    @staticmethod
    def _parse_user(user: str) -> dict:
        """Recover the structured context from the user prompt text."""
        ctx: dict = {}
        m = re.search(r"as_of\):\s*([0-9-]+)", user)
        if m:
            ctx["as_of"] = m.group(1)
        ctx["telemetry"] = _grab_json_after(user, "Strategy telemetry:")
        ctx["detector_alarms"] = _grab_json_after(user, "visible so far:")
        return ctx


def _grab_json_after(text: str, marker: str) -> dict:
    i = text.find(marker)
    if i == -1:
        return {}
    seg = text[i + len(marker):]
    start = seg.find("{")
    if start == -1:
        return {}
    depth = 0
    for j in range(start, len(seg)):
        if seg[j] == "{":
            depth += 1
        elif seg[j] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(seg[start : j + 1])
                except json.JSONDecodeError:
                    return {}
    return {}


class OllamaModel(LocalModel):
    """Thin client for a locally-served Ollama model. Import-safe; lazy network use.

    If ``json_schema`` is given it is passed as Ollama's ``format`` (structured
    outputs): decoding is grammar-constrained server-side, so the response is
    guaranteed to have the schema's shape. Small models otherwise tend to echo the
    schema wrapper instead of filling it in.
    """

    def __init__(self, model: str = "llama3.1", host: str = "http://localhost:11434",
                 temperature: float = 0.0, timeout: float = 300.0,
                 json_schema: dict | None = None):
        self.model = model
        self.host = host.rstrip("/")
        self.temperature = temperature
        self.timeout = timeout
        self.json_schema = json_schema
        self.name = f"ollama:{model}"

    def complete(self, system: str, user: str) -> dict:
        import urllib.request  # stdlib only; no hard dependency

        payload = {
            "model": self.model,
            "prompt": f"{system}\n\n{user}",
            "stream": False,
            "format": self.json_schema if self.json_schema is not None else "json",
            "options": {"temperature": self.temperature},
        }
        req = urllib.request.Request(
            f"{self.host}/api/generate",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            body = json.loads(resp.read().decode())
        return self._extract_json(body.get("response", ""))


def make_model(spec: str) -> LocalModel:
    """Build a LocalModel from a CLI spec: ``stub`` or ``ollama:<model-name>``.

    ``ollama:`` specs keep everything after the first colon as the model name, so
    tags work naturally (e.g. ``ollama:llama3.2:3b``). Ollama models are constrained
    to the supervisor assessment schema via structured outputs.
    """
    if spec == "stub":
        return OfflineStubModel()
    if spec.startswith("ollama:"):
        name = spec.split(":", 1)[1]
        if not name:
            raise ValueError("empty ollama model name; expected e.g. 'ollama:llama3.2:3b'")
        from .schemas import ASSESSMENT_JSON_SCHEMA
        return OllamaModel(model=name, json_schema=ASSESSMENT_JSON_SCHEMA)
    raise ValueError(f"unknown model spec {spec!r}; expected 'stub' or 'ollama:<name>'")
