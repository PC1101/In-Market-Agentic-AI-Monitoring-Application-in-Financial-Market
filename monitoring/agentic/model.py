"""Local-model integration for the agentic layer.

``LocalModel`` is the pluggable interface every agent talks to. Two implementations ship:

  * ``OfflineStubModel`` — a deterministic, rule-based stand-in that reads the context and
    emits a schema-valid assessment WITHOUT any LLM. It lets the whole agentic pipeline
    (prompting → schema validation → logging → evaluation) run and be tested offline,
    and remains the CI model. It is NOT the research model — it is test scaffolding.

  * ``OllamaModel`` — a thin client for a locally-served model (e.g. Ollama on
    localhost:11434). It is import-safe with no server running; ``complete`` only reaches
    out when actually called.

Both return a parsed dict; the runners validate it against the relevant schema.

Structured outputs: ``complete`` takes an optional per-call ``json_schema``. For Ollama
it is passed as the ``format`` (grammar-constrained decoding server-side), so the
response is guaranteed to have the schema's shape — small models otherwise tend to echo
the schema wrapper instead of filling it in. Per-call (rather than pinned at
construction) because one model instance serves BOTH the supervisor and the news agent,
which have different output contracts.
"""

from __future__ import annotations

import json
import os
import re
from abc import ABC, abstractmethod

from .schemas import State, Action


def _extract_as_of(text: str) -> str:
    """Pull the ``as_of`` date out of a rendered user prompt (shared by the
    supervisor and news-mode branches of ``OfflineStubModel``).
    Returns the Condition-B masked sentinel when found."""
    if "XXXX-XX-XX" in text:
        return "XXXX-XX-XX"
    m = re.search(r"as_of\):\s*([0-9-]+)", text)
    return m.group(1) if m else ""


class LocalModel(ABC):
    """Interface for a local text-completion model returning JSON."""

    name: str = "local-model"

    @abstractmethod
    def complete(self, system: str, user: str, json_schema: dict | None = None) -> dict:
        """Return a parsed JSON object (the agent's assessment).

        Args:
            json_schema: optional output contract for structured decoding;
                implementations that cannot constrain decoding may ignore it.
        """
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
    """Deterministic rule-based stand-in (no LLM). For offline pipeline tests.

    Stands in for BOTH the Performance Supervisor Agent and the News Context Agent,
    dispatching on which system prompt it is given.
    """

    name = "offline-stub"

    def complete(self, system: str, user: str, json_schema: dict | None = None) -> dict:
        # Dispatch on the role declaration, not a bare substring: the supervisor-v3
        # system prompt also *mentions* the News Context Agent.
        if "You are a News Context Agent" in system:
            return self._complete_news(user)
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
    def _complete_news(user: str) -> dict:
        """News-mode branch: derive a schema-valid ``NewsFlags`` payload from the
        rendered news-agent user prompt, without any LLM."""
        n_articles = user.count("\n- [")
        risky = any(t in user.lower() for t in ("meltdown", "crash", "liquidat", "crisis"))
        risk = "HIGH" if (risky and n_articles >= 3) else ("ELEVATED" if risky else "LOW")
        return {
            "overall_risk": risk,
            "risk_flags": (
                [{"flag": "stress_language_in_news", "evidence": "risk terms present in headlines"}]
                if risky else []
            ),
            "narrative": f"Reviewed {n_articles} filtered articles; "
                         + ("stress language present." if risky else "no stress language."),
            "confidence": 0.5,
            "as_of": _extract_as_of(user),
            "n_articles": n_articles,
        }

    @staticmethod
    def _parse_user(user: str) -> dict:
        """Recover the structured context from the user prompt text."""
        ctx: dict = {}
        as_of = _extract_as_of(user)
        if as_of:
            ctx["as_of"] = as_of
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

    A ``json_schema`` (per ``complete`` call, or as a constructor default) is passed
    as Ollama's ``format`` (structured outputs): decoding is grammar-constrained
    server-side, so the response is guaranteed to have the schema's shape.
    """

    def __init__(self, model: str = "qwen2.5:1.5b", host: str = "http://localhost:11434",
                 temperature: float = 0.0, timeout: float = 1800.0,
                 json_schema: dict | None = None):
        # 1800 s: GFC/mc09 windows have 5-8k risk articles; combined news+supervisor
        # prompts on those days can cause slow inference when GPU drops to Eco mode.
        self.model = model
        self.host = host.rstrip("/")
        self.temperature = temperature
        self.timeout = timeout
        self.json_schema = json_schema
        self.name = f"ollama:{model}"

    def complete(self, system: str, user: str, json_schema: dict | None = None) -> dict:
        import urllib.request  # stdlib only; no hard dependency

        schema = json_schema if json_schema is not None else self.json_schema
        payload = {
            "model": self.model,
            "prompt": f"{system}\n\n{user}",
            "stream": False,
            "format": schema if schema is not None else "json",
            "options": {"temperature": self.temperature},
        }
        req = urllib.request.Request(
            f"{self.host}/api/generate",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        # Retry once on malformed JSON (small models sometimes emit garbage on
        # very long prompts).
        for attempt in range(2):
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = json.loads(resp.read().decode())
            try:
                return self._extract_json(body.get("response", ""))
            except (json.JSONDecodeError, ValueError):
                if attempt == 0:
                    continue
                raise


def default_model(spec: str | None = None) -> LocalModel:
    """Build a LocalModel from a spec string.

    Specs: "stub" | "ollama" (qwen2.5:1.5b) | "ollama:<model-name>" — everything
    after the first colon is the model name, so tags work naturally (e.g.
    "ollama:qwen2.5:1.5b"). Falls back to the MONITOR_MODEL env var, then to the
    offline stub, so tests and CI never require a running Ollama server.

    Default Ollama model is ``qwen2.5:1.5b`` (preregistration §6.2).
    """
    spec = spec or os.environ.get("MONITOR_MODEL", "stub")
    if spec in ("stub", "offline"):
        return OfflineStubModel()
    if spec == "ollama":
        return OllamaModel(model="qwen2.5:1.5b")
    if spec.startswith("ollama:"):
        name = spec.split(":", 1)[1]
        if not name:
            raise ValueError("empty ollama model name; expected e.g. 'ollama:qwen2.5:1.5b'")
        return OllamaModel(model=name)
    raise ValueError(f"unknown model spec {spec!r}; use 'stub', 'ollama', or 'ollama:<name>'")


def make_model(spec: str) -> LocalModel:
    """Build a LocalModel from an explicit CLI spec: ``stub`` or ``ollama:<name>``.

    Unlike ``default_model`` there is no env-var or stub fallback — an empty or
    unknown spec is an error (CLI callers should fail loudly, not silently degrade).
    """
    if not spec:
        raise ValueError("empty model spec; expected 'stub' or 'ollama:<name>'")
    return default_model(spec)
