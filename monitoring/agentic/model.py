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

from .schemas import State, Action, RiskFlag, NewsIntensity


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
        # Dispatch: news-agent prompts carry the "Filtered headlines" marker.
        if "Filtered headlines" in user:
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

        # News context (supervisor-v3 prompts): stressful news corroborates.
        news = ctx.get("news") or {}
        news_flags = [f for f in news.get("risk_flags", []) if f != RiskFlag.NONE]
        news_hot = bool(news_flags) and news.get("news_intensity") in (
            NewsIntensity.ELEVATED, NewsIntensity.HIGH)
        if news_hot:
            severity = max(severity, 1)
            if news.get("news_intensity") == NewsIntensity.HIGH:
                severity = min(3, severity + 1)

        state = [State.NORMAL, State.WATCH, State.ALERT, State.CRITICAL][severity]
        action = [Action.HOLD, Action.INVESTIGATE, Action.REDUCE, Action.HALT][severity]
        confidence = min(0.95, 0.4 + 0.15 * severity + 0.1 * min(n_fired, 3))

        cited = [k for k, v in alarms.items() if v]
        cause = (
            f"{n_fired} classical detector(s) fired; recent mean daily return "
            f"{mean_ret:+.4f}, drawdown {dd:+.2%}, worst day {worst:+.2%}."
        )
        if news_hot:
            cited.append("news_context")
            cause += (
                f" News context corroborates: {news.get('news_intensity')} intensity, "
                f"flags {', '.join(news_flags)}."
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
        ctx["news"] = _grab_json_after(user, "News context summary:")
        return ctx

    # Map news-filter lexicon categories -> risk-flag vocabulary.
    _CATEGORY_FLAGS = {
        "liquidity_credit": RiskFlag.LIQUIDITY_STRESS,
        "market_stress": RiskFlag.MARKET_SELLOFF,
        "fund_quant_stress": RiskFlag.FUND_STRESS,
        "macro_policy": RiskFlag.MACRO_DETERIORATION,
    }

    def _complete_news(self, user: str) -> dict:
        """Deterministic news-context summary from the prompt text (no LLM)."""
        m = re.search(r"as_of\):\s*([0-9-]+)", user)
        as_of = m.group(1) if m else ""
        signal = _grab_json_after(user, "Aggregate news signal:")

        # Headline lines look like: "- [2007-08-10] (market_stress) Stocks plunge..."
        parsed = re.findall(r"^- \[([0-9-]+)\] \(([^)]*)\) (.+)$", user, re.MULTILINE)
        flags: list[str] = []
        for _, cats, _ in parsed:
            for cat in (c.strip() for c in cats.split(",") if c.strip()):
                flag = self._CATEGORY_FLAGS.get(cat)
                if flag and flag not in flags:
                    flags.append(flag)

        stress = signal.get("stress_last_day")
        max_stress = max(
            (v for v in (stress, signal.get("max_stress_3d")) if isinstance(v, (int, float))),
            default=None,
        )
        n_hits = int(signal.get("n_hits_3d", 0) or 0)

        if not flags:
            flags = [RiskFlag.NONE]
            intensity = NewsIntensity.LOW
        elif (max_stress is not None and max_stress >= 0.6) or n_hits >= 10:
            intensity = NewsIntensity.HIGH
        elif (max_stress is not None and max_stress >= 0.3) or n_hits > 0:
            intensity = NewsIntensity.ELEVATED
        else:
            intensity = NewsIntensity.LOW

        cited = [h for _, _, h in parsed[-3:]]
        if flags == [RiskFlag.NONE]:
            narrative = "No stress-relevant headlines in the visible window."
            cited = []
        else:
            narrative = (
                f"{len(parsed)} stress-matched headline(s) visible; "
                f"flags {', '.join(flags)}; "
                f"recent stress {max_stress if max_stress is not None else 'n/a'}."
            )
        confidence = 0.5 if flags == [RiskFlag.NONE] else min(
            0.9, 0.5 + 0.05 * len(parsed))
        return {
            "as_of": as_of,
            "risk_flags": flags,
            "narrative": narrative[:600],
            "news_intensity": intensity,
            "confidence": round(confidence, 2),
            "headlines_cited": cited,
        }


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


def make_model(spec: str, agent: str = "supervisor") -> LocalModel:
    """Build a LocalModel from a CLI spec: ``stub`` or ``ollama:<model-name>``.

    ``ollama:`` specs keep everything after the first colon as the model name, so
    tags work naturally (e.g. ``ollama:llama3.2:3b``). Ollama models are
    grammar-constrained via structured outputs to the schema of ``agent``:
    ``"supervisor"`` (default, preserves Week-2 behavior) or ``"news_context"``.
    The stub dispatches on the prompt itself, so ``agent`` is ignored for it.
    """
    if spec == "stub":
        return OfflineStubModel()
    if spec.startswith("ollama:"):
        name = spec.split(":", 1)[1]
        if not name:
            raise ValueError("empty ollama model name; expected e.g. 'ollama:llama3.2:3b'")
        from .schemas import ASSESSMENT_JSON_SCHEMA, NEWS_CONTEXT_JSON_SCHEMA
        schemas = {"supervisor": ASSESSMENT_JSON_SCHEMA,
                   "news_context": NEWS_CONTEXT_JSON_SCHEMA}
        if agent not in schemas:
            raise ValueError(f"unknown agent {agent!r}; expected {tuple(schemas)}")
        return OllamaModel(model=name, json_schema=schemas[agent])
    raise ValueError(f"unknown model spec {spec!r}; expected 'stub' or 'ollama:<name>'")
