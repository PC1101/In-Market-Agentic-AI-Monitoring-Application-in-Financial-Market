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


class DeadRunnerError(RuntimeError):
    """Ollama runner subprocess is dead and could not be recovered after retries."""


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


class VLLMModel(LocalModel):
    """Client for a remote vLLM server exposing the OpenAI-compatible API.

    Talks to ``/v1/chat/completions`` with ``response_format: json_object``
    and ``chat_template_kwargs: {enable_thinking: false}`` (for Qwen3.x models
    that default to thinking mode). Structured output is enforced server-side
    via guided decoding when a ``json_schema`` is provided.

    Designed for vast.ai / remote GPU deployments where the model is served by
    ``vllm serve <model> --port 18000`` and reached via SSH tunnel.
    """

    RETRY_TIMEOUT: int = 120

    def __init__(self, model: str = "Qwen/Qwen3.5-9B",
                 host: str = "http://localhost:18000",
                 temperature: float = 0.0, timeout: float = 300.0,
                 max_tokens: int = 1024,
                 json_schema: dict | None = None):
        self.model = model
        self.host = host.rstrip("/")
        self.temperature = temperature
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.json_schema = json_schema
        self.name = f"vllm:{model}"

    def complete(self, system: str, user: str, json_schema: dict | None = None) -> dict:
        import urllib.request, urllib.error

        schema = json_schema if json_schema is not None else self.json_schema

        # Build response_format: guided JSON if schema provided, else json_object
        if schema is not None:
            response_format = {
                "type": "json_schema",
                "json_schema": {"name": "output", "schema": schema},
            }
        else:
            response_format = {"type": "json_object"}

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": False,
            "response_format": response_format,
            # Disable Qwen3.x thinking mode — emit JSON directly
            "chat_template_kwargs": {"enable_thinking": False},
        }

        for http_attempt in range(3):
            timeout = self.timeout if http_attempt == 0 else self.RETRY_TIMEOUT
            headers = {"Content-Type": "application/json"}
            _api_key = os.environ.get("VLLM_API_KEY")
            if _api_key:
                headers["Authorization"] = f"Bearer {_api_key}"
            req = urllib.request.Request(
                f"{self.host}/v1/chat/completions",
                data=json.dumps(payload).encode(),
                headers=headers,
            )
            try:
                for attempt in range(2):
                    with urllib.request.urlopen(req, timeout=timeout) as resp:
                        body = json.loads(resp.read().decode())
                    content = (body.get("choices", [{}])[0]
                               .get("message", {}).get("content", ""))
                    if not content:
                        raise urllib.error.URLError(
                            "vLLM returned empty content; "
                            f"finish_reason={body.get('choices', [{}])[0].get('finish_reason')}"
                        )
                    try:
                        return self._extract_json(content)
                    except (json.JSONDecodeError, ValueError):
                        if attempt == 0:
                            continue
                        raise
            except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
                if http_attempt < 2:
                    import time as _time
                    _time.sleep(5)  # brief backoff, no runner reset needed
                    continue
                raise DeadRunnerError(
                    f"vLLM server unreachable after {http_attempt+1} attempts: {exc}"
                )


class OllamaModel(LocalModel):
    """Thin client for a locally-served Ollama model. Import-safe; lazy network use.

    A ``json_schema`` (per ``complete`` call, or as a constructor default) is passed
    as Ollama's ``format`` (structured outputs): decoding is grammar-constrained
    server-side, so the response is guaranteed to have the schema's shape.
    """

    # After a reset, retry with a shorter timeout to detect recurring death quickly.
    RETRY_TIMEOUT: int = 120  # seconds

    def __init__(self, model: str = "qwen2.5:1.5b", host: str = "http://localhost:11434",
                 temperature: float = 0.0, timeout: float = 300.0,
                 json_schema: dict | None = None):
        # 300 s: qwen2.5:1.5b completes most prompts in <30s; 5 min is generous
        # even under thermal throttling. Fail fast so the resilient loop can reset.
        self.model = model
        self.host = host.rstrip("/")
        self.temperature = temperature
        self.timeout = timeout
        self.json_schema = json_schema
        self.name = f"ollama:{model}"

    def _reset_runner(self) -> None:
        """Stop and restart the Ollama runner subprocess.

        Handles the "dead-runner" pattern: Ollama returns HTTP 200 with done=false
        or an empty response field when the underlying runner subprocess has died
        (typically from GPU thermal throttling or memory pressure). The fix is to
        explicitly stop the model (which kills the runner), wait for GPU cooldown,
        then trigger a fresh generate to force the runner to reload weights.
        """
        import subprocess, time as _time
        subprocess.run(["ollama", "stop", self.model], capture_output=True)
        _time.sleep(10)
        # Warm-up: a tiny generate call forces the runner to reload.
        try:
            import urllib.request
            req = urllib.request.Request(
                f"{self.host}/api/generate",
                data=json.dumps({"model": self.model, "prompt": "hi",
                                 "stream": False}).encode(),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                resp.read()
        except Exception:
            pass  # best effort

    def complete(self, system: str, user: str, json_schema: dict | None = None) -> dict:
        import urllib.request, urllib.error  # stdlib only; no hard dependency

        schema = json_schema if json_schema is not None else self.json_schema
        payload = {
            "model": self.model,
            "prompt": f"{system}\n\n{user}",
            "stream": False,
            "format": schema if schema is not None else "json",
            "options": {"temperature": self.temperature},
        }

        # Outer loop: retry with runner reset on infra failures (HTTP 500,
        # dead-runner 200-OK, connection refused, socket timeout).
        for http_attempt in range(3):
            timeout = self.timeout if http_attempt == 0 else self.RETRY_TIMEOUT
            req = urllib.request.Request(
                f"{self.host}/api/generate",
                data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"},
            )
            try:
                # Inner loop: retry once on malformed JSON (small models sometimes
                # emit garbage on very long prompts).
                for attempt in range(2):
                    with urllib.request.urlopen(req, timeout=timeout) as resp:
                        body = json.loads(resp.read().decode())
                    # Detect dead-runner: HTTP 200 but done=False or empty response
                    if not body.get("done") or not body.get("response"):
                        raise urllib.error.URLError("dead runner: done=%s, response empty"
                                                    % body.get("done"))
                    try:
                        return self._extract_json(body.get("response", ""))
                    except (json.JSONDecodeError, ValueError):
                        if attempt == 0:
                            continue
                        raise
            except (urllib.error.URLError, TimeoutError, ConnectionError):
                # Covers HTTPError (inc. 500), connection refused, dead-runner,
                # socket timeouts, and connection resets.
                if http_attempt < 2:
                    self._reset_runner()
                    continue
                raise DeadRunnerError(
                    f"Ollama runner unrecoverable after {http_attempt+1} reset attempts"
                )


def default_model(spec: str | None = None) -> LocalModel:
    """Build a LocalModel from a spec string.

    Specs:
      "stub"                     — offline rule-based stand-in (CI/tests)
      "ollama"                   — Ollama with default qwen2.5:1.5b
      "ollama:<model-name>"      — Ollama with a specific model
      "vllm"                     — vLLM with default Qwen/Qwen3.5-9B @ localhost:18000
      "vllm:<model>@<host:port>" — vLLM with explicit model and endpoint

    Falls back to the MONITOR_MODEL env var, then to the offline stub, so tests
    and CI never require a running server.
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
    if spec == "vllm":
        return VLLMModel()
    if spec.startswith("vllm:"):
        rest = spec[len("vllm:"):]
        if "@" in rest:
            model_name, host = rest.rsplit("@", 1)
            if not host.startswith("http"):
                host = f"http://{host}"
            return VLLMModel(model=model_name, host=host)
        return VLLMModel(model=rest)
    raise ValueError(f"unknown model spec {spec!r}; use 'stub', 'ollama:<name>', or 'vllm:<model>@<host:port>'")


def make_model(spec: str) -> LocalModel:
    """Build a LocalModel from an explicit CLI spec: ``stub`` or ``ollama:<name>``.

    Unlike ``default_model`` there is no env-var or stub fallback — an empty or
    unknown spec is an error (CLI callers should fail loudly, not silently degrade).
    """
    if not spec:
        raise ValueError("empty model spec; expected 'stub' or 'ollama:<name>'")
    return default_model(spec)
