"""Benchmark candidate local LLMs for the Performance Supervisor Agent.

Closes the Week-1 deliverable "local AI model selected, benchmarked, and confirmed
running locally". Replays the same 8 event-onset contexts that ``run_classical.py``
feeds the agentic scaffold (4 event windows x 2 strategies) against one or more
candidate models, and measures per call:

  * wall-clock latency of each model call,
  * JSON validity / schema validity (including the runner's one repair retry),
  * the resulting state/action (sanity: event onsets should not come back NORMAL/HOLD).

Writes ``results/model_benchmark.json`` (summary) and appends every invocation to
``results/model_benchmark_log.jsonl`` (auditable trail, same RunLogger as the main run).

Usage (from ``monitoring/``):
    python benchmark_model.py                                   # default: ollama:llama3.2:3b
    python benchmark_model.py --models ollama:llama3.2:3b ollama:qwen2.5:3b stub
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from time import perf_counter

from run_classical import STRATEGY_CURVES, _new_detectors, _windows_in_curve
from pnl_loader import load_pnl, returns_series
from detectors import aggregate_alarms
from agentic import as_of_context, RunLogger, run_supervisor
from agentic.model import LocalModel, make_model
from agentic.prompts import SUPERVISOR_PROMPT_VERSION
from agentic.schemas import SchemaError

RESULTS = Path(__file__).resolve().parent / "results"


class TimedModel(LocalModel):
    """Wraps a LocalModel and records the latency of every ``complete`` call."""

    def __init__(self, inner: LocalModel):
        self.inner = inner
        self.name = inner.name
        self.call_latencies: list[float] = []

    def complete(self, system: str, user: str) -> dict:
        t0 = perf_counter()
        out = self.inner.complete(system, user)
        self.call_latencies.append(perf_counter() - t0)
        return out


def build_event_contexts() -> list[tuple[str, dict]]:
    """Rebuild the causal event-onset contexts exactly as run_classical.py does."""
    contexts: list[tuple[str, dict]] = []
    for strat, paths in STRATEGY_CURVES.items():
        for path in paths:
            if not Path(path).exists():
                continue
            series = returns_series(load_pnl(path))
            events = [w for w in _windows_in_curve(series) if w.kind == "event"]
            if not events:
                continue
            results = [d.detect(series) for d in _new_detectors()]
            alarms = {r.detector: r.alarms for r in results}
            alarms["aggregate"] = aggregate_alarms(results, window_days=5, min_detectors=2)
            for w in events:
                onset = w.onset_ts
                if onset not in series.index:
                    later = series.index[series.index >= onset]
                    if len(later) == 0:
                        continue
                    onset = later[0]
                contexts.append((f"{strat}:{w.name}", as_of_context(series, onset, alarms)))
    return contexts


def benchmark_model(spec: str, contexts, logger: RunLogger) -> dict:
    base = make_model(spec)
    rows = []
    for label, ctx in contexts:
        timed = TimedModel(base)
        t0 = perf_counter()
        assessment, error = None, None
        try:
            assessment = run_supervisor(ctx, timed, logger=None)
        except SchemaError as e:
            error = f"schema: {e}"
        except Exception as e:  # connection refused, timeout, bad JSON, ...
            error = f"{type(e).__name__}: {e}"
        total_s = perf_counter() - t0

        row = {
            "model": base.name,
            "label": label,
            "as_of": ctx["as_of"],
            "valid": assessment is not None,
            "needed_repair": len(timed.call_latencies) > 1,
            "n_model_calls": len(timed.call_latencies),
            "call_latencies_s": [round(x, 3) for x in timed.call_latencies],
            "total_s": round(total_s, 3),
            "state": assessment.state if assessment else None,
            "action": assessment.action if assessment else None,
            "error": error,
        }
        rows.append(row)
        logger.log_invocation(
            agent="performance_supervisor",
            prompt_version=SUPERVISOR_PROMPT_VERSION,
            as_of=ctx["as_of"],
            raw_output=None,
            assessment=assessment.to_dict() if assessment else None,
            error=error,
            latency_s=row["total_s"],
            extra={"model": base.name, "label": label, "benchmark": True,
                   "needed_repair": row["needed_repair"]},
        )
        status = row["state"] or "FAILED"
        print(f"  {label:35} {row['total_s']:7.2f}s  {status:8} "
              f"{'(repair)' if row['needed_repair'] else ''}{error or ''}")

    lat = [x for r in rows for x in r["call_latencies_s"]]
    summary = {
        "model": base.name,
        "n_contexts": len(rows),
        "n_valid": sum(r["valid"] for r in rows),
        "n_first_try_valid": sum(r["valid"] and not r["needed_repair"] for r in rows),
        "mean_call_latency_s": round(statistics.mean(lat), 3) if lat else None,
        "median_call_latency_s": round(statistics.median(lat), 3) if lat else None,
        "max_call_latency_s": round(max(lat), 3) if lat else None,
        "states": sorted({r["state"] for r in rows if r["state"]}),
        "rows": rows,
    }
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--models", nargs="+", default=["ollama:llama3.2:3b"],
                    help="model specs: 'stub' or 'ollama:<name>' (default: ollama:llama3.2:3b)")
    args = ap.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    logger = RunLogger(RESULTS / "model_benchmark_log.jsonl")

    print("Building event-onset contexts (runs the classical detectors)...")
    contexts = build_event_contexts()
    print(f"{len(contexts)} contexts.\n")

    summaries = []
    for spec in args.models:
        print(f"## {spec}")
        summaries.append(benchmark_model(spec, contexts, logger))
        print()

    out = RESULTS / "model_benchmark.json"
    out.write_text(json.dumps(summaries, indent=2, default=str))

    header = f"{'model':24} {'valid':>8} {'1st-try':>8} {'mean lat':>9} {'max lat':>9}"
    print(header)
    print("-" * len(header))
    for s in summaries:
        print(f"{s['model']:24} {s['n_valid']}/{s['n_contexts']:>4} "
              f"{s['n_first_try_valid']:>8} "
              f"{s['mean_call_latency_s'] if s['mean_call_latency_s'] is not None else 'n/a':>8}s "
              f"{s['max_call_latency_s'] if s['max_call_latency_s'] is not None else 'n/a':>8}s")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
