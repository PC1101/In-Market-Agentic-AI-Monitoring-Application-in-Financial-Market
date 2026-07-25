# BATCH A — Provenance Header + Onset Re-Anchoring (code + tests only)

> For the executing agent: this batch is PURE CODE + TESTS. You must NOT run any
> Ollama/LLM pipeline, NOT invoke `run_agentic.py` outside pytest tmp_path fixtures,
> NOT delete or move any file in `monitoring/results/`, and NOT touch anything
> outside `monitoring/`. Do exactly what is written here — no extra refactors,
> no docstring/comment additions to untouched code, no "improvements".

## Shared context (read once)

Repo root: `C:\Users\dungh\Downloads\In-Market-Agentic-AI-Monitoring-Application-in-Financial-Market`
Working dir for all commands: `monitoring/` (run `python -m pytest -q` from there).
Windows 10, bash shell, system Python 3.11 (`python` works directly, no venv).
Current state: 213 tests green in `monitoring/tests/` (25 files). `monitoring/results/`
contains 8 REAL-model event JSONLs and 1 `dev_analysis.json` that are the project's
primary evidence — treat as read-only.

This is an agentic-vs-classical strategy-monitoring experiment. Two fixes here:
1. **Provenance**: JSONL run logs record no model name / curve path / git SHA, so stub
   vs real-model runs are indistinguishable after the fact.
2. **Onset re-anchoring**: event onset dates in `windows.py` are hardcoded market-event
   dates, not verified strategy-break dates. We add a deterministic curve-derived onset
   as a SENSITIVITY scoring — the primary scoring must remain bit-identical.

## Task 1 — Provenance `run_meta` header

1. `monitoring/run_agentic.py`, in `main()`, immediately after `logger = RunLogger(out)`:
   ```python
   logger.log({"agent": "run_meta", "model": model.name, "curve": str(curve),
               "git_sha": _git_sha(), "window": args.window,
               "strategy": args.strategy, "condition": args.condition})
   ```
   Add a small module-level helper:
   ```python
   def _git_sha() -> str:
       try:
           import subprocess
           return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                                 capture_output=True, text=True, timeout=5,
                                 cwd=Path(__file__).parent).stdout.strip() or "unknown"
       except Exception:
           return "unknown"
   ```
2. `monitoring/run_classical.py`: in the summary written to
   `results/classical_summary.json`, include the resolved curve path(s) per strategy
   (the values already computed from `STRATEGY_CURVES`). Do NOT change detector logic.
   NOTE: do not actually re-run `run_classical.py` in this batch — another batch does
   that deliberately. Just make the code change.
3. `monitoring/scripts/run_remaining_events.py` → `count_supervisor_days()` (lines ~24–36):
   verify it filters records by `agent == "performance_supervisor"` (or equivalent) so a
   `run_meta` first line cannot be miscounted. Only change it if it would miscount.
4. Test (`monitoring/tests/test_run_agentic.py`): using the existing stub-model
   end-to-end fixture pattern (tmp_path, `OfflineStubModel`), assert the first line of
   the produced JSONL parses and has `agent == "run_meta"` with keys
   `model`, `curve`, `git_sha`.

## Task 2 — Onset re-anchoring (override + sensitivity scoring)

1. NEW `monitoring/onset.py`:
   ```python
   def curve_onset(series: pd.Series, window) -> pd.Timestamp:
       """Deterministic curve-derived onset: within [window.start_ts, window.end_ts],
       find the deepest peak-to-trough drawdown of the cumulative curve; return the
       peak date preceding that trough. No detectors, no tunable parameters."""
   ```
   Implementation: restrict `series` (daily returns) to the window, build cumulative
   curve `(1+r).cumprod()`, compute running max, drawdown = curve/runmax − 1, trough =
   idxmin of drawdown, onset = last date ≤ trough where drawdown == 0 (the peak).
   Raise `ValueError` for calm windows (`window.onset is None`) or empty slices.
2. `monitoring/metrics.py` → `evaluate_window()` (TP-zone logic ~lines 62–75): add
   parameter `override_onset=None`; use
   `onset = override_onset if override_onset is not None else window.onset_ts`.
   No other behavioral change.
3. `monitoring/agentic/alarm_extraction.py` → `evaluate_agentic_window()`
   (~lines 145–149): same `override_onset=None` parameter, same substitution.
4. `monitoring/scripts/analyze_dev_results.py`: after the existing (primary) analysis,
   compute a second scoring pass using `curve_onset()` per (window × strategy) —
   loading each strategy series the same way the script already does — and write it to
   `dev_analysis.json` under a new top-level key `"onset_sensitivity"` containing:
   per-pair `{curve_onset, classical_latency, agentic_latency}` and the permutation-test
   result under the alternative onsets. The PRIMARY sections must be unchanged.
   NOTE: this script reads real-model JSONLs from `results/` — reading is fine,
   overwriting `dev_analysis.json` with primary values identical to before is the
   acceptance criterion (diff the file: only `onset_sensitivity` may be new/changed).
   IMPORTANT: back up `results/dev_analysis.json` to `results/dev_analysis.json.bak`
   before first regeneration; delete the .bak only after the diff check passes.
5. `monitoring/freeze_gate.py`: add two checks — `onset` module imports and exposes
   `curve_onset`; `results/dev_analysis.json` contains key `onset_sensitivity`.
6. Tests:
   - NEW `monitoring/tests/test_onset.py`: synthetic return series with an engineered
     peak→trough (e.g., flat, then +1% for 10 days, then −3% for 10 days, then flat)
     inside a synthetic event `Window`; assert `curve_onset` returns the known peak
     date; assert ValueError on a calm window.
   - `monitoring/tests/test_metrics.py` and `test_alarm_extraction.py`: one test each —
     `override_onset` shifts latency by exactly the shift in onset; `None` reproduces
     the default result.

## Acceptance criteria (all must hold)

1. `cd monitoring && python -m pytest -q` → all green (213 existing + new tests).
2. `git diff results/` shows changes ONLY to `dev_analysis.json`, and within it ONLY
   the added `onset_sensitivity` key (primary numbers bit-identical).
3. No files deleted, moved, or created under `results/` except `dev_analysis.json`
   edits and the temporary `.bak`.
4. `python freeze_gate.py` runs; the two new checks pass (other checks unchanged).

## Forbidden actions
- No `--model ollama:*` invocations of anything.
- No edits to `windows.py` onset values.
- No commits unless the user asked; leave the working tree for review.
