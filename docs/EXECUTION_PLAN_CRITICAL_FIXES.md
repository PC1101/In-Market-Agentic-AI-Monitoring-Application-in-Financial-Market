# Execution Plan: Critical Fixes + Codebase Cleanup

> Handoff document for an executing agent. Self-contained: includes context, exact file
> paths, function names, verification commands. Repo root:
> `C:\Users\dungh\Downloads\In-Market-Agentic-AI-Monitoring-Application-in-Financial-Market`
> All `monitoring/` commands run from inside `monitoring/`. System Python 3.11 (no venv
> despite CLAUDE.md; `python` works directly). Ollama serves `qwen2.5:1.5b` locally.

## Context

A quant-researcher critique of this agentic-vs-classical monitoring experiment found:

- **CRITICAL-1**: The FPR co-primary endpoint (H1: "agentic is faster AND lower FPR")
  has only ever been measured with the **stub model**. The 4 calm-window JSONLs in
  `monitoring/results/` (`agentic_calm_*.jsonl`) are stub artifacts. Real-model
  (qwen2.5:1.5b) calm runs have never been done.
- **CRITICAL-2**: Survivorship/PIT asymmetry between strategies. NOTE exploration
  finding: `run_classical.py:_prefer_pit()` already auto-prefers PIT curves, and
  `Stat Arb/statsArb-dev/results/full_universe/{baseline_2007_2015_pit, calm_2004_2006_pit,
  calm_2013_2014_pit}/equity_curve.csv` all exist (built Jul 12, cover 2004–2014).
  So the dev set is *probably* already PIT. The real gaps: (a) nothing records which
  curve a run actually used (provenance), (b) no AL_PCA PIT curve reaches the test
  window china_deval_2015 (2015-08-11); `baseline_2007_2015_pit` ends 2014-12-31 but
  `build_pit_sleeves.py` prices span to 2016-12-31 so an extension is buildable.
- **HIGH-1**: Event onset dates in `monitoring/windows.py` are hardcoded *market-event*
  dates, not verified *strategy-break* dates. Both arms are scored against a possibly
  wrong ground truth; the news-informed agentic arm is advantaged by market-date anchoring.
- **HIGH-2**: JSONL logs record no model name, curve path, or git SHA — stub vs real
  runs are indistinguishable after the fact.

Plus: the repo is cluttered (dead third-party lib, stub artifacts mixed with real
results, caches, exploratory notebooks). Clean it up for modularity before eval freeze.

Current dev-set state (all real-model qwen2.5:1.5b, complete):
8/8 event JSONLs; `dev_analysis.json` shows latency −9.4d, Holm-adj p=0.0156 (reject);
FPR p=0.5158 (stub — invalid). 213 tests pass in `monitoring/tests/` (25 files).

---

## Phase 1 — Provenance header (do first; small, unblocks 2+3)

1. **`monitoring/run_agentic.py`** (`main()`): immediately after `logger = RunLogger(out)`,
   log a first meta record:
   ```python
   logger.log({"agent": "run_meta", "model": model.name, "curve": str(curve),
               "git_sha": <git rev-parse --short HEAD via subprocess, fallback "unknown">,
               "window": args.window, "strategy": args.strategy,
               "condition": args.condition})
   ```
2. **`monitoring/run_classical.py`**: include the resolved curve path per strategy in
   the `classical_summary.json` output.
3. **`monitoring/scripts/run_remaining_events.py`** (`count_supervisor_days`, lines 24–36):
   confirm it filters by `agent == "performance_supervisor"` so the meta line is ignored;
   adjust if needed.
4. **Test**: add to `monitoring/tests/test_run_agentic.py` — after a stub run, first JSONL
   line has `agent == "run_meta"` with model/curve keys.

Verify: `python run_agentic.py --window quant_meltdown_2007 --strategy AL_PCA --model stub`
→ **but note this deletes/recreates the real-model JSONL** (`out.unlink()` in main).
Instead test via pytest with tmp_path only; do NOT overwrite real results files.
Run `python -m pytest -q` → all green.

## Phase 2 — CRITICAL-1: Real-model calm FPR runs (4 cells; hours of runtime)

1. **Archive stub calm files FIRST** (resume logic counts supervisor days and would
   otherwise SKIP these cells): `git mv` the 4 files
   `monitoring/results/agentic_calm_{2004_2006,2013_2014}_{AL_PCA,JT_MOM}.jsonl`
   → `monitoring/results/archive/stub/`.
2. **New `monitoring/scripts/run_calm_windows.py`** — clone the pattern of
   `run_remaining_events.py` (singleton `results/.run_lock`, per-window
   `results/.lock_{window}_{strategy}`, resume-if-≥97%-days, `encoding="utf-8"` +
   `PYTHONUTF8=1` subprocess env, sequential):
   - Windows: `calm_2004_2006`, `calm_2013_2014` × strategies `AL_PCA`, `JT_MOM`.
   - Compute expected day counts at runtime from the strategy curve (via
     `pnl_loader.returns_series` + window bounds), not hardcoded (~750 and ~500 days).
   - Model: `ollama:qwen2.5:1.5b`. Triage will skip most calm days by design; print
     the triage-mode counts (skip rate is itself a result).
3. **`monitoring/scripts/analyze_dev_results.py`**: read the Phase-1 `run_meta` header
   in each calm JSONL and **hard-fail the FPR section if model is stub** (replace the
   current soft warning at lines ~291–292). Regenerate `dev_analysis.json`.

Verify: `python scripts/run_calm_windows.py` → 4/4 complete;
`python scripts/analyze_dev_results.py` → real-model FPR + updated Holm section.
Optionally start `python scripts/notify_when_done.py` for completion notification.
Runtime warning: heavy-news days cost ~8s each; run in background, don't poll.

## Phase 3 — CRITICAL-2: PIT verification + test-set PIT extension (parallel with Phase 2 only if Ollama/CPU allows; the backtest is CPU-heavy, prefer sequential)

1. **Verify dev runs used PIT**: PIT curves built Jul 12; dev agentic runs Jul 20–23, and
   `_prefer_pit` auto-selects PIT — so PIT was almost certainly used. Confirm by running
   `python run_classical.py` and diffing detection/latency values in the regenerated
   `classical_summary.json` against the classical values inside `dev_analysis.json`
   (e.g. qm07/AL_PCA lat=7, gfc08/AL_PCA lat=14, mc09/AL_PCA lat=1, dg11/AL_PCA lat=11,
   qm07/JT lat=7, dg11/JT lat=7, gfc08/JT & mc09/JT not detected).
   - If they MATCH → dev baseline is PIT; record confirmation in the deviations log.
   - If they DIFFER → dev classical baseline was snapshot-based: re-run
     `python calibrate_classical.py` then `python scripts/analyze_dev_results.py`,
     and document the change prominently.
2. **Extend the AL_PCA PIT curve through 2015** (needed for test window china_deval_2015):
   ```
   python "Stat Arb/statsArb-dev/run_full_universe.py" --pit --tag baseline_2007_2016
   ```
   (from repo root; ~30–45 min). Output:
   `Stat Arb/statsArb-dev/results/full_universe/baseline_2007_2016_pit/equity_curve.csv`.
   Note: `run_full_universe.py` may need its end-date argument/config set to 2016; read
   its argparse first. PIT sleeves in `data/pit/` already cover prices to 2016-12-31;
   if a sleeve rebuild is required, use
   `python "Stat Arb/statsArb-dev/scripts/build_pit_sleeves.py"` (inputs:
   `sp500-master/sp500_ticker_start_end.csv`).
3. **`monitoring/run_classical.py`** `STRATEGY_CURVES["AL_PCA"]`: prepend
   `_prefer_pit("baseline_2007_2016")` so the longest curve wins coverage selection
   (run_agentic's `main()` picks the first curve covering the window).
4. **Docs**: append a deviations-log entry (PIT status confirmation + curve extension)
   to `docs/WEEK4_STATUS.md` (or a new `docs/DEVIATIONS.md` if none exists).

Verify: new curve spans 2007-01→2015-12; `run_classical.py` prints PIT paths;
dev numbers unchanged (or re-analysis documented).

## Phase 4 — HIGH-1: Onset re-anchoring (dual scoring; pure code, no LLM runs)

1. **New `monitoring/onset.py`**: `curve_onset(series: pd.Series, window: Window) -> pd.Timestamp`
   — deterministic pre-registered rule: within [window.start_ts, window.end_ts], find the
   deepest peak-to-trough drawdown episode of the cumulative curve; onset = the peak date
   preceding that trough. No detectors, no tunable parameters.
2. **`monitoring/metrics.py:evaluate_window()`** (TP-zone logic ~lines 62–75) and
   **`monitoring/agentic/alarm_extraction.py:evaluate_agentic_window()`** (~lines 145–149):
   add param `override_onset: pd.Timestamp | None = None`;
   `onset = override_onset if override_onset is not None else window.onset_ts`.
   No behavior change when None.
3. **`monitoring/scripts/analyze_dev_results.py`**: compute both scorings —
   hardcoded onset (primary, unchanged) and curve-derived (sensitivity) — and write the
   second under `"onset_sensitivity"` in `dev_analysis.json`, including per-pair latencies
   and the permutation-test p under the alternative onsets.
4. **Tests**: new `monitoring/tests/test_onset.py` (synthetic curve, known trough →
   known onset); override-param tests added to `test_metrics.py` and
   `test_alarm_extraction.py`.
5. **`monitoring/freeze_gate.py`**: add checks — `onset.py` importable;
   `dev_analysis.json` contains `onset_sensitivity`.

Verify: primary numbers bit-identical to current `dev_analysis.json` when override=None;
both scorings printed side by side; full pytest green.

## Phase 5 — Cleanup (after Phase 2's archive step; final gate)

Exploration confirmed: **no tests read `monitoring/results/` or import from
`monitoring/scripts/`; nothing imports arbitragelab.** Use `git rm -r` / `git mv` for
tracked paths (history preserves everything); plain delete for untracked caches.

DELETE:
- all `__pycache__/` and `.pytest_cache/` dirs under `monitoring/`, `XSectional/`, `Stat Arb/`
- `Stat Arb/arbitragelab-master/` (52 MB third-party reference repo, never imported)
- stale `monitoring/results/.lock_*` and `.run_lock` files (only if no run is active)
- `monitoring/results/agent_log.jsonl` (orphan single-run log)
- `docs/superpowers/` (empty/reference)

ARCHIVE (git mv):
- `sp500-master/*.ipynb` (4 exploratory notebooks) → `sp500-master/notebooks/`
- `monitoring/results/model_benchmark.json`, `model_benchmark_log.jsonl` →
  `monitoring/results/archive/` (Week-1 model-selection evidence, keep for writeup)
- (already done in Phase 2) stub calm JSONLs → `monitoring/results/archive/stub/`

KEEP (do NOT delete, despite looking like cruft):
- `monitoring/benchmark_model.py` (documents the llama3.2→qwen2.5 model deviation)
- all 5 files in `monitoring/scripts/`
- all `sp500-master/*.csv` (PIT provenance chain)
- `data/fnspid/store/` (857 MB news cache, expensive to rebuild)
- all 8 real-model event JSONLs in `monitoring/results/`
- `Stat Arb/statsArb-dev/results/full_universe/*` including non-PIT dirs (fallback +
  snapshot-vs-PIT comparison evidence)

## Final verification

1. `cd monitoring && python -m pytest -q` → all green (~220+ tests after additions)
2. `cd XSectional && python -m pytest -q` → all green
3. `cd monitoring && python freeze_gate.py` → passes incl. new onset-sensitivity check
4. `dev_analysis.json` contains: real-model FPR (no stub warning), onset_sensitivity block
5. `python run_classical.py` prints PIT curve paths for AL_PCA incl. 2015 coverage
6. Do NOT run anything against the test-set windows — they stay blind until eval freeze.

## Known hazards for the executor

- `run_agentic.py` **deletes** the output JSONL on start (`out.unlink()`). Never invoke
  it against a window×strategy whose real-model JSONL you want to keep, unless resuming
  via the scripts' skip logic.
- Only one Ollama-consuming run at a time (past HTTP 500s under contention).
- Windows shell is bash; always pass `encoding="utf-8"` / `PYTHONUTF8=1` to subprocesses
  (cp1252 crashes on `×`).
- Laptop dGPU: Eco Mode disables the RTX 3050 Ti → Ollama falls back to CPU (~10× slower).
