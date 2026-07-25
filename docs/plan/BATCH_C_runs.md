# BATCH C — Real-Model Calm FPR Runs + PIT Verification/Extension (HIGH RISK)

> For the executing agent: this batch launches multi-hour compute. It contains the
> project's two known foot-guns. Read "Known hazards" before doing ANYTHING.
> Prerequisite: BATCH A must be merged first (the `run_meta` provenance header and the
> analyzer's stub hard-fail depend on it). Do not run BATCH B's lock cleanup while this
> batch is active.

## Shared context (read once)

Repo root: `C:\Users\dungh\Downloads\In-Market-Agentic-AI-Monitoring-Application-in-Financial-Market`
Working dir for monitoring commands: `monitoring/`. Windows 10, bash shell, Python 3.11.
Ollama serves `qwen2.5:1.5b` locally (RTX 3050 Ti; if laptop "Eco Mode" is on, the dGPU
is off and inference is ~10× slower on CPU — supervisor calls should take ~4s, not ~40s;
if you observe ~40s, stop and tell the user to disable Eco Mode).

Experiment: agentic (LLM) vs classical detectors monitoring strategy PnL. Dev-set event
results are DONE (8/8 real-model JSONLs in `monitoring/results/` — never overwrite).
Two critical gaps remain:
- **FPR endpoint**: calm-window agentic runs were only ever done with the stub model;
  H1's "lower false-positive rate" claim is unmeasured. Fix: run 2 calm windows ×
  2 strategies with the real model.
- **PIT status**: `run_classical.py:_prefer_pit()` auto-prefers point-in-time curves and
  PIT curves exist for 2004–2014, so the dev baseline is *probably* PIT — verify it.
  Also, no AL_PCA PIT curve covers the test window china_deval_2015 (2015-08-11);
  `baseline_2007_2015_pit` ends 2014-12-31 but PIT price sleeves reach 2016-12-31,
  so build an extended curve.

## Known hazards (memorize)
1. **`run_agentic.py` DELETES its output JSONL on start** (`out.unlink()` in `main()`).
   NEVER invoke it directly for a window×strategy that already has a completed
   real-model JSONL. This mistake already destroyed a 74-day result once. Always go
   through the orchestration scripts, whose resume logic skips completed cells.
2. **One Ollama consumer at a time.** Concurrent runs previously caused HTTP 500s.
   Do not parallelize calm runs; do not run the AL backtest simultaneously if CPU
   contention slows Ollama badly (prefer strictly sequential: Task 1 → Task 2).
3. **Subprocess encoding**: always `encoding="utf-8"` and env `PYTHONUTF8=1`
   (cp1252 crashes on the `×` character the runners print).
4. If a run fails mid-window: the resume logic (≥97% supervisor-day threshold) handles
   partial windows — just rerun the script. Do NOT delete partial JSONLs.
5. Do NOT retry a failing command in a loop. One retry max, then report to the user.
6. **Never run anything against test-set windows** (flash_crash_2010, china_deval_2015,
   volmageddon_2018, covid_2020, calm_2012, calm_2017). They stay blind until freeze.

## Task 1 — Real-model calm FPR runs (4 cells, hours)

1. **Archive stub calm files FIRST** (mandatory — resume logic counts supervisor days
   and would SKIP the cells otherwise). If not already done by BATCH B:
   `git mv monitoring/results/agentic_calm_{2004_2006,2013_2014}_{AL_PCA,JT_MOM}.jsonl monitoring/results/archive/stub/`
   (create the dir; use two `git mv` commands per file if brace expansion misbehaves).
2. **NEW `monitoring/scripts/run_calm_windows.py`** — copy the structure of
   `monitoring/scripts/run_remaining_events.py` (read it first: singleton
   `results/.run_lock` + per-window `results/.lock_{window}_{strategy}` locks,
   `count_supervisor_days()` resume with 97% threshold, sequential subprocess calls
   with `encoding="utf-8"` + `PYTHONUTF8=1`). Differences:
   - Cells: `calm_2004_2006`, `calm_2013_2014` × `AL_PCA`, `JT_MOM`.
   - Expected day counts computed at RUNTIME from the strategy curve (use
     `pnl_loader.load_pnl` + `returns_series` + the window bounds from
     `windows.get_window`), not hardcoded — AL_PCA's calm_2004_2006 uses a different
     curve file than its baseline. Roughly ~750 and ~500 trading days.
   - Model arg: `--model ollama:qwen2.5:1.5b`.
   - After each cell, print the triage-mode counts (the skip rate on calm data is
     itself a headline result) and mean supervisor latency.
   - Do NOT auto-run the analyzer at the end until all 4 cells are complete.
3. **`monitoring/scripts/analyze_dev_results.py`**: (if BATCH A didn't already) read the
   `run_meta` first line of each calm JSONL and HARD-FAIL the FPR section (raise, don't
   warn) if `model` is the stub. Replace the existing soft "NOTE: Stub-model calm FPR"
   print. Then run it to regenerate `results/dev_analysis.json`.
4. Optional: start `python scripts/notify_when_done.py` in the background for the user.
   Do not poll; launch the runner in the background and wait for its completion signal.

**Acceptance:** 4 calm JSONLs exist with `run_meta.model == "ollama:qwen2.5:1.5b"`
(or the model's reported name); day counts ≥97% of expected; `dev_analysis.json` FPR
section computed from real-model data with no stub warning; event-window primary
latency numbers unchanged.

## Task 2 — PIT verification + extended AL_PCA PIT curve

Run AFTER Task 1's Ollama work finishes (CPU contention).

1. **Verify dev classical baseline used PIT curves:**
   - `cd monitoring && python run_classical.py` (regenerates
     `results/classical_summary.json`; with BATCH A merged it now records curve paths).
   - Diff the classical detection/latency values against the classical values embedded
     in `results/dev_analysis.json`. Expected (current dev results):
     qm07/AL lat=7, gfc08/AL lat=14, mc09/AL lat=1, dg11/AL lat=11, qm07/JT lat=7,
     dg11/JT lat=7, gfc08/JT and mc09/JT not detected.
   - MATCH → dev baseline is PIT-confirmed; write one line to the deviations log
     (`docs/WEEK4_STATUS.md`, or create `docs/DEVIATIONS.md`).
   - MISMATCH → dev baseline was snapshot-based: rerun `python calibrate_classical.py`
     then `python scripts/analyze_dev_results.py`; document the before/after numbers
     prominently in the deviations log and REPORT THIS to the user before proceeding.
2. **Build the extended PIT curve (2007→2015+):**
   - First read `"Stat Arb/statsArb-dev/run_full_universe.py"` argparse/config to see
     how the date range and `--tag` are set. PIT sleeves (`data/pit/{etf}_pit.csv`)
     already cover prices to 2016-12-31; if a sleeve is missing/short, rebuild with
     `python "Stat Arb/statsArb-dev/scripts/build_pit_sleeves.py"` (input:
     `sp500-master/sp500_ticker_start_end.csv`).
   - From repo root:
     `python "Stat Arb/statsArb-dev/run_full_universe.py" --pit --tag baseline_2007_2016`
     (~30–45 min; run in background, no polling).
   - Expected output:
     `Stat Arb/statsArb-dev/results/full_universe/baseline_2007_2016_pit/equity_curve.csv`
     spanning ≥ 2007-01 → 2015-12 (must cover 2015-10-16 for china_deval_2015).
3. **`monitoring/run_classical.py`** `STRATEGY_CURVES["AL_PCA"]`: PREPEND
   `_prefer_pit("baseline_2007_2016")` to the list (coverage selection picks the first
   curve covering a window, so longest-first ordering is the mechanism).
4. Rerun `python run_classical.py` and confirm dev-window classical numbers are
   UNCHANGED (the 2007–2014 portion of the extended curve should reproduce the same
   detections; if it does not — the extended backtest differs on overlapping dates —
   STOP and report to the user with the diff; do not silently adopt the new numbers).

**Acceptance:** PIT confirmation (or documented re-analysis); extended PIT curve on
disk covering 2015; `run_classical.py` prints the extended-PIT path; dev numbers
unchanged or explicitly escalated; deviations-log entry written.

## Final checks for this batch
1. `cd monitoring && python -m pytest -q` → all green.
2. `python freeze_gate.py` → runs (full pass not required until all batches merge).
3. Report: skip rates + FPR per calm cell, PIT verification outcome, extended-curve
   date range, and anything you had to deviate on.
4. No `git commit` / `git push` — leave changes for user review.
