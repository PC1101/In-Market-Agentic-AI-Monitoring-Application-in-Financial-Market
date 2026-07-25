# Batch D Phase 1 — Correctness & Bias Audit Report

**Date:** 2026-07-24
**Status:** All 5 checks complete. No pipeline blockers found. Two items require documentation before freeze.

---

## Check 1 — Provenance Audit

**Verdict: FAIL (expected — triggers Phase D3 rerun)**

Checked first JSONL line of all 11 `results/agentic_*.jsonl` files:

| File | Status | Notes |
|---|---|---|
| `agentic_calm_2004_2006_AL_PCA.jsonl` | OK | run_meta: model=ollama:qwen2.5:1.5b, pit=True, sha=5b06eb4 |
| `agentic_calm_2004_2006_JT_MOM.jsonl` | OK | run_meta: model=ollama:qwen2.5:1.5b, pit=False* |
| `agentic_calm_2013_2014_AL_PCA.jsonl` | OK | run_meta: model=ollama:qwen2.5:1.5b, pit=True |
| `agentic_calm_2013_2014_JT_MOM.jsonl` | IN PROGRESS | Batch C Task 1 still running |
| `agentic_downgrade_2011_AL_PCA.jsonl` | NO_META | Pre-Batch-A — formal trigger for D3 rerun |
| `agentic_downgrade_2011_JT_MOM.jsonl` | NO_META | Pre-Batch-A — formal trigger for D3 rerun |
| `agentic_gfc_lehman_2008_AL_PCA.jsonl` | NO_META | Pre-Batch-A — formal trigger for D3 rerun |
| `agentic_gfc_lehman_2008_JT_MOM.jsonl` | NO_META | Pre-Batch-A — formal trigger for D3 rerun |
| `agentic_momentum_crash_2009_AL_PCA.jsonl` | NO_META | Pre-Batch-A — formal trigger for D3 rerun |
| `agentic_momentum_crash_2009_JT_MOM.jsonl` | NO_META | Pre-Batch-A — formal trigger for D3 rerun |
| `agentic_quant_meltdown_2007_AL_PCA.jsonl` | NO_META | Pre-Batch-A — formal trigger for D3 rerun |
| `agentic_quant_meltdown_2007_JT_MOM.jsonl` | NO_META | Pre-Batch-A — formal trigger for D3 rerun |

\* JT_MOM curve path does not contain `_pit` in the filename — this is expected and correct.
  The JT momentum PIT construction is baked into the XSectional backtest logic
  (`run_daily_pnl.py` without `--survivorship`), not a separate curve file.

**Action:** Phase D3 must archive these 8 JSONLs and rerun all event cells with
`run_remaining_events.py`. This is the authoritative trigger; do not skip D3.

---

## Check 2a — Curve Coverage Matrix

**Verdict: PASS for DEV; TEST has 4 AL_PCA gaps (expected pre-Batch-C-Task-2)**

| Window | Tier | AL_PCA | JT_MOM |
|---|---|---|---|
| quant_meltdown_2007 | DEV | baseline_2007_2015_pit ✓ | equity_curve_daily ✓ |
| gfc_lehman_2008 | DEV | baseline_2007_2015_pit ✓ | equity_curve_daily ✓ |
| momentum_crash_2009 | DEV | baseline_2007_2015_pit ✓ | equity_curve_daily ✓ |
| downgrade_2011 | DEV | baseline_2007_2015_pit ✓ | equity_curve_daily ✓ |
| calm_2004_2006 | DEV | calm_2004_2006_pit ✓ | equity_curve_daily ✓ |
| calm_2013_2014 | DEV | baseline_2007_2015_pit ✓ | equity_curve_daily ✓ |
| flash_crash_2010 | TEST | baseline_2007_2015_pit ✓ | equity_curve_daily ✓ |
| china_deval_2015 | TEST | **MISSING** (curve ends 2014-12-31) | equity_curve_daily ✓ |
| volmageddon_2018 | TEST | **MISSING** (curve ends 2014-12-31) | equity_curve_daily ✓ |
| covid_2020 | TEST | **MISSING** (curve ends 2014-12-31) | equity_curve_daily ✓ |
| calm_2012 | TEST | baseline_2007_2015_pit ✓ | equity_curve_daily ✓ |
| calm_2017 | TEST | **MISSING** (curve ends 2014-12-31) | equity_curve_daily ✓ |

**Updated (2026-07-24 post Batch C):** `baseline_2007_2016_pit` extension was not buildable
— all 11 PIT sleeve files (`data/pit/xlf_pit.csv` etc.) are absent, and the prerequisite
`XSectional/data/sp500_prices_pit.csv` does not exist. There is no path to extending the
AL_PCA PIT curve beyond 2014-12-31 without first rebuilding the sleeve files from raw
sector-ETF price data (multi-day effort, out of scope before freeze).

**Corrected AL_PCA coverage:** 6 dev windows only + flash_crash_2010 + calm_2012 of test
= **8 windows total**. china_deval_2015, volmageddon_2018, covid_2020, calm_2017 are all
JT_MOM-only (structural limitation).

**Implication for the report:** The AL_PCA arm covers 8 of 12 windows (6 dev + 2 test).
The remaining 4 test windows are JT_MOM-only. This must be stated as a limitation.

---

## Check 2b — News Store Coverage

**Verdict: PASS — no missing years, no blockers**

All critical years present in `data/fnspid/store/` as parquet partitions (2003–2020):

| Year | Tier | Articles | Coverage |
|---|---|---|---|
| 2007 | DEV | 464,439 | 2007-01-01 → 2007-12-31 |
| 2008 | DEV | 1,056,009 | 2008-01-01 → 2008-12-31 |
| 2009 | DEV | 961,342 | 2009-01-01 → 2009-12-31 |
| 2010 | TEST | 1,211,112 | 2010-01-01 → 2010-12-31 |
| 2011 | DEV | 1,586,808 | 2011-01-01 → 2011-12-31 |
| 2013 | DEV | 1,259,792 | 2013-01-01 → 2013-12-31 |
| 2014 | DEV | 1,256,672 | 2014-01-01 → 2014-12-31 |
| 2015 | TEST | 1,471,301 | 2015-01-01 → 2015-12-31 |
| 2017 | TEST | 515,523 | 2017-01-01 → 2017-12-31 |
| 2018 | TEST | 698,590 | 2018-01-01 → 2018-12-31 |
| 2020 | TEST | 351,832 | 2020-01-01 → 2020-12-31 |

**Note:** Article counts for 2017 (515K), 2018 (699K), and 2020 (352K) are materially
lower than the 2011–2015 peak (~1–1.6M/year). This may cause the triage's
`COVERAGE_DAYS=7` gate to be more permissive (fewer articles → lower FinBERT stress
signal → more skip days) in those test windows. This is a pre-registered behavior
(calm-by-coverage), not a bug, but worth reporting in the limitations section.

---

## Check 3 — Lookahead Spot-Check

**Verdict: PASS (data pipeline is clean; 1 LLM memorization artifact noted)**

Code-level guards:
- `assert_no_lookahead` called **2 times** in `run_agentic.py` (news articles + macro context)
- `mask_dates_in_*` called **4 times** in `run_agentic.py` (conditions A/B masking paths)
- `as_of_context(series, day, ...)` slices the return series at `day` (≤ as_of) ✓
- Same guards present in `run_leakage.py` (conditions A, B, C) ✓

Raw-output scan: sampled all 548 supervisor/news_context entries in the two earliest event
windows (quant_meltdown_2007, gfc_lehman_2008). Found **1 occurrence** of a future ISO date:

> `agentic_quant_meltdown_2007_JT_MOM.jsonl` — `as_of=2007-08-08` — LLM output contained `"as_of": "2008-09-15"`

**Classification: LLM pretraining memorization artifact, NOT input lookahead.**

The LLM hallucinated the Lehman bankruptcy date (2008-09-15) into its own JSON response
when asked to state `as_of`. The input context was correctly limited to data through
2007-08-08 — no future data was in the prompt. This is confirmed because:
1. `record["as_of"]` (set by the Python runner) = `2007-08-08` ✓
2. Scoring uses `record["as_of"]`, never `record["assessment"]["as_of"]`
3. The alarm extraction code reads `rec["as_of"]` for alarm timestamps, not the nested assessment field

**Implication:** This confirms that qwen2.5:1.5b has prior knowledge of major financial events.
This is precisely what the Batch D Phase 2 leakage harness (Condition B = date masking) is
designed to quantify. Flag this incident in the leakage discussion section of the report.

---

## Check 4 — Calibration Hygiene

**Verdict: PASS**

- `calibrate_classical.py`: imports and iterates `DEV_WINDOWS` only; `TEST_WINDOWS` appears
  only in a docstring comment, never in the calibration loop ✓
- `calibration_grid.json`: both strategies present; no test window names in the JSON ✓
- Best calibrated configs applied by `run_classical.py` (HMM threshold=0.5 for both strategies — the 1× multiplier, meaning the default is already near-optimal on the dev set):

| Strategy | Detector | Best params | Dev recall | Dev FPR/day |
|---|---|---|---|---|
| AL_PCA | hmm | threshold=0.5 | 1.0 | 0.0183 |
| AL_PCA | page_hinkley | delta=0.25, lambda_=16 | 0.25 | 0.0 |
| JT_MOM | hmm | threshold=0.5 | 0.50 | 0.0040 |
| JT_MOM | page_hinkley | delta=0.5, lambda_=4 | 0.75 | 0.0087 |

- `freeze_gate.py` blinding check: PASS (no test-set result files in results/) ✓

---

## Check 5 — Arm Symmetry

**Verdict: PASS with one documented asymmetry**

Both arms:
- Use **tolerance_days = 21** for event detection ✓
- Apply **`override_onset`** parameter consistently (both added in Batch A) ✓
- Use **`np.busday_count`** for trading-day counts ✓
- Score against identical `Window` objects from `windows.py` ✓

**Documented asymmetry (not a bug — report in methods):**

| Aspect | Classical arm | Agentic arm |
|---|---|---|
| Ensemble dedup | `aggregate_alarms(window_days=5)` — calendar-day rolling window; fires when ≥2 detectors alarm within 5 calendar days | `cluster_starts(dedup_days=5)` — trading-day cooldown; FP cluster starts when no prior alarm in previous 5 trading days |
| Dedup purpose | Aggregation (quorum rule for TP detection) | False-positive rate deduplication |
| Day-count basis | Calendar days | Trading days |

These serve different mathematical purposes and are not directly comparable, but both implement
"5-day" cooldown semantics consistent with the preregistration. The agentic FPR denominator
uses trading days for both arms (passed as `n_trading_days`). The asymmetry should be stated
in the methods section.

---

## Summary

| Check | Verdict | Blocker? | Action |
|---|---|---|---|
| 1 — Provenance | FAIL (expected) | → D3 | Archive 8 event JSONLs; rerun with real model |
| 2a — Curve coverage | PASS / GAP | No | 4 AL_PCA test gaps: all structural (baseline_2007_2016_pit unbuildable; PIT sleeves absent). AL_PCA covers 6 dev + flash_crash_2010 + calm_2012 only. Report as limitation. |
| 2b — News coverage | PASS | No | 2017/2018/2020 article counts lower — note in limitations |
| 3 — Lookahead | PASS | No | 1 LLM memorization artifact found; quantify via leakage harness in D2 |
| 4 — Calibration | PASS | No | Clean; HMM default threshold selected for both strategies |
| 5 — Arm symmetry | PASS | No | Dedup semantics differ (calendar vs trading days) — document in methods |

**No pipeline blockers.** Batch C Task 1 complete (all 4 calm cells done, real model).
Batch C Task 2 blocked/closed (structural). Proceed to Phase D2 (leakage harness validation)
then D3 (dev-set rerun).
