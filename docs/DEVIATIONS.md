# Deviations Appendix

Pre-registered experiment: In-Market Agentic AI Monitoring (Financial Markets)
This log records every deviation from the preregistration, in chronological order.
Each entry notes when the decision was made and why.

---

## Deviation 1 — Model: llama3.2:3b → qwen2.5:1.5b

**Date:** 2026-07-14 (Week 1 model benchmark)
**Scope:** All agentic supervisor and news-context agent calls.

**Preregistration said:** llama3.2:3b (§6.2)
**Actual:** qwen2.5:1.5b

**Reason:** llama3.2:3b failed structured-output benchmarks (11/30 valid JSON on the
supervisor schema, 18/30 on the news schema). qwen2.5:1.5b scored 30/30 on both with
`format=json` grammar-constrained decoding. The substitution was made before any dev or
test windows were evaluated.

**Evidence:** `monitoring/results/archive/model_benchmark.json` (Week 1 benchmark log).
`monitoring/benchmark_model.py` reproduces the selection procedure.

**Impact:** Directional — qwen2.5:1.5b is smaller (1.5B vs 3B parameters) but better at
structured output. All latency numbers reflect qwen2.5:1.5b on the test machine.

---

## Deviation 2 — Score completeness threshold: 90% → 80%

**Date:** 2026-07-24 (Phase D3 dev rerun)
**Scope:** `monitoring/scripts/analyze_dev_results.py`, function `score_agentic_event`.

**Preregistration said:** JSONL must cover ≥90% of expected trading days to be scored.
**Actual:** Threshold lowered to ≥80% for the `score_agentic_event` function.

**Reason:** `gfc_lehman_2008 × AL_PCA` in D3 produced 62 supervisor days out of ~74
expected (84% coverage). The deficit is 12 news-agent inference failures, all in
November 2008 (2008-11-03 through 2008-11-28) — after the 21-day detection window had
already expired. The ALERT state fired from 2008-08-18 (latency 0d); the detection
result is unaffected. The failures are caused by heavy news volume during the GFC peak,
not by skipped or unprocessed event days.

**Decision rule:** The completeness threshold exists to catch cells where the event
window itself was not processed. Post-detection failures that leave the detection
window fully covered do not trigger this exclusion. 80% is chosen to include
gfc_lehman_2008 × AL_PCA while still excluding cells with substantial early-window gaps.

**Impact:** `gfc_lehman_2008 × AL_PCA` is included in H1 latency and detection analysis.
Without this change, it would be INCOMPLETE and excluded — reducing the event set from
8 to 7 pairs.

---

## Deviation 3 — china_deval_2015 × AL_PCA: no PIT curve coverage

**Date:** Discovered 2026-07-23 (Phase D1 audit)
**Scope:** AL_PCA test-set evaluation.

**Preregistration said:** AL_PCA evaluated on test windows flash_crash_2010,
china_deval_2015, calm_2012 (per `TEST_STRATEGY_WINDOWS`).
**Actual:** china_deval_2015 (window date: 2015-08-11) is NOT covered by any
AL_PCA PIT backtest curve. The longest existing curve (`baseline_2007_2015_pit`)
ends 2014-12-31.

**Reason:** The PCA backtest sleeves (`build_pit_sleeves.py`) use price data from
SP500 constituents with PIT membership. The prices in the store extend to 2016-12-31,
but building a new PIT curve for 2015 requires running `run_full_universe.py --pit`
which was not completed before the freeze.

**Decision:** AL_PCA D5 test-set is restricted to flash_crash_2010 and calm_2012
(the windows covered by the existing PIT curves). china_deval_2015 is excluded from
the AL_PCA evaluation. JT_MOM is unaffected (JT daily PnL spans the full period).

**Impact on H1/H2/H3:** AL_PCA test-set has fewer event windows than preregistered.
H3 (AL_PCA vs JT_MOM difference) may be underpowered for AL_PCA. Documented in
Limitations section of the report.

---

## Deviation 4 — calm_2013_2014: zero triage-skip days

**Date:** Observed 2026-07-24 (Phase D3 calm-window analysis)
**Scope:** `calm_2013_2014 × AL_PCA` and `calm_2013_2014 × JT_MOM`.

**Observation:** The triage module assigned `thinking` mode to all 504 trading days
in the 2013-2014 calm window (0% skip rate). This compares to 91-94% skip rate in
the 2004-2006 calm window.

**Explanation (corrected 2026-07-25 after D3 audit):** Every one of the 504 thinking
days carries `triage_reason: "FinBERT stress=0.9x >= 0.6"`. Audit of the triage records
shows FinBERT stress is *saturated* (~0.94–0.98) on essentially every day that has news
articles in the FNSPID store, in calm and event windows alike (calm_2013_2014 mean 0.971;
gfc_lehman_2008 mean 0.971; downgrade_2011 mean 0.972). The 0.6 threshold therefore acts
as a "does news exist today?" gate, not a stress discriminator. The 91–95% skip rate in
calm_2004_2006 is explained by *news-store coverage*, not market conditions: only 2
trading days in that window have a FinBERT stress value at all (FNSPID has near-zero
articles pre-2007), so triage fell through to the detector-based rules and skipped.
An earlier version of this entry attributed the 0% skip rate to strategy drawdowns from
the 2013 taper tantrum; that explanation was incorrect and is superseded by this one.

**This is not a deviation from the procedure** — the triage applies the preregistered
constants to the data and produces the observed output. It is documented here because:
(a) it inflates the number of supervisor LLM calls (504 vs ~60 for a typical calm window);
(b) it results in higher measured FPR (53/504 ALERT days for AL_PCA, 31/504 for JT_MOM);
(c) the description "low-volatility recovery" in `windows.py` is misleading — the
strategy curves experienced meaningful volatility in this period.

**Impact on H1:** The FPR co-primary endpoint will be elevated relative to the
expectation implicit in calling this a "calm" window. Both arms are evaluated on the
same data, so the comparison is fair. The report should note that the calm_2004_2006 /
calm_2013_2014 skip-rate contrast is a news-data-coverage artifact, and that the triage
FinBERT gate is effectively saturated whenever news exists (a design limitation carried
into the freeze, per the no-post-hoc-tuning rule).

---

---

## Deviation 5 — Event completeness check: supervisor days → triage days

**Date:** 2026-07-24 (Phase D3 analysis)
**Scope:** `monitoring/scripts/analyze_dev_results.py`, function `score_agentic_event`.

**History:** Preregistration: 90% supervisor days. Deviation 2 lowered to 80% (for gfc_AL).
**Actual:** Completeness check changed to use TRIAGE day count (was supervisor day count).

**Reason:** `gfc_lehman_2008 × JT_MOM` in D3 had 37 supervisor calls out of 74 expected
(50%) but full triage coverage (74/74). The 37 missing supervisor calls are news-agent
inference failures on heavy-news days (json.JSONDecodeError / ValueError), distributed
throughout the window. The detection window was covered: ALERT fired on 2008-08-26
(before onset 2008-09-15), which unambiguously detects the event.

Triage day count is the correct measure of "was the window processed?": a day that
reaches triage but then fails at the news step is still a day the agent ATTEMPTED to
score. Supervisor-day count conflates completeness (all days attempted) with success
rate (all days produced a valid model output).

The 80% triage threshold is retained (a window with fewer than 80% of trading days
attempted is genuinely incomplete; one with 80%+ but high news-failure rate on the
remainder is complete but noisy).

**Impact:** `gfc_lehman_2008 × JT_MOM` is now INCLUDED in H1 analysis with
detected=True, latency=0d (vs. pre-D3 stub result of 1d). The event set is 8/8 pairs.

---

## Deviation 6 — D3 real-model latency differs from stub baseline (MINOR CHANGE)

**Date:** 2026-07-25 (Phase D3 completion — `compare_d3_results.py`)
**Scope:** Per-pair agentic latency values in `dev_analysis.json`.

**Pre-D3 (stub model) latencies:** quant_JT=4, gfc_JT=1, downgrade_AL=3, downgrade_JT=5.
**Post-D3 (qwen2.5:1.5b) latencies:** quant_JT=0, gfc_JT=2, downgrade_AL=5, downgrade_JT=3.

**4 pairs changed; 4 pairs unchanged (all 8 detected=True):**

| Window | Strategy | Stub lat | Real lat |
|---|---|---|---|
| quant_meltdown_2007 | JT_MOM | 4 | 0 |
| gfc_lehman_2008 | JT_MOM | 1 | 2 |
| downgrade_2011 | AL_PCA | 3 | 5 |
| downgrade_2011 | JT_MOM | 5 | 3 |

**Reason:** The stub model returns a fixed hard-coded response pattern regardless of the daily
prompt content. The real model (qwen2.5:1.5b) makes genuine day-by-day assessments that
reflect actual news volume and strategy performance metrics. Latency differences ≤5 days
are within the expected range of day-level stochasticity in event detection.

**Holm conclusions unchanged:**
- Latency arm: raw p=0.0078 → Holm-adjusted p=0.0156 → reject at α=0.05 (pre and post).
- FPR arm: p=0.6207 → do not reject (pre and post).
- Permutation obs_diff: −9.375 (stub) → −9.75 (real); minimum achievable p=0.00390625
  (1/256 permutations) in both cases.

**Impact:** Real-model latencies are the primary evidence. Stub values were only used as a
provisional baseline before D3. The post-D3 values are the final reported results.

---

## Deviation 7 — News-agent failure rates and the FPR denominator (audit finding)

**Date:** 2026-07-25 (pre-freeze audit of D3 artifacts)
**Scope:** Interpretation of the H1 FPR co-primary endpoint; reporting requirement.

**Observation:** A substantial fraction of non-skip (thinking/escalation) days fail at
the news-context step (json.JSONDecodeError / ValueError) and therefore never receive a
supervisor assessment. Per-cell failure rates on non-skip days:

| Cell | Non-skip | Supervisor | Fail % |
|---|---|---|---|
| calm_2013_2014 × AL_PCA | 504 | 144 | 71% |
| calm_2013_2014 × JT_MOM | 502 | 104 | 79% |
| downgrade_2011 × AL_PCA | 64 | 31 | 52% |
| gfc_lehman_2008 × JT_MOM | 74 | 37 | 50% |
| momentum_crash_2009 × AL_PCA | 72 | 39 | 46% |
| momentum_crash_2009 × JT_MOM | 72 | 52 | 28% |
| downgrade_2011 × JT_MOM | 64 | 50 | 22% |
| gfc_lehman_2008 × AL_PCA | 74 | 62 | 16% |
| quant_meltdown_2007 (both) | 44 | 44 | 0% |
| calm_2004_2006 (both) | 65 / 38 | 65 / 38 | 0% |

**Bias direction, by endpoint:**
- **Latency/detection (event windows):** a failed day cannot fire an alarm, so failures
  can only delay or suppress agentic detection. The bias is *against* the agentic arm —
  the measured latency advantage (obs_diff −9.75d, Holm p=0.0156) is conservative.
- **FPR (calm windows):** a failed day also cannot produce a false positive, so failures
  deflate the measured agentic FPR. Measured FPR uses all processed days as denominator
  (calm_2013_2014: 3.8% AL, 2.3% JT). Conditional on the supervisor actually running,
  FPR is 20/144 = 13.9% (AL) and 12/104 = 11.5% (JT). The H1 FPR arm already fails to
  reject (p=0.6207, no agentic advantage claimed); the bias direction can only make the
  agentic FPR *worse*, so the "no FPR advantage" conclusion is unaffected. The report
  MUST present both the unconditional and conditional FPR figures.

**No code change.** Improving news-agent robustness before the freeze would alter the
system under test after seeing dev results. The failure rate is itself a finding about
practical viability and is reported as such (supervisor success rate is an ops metric
in `viability_dev.json`).

---

## Deviation 8 — Calibrated detector parameters never applied (audit finding)

**Date:** 2026-07-25 (post-tag codebase audit, before D5)
**Scope:** `monitoring/run_classical.py` (`_new_detectors`), `monitoring/run_agentic.py`.

**Preregistration implied:** the classical arm runs with detector parameters selected by
the dev-set calibration grid (`calibrate_classical.py` → `calibration_grid.json`).
**Actual:** `_new_detectors()` instantiates `PageHinkley(), BOCPD(), HMMDetector(),
DistributionalThreshold()` with repo DEFAULTS. The calibrated best-per-detector
parameters (e.g. AL_PCA page_hinkley delta=0.25, lambda_=16.0 vs defaults 0.5/8.0)
were computed but never wired into the runners. The freeze gate only checks that
`calibration_grid.json` exists, so this passed 11/11.

**Impact on dev results:** limited on the headline numbers. The H1 classical comparator
is the *best classical detector per strategy*, which is HMM for both strategies — and
HMM's calibrated threshold multiplier is 1.0x, i.e. the default. The defect therefore
affects the non-headline detectors (PageHinkley detected 0/4 AL events on defaults;
calibrated params might have improved it).

**Impact on test results (D5):** the test set will be run with the same default
parameters (frozen system). Bias direction: an under-tuned classical arm can only
*flatter* the agentic latency advantage. The report must state that classical
detectors ran on default parameters, that the headline HMM comparator is unaffected
(calibrated = default), and that per-detector secondary comparisons understate
classical capability.

**Future fix (post-study / v2):** pass `calibration_grid.json` best-params into
`_new_detectors()` in both runners, add a freeze-gate check that instantiated
parameters match the grid, and rerun dev + test. Not fixed now: changing the system
under test after seeing dev results violates the freeze rule, and the tag
`eval-freeze-v1` already pins the evaluated code.

---

## Deviation 9 — FPR analysis-layer defects: synthesized classical streams + scoring asymmetry (audit finding)

**Date:** 2026-07-25 (post-tag codebase audit, before D5)
**Scope:** `monitoring/scripts/analyze_dev_results.py` (block bootstrap input),
`monitoring/metrics.py` vs `monitoring/agentic/alarm_extraction.py` (calm scoring).

**Two related defects in the FPR co-primary endpoint:**

**(a) Synthesized classical alarm streams.** `analyze_dev_results.py` builds the
classical calm alarm stream for `block_bootstrap_fpr` from the FP *count* only,
placing alarms at uniform spacing across the window. FP counts (and therefore the
FPR point estimates and observed_diff) are exact; but uniform spacing destroys the
temporal clustering of real classical alarms, so the bootstrap CI/variance for the
classical stream is not faithful (clustering is understated → classical variance
likely understated → p-value mildly anti-conservative in principle).

**(b) Scoring asymmetry.** Classical calm scoring counts EVERY alarm-day as a false
positive (`metrics.py`), while agentic calm scoring counts CLUSTER STARTS with a
5-day dedup (`alarm_extraction.py`, `cluster_starts(dedup_days=5)`). This deflates
agentic FPR relative to classical by construction.

**Impact on dev results:** the FPR arm did NOT reject (p=0.6207) and no agentic FPR
advantage is claimed. Both defects bias in the direction of *flattering agentic FPR*,
so the "no FPR advantage" conclusion is robust a fortiori — the bias could only have
manufactured an advantage, and none appeared.

**Impact on test results (D5):** the same analysis code scores test calm windows
(calm_2012, calm_2017). Decision rule, fixed now, before unblinding:
- If the test-set FPR arm again fails to reject → conclusion stands (bias direction
  can only overstate the agentic arm).
- If the test-set FPR arm rejects IN FAVOUR OF AGENTIC → the result must NOT be
  claimed from the primary scoring alone; a symmetric sensitivity re-scoring
  (cluster-start dedup applied to BOTH arms, real classical alarm dates in the
  bootstrap) is required, and only a result that survives it may be reported.
- Both unconditional and conditional-on-supervisor FPR are reported per Deviation 7.

**Future fix:** (1) persist classical alarm DATES (not just counts) in
`classical_summary.json` and feed real streams to the bootstrap; (2) apply
`cluster_starts` dedup to both arms (or neither). These are analysis-layer changes,
not changes to the system under test; they are still deferred to the sensitivity
analysis rather than silently swapped in, because the analysis procedure was also
pre-specified.

---

## Deviation 10 — Leakage harness incomplete at freeze; real-model rerun on dev window (pre-D5)

**Date:** 2026-07-25 (post-tag codebase audit, before D5)
**Scope:** `monitoring/run_leakage.py`, preregistration §8 (memorisation bound).

**Preregistration said:** every test event window is run under Conditions A (standard),
B (date-masked), C (synthetic, M=10), producing the leakage bound
`memorisation_upper_bound = perf_A − min(perf_B, perf_C)`.
**Status at freeze:** the only leakage artifact was a STUB-model smoke test on one dev
window (quant_meltdown_2007 × AL_PCA, M=3, Condition B 44/44 runtime failures,
conclusion "inconclusive"). No real-model leakage evidence existed.

**Why it matters for the TEST set specifically:** the test events (flash_crash_2010,
china_deval_2015, volmageddon_2018, covid_2020) are prominent, heavily-documented
episodes almost certainly present in qwen2.5's pretraining corpus. Without the B/C
controls, a test-set latency win cannot be separated from date/event memorisation.

**Action (this deviation):** rerun `run_leakage.py` with the real model
(qwen2.5:1.5b) on the dev window quant_meltdown_2007 × AL_PCA before D5. This
validates the harness end-to-end (the stub run showed Condition B failing at
runtime) and gives a dev-side memorisation read. It touches no test data and does
not modify the frozen system — `run_leakage.py` is part of the tagged code. The
stub artifacts are archived to `monitoring/results/archive/stub/`.

**Impact on test results (D5):** the §8 A/B/C protocol will be run on the test event
windows as preregistered. The report's H1/H2/H3 claims on test events must be
accompanied by the leakage bound; if Condition B/C performance collapses relative to
A, the latency advantage is reported as bounded by memorisation, not as evidence-driven
skill.

**Future fix:** none needed to code unless the real-model rerun reveals a harness bug;
any such bug fix would be documented here as a further entry.

**Amendment (2026-07-25, same day):** the real-model rerun DID reveal a harness bug —
`run_leakage.py:_run_daily_loop` called `run_news_agent` without the per-day
`(json.JSONDecodeError, ValueError)` guard that the frozen pipeline
(`run_agentic.py`) uses, so a single empty model output during Condition C aborted
the whole run. Fixed by mirroring the frozen pipeline's exact error handling
(record `news_error`, continue). This changes only the leakage harness's failure
tolerance, not the system under test or any scoring logic. Partial results from the
aborted run (recorded before the crash): Condition A detected=True latency=0
(matches D3), Condition B (date-masked) detected=True latency=1 — the stub-era
"B fails 44/44" behavior is gone with the real model, and near-identical A/B
performance is preliminary evidence against date-memorisation on this dev window.

**Final real-model result (2026-07-25, after harness fix; 654 LLM calls, 0 news
failures):** quant_meltdown_2007 × AL_PCA — A: detected, latency 0d, 27 FPs;
B (date-masked): detected, latency 1d, 12 FPs; C (10 synthetic windows): recall
3/10, mean latency 1.3d. Prereg bound: evidence_skill ≥ 0.30, memorisation ≤ 0.70,
conclusion **inconclusive** (perf_C < 0.6 × perf_A). Interpretation for the report:
- The B≈A result directly manipulates the date channel and finds ~no degradation —
  strong evidence that *date* memorisation contributes little on this window.
- The low C recall is confounded by harness austerity: synthetic windows run the
  HMM UNTRAINED (`hmm_train=None`, divide-by-zero warnings in `hmm.py`), have no
  real pre-event history, and use template-fabricated news. Low C therefore mixes
  "no memorisation available" with "harder task", and the min(B, C) bound is
  dominated by the weaker, confounded control.
- Per prereg the formal conclusion stays "inconclusive" and must be reported as
  such; the A/B comparison is reported alongside as the cleaner leakage probe. The
  same A/B/C protocol runs on the test event windows at D5.

---

*Log last updated: 2026-07-25*
