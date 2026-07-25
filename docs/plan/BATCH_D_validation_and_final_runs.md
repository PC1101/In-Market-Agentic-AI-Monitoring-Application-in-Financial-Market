# BATCH D — Pipeline Validation, Dev Rerun, Test-Set Unblinding

**Goal:** confirm the experiment has no biases or setup errors, demonstrate practical
viability, then produce the final, provenance-clean dev results and the one-shot
blinded test results for the report.

**Prerequisite:** Batch C complete and verified — real-model calm FPR (4 cells with
`run_meta` headers), PIT curve extension, analyzer hard-fails on stub calm data,
`freeze_gate.py` all PASS.

**Ground rules (apply to every phase):**
- One Ollama consumer at a time. Check GPU is active (~4s/call, not ~40s) before long runs.
- Never delete anything in `monitoring/results/` — archive to `results/archive/<reason>/` via `git mv`.
- `run_agentic.py` unlinks its output file on start: never re-launch a completed cell; use the resume-safe orchestrators only.
- No commits, no pushes, no tags unless the step explicitly says so AND the user confirms.
- Every anomaly or judgment call → append to the deviations log in `docs/`.

---

## Phase D1 — Correctness & bias audit (read-only, no LLM, ~minutes)

The point: prove, from artifacts, that nothing in the scored data could be contaminated.

1. **Provenance audit.** For every `agentic_*.jsonl` in `monitoring/results/` (not archive):
   first line must be `run_meta` with `model == "qwen2.5:1.5b"`, a `*_pit` curve path,
   and a git SHA. Expected finding: the 8 pre-Batch-A event JSONLs have NO run_meta —
   record this; it is the formal trigger for the Phase D3 rerun.
2. **Coverage matrix.** Build a table (window × strategy × {curve coverage, article count
   in news store, macro vintage coverage}) for ALL dev + test windows.
   - AL_PCA: confirm which test windows the longest PIT curve covers (expected:
     flash_crash_2010, china_deval_2015, calm_2012 at most; 2018/2020/2017 impossible —
     sleeve prices end 2016-12-31). Confirm this matches `TEST_STRATEGY_WINDOWS`.
   - **News store range check (potential BLOCKER):** verify the FNSPID/NYT store has
     articles for 2017, 2018 Q1, and 2020 H1 (JT test windows). If coverage is absent or
     thin, triage behavior on those windows is undefined — resolve (extend the store)
     BEFORE the freeze tag, and document.
3. **Lookahead spot-check.** Sample ≥20 supervisor prompts from dev JSONLs across windows;
   assert no date string > `as_of` appears in any prompt (script it — regex all ISO dates
   in the prompt text, compare to as_of). Verify `assert_no_lookahead` is called on every
   path in `run_agentic.run_window` and `run_leakage._run_daily_loop`.
4. **Calibration hygiene.** Confirm `calibrate_classical.py` reads only DEV windows;
   confirm `calibration_grid.json` content matches what `run_classical.py` applies;
   confirm `freeze_gate` blinding check passes (no test artifacts anywhere, including
   `results/archive/`).
5. **Symmetry check.** Confirm classical and agentic arms are scored with identical
   machinery: same `cluster_starts` dedup (5 days), same tolerance (21 days), same
   trading-day calendars, same onset (and same `override_onset` in the sensitivity pass).
   Grep `analyze_dev_results.py` and confirm no arm-specific special-casing.

**Output:** `docs/plan/D1_audit_report.md` — table of checks, pass/fail, findings.
Any FAIL here blocks all later phases.

## Phase D2 — Leakage / memorisation control validation (LLM, moderate)

The §8 leakage harness runs on TEST windows and therefore belongs post-freeze (D5).
Here we only validate its mechanics on DEV data so it works first-time under the freeze.

1. Stub dry-run: `python run_leakage.py --strategy AL_PCA --model stub` restricted to one
   dev event window (add a `--window` restriction flag if absent — a small, tested code
   change is allowed pre-freeze). Verify it produces the A/B/C metrics and the
   memorisation bound JSON without error.
2. Real-model smoke: same single dev window with `--model ollama:qwen2.5:1.5b`. Verify
   condition B prompts actually contain masked dates (`XXXX-XX-XX`) by inspecting the log,
   and condition C synthetic windows contain no real dates/tickers.
3. Record per-condition runtime → estimate full test-set leakage cost for D5 planning.
4. Optional but recommended for the report: full condition-B (date-masked) rerun of the
   4 dev event windows × 2 strategies. If dev A ≈ dev B (detection/latency), that is
   direct evidence the monitor is evidence-driven, quotable before test unblinding.

**Output:** leakage harness validated + dev A-vs-B comparison in `dev_analysis.json`
(new key `leakage_dev`, additive only — primary numbers must remain bit-identical).

## Phase D3 — Dev-set rerun (final dev results; LONG, hours)

**Why rerun:** (a) existing event JSONLs lack run_meta (cannot prove model/curve),
(b) Batch C may have changed the preferred AL_PCA curve (baseline_2007_2016_pit),
so inputs differ from the original runs. Both are disqualifying for a report.

1. Archive current 8 event JSONLs → `results/archive/pre_freeze/` (`git mv`).
2. Verify `run_remaining_events.py` resume counter tolerates `run_meta` lines
   (`count_supervisor_days` filters by agent name — confirm with a unit test if absent).
3. Run all 8 event cells (4 dev event windows × 2 strategies), real model, sequential,
   via the resume-safe orchestrator. Calm cells from Batch C are kept (already have
   run_meta) — do NOT rerun them.
4. Regenerate `dev_analysis.json`. Compare against the archived version:
   - Same detection pattern and conclusions → note "reproduced under provenance".
   - Different → investigate before proceeding; document in deviations log. Do NOT
     iterate/rerun to chase better numbers — one rerun, whatever it says.
5. Rerun `analyze_dev_results.py` onset-sensitivity and Holm sections on the new data.

**Acceptance:** all 12 dev cells (8 event + 4 calm) carry run_meta with real model +
PIT curve; `dev_analysis.json` final; conclusions vs archived version documented either way.

## Phase D4 — Practical viability assessment (analysis only, cheap)

For the report's "is this usable in practice" section. Dev data only. Explicitly
labelled exploratory (not preregistered).

1. **Ops metrics from JSONLs:** per-day wall-clock (p50/p95 of `latency_s`), triage
   skip-rate per window, escalation mix, runtime-failure rate, total LLM calls per
   trading day. Verdict criterion: can the daily loop complete comfortably within a
   pre-open window on commodity hardware.
2. **Economic simulation:** simple pre-declared rule — on ALERT scale exposure to 50%,
   on CRITICAL to 0%, restore after N=5 clean days. Apply to the dev PnL curves with
   next-day execution lag. Report: max-drawdown avoided, annualised return drag from
   false positives, same for the classical-alarm rule and a no-monitor baseline.
3. **Failure-mode review:** read every FP cluster and every late/missed detection in the
   dev set; 1-paragraph qualitative note each (news absent? triage skipped? model
   misread?). Goes straight into the report's limitations section.

**Output:** `results/viability_dev.json` + a short markdown section for the report.

## Phase D5 — Freeze, then test-set one-shot (LONG; run attended)

**Order is mandatory. Nothing below the tag happens before the tag.**

1. `python freeze_gate.py` → all PASS. Finalize the deviations appendix. User reviews
   and commits everything. Tag `eval-freeze-v1` (user action or explicit user approval).
2. **Classical arm on test windows:** frozen thresholds from `calibration_grid.json`,
   zero recalibration. Both strategies where curve coverage exists (JT: all 6;
   AL_PCA: per `TEST_STRATEGY_WINDOWS`).
3. **Agentic arm on test windows:** resume-safe orchestrator (extend
   `run_calm_windows.py` pattern), real model, sequential. Runtime failures: resume,
   never restart; count them via `count_runtime_failures` and report them.
4. **Leakage harness (§8):** conditions A/B/C on the test event windows, both strategies
   where covered. Report `evidence_skill_lower_bound` and `memorisation_upper_bound`.
5. **Single scoring pass** → `results/test_analysis.json`: same analyzer machinery,
   Holm-corrected co-primary endpoints (latency + FPR) for H1, per-strategy splits for
   H2/H3, onset sensitivity as secondary.
6. **Hard rules:** no prompt edits, no threshold changes, no second runs, no peeking at
   partial results to alter anything. Whatever the numbers are, they go in the report.
   Any unavoidable deviation → deviations log with timestamp and justification.

## Phase D6 — Report assembly

1. Tables: per-window detection/latency/FPR (dev and test, both arms, both strategies);
   H1/H2/H3 verdicts with test statistics and Holm-adjusted p-values.
2. Leakage bound section (dev A-vs-B + test A/B/C).
3. Onset sensitivity section (hardcoded vs curve-derived, dev and test).
4. Viability section (D4).
5. Deviations appendix (complete list: qwen2.5:1.5b model swap, any curve changes,
   AL_PCA test-coverage asymmetry, anything from D3/D5).
6. Limitations: AL_PCA cannot be evaluated on 2017–2020 test windows; single LLM;
   single news source; dev-set-tuned triage constants.

---

## Execution order & cost profile

```
D1 audit            minutes    read-only          BLOCKER gate
D2 leakage-val      ~1-2 cells LLM smoke          can overlap D1
D3 dev rerun        hours      8 real-model cells needs D1 pass; attended start
D4 viability        minutes    analysis only      after D3 (uses final dev JSONLs)
D5 freeze + test    hours      ~10+ cells + leakage  needs D1–D4 + user tag approval
D6 report           writing    —                  after D5
```

## Known open questions (resolve in D1, before freeze)
1. Does the news store cover 2017 / 2018-Q1 / 2020-H1? If not → extend before freeze.
2. Does `TEST_STRATEGY_WINDOWS` match actual AL_PCA curve coverage after Batch C's
   extension? Reconcile and document the final per-strategy test roster.
3. Were HMM training returns for test windows drawn strictly pre-window (no test-period
   data in detector training)? Verify `_hmm_training_returns` cutoffs for 2018/2020.
