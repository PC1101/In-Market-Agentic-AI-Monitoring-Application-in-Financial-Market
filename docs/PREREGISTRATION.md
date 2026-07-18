# Pre-Registration: Confirmatory Evaluation of Agentic vs. Classical Monitoring

**Status: DRAFT v0.1 — for review by both project members and the supervisor.
Nothing in Section 3's test set may be run until this document is approved and
the freeze tag exists.**

| | |
|---|---|
| Project | In-Market Agentic AI Monitoring (Financial Markets) |
| Authors | Paul Chen, Oscar Pham |
| Drafted | 2026-07-18 |
| Approved | *(pending — supervisor signature/date)* |
| Freeze tag | `eval-freeze-v1` *(to be created at approval; see §9)* |

---

## 1. Hypotheses and endpoints

- **H1**: agentic monitoring detects strategy regime breaks *faster* (lower
  detection latency) **and** with a *lower false-positive rate* than classical
  monitoring. H1 has **two co-primary endpoints — latency and FPR — and both
  must individually reject at Holm-adjusted α = 0.05** on the blind test set.
  Recall is a supporting (secondary) endpoint: a monitor that misses events
  cannot win on latency alone, so latency is scored with non-detections as
  censored observations (§5), which penalises misses within the latency
  endpoint itself.
- **H2**: the advantage generalises across both strategies (AL PCA, JT
  momentum) — assessed as consistency of the H1 effect signs per strategy.
- **H3**: the advantage differs by strategy — assessed as a strategy × method
  interaction (§7.4).

## 2. Design overview

A development/test split in which the split follows the existing
contamination. All windows that any design decision has ever touched — the
four famous VRI event windows and both original calm windows — become the
**development set**, on which both arms are calibrated freely. The
**confirmatory test set** consists of event and calm windows that no pipeline
run, prompt iteration, or results inspection has ever touched. Both arms are
frozen (git tag) before the first test-set run.

This replaces the earlier framing of "a-priori vs. calibrated" with a
symmetric protocol: **both arms are calibrated, on the same data, then both
are frozen and evaluated blind.**

## 3. Windows

### 3.1 Development set (contaminated — calibration and descriptive results only)

| window | kind | onset | contamination |
|---|---|---|---|
| quant_meltdown_2007 | event | 2007-08-06 | prompt iteration v1→v3; pipeline smoke tests |
| gfc_lehman_2008 | event | 2008-09-15 | onset assessments run and inspected (llama3.2:3b benchmark + E2E) |
| momentum_crash_2009 | event | 2009-03-09 | same |
| downgrade_2011 | event | 2011-08-05 | same |
| calm_2004_2006 | calm | — | triage tallies inspected; coverage-gate iteration |
| calm_2013_2014 | calm | — | triage tallies inspected (never-SKIPs anomaly known) |

### 3.2 Confirmatory test set (blind — no runs before freeze)

| window | kind | proposed range | proposed onset | strategies |
|---|---|---|---|---|
| flash_crash_2010 | event | 2010-04-15 – 2010-07-15 | 2010-05-06 | AL PCA + JT |
| china_deval_2015 | event | 2015-07-15 – 2015-10-16 | 2015-08-11 | AL PCA† + JT |
| volmageddon_2018 | event | 2018-01-15 – 2018-04-13 | 2018-02-05 | JT only |
| covid_2020 | event | 2020-02-03 – 2020-05-29 | 2020-02-24 | JT only |
| calm_2012 | calm | 2012-01-03 – 2012-12-31 | — | AL PCA + JT |
| calm_2017 | calm | 2017-01-03 – 2017-12-29 | — | JT only |

† Conditional on the AL PCA PIT curve covering the full window (curve ends
2015/16); verified as a pre-freeze checklist item (§11), *before* any agent
runs on it.

Onset labels above are proposed from event dates (flash crash day; CNY
devaluation announcement; XIV collapse day; first COVID selloff day) and are
**fixed at approval** — they may be corrected by the supervisor before the
freeze, never after.

## 4. Contamination register (why the split is what it is)

1. Prompts were iterated (v1→v3) while observing quant_meltdown_2007 output.
2. The llama3.2:3b benchmark and the full 6-window E2E were run and inspected
   by all project members (2026-07-16) — every original window's results are
   known.
3. Detector *defaults* were set from synthetic calibration only, but their
   published Week-2 metrics on all six windows are known — so any *future*
   manual re-tuning would be in-sample. The grid protocol (§6.1) replaces
   manual tuning.
4. No member has ever run any pipeline on, or inspected model output for, the
   §3.2 windows. This claim is part of the sign-off.

## 5. Detection and scoring rules (both arms, fixed now)

- **Classical detection**: an aggregate alarm (≥2 distinct detectors within 5
  days, 5-day cooldown) in `[onset, onset+21 trading days]`.
- **Agentic detection**: first day in `[onset, onset+21 trading days]` whose
  supervisor assessment has `state ∈ {ALERT, CRITICAL}`.
- **Latency**: trading days from onset to first detection; non-detection is
  censored at 21 days (§7.2).
- **False positive**: a detection-cluster start (classical alarm, or agentic
  ALERT/CRITICAL day with no ALERT/CRITICAL in the prior 5 trading days) on a
  calm window. FPR/day = clusters ÷ trading days.
- **Agentic runtime failures** (schema-invalid after one repair retry, or
  `CLASSICAL_ESCALATION` due to LLM failure) count as *no detection* for the
  agentic arm on event days and are reported; the agentic arm may not silently
  inherit classical alarms it escalated to.

## 6. Calibration protocol (development set only)

### 6.1 Classical arm

- Grid: for each detector, its key sensitivity parameters at
  {0.5×, 1×, 2×} of current repo defaults — Page-Hinkley (`delta`, `lambda_`),
  BOCPD (`hazard`, `min_run`), HMM (stressed-state `threshold`), distributional
  (z-threshold, vol-ratio threshold). Full factorial within detector;
  aggregation rule itself stays fixed (≥2 in 5d).
- Objective, computed on the dev set per strategy:
  **J = recall − 50 × FPR/day**, tie-break lower mean (censored) latency.
  (Scale rationale: 50 trading days ≈ trading one extra false alarm per ~year
  against catching one more event in four.)
- Primary configuration is **per-strategy** (matches deployment practice);
  the shared-config result is also recorded for the H3 analysis (§7.4).

### 6.2 Agentic arm

- Frozen components: prompts `supervisor-v3`, `news-context-v1`,
  `news-context-v1-thinking`; triage constants (STRESS_CHEAP = 0.30,
  STRESS_THINKING = 0.60, HIT_Z_THINKING = 2.0, RECENT_DAYS = 3,
  COVERAGE_DAYS = 7); FinBERT stress scorer; model `llama3.2:3b` via Ollama,
  temperature 0, grammar-constrained structured outputs, one repair retry;
  T-1 news cutoff and all lookahead guardrails.
- Remaining dev-set iteration budget before freeze: **at most 10 further
  prompt/threshold variants**, each version-tagged and selected by the same J
  as §6.1. After the freeze tag: zero.

## 7. Statistical analysis plan (test set)

Analysis code (`monitoring/significance.py`) is written and tested against
*development* logs before any test run exists (§9).

1. **Recall (secondary)**: exact McNemar test on paired detection outcomes,
   paired by window × strategy (6 pairs: 2010×2, 2015×2†, 2018, 2020).
   Independence caveat (shared event dates across strategies; daily PnL
   correlation ≈ 0.003) reported alongside.
2. **Latency (co-primary)**: exact permutation test (all 2^n sign
   assignments) on the paired difference in restricted mean detection time,
   censored at 21 trading days.
3. **FPR (co-primary)**: difference in FPR/day on test calm windows with a
   moving-block bootstrap (block = 10 trading days, 10,000 resamples) CI;
   Fisher's exact test reported as a descriptive companion.
4. **H2/H3**: per-strategy effect signs (H2); strategy × method interaction
   on latency and FPR via the same permutation/bootstrap machinery (H3).
5. **Placebo onsets**: K = 1,000 pseudo-onsets sampled uniformly in test calm
   windows (≥42 trading days apart, ≥21 days from window edges); empirical
   null for "detection within 21 days" for both arms. The agentic arm must
   beat its own placebo rate — this is the direct test against
   "the monitor is just permanently alarmed."
6. **Bayesian companion**: Beta-Binomial (Jeffreys prior) posteriors on
   recall for both arms; report P(recall_agentic > recall_classical | data)
   and 95% credible intervals — the honest headline given small n.
7. **Multiplicity**: Holm correction across the two co-primary endpoints.
   Secondary/companion analyses are reported unadjusted and labelled as such.

## 8. Pretraining-leakage controls (agentic arm, test set)

Temporal out-of-sample is not out-of-corpus: all test events predate the
LLM's training cutoff. Three conditions are run on every test event window:

- **A. Standard** — the frozen pipeline as-is.
- **B. Date-masked** — identical contexts with all dates removed from prompts
  (telemetry, alarms, headlines retained).
- **C. Synthetic** — M = 10 synthetic event windows: block-bootstrapped calm
  returns with injected crash templates (drawdown/vol scaled to dev-event
  severities) and template-generated stress headlines at fabricated dates.

Reported bound: evidence-driven skill ≥ performance on C; memorisation
component ≤ (A − min(B, C)). If A ≫ C, the write-up states that real-event
performance is materially attributable to pretraining memory. The two-pass
with/without-events control (Week-4 VRI item) covers the classical/HMM
training-span analogue.

## 9. Freeze and blinding mechanics

1. This document is approved → both arms' final configs land on the
   integration branch → annotated tag **`eval-freeze-v1`**.
2. `significance.py` and the leakage-control harness are merged and tested
   against dev logs *before* the tag.
3. Test-set runs execute on the frozen tag only (Oscar's GPU). No config,
   prompt, threshold, or scoring change after the tag. Re-runs of the same
   frozen config (e.g. crash recovery) are permitted and logged.
4. Any deviation is recorded in the **Deviations** appendix below with date,
   reason, and impact — before results of the deviated run are inspected.

## 10. Reporting commitments

- All §7 endpoints are reported regardless of direction or significance.
- Development-set results are always labelled *descriptive/calibration*;
  §3.2 results are the only ones labelled *confirmatory*.
- The §8 leakage bound is reported next to every confirmatory agentic result.
- Known limitations reported: n = 4 test events (6 window×strategy pairs);
  JT-only windows limit H2/H3 on 2018/2020; FNSPID coverage differences
  across eras; 4 GB VRAM confines the "thinking" tier to the same 3B model.

## 11. Pre-freeze checklist (blockers for the tag)

- [ ] Supervisor approval of this document, incl. test windows and onsets
- [ ] AL PCA PIT curve coverage verified through 2015-10-16 (else 2015 → JT-only)
- [ ] News caches built for all §3.2 windows (FNSPID post-2009 + NYT months)
- [ ] Branch convergence: `oscar` → integration branch; PR #2 closed as superseded
- [ ] Macro block wired into supervisor-v3 prompts (or explicitly deferred, recorded here)
- [ ] `significance.py` + leakage harness merged, tested on dev logs
- [ ] Classical grid calibration run on dev; winning configs recorded here
- [ ] Agentic dev iterations (≤10) finished; final prompt versions recorded here

## Deviations

*(none)*
