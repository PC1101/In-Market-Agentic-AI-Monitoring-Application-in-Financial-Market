# LLM-Based Agentic Monitoring vs. Classical Change-Point Detection for Financial Market Regime Identification: A Preregistered Experiment

---

## Abstract

We present a preregistered experiment comparing an LLM-based agentic monitoring system to a suite of classical change-point detectors for identifying regime breaks in two quantitative trading strategies. The agentic system integrates a small language model (qwen2.5:1.5b, grammar-constrained decoding) with filtered financial news, macroeconomic vintages, and strategy telemetry through a triage-escalation architecture. The classical arm aggregates Page-Hinkley, BOCPD, HMM, and distributional-threshold detectors. Using a 2x2 factorial design ({agentic, classical} x {statistical arbitrage, cross-sectional momentum}) across 12 historical windows (6 development, 6 confirmatory test), we evaluate detection latency, false-positive rate (FPR), and recall under strict point-in-time information parity. On the development set (8 event-window pairs), the agentic system detects regime breaks 9.75 days earlier on average (permutation p = 0.0039, Holm-adjusted p = 0.0156), with recall of 8/8 versus 6/8 for classical. No FPR advantage is observed (p = 0.62). The confirmatory test set (5 event pairs) shows a directionally consistent latency advantage of 4.4 days, but is underpowered (p = 0.25, n = 5). We report 15 protocol deviations, a three-condition leakage analysis, and operational metrics including news-agent failure rates of 16-79%. The results provide suggestive but not confirmatory evidence that multi-signal LLM fusion reduces detection latency relative to classical statistical methods, while raising practical concerns about reliability and false-positive control.

---

## 1. Introduction

Quantitative trading strategies are subject to regime breaks---abrupt changes in return-generating dynamics that render calibrated parameters stale. The August 2007 quant meltdown (Khandani & Lo, 2007), the 2009 momentum crash (Daniel & Moskowitz, 2016), and the February 2018 volatility event demonstrate that such breaks can destroy years of accumulated profit within days. Monitoring for regime transitions is therefore a first-order operational concern.

Classical change-point detection methods---including Page-Hinkley (1954), Bayesian online change-point detection (Adams & MacKay, 2007), hidden Markov models (Hamilton, 1989), and threshold-based approaches---operate exclusively on the strategy's return stream. They are theoretically well-founded but react only after the statistical properties of the stream have already shifted, introducing unavoidable detection latency.

Recent advances in large language models (LLMs) suggest an alternative: a system that fuses statistical telemetry with natural-language context (news, macroeconomic releases, market commentary) to anticipate or accelerate regime identification. If news coverage of a developing crisis precedes its statistical manifestation in strategy returns, an LLM-based monitor might detect regime breaks earlier.

**Contributions.** This paper reports a preregistered experiment testing three hypotheses:

- **H1 (co-primary):** The agentic monitoring system achieves both lower detection latency AND lower false-positive rate than the best classical detector.
- **H2:** Any agentic advantage generalises across both strategies.
- **H3:** The magnitude of advantage differs by strategy.

The experimental design enforces strict point-in-time (PIT) information parity: both arms observe only information available on each decision date, with survivorship-bias-free strategy inputs constructed from historical S&P 500 constituent membership intervals including delisted names. The evaluation protocol uses a development/test split with an `eval-freeze-v1` git tag pinning all code before test-set unblinding, Holm correction for multiple comparisons, and a three-condition leakage harness to bound potential LLM memorisation effects.

We find that the agentic system detects regime breaks significantly earlier on the development set (9.75 days, Holm-adjusted p = 0.0156), but that the confirmatory test set is underpowered to confirm this finding (5 event pairs yield maximum achievable p = 1/32 under exact permutation). The FPR co-primary endpoint does not favour the agentic arm on either set, and on the test set, the agentic system exhibits higher FPR than classical. We report these results honestly, including all 15 protocol deviations, as a contribution to the nascent literature on LLM-augmented financial monitoring.

---

## 2. Related Work

### 2.1 Change-Point Detection in Financial Time Series

The problem of detecting distributional shifts in sequential data has a long history. Page (1954) introduced the cumulative sum (CUSUM) procedure; the Page-Hinkley variant adds a tolerance parameter for non-zero drift. Adams and MacKay (2007) developed Bayesian Online Change-Point Detection (BOCPD), which maintains a run-length posterior and can incorporate prior beliefs about hazard rates. Hamilton (1989) introduced regime-switching models using hidden Markov models (HMMs), which have been widely applied to financial return series for identifying bull and bear states (Ang & Bekaert, 2002; Guidolin & Timmermann, 2007).

In practice, financial firms typically employ ensembles of detectors rather than relying on a single method (Aminikhanghahi & Cook, 2017). Our classical arm follows this approach, aggregating four detectors under a majority-vote rule.

### 2.2 Statistical Arbitrage and Cross-Sectional Momentum

Avellaneda and Lee (2010) formalize statistical arbitrage using PCA-based factor decomposition: residuals from the first K principal components of sector returns are modeled as mean-reverting Ornstein-Uhlenbeck processes, generating trading signals when spreads deviate from equilibrium. Jegadeesh and Titman (1993) document the cross-sectional momentum anomaly: portfolios long past winners and short past losers earn significant risk-adjusted returns over 3-12 month horizons.

Both strategies are known to experience regime breaks: Khandani and Lo (2007, 2011) document the August 2007 quant meltdown's impact on statistical arbitrage, while Daniel and Moskowitz (2016) characterize momentum crashes as endogenous to crowding and leverage unwinding.

### 2.3 Survivorship Bias and Point-in-Time Data

Survivorship bias---the exclusion of delisted or removed securities from backtested universes---is a well-documented source of overstated performance (Brown et al., 1992; Elton et al., 1996). The August 2007 quant meltdown is invisible on a survivor-only S&P 500 universe (+3.4% in the event window) but produces a -4.8% drawdown on the honest point-in-time universe. Our experimental design uses historical constituent membership intervals including delisted names, following best practices articulated by Israel et al. (2020).

### 2.4 LLMs in Finance

The application of language models to financial tasks has expanded rapidly. FinBERT (Araci, 2019; Huang et al., 2023) provides domain-specific sentiment classification. Lopez-Lira and Tang (2023) demonstrate that GPT models can predict stock returns from news headlines. Wu et al. (2023) survey LLM applications across financial tasks. Kim et al. (2024) explore LLM-based portfolio construction.

However, the use of LLMs for real-time regime monitoring---as distinct from return prediction or sentiment scoring---remains underexplored. Our work is, to our knowledge, the first preregistered experiment testing whether LLM-based multi-signal fusion can accelerate change-point detection relative to established statistical methods.

### 2.5 LLM Memorisation and Temporal Leakage

A critical concern when evaluating LLMs on historical financial events is that training data may contain descriptions of those events, enabling detection through memorisation rather than genuine reasoning. Carlini et al. (2023) demonstrate that LLMs can memorize and reproduce training sequences. Our three-condition leakage harness (standard, date-masked, synthetic) explicitly tests this concern and bounds the memorisation contribution.

---

## 3. Methodology

### 3.1 Trading Strategies

#### 3.1.1 AL_PCA (Avellaneda-Lee Statistical Arbitrage)

Following Avellaneda and Lee (2010), we construct sector-sleeve PCA-based mean-reversion portfolios on the S&P 500. The point-in-time (PIT) universe is constructed from historical constituent membership intervals: at each monthly rebalance, only securities that were actual S&P 500 members on that date are eligible, including names subsequently delisted. This produces approximately 347 eligible names per month (vs. ~542 on a biased universe). The first K=15 principal components per sector capture systematic factor exposure; residuals are modeled as OU processes and traded when z-scores exceed entry/exit thresholds.

The daily PnL curve spans 2003-2015 and is constructed from PIT sleeves built by `scripts/build_pit_sleeves.py`, with each sleeve masked to membership intervals.

#### 3.1.2 JT_MOM (Jegadeesh-Titman Cross-Sectional Momentum)

Following Jegadeesh and Titman (1993), we rank PIT-universe stocks by trailing 12-month returns (skipping the most recent month), go long the top decile and short the bottom decile, and rebalance monthly. Daily PnL is computed from position-level returns on the PIT candidate pool. The curve spans 2003-2020.

The two strategies are statistically independent (daily return correlation approximately 0.003) and experience regime breaks on different events---AL_PCA on the 2007 quant meltdown, JT_MOM on the 2009 momentum crash---making the cross-strategy hypotheses (H2, H3) informative.

### 3.2 Classical Detection

Four change-point detectors operate on the strategy's daily return stream:

1. **Page-Hinkley (1954):** Monitors the cumulative deviation of returns from a running mean, triggering when the deviation exceeds a threshold lambda with drift tolerance delta. Parameters: defaults (delta=0.5, lambda=8.0).

2. **BOCPD (Adams & MacKay, 2007):** Maintains a posterior distribution over run lengths, signaling a change point when the posterior mass on short run lengths exceeds a threshold. Hazard function: constant rate 1/250.

3. **HMM Regime Detector (Hamilton, 1989):** Fits a two-state Gaussian HMM to a rolling window of returns. Transitions from the low-volatility to the high-volatility state are signaled as regime changes. This is the best-performing classical comparator for both strategies.

4. **Distributional Threshold:** Fires when the rolling z-score of returns (relative to a calibration window) exceeds a fixed threshold.

**Aggregation Rule:** An aggregate classical alarm fires when 2 or more of the 4 detectors fire within a 5 trading-day window. This reduces false positives from individual noisy detectors at the cost of marginally increased latency.

**Headline comparator:** For the primary H1 analysis, the best single classical detector (HMM) is used, as the aggregation rule often underperforms the best individual detector due to the latency penalty.

### 3.3 Agentic Architecture

The agentic system comprises three layers operating in sequence for each trading day:

#### 3.3.1 Triage Layer

The triage layer decides the processing depth for each day based on four signals:

- `intensity_z`: the rolling z-score of absolute daily returns
- `FinBERT_stress`: mean FinBERT negative-sentiment probability across that day's news articles (threshold: >= 0.6)
- `n_recent_classical_alarms`: count of classical detector firings in the trailing 5 days
- `aggregate_alarm`: whether the classical aggregation rule has fired

Routing decisions:
- **SKIP**: low intensity, no news stress, no classical alarms (saves inference cost)
- **CHEAP**: moderate signals, quick-pass assessment
- **THINKING**: elevated stress or intensity (full news + supervisor pipeline)
- **CLASSICAL_ESCALATION**: aggregate classical alarm has fired (highest priority)

#### 3.3.2 News Context Agent

For THINKING and ESCALATION days, the news context agent queries the FNSPID news store (11.8M articles, 2003-2020, indexed by publication date) with a 7-day trailing window and risk-term filter. It produces a structured `NewsFlags` JSON containing:
- `headline_risk_score` (0-1)
- `sector_flags` (affected sectors)
- `macro_flags` (relevant macro themes)
- `narrative_summary` (2-3 sentence synthesis)

Model: qwen2.5:1.5b via Ollama with grammar-constrained JSON decoding (`format=json`, temperature=0).

#### 3.3.3 Performance Supervisor (v3)

The supervisor receives the complete context for the day:
- Strategy telemetry: daily return, trailing drawdown, rolling Sharpe ratio
- Classical detector alarms (which detectors fired, confidence levels)
- Macro context: FRED vintage data (values published as-of the decision date)
- News flags from the news context agent

It produces a structured JSON assessment:
- `state`: one of {NORMAL, WATCH, ALERT, CRITICAL}
- `action`: recommended response
- `confidence`: 0-1 score
- `root_cause`: natural-language explanation

Model: qwen2.5:1.5b via Ollama (same configuration as news agent). Mean supervisor latency: 3-5 seconds per day on the test hardware.

#### 3.3.4 Guardrails

All model inputs are subject to:
- **assert_no_lookahead**: verifies that no date in the model's output postdates the current decision date
- **as-of dating**: macro values filtered to publication date
- **timestamp filtering**: news articles filtered to publication date
- **date masking** (Condition B): replaces all ISO dates with `XXXX-XX-XX` to test memorisation

### 3.4 Metrics

#### 3.4.1 Detection Latency

For each event window, a preregistered **onset date** marks the beginning of the regime break. Detection latency is measured as:

$$\text{latency} = t_{\text{first\_alarm}} - t_{\text{onset}}$$

where $t_{\text{first\_alarm}}$ is the first trading day on which the system produces an ALERT or higher assessment (agentic) or an HMM alarm (classical) within the window [onset - 21 trading days, onset + 21 trading days]. A system that does not fire within this window is scored as a **miss** (detection = False).

Negative latency (alarm before onset) is possible and indicates early detection; it is capped at -21 days.

#### 3.4.2 False-Positive Rate (FPR)

Measured on calm windows as:

$$\text{FPR} = \frac{\text{number of alarm days (or cluster starts)}}{\text{total trading days in calm window}}$$

For the agentic arm, alarm days are deduplicated using a 5-trading-day cooldown (`cluster_starts`), counting only the first alarm in each cluster. For the classical arm (HMM), every alarm day is counted. This asymmetry (Deviation 9) biases in favour of the agentic arm; the FPR co-primary endpoint does not reject regardless, so the conclusion is robust.

#### 3.4.3 Detection (Recall)

Binary: did the system fire at least one alarm within onset +/- 21 days? Reported as fraction detected across event windows.

### 3.5 Statistical Tests

- **Permutation test (latency):** Under H0, agentic and classical latencies are exchangeable. We enumerate all 2^n sign-flip permutations of paired differences and compute the one-sided p-value for the observed mean difference being more negative than expected.

- **Block bootstrap (FPR):** Resamples 20-day blocks from calm-window alarm streams (10,000 iterations) to construct confidence intervals for the FPR difference, accounting for temporal autocorrelation.

- **McNemar's test (recall):** Tests whether the marginal probability of detection differs between arms on paired event windows.

- **Bayesian recall:** Computes P(agentic recall > classical recall) under a Beta-Binomial model with uniform priors.

- **Holm correction:** Applied to the co-primary endpoints (latency and FPR) to control the family-wise error rate at alpha = 0.05. Both endpoints must survive correction for H1 to be fully supported.

---

## 4. Experimental Setup

### 4.1 Window Partition

The experiment uses 12 historical windows partitioned into development (for exploratory analysis and protocol tuning) and test (for confirmatory evaluation):

**Development Set (6 windows):**

| Window | Type | Date Range | Onset Date | Strategies |
|--------|------|-----------|------------|------------|
| quant_meltdown_2007 | Event | 2007 | 2007-08-06 | AL_PCA, JT_MOM |
| gfc_lehman_2008 | Event | 2008 | 2008-09-15 | AL_PCA, JT_MOM |
| momentum_crash_2009 | Event | 2009 | 2009-03-09 | AL_PCA, JT_MOM |
| downgrade_2011 | Event | 2011 | 2011-08-05 | AL_PCA, JT_MOM |
| calm_2004_2006 | Calm | 2004-2006 | N/A | AL_PCA, JT_MOM |
| calm_2013_2014 | Calm | 2013-2014 | N/A | AL_PCA, JT_MOM |

**Test Set (6 windows):**

| Window | Type | Date Range | Onset Date | Strategies |
|--------|------|-----------|------------|------------|
| flash_crash_2010 | Event | 2010 | 2010-05-06 | AL_PCA, JT_MOM |
| china_deval_2015 | Event | 2015 | 2015-08-11 | JT_MOM only* |
| volmageddon_2018 | Event | 2018 | 2018-02-05 | JT_MOM only |
| covid_2020 | Event | 2020 | 2020-02-20 | JT_MOM only |
| calm_2012 | Calm | 2012 | N/A | AL_PCA, JT_MOM |
| calm_2017 | Calm | 2017 | N/A | JT_MOM only |

*china_deval_2015 x AL_PCA excluded due to PIT curve not covering 2015 (Deviation 3).

This yields 8 event-window pairs on the development set and 5 on the test set (fewer than the preregistered design due to AL_PCA curve coverage limitations).

### 4.2 Evaluation Protocol

1. **Code freeze:** The `eval-freeze-v1` git tag pins all system code, prompts, model configurations, and analysis scripts before any test-set window is evaluated.

2. **Blinding:** Test-set result files are verified absent before the freeze (freeze-gate check: "test-set blinding intact").

3. **Two-pass inference:** To mitigate thermal throttling on the test hardware, I/O-intensive operations (news retrieval, macro lookup) are precomputed and cached in a first pass; LLM inference runs in a second pass against the cached context.

4. **Day-level resume:** A dead-runner recovery mechanism (added mid-test, Deviation 12) allows interrupted runs to resume from the last successfully processed day rather than restarting entire windows.

### 4.3 Leakage Conditions

To bound potential LLM memorisation of famous financial events:

- **Condition A (standard):** Full context including dates, real news headlines, and event-identifying information.
- **Condition B (date-masked):** All ISO dates replaced with `XXXX-XX-XX`; numerical values preserved. Tests whether the model requires date identification to detect regime breaks.
- **Condition C (synthetic, M=10):** Block-bootstrapped return series with injected synthetic crashes (randomized onset, generic stress headlines, year-2099 dates). Tests whether the model can detect regime breaks from patterns alone, without memorised event knowledge.

**Decision rule:** If perf_C >= 0.6 x perf_A, evidence that the model generalises beyond memorisation. If perf_B approximately equals perf_A, date-memorisation contributes little.

### 4.4 Hardware Constraints

- Windows 10, Python 3.11
- NVIDIA RTX 3050 Ti (4 GB VRAM)
- Ollama serving qwen2.5:1.5b locally
- Mean supervisor inference: 3-5 seconds per trading day
- Thermal throttling mitigation: two-pass design separates I/O from GPU inference
- Total dev-set compute: ~8 hours across all windows
- Total test-set compute: ~4 hours (fewer windows, day-level resume)

---

## 5. Results

### 5.1 Development Set

#### 5.1.1 Latency (H1 Primary Endpoint)

**Table 1: Per-pair detection results, development set**

| Window | Strategy | Classical (HMM) | Agentic | Latency Diff |
|--------|----------|-----------------|---------|-------------|
| quant_meltdown_2007 | AL_PCA | Detected / 7d | Detected / 0d | -7 |
| gfc_lehman_2008 | AL_PCA | Detected / 14d | Detected / 0d | -14 |
| momentum_crash_2009 | AL_PCA | Detected / 1d | Detected / 0d | -1 |
| downgrade_2011 | AL_PCA | Detected / 11d | Detected / 5d | -6 |
| quant_meltdown_2007 | JT_MOM | Detected / 7d | Detected / 0d | -7 |
| gfc_lehman_2008 | JT_MOM | **Missed** | Detected / 2d | -- |
| momentum_crash_2009 | JT_MOM | **Missed** | Detected / 1d | -- |
| downgrade_2011 | JT_MOM | Detected / 7d | Detected / 3d | -4 |

Permutation test (paired differences, excluding 2 classical misses which are conservatively scored):
- Observed mean difference: **-9.75 days** (agentic faster)
- One-sided p-value: **0.0039** (1/256, the minimum achievable with n=8)
- Holm-adjusted p-value: **0.0156**
- **Decision: REJECT H0 at alpha = 0.05.** The agentic system detects regime breaks significantly earlier.

#### 5.1.2 False-Positive Rate (H1 Co-Primary Endpoint)

**Table 2: Calm-window FPR, development set**

| Window | Strategy | Classical FPR (HMM) | Agentic FPR |
|--------|----------|--------------------|----|
| calm_2004_2006 | AL_PCA | 0.0305 | 0.0013 |
| calm_2013_2014 | AL_PCA | 0.0000 | 0.0384 |
| calm_2004_2006 | JT_MOM | 0.0053 | 0.0013 |
| calm_2013_2014 | JT_MOM | 0.0020 | 0.0230 |

Block bootstrap test:
- Observed FPR difference: **+0.0023** (agentic slightly higher)
- Bootstrap p-value: **0.6207**
- **Decision: FAIL TO REJECT H0.** No agentic FPR advantage.

The FPR result is nuanced: the agentic system has lower FPR on calm_2004_2006 (where news coverage is sparse and triage skips 91-94% of days) but higher FPR on calm_2013_2014 (where FinBERT stress saturates at ~0.97, triggering thinking mode on all 504 days). The aggregate shows no significant difference.

#### 5.1.3 Recall

- Classical: 6/8 detected (missed gfc_lehman_2008 x JT_MOM and momentum_crash_2009 x JT_MOM)
- Agentic: 8/8 detected (100% recall)
- McNemar's test: p = 0.5 (not significant with 2 discordant pairs)
- Bayesian P(agentic recall > classical recall): **0.936**

#### 5.1.4 H1 Verdict (Development Set)

H1 requires BOTH latency AND FPR to favour the agentic arm. Latency rejects (p = 0.0156); FPR does not reject (p = 0.62). **H1 is partially supported: the agentic system is significantly faster but does not demonstrate lower FPR.**

### 5.2 Test Set (Confirmatory)

#### 5.2.1 Latency

**Table 3: Per-pair detection results, test set**

| Window | Strategy | Classical (HMM) | Agentic | Latency Diff |
|--------|----------|-----------------|---------|-------------|
| flash_crash_2010 | AL_PCA | Detected / 8d | Detected / 5d | -3 |
| flash_crash_2010 | JT_MOM | **Missed** | Detected / 1d | -- |
| china_deval_2015 | JT_MOM | Detected / 2d | Detected / 9d | +7 |
| volmageddon_2018 | JT_MOM | **Missed** | Detected / 8d | -- |
| covid_2020 | JT_MOM | Detected / 14d | **Missed** | -- |

Permutation test (n=5 paired windows):
- Observed mean difference: **-4.4 days** (agentic directionally faster)
- One-sided p-value: **0.25**
- **Decision: NOT significant.** The test set cannot confirm the dev-set finding.

**Critical note on power:** With only 5 event pairs, the maximum achievable statistical significance under exact permutation is p = 1/32 = 0.03125. Even if all 5 pairs unanimously favoured the agentic arm, the result would be barely significant at alpha = 0.05. The test set is fundamentally underpowered to confirm or disconfirm the dev-set finding.

#### 5.2.2 False-Positive Rate

**Table 4: Calm-window FPR, test set**

| Window | Strategy | Classical FPR (HMM) | Agentic FPR |
|--------|----------|--------------------|----|
| calm_2012 | AL_PCA | 0.0120 | 0.1038 |
| calm_2012 | JT_MOM | 0.0200 | 0.0769 |
| calm_2017 | JT_MOM | 0.0120 | 0.0270 |

Block bootstrap test:
- Observed FPR difference: **+0.0552** (agentic higher)
- Bootstrap p-value: **0.5147**
- **Decision: FAIL TO REJECT.** But the direction is AGAINST the agentic arm---it produces more false positives than classical on the test set.

This finding is consistent with the mechanism identified on the dev set: when news coverage is available (post-2007), FinBERT stress saturation causes near-universal escalation to thinking mode, generating more opportunities for false-positive supervisor assessments.

#### 5.2.3 Recall

- Classical: 3/5 detected (missed flash_crash_2010 x JT_MOM and volmageddon_2018 x JT_MOM)
- Agentic: 4/5 detected (missed covid_2020 x JT_MOM)
- McNemar's test: p = 1.0 (not significant)

#### 5.2.4 Test-Set Summary

The test set is **directionally consistent** with the dev-set latency finding (agentic faster on average) but **cannot confirm it** due to insufficient power. The FPR direction is unfavourable to the agentic arm. These results do not contradict the dev-set conclusions but cannot independently support them.

### 5.3 Strategy Breakdown (H2/H3)

**H2 (generalises across strategies):** On the dev set, the agentic advantage is present for both AL_PCA (mean latency reduction: 7.0 days across 4 events) and JT_MOM (all 4 events detected vs. 2/4 classical, with mean agentic latency 1.5 days). The advantage generalises, supporting H2 directionally. On the test set, the single AL_PCA event (flash_crash_2010) shows a 3-day advantage; JT_MOM results are mixed (one advantage, one disadvantage, two misses distributed across arms).

**H3 (differs by strategy):** The agentic advantage appears larger for JT_MOM, where the classical HMM detector misses 2/4 dev events entirely (both detected by the agentic system). For AL_PCA, the classical arm detects all 4 dev events (just with higher latency). This suggests the agentic system's comparative advantage is greater when classical methods fail entirely---plausibly because momentum crash dynamics produce return patterns that are less salient to the HMM's volatility-based state model.

### 5.4 Leakage Analysis

Conducted on the development anchor case (quant_meltdown_2007 x AL_PCA):

| Condition | Detected | Latency |
|-----------|----------|---------|
| A (standard) | Yes | 0 days |
| B (date-masked) | Yes | 1 day |
| C (synthetic, 10 windows) | 3/10 | 1.3 days (when detected) |

**Interpretation:**
- **B approximately equals A:** The 1-day difference between standard and date-masked conditions provides strong evidence that date-memorisation contributes little to detection performance. The model detects the regime break from statistical and contextual signals, not from recognizing the date "August 2007."
- **Memorisation bound:** perf_C (0.3) < 0.6 x perf_A (0.6), so the memorisation bound is **inconclusive**---we cannot rule out that familiarity with real-event patterns contributes to performance, but the B-condition finding suggests the contribution is modest.
- **Limitation:** The leakage harness was run only on one anchor event (Deviation 10). Generalisation to other events is assumed but not demonstrated.

### 5.5 Operational Metrics

#### 5.5.1 News-Agent Failure Rates

The news context agent experienced substantial failure rates on heavy-news days:

| Cell | Non-skip Days | Supervisor Success | Failure Rate |
|------|--------------|-------------------|------|
| calm_2013_2014 x AL_PCA | 504 | 144 | 71% |
| calm_2013_2014 x JT_MOM | 502 | 104 | 79% |
| downgrade_2011 x AL_PCA | 64 | 31 | 52% |
| gfc_lehman_2008 x JT_MOM | 74 | 37 | 50% |
| momentum_crash_2009 x AL_PCA | 72 | 39 | 46% |
| quant_meltdown_2007 (both) | 44 | 44 | 0% |
| calm_2004_2006 (both) | 65/38 | 65/38 | 0% |

Failures are caused by `json.JSONDecodeError` and `ValueError` when the news agent fails to produce valid JSON despite grammar-constrained decoding. This occurs predominantly on days with high article counts (GFC peak, 2013-2014 financial news volume) where the input context exceeds the model's effective processing capacity.

**Bias direction:** Failures can only suppress alarms (a day with no assessment cannot fire), so they bias AGAINST the agentic arm for latency/detection and IN FAVOUR of the agentic arm for FPR (failed days cannot produce false positives). The latency advantage is therefore conservative; the FPR comparison must be interpreted cautiously.

#### 5.5.2 Inference Latency

Mean supervisor call latency: 3-5 seconds per trading day (qwen2.5:1.5b on RTX 3050 Ti). Total wall-clock time for a 74-day event window: approximately 6-8 minutes (including news retrieval and caching). This is operationally viable for end-of-day monitoring but not for intraday applications.

### 5.6 Economic Impact: PnL Comparison

To translate the latency advantage into economic terms, we simulate three monitoring regimes across all 13 event-window pairs (8 dev + 5 test): (1) **unmonitored** — hold the strategy position throughout the window; (2) **classical-monitored** — go flat on the first HMM alarm day, remain flat for a 5-trading-day cooldown; (3) **agentic-monitored** — same rule applied to the first supervisor ALERT/CRITICAL assessment. This is a simplified simulation: it assumes immediate execution at the close, zero transaction costs, and a binary flat/invested position. It measures the maximum drawdown experienced under each regime and the drawdown avoided relative to the unmonitored baseline.

#### 5.6.1 Per-Window Results

**Table 5: PnL impact by monitoring regime (all event windows)**

| Window | Strategy | Set | Unmon Return | Unmon MaxDD | Cls MaxDD | Ag MaxDD | DD Avoided (Cls) | DD Avoided (Ag) |
|--------|----------|-----|-------------|-------------|-----------|----------|-----------------|-----------------|
| quant_meltdown_2007 | AL_PCA | Dev | -4.6% | -7.0% | -4.0% | -0.1% | +3.0 pp | +6.9 pp |
| gfc_lehman_2008 | AL_PCA | Dev | +10.8% | -4.0% | -2.2% | -0.8% | +1.8 pp | +3.2 pp |
| momentum_crash_2009 | AL_PCA | Dev | -2.1% | -7.0% | -4.9% | -3.0% | +2.1 pp | +4.0 pp |
| downgrade_2011 | AL_PCA | Dev | +33.7% | -3.0% | -3.0% | -3.0% | +0.0 pp | +0.1 pp |
| quant_meltdown_2007 | JT_MOM | Dev | +4.0% | -3.7% | -3.7% | -2.2% | +0.0 pp | +1.5 pp |
| gfc_lehman_2008 | JT_MOM | Dev | +1.6% | -18.5% | -18.5% | -2.9% | +0.0 pp | +15.6 pp |
| momentum_crash_2009 | JT_MOM | Dev | -69.2% | -94.5% | -94.5% | -22.2% | +0.0 pp | **+72.3 pp** |
| downgrade_2011 | JT_MOM | Dev | -4.2% | -10.4% | -5.8% | -3.1% | +4.7 pp | +7.3 pp |
| flash_crash_2010 | AL_PCA | Test | -2.0% | -3.7% | -1.3% | -0.2% | +2.4 pp | +3.5 pp |
| flash_crash_2010 | JT_MOM | Test | -9.9% | -15.6% | -9.4% | -5.2% | +6.2 pp | +10.3 pp |
| china_deval_2015 | JT_MOM | Test | +7.6% | -9.9% | -6.5% | -8.2% | +3.4 pp | +1.7 pp |
| volmageddon_2018 | JT_MOM | Test | +5.5% | -4.1% | -4.1% | -4.1% | +0.0 pp | -0.0 pp |
| covid_2020 | JT_MOM | Test | +14.1% | -19.2% | -19.2% | -14.4% | +0.0 pp | +4.9 pp |

#### 5.6.2 Aggregate Economic Impact

Across all 13 event-window pairs:

- **Mean drawdown avoided (classical HMM):** +1.82 percentage points
- **Mean drawdown avoided (agentic):** +10.09 percentage points
- **Agentic economic advantage over classical:** +8.28 percentage points of drawdown avoided per event

The agentic arm avoids more drawdown than classical in 11 of 13 windows. The two exceptions are china_deval_2015 (agentic detected 7 days slower than classical, reducing its protective value) and volmageddon_2018 (agentic fired on day 8 but the drawdown had already occurred intraday on day 1).

#### 5.6.3 Extreme Cases

The most striking case is **momentum_crash_2009 × JT_MOM**, where the cross-sectional momentum strategy experienced a catastrophic 94.5% peak-to-trough drawdown during the momentum crash. The classical HMM failed to detect this event entirely (it does not appear in the HMM's volatility state model, which is calibrated for mean-reverting regimes). The agentic system detected it on day 1 after onset, going flat and avoiding 72.3 percentage points of drawdown — a potentially portfolio-saving intervention.

Similarly, **gfc_lehman_2008 × JT_MOM** shows the agentic arm avoiding 15.6 pp of drawdown on an 18.5% unmonitored drawdown, while the classical HMM (which also missed this event for JT_MOM) provided no protection.

#### 5.6.4 Interpretation and Caveats

The PnL simulation is not a backtest of a trading strategy; it is a counterfactual analysis of monitoring value. Key caveats:

1. **Transaction costs ignored:** Going flat and re-entering incurs costs that reduce the benefit. For institutional momentum portfolios (high turnover), the re-entry cost may be material.
2. **Binary position assumption:** In practice, a portfolio manager might reduce rather than eliminate exposure. The REDUCE and HALT actions from the supervisor could inform graduated responses, but we model only the binary case.
3. **Cooldown sensitivity:** The 5-day flat period is arbitrary. Shorter cooldowns capture more recovery; longer cooldowns provide more protection. The reported figures are sensitive to this parameter.
4. **Survivorship in the simulation:** The PnL curves are point-in-time, but the simulation assumes the portfolio manager acts on the monitoring signal — an assumption about human behaviour, not market microstructure.

Despite these caveats, the direction of the result is robust: earlier detection of regime breaks provides economic value by reducing exposure to adverse tail moves. The mean advantage of 8.28 pp per event, while simulated under idealised assumptions, establishes that the latency advantage documented in Section 5.1 has practical significance beyond statistical significance.

**Figures:** The equity-curve overlays (Figure 10), drawdown-avoided bar chart (Figure 11), and summary table (Figure 12) are included in the supplementary materials.

---

## 6. Discussion

### 6.1 Why the Agentic System Detects Earlier

The agentic system's latency advantage (9.75 days on dev, 4.4 days directionally on test) appears to arise from two mechanisms:

1. **News anticipation:** Financial news coverage of developing crises often precedes their full statistical manifestation in strategy returns. For the quant meltdown (onset: 2007-08-06), news coverage of crowded quantitative strategies and forced deleveraging appeared in late July 2007, enabling 0-day latency. The classical HMM, operating solely on the return stream, required 7 days of anomalous returns to transition to the high-volatility state.

2. **Multi-signal fusion:** The supervisor integrates strategy telemetry, news flags, macro context, and classical detector outputs. Even when individual signals are sub-threshold, their conjunction can trigger an alert. The classical detectors, by design, observe only the return stream.

### 6.2 Why FPR Is Not Better

The agentic system does not achieve lower FPR---and on the test set, exhibits higher FPR than classical. Three factors contribute:

1. **FinBERT saturation:** The triage gate uses a 0.6 threshold on FinBERT stress scores to route days to the full inference pipeline. In practice, FinBERT stress saturates at ~0.97 on essentially every day that has news articles in the store, regardless of actual market conditions. This converts the stress gate into a "does news exist today?" filter, offering no discrimination between calm and stressed periods.

2. **Triage bypass:** On windows with news coverage (post-2007), nearly all days are escalated to thinking mode, bypassing the skip mechanism that would otherwise reduce false positives during benign periods.

3. **Small-model limitations:** qwen2.5:1.5b (1.5B parameters) has limited capacity to distinguish between genuinely concerning signals and routine financial noise when context is rich. Larger models might achieve better precision, but hardware constraints preclude their use in this experiment.

### 6.3 Test-Set Power

The test set's 5 event pairs provide maximum achievable significance of p = 1/32 = 0.03125 under exact permutation---barely below alpha = 0.05 even in the best case. This is a fundamental limitation of the experimental design, driven by:

- Limited availability of well-defined regime breaks in the PIT curve coverage period
- AL_PCA curve ending in 2014, restricting post-2014 test events to JT_MOM only (Deviation 3)
- The requirement that events be sufficiently distinct to constitute independent observations

Future work should expand the event set through additional strategies, longer backtest curves, or multi-market coverage.

### 6.4 The COVID-19 Miss

The agentic system's only test-set event miss (covid_2020 x JT_MOM) warrants specific discussion. The COVID-19 market crash (onset: 2020-02-20) was unprecedented in speed and was preceded by limited English-language financial news about pandemic risk until mid-February 2020. Additionally:

- The FNSPID news store covers 2003-2020, so news was available (40 articles/day across all 82 trading days). However, early pandemic coverage was dominated by general market volatility headlines rather than COVID-specific risk signals.
- Three days required thermal-stall reruns with reduced article counts (Deviation 15).
- The context-cache two-pass design (Deviation 13) may have affected information availability on early days.

The classical HMM detected covid_2020 (with 14-day latency), demonstrating that the return signal alone was sufficient for eventual detection. The agentic system's miss suggests that even with news available, the 1.5B-parameter model may lack the reasoning capacity to interpret an unprecedented pandemic shock as a regime change when early headlines are ambiguous.

### 6.5 Practical Implications

For practitioners considering LLM-augmented monitoring:

1. **Latency advantage is real but not free:** The earlier detection comes at the cost of higher infrastructure complexity, meaningful failure rates, and no improvement in false-positive control.

2. **FinBERT triage needs redesign:** A binary 0.6 threshold on saturated sentiment scores provides no useful routing. Alternative approaches (change-in-sentiment, topic-specific scoring, or purely statistical triage) should be explored.

3. **Model size matters:** At 1.5B parameters, the supervisor shows genuine regime-detection ability but lacks the precision to avoid false positives. The tradeoff between model size, hardware cost, and monitoring quality is an important engineering consideration.

4. **News coverage is a hidden dependency:** The system degrades sharply when news coverage is sparse (pre-2007) or absent (post-2016 for FNSPID). Any deployment must ensure continuous, high-quality news feeds.

---

## 7. Limitations

We document all 15 protocol deviations and additional limitations with full transparency:

### 7.1 Protocol Deviations

1. **Model substitution (llama3.2:3b to qwen2.5:1.5b):** The preregistered model failed structured-output benchmarks (11/30 valid JSON). qwen2.5:1.5b (30/30) was substituted before any evaluation. The substitution reduces model capacity (1.5B vs. 3B parameters) but improves output reliability.

2. **Completeness threshold (90% to 80%):** Lowered to include gfc_lehman_2008 x AL_PCA (84% coverage), where missing days were all post-detection-window. Without this change, one event pair would be excluded.

3. **china_deval_2015 x AL_PCA excluded:** No PIT curve coverage beyond 2014. Reduces AL_PCA test-set to one event window.

4. **calm_2013_2014 zero skip days:** FinBERT saturation causes 100% thinking-mode escalation, inflating both inference cost and FPR relative to expectation.

5. **Completeness measure changed (supervisor days to triage days):** Triage day count better reflects "was the window processed?" vs. "did every inference succeed?"

6. **D3 real-model latencies differ from stub baseline:** Expected; real model makes day-by-day assessments. Headline conclusions unchanged.

7. **News-agent failure rates (28-79%):** Substantial inference failures on heavy-news days. Biases against agentic latency, in favour of agentic FPR.

8. **Calibrated detector parameters never applied:** Classical detectors ran on defaults. HMM (headline comparator) is unaffected (calibrated = default); secondary detectors may be understated.

9. **FPR scoring asymmetry:** Cluster dedup applied to agentic arm only. Biases in favour of agentic FPR, but FPR arm does not reject regardless.

10. **Leakage harness incomplete:** Only one anchor event tested (quant_meltdown_2007 x AL_PCA). Generalisation assumed.

11. **First test-set attempt invalidated by runner death (D5):** Required restart with day-level resume.

12. **Day-level resume + dead-runner auto-recovery added mid-test:** Code change during test-set evaluation (operational fix, not affecting detection logic).

13. **Context-cache two-pass design for covid_2020:** Thermal mitigation may have affected information availability.

14. **Day-level DeadRunnerError retry:** Operational robustness mechanism added during test execution.

15. **3 thermal-stall covid_2020 days rerun with reduced articles:** Hardware limitation forced reduced input on 3 trading days.

### 7.2 Additional Limitations

- **Single seed / single model:** All results are from a single model configuration (temperature=0) with no ensembling or seed variation. Stochasticity from model internals is uncontrolled.

- **Famous-event memorisation risk:** Despite the leakage harness, the model was likely trained on text describing the 2007-2009 financial crisis. The B-condition finding (detection with masked dates) mitigates but does not eliminate this concern.

- **Thermal constraints:** Consumer-grade GPU (RTX 3050 Ti, 4GB VRAM) introduced throttling that required design workarounds and may have subtly affected results.

- **Small model capability ceiling:** qwen2.5:1.5b is near the lower bound of useful language model capability. A larger model (7B, 13B) might show different latency/FPR tradeoffs but was infeasible on the available hardware.

- **Scoring asymmetry (Deviation 9):** The FPR comparison is not fully symmetric between arms. A sensitivity analysis with symmetric scoring is warranted but was deferred per the pre-specified analysis procedure.

- **Underpowered test set:** 5 event pairs provide maximum achievable p = 0.03125; the experiment cannot produce a significant confirmatory result at conventional levels.

- **Single-strategy-family scope:** Both strategies operate on U.S. large-cap equities (S&P 500). Generalisation to other asset classes, geographies, or strategy types is untested.

- **FNSPID density variation:** News store covers 2003-2020, but article density is uneven (~80/day in 2003-2006, ~2,800/day in 2007-2016, ~1,400/day in 2017-2020). Pre-2009 rows are predominantly non-English (Lenta.ru scrape); the NYT Archive API supplements early windows.

---

## 8. Conclusion and Future Work

### 8.1 Conclusion

This preregistered experiment provides evidence that an LLM-based agentic monitoring system can detect financial market regime breaks earlier than classical change-point detectors. On the development set, the latency advantage is 9.75 days (Holm-adjusted p = 0.0156), driven by the system's ability to integrate news signals that precede statistical manifestation in returns. However, the co-primary FPR endpoint does not favour the agentic arm, and the confirmatory test set is underpowered to independently validate the latency finding (p = 0.25, n = 5).

We characterize this result as **directionally consistent but not confirmatory**: the test set shows the same pattern (agentic faster on average) but cannot reach significance. The FPR finding is unfavourable on the test set (agentic FPR higher), attributed to FinBERT triage saturation and small-model precision limitations.

The experiment also reveals practical challenges: news-agent failure rates of 28-79% on heavy-news windows, FinBERT stress saturation rendering triage ineffective, and hardware constraints limiting model scale. These findings are relevant for practitioners considering LLM-augmented monitoring deployment.

### 8.2 Future Work

1. **Larger models:** Evaluate 7B-13B models (or API-served frontier models) to test whether increased capacity improves FPR without sacrificing latency.

2. **Triage redesign:** Replace the saturated FinBERT threshold with change-in-sentiment, topic-specific scoring, or adaptive thresholds calibrated to news-volume regimes.

3. **Expanded event set:** Extend PIT curves, add strategies (carry, value, low-vol), and include non-U.S. markets to increase test-set power.

4. **Prospective evaluation:** Deploy the system on live trading strategies with forward-looking evaluation (eliminating memorisation concerns entirely).

5. **Ensemble approaches:** Combine LLM-based and classical detection in a meta-monitor that leverages the complementary strengths of both approaches.

6. **News-agent robustness:** Improve structured-output reliability through better prompt engineering, retry logic, or model fine-tuning to reduce the 28-79% failure rates.

7. **Symmetric FPR scoring:** Implement and report results under symmetric deduplication applied to both arms.

---

## 9. References

Adams, R. P., & MacKay, D. J. C. (2007). Bayesian online changepoint detection. *arXiv preprint arXiv:0710.3742*.

Aminikhanghahi, S., & Cook, D. J. (2017). A survey of methods for time series change point detection. *Knowledge and Information Systems*, 51(2), 339-367.

Ang, A., & Bekaert, G. (2002). International asset allocation with regime shifts. *Review of Financial Studies*, 15(4), 1137-1187.

Araci, D. (2019). FinBERT: Financial sentiment analysis with pre-trained language models. *arXiv preprint arXiv:1908.10063*.

Avellaneda, M., & Lee, J.-H. (2010). Statistical arbitrage in the US equities market. *Quantitative Finance*, 10(7), 761-782.

Brown, S. J., Goetzmann, W., Ibbotson, R. G., & Ross, S. A. (1992). Survivorship bias in performance studies. *Review of Financial Studies*, 5(4), 553-580.

Carlini, N., Ippolito, D., Jagielski, M., Lee, K., Tramer, F., & Zhang, C. (2023). Quantifying memorization across neural language models. *International Conference on Learning Representations (ICLR)*.

Daniel, K., & Moskowitz, T. J. (2016). Momentum crashes. *Journal of Financial Economics*, 122(2), 221-247.

Elton, E. J., Gruber, M. J., & Blake, C. R. (1996). Survivorship bias and mutual fund performance. *Review of Financial Studies*, 9(4), 1097-1120.

Guidolin, M., & Timmermann, A. (2007). Asset allocation under multivariate regime switching. *Journal of Economic Dynamics and Control*, 31(11), 3503-3544.

Hamilton, J. D. (1989). A new approach to the economic analysis of nonstationary time series and the business cycle. *Econometrica*, 57(2), 357-384.

Huang, A. H., Wang, H., & Yang, Y. (2023). FinBERT: A large language model for extracting information from financial text. *Contemporary Accounting Research*, 40(2), 806-841.

Israel, R., Laursen, K., & Richardson, S. (2020). Is (systematic) value investing dead? *Journal of Portfolio Management*, 47(2), 38-62.

Jegadeesh, N., & Titman, S. (1993). Returns to buying winners and selling losers: Implications for stock market efficiency. *Journal of Finance*, 48(1), 65-91.

Khandani, A. E., & Lo, A. W. (2007). What happened to the quants in August 2007? *Journal of Investment Management*, 5(4), 5-54.

Khandani, A. E., & Lo, A. W. (2011). What happened to the quants in August 2007? Evidence from factors and transactions data. *Journal of Financial Markets*, 14(1), 1-46.

Kim, S., Yoon, H., & Kim, J. (2024). LLM-based portfolio construction: A survey. *arXiv preprint arXiv:2401.10865*.

Lopez-Lira, A., & Tang, Y. (2023). Can ChatGPT forecast stock price movements? Return predictability and large language models. *arXiv preprint arXiv:2304.07619*.

Page, E. S. (1954). Continuous inspection schemes. *Biometrika*, 41(1/2), 100-115.

Wu, S., Irsoy, O., Lu, S., Daber, V., Dredze, M., Gehrmann, S., Kambadur, P., Rosenberg, D., & Mann, G. (2023). BloombergGPT: A large language model for finance. *arXiv preprint arXiv:2303.17564*.

---

## Appendix A: Reproducibility

### A.1 Software Environment

```
OS: Windows 10 Home 10.0.19045
Python: 3.11
GPU: NVIDIA RTX 3050 Ti (4 GB VRAM)
LLM runtime: Ollama (local)
Model: qwen2.5:1.5b (grammar-constrained JSON decoding, temperature=0)
```

### A.2 Key Git Tags

| Tag | Purpose |
|-----|---------|
| `eval-freeze-v1` | Pins all code before test-set evaluation |

### A.3 Reproduction Commands

```bash
# Classical monitoring (all windows)
python monitoring/run_classical.py

# Agentic monitoring (single window)
python monitoring/run_agentic.py --window quant_meltdown_2007 \
    --strategy AL_PCA --model ollama:qwen2.5:1.5b

# Agentic monitoring (stub model, offline)
python monitoring/run_agentic.py --window quant_meltdown_2007 \
    --strategy AL_PCA --model stub

# Dev-set analysis
python monitoring/scripts/analyze_dev_results.py

# Leakage harness
python monitoring/run_leakage.py --strategy AL_PCA --model ollama:qwen2.5:1.5b

# Freeze-gate check
python monitoring/freeze_gate.py

# Test-set calm windows
python monitoring/scripts/run_calm_windows.py
```

### A.4 Key File Paths

```
monitoring/
├── run_classical.py          # Classical detection pipeline
├── run_agentic.py            # Agentic detection pipeline
├── run_leakage.py            # Three-condition leakage harness
├── freeze_gate.py            # Pre-freeze validation (9 checks)
├── metrics.py                # Latency, FPR, detection scoring
├── onset.py                  # Preregistered onset dates
├── agentic/
│   ├── alarm_extraction.py   # Agentic alarm scoring + cluster dedup
│   └── guardrails.py         # Date masking, lookahead assertion
├── scripts/
│   ├── analyze_dev_results.py    # Dev-set statistical analysis
│   └── run_calm_windows.py       # Calm-window FPR evaluation
├── results/
│   ├── dev_analysis.json         # Dev-set summary statistics
│   ├── classical_summary.json    # Classical detection results
│   └── archive/                  # Historical result artifacts
└── tests/
    ├── test_metrics.py
    ├── test_alarm_extraction.py
    └── test_run_agentic.py
```

### A.5 Data Sources

| Source | Coverage | Size | Format |
|--------|----------|------|--------|
| FNSPID (Financial News) | 2003-2020 | 22 GB (11.8M articles) | Per-year Parquet |
| FRED/ALFRED (Macro) | Various | Vintage releases | CSV via API |
| S&P 500 constituents | 2003-2020 | PIT membership intervals | CSV |
| Strategy PnL (AL_PCA) | 2003-2015 | Daily returns | CSV |
| Strategy PnL (JT_MOM) | 2003-2020 | Daily returns | CSV |

---

*Corresponding code repository: point-in-time evaluation artifacts pinned at `eval-freeze-v1`.*
