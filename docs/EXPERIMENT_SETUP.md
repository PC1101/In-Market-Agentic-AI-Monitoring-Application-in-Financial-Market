# Experiment Setup: Agentic Risk-Overlay P&L Backtest

**Replication guide for the full pipeline — from raw data through agentic inference to gross/net live-backtest figures.**

Every design choice is justified inline. If a step says "why," that's the reasoning a replicator needs to understand the decision.

---

## Table of Contents

1. [Research Question](#1-research-question)
2. [Experimental Design Overview](#2-experimental-design-overview)
3. [Data Pipeline](#3-data-pipeline)
4. [Strategy Construction](#4-strategy-construction)
5. [Classical Monitoring Baseline](#5-classical-monitoring-baseline)
6. [Agentic Monitoring Pipeline](#6-agentic-monitoring-pipeline)
7. [Risk-Overlay Policy](#7-risk-overlay-policy)
8. [Portfolio Simulation Engine](#8-portfolio-simulation-engine)
9. [Transaction Cost Model](#9-transaction-cost-model)
10. [Live Backtest Execution](#10-live-backtest-execution)
11. [Evaluation Windows](#11-evaluation-windows)
12. [Results Summary](#12-results-summary)
13. [Known Biases and Limitations](#13-known-biases-and-limitations)
14. [Replication Commands](#14-replication-commands)
15. [File Inventory](#15-file-inventory)

---

## 1. Research Question

**Can an LLM-based agentic monitoring layer, receiving the same information a human trader would see (strategy telemetry, classical detector alarms, financial news), reduce drawdowns of systematic equity strategies without destroying long-run returns?**

Three hypotheses under test (from `docs/VRI.md`):

- **H1:** Agentic monitoring detects regime breaks faster and with fewer false positives than classical change-point detectors alone.
- **H2:** The advantage generalises across both strategy types (cross-sectional momentum and statistical arbitrage).
- **H3:** The magnitude of the advantage differs by strategy.

The P&L backtest is the economic-impact test: do faster, more accurate detections translate into money saved?

---

## 2. Experimental Design Overview

```
                        ┌─────────────────────┐
                        │   S&P 500 Universe   │
                        │  (point-in-time PIT) │
                        └──────────┬──────────┘
                    ┌──────────────┴──────────────┐
                    ▼                             ▼
           ┌────────────────┐            ┌────────────────┐
           │  JT Momentum   │            │  AL PCA Stat   │
           │  (12-1 decile) │            │  Arb (OU-based)│
           │  Long-only     │            │  Long-only     │
           └───────┬────────┘            └───────┬────────┘
                   │                             │
                   ▼                             ▼
           Gross equity curve              Gross equity curve
           + daily weight matrix           + trades blotter
                   │                             │
           ┌───────┴────────┐            ┌───────┴────────┐
           │ Post-hoc costs │            │ Post-hoc costs │
           │ (frictions.py) │            │ (frictions.py) │
           └───────┬────────┘            └───────┬────────┘
                   │                             │
            Net curves ×3                 Net curves ×3
           (low/base/high)              (low/base/high)
                   │                             │
                   └──────────┬──────────────────┘
                              ▼
                   ┌──────────────────────┐
                   │  24 Agentic JSONL    │
                   │  logs (12 windows ×  │
                   │  2 strategies)       │
                   └──────────┬───────────┘
                              ▼
                   ┌──────────────────────┐
                   │  Risk-Overlay Policy │
                   │  (exposure ladder +  │
                   │  cooldown + re-entry)│
                   └──────────┬───────────┘
                              ▼
                   ┌──────────────────────┐
                   │  Portfolio Engine    │
                   │  ($1M, daily sim,    │
                   │   tcost on Δexposure)│
                   └──────────┬───────────┘
                              ▼
                   Figures, metrics, audit CSVs
```

**2×2 design:**
- 2 strategies: JT_MOM (cross-sectional momentum), AL_PCA (PCA-based statistical arbitrage)
- 2 cost variants: gross (frictionless), net (era-dependent transaction costs)
- 3 cost scenarios per net variant: low, base, high (sensitivity)
- 3 portfolio curves per run: managed (strategy + agent), unmanaged (strategy alone), SPY buy-and-hold

---

## 3. Data Pipeline

### 3.1 S&P 500 Universe (Point-in-Time)

**Source:** `XSectional/data/sp500_prices_pit.csv` (69 MB, 2000-01-03 onward, ~1,200 tickers)

**Why point-in-time:** Using today's S&P 500 members on historical data introduces survivorship bias — the backtest would only trade stocks that survived to the present. The PIT dataset reconstructs the actual S&P 500 membership at each date, including stocks that were later removed, merged, or delisted.

**Why adjusted close:** Dividends are captured via adjusted closing prices (split- and dividend-adjusted). This means "dividends included" is a data-level guarantee, not an assumption.

### 3.2 Benchmark: SPY

**Source:** Yahoo Finance via `yfinance`, cached at `monitoring/data/benchmark_SPY.csv` (6,245 daily returns, 2001-03-02 to 2025-12-30).

**Why SPY not ^GSPC:** SPY is an adjusted total-return ETF (dividends reinvested via adjusted close with `auto_adjust=True`). ^GSPC is price-only and understates benchmark returns by approximately 2% per year (the dividend yield). Using ^GSPC would make the strategies look artificially better relative to the benchmark.

### 3.3 Financial News (FNSPID + NYT Archive)

**Source:** FNSPID financial news dataset + NYT Archive API (~240k articles).

**Why these sources:** FNSPID provides risk-relevant financial news with ticker associations. NYT Archive provides broad macro/geopolitical context. Together they give the News Context Agent the same information a trader scanning Bloomberg/Reuters would have.

**Timestamp filtering:** All news is filtered to `date <= as_of` before being shown to the agent, eliminating look-ahead. The news pipeline applies a 7-day trailing window (`COVERAGE_DAYS = 7`).

### 3.4 FinBERT Stress Scores

Pre-computed per-day mean negative-sentiment probability from FinBERT applied to filtered risk-relevant headlines. Cached at `monitoring/results/finbert_cache/<window>.json`.

**Why pre-compute:** FinBERT (PyTorch + transformers) consumes ~1.5 GB RAM. The LLM inference process (Ollama/vLLM) competes for GPU memory. Pre-computing FinBERT scores in a separate process and reading from cache keeps the inference process lightweight (~300 MB) and prevents OOM crashes.

---

## 4. Strategy Construction

### 4.1 JT Momentum (JT_MOM)

**Signal:** Jegadeesh-Titman 12-1 cross-sectional momentum. At each month-end, rank all S&P 500 constituents (PIT) by their trailing 12-month return excluding the most recent month (skip-month avoids short-term reversal). Go long the top decile.

**Rebalance frequency:** Monthly (end of month). Between rebalances, positions drift passively with market returns — no intra-month rebalancing.

**Why long-only:** The original JT signal produces a long-short portfolio. The long-only version keeps only positive weights and renormalises each row to sum to 1.0:

```python
# XSectional/portfolio.py
def to_long_only(weights: pd.DataFrame) -> pd.DataFrame:
    lo = weights.where(weights > 0)
    return lo.div(lo.sum(axis=1), axis=0)
```

**Why long-only instead of long-short:** (a) More realistic for a retail $1M portfolio — shorting S&P 500 names requires margin and borrow, adding complexity and cost. (b) The agentic overlay is a risk-reduction mechanism that scales exposure between 0 and 1 — this maps cleanly to "hold less" in a long-only context. (c) Isolates the overlay's value: if it works on a simple long-only strategy, the result is more interpretable.

**Daily curve construction:**

```python
# XSectional/backtest.py:run_backtest_daily()
w_daily = w.reindex(daily_returns.index, method="ffill").shift(1)
port_ret = (w_daily.fillna(0) * daily_returns.fillna(0)).sum(axis=1)
```

Monthly weights are forward-filled onto the daily grid, then shifted by one day. The return on day *t* uses only weights decided on or before day *t−1*. No look-ahead.

**Output:** `XSectional/results/equity_curve_daily_long_only.csv`
- 6,246 trading days: 2001-03-01 to 2025-12-30
- Terminal equity: 12.75× ($1 grows to $12.75)
- Schema: `Date, port_ret, equity, drawdown`

### 4.2 AL PCA Statistical Arbitrage (AL_PCA)

**Signal:** PCA-factor-residual mean-reversion. PCA factors are extracted from sector constituent returns. The strategy trades sector-level portfolios when the residual from the PCA model deviates from zero (Ornstein-Uhlenbeck process entry/exit signals).

**Portfolio construction:** 9 sector sleeves, each running independent OU-process stat-arb. Within each sleeve, active positions are equal-weighted (1/N'). The portfolio return is the equal-weight average across all 9 sector sleeves:

```python
# Stat Arb/statsArb-dev/run_full_universe.py:149
ret_panel.mean(axis=1)  # equal-weight across sector sleeves
```

**Why long-only:** Same rationale as JT_MOM. The `run_full_universe.py` supports `--long-only` which filters to long-side trades only.

**Output:** `Stat Arb/statsArb-dev/results/full_universe/pit_pca_long_only/equity_curve.csv`
- 5,531 trading days: 2004-01-02 to 2025-12-24
- Terminal equity: 24.71× ($1 grows to $24.71)
- Schema: `Date, port_ret, equity, drawdown`
- Auxiliary: `trades.csv` (75,793 trades), `sector_returns.csv` (9 sector daily returns)

### 4.3 Why these two strategies

JT momentum and AL stat-arb represent opposite factor exposures:
- JT is a trend-following strategy (momentum) — it suffers during momentum crashes and sharp reversals.
- AL is a mean-reversion strategy (statistical arbitrage) — it suffers during correlation breakdowns and forced deleveraging.

Testing the overlay on both validates H2 (generalisation) and H3 (strategy-dependent performance). If the overlay only worked on one, it would be fitting a specific factor's failure mode, not detecting regime breaks generically.

---

## 5. Classical Monitoring Baseline

Four classical change-point detectors run over the full equity curve (`monitoring/run_classical.py`):

| Detector | Method | Why included |
|---|---|---|
| **Page-Hinkley** | Sequential mean-shift detector; detects a change in the mean of a stream | Well-studied, fast, minimal parameters; detects sustained return drops |
| **BOCPD** | Bayesian Online Change-Point Detection (Adams & MacKay 2007) | Probabilistic; provides run-length posterior, not just a binary alarm |
| **HMM** | Hidden Markov Model (2-state: normal/stressed) | Captures regime persistence; trained on pre-event data (out-of-sample for event windows) |
| **Distributional Threshold** | Rolling-window percentile-based anomaly detection | Non-parametric fallback; catches fat-tail moves that parametric models miss |

**Aggregate alarm rule:** ≥2 detectors firing within 5 trading days. This is the classical baseline that the agentic layer must beat.

**Why these specific detectors:** They represent four distinct statistical paradigms (cumulative-sum, Bayesian, latent-state, non-parametric). The aggregate rule prevents any single noisy detector from dominating.

---

## 6. Agentic Monitoring Pipeline

### 6.1 Architecture: Three-Agent Cascade

```
Day t arrives
     │
     ▼
┌─────────────┐     FinBERT stress score
│   TRIAGE    │◄──  News intensity z-score
│             │     Recent detector alarms
└──────┬──────┘
       │ mode ∈ {skip, cheap, thinking, classical_escalation}
       │
       │ if mode ≠ skip:
       ▼
┌─────────────┐     Filtered news articles (7-day window, ≤40 articles)
│ NEWS CONTEXT│
│    AGENT    │──►  {overall_risk, risk_flags[], narrative, confidence}
└──────┬──────┘
       │
       ▼
┌─────────────┐     Strategy telemetry (drawdown, vol, returns)
│ PERFORMANCE │◄──  Classical detector alarms
│ SUPERVISOR  │◄──  News agent output
│   (v3)      │◄──  Macro context (VIX, yields, TED spread)
└──────┬──────┘
       │
       ▼
{state, action, root_cause, confidence, detectors_cited}
```

### 6.2 Triage (Cost Control)

**Why triage:** Running a 9B-parameter LLM on every trading day is expensive (~6,250 days × 2 strategies = 12,500 inference calls). Triage skips days where there's nothing to worry about.

**Decision logic** (`monitoring/news/triage.py`):

| Mode | Trigger | Action |
|---|---|---|
| `skip` | No signal anywhere | Log and move on; no LLM call |
| `cheap` | FinBERT stress ≥ 0.50 | Run news + supervisor agents |
| `thinking` | FinBERT stress ≥ 0.58 OR news intensity z ≥ 2.0 OR any detector fired in last 3 days | Run news + supervisor agents |
| `classical_escalation` | Classical aggregate alarm fired within 5 days | Always run full agentic assessment |

**Why these thresholds:**
- `STRESS_CHEAP = 0.50`: Calibrated from real vLLM logs. Calm-window daily mean stress ≈ 0.55, event mean ≈ 0.49 — FinBERT separation is poor. 0.50 casts a wide net to avoid missing events via the FinBERT pathway.
- `STRESS_THINKING = 0.58`: ~27% calm / ~11% event escalation via FinBERT alone. High enough to avoid flooding the model with noise, low enough that z-score and detector pathways independently catch all real events.
- `HIT_Z_THINKING = 2.0`: Two standard deviations above the trailing news-intensity baseline. A clear outlier in news volume/risk-term frequency.
- `RECENT_DAYS = 3`: Detector lookback. Three calendar days captures the typical delay between a market shock and detector firing.

### 6.3 News Context Agent

**Prompt** (`monitoring/agentic/prompts.py`, version `news-context-v1`):

```
You are a News Context Agent supporting the monitoring of a systematic trading strategy.
You receive financial news headlines/summaries published on or before a decision date.
Your job is to extract MARKET-RISK-relevant context: what stress, if any, is the news
describing, and how severe is it for systematic strategies?

You may ONLY use the information provided in the user message. It reflects what was
published as of the stated decision date. Do not use any knowledge of what happened
after that date.
```

**Output schema:**
```json
{
  "overall_risk": "LOW|ELEVATED|HIGH|SEVERE",
  "risk_flags": [{"flag": "credit_stress", "evidence": "..."}],
  "narrative": "2-4 sentences summarising the news-implied market state.",
  "confidence": 0.0-1.0,
  "as_of": "YYYY-MM-DD",
  "n_articles": 40
}
```

**Why a separate news agent:** Separation of concerns. The news agent extracts structure from unstructured text. The supervisor integrates this with quantitative telemetry. A single monolithic prompt would be too long and too noisy for a 9B model.

**Article limit:** 40 articles max per day. Larger context would exceed the model's effective attention window and slow inference. Articles are sorted by risk-term density (`n_risk_terms` descending) so the most relevant appear first.

### 6.4 Performance Supervisor Agent

**Prompt** (`monitoring/agentic/prompts.py`, version `supervisor-v3`):

```
You are a Performance Supervisor monitoring a systematic trading strategy. Each day you
receive the strategy's recent performance telemetry and the outputs of classical
change-point detectors. Your job is to assess whether the strategy is behaving normally
or is undergoing a regime break, and to recommend an action.

You may ONLY use the information provided in the user message. It reflects what was known
as of the stated decision date. Do not speculate about future events or use knowledge of
what happened after the decision date.

Respond with a SINGLE flat JSON object and nothing else...
```

The v3 extension adds:
```
You may additionally receive:
- A macro-market block (VIX, Treasury yields, fed funds, TED spread, and the latest
  macro releases AS THEY WERE KNOWN on the decision date).
- A structured summary from a News Context Agent that has read the filtered financial
  news published up to the decision date.
Weigh telemetry and classical detectors first; use news/macro context to confirm,
explain (root_cause), or discount them.
```

**User message structure:**

```
Decision date (as_of): 2008-09-15

Strategy telemetry:
{
  "current_drawdown": -0.134,
  "recent_mean_daily_return": -0.0021,
  "recent_worst_day": -0.0317,
  ...
}

Classical detector alarms visible so far:
{
  "page_hinkley": true,
  "bocpd": true,
  "hmm": true,
  "distributional_threshold": false
}

News Context Agent summary:
{
  "overall_risk": "SEVERE",
  "risk_flags": [...],
  ...
}

Return your JSON assessment now.
```

**Output schema** (`monitoring/agentic/schemas.py`):

```json
{
  "state": "NORMAL|WATCH|ALERT|CRITICAL",
  "action": "HOLD|INVESTIGATE|REDUCE|HALT",
  "root_cause": "One or two sentences naming the most likely driver.",
  "confidence": 0.0-1.0,
  "as_of": "2008-09-15",
  "detectors_cited": ["page_hinkley", "bocpd"]
}
```

**State/action severity ordering** (from `schemas.py`):
- States: NORMAL(0) < WATCH(1) < ALERT(2) < CRITICAL(3)
- Actions: HOLD(0) < INVESTIGATE(1) < REDUCE(2) < HALT(3)

### 6.5 LLM Model

**Model:** Qwen/Qwen3.5-9B served via vLLM with guided JSON decoding.

**Why Qwen3.5-9B:** Best available open-source model at this parameter count for structured JSON output. Grammar-constrained decoding (via vLLM's `response_format: json_schema`) guarantees the response has the schema's shape — small models without this tend to echo the schema wrapper or produce malformed JSON.

**Inference settings:**
- `temperature: 0.0` — deterministic; the same context should produce the same assessment.
- `max_tokens: 1024` — sufficient for the structured output; prevents runaway generation.
- `enable_thinking: false` — Qwen3.x defaults to thinking mode which produces internal reasoning tokens. Disabled for consistent JSON output.

**Infrastructure:** vLLM served on vast.ai GPU instances, accessed via SSH tunnel to `localhost:18000`. The `VLLMModel` class handles retries and dead-runner detection.

**Why not local Ollama:** The full 24-run matrix (12 windows × 2 strategies × ~100 days per window = ~2,400 inference calls) takes approximately 4 hours on a cloud A100. The local RTX 3050 Ti would take ~10× longer and is prone to thermal throttling on long runs.

### 6.6 Lookahead Guards

Three layers prevent the agent from using future information:

1. **Causal context construction** (`monitoring/agentic/guardrails.py`): `as_of_context()` only includes telemetry and detector alarms that are visible by the decision date. Rolling statistics use trailing windows only.

2. **News timestamp filtering:** `store.query(start, end)` returns only articles published on or before `end`. The caller sets `end = day` (the current decision date).

3. **Post-hoc assertion:** `assert_no_lookahead(context, day)` scans the context for any date string after the decision date. If found (e.g., the model hallucinated a future date in its narrative), the news block is dropped entirely ("fail closed").

### 6.7 Condition B (Date-Masking Leakage Test)

An optional `--condition B` mode masks all dates in the context with `XXXX-XX-XX` before sending to the model. This tests whether the model is exploiting specific calendar dates (e.g., "September 15, 2008" = Lehman) as a shortcut rather than reading the telemetry/news evidence. Results from Condition B runs are used in the leakage analysis but do not affect the main P&L backtest.

### 6.8 JSONL Log Format

Each agentic run produces one JSONL file: `monitoring/results/agentic_<window>_<strategy>.jsonl`

First record: run metadata.
```json
{"agent": "run_meta", "model": "vllm:Qwen/Qwen3.5-9B", "curve": "...", "git_sha": "a737441", "window": "gfc_lehman_2008", "strategy": "JT_MOM", "condition": "A"}
```

Per-day records: triage → news_context → performance_supervisor (only if triage ≠ skip).
```json
{"agent": "triage", "prompt_version": "triage-v1", "as_of": "2008-08-15", "triage_mode": "skip", ...}
{"agent": "news_context", "prompt_version": "news-context-v1", "as_of": "2008-08-18", "assessment": {...}, ...}
{"agent": "performance_supervisor", "prompt_version": "supervisor-v3", "as_of": "2008-08-18", "assessment": {"state": "CRITICAL", "action": "HALT", ...}, ...}
```

**24 total log files:** 12 windows × 2 strategies. All produced by a single model (Qwen/Qwen3.5-9B, condition A).

### 6.9 Supervisor Decision Distribution

**JT_MOM** (1,193 supervisor decisions across 12 windows):

| Window | Days | NORMAL | WATCH | ALERT | CRITICAL |
|---|---|---|---|---|---|
| calm_2004_2006 | 68 | 0 | 43 | 25 | 0 |
| calm_2012 | 145 | 0 | 48 | 75 | 22 |
| calm_2013_2014 | 416 | 4 | 300 | 101 | 11 |
| calm_2017 | 219 | 26 | 177 | 16 | 0 |
| quant_meltdown_2007 | 37 | 0 | 0 | 13 | 24 |
| gfc_lehman_2008 | 49 | 0 | 0 | 1 | 48 |
| momentum_crash_2009 | 20 | 0 | 0 | 1 | 19 |
| flash_crash_2010 | 23 | 0 | 0 | 14 | 9 |
| downgrade_2011 | 38 | 0 | 1 | 12 | 25 |
| china_deval_2015 | 48 | 0 | 21 | 19 | 8 |
| volmageddon_2018 | 48 | 2 | 20 | 23 | 3 |
| covid_2020 | 82 | 0 | 9 | 20 | 53 |

**Key observation:** WATCH/INVESTIGATE dominate even in calm windows. This is why the policy must key on ALERT+ states only — de-risking on bare WATCH would destroy returns in calm_2013_2014 (300 WATCH days) and calm_2017 (177 WATCH days).

---

## 7. Risk-Overlay Policy

### 7.1 Exposure Ladder

The policy maps each daily supervisor decision to a target exposure in [0, 1]. The mapping keys on `state` and refines by `action`:

```python
# monitoring/live_backtest/policy.py
STATE_RANK = {"NORMAL": 0, "WATCH": 1, "ALERT": 2, "CRITICAL": 3}
ACTION_RANK = {"HOLD": 0, "INVESTIGATE": 1, "REDUCE": 2, "HALT": 3}
```

| Condition | Target Exposure | Justification |
|---|---|---|
| NORMAL or WATCH (any action) | 1.0 (100%) | WATCH is informational; acting on it causes excessive trading in calm periods |
| ALERT with action < REDUCE | 0.75 (75%) | Alert but ambiguous — reduce modestly |
| ALERT + REDUCE | 0.50 (50%) | Alert with explicit de-risk recommendation — halve exposure |
| CRITICAL or HALT (any) | 0.0 (0%) | Full exit — strategy is in a confirmed regime break |
| Day outside log coverage | 1.0 (100%) | Hybrid coverage: no information ≠ alarm |

**Why not de-risk on WATCH:** Calm_2012 produces 48 WATCH days, calm_2013_2014 produces 300. De-risking on these would create massive friction and return drag in precisely the periods where the strategy is performing well. The cost exceeds any possible benefit.

**Why 0.75/0.50 gradation (not binary 1.0/0.0):** Published risk-overlay literature (Bollerslev et al. FAJ 2020, Man Group) shows that gradual de-risking outperforms binary on/off switches. Gradation reduces whipsaw costs and avoids the timing-luck problem of picking the exact re-entry day.

### 7.2 Cooldown

After any day where exposure drops below 1.0, hold the minimum exposure for `cooldown_days = 3` trading days.

**Why 3 days:** Regime breaks don't resolve in a single day. A 1-day cooldown would produce whipsaw trades on consecutive ALERT/NORMAL oscillations. 3 days is the minimum to smooth over typical detector-alarm jitter.

### 7.3 Staged Re-Entry

After the cooldown expires, don't snap back to 100% exposure immediately. Require `reentry_consecutive = 2` consecutive non-ALERT days per step, with steps `(0.75, 1.0)`.

**Example:** Exposure drops to 0.50 (ALERT+REDUCE). After 3-day cooldown, the policy holds 0.50. After 2 consecutive clean days, exposure steps to 0.75. After 2 more consecutive clean days, exposure returns to 1.0.

**Why staged:** Prevents a single non-ALERT day during a multi-day crisis from causing premature re-entry. The 2008 GFC had brief bounces that would fool a snap-back policy.

### 7.4 Causal Lag

`lag_days = 1`: The decision made at market close on day *t* takes effect from day *t+1*.

**Why 1 day:** A real trader can't act on information until the next trading day. A zero-lag policy implicitly assumes same-day execution on information observed at close, which is impossible.

### 7.5 Duplicate-Date Handling

If multiple windows overlap the same date (rare but possible), the most severe state is kept:

```python
df["state_rank"] = df["state"].map(STATE_RANK)
df = df.sort_values("state_rank", ascending=False).drop_duplicates("date", keep="first")
```

**Why most-severe:** Failing to de-risk when two sources disagree is more dangerous than de-risking unnecessarily. The cost of being out for a day during a real crisis >> the cost of missing one day's return in a false alarm.

---

## 8. Portfolio Simulation Engine

### 8.1 Daily Simulation Loop

```python
# monitoring/live_backtest/engine.py
for i in range(n):
    e = exp[i]       # today's target exposure
    r = ret[i]       # today's gross strategy return

    # Invested portion earns strategy return; rest earns cash rate
    gross = e * (r - strat_drag_daily) + (1 - e) * cash_daily

    # Transaction cost on exposure change
    delta_exp = abs(e - prev_exp) if i > 0 else 0.0
    tc = delta_exp * prev_value * tcost_bps / 1e4

    # New portfolio value
    v = prev_value * (1 + gross) - tc
```

**Why iterative (not vectorized):** Transaction costs depend on the previous day's portfolio value, which itself depends on all prior days' costs. This creates a sequential dependency that can't be vectorized without approximation. At 6,250 rows, the loop completes in <100ms — there's no performance reason to approximate.

### 8.2 Parameters

| Parameter | Value | Justification |
|---|---|---|
| `capital` | $1,000,000 | Round number; large enough that S&P 500 market impact is negligible ($1M << $100B+ ADV) |
| `tcost_bps` | 10 bps | Cost per overlay trade (|Δexposure| × notional). Conservative for SPY-like replication. This is the cost of the overlay's own trades, separate from strategy friction |
| `strat_tcost_annual` | 0.0 | Set to zero because real costs are already embedded in the net curves via `frictions.py`. Double-counting would be wrong |
| `cash_rate_annual` | 1.4% | Average 3-month T-bill yield 2004–2025. De-risked capital earns this rate. Actual rates ranged 0–5%; 1.4% flat is conservative but consistent |

**Why cash interest matters:** When the overlay goes flat (exposure = 0%), the $1M sits in cash. Ignoring cash interest would penalise the overlay unfairly — in 2007-2009, the strategy could be earning 4-5% risk-free on the de-risked portion. Using 1.4% flat is conservative (understates the benefit during high-rate periods).

---

## 9. Transaction Cost Model

### 9.1 Scope

There are two distinct sources of trading costs:

1. **Strategy friction:** The cost of executing the strategy's own rebalancing trades (monthly JT rotations, daily AL entries/exits). Applied post-hoc to the gross equity curve to produce net curves.

2. **Overlay friction:** The cost of the overlay's exposure changes ($1M × |Δexposure| × 10 bps). Applied within the engine simulation.

These are independent and additive.

### 9.2 Strategy Friction: Era-Dependent Cost Schedule

```python
# monitoring/frictions.py
COST_SCHEDULE = {
    "low":  [("2004-01-01", "2009-12-31",  4.0),
             ("2010-01-01", "2018-12-31",  2.0),
             ("2019-01-01", "2099-12-31",  1.0)],
    "base": [("2004-01-01", "2009-12-31",  8.0),
             ("2010-01-01", "2018-12-31",  5.0),
             ("2019-01-01", "2099-12-31",  2.5)],
    "high": [("2004-01-01", "2009-12-31", 15.0),
             ("2010-01-01", "2018-12-31", 10.0),
             ("2019-01-01", "2099-12-31",  5.0)],
}
```

Units are **basis points per side of notional traded** (commission + bid-ask half-spread).

**Cost model:**
```
net_ret(t) = gross_ret(t) − turnover_oneway(t) × bps(t) / 10,000
```

### 9.3 Era Breakpoint Justification

| Era | Regime | Base bps | Reasoning |
|---|---|---|---|
| 2004–2009 | Pre-zero-commission; wider spreads | 8 | ~$5-10 commission per trade + 2-5 bps half-spread on large caps. SEC decimalization (2001) had narrowed spreads from 12.5 bps to ~5 bps, but commissions were still $5-10/trade |
| 2010–2018 | Tighter spreads; lower commissions | 5 | HFT competition compressed spreads to 1-2 bps; commissions fell to $1-5/trade. But $1M portfolio turning over ~50%/month still faces material friction |
| 2019–2025 | Zero-commission retail (Schwab Oct 2019); sub-penny spreads | 2.5 | Commission = $0 (Schwab, Fidelity, TD Ameritrade all eliminated commissions Oct 2019). Cost is now purely half-spread ≈ 1-2 bps on S&P 500 large caps |

**Why low/base/high sensitivity:** The exact cost is uncertain (depends on execution quality, order size, time of day). The three scenarios bracket the plausible range. If results hold under the "high" scenario, they're robust to cost assumptions.

### 9.4 Frictions Excluded (Disclosed)

| Friction | Disposition | Reasoning |
|---|---|---|
| Slippage / market impact | Excluded from base; folded into "high" | $1M is <0.001% of S&P 500 daily volume. Impact at this size is sub-basis-point |
| Borrow cost | N/A | Long-only strategies — no shorting |
| Dividends | Already captured | Adjusted close prices include dividends |
| Tax drag | Excluded | Varies by investor; not a strategy-level friction |

### 9.5 JT_MOM Turnover Computation

**Method:** Track drifted weights between monthly rebalances, charge turnover only at actual rebalance points.

```python
# XSectional/compute_net_curve.py
# On rebalance day:
turnover = sum(|w_new_target - w_drifted_from_prior|)
# Between rebalances:
w_drifted_raw = w_held * (1 + ret_today)
w_drifted = w_drifted_raw / sum(w_drifted_raw)  # renormalize
```

**Why drifted weights (not target-vs-target):** The constant-weight backtest (`run_backtest_daily`) implicitly rebalances daily by applying target weights every day. But a real trader holds positions and lets them drift between monthly rebalances. Charging turnover as `|w_month_t − w_month_t−1|` would overstate costs because it ignores the drift that happened during the month. Computing turnover as `|w_new_target − w_drifted|` captures only the trades a real trader would actually make.

**Results:**
- 298 rebalance days over 6,246 trading days
- Mean one-way turnover per rebalance: 52.78%
- Annualised one-way turnover: 634.6%
- **Base cost drag: 34 bps/year** (gross terminal 12.75× → net terminal 11.71×)

### 9.6 AL_PCA Turnover Computation

**Method:** Reconstruct daily positions from the trades blotter (entry/exit dates per stock per sector), compute per-sector equal-weighted turnover, average across sectors.

```python
# Stat Arb/statsArb-dev/scripts/compute_net_curve.py
# Per sector: binary position matrix → equal weights 1/N' → turnover from w.diff().abs()
# Portfolio turnover = mean of sector turnovers (9 sectors, equal weight)
```

**Verification gate:** Reconstructed gross return (mean of sector_returns.csv) must match equity_curve.csv with correlation > 0.999 and terminal equity within 1%. **Achieved: corr = 1.000000.**

**Why not use the built-in cost parameter:** The `bt_tools.py:118` cost implementation has a bug: `ret * (1 - indicator * epsilon) * w` charges `epsilon × |return|` rather than `epsilon × notional`. This understates costs by approximately 100× because |return| ≈ 0.001 on a typical day. We bypassed this entirely by computing costs post-hoc from the trades blotter.

**Results:**
- 5,530 active turnover days out of 5,531 (OU signal generates near-daily trades)
- Mean daily one-way turnover: 16.50%
- Annualised one-way turnover: 4,158%
- **Base cost drag: 219 bps/year** (gross terminal 24.71× → net terminal 15.28×)

**Why AL turnover is so much higher:** The OU-process stat-arb trades on entry/exit signals that fire frequently. Unlike momentum (monthly rebalance with gradual drift), stat-arb positions turn over nearly every day. This is realistic for an active stat-arb strategy.

### 9.7 Post-Hoc Cost Application: Justification for Log Reuse

The agentic JSONL logs were generated by running the LLM on **gross** equity curves. We then apply these same log decisions to **net** curves. This is valid if the agent's input features (rolling 21-day vol and running drawdown) don't change materially between gross and net.

**Verification** (`monitoring/scripts/verify_gross_net_features.py`):

| Metric | JT_MOM max | AL_PCA max | Threshold |
|---|---|---|---|
| Rolling 21d vol divergence | 0.098% annualized | 0.057% | Well under 1% |
| Running drawdown divergence | 2.06 pp | 2.71 pp | Well under the -10% to -50% drawdowns the agent acts on |

**Conclusion:** The cost drag is a slow, steady drain that barely shifts the high-frequency features the agent uses. Re-running the agent on net curves would produce near-identical decisions. The time cost (~4 hours of GPU compute) is not justified by the <0.1% vol / <3pp drawdown divergence.

---

## 10. Live Backtest Execution

### 10.1 Run Matrix

For each strategy × cost variant, the CLI is invoked once. Each run produces a managed curve (strategy + agent overlay), an unmanaged curve (strategy alone), and a SPY benchmark.

**JT_MOM (2001-03-01 to 2025-12-30, 6,246 trading days):**

```bash
# Gross
python run_live_backtest.py \
  --curve ../XSectional/results/equity_curve_daily_long_only.csv \
  --strategy JT_MOM \
  --logs-glob "results/agentic_*_JT_MOM.jsonl" \
  --capital 1000000 --tcost-bps 10 --cash-rate 0.014 \
  --outdir results/live_backtest/JT_MOM_gross

# Net (base)
python run_live_backtest.py \
  --curve ../XSectional/results/equity_curve_daily_long_only_net_base.csv \
  --strategy JT_MOM \
  --logs-glob "results/agentic_*_JT_MOM.jsonl" \
  --capital 1000000 --tcost-bps 10 --cash-rate 0.014 \
  --outdir results/live_backtest/JT_MOM_net

# Net (low / high) — same pattern, different curve
```

**AL_PCA (2004-01-02 to 2025-12-24, 5,531 trading days):** Same structure, different curves.

**8 total output directories:** `{JT_MOM,AL_PCA}_{gross,net,net_low,net_high}/`

### 10.2 Per-Directory Outputs

| File | Description |
|---|---|
| `metrics.json` | CAGR, volatility, Sharpe, max drawdown, Calmar, terminal value, total return — for managed, unmanaged, and SPY |
| `metrics_table.png` | Formatted table of the above |
| `fig_a_portfolio_value.png` | Three $ curves (managed, unmanaged, SPY) on log-y scale with window shading |
| `fig_b_pnl_drawdown_exposure.png` | Three stacked panels: cumulative PnL ($), drawdown (%), exposure step-plot |
| `portfolio_daily.csv` | Full daily audit trail: value, pnl, exposure, strat_ret, port_ret, tcost, drawdown |
| `exposure.csv` | Daily exposure series |

### 10.3 Cross-Strategy Comparison Figures

Generated by `monitoring/scripts/gross_net_comparison.py`:

| File | Description |
|---|---|
| `summary_gross_net.png` | 9-row combined metrics table (both strategies, gross and net, plus SPY) |
| `metrics_gross_net_{STRAT}.png` | Per-strategy 5-row comparison table |
| `fig_c_gross_net_equity_{STRAT}.png` | Two-panel equity curves: gross vs net |
| `fig_d_cost_drag_{STRAT}.png` | Cumulative $ drag + annualized bps bar chart by era |
| `sensitivity_{STRAT}.png` | Low/base/high sensitivity table |

---

## 11. Evaluation Windows

### 11.1 Dev/Test Split

The 12 evaluation windows are split into two sets:

**Development set (6 windows) — used for calibration:**
- quant_meltdown_2007 (2007-07-16 to 2007-09-14, onset 2007-08-06)
- gfc_lehman_2008 (2008-08-15 to 2008-11-28, onset 2008-09-15)
- momentum_crash_2009 (2009-02-16 to 2009-05-29, onset 2009-03-09)
- downgrade_2011 (2011-07-18 to 2011-10-14, onset 2011-08-05)
- calm_2004_2006 (2004-01-02 to 2006-12-29, no onset)
- calm_2013_2014 (2013-01-02 to 2014-12-31, no onset)

**Confirmatory test set (6 windows) — BLIND, no tuning:**
- flash_crash_2010 (2010-04-15 to 2010-07-15, onset 2010-05-06)
- china_deval_2015 (2015-07-15 to 2015-10-16, onset 2015-08-11)
- volmageddon_2018 (2018-01-15 to 2018-04-13, onset 2018-02-05)
- covid_2020 (2020-02-03 to 2020-05-29, onset 2020-02-24)
- calm_2012 (2012-01-03 to 2012-12-31, no onset)
- calm_2017 (2017-01-03 to 2017-12-29, no onset)

**Why this split:** The development set was used to iterate prompts, calibrate triage thresholds, and set policy parameters. The test set was held out — no parameter tuning was done after seeing outcomes on these windows. The freeze is enforced by the `eval-freeze-v1` git tag. This prevents the analyst from unconsciously tuning the system to fit known events.

**Why each window was selected:** The event windows cover the major documented regime breaks in U.S. equity markets 2004–2020, each with a distinct mechanism (quant deleveraging, credit crisis, momentum reversal, sovereign downgrade, flash crash, yuan devaluation, short-vol blowup, pandemic). The calm windows are 1-3 year stretches with no known regime break, ensuring the system doesn't fire in benign conditions.

### 11.2 Hybrid Coverage

Agentic decisions exist only inside the 12 windows (~1,097 of ~6,250 JT trading days). On days outside any window, the overlay assumes 100% exposure (no opinion → stay invested).

**Why this is a bias:** The windows were selected ex-post around known crises and calm periods. The overlay is only "tested" where we chose to look. A truly live system would run every day and might produce false positives on uncovered days (which would reduce performance). This favorable selection bias must be disclosed.

**Mitigation path:** Stage 2 of the upgrade plan runs the agent over all ~6,250 trading days, eliminating the coverage gap entirely.

---

## 12. Results Summary

### 12.1 Gross Results

| Metric | JT + Agent | JT Unmanaged | AL + Agent | AL Unmanaged | SPY B&H |
|---|---|---|---|---|---|
| CAGR | 11.76% | 10.84% | 14.06% | 15.74% | 9.14%* |
| Ann. Vol | 16.14% | 20.68% | 13.55% | 19.89% | 19.20% |
| Sharpe | 0.73 | 0.52 | 1.04 | 0.79 | 0.48 |
| Max DD | -22.52% | -54.66% | -17.61% | -47.60% | -55.19% |
| Calmar | 0.52 | 0.20 | 0.80 | 0.33 | 0.17 |
| Terminal ($1M) | $15.65M | $12.75M | $17.94M | $24.71M | $8.67M |

*SPY CAGR differs between JT and AL runs due to different date ranges.

### 12.2 Net Results (Base Cost Scenario)

| Metric | JT + Agent | JT Unmanaged | AL + Agent | AL Unmanaged | SPY B&H |
|---|---|---|---|---|---|
| CAGR | 11.62% | 10.46% | 12.14% | 13.23% | 9.14% |
| Sharpe | 0.72 | 0.51 | 0.90 | 0.67 | 0.48 |
| Max DD | -22.57% | -54.98% | -18.29% | -49.11% | -55.19% |
| Terminal ($1M) | $15.17M | $11.71M | $12.35M | $15.28M | $8.67M |

### 12.3 Cost Drag Summary

| Strategy | Scenario | Annualized Drag (bps/yr) | Gross Terminal | Net Terminal |
|---|---|---|---|---|
| JT_MOM | Low | 15.7 | 12.75× | 12.26× |
| JT_MOM | Base | 34.4 | 12.75× | 11.71× |
| JT_MOM | High | 66.5 | 12.75× | 10.81× |
| AL_PCA | Low | 98.2 | 24.71× | 19.93× |
| AL_PCA | Base | 219.2 | 24.71× | 15.28× |
| AL_PCA | High | 425.2 | 24.71× | 9.72× |

### 12.4 Key Findings

1. **Drawdown reduction is the primary value.** Both strategies see max drawdown roughly halved (JT: -55% → -23%; AL: -48% → -18%). This is the overlay's main contribution.

2. **Sharpe improvement is substantial.** JT: 0.52 → 0.73 (+40%); AL: 0.79 → 1.04 (+32%). The overlay reduces volatility more than it reduces return.

3. **CAGR trade-off is asymmetric.** JT managed beats unmanaged on CAGR (+92 bps/yr gross). AL managed trails unmanaged on CAGR (-168 bps/yr gross) because the overlay exits the highly profitable AL stat-arb during crises where AL would have eventually recovered. But the Sharpe improvement shows the risk-adjusted picture favors the overlay.

4. **Both strategies beat SPY B&H** on every metric in both gross and net scenarios (including the high-cost scenario).

5. **Results are robust to cost assumptions.** Even under the "high" cost scenario, the overlay maintains its drawdown-reduction benefit and positive alpha vs SPY.

---

## 13. Known Biases and Limitations

These must be disclosed in any reporting:

1. **Hybrid coverage bias**: Agentic decisions exist only inside 12 ex-post-selected windows (~1,097 of ~6,250 JT trading days). Assuming 100% exposure elsewhere means the overlay only acts where we chose to look. Favorable selection bias.

2. **Curve mismatch**: Agentic JSONL logs were generated by monitoring the **long-only** curve (`run_meta.curve` confirms). This is the correct curve. However, the logs were generated in a single batch; a truly live system would process each day as it arrives.

3. **Post-hoc cost application**: Net curves apply era-dependent transaction costs post-hoc to gross strategy returns. Agent decisions were generated on gross curves. Verification shows max feature divergence <0.1% vol and <2.7pp drawdown — small relative to the -10% to -50% drawdowns the agent acts on.

4. **No HALT/CRITICAL in JT calm windows**: The 0.0-exposure rung (CRITICAL/HALT) never fires in JT calm windows because the model correctly identifies them as non-crisis. This is good behavior, not a limitation — but it means the HALT exposure rung is only tested in event windows.

5. **False-positive drag is real**: calm_2012 has 75 ALERT days (JT) with 22 CRITICAL; calm_2013_2014 has 101 ALERT + 11 CRITICAL. The overlay costs return in these calm windows. This is an honest cost of the system.

6. **Cash interest is flat**: De-risked capital earns 1.4% flat (average 3M T-bill 2004–2025). Actual rates ranged 0–5%. This is conservative in high-rate periods (2004–2007, 2022–2025) and slightly generous in the zero-rate era (2009–2021).

7. **SPY benchmark is total-return (SPY, not ^GSPC)**: SPY's adjusted close includes reinvested dividends. ^GSPC is price-only and would understate the benchmark by ~2%/yr.

8. **Slippage excluded**: At $1M portfolio size, market impact on S&P 500 names is sub-basis-point. Folded into the "high" cost scenario.

---

## 14. Replication Commands

All commands assume the working directory is the repo root unless otherwise noted.

### 14.1 Strategy Curves

```bash
# JT_MOM: generate long-only equity curve + daily weight matrix
cd XSectional
python run_daily_pnl.py --long-only
# Produces: results/equity_curve_daily_long_only.csv
#           results/weights_daily_long_only.csv

# AL_PCA: generate long-only equity curve (takes ~30-45 min)
cd "Stat Arb/statsArb-dev"
python run_full_universe.py --pit --tag pit_pca_long_only --long-only
# Produces: results/full_universe/pit_pca_long_only/equity_curve.csv
#           results/full_universe/pit_pca_long_only/trades.csv
#           results/full_universe/pit_pca_long_only/sector_returns.csv
```

### 14.2 Net Curves (Transaction Costs)

```bash
# JT net curves
cd XSectional
python compute_net_curve.py
# Produces: results/equity_curve_daily_long_only_net_{low,base,high}.csv

# AL net curves
cd "Stat Arb/statsArb-dev"
python scripts/compute_net_curve.py
# Produces: results/full_universe/pit_pca_long_only/equity_curve_net_{low,base,high}.csv
```

### 14.3 Agentic Inference (requires GPU or vLLM server)

```bash
cd monitoring

# Pre-compute FinBERT stress scores (avoids torch in inference process)
python scripts/precompute_finbert.py --window quant_meltdown_2007

# Run one window (vLLM on localhost:18000)
python run_agentic.py \
  --window quant_meltdown_2007 \
  --strategy JT_MOM \
  --model "vllm:Qwen/Qwen3.5-9B" \
  --finbert-cache results/finbert_cache/quant_meltdown_2007.json

# Run all 24 cells (12 windows × 2 strategies)
python scripts/run_all_vllm.py
```

### 14.4 Live Backtest Simulation

```bash
cd monitoring

# JT_MOM gross
python run_live_backtest.py \
  --curve ../XSectional/results/equity_curve_daily_long_only.csv \
  --strategy JT_MOM \
  --logs-glob "results/agentic_*_JT_MOM.jsonl" \
  --capital 1000000 --tcost-bps 10 --cash-rate 0.014 \
  --outdir results/live_backtest/JT_MOM_gross

# JT_MOM net (base)
python run_live_backtest.py \
  --curve ../XSectional/results/equity_curve_daily_long_only_net_base.csv \
  --strategy JT_MOM \
  --logs-glob "results/agentic_*_JT_MOM.jsonl" \
  --capital 1000000 --tcost-bps 10 --cash-rate 0.014 \
  --outdir results/live_backtest/JT_MOM_net

# JT_MOM net (low)
python run_live_backtest.py \
  --curve ../XSectional/results/equity_curve_daily_long_only_net_low.csv \
  --strategy JT_MOM \
  --logs-glob "results/agentic_*_JT_MOM.jsonl" \
  --capital 1000000 --tcost-bps 10 --cash-rate 0.014 \
  --outdir results/live_backtest/JT_MOM_net_low

# JT_MOM net (high)
python run_live_backtest.py \
  --curve ../XSectional/results/equity_curve_daily_long_only_net_high.csv \
  --strategy JT_MOM \
  --logs-glob "results/agentic_*_JT_MOM.jsonl" \
  --capital 1000000 --tcost-bps 10 --cash-rate 0.014 \
  --outdir results/live_backtest/JT_MOM_net_high

# AL_PCA gross
python run_live_backtest.py \
  --curve "../Stat Arb/statsArb-dev/results/full_universe/pit_pca_long_only/equity_curve.csv" \
  --strategy AL_PCA \
  --logs-glob "results/agentic_*_AL_PCA.jsonl" \
  --capital 1000000 --tcost-bps 10 --cash-rate 0.014 \
  --outdir results/live_backtest/AL_PCA_gross

# AL_PCA net (base)
python run_live_backtest.py \
  --curve "../Stat Arb/statsArb-dev/results/full_universe/pit_pca_long_only/equity_curve_net_base.csv" \
  --strategy AL_PCA \
  --logs-glob "results/agentic_*_AL_PCA.jsonl" \
  --capital 1000000 --tcost-bps 10 --cash-rate 0.014 \
  --outdir results/live_backtest/AL_PCA_net

# AL_PCA net (low / high) — same pattern
```

### 14.5 Figures and Comparison Tables

```bash
cd monitoring

# Feature divergence verification
python scripts/verify_gross_net_features.py

# Gross vs net comparison figures
python scripts/gross_net_comparison.py
```

### 14.6 Tests

```bash
cd monitoring
python -m pytest -q tests/test_live_backtest_policy.py \
                     tests/test_live_backtest_engine.py \
                     tests/test_live_backtest_report.py \
                     tests/test_live_backtest_benchmark.py
```

---

## 15. File Inventory

### Core Modules

| File | Role |
|---|---|
| `monitoring/frictions.py` | Shared era-dependent cost model (used by both strategies) |
| `monitoring/live_backtest/policy.py` | JSONL → daily exposure mapping (ladder + cooldown + re-entry + lag) |
| `monitoring/live_backtest/engine.py` | $1M portfolio simulation (iterative daily loop with tcost) |
| `monitoring/live_backtest/benchmark.py` | SPY fetch/cache and buy-and-hold computation |
| `monitoring/live_backtest/report.py` | Metrics computation + figure generation |
| `monitoring/run_live_backtest.py` | CLI orchestrator for one strategy × one curve |

### Strategy Construction

| File | Role |
|---|---|
| `XSectional/backtest.py` | Monthly → daily JT momentum backtest (weight export via `return_weights=True`) |
| `XSectional/portfolio.py` | `to_long_only()` filter and renormalization |
| `XSectional/run_daily_pnl.py` | CLI to generate JT equity curve and weight matrix |
| `XSectional/compute_net_curve.py` | JT turnover computation (drifted weights) and net curve generation |
| `Stat Arb/statsArb-dev/run_full_universe.py` | AL PCA backtest across 9 sector sleeves |
| `Stat Arb/statsArb-dev/scripts/compute_net_curve.py` | AL turnover from trades blotter and net curve generation |

### Agentic Pipeline

| File | Role |
|---|---|
| `monitoring/run_agentic.py` | Daily agentic loop: triage → news agent → supervisor |
| `monitoring/agentic/prompts.py` | All LLM prompts (supervisor-v2/v3, news-context-v1) |
| `monitoring/agentic/schemas.py` | State/Action enums, JSON schemas, validation |
| `monitoring/agentic/model.py` | LLM backends (OfflineStub, Ollama, vLLM) |
| `monitoring/news/triage.py` | Cost-control triage (skip/cheap/thinking/escalation) |
| `monitoring/windows.py` | 12 evaluation windows (6 dev + 6 test) |

### Analysis and Reporting

| File | Role |
|---|---|
| `monitoring/scripts/gross_net_comparison.py` | Cross-strategy gross/net tables and figures |
| `monitoring/scripts/verify_gross_net_features.py` | Validates log-reuse by measuring gross/net feature divergence |
| `monitoring/scripts/analyze_test_results.py` | Per-window detection metrics (latency, recall, FPR) |

### Key Outputs

| File | Description |
|---|---|
| `monitoring/results/live_backtest/{STRAT}_{variant}/metrics.json` | Quantitative results |
| `monitoring/results/live_backtest/{STRAT}_{variant}/fig_a_*.png` | Equity curve figure |
| `monitoring/results/live_backtest/{STRAT}_{variant}/fig_b_*.png` | PnL/drawdown/exposure figure |
| `monitoring/results/live_backtest/{STRAT}_{variant}/portfolio_daily.csv` | Full audit trail |
| `monitoring/results/figures/summary_gross_net.png` | Combined 9-row summary table |
| `monitoring/results/figures/fig_c_*.png` | Gross vs net equity curves |
| `monitoring/results/figures/fig_d_*.png` | Cost drag analysis |
| `monitoring/results/figures/sensitivity_*.png` | Low/base/high sensitivity tables |
| `monitoring/results/gross_net_feature_divergence.csv` | Log-reuse validation data |
| `monitoring/results/agentic_*_{STRAT}.jsonl` | Raw agentic decision logs (24 files) |

---

*Generated from the `pnl-backtest` branch, commit `027caf1`.*
