# Experimental Setup: Agentic vs. Classical In-Market Regime Monitoring

**Project:** In-market agentic-AI monitoring of quantitative trading strategies
**Scope of this document:** the experimental design, data methodology, system
architecture, evaluation protocol, and the multi-market extension (US S&P 500 →
global energy), including a worked case study of the agentic layer applied to the
March-2020 oil crash.

---

## 1. Research question and hypotheses

Quantitative trading strategies degrade when the market **regime** they rely on
breaks (e.g., a mean-reversion book during a momentum crash). The question:

> Can an **LLM-agent monitoring layer**, fed news and macro context under strict
> information parity, detect strategy regime breaks **faster** and with **fewer
> false positives** than classical statistical change-point detectors?

Formal hypotheses:

- **H1 — Superiority.** Agentic monitoring detects regime breaks with **lower
  detection latency** *and* a **lower false-positive rate** than the classical layer.
- **H2 — Generalisation.** The advantage holds across **both** strategies (a
  statistical-arbitrage book and a cross-sectional momentum book).
- **H3 — Heterogeneity.** The size of the advantage **differs by strategy**.

The design is a **2 × 2 × N**: {classical, agentic} × {strategy A, strategy B} ×
{N labelled evaluation windows}.

---

## 2. Core methodological principle: information parity (no look-ahead)

The single most important control. **Neither layer may see anything that was not
knowable on the decision date.** This is enforced at four levels:

1. **Survivorship-bias-free universe.** Strategy universes are reconstructed
   point-in-time, *including delisted names*. This is not cosmetic: the August-2007
   quant meltdown is **invisible** on a survivor-only S&P 500 (+3.4% in the window)
   and only appears on the honest universe (−4.8%, worst day −4.7%).
2. **As-of macro data.** Macro series are pulled from **ALFRED vintages** (the value
   as *published* by the decision date), not the latest revised value — using the
   revised value would be look-ahead.
3. **Timestamp-filtered news.** Every news article is filtered by its publication
   timestamp to the decision date.
4. **Fail-closed lookahead assertion.** Every context object passed to an agent is
   run through a `assert_no_lookahead` guard before the LLM call.

A separate **pre-training-leakage control** (see §8) addresses a subtler risk unique
to LLMs: the model may "remember" a famous crash from its training data.

---

## 3. Strategies under monitoring

Two deliberately **independent** strategies (daily return correlation ≈ 0.003) that
break on *different* events — so a monitor that only works on one is exposed:

| Strategy | Type | Characteristic break |
|---|---|---|
| **AL PCA** | Avellaneda–Lee statistical arbitrage (PCA-defactored mean reversion) | 2007 quant meltdown |
| **JT momentum** | Jegadeesh–Titman cross-sectional momentum (long top decile / short bottom) | 2009 momentum crash |

Each strategy emits a daily PnL curve with a fixed schema (`Date, port_ret, equity,
drawdown`) that both monitoring layers consume identically.

---

## 4. Classical monitoring layer (the baseline)

Four statistical change-point detectors run **continuously** over each daily PnL
curve (so every window inherits genuine warm-up history):

- **Page–Hinkley** (cumulative-deviation test)
- **BOCPD** (Bayesian online change-point detection)
- **2-state HMM** (regime-switching; HMM parameters fit **out-of-sample** on
  pre-event data)
- **Distributional threshold** (rolling tail test)

**Aggregation rule:** an alarm fires when **≥ 2 detectors trigger within 5 days**.
This is the classical baseline against which the agentic layer is measured.

---

## 5. Agentic monitoring layer (the treatment)

A pipeline of LLM agents (local `qwen2.5:3b` via Ollama, structured-JSON outputs)
fed the same point-in-time data:

```
Guardrails ─► Triage ─► News Context Agent ─► Performance Supervisor ─► JSONL audit log
(as-of,      (skip /   (reads filtered news, (telemetry + classical
 timestamp,   cheap /   emits risk flags +    alarms + news summary
 lookahead)   thinking/ narrative)            → {state, action,
              escalate)                         root_cause, confidence})
```

- **Guardrails** enforce information parity (§2) and fail closed.
- **Triage** decides how much model to spend per day: `skip` (quiet days),
  `thinking`, or `classical_escalation` — controlling inference cost.
- **News Context Agent** ingests the day's filtered news and emits structured risk
  signals.
- **Performance Supervisor (v3 prompt)** ingests strategy telemetry, the classical
  detector alarms, and the news summary, and emits a validated JSON assessment:
  `state ∈ {NORMAL, WATCH, ALERT}`, `action`, `root_cause`, `confidence`.
- Every model call is logged to JSONL with prompt version, raw output, validated
  assessment, and latency — for full auditability and blind qualitative scoring.

---

## 6. Multi-market design

To test **H2 (generalisation)** beyond a single market, the system was refactored
behind a **market-provider abstraction** (`MarketProfile` + provider interfaces for
universe, strategy PnL, news, macro, calendar). Each market plugs in its own data
sources behind stable interfaces; the classical + agentic pipeline is shared and
identical across markets. The US S&P 500 path is preserved **byte-identical** through
the refactor (a golden-path regression test enforces this).

| Market | Universe | News | Macro | Status |
|---|---|---|---|---|
| **US S&P 500** | PIT membership incl. delisted | FNSPID (11.8M articles) | FRED/ALFRED (rates, VIX) | Reference |
| **Global energy** | Curated ~34 megacaps (US-listed + ADR) | GDELT (point-in-time) | FRED commodity curves (WTI, Brent, Henry Hub) | Implemented |
| **ASX 200** | — | — | — | Deferred (survivorship-free data not freely available) |

**Energy-market design notes.** Energy regime breaks are commodity-driven, so the
macro provider carries oil/gas curves rather than only rates. Because energy is a
*single sector*, the AL statistical-arbitrage engine is run in **PCA-defactoring
mode** over the whole universe (PCA removes the common oil factor; the strategy
trades the mean-reverting residuals) rather than the sector-ETF sleeve scheme used
for the broad S&P 500. The curated fixed universe carries a documented, minor
survivorship caveat (energy megacaps rarely delist; M&A exits are enumerated).

---

## 7. Evaluation design

**Windows.** Each market is evaluated on labelled slices of the timeline, split into
a **development set** (used for calibration and prompt iteration — treated as
contaminated) and a **blind confirmatory test set** (no parameter tuning or prompt
changes after seeing outcomes; access gated by a freeze tag). US uses 4 event + 2
calm dev windows and 6 blind test windows. Energy uses windows within the news-data
coverage range (e.g., the **2020 oil crash**, the **2022 energy spike**, plus calm
controls). Each **event** window carries a ground-truth **onset** date; each **calm**
window carries none (any alarm in a calm window is a false positive).

**Metrics** (computed per detector/agent, per window, and aggregated):

- **Detection latency** — trading days from onset to first alarm (H1: lower is better)
- **False-positive rate** — alarms per day inside calm windows (H1: lower is better)
- **Precision / recall** — on event detection
- **Qualitative reasoning score** — blind rubric on agent root-cause narratives

---

## 8. Methodological controls (validity threats and mitigations)

| Threat | Mitigation |
|---|---|
| Survivorship bias | PIT universe incl. delisted; gating test that a known crash is invisible on survivor-only data |
| Look-ahead in macro | ALFRED vintages (as-published values) |
| Look-ahead in news | Publication-timestamp filtering + fail-closed lookahead assertion |
| **LLM pre-training leakage** | **Two-pass evaluation:** Condition A (standard), Condition B (dates masked `XXXX-XX-XX` so the model cannot anchor on famous dates), Condition C (synthetic crash windows with fabricated year-2099 dates and no real event names). A leakage bound is computed: memorisation ≤ A − min(B, C). |
| HMM fit on evaluation data | HMM parameters estimated out-of-sample on pre-event history |
| Cross-strategy confounding | Two statistically independent strategies (corr ≈ 0.003) |

---

## 9. Case study: the agentic layer on the 2020 oil crash

A worked demonstration on the **global-energy** market, window = 2020 oil crash
(2020-02-14 → 2020-05-29), strategy = AL PCA. The agentic monitor ran with real
`qwen2.5:3b` inference over 73 trading days (triage: 56 skip / 12 thinking / 5
escalation), producing **17 structured supervisor assessments**. The agent raised
**`ALERT` on 2020-03-10** (confidence 0.85) — one day after the crash onset
(2020-03-09) — and held `WATCH`/`ALERT` through the turmoil.

**Illustrative PnL overlay** (not the primary experiment, which is detection metrics —
this is a downstream what-if). Mapping the monitor's state to exposure
(NORMAL → full, WATCH → half, ALERT → cash), applied with a **1-day lag (no
look-ahead)**, over a $1M account for the window:

| | Return | Max drawdown |
|---|---|---|
| Baseline (buy & hold energy) | −29.4% | −62.8% |
| Strategy (AL PCA) | +12.1% | −8.8% |
| **Strategy + agentic monitor** | **+8.4%** | **−3.6%** |

The monitor **more than halved the strategy's max drawdown** (−8.8% → −3.6%) at a
cost of ~3.7 pts of return — the risk-management trade a fast monitor is meant to
deliver. See `monitoring/results/energy_pnl_3way.png`. *(A momentum + agentic
version is produced the same way.)*

> **Framing for interpretation.** The core hypotheses (H1–H3) are about **detection
> quality** (latency, FPR, precision/recall), not PnL. The overlay above is an
> intuitive illustration of *why* faster detection matters; it is not itself the
> claimed result, and its exposure-mapping rule is a modelling choice stated
> explicitly.

---

## 10. Reproducibility

All code is version-controlled. Representative commands (run from `monitoring/` in
the project venv):

```bash
# Classical monitoring, either market
python run_classical.py --market sp500  --model stub
python run_classical.py --market energy --model stub

# Agentic monitoring, one window (real LLM)
python run_agentic.py --market energy --window oil_crash_2020 \
    --strategy AL_PCA --model ollama:qwen2.5:3b

# Pre-training leakage harness (Conditions A/B/C)
python run_leakage.py --strategy AL_PCA --model stub
```

Every agentic run emits a JSONL audit log (prompt version, raw + validated output,
latency) enabling blind qualitative scoring and full replay. Strategy PnL curves,
evaluation-window definitions with onsets, and the point-in-time data builders are
all in the repository.

---

## 11. Scope and limitations (stated honestly)

- **ASX 200** is deferred: survivorship-free constituent + delisted-price data is not
  freely available, and the project's validity depends on that data being honest.
- The **energy universe** is a curated fixed set (not the full point-in-time index),
  with a documented minor survivorship caveat justified by energy-megacap stability.
- The **local `qwen2.5:3b`** model is a small, fast model chosen for cost and
  reproducibility; results characterise this model class, not the frontier.
- The **PnL overlay** (§9) is an illustration, not the primary metric.
- Energy news coverage (**GDELT DOC 2.0**) begins ~2017, constraining energy windows
  to that range; pre-2017 energy events require the GDELT GKG raw files.
