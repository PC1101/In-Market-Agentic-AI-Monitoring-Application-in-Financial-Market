# Week 2 Status — Classical monitoring + agentic structure

Against the VRI Week-2 checklist. Code lives in the new top-level `monitoring/` package
plus a daily-PnL addition to `XSectional/`.

## Deliverables

| VRI Week-2 item | Status | Where |
|---|---|---|
| Page-Hinkley, BOCPD, HMM, distributional detectors, unit-tested | ✅ | `monitoring/detectors/`, `monitoring/tests/` |
| Aggregation rule (≥2 detectors within 5 days) | ✅ | `monitoring/detectors/aggregate.py` |
| Classical monitoring run end-to-end, both strategies × 6 windows | ✅ | `monitoring/run_classical.py` — AL PCA + JT momentum, all 6 windows |
| Per-detector + aggregated metrics (latency, FPR, precision, recall) | ✅ | `monitoring/metrics.py` |
| Agentic framework scaffolded (prompts, JSON schema, local model, logging) | ✅ | `monitoring/agentic/` |
| Information-parity guardrails (as-of dating, timestamp filtering) | ✅ | `monitoring/agentic/guardrails.py` |

Tests: **56** in `monitoring/` + **13** new in `XSectional/` (6 daily-backtest,
7 PIT-universe) — all passing.

## Design decisions (agreed)
- Single strategy-agnostic `monitoring/` package consuming the locked daily PnL schema
  `Date,port_ret,equity,drawdown`.
- Detectors implemented from scratch (NumPy/SciPy) — transparent for the methodology write-up.
- JT (XSectional) extended to emit a **daily** curve (`run_backtest_daily`) so both
  strategies are monitored identically at daily frequency.

## Detectors (all online / causal)
- **Page-Hinkley** — volatility-normalized (Welford) so thresholds are in σ-units and
  transfer across strategies; two-sided mean-shift.
- **BOCPD** — Adams & MacKay run-length posterior, Normal-inverse-Gamma predictive;
  alarms on MAP-run-length collapse.
- **HMM** — 2-state Gaussian HMM (Baum-Welch fit, causal forward filter); stressed =
  higher-variance state. Supports an explicit historical `train_returns` for strict parity.
- **Distributional threshold** — recent-vs-baseline z-score + volatility-ratio (scale-free).

## Evaluation design
Six windows in `monitoring/windows.py`: 4 event (quant-meltdown 2007, GFC/Lehman 2008,
momentum-crash 2009, downgrade 2011) + 2 calm (2004-06, 2013-14). Detectors run
*continuously* over each daily curve (real warm-up history for every window); alarms are
then scored per window. An alarm is a true positive within `[onset, onset+21d]`.

## First results — AL PCA (real data, all 6 windows, point-in-time sleeves)

| detector | recall | precision | mean latency (d) | FPR/day (calm) |
|---|---|---|---|---|
| page_hinkley | 0.25 | 0.33 | **3.0** | 0.0008 |
| bocpd | 0.25 | 0.05 | 11.0 | 0.0143 |
| hmm | **1.00** | 0.12 | 10.8 | 0.0183 |
| distributional | 0.25 | 0.14 | 11.0 | 0.0032 |
| aggregate | 0.25 | 0.09 | 11.0 | 0.0072 |

Reading: on the honest universe, Page-Hinkley catches the Aug-2007 meltdown in **3 days**
and HMM now detects **all four events** (the PIT curve has real vol regimes to find),
but precision degrades across the board — the PIT calm windows are genuinely noisier
(calm 2004-06: 10.3% vol, −14.5% maxDD), so detectors co-fire on non-events and the
aggregate loses the pristine precision it showed on the sanitized snapshot curve
(1.00 → 0.09). **This is the honest baseline**: classical monitoring on realistic data
is hard, which is precisely the premise H1 tests the agentic layer against. Detector
calibration (open item 1) is now clearly load-bearing.

### AL PCA snapshot → PIT: what the honest universe changed

| window | snapshot | PIT |
|---|---|---|
| baseline 2007-2015 | +41% cum, 5.2% vol, Sharpe 0.86 | +98% cum, 17.7% vol, Sharpe 0.57 |
| calm 2004-2006 | +11.4% | −3.0% |
| calm 2013-2014 | +1.9% | +3.5% |
| **Aug-2007 meltdown window** | **+3.4% (invisible!)** | **−4.8%, worst day −4.7%** |

The headline finding: **the Aug-2007 quant meltdown — the VRI gating test — was
invisible on the 2020-snapshot universe** (survivors sailed through: +3.4%) and only
appears on the PIT universe (−4.8% with the classic sharp single-day loss, cf.
Khandani & Lo 2007). The Week-1 gating test genuinely passes only under PIT data.
Curve correlation snapshot-vs-PIT: 0.21 (2007-15), 0.26 (2004-06), 0.74 (2013-14) —
survivorship distortion is largest exactly where the events are.

PIT sleeve construction: `Stat Arb/statsArb-dev/scripts/build_pit_sleeves.py`
(514 historical members with price data in 2003-2016; 491 sector-classified via
yfinance, 23 unclassified — audited in `data/pit/sector_map.csv`; sleeves masked to
membership intervals; XLC/XLRE skipped pre-inception exactly as in the snapshot
baseline). Run with `run_full_universe.py --pit`.

## First results — JT momentum (real data, all 6 windows, point-in-time universe)

| detector | recall | precision | mean latency (d) | FPR/day (calm) |
|---|---|---|---|---|
| page_hinkley | 0.00 | — | — | 0.0000 |
| bocpd | 0.75 | 0.12 | 9.0 | 0.0151 |
| hmm | 0.25 | 0.17 | 11.0 | 0.0008 |
| distributional | 0.25 | 0.25 | 10.0 | 0.0008 |
| **aggregate** | **0.25** | **0.25** | **11.0** | **0.0016** |

JT curve generated by `python XSectional/run_daily_pnl.py` (6,246 trading days, 2001-2025)
on the **point-in-time universe** (see below).

## Point-in-time universe (survivorship-bias control) — JT wired in

`XSectional/universe.py` loads `sp500-master/sp500_ticker_start_end.csv` (membership
intervals incl. delisted names) and masks the momentum score matrix so that at each
monthly rebalance only that date's actual index members are candidates.
`data.load_prices_pit()` downloads the union of historical members (1,081 tickers;
697 have yfinance data, 384 unavailable — recorded in `data/pit_download_report.csv`).

Effects (PIT vs old survivorship curve — kept at
`results/equity_curve_daily_survivorship.csv` for comparison):
- Full-period CAGR −5.65% → **−6.35%**; max drawdown −85% → −88%; daily corr 0.938.
- The 2009 momentum crash **deepens** (−44% → −53% in the event window) — the event
  signature the monitors must catch is sharper, not softer, under the honest universe.
- Candidate pool: ~542/month (biased) → ~347/month (true members with data).
- **Residual bias is measured, not silent**: member coverage rises from ~49% (2001) to
  ~98% (2025) (`results/pit_coverage.csv`) — early-sample dead names without vendor
  price data remain missing; flag as a data limitation in the write-up.

Delisted-name handling (documented): membership ends on the removal date; a held name
whose prices stop mid-hold books 0 return thereafter (implicit exit at last traded
price) — slightly optimistic for shorts of names that collapsed further off-exchange.

**Cross-strategy read (early H3 signal):** the detector that shines differs by strategy —
Page-Hinkley flags AL PCA's Aug-2007 stat-arb meltdown, while BOCPD/HMM dominate on JT
momentum (BOCPD catches 3/4 events incl. the 2009 momentum crash). This is exactly the
kind of strategy-dependent behaviour H3 predicts, though calibration confounds remain.

The agentic scaffold runs end-to-end offline: `run_classical.py` produces
`monitoring/results/agent_log.jsonl` with **8** schema-valid assessments (4 event windows ×
2 strategies), each built only from information available as of the decision date.

## Cross-strategy comparison (both PIT, back-to-back)

Strategy level (2007-2015): AL PCA +98.5% cum / Sharpe +0.57 / maxDD −20%;
JT momentum −65.7% cum / Sharpe −0.45 / maxDD −80.5%. Daily correlation **+0.003** —
the two books are statistically independent.

Event-window signatures are near-perfect mirror images (the H3 design working):

| event window | AL PCA | JT momentum |
|---|---|---|
| quant meltdown 07 | **−4.8%** (worst day −4.7%) | +4.0% |
| GFC / Lehman 08 | +11.1% | −1.4% (worst day −8.4%) |
| momentum crash 09 | −2.3% | **−53.3%** (worst day −11.7%) |
| US downgrade 11 | +33.9% | −4.5% |

Each strategy breaks on exactly one of the two gating events and shrugs off the other's.
Regime contrast for detectors: calm-window vol → worst-event vol is 4.0%→66% (16.5×)
for AL and 8.9%→68% (7.7×) for JT — large, detectable signatures in both books.

## Open items / next
1. **Per-strategy detector calibration**: several detectors have low precision (BOCPD both
   strategies; distributional on JT) and PH/HMM miss events on one strategy each. Tune
   per-strategy: `delta`/`lambda_` (PH), `hazard`/`min_run` (BOCPD), stressed-state
   `threshold` (HMM). Consider fitting HMM on an explicit historical training span.
   **Caution — no in-sample tuning:** parameters must NOT be optimised against the same
   six windows used for the Week-5 evaluation (that would inflate the classical baseline
   and bias H1). Acceptable protocols: fix defaults a priori from synthetic calibration
   (current approach), tune on non-evaluation years, or pre-register the grid and report
   sensitivity. Decide + document with supervisor before Week 5.
2. ~~Point-in-time universe — AL PCA side~~ **Done**: PIT sleeves built and all three
   AL windows regenerated (`*_pit` tags); monitoring auto-prefers them. Residuals: 23
   unclassified tickers; sector classification uses today's GICS labels for the whole
   history (same simplification as the snapshot baseline); truly dead names without
   yfinance data remain absent (measured in the JT coverage report).
3. **Local LLM**: swap `OfflineStubModel` for the Week-1 local model via the `OllamaModel`
   client (or equivalent) — the `LocalModel` interface is ready.
4. **Confirm the 4th event window** (2011 downgrade) and event onset dates with the supervisor.

## Run it
```bash
cd monitoring && pip install -r requirements.txt
python -m pytest -q          # 56 tests
python run_classical.py      # end-to-end; writes results/classical_summary.json + agent_log.jsonl
```
