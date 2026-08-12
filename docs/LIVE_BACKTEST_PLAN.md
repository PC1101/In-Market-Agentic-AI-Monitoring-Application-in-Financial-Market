# Live Trading Backtest Plan: Long-Only Strategy + Agentic Risk Overlay ($1M, vs S&P 500)

Status: PLANNED (not yet implemented)
Scope now: JT momentum full period 2001–2025. Later: AL stat arb via the same module (see plug-in recipe at the bottom).

## Context

We have precomputed: (a) JT momentum daily PnL 2001–2025, (b) agentic monitoring logs (daily
`performance_supervisor` decisions) for 12 evaluation windows × JT_MOM, (c) evidence the agent
detects events earlier than classical detectors. Goal: simulate a live trader starting with
**$1,000,000** running **long-only JT momentum** on the S&P 500 universe with the **agentic layer
as a risk overlay**, compared against S&P 500 buy-and-hold. Deliverables: portfolio-value graph,
PnL/drawdown/exposure graph, metrics table.

The module must be **strategy-agnostic** so that once AL stat arb is re-run (long-only, through
2025) with its agentic logs, the same CLI reproduces the full-period AL backtest.

## Verified facts (do not re-derive)

- JT curve `XSectional/results/equity_curve_daily.csv` (2001-03-01 → 2025-12-30) is **LONG-SHORT**
  → a long-only rerun is required (fast; prices cached at `XSectional/data/sp500_prices_pit.csv`).
- Agentic JSONLs: 12 JT_MOM files at `monitoring/results/agentic_<window>_JT_MOM.jsonl`,
  ~1,097 daily `performance_supervisor` records. Vocabulary (from `monitoring/agentic/schemas.py`,
  `monitoring/agentic/model.py:109`): states `NORMAL < WATCH < ALERT < CRITICAL`; actions
  `HOLD < INVESTIGATE < REDUCE < HALT`. JT logs contain only HOLD/INVESTIGATE/REDUCE and
  NORMAL/WATCH/ALERT (never HALT/CRITICAL). WATCH/INVESTIGATE dominate even in calm windows →
  **policy must key on ALERT+ only; never de-risk on bare INVESTIGATE** or calm years get destroyed.
- No SPY/^GSPC data in repo; yfinance 1.4.0 + matplotlib 3.7.5 installed (system Python 3.11,
  no venv). Monitoring tests: `python -m pytest -q` from `monitoring/` (pytest.ini testpaths=tests).
- `XSectional/run_daily_pnl.py` currently has only a `--survivorship` flag.
- JT long weights are equal-weight 1/n_long, so a long-only filter already sums to 1.0 per row
  (renormalize defensively anyway).
- Existing overlay prototype: `monitoring/scripts/pnl_comparison.py` (flat-on-ALERT, 5-day
  cooldown, per-window). Reuse its JSONL parsing approach and figure palette.

## Phase 1 — JT long-only curve (XSectional)

1. `XSectional/portfolio.py` — add:
   ```python
   def to_long_only(weights: pd.DataFrame) -> pd.DataFrame:
       """Keep positive weights only; renormalize each row to sum to 1.0."""
   ```
   Mask non-positive weights to NA, divide each row by its row-sum.
2. `XSectional/run_daily_pnl.py` — add `--long-only` flag: apply `to_long_only` after
   `construct_portfolio(scores)`, write `results/equity_curve_daily_long_only.csv`.
   Reuse `run_backtest_daily` / `write_daily_equity_curve` untouched.

## Phase 2 — New package `monitoring/live_backtest/`

Files: `__init__.py`, `policy.py`, `engine.py`, `benchmark.py`, `report.py`, `README.md`;
CLI at `monitoring/run_live_backtest.py` (repo convention: runners at monitoring/ root, which
guarantees clean imports of `pnl_loader` and `windows`).

### policy.py — agentic decisions → daily exposure in [0,1]

Graded ladder grounded in published risk-overlay practice (drawdown-control indices, conditional
volatility targeting — QuantPedia "An Introduction to Volatility Targeting"; Bollerslev et al.
"Conditional Volatility Targeting" FAJ 2020; CSSA "Building a Risk Control Index with Drawdown
Protection"; Man Group "The Impact of Volatility Targeting"). Cite in module docstring.

```python
@dataclass(frozen=True)
class PolicyConfig:
    exposure_normal: float = 1.0        # NORMAL/WATCH, or any day outside log coverage
    exposure_alert: float = 0.50        # ALERT with action below REDUCE
    exposure_alert_reduce: float = 0.25 # ALERT + REDUCE
    exposure_halt: float = 0.0          # CRITICAL state or HALT action (unused for JT; kept for AL)
    cooldown_days: int = 5              # hold min exposure N trading days after trigger
    reentry_consecutive: int = 3        # consecutive non-ALERT days required per re-entry step
    reentry_steps: tuple = (0.5, 1.0)   # staged ramp back up
    lag_days: int = 1                   # decision at close t -> exposure from t+1 (causal)

def load_decisions(log_paths: list[Path]) -> pd.DataFrame:
    """Parse performance_supervisor records from JSONLs -> DataFrame indexed by date with
    columns state, action, confidence, window. Parsing lifted from
    scripts/pnl_comparison.py (agent=='performance_supervisor', tolerant of malformed lines).
    On duplicate dates keep the most severe state."""

def target_exposure(decisions: pd.DataFrame, cfg: PolicyConfig) -> pd.Series:
    """Raw per-day ladder mapping (no smoothing)."""

def build_exposure(trading_days: pd.DatetimeIndex, decisions: pd.DataFrame,
                   cfg: PolicyConfig) -> pd.Series:
    """Full-period exposure: ladder + cooldown + staged re-entry; 1.0 on days with no
    decision record (hybrid coverage); then .shift(lag_days).fillna(1.0)."""
```
Severity constants `STATE_RANK`/`ACTION_RANK` follow `model.py:109` ordering.

### engine.py — $1M portfolio simulation

```python
@dataclass(frozen=True)
class EngineConfig:
    capital: float = 1_000_000.0
    tcost_bps: float = 10.0        # on |Δexposure| × notional (overlay trades)
    cash_rate_annual: float = 0.0  # optional T-bill on uninvested sleeve

def simulate(strategy_returns: pd.Series, exposure: pd.Series, cfg: EngineConfig) -> pd.DataFrame:
    """Columns: value, pnl, exposure, strat_ret, port_ret, tcost, drawdown.
    Day t: gross  = e_t*r_t + (1-e_t)*cash_daily
           tcost_t = |e_t - e_{t-1}| * V_{t-1} * tcost_bps/1e4
           V_t     = V_{t-1}*(1+gross) - tcost_t"""
```
Iterative loop (not vectorized) so tcost compounds correctly; ~6,250 rows is trivial.

### benchmark.py

```python
def fetch_benchmark(ticker="SPY", start=..., end=..., cache_dir=MONITORING/"data") -> pd.Series  # daily returns
def buy_and_hold(returns: pd.Series, capital: float) -> pd.Series                                # value series
```
yfinance `auto_adjust=True` (SPY adjusted close ≈ total return incl. dividends); cache to
`monitoring/data/benchmark_<TICKER>.csv` (schema `Date,ret`); load cache without network if
present. `^GSPC` supported but flagged price-only. Align to strategy calendar via reindex-intersection.

### report.py

```python
def compute_metrics(value: pd.Series, rf=0.0) -> dict   # CAGR, ann_vol, sharpe, max_dd, calmar, terminal_value, total_return
def write_metrics_table(results: dict[str, pd.Series], outdir) -> Path  # metrics.json + metrics_table.png; alpha-vs-benchmark rows
def fig_portfolio_value(...)   # fig_a_portfolio_value.png
def fig_pnl_drawdown(...)      # fig_b_pnl_drawdown_exposure.png
```
Figure specs:
- **(a)** three $ curves — "Strategy (unmanaged)" gray `#888888`, "Strategy + Agent" red
  `#E8553A`, "S&P 500 B&H" blue `#2176AE` (palette from `pnl_comparison.py`); log-y (25-year
  span); shaded spans for all 12 windows from `windows` module (event=light red, calm=light
  gray); onset dates as dashed vlines.
- **(b)** 3 stacked panels sharing x: cumulative PnL ($), drawdown (%), exposure step-plot
  (fill_between 0→e), same shading.
Outputs → `monitoring/results/live_backtest/<STRATEGY>/` plus `exposure.csv`,
`portfolio_daily.csv` for audit.

### CLI — `monitoring/run_live_backtest.py`

```
python run_live_backtest.py \
  --curve ../XSectional/results/equity_curve_daily_long_only.csv \
  --strategy JT_MOM \
  --logs-glob "results/agentic_*_JT_MOM.jsonl" \
  --capital 1000000 --tcost-bps 10 --benchmark SPY \
  [--start 2001-03-01 --end 2025-12-30] [--cooldown 5] [--lag 1] \
  [--outdir results/live_backtest/JT_MOM] [--no-fetch]
```
Flow: `pnl_loader.load_pnl`/`returns_series` (reuse causal loader) → `policy.load_decisions` +
`build_exposure` → `engine.simulate` (managed) AND `simulate` with exposure≡1 (unmanaged) →
benchmark → report.

Add `yfinance>=0.2.54` and `matplotlib>=3.7.0` to `monitoring/requirements.txt`.

## Phase 3 — Tests (`monitoring/tests/`, no network, tmp_path fixtures, deterministic)

- `test_live_backtest_policy.py`: ladder mapping for each (state, action) combo incl. unknown
  action → state fallback; days without records → 1.0; cooldown pins min exposure N days;
  staged re-entry 0.25→0.5→1.0 requires `reentry_consecutive` clean days per step; `lag_days=1`
  shifts exposure; duplicate-date records keep most severe; JSONL parser skips malformed lines
  and non-supervisor agents (synthetic JSONL via tmp_path).
- `test_live_backtest_engine.py`: exposure≡1 & tcost=0 → value equals `capital*cumprod(1+r)`;
  exposure≡0 → flat; hand-computed 3-day tcost example; `pnl == value.diff()`; cash accrual.
- `test_live_backtest_report.py`: closed-form CAGR/Sharpe/maxDD on synthetic series.
- `test_live_backtest_benchmark.py`: buy_and_hold math from fixture CSV; cache-load path (no yfinance call).

## Phase 4 — Verification

```
cd XSectional && python run_daily_pnl.py --long-only
cd ../monitoring && python -m pytest -q tests/test_live_backtest_policy.py tests/test_live_backtest_engine.py tests/test_live_backtest_report.py tests/test_live_backtest_benchmark.py
python run_live_backtest.py --curve ../XSectional/results/equity_curve_daily_long_only.csv --strategy JT_MOM --logs-glob "results/agentic_*_JT_MOM.jsonl"
python -m pytest -q   # full suite regression
```
Sanity checks: exposure dips only inside windows; terminal values plausible; figures render.

## AL stat arb plug-in recipe (also goes in live_backtest/README.md)

User must later produce:
1. **Long-only full-period AL equity curve** CSV in the locked schema
   `Date,port_ret,equity,drawdown` — re-run the stat arb full-universe backtest through 2025
   with `bt.run(long_only=True)` (flag already exists in `Stat Arb/statsArb-dev/src/backtest.py`).
2. **Agentic JSONLs** named `monitoring/results/agentic_<window>_AL_PCA.jsonl` with
   `performance_supervisor` records — re-run `monitoring/run_agentic.py` on the new curve for
   all 12 windows (the 7 existing AL files cover only ≤2015 windows and monitored the old curve).

Then:
```
python run_live_backtest.py --curve "<AL long-only curve.csv>" --strategy AL_PCA --logs-glob "results/agentic_*_AL_PCA.jsonl"
```
v2 extension (out of scope now): `--combine JT_MOM,AL_PCA --weights 0.5,0.5` averaging the two
managed daily return series before `engine.simulate`.

## Risks / limitations (disclose in README + report footer)

1. **Hybrid coverage bias**: agentic decisions exist only inside 12 ex-post-selected windows
   (~1,097 of ~6,250 trading days); assuming 100% exposure elsewhere means the overlay only acts
   where we chose to look. Favorable selection bias — must be disclosed.
2. **Curve mismatch**: JT JSONLs were generated monitoring the LONG-SHORT curve
   (`run_meta.curve` confirms); we apply them to the long-only curve. Defensible (regime breaks
   are market-wide) but an approximation — ideally re-run `run_agentic.py` on the long-only
   curve later (see "Accuracy upgrade path" below).
3. **False-positive drag is real**: calm_2012 has 40 ALERT/REDUCE days, calm_2013_2014 has 31 —
   the overlay costs return there. Honest cost; report it.
4. The 0.0-exposure rung never fires for JT (no HALT/CRITICAL in logs) — keep for generality/AL.
5. **Post-hoc cost application**: net curves apply era-dependent transaction costs
   (commission + half-spread) post-hoc to gross strategy returns. Agent decisions were generated
   on gross curves; verification shows max feature divergence <0.1% vol and <2.7pp drawdown
   (see `results/gross_net_feature_divergence.csv`). Slippage/market impact excluded
   ($1M << S&P 500 ADV; folded into "high" cost scenario).
6. ^GSPC is price-only (understates benchmark ~2%/yr); default SPY mitigates.
7. Outperformance vs S&P is an empirical outcome, not guaranteed by construction.
8. **Cash interest**: de-risked capital earns 1.4% flat (avg 3M T-bill 2004–2025);
   actual rates varied 0–5%. Conservative but consistent.

## Realistic frictions (added Week 4)

Era-dependent per-side cost schedule (bps of notional traded):

| Era | Low | Base | High |
|---|---|---|---|
| 2004–2009 | 4 | 8 | 15 |
| 2010–2018 | 2 | 5 | 10 |
| 2019–2025 | 1 | 2.5 | 5 |

Applied to actual turnover (JT: monthly rebalance vs drifted weights, ~52%/rebalance,
~34 bps/yr base drag; AL: OU signal entry/exit from trades blotter, ~219 bps/yr base drag).

Results in `results/live_backtest/{STRAT}_{gross,net,net_low,net_high}/`.
Figures in `results/figures/{summary_gross_net,metrics_gross_net_*,fig_c_*,fig_d_*,sensitivity_*}.png`.
Module: `monitoring/frictions.py`.

## Accuracy upgrade path (staged; each step removes one bias)

- **Stage 0 (this plan)**: overlay on existing logs. Fast; validates all plumbing, figures,
  engine. Known biases #1 and #2 above.
- **Stage 1**: after the long-only curve exists, re-run `run_agentic.py` for the 12 JT windows
  on the long-only curve (medium compute: 12 windows × Ollama two-pass) → removes bias #2.
  Re-run the CLI unchanged (same logs-glob).
- **Stage 2 (gold standard)**: run the agent over every trading day 2001–2025 (~6,250 days;
  days of Ollama compute) → removes bias #1. The module needs no changes: `build_exposure`
  already consumes whatever coverage the logs provide.

## Critical files

- Modify: `XSectional/portfolio.py`, `XSectional/run_daily_pnl.py`, `monitoring/requirements.txt`
- Create: `monitoring/live_backtest/{__init__,policy,engine,benchmark,report}.py`,
  `monitoring/live_backtest/README.md`, `monitoring/run_live_backtest.py`,
  `monitoring/tests/test_live_backtest_{policy,engine,report,benchmark}.py`
- Reuse: `monitoring/pnl_loader.py`, `monitoring/windows.py`,
  `monitoring/scripts/pnl_comparison.py` (JSONL parsing + palette)
