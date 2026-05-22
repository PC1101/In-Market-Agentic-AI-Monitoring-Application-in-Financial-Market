# XSectional — Cross-Sectional Momentum Model Design

**Date:** 2026-05-22  
**Module:** `XSectional/`  
**Status:** Approved  

---

## 1. Overview

`XSectional` is a cross-sectional momentum backtesting module within the In-Market Agentic AI Monitoring Application. It implements the classic Jegadeesh & Titman (1993) 12–1 momentum strategy on the S&P 500 universe, adapted from the [Fisjo/momentum-strategy-backtest](https://github.com/Fisjo/momentum-strategy-backtest) reference implementation.

The module downloads historical price data, computes momentum scores, constructs a monthly long-short portfolio, simulates performance from 2000 to 2025, and produces a performance tearsheet.

---

## 2. Strategy Definition

- **Universe:** S&P 500 constituent tickers (current composition via Wikipedia or hardcoded list)
- **Signal:** 12–1 momentum — cumulative return over the past 12 months, excluding the most recent month (to avoid short-term reversal)
- **Ranking:** All stocks ranked by momentum score each month
- **Portfolio construction:**
  - **Long:** Top 20% (quintile 5) by momentum score
  - **Short:** Bottom 20% (quintile 1) by momentum score
  - Equal weighting within each leg
- **Rebalancing:** Monthly, at end of month
- **Backtest period:** January 2000 – December 2025
- **Data source:** `yfinance` (adjusted close prices)

---

## 3. File Structure

```
XSectional/
├── config.py        # Central configuration (universe, lookback window, top/bottom %, dates)
├── data.py          # Download and cache S&P 500 adjusted close prices via yfinance
├── signals.py       # Compute 12–1 momentum scores and rank stocks cross-sectionally
├── portfolio.py     # Construct long-short portfolio weights from monthly rankings
├── backtest.py      # Month-by-month simulation loop, compute portfolio returns
├── report.py        # Generate tearsheet: equity curve, Sharpe, drawdown, annual returns
├── main.py          # Entry point — orchestrates the full pipeline
└── data/            # Local cache directory for downloaded price data (CSV)
```

---

## 4. Module Responsibilities

### `config.py`
Central configuration object. All tunable parameters live here — no magic numbers elsewhere.

```python
START_DATE = "2000-01-01"
END_DATE   = "2025-12-31"
LOOKBACK_MONTHS = 12     # Momentum lookback window
SKIP_MONTHS     = 1      # Months to skip before lookback (reversal avoidance)
TOP_QUANTILE    = 0.20   # Long leg: top 20%
BOTTOM_QUANTILE = 0.20   # Short leg: bottom 20%
REBALANCE_FREQ  = "ME"   # Month-end rebalancing (pandas offset alias)
DATA_DIR        = "data/"
```

### `data.py`
- Downloads adjusted close prices for all S&P 500 tickers using `yfinance.download()`
- Caches results to `data/sp500_prices.csv` to avoid repeated API calls
- Exposes a single function: `load_prices() -> pd.DataFrame` (dates × tickers)
- Handles missing tickers gracefully (drops tickers with insufficient history)

### `signals.py`
- Takes price DataFrame as input
- Computes monthly returns: `prices.resample("ME").last().pct_change()`
- Computes 12–1 momentum score: rolling 12-month cumulative return, shifted by 1 month
- Returns a DataFrame of momentum scores (dates × tickers)
- Exposes: `compute_momentum_scores(prices: pd.DataFrame) -> pd.DataFrame`

### `portfolio.py`
- Takes momentum scores DataFrame as input
- Each month: ranks stocks, assigns long (+1/n) to top 20%, short (−1/n) to bottom 20%, zero to all others
- Returns a DataFrame of portfolio weights (dates × tickers)
- Exposes: `construct_portfolio(scores: pd.DataFrame) -> pd.DataFrame`

### `backtest.py`
- Takes portfolio weights + price returns as inputs
- Computes portfolio return each month: `(weights × forward_returns).sum(axis=1)`
- Returns a time series of monthly portfolio returns
- Exposes: `run_backtest(weights: pd.DataFrame, returns: pd.DataFrame) -> pd.Series`

### `report.py`
- Takes monthly returns series as input
- Computes and prints: Sharpe ratio, annualised return, max drawdown, Calmar ratio
- Plots: cumulative equity curve, rolling 12-month Sharpe, annual return bar chart, drawdown chart
- Saves charts to `data/tearsheet.png`
- Exposes: `generate_report(returns: pd.Series) -> None`

### `main.py`
Orchestrates the pipeline end-to-end:
```python
prices    = load_prices()
scores    = compute_momentum_scores(prices)
weights   = construct_portfolio(scores)
returns   = run_backtest(weights, prices)
generate_report(returns)
```

---

## 5. Data Flow

```
yfinance API
    ↓
data.py  ──────────────────────────────→  data/sp500_prices.csv (cache)
    ↓
signals.py  (monthly returns + 12–1 momentum scores)
    ↓
portfolio.py  (long-short weights, equal-weighted per leg)
    ↓
backtest.py  (monthly P&L series, 2000–2025)
    ↓
report.py  (tearsheet: equity curve, Sharpe, drawdown, annual returns)
```

---

## 6. Output / Tearsheet

The report will include:

| Metric | Description |
|---|---|
| Annualised Return | Geometric mean annual return of the L/S portfolio |
| Sharpe Ratio | Annualised Sharpe (assume 0% risk-free rate for simplicity) |
| Max Drawdown | Largest peak-to-trough decline |
| Calmar Ratio | Annualised return ÷ max drawdown |

**Charts (saved to `data/tearsheet.png`):**
1. Cumulative equity curve (log scale)
2. Annual return bar chart (year by year)
3. Rolling 12-month Sharpe ratio
4. Drawdown chart

---

## 7. Error Handling

- **Missing tickers:** Tickers with >20% missing data over the backtest period are dropped silently
- **Insufficient history:** Any ticker without at least 13 months of history at ranking time is excluded from that month's ranking
- **API failures:** If `yfinance` download fails, the module falls back to the cached CSV; raises `FileNotFoundError` if no cache exists
- **Empty portfolio:** If fewer than 10 stocks qualify for a leg in any month, that month's return is recorded as 0 (logged as a warning)

---

## 8. Dependencies

```
yfinance>=0.2.0
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
```

Install via: `pip install yfinance pandas numpy matplotlib`

---

## 9. Known Limitations

- **Survivorship bias:** `yfinance` only returns currently listed tickers; delisted stocks are excluded, which inflates historical returns
- **No transaction costs:** Bid-ask spread, commissions, and market impact are not modelled
- **Static universe:** Uses current S&P 500 composition, not historical point-in-time membership
- **No slippage:** Assumes execution at exact month-end close prices

These are acceptable for an initial research prototype. A production version would address all four.

---

## 10. Reference

- Jegadeesh, N. & Titman, S. (1993). *Returns to Buying Winners and Selling Losers: Implications for Stock Market Efficiency.* Journal of Finance.
- [Fisjo/momentum-strategy-backtest](https://github.com/Fisjo/momentum-strategy-backtest)
