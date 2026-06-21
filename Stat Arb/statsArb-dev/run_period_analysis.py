"""
Baseline stat-arb performance across 4 distinct market regimes, overlaid
with S&P 500 (SPY) buy-and-hold.

Periods
-------
  GFC         2007-01-03  ->  2009-03-09   (pre-crisis through market trough)
  Recovery    2009-03-10  ->  2019-01-02   (post-GFC bull market / calm window)
  COVID       2019-01-03  ->  2020-11-20   (late cycle + COVID crash + recovery)
  Full        2007-01-03  ->  2020-11-20   (entire sample)

Outputs (results/period_analysis/)
-------
  period_analysis.png   - 2x2 chart: each period with strategy vs S&P 500
  period_metrics.csv    - headline metrics table
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

from src import backtest, bt_tools  # noqa: E402

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PERIODS = [
    ("GFC\n(2007–2009)",        "2007-01-03", "2009-03-09"),
    ("Recovery\n(2009–2019)",   "2009-03-10", "2019-01-02"),
    ("COVID\n(2019–2020)",      "2019-01-03", "2020-11-20"),
    ("Full Period\n(2007–2020)","2007-01-03", "2020-11-20"),
]

ETF        = "xlf"
DEFACTORING = "etf"
N_WINDOW   = 60
KAPPA_MIN  = 252.0 / 30.0   # ~8.4 / yr

with open("configs/optimise_trading_rules.yml") as fh:
    CFG = yaml.load(fh, Loader=yaml.SafeLoader)

PRICES_PATH = CFG["prices_file_path"][ETF]
COST        = tuple(CFG.get("transaction_cost", (0.0005, 0.0005)))
SL          = CFG.get("sl", -0.10)

# ---------------------------------------------------------------------------
# S&P 500 data  (try yfinance, fall back to XLF as sector proxy)
# ---------------------------------------------------------------------------

def load_spy():
    """Download SPY adjusted closes. Falls back to None on failure."""
    try:
        import yfinance as yf
        raw = yf.download("SPY", start="2006-01-01", end="2020-11-21",
                          auto_adjust=True, progress=False)
        # yfinance may return multi-level columns; flatten to a plain Series
        if isinstance(raw.columns, pd.MultiIndex):
            spy = raw[("Close", "SPY")]
        else:
            spy = raw["Close"]
        spy = spy.squeeze()                                # ensure Series
        spy.index = pd.to_datetime(spy.index).tz_localize(None)
        spy.name = "SPY"
        print("  SPY data downloaded via yfinance.")
        return spy
    except Exception as e:
        print(f"  yfinance unavailable ({e}). Using XLF as sector proxy for S&P 500.")
        return None

# ---------------------------------------------------------------------------
# Run one period
# ---------------------------------------------------------------------------

def run_period(st_dt: str, ed_dt: str):
    """Run baseline backtest for one period. Returns port_ret DataFrame."""
    model = backtest.bt(
        prices_file_path=PRICES_PATH,
        etf_name=ETF,
        st_dt=st_dt,
        ed_dt=ed_dt,
        n_window=N_WINDOW,
        defactoring=DEFACTORING,
        performance_only=True,
        kappa_min=KAPPA_MIN,
        progress=False,
    )
    sharpe, maxdd, endpnl = model.run(
        weighting_scheme=CFG.get("weighting_scheme", "equal_weighted"),
        sl=SL,
        long_only=False,
        transaction_cost=COST,
    )
    return model.port_ret, sharpe, maxdd, endpnl

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def rebase(series: pd.Series) -> pd.Series:
    """Rebase a price/equity series to 1.0 at its first value."""
    s = series.dropna()
    return s / s.iloc[0]

def spy_cum_ret(spy_prices: pd.Series, st_dt: str, ed_dt: str) -> pd.Series:
    """Slice SPY to the period and rebase to 1.0."""
    sl = spy_prices.loc[st_dt: ed_dt]
    return rebase(sl)

def etf_cum_ret(port_ret: pd.DataFrame, etf_col: str) -> pd.Series:
    """Rebase the ETF buy-and-hold column from port_ret."""
    if etf_col in port_ret.columns:
        return (port_ret[etf_col] + 1).cumprod()
    return None

def period_ann_ret(cum_pnl: pd.Series) -> float:
    n_years = len(cum_pnl) / 252
    return float(cum_pnl.iloc[-1] ** (1 / n_years) - 1) if n_years > 0 else np.nan

def spy_sharpe(ret: pd.Series) -> float:
    r = ret.squeeze().pct_change().dropna()
    std = float(r.std())
    return float(np.sqrt(252) * r.mean() / std) if std > 0 else np.nan

# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

STRATEGY_COLOR = "#1f77b4"   # blue
SPY_COLOR      = "#d62728"   # red
ETF_COLOR      = "#7f7f7f"   # grey

def plot_panel(ax, label, port_ret, spy_slice, sharpe, maxdd, endpnl):
    """Draw one 2x2 panel: strategy equity curve + SPY + metrics annotation."""

    # Strategy cumulative PnL (already rebased to 1.0 from backtest start)
    strat = port_ret["cum_pnl"].dropna()
    strat = rebase(strat)
    strat.plot(ax=ax, color=STRATEGY_COLOR, linewidth=1.5, label="Stat-arb (baseline)")

    # S&P 500
    if spy_slice is not None and len(spy_slice) > 0:
        spy_slice.plot(ax=ax, color=SPY_COLOR, linewidth=1.2,
                       linestyle="--", label="S&P 500 (SPY)")
        spy_end = float(spy_slice.iloc[-1])
        spy_sr  = spy_sharpe(spy_slice)
    else:
        spy_end = np.nan
        spy_sr  = np.nan

    # ETF buy-and-hold line (from the port_ret ETF column if present)
    etf_col = ETF.upper()
    if etf_col in port_ret.columns:
        etf_bh = rebase((port_ret[etf_col] + 1).cumprod().dropna())
        etf_bh.plot(ax=ax, color=ETF_COLOR, linewidth=0.9,
                    linestyle=":", label=f"{etf_col} B&H")

    ax.axhline(1.0, color="black", linewidth=0.5, linestyle=":")
    ax.set_title(label, fontsize=11, fontweight="bold")
    ax.set_ylabel("Equity (base = 1.0)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

    # Metrics box
    strat_end = float(strat.iloc[-1])
    txt = (
        f"  Stat-arb\n"
        f"    Sharpe : {sharpe:+.2f}\n"
        f"    MaxDD  : {maxdd:.1%}\n"
        f"    End PnL: {strat_end:.2f}x\n"
        f"\n"
        f"  S&P 500 (SPY)\n"
        f"    Sharpe : {spy_sr:+.2f}\n"
        f"    End PnL: {spy_end:.2f}x"
    )
    ax.text(0.02, 0.03, txt,
            transform=ax.transAxes,
            fontsize=7.5,
            verticalalignment="bottom",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="0.75", alpha=0.9),
            family="monospace")

    ax.legend(loc="upper left", fontsize=8)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Loading S&P 500 (SPY) data...")
    spy_prices = load_spy()

    results = []
    port_rets = []

    for i, (label, st_dt, ed_dt) in enumerate(PERIODS, 1):
        clean_label = label.replace("\n", " ")
        print(f"\n[{i}/4] Running baseline - {clean_label}  ({st_dt} to {ed_dt})")
        port_ret, sharpe, maxdd, endpnl = run_period(st_dt, ed_dt)
        print(f"       Sharpe {sharpe:+.3f}   MaxDD {maxdd:.2%}   EndPnL {endpnl:.3f}x")
        port_rets.append(port_ret)

        # SPY for this period
        spy_sl = None
        if spy_prices is not None:
            raw = spy_prices.loc[st_dt: ed_dt]
            spy_sl = rebase(raw) if len(raw) > 0 else None
            spy_end = float(spy_sl.iloc[-1]) if spy_sl is not None else np.nan
            spy_sr  = spy_sharpe(spy_prices.loc[st_dt: ed_dt])
        else:
            spy_end = np.nan
            spy_sr  = np.nan

        results.append({
            "period":      clean_label,
            "start":       st_dt,
            "end":         ed_dt,
            "strat_sharpe": round(sharpe, 3),
            "strat_maxdd":  round(maxdd, 4),
            "strat_endpnl": round(endpnl, 3),
            "spy_sharpe":   round(spy_sr, 3) if not np.isnan(spy_sr) else "",
            "spy_endpnl":   round(spy_end, 3) if not np.isnan(spy_end) else "",
        })

    # ---- Plot ----
    print("\nGenerating 2x2 chart...")
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    axs = axs.flatten()

    for i, (ax, (label, st_dt, ed_dt)) in enumerate(zip(axs, PERIODS)):
        port_ret = port_rets[i]
        _, sharpe, maxdd, endpnl = (
            None,
            results[i]["strat_sharpe"],
            results[i]["strat_maxdd"],
            results[i]["strat_endpnl"],
        )
        spy_sl = None
        if spy_prices is not None:
            raw = spy_prices.loc[st_dt: ed_dt]
            spy_sl = rebase(raw) if len(raw) > 0 else None

        plot_panel(axs[i], label, port_ret, spy_sl, sharpe, maxdd, endpnl)

    fig.suptitle(
        f"XLF Stat-arb Baseline — Performance Across Market Regimes vs S&P 500",
        fontsize=13, fontweight="bold", y=1.01,
    )
    fig.tight_layout()

    out_dir = os.path.join("results", "period_analysis")
    os.makedirs(out_dir, exist_ok=True)
    chart_path = os.path.join(out_dir, "period_analysis.png")
    fig.savefig(chart_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Chart saved: {chart_path}")

    # ---- Metrics CSV ----
    metrics_df = pd.DataFrame(results)
    csv_path = os.path.join(out_dir, "period_metrics.csv")
    metrics_df.to_csv(csv_path, index=False)
    print(f"Metrics saved: {csv_path}")

    # ---- Console summary table ----
    print("\n" + "=" * 70)
    print(f"{'Period':<26} {'Strat Sharpe':>13} {'Strat MaxDD':>12} "
          f"{'Strat End':>10} {'SPY End':>9}")
    print("-" * 70)
    for r in results:
        print(f"{r['period']:<26} {r['strat_sharpe']:>13.3f} "
              f"{r['strat_maxdd']:>11.1%}  "
              f"{r['strat_endpnl']:>9.3f}x  "
              f"{str(r['spy_endpnl']):>7}x")
    print("=" * 70)


if __name__ == "__main__":
    main()
