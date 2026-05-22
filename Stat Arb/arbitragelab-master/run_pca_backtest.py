"""
Run an Avellaneda-Lee (PCA approach) stat-arb backtest using arbitragelab's
PCAStrategy WITHOUT installing the full arbitragelab package, and produce a
detailed performance report (yearly PnL, monthly returns, daily-PnL stats, risk
metrics) plus a multi-panel chart.

Why the direct load: importing arbitragelab the normal way runs
arbitragelab/__init__.py, which imports the ENTIRE library (codependence -> POT,
copula -> cvxpy, jupyter-dash, ...) and pins pandas==2.0.0 / numpy==1.23.5.
pca_approach.py itself only needs numpy + pandas + scikit-learn, so we load that
single file directly and skip all of that.

Pipeline:
  1. PCAStrategy.get_signals() -> target dollar weights per asset per day
     (arbitragelab provides this; it has no built-in backtester).
  2. Portfolio return series:
        port_ret(t+1) = sum_i w_i(t) * r_i(t+1) / sum_i |w_i(t)|
     i.e. next-day P&L normalized by gross exposure (the book is ~dollar-neutral),
     so the Sharpe is per-dollar-deployed.
  3. Report + save: equity_curve.csv, target_weights.csv, yearly_pnl.csv,
     monthly_returns.csv, performance.png.

Examples:
    python run_pca_backtest.py
    python run_pca_backtest.py --k 12 --residual-window 60 --corr-window 252
    python run_pca_backtest.py --prices path/to/your_prices.csv --show
"""
import argparse
import importlib.util
import os
import sys

import matplotlib
if "--show" not in sys.argv:           # pick backend before importing pyplot
    matplotlib.use("Agg")
import matplotlib.pyplot as plt        # noqa: E402
import numpy as np                     # noqa: E402
import pandas as pd                    # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PRICES = os.path.join(HERE, "tests", "test_data", "stock_prices.csv")
TRADING_DAYS = 252


def load_pca_strategy():
    """Load PCAStrategy straight from its source file, bypassing the package init."""
    path = os.path.join(HERE, "arbitragelab", "other_approaches", "pca_approach.py")
    spec = importlib.util.spec_from_file_location("al_pca_approach", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.PCAStrategy


def backtest(target_weights, returns):
    """Convert target dollar weights into a normalized portfolio return series."""
    w = target_weights
    ret = returns[w.columns]
    ret_next = ret.shift(-1).reindex(w.index)               # weights at t earn r(t+1)
    gross = w.abs().sum(axis=1).replace(0, np.nan)          # gross dollar exposure
    port_ret = ((w * ret_next).sum(axis=1) / gross).dropna()

    equity = (1 + port_ret).cumprod()
    drawdown = equity / equity.cummax() - 1
    return pd.DataFrame({"port_ret": port_ret, "equity": equity, "drawdown": drawdown})


# --------------------------------------------------------------------------- #
#  Statistics
# --------------------------------------------------------------------------- #
def headline_stats(perf):
    r, eq = perf["port_ret"], perf["equity"]
    n = len(r)
    years = n / TRADING_DAYS
    total = eq.iloc[-1] - 1
    cagr = eq.iloc[-1] ** (1 / years) - 1 if years > 0 else np.nan
    ann_vol = r.std() * np.sqrt(TRADING_DAYS)
    sharpe = np.sqrt(TRADING_DAYS) * r.mean() / r.std() if r.std() else np.nan
    downside = r[r < 0].std() * np.sqrt(TRADING_DAYS)
    sortino = (r.mean() * TRADING_DAYS) / downside if downside else np.nan
    maxdd = perf["drawdown"].min()
    calmar = cagr / abs(maxdd) if maxdd else np.nan
    return {
        "Trading days": n, "Years": round(years, 2),
        "Total return": total, "CAGR": cagr,
        "Ann. volatility": ann_vol, "Sharpe": sharpe, "Sortino": sortino,
        "Max drawdown": maxdd, "Calmar": calmar,
        "Daily mean": r.mean(), "Daily std": r.std(),
        "Best day": r.max(), "Worst day": r.min(),
        "% positive days": (r > 0).mean(), "Skew": r.skew(), "Kurtosis": r.kurtosis(),
    }


def yearly_table(perf):
    r = perf["port_ret"]
    g = r.groupby(r.index.year)
    tbl = pd.DataFrame({
        "days": g.size(),
        "return": g.apply(lambda x: (1 + x).prod() - 1),
        "sharpe": g.apply(lambda x: np.sqrt(TRADING_DAYS) * x.mean() / x.std() if x.std() else np.nan),
        "vol": g.apply(lambda x: x.std() * np.sqrt(TRADING_DAYS)),
        "best_day": g.max(),
        "worst_day": g.min(),
        "pos_%": g.apply(lambda x: (x > 0).mean()),
    })
    tbl["max_dd"] = perf["drawdown"].groupby(perf.index.year).min()
    tbl["end_equity"] = perf["equity"].groupby(perf.index.year).last()
    return tbl


def monthly_matrix(perf):
    r = perf["port_ret"]
    m = r.groupby([r.index.year, r.index.month]).apply(lambda x: (1 + x).prod() - 1)
    return m.unstack()  # rows = year, cols = month


def print_report(stats, ytbl, mmat):
    print("\n=== Headline performance ===")
    for k, v in stats.items():
        if k in ("Trading days", "Years"):
            print(f"{k:<18}: {v}")
        elif k in ("Sharpe", "Sortino", "Calmar", "Skew", "Kurtosis"):
            print(f"{k:<18}: {v:.3f}")
        else:
            print(f"{k:<18}: {v:.2%}")

    print("\n=== Yearly PnL ===")
    disp = ytbl.copy()
    for c in ("return", "vol", "best_day", "worst_day", "pos_%", "max_dd"):
        disp[c] = (disp[c] * 100).round(2)
    disp["sharpe"] = disp["sharpe"].round(2)
    disp["end_equity"] = disp["end_equity"].round(3)
    disp = disp.rename(columns={"return": "ret_%", "vol": "vol_%", "best_day": "best_%",
                                "worst_day": "worst_%", "pos_%": "pos_d_%", "max_dd": "maxdd_%"})
    print(disp.to_string())

    print("\n=== Monthly returns (%) ===")
    mm = (mmat * 100).round(2)
    mm.columns = [pd.Timestamp(2000, c, 1).strftime("%b") for c in mm.columns]
    mm["YEAR"] = ((mmat + 1).prod(axis=1) - 1) * 100
    mm["YEAR"] = mm["YEAR"].round(2)
    print(mm.to_string())


# --------------------------------------------------------------------------- #
#  Plots
# --------------------------------------------------------------------------- #
def make_plots(perf, ytbl, mmat, stats, save_path, show):
    fig, axs = plt.subplots(3, 2, figsize=(15, 13))

    # Equity curve
    perf["equity"].plot(ax=axs[0, 0])
    axs[0, 0].axhline(1.0, lw=0.6, ls="--", color="k")
    axs[0, 0].set_title("Equity curve (per gross $)")
    axs[0, 0].text(0.02, 0.04,
                   f"Sharpe {stats['Sharpe']:.2f}\nCAGR {stats['CAGR']:.1%}\n"
                   f"MaxDD {stats['Max drawdown']:.1%}",
                   transform=axs[0, 0].transAxes, va="bottom",
                   bbox=dict(boxstyle="round", fc="white", ec="0.7"))

    # Drawdown
    perf["drawdown"].plot(ax=axs[0, 1], color="firebrick")
    axs[0, 1].fill_between(perf.index, perf["drawdown"], 0, color="firebrick", alpha=0.3)
    axs[0, 1].set_title("Drawdown")

    # Yearly returns bar
    yr = ytbl["return"] * 100
    axs[1, 0].bar(yr.index.astype(str), yr.values,
                  color=["seagreen" if v >= 0 else "firebrick" for v in yr.values])
    axs[1, 0].axhline(0, lw=0.6, color="k")
    axs[1, 0].set_title("Yearly return (%)")
    axs[1, 0].tick_params(axis="x", rotation=45)

    # Daily return distribution
    axs[1, 1].hist(perf["port_ret"] * 100, bins=80, color="steelblue")
    axs[1, 1].axvline(0, lw=0.8, ls="--", color="k")
    axs[1, 1].set_title("Daily return distribution (%)")

    # Rolling 126-day annualized Sharpe
    r = perf["port_ret"]
    roll = np.sqrt(TRADING_DAYS) * r.rolling(126).mean() / r.rolling(126).std()
    roll.plot(ax=axs[2, 0], color="darkorange")
    axs[2, 0].axhline(0, lw=0.6, color="k")
    axs[2, 0].set_title("Rolling 126-day annualized Sharpe")

    # Monthly returns heatmap
    ax = axs[2, 1]
    data = mmat.values * 100
    im = ax.imshow(data, aspect="auto", cmap="RdYlGn",
                   vmin=-np.nanmax(np.abs(data)), vmax=np.nanmax(np.abs(data)))
    ax.set_xticks(range(mmat.shape[1]))
    ax.set_xticklabels([pd.Timestamp(2000, c, 1).strftime("%b") for c in mmat.columns])
    ax.set_yticks(range(mmat.shape[0]))
    ax.set_yticklabels(mmat.index)
    ax.set_title("Monthly returns heatmap (%)")
    for i in range(mmat.shape[0]):
        for j in range(mmat.shape[1]):
            if not np.isnan(data[i, j]):
                ax.text(j, i, f"{data[i, j]:.1f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("PCA (Avellaneda-Lee) stat-arb performance", fontsize=14)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=110)
    if show:
        plt.show()
    else:
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prices", default=DEFAULT_PRICES,
                        help="CSV of prices (Date index + one column per asset). "
                             "Default: bundled tests/test_data/stock_prices.csv.")
    parser.add_argument("--n-components", type=int, default=15, help="PCA factors (default 15).")
    parser.add_argument("--k", type=float, default=8.4,
                        help="Min mean-reversion speed to trade (paper: 252/30=8.4).")
    parser.add_argument("--corr-window", type=int, default=252, help="PCA corr look-back (252).")
    parser.add_argument("--residual-window", type=int, default=60, help="Residual look-back (60).")
    parser.add_argument("--sbo", type=float, default=1.25, help="Long entry band (1.25).")
    parser.add_argument("--sso", type=float, default=1.25, help="Short entry band (1.25).")
    parser.add_argument("--ssc", type=float, default=0.5, help="Long close band (0.5).")
    parser.add_argument("--sbc", type=float, default=0.75, help="Short close band (0.75).")
    parser.add_argument("--show", action="store_true", help="Display the chart interactively.")
    args = parser.parse_args()

    PCAStrategy = load_pca_strategy()

    prices = pd.read_csv(args.prices, parse_dates=True, index_col="Date").sort_index()
    returns = prices.pct_change().iloc[1:]
    print(f"data: {os.path.relpath(args.prices, HERE)}  "
          f"{returns.shape[0]} days x {returns.shape[1]} assets  "
          f"({returns.index.min().date()} -> {returns.index.max().date()})")
    print(f"params: n_components={args.n_components} k={args.k} "
          f"corr_window={args.corr_window} residual_window={args.residual_window} "
          f"sbo={args.sbo} sso={args.sso} ssc={args.ssc} sbc={args.sbc}")

    strategy = PCAStrategy(n_components=args.n_components)
    print("Generating target weights with PCAStrategy.get_signals() "
          "(no progress bar in the library; this takes a moment)...")
    target_weights = strategy.get_signals(
        returns, k=args.k, corr_window=args.corr_window,
        residual_window=args.residual_window,
        sbo=args.sbo, sso=args.sso, ssc=args.ssc, sbc=args.sbc, size=1,
    )

    perf = backtest(target_weights, returns)
    stats = headline_stats(perf)
    ytbl = yearly_table(perf)
    mmat = monthly_matrix(perf)
    print_report(stats, ytbl, mmat)

    out_dir = os.path.join(HERE, "results")
    os.makedirs(out_dir, exist_ok=True)
    perf.to_csv(os.path.join(out_dir, "equity_curve.csv"))
    target_weights.to_csv(os.path.join(out_dir, "target_weights.csv"))
    ytbl.to_csv(os.path.join(out_dir, "yearly_pnl.csv"))
    mmat.to_csv(os.path.join(out_dir, "monthly_returns.csv"))
    make_plots(perf, ytbl, mmat, stats, os.path.join(out_dir, "performance.png"), args.show)
    print(f"\nSaved equity_curve.csv, target_weights.csv, yearly_pnl.csv, "
          f"monthly_returns.csv, performance.png to: {os.path.relpath(out_dir, HERE)}/")


if __name__ == "__main__":
    main()
