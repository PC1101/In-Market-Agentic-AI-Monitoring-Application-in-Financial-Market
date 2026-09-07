"""
Avellaneda-Lee stat-arb backtest over the ENERGY universe in PCA-defactoring mode.

Unlike run_full_universe.py (which trades one sector-ETF sleeve at a time and hedges
each name against its sector ETF), this runner trades the whole 34-name energy
constituent universe as ONE dollar-neutral long/short book, defactored against the
5 energy PCA factors (results/pca_factoring/ret_pca_port_energy.csv). It emits the
canonical Date,port_ret,equity,drawdown curve consumed by the monitoring layer.

Integration notes (why this file exists rather than calling bt directly):
  * The energy prices.csv (Task 2) uses BARE ticker columns (APA, BKR, ...), but
    bt.__init__ selects only columns containing 'Adj. Close'. We therefore build a
    bt-compatible derived file (data/energy_universe/prices_al.csv) that renames each
    ticker to 'TICKER Adj. Close'.
  * bt still needs the etf_name column present: ou_process pops it from the traded
    universe (so XLE is NOT traded) and uses it only for a shape assert (its VALUES
    are irrelevant in pca mode), and backtest.run() concats it as a benchmark column.
    We attach 'XLE Adj. Close' from the existing SPDR koyfin file as that benchmark.
    XLE is thus a benchmark/reference only — never part of the 34-name traded set.

Examples:
    python run_energy_al.py --start 2020-01-01 --end 2020-06-30   # fast validation
    python run_energy_al.py --start 2016-06-01 --end 2022-12-30    # full window
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import numpy as np   # noqa: E402
import pandas as pd  # noqa: E402

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

from src import backtest  # noqa: E402

RAW_PRICES = "data/energy_universe/prices.csv"
PCA_PATH = "results/pca_factoring/ret_pca_port_energy.csv"
XLE_SRC = "data/xle/xle_koyfin_20201122_071247976.csv"   # SPDR energy ETF, has 'XLE Adj. Close'
DERIVED_PRICES = "data/energy_universe/prices_al.csv"
OUT = "results/equity_curve_energy_al.csv"

ETF_NAME = "XLE"
N_WINDOW = 60
DEFAULT_KAPPA_MIN = 252.0 / 30.0


def build_bt_prices():
    """Build a bt-compatible prices file: 'TICKER Adj. Close' columns + an XLE
    benchmark column. Returns the (sorted) DatetimeIndex of the price panel."""
    raw = pd.read_csv(RAW_PRICES, index_col="Date", parse_dates=True).sort_index()
    raw.columns = [f"{c} Adj. Close" for c in raw.columns]

    xle = pd.read_csv(XLE_SRC, index_col="Date", parse_dates=True).sort_index()["XLE Adj. Close"]
    # Align the benchmark to the energy panel's trading calendar. The koyfin file
    # ends 2020-11-20; for later dates we forward-fill (the benchmark column is
    # discarded before the canonical curve is written, and its VALUES do not enter
    # the pca-mode strategy at all, so this only affects an unused reference).
    raw[f"{ETF_NAME} Adj. Close"] = xle.reindex(raw.index).ffill().bfill()

    os.makedirs(os.path.dirname(DERIVED_PRICES), exist_ok=True)
    raw.to_csv(DERIVED_PRICES)
    return raw.index


def snap_window(idx, start, end):
    """Snap requested start/end to real trading days in the price index, and make
    sure there is at least N_WINDOW days of history before the start."""
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    on_or_after = idx[idx >= start_ts]
    on_or_before = idx[idx <= end_ts]
    if on_or_after.empty or on_or_before.empty:
        raise SystemExit(f"window {start}..{end} falls outside data range "
                         f"{idx.min().date()}..{idx.max().date()}")
    st = on_or_after.min()
    ed = on_or_before.max()
    if idx.get_loc(st) < N_WINDOW:
        st = idx[N_WINDOW]
    if idx.get_loc(ed) <= idx.get_loc(st):
        raise SystemExit("window too short after snapping to trading days")
    return st, ed


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--start", default="2016-06-01")
    p.add_argument("--end", default="2022-12-30")
    p.add_argument("--kappa-min", type=float, default=DEFAULT_KAPPA_MIN)
    p.add_argument("--cost", type=float, default=0.0,
                   help="Per-side transaction cost fraction (default 0 = frictionless).")
    p.add_argument("--sl", type=float, default=-0.10, help="Stop-loss (default -0.10).")
    p.add_argument("--out", default=OUT, help="Output canonical curve path.")
    args = p.parse_args()

    idx = build_bt_prices()
    st_dt, ed_dt = snap_window(idx, args.start, args.end)
    kappa_min = args.kappa_min if args.kappa_min > 0 else None

    print(f"AL energy (pca defactoring)  window {st_dt.date()}..{ed_dt.date()}  "
          f"kappa_min={kappa_min if kappa_min is not None else 'off'}  "
          f"cost={args.cost} ({'frictionless' if args.cost == 0 else f'{args.cost*1e4:g} bps'})")

    model = backtest.bt(
        prices_file_path=DERIVED_PRICES,
        etf_name=ETF_NAME,
        st_dt=st_dt.strftime("%Y-%m-%d"),
        ed_dt=ed_dt.strftime("%Y-%m-%d"),
        n_window=N_WINDOW,
        defactoring="pca",
        performance_only=True,
        kappa_min=kappa_min,
        progress=True,
        pca_ret_path=PCA_PATH,
    )
    sharpe, maxdd, endpnl = model.run(
        weighting_scheme="equal_weighted", sl=args.sl, long_only=False,
        transaction_cost=(args.cost, args.cost),
    )

    # cum_ret is the DAILY book return (bt_tools.port_performance:
    # cum_ret = long_ret - cum_short_ret); cum_pnl is its running cumprod.
    daily = model.port_ret["cum_ret"].astype(float)
    equity = (1.0 + daily).cumprod()
    drawdown = equity / equity.cummax() - 1.0

    # Sanity: our rebuilt equity must match the engine's own cum_pnl.
    engine_pnl = model.port_ret["cum_pnl"].astype(float)
    max_dev = float((equity - engine_pnl).abs().max())
    assert max_dev < 1e-9, f"rebuilt equity diverges from engine cum_pnl (max dev {max_dev})"

    out = pd.DataFrame({"port_ret": daily, "equity": equity, "drawdown": drawdown})
    out.index.name = "Date"
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out.to_csv(args.out)

    n_nonzero = int((daily != 0).sum())
    print(f"\n=== AL energy PCA summary ===")
    print(f"Trading days   : {len(out)}  (nonzero-return days: {n_nonzero})")
    print(f"Sharpe ratio   : {sharpe:.3f}")
    print(f"Max drawdown   : {drawdown.min():.2%}")
    print(f"Final equity   : {equity.iloc[-1]:.4f}  (EndPnL {endpnl:.4f})")
    print(f"equity==cum_pnl: max deviation {max_dev:.2e}")
    print(f"Wrote canonical curve -> {args.out}")


if __name__ == "__main__":
    main()
