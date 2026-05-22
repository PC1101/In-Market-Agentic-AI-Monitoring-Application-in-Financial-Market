"""
Grid-search the Avellaneda-Lee stat-arb trading rules WITHOUT ray.

Sweeps three dimensions and ranks every combination by Sharpe ratio:
  * kappa_min  - OU mean-reversion-speed floor (signal selectivity)
  * entry/exit - symmetric s-score bands (s_o = entry, s_c = close)
  * stop-loss  - per-trade cumulative-return stop

Efficiency: kappa_min changes the s-scores (the slow OU fit), so the model is
re-fit ONCE per kappa value; the cheap threshold/stop-loss combos are then swept
by reusing that fitted model via bt.run(). A tqdm bar tracks progress.

The symmetric band maps to backtest thresholds as:
    s_bo=-s_o, s_bc=-s_c, s_so=+s_o, s_sc=+s_c   (requires s_o > s_c > 0)

Examples (run from anywhere; the script cd's to its own directory):

    python optimise.py                                  # XLF, PCA, default grids
    python optimise.py --etf xlf --start 2019-01-02 --end 2020-11-19
    python optimise.py --kappa-grid 0,15,30 --sl-grid 0.05,0.10,none
"""
import argparse
import itertools
import os
import sys

import matplotlib
matplotlib.use("Agg")        # headless: never try to open a window
import pandas as pd          # noqa: E402

try:
    from tqdm import tqdm
except ImportError:          # progress bar optional
    def tqdm(iterable=None, **kwargs):
        return iterable

import yaml                  # noqa: E402

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

from src import backtest     # noqa: E402

DEFAULT_WINDOW = ("2007-01-03", "2015-01-02")


def load_cfg():
    with open("configs/optimise_trading_rules.yml", "r") as fh:
        return yaml.load(fh, Loader=yaml.SafeLoader)


def parse_floats(text):
    return [float(x) for x in text.split(",") if x.strip() != ""]


def parse_kappa(text):
    ## <= 0 means "no kappa filter" (None)
    return [None if float(x) <= 0 else float(x) for x in text.split(",") if x.strip() != ""]


def parse_sl(text):
    ## Stop-loss given as a positive magnitude (e.g. 0.10 = -10% stop); 'none' off.
    ## Positive avoids argparse treating a leading '-' as a flag.
    out = []
    for x in text.split(","):
        x = x.strip()
        if x == "":
            continue
        if x.lower() in ("none", "off"):
            out.append(None)
        else:
            out.append(-abs(float(x)))
    return out


def parse_args(cfg):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--etf", default="xlf", choices=sorted(cfg["prices_file_path"].keys()))
    p.add_argument("--defactoring", default="pca", choices=["pca", "etf"])
    p.add_argument("--start", default=None)
    p.add_argument("--end", default=None)
    p.add_argument("--n-window", type=int, default=60)
    p.add_argument("--kappa-grid", default="0,15,30",
                   help="OU speed floors; 0 disables the filter (default: 0,15,30).")
    p.add_argument("--entry-grid", default="1.0,1.25,1.5,1.75,2.0",
                   help="Entry s-score bands s_o (default: 1.0,1.25,1.5,1.75,2.0).")
    p.add_argument("--exit-grid", default="0.25,0.5,0.75",
                   help="Close s-score bands s_c (default: 0.25,0.5,0.75).")
    p.add_argument("--sl-grid", default="0.05,0.10,none",
                   help="Stop-loss magnitudes as POSITIVE numbers (0.10 = -10%% stop); "
                        "'none' disables (default: 0.05,0.10,none).")
    p.add_argument("--cost", type=float, default=None,
                   help="Per-side transaction cost as a return fraction "
                        "(0.0005 = 5 bps). Use 0 for frictionless. Default: config value.")
    p.add_argument("--top", type=int, default=15, help="How many best configs to print.")
    return p.parse_args()


def main():
    cfg = load_cfg()
    args = parse_args(cfg)

    cfg_start, cfg_end = cfg.get("bt_dt", {}).get(args.etf, DEFAULT_WINDOW)
    st_dt = args.start or cfg_start
    ed_dt = args.end or cfg_end
    prices_file_path = cfg["prices_file_path"][args.etf]

    kappa_grid = parse_kappa(args.kappa_grid)
    entry_grid = parse_floats(args.entry_grid)
    exit_grid = parse_floats(args.exit_grid)
    sl_grid = parse_sl(args.sl_grid)

    # Valid bands only: close must sit inside entry (s_o > s_c > 0).
    band_combos = [(s_o, s_c) for s_o, s_c in itertools.product(entry_grid, exit_grid)
                   if s_o > s_c > 0]
    combos_per_kappa = list(itertools.product(band_combos, sl_grid))

    weighting_scheme = cfg.get("weighting_scheme", "equal_weighted")
    long_only = cfg.get("long_only", False)
    transaction_cost = ((args.cost, args.cost) if args.cost is not None
                        else tuple(cfg.get("transaction_cost", (0.0005, 0.0005))))

    print(f"ETF={args.etf.upper()}  defactoring={args.defactoring}  window={st_dt}..{ed_dt}")
    print(f"cost/side  : {transaction_cost[0]} "
          f"({'frictionless' if transaction_cost[0] == 0 else f'{transaction_cost[0] * 1e4:g} bps'})")
    print(f"kappa grid : {kappa_grid}")
    print(f"bands      : {len(band_combos)} (entry x exit, filtered)")
    print(f"stop-loss  : {sl_grid}")
    print(f"total runs : {len(kappa_grid)} fits x {len(combos_per_kappa)} combos "
          f"= {len(kappa_grid) * len(combos_per_kappa)} backtests\n")

    records = []
    for kappa_min in kappa_grid:
        # Expensive step: fit the OU process / generate s-scores once per kappa.
        model = backtest.bt(
            prices_file_path=prices_file_path,
            etf_name=args.etf,
            st_dt=st_dt,
            ed_dt=ed_dt,
            n_window=args.n_window,
            defactoring=args.defactoring,
            performance_only=True,
            kappa_min=kappa_min,
            progress=True,
        )

        kappa_label = "off" if kappa_min is None else f"{kappa_min:g}"
        for (s_o, s_c), sl in tqdm(combos_per_kappa,
                                   desc=f"sweep kappa={kappa_label}", unit="bt"):
            s_thresholds = {"s_bo": -s_o, "s_bc": -s_c, "s_so": s_o, "s_sc": s_c}
            sharpe, maxdd, endpnl = model.run(
                weighting_scheme=weighting_scheme,
                sl=sl,
                long_only=long_only,
                transaction_cost=transaction_cost,
                s_thresholds=s_thresholds,
            )
            records.append({
                "kappa_min": kappa_min,
                "entry_s": s_o,
                "exit_s": s_c,
                "stop_loss": sl,
                "sharpe": sharpe,
                "max_dd": maxdd,
                "end_pnl": endpnl,
            })

    results = pd.DataFrame(records).sort_values("sharpe", ascending=False,
                                                na_position="last").reset_index(drop=True)

    out_dir = os.path.join("results", "run_backtest", f"{args.etf}_{args.defactoring}")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "optimise_results.csv")
    results.to_csv(out_path, index=False)

    show = results.head(args.top).copy()
    show["max_dd"] = (show["max_dd"] * 100).round(2)
    show["end_pnl"] = show["end_pnl"].round(3)
    show["sharpe"] = show["sharpe"].round(3)
    ## None -> "off" so the kappa-filter / no-stop rows don't read as NaN errors
    show["kappa_min"] = show["kappa_min"].apply(lambda v: "off" if pd.isna(v) else f"{v:g}")
    show["stop_loss"] = show["stop_loss"].apply(lambda v: "off" if pd.isna(v) else f"{v:g}")
    show = show.rename(columns={"max_dd": "max_dd_%"})
    print(f"\n=== Top {min(args.top, len(show))} configs by Sharpe ===")
    with pd.option_context("display.width", 120):
        print(show.to_string(index=False))
    print(f"\nFull ranked grid ({len(results)} rows) saved to: {out_path}")


if __name__ == "__main__":
    main()
