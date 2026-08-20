"""Plot the energy market's JT momentum + AL PCA equity curves as a $1M account.

Renders account value (starting at $1,000,000) and drawdown for both strategies
from the committed energy curves, saved to monitoring/results/energy_pnl_1M.png.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
START_CAPITAL = 1_000_000
CURVES = {
    "JT momentum": REPO / "XSectional" / "results" / "equity_curve_energy.csv",
    "AL PCA stat-arb": REPO / "Stat Arb" / "statsArb-dev" / "results" / "equity_curve_energy_al.csv",
}
COLORS = {"JT momentum": "#c2410c", "AL PCA stat-arb": "#1d4ed8"}
OUT = REPO / "monitoring" / "results" / "energy_pnl_1M.png"


def _load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
    # Normalise so the account starts at exactly $1M on day one.
    df["account"] = START_CAPITAL * df["equity"] / df["equity"].iloc[0]
    df["dd"] = df["account"] / df["account"].cummax() - 1.0
    return df


def main() -> None:
    curves = {k: _load(p) for k, p in CURVES.items()}

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12, 7), height_ratios=[3, 1], sharex=True,
        gridspec_kw={"hspace": 0.08})

    for name, df in curves.items():
        final = df["account"].iloc[-1]
        ret = final / START_CAPITAL - 1
        maxdd = df["dd"].min()
        ax1.plot(df.index, df["account"], color=COLORS[name], lw=1.6,
                 label=f"{name}:  ${final:,.0f}  ({ret:+.1%},  maxDD {maxdd:.1%})")
        ax2.fill_between(df.index, df["dd"] * 100, 0, color=COLORS[name], alpha=0.35)

    ax1.axhline(START_CAPITAL, color="#6b7280", lw=1.0, ls="--", alpha=0.8)
    ax1.set_title("Global energy market — account value from $1,000,000 (2016–2022)",
                  fontsize=13, fontweight="bold")
    ax1.set_ylabel("Account value (USD)")
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1e6:.2f}M"))
    ax1.legend(loc="upper left", fontsize=10, frameon=False)
    ax1.grid(True, alpha=0.25)

    ax2.set_ylabel("Drawdown (%)")
    ax2.set_xlabel("Date")
    ax2.grid(True, alpha=0.25)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=140, bbox_inches="tight")
    print(f"wrote {OUT}")
    for name, df in curves.items():
        print(f"  {name}: start $1,000,000 -> end ${df['account'].iloc[-1]:,.0f} "
              f"({df['account'].iloc[-1]/START_CAPITAL - 1:+.1%}), maxDD {df['dd'].min():.1%}")


if __name__ == "__main__":
    main()
