# report.py — compute performance metrics and render tearsheet PNG

import os

import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe in all environments
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config


def compute_metrics(returns: pd.Series) -> dict:
    """
    Compute annualised performance metrics from a monthly returns Series.

    Returns a dict with:
      Annualised Return    — geometric mean annualised return
      Annualised Volatility — annualised standard deviation of monthly returns
      Sharpe Ratio         — Ann. Return / Ann. Volatility (risk-free = 0)
      Max Drawdown         — largest peak-to-trough decline (≤ 0)
      Calmar Ratio         — Ann. Return / |Max Drawdown| (0 if drawdown is 0)
    """
    n = len(returns)
    ann_return = (1 + returns).prod() ** (12 / n) - 1
    ann_vol = returns.std() * np.sqrt(12)
    sharpe = (ann_return / ann_vol) if ann_vol > 0 else 0.0

    cum = (1 + returns).cumprod()
    rolling_max = cum.cummax()
    drawdown = (cum - rolling_max) / rolling_max
    max_dd = drawdown.min()

    calmar = (ann_return / abs(max_dd)) if max_dd < 0 else 0.0

    return {
        "Annualised Return": ann_return,
        "Annualised Volatility": ann_vol,
        "Sharpe Ratio": sharpe,
        "Max Drawdown": max_dd,
        "Calmar Ratio": calmar,
    }


def generate_report(returns: pd.Series) -> None:
    """
    Print performance metrics to stdout and save a 4-panel tearsheet PNG
    to {config.DATA_DIR}/tearsheet.png.

    Panels:
      1. Cumulative equity curve (log scale)
      2. Annual return bar chart
      3. Rolling 12-month Sharpe ratio
      4. Drawdown chart
    """
    metrics = compute_metrics(returns)

    print("\n=== XSectional Momentum Tearsheet ===")
    for label, value in metrics.items():
        if "Drawdown" in label or "Return" in label or "Volatility" in label:
            print(f"  {label}: {value:.2%}")
        else:
            print(f"  {label}: {value:.2f}")

    cum = (1 + returns).cumprod()
    rolling_max = cum.cummax()
    drawdown = (cum - rolling_max) / rolling_max
    rolling_sharpe = returns.rolling(12).apply(
        lambda x: (x.mean() * 12) / (x.std() * np.sqrt(12)) if x.std() > 0 else 0.0,
        raw=True,
    )
    annual_returns = returns.resample("YE").apply(
        lambda x: (1 + x).prod() - 1
    )

    fig = plt.figure(figsize=(14, 12))
    gs = gridspec.GridSpec(4, 1, hspace=0.45)

    # Panel 1: Equity curve
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(cum.index, cum.values, linewidth=1.5)
    ax1.set_title("Cumulative Equity Curve (log scale)", fontsize=11)
    ax1.set_ylabel("Growth of $1")
    ax1.set_yscale("log")
    ax1.grid(True, alpha=0.3)

    # Panel 2: Annual returns
    ax2 = fig.add_subplot(gs[1])
    colors = ["steelblue" if r >= 0 else "tomato" for r in annual_returns.values]
    ax2.bar(annual_returns.index.year, annual_returns.values * 100, color=colors)
    ax2.set_title("Annual Returns (%)", fontsize=11)
    ax2.set_ylabel("Return (%)")
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.grid(True, alpha=0.3, axis="y")

    # Panel 3: Rolling Sharpe
    ax3 = fig.add_subplot(gs[2])
    ax3.plot(rolling_sharpe.index, rolling_sharpe.values, linewidth=1.2, color="darkgreen")
    ax3.set_title("Rolling 12-Month Sharpe Ratio", fontsize=11)
    ax3.set_ylabel("Sharpe")
    ax3.axhline(0, color="black", linewidth=0.8)
    ax3.grid(True, alpha=0.3)

    # Panel 4: Drawdown
    ax4 = fig.add_subplot(gs[3])
    ax4.fill_between(
        drawdown.index, drawdown.values * 100, 0,
        color="tomato", alpha=0.5, linewidth=0,
    )
    ax4.set_title("Drawdown (%)", fontsize=11)
    ax4.set_ylabel("Drawdown (%)")
    ax4.grid(True, alpha=0.3)

    os.makedirs(config.DATA_DIR, exist_ok=True)
    output_path = os.path.join(config.DATA_DIR, "tearsheet.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Tearsheet saved → {output_path}")
