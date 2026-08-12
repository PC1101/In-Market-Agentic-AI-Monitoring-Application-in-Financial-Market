"""Verify that gross vs net curves produce near-identical agent input features.

Justifies reusing existing agentic JSONL logs (generated on gross curves) with
net return streams. Computes max divergence of rolling 21d vol and running
drawdown between gross and net-base curves across all 12 evaluation windows.

Usage (from monitoring/):
    python scripts/verify_gross_net_features.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from windows import ALL_WINDOWS_FULL  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

RESULTS = Path(__file__).resolve().parents[1] / "results"

STRATEGY_CURVES = {
    "JT_MOM": {
        "gross": Path(__file__).resolve().parents[2] / "XSectional" / "results" / "equity_curve_daily_long_only.csv",
        "net":   Path(__file__).resolve().parents[2] / "XSectional" / "results" / "equity_curve_daily_long_only_net_base.csv",
    },
    "AL_PCA": {
        "gross": Path(__file__).resolve().parents[2] / "Stat Arb" / "statsArb-dev" / "results" / "full_universe" / "pit_pca_long_only" / "equity_curve.csv",
        "net":   Path(__file__).resolve().parents[2] / "Stat Arb" / "statsArb-dev" / "results" / "full_universe" / "pit_pca_long_only" / "equity_curve_net_base.csv",
    },
}


def load_curve(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, index_col=0, parse_dates=True)


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute rolling 21d vol and running drawdown — the key agent input features."""
    ret = df["port_ret"]
    vol_21d = ret.rolling(21).std() * np.sqrt(252)
    equity = df["equity"]
    drawdown = equity / equity.cummax() - 1.0
    return pd.DataFrame({"vol_21d": vol_21d, "drawdown": drawdown}, index=df.index)


def main():
    report_rows = []

    for strat, paths in STRATEGY_CURVES.items():
        gross = load_curve(paths["gross"])
        net = load_curve(paths["net"])

        feat_gross = compute_features(gross)
        feat_net = compute_features(net)

        # Align
        idx = feat_gross.index.intersection(feat_net.index)
        fg = feat_gross.reindex(idx)
        fn = feat_net.reindex(idx)

        for w in ALL_WINDOWS_FULL:
            mask = (idx >= w.start_ts) & (idx <= w.end_ts)
            if mask.sum() == 0:
                continue

            vol_diff = (fg.loc[mask, "vol_21d"] - fn.loc[mask, "vol_21d"]).abs()
            dd_diff = (fg.loc[mask, "drawdown"] - fn.loc[mask, "drawdown"]).abs()

            report_rows.append({
                "strategy": strat,
                "window": w.name,
                "n_days": int(mask.sum()),
                "max_vol_diff_pct": round(float(vol_diff.max()) * 100, 4),
                "mean_vol_diff_pct": round(float(vol_diff.mean()) * 100, 4),
                "max_dd_diff_pp": round(float(dd_diff.max()) * 100, 4),
                "mean_dd_diff_pp": round(float(dd_diff.mean()) * 100, 4),
            })

    report = pd.DataFrame(report_rows)
    out_path = RESULTS / "gross_net_feature_divergence.csv"
    report.to_csv(out_path, index=False)

    print("\n" + "=" * 80)
    print("GROSS vs NET FEATURE DIVERGENCE (justifies log reuse)")
    print("=" * 80)
    print(report.to_string(index=False))
    print()

    max_vol = report["max_vol_diff_pct"].max()
    max_dd = report["max_dd_diff_pp"].max()
    print(f"Global max vol divergence:      {max_vol:.4f}% annualised")
    print(f"Global max drawdown divergence: {max_dd:.4f} pp")
    print()

    if max_vol < 1.0 and max_dd < 5.0:
        print("PASS: divergence is small relative to agent decision thresholds")
        print("      (vol: <1% ann.; drawdown: <5pp vs typical -10% to -50% events).")
        print("      Agentic log reuse is justified with disclosure.")
    else:
        print("WARNING: non-trivial divergence detected — consider re-running agentic cells.")

    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
