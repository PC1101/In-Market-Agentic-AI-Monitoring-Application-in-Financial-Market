"""
Build PCA factor-portfolio returns for the ENERGY universe.

Analog of src/data/hist_pca_factoring.py's S&P 500 ret_pca_port.csv, but for the
34-name energy universe built by scripts/build_energy_prices.py (Task 2), and used
as the `pca_ret` input to the AL engine's defactoring='pca' mode (src/ou_process.py).

Reuses the non-Ray, pure `pca_port` function from src/pca_factoring.py in a plain
Python loop -- no Ray, no tqdm. 34 names x ~1700 days is fast enough without them.

n_components choice: see docstring note below / task report. ou_process.py consumes
pca_ret generically (however many columns are present -- no hardcoded factor count),
so we use n_components=5 rather than matching the S&P universe's 15.

Usage (run from inside Stat Arb/statsArb-dev):
    ../../.venv/bin/python scripts/build_energy_pca.py
"""
import os
import sys

import numpy as np
import pandas as pd

# src/pca_factoring.py opens configs/optimise_trading_rules.yml with a relative
# path at import time, so chdir to the engine dir (this script's parent) first,
# mirroring the pattern in run_full_universe.py.
HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)  # Stat Arb/statsArb-dev
os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

from src.pca_factoring import pca_port  # noqa: E402

N_EST = 60
N_COMPONENTS = 5  # see module docstring / task report: ou_process consumes pca_ret
                   # generically, so we don't need to match the S&P universe's 15.

PRICES_PATH = os.path.join(PROJECT_ROOT, "data", "energy_universe", "prices.csv")
OUT_PATH = os.path.join(PROJECT_ROOT, "results", "pca_factoring", "ret_pca_port_energy.csv")


def main():
    prices = pd.read_csv(PRICES_PATH, index_col="Date", parse_dates=True)
    mkt_ret_df = prices.pct_change()

    rows = []
    for i in range(N_EST, mkt_ret_df.shape[0]):
        window = (
            mkt_ret_df.iloc[i - N_EST : i]
            .replace(0, np.nan)
            .dropna(axis=1, how="all")
            .dropna(axis=0, how="all")
            .dropna(axis=1, how="any")
        )
        cur_ret = mkt_ret_df.iloc[i]
        dt = mkt_ret_df.iloc[i].name
        rows.append(pca_port((window, dt, cur_ret, N_COMPONENTS)))

    ret_pca_port = pd.DataFrame(
        rows, columns=["Date"] + [f"pca_{n}" for n in range(N_COMPONENTS)]
    ).set_index("Date")

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    ret_pca_port.to_csv(OUT_PATH)

    print(ret_pca_port.shape)
    print(ret_pca_port.head())
    print(f"date range: {ret_pca_port.index.min()} .. {ret_pca_port.index.max()}")
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
