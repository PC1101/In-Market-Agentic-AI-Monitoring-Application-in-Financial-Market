from pathlib import Path
import pandas as pd
from pnl_loader import load_pnl, REQUIRED_COLUMNS

REPO = Path(__file__).resolve().parents[2]
JT_ENERGY = REPO / "XSectional" / "results" / "equity_curve_energy.csv"
AL_ENERGY = REPO / "Stat Arb" / "statsArb-dev" / "results" / "equity_curve_energy_al.csv"


def test_energy_al_curve_has_valid_schema():
    assert AL_ENERGY.exists()
    df = load_pnl(AL_ENERGY)
    assert list(df.columns) == list(REQUIRED_COLUMNS)
    assert len(df) > 100
    assert df["port_ret"].abs().sum() > 0   # non-degenerate


def test_energy_jt_curve_has_valid_schema():
    assert JT_ENERGY.exists(), "run: python XSectional/run_energy_pnl.py"
    df = load_pnl(JT_ENERGY)
    assert list(df.columns) == list(REQUIRED_COLUMNS)
    assert len(df) > 250
    assert df["equity"].iloc[-1] > 0
    # Guard against a degenerate (all-skipped-rebalance) curve: with the
    # curated ~34-name energy universe, the shared config's 20%-quantile /
    # min-10-per-leg defaults skip every month and produce a flat curve
    # (port_ret == 0 throughout, equity constant at 1.0). A real momentum
    # book must show nonzero daily PnL and a non-constant equity path.
    assert df["port_ret"].abs().sum() > 0
    assert df["equity"].nunique() > 1
