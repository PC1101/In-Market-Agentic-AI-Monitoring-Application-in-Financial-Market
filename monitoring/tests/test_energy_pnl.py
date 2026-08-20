from pathlib import Path
import pandas as pd
from pnl_loader import load_pnl, REQUIRED_COLUMNS

REPO = Path(__file__).resolve().parents[2]
JT_ENERGY = REPO / "XSectional" / "results" / "equity_curve_energy.csv"


def test_energy_jt_curve_has_valid_schema():
    assert JT_ENERGY.exists(), "run: python XSectional/run_energy_pnl.py"
    df = load_pnl(JT_ENERGY)
    assert list(df.columns) == list(REQUIRED_COLUMNS)
    assert len(df) > 250
    assert df["equity"].iloc[-1] > 0
