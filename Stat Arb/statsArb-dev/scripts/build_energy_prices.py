"""Write the curated energy universe's adjusted-close prices as a bt-format CSV.

Downloads the curated ~34-name global-energy universe's daily adjusted-close
prices (``monitoring/providers/energy/universe.py``, free via yfinance) and
writes them to a CSV indexed by ``Date`` with one column per ticker — the
format the AL stat-arb engine's ``bt`` class (``src/backtest.py``) reads via
``prices_file_path``.
"""
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO = Path(__file__).resolve().parents[3]
MONITORING_DIR = REPO / "monitoring"


def _prepend_to_syspath(path: Path) -> None:
    """Force `path` to the front of sys.path, deduping any prior occurrence.

    Running `python scripts/build_energy_prices.py` auto-inserts SCRIPT_DIR
    into sys.path[0] *before* this module's code runs, so a plain
    ``if str(p) not in sys.path: sys.path.insert(0, ...)`` guard silently
    no-ops when MONITORING_DIR is already present elsewhere on sys.path but
    not first. Removing-then-reinserting guarantees MONITORING_DIR wins.
    """
    p = str(path)
    if p in sys.path:
        sys.path.remove(p)
    sys.path.insert(0, p)


_prepend_to_syspath(MONITORING_DIR)

from providers.energy.universe import load_prices  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "data" / "energy_universe" / "prices.csv"


def main() -> None:
    px = load_prices(start="2016-01-01", end="2022-12-31").dropna(axis=1, how="all")
    px.index.name = "Date"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    px.to_csv(OUT)
    print(f"wrote {OUT}: {px.shape[1]} tickers x {len(px)} days")


if __name__ == "__main__":
    main()
