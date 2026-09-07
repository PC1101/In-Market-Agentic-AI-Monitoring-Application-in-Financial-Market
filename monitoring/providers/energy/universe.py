"""Curated global-energy megacap universe + free (yfinance) price loader.

Why a curated fixed universe (design §6, §11.2): the full S&P Global 1200 Energy
index with survivorship-free point-in-time membership is paywalled, but the global
energy sector is dominated by a small, very stable set of megacaps that rarely
delist. A curated ~35-name universe of major global energy companies (US-listed or
US-listed ADR, so prices are free via yfinance) is representative and
methodologically defensible, with the residual survivorship caveat documented in
``EXITED`` below. Energy regime breaks are commodity-driven (free FRED curves), so
the universe need only be representative, not index-exact.

All tickers resolve on yfinance without an API key. ADRs are used for non-US majors
(SHEL, BP, TTE, E, EQNR, SU, CVE, IMO, ENB, TRP) so a single free source covers the
whole universe.
"""
from __future__ import annotations

import pandas as pd

#: Curated global-energy universe, grouped by sub-sector. Ticker -> display name.
UNIVERSE: dict[str, str] = {
    # Integrated / international majors (ADRs for non-US)
    "XOM": "ExxonMobil", "CVX": "Chevron", "SHEL": "Shell", "BP": "BP",
    "TTE": "TotalEnergies", "COP": "ConocoPhillips", "E": "Eni", "EQNR": "Equinor",
    "SU": "Suncor Energy", "CVE": "Cenovus Energy", "IMO": "Imperial Oil",
    # US exploration & production
    "EOG": "EOG Resources", "OXY": "Occidental", "DVN": "Devon Energy",
    "APA": "APA Corp", "FANG": "Diamondback", "OVV": "Ovintiv",
    # NOTE: Hess (HES) excluded — not reliably served by the free yfinance source
    # (a Yahoo coverage gap, not an M&A exit); revisit if a paid feed is added.
    # NOTE: Coterra (CTRA) intentionally excluded — it was formed in 2021 (Cabot +
    # Cimarex), so it does not exist in a 2007-2020 study window. Cabot (COG) and
    # Cimarex (XEC, see EXITED) are its pre-merger constituents.
    # Oilfield services
    "SLB": "SLB (Schlumberger)", "HAL": "Halliburton", "BKR": "Baker Hughes",
    "NOV": "NOV Inc", "FTI": "TechnipFMC",
    # Midstream (ADRs for ENB, TRP)
    "KMI": "Kinder Morgan", "WMB": "Williams", "OKE": "ONEOK",
    "ENB": "Enbridge", "TRP": "TC Energy", "EPD": "Enterprise Products",
    "ET": "Energy Transfer", "MPLX": "MPLX", "LNG": "Cheniere Energy",
    # Refiners
    "MPC": "Marathon Petroleum", "VLO": "Valero", "PSX": "Phillips 66",
}

#: Names that exited via M&A during a typical 2007-2020 study window — the
#: documented survivorship caveat. Included so the caveat is explicit and the
#: names can be added to a point-in-time run if their pre-exit prices are pulled.
EXITED: dict[str, str] = {
    "PXD": "Pioneer Natural Resources (-> ExxonMobil, 2024)",
    "MRO": "Marathon Oil (-> ConocoPhillips, 2024)",
    "APC": "Anadarko Petroleum (-> Occidental, 2019)",
    "BG.L": "BG Group (-> Shell, 2016)",
    "XEC": "Cimarex Energy (-> Cabot => Coterra, 2021)",
}


def tickers() -> list[str]:
    """Sorted list of the curated universe tickers."""
    return sorted(UNIVERSE)


def load_prices(start: str = "2007-01-01", end: str = "2020-12-31",
                symbols: list[str] | None = None) -> pd.DataFrame:
    """Adjusted close price panel for the universe (free, via yfinance).

    Returns a DataFrame indexed by date with one column per ticker. Requires
    network + the ``yfinance`` package; kept out of import path so the module
    imports without a network call.
    """
    import yfinance as yf

    syms = symbols if symbols is not None else tickers()
    df = yf.download(syms, start=start, end=end, progress=False, auto_adjust=True)
    close = df["Close"] if "Close" in df else df
    return close.sort_index()
