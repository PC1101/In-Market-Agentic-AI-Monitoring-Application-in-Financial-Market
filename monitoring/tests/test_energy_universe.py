"""Structure tests for the curated global-energy universe (no network)."""
from providers.energy import universe


def test_universe_is_nonempty_and_deduped():
    ts = universe.tickers()
    assert len(ts) >= 30, "curated energy universe should be ~30-40 names"
    assert len(ts) == len(set(ts)), "no duplicate tickers"


def test_universe_sorted():
    assert universe.tickers() == sorted(universe.tickers())


def test_major_integrated_names_present():
    # The stable megacaps that anchor the survivorship argument.
    for t in ("XOM", "CVX", "SHEL", "BP", "TTE", "COP"):
        assert t in universe.UNIVERSE


def test_exited_names_documented_for_survivorship_caveat():
    # The caveat must be explicit, not silent.
    assert universe.EXITED, "M&A exits must be documented"
    assert "APC" in universe.EXITED  # Anadarko -> Occidental 2019
