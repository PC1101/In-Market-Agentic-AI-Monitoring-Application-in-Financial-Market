"""Provider Protocol conformance + sp500 equivalence to the pre-refactor code path."""
import pandas as pd
import pytest

from providers.base import StrategyPnLProvider, NewsProvider, MacroProvider
from config.profiles import MarketProfile, get_profile


def test_a_minimal_impl_satisfies_strategy_pnl_protocol():
    class Fake:
        def strategies(self):
            return ["AL_PCA"]

        def returns(self, strategy, as_of=None):
            return pd.Series([0.0], index=pd.to_datetime(["2007-08-01"]), name="port_ret")

    assert isinstance(Fake(), StrategyPnLProvider)


def test_missing_method_fails_protocol():
    class Broken:
        def strategies(self):
            return []

    assert not isinstance(Broken(), StrategyPnLProvider)


def test_unknown_market_raises():
    with pytest.raises(KeyError):
        get_profile("does_not_exist")


def test_profile_dataclass_holds_providers():
    prof = MarketProfile(key="x", timezone="UTC", base_ccy="USD",
                         pnl=None, news=None, macro=None, windows=[])
    assert prof.key == "x" and prof.timezone == "UTC"


def test_sp500_profile_providers_conform():
    prof = get_profile("sp500")
    assert isinstance(prof.pnl, StrategyPnLProvider)
    assert isinstance(prof.news, NewsProvider)
    assert isinstance(prof.macro, MacroProvider)


def test_sp500_pnl_matches_direct_load():
    from pnl_loader import load_pnl, returns_series
    from providers.sp500.pnl import AL_CURVE_PATHS

    prof = get_profile("sp500")
    direct = returns_series(load_pnl(AL_CURVE_PATHS[0]))
    via = prof.pnl.returns("AL_PCA")
    # Wrapper concatenates curves in order and keeps first on duplicate dates,
    # so the first curve's values must survive unchanged.
    pd.testing.assert_series_equal(via.loc[direct.index], direct)
