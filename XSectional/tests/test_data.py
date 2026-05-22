import pytest
import pandas as pd
import numpy as np
from data import drop_missing, load_prices


def test_drop_missing_removes_high_missing_tickers():
    df = pd.DataFrame({
        "A": [1.0, 2.0, 3.0, 4.0, 5.0],     # 0% missing  → keep
        "B": [None, None, None, None, 1.0],   # 80% missing → drop
        "C": [1.0, None, 3.0, 4.0, 5.0],     # 20% missing → keep (≤ threshold)
    })
    result = drop_missing(df, threshold=0.20)
    assert "A" in result.columns
    assert "C" in result.columns
    assert "B" not in result.columns


def test_drop_missing_keeps_complete_tickers():
    df = pd.DataFrame({
        "X": [1.0, 2.0, 3.0],
        "Y": [None, 2.0, None],   # 67% missing → drop
    })
    result = drop_missing(df, threshold=0.20)
    assert list(result.columns) == ["X"]


def test_drop_missing_empty_dataframe():
    df = pd.DataFrame()
    result = drop_missing(df, threshold=0.20)
    assert result.empty


def test_load_prices_returns_dataframe_from_cache(tmp_path, monkeypatch):
    import data as data_module
    dates = pd.date_range("2000-01-01", periods=10)
    fake_prices = pd.DataFrame(
        np.ones((10, 3)) * 100.0,
        index=dates,
        columns=["AAPL", "MSFT", "GOOG"],
    )
    cache_path = tmp_path / "sp500_prices.csv"
    fake_prices.to_csv(cache_path)

    monkeypatch.setattr(data_module, "SP500_CACHE", str(cache_path))
    prices = load_prices()

    assert isinstance(prices, pd.DataFrame)
    assert prices.shape == (10, 3)
    assert list(prices.columns) == ["AAPL", "MSFT", "GOOG"]


def test_load_prices_raises_when_no_cache_and_no_network(tmp_path, monkeypatch):
    """When cache is missing and download is mocked to fail, an exception is raised."""
    import data as data_module
    monkeypatch.setattr(data_module, "SP500_CACHE", str(tmp_path / "missing.csv"))

    def _fail(*args, **kwargs):
        raise RuntimeError("network unavailable")

    monkeypatch.setattr(data_module, "download_prices", _fail)
    monkeypatch.setattr(data_module, "get_sp500_tickers", lambda: ["AAPL"])

    with pytest.raises(Exception):
        load_prices()
