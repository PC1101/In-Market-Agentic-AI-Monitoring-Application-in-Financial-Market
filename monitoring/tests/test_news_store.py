"""build_store: chunked CSV -> per-year parquet; NewsStore: date/ticker queries."""
import pandas as pd
import pytest

from news.store import build_store, NewsStore, detect_columns


@pytest.fixture
def raw_csv(tmp_path):
    df = pd.DataFrame({
        "Date": [
            "2007-08-06 09:00:00 UTC", "2007-08-09 12:00:00 UTC",
            "2010-05-06 14:45:00 UTC", "1999-01-04 10:00:00 UTC",   # out of range
            "not-a-date",                                            # unparseable
        ],
        "Article_title": ["Quant funds hit", "Selloff deepens", "Flash crash", "Old", "Bad"],
        "Stock_symbol": ["GS", "", "SPY", "IBM", "XX"],
        "Lsa_summary": ["s1", "s2", "s3", "s4", "s5"],
        "Publisher": ["P"] * 5,
        "Url": ["u"] * 5,
    })
    p = tmp_path / "raw.csv"
    df.to_csv(p, index=False)
    return p


def test_detect_columns(raw_csv):
    m = detect_columns(raw_csv)
    assert m["Date"] == "date" and m["Article_title"] == "title"
    assert m["Stock_symbol"] == "ticker" and m["Lsa_summary"] == "summary"


def test_build_store_filters_dates_and_partitions_by_year(tmp_path, raw_csv):
    out = tmp_path / "store"
    n = build_store(raw_csv, out, start="2003-01-01", end="2016-12-31", chunksize=2)
    assert n == 3  # 1999 row and unparseable row dropped
    assert (out / "year=2007.parquet").exists()
    assert (out / "year=2010.parquet").exists()
    assert not (out / "year=1999.parquet").exists()


def test_query_by_date_range(tmp_path, raw_csv):
    out = tmp_path / "store"
    build_store(raw_csv, out, start="2003-01-01", end="2016-12-31")
    store = NewsStore(out)
    df = store.query("2007-08-01", "2007-08-31")
    assert len(df) == 2
    assert list(df.columns) == ["date", "ticker", "title", "summary", "publisher", "url"]
    assert df["date"].is_monotonic_increasing
    assert df["date"].dt.tz is None  # naive timestamps, comparable to curve index


def test_query_by_ticker(tmp_path, raw_csv):
    out = tmp_path / "store"
    build_store(raw_csv, out, start="2003-01-01", end="2016-12-31")
    df = NewsStore(out).query("2007-01-01", "2010-12-31", tickers=["SPY"])
    assert len(df) == 1 and df.iloc[0]["title"] == "Flash crash"


def test_query_missing_years_returns_empty(tmp_path, raw_csv):
    out = tmp_path / "store"
    build_store(raw_csv, out, start="2003-01-01", end="2016-12-31")
    df = NewsStore(out).query("2013-01-01", "2013-12-31")
    assert df.empty and list(df.columns) == ["date", "ticker", "title", "summary", "publisher", "url"]
