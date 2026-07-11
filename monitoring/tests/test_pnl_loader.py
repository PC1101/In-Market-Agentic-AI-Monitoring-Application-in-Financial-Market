import pandas as pd
import pytest

from pnl_loader import load_pnl, returns_series, as_of, REQUIRED_COLUMNS


def _write_csv(tmp_path, text, name="equity_curve.csv"):
    p = tmp_path / name
    p.write_text(text)
    return p


def test_load_explicit_date_column(tmp_path):
    p = _write_csv(tmp_path, "Date,port_ret,equity,drawdown\n"
                            "2007-01-03,0.0,1.0,0.0\n"
                            "2007-01-04,0.01,1.01,0.0\n")
    df = load_pnl(p)
    assert list(df.columns) == list(REQUIRED_COLUMNS)
    assert df.index.name == "Date"
    assert isinstance(df.index, pd.DatetimeIndex)
    assert len(df) == 2


def test_load_unnamed_index_column(tmp_path):
    # arbitragelab driver writes an unnamed leading date column
    p = _write_csv(tmp_path, ",port_ret,equity,drawdown\n"
                            "2008-12-31,0.012,1.012,0.0\n"
                            "2009-01-02,0.003,1.015,0.0\n")
    df = load_pnl(p)
    assert list(df.columns) == list(REQUIRED_COLUMNS)
    assert len(df) == 2


def test_missing_column_raises(tmp_path):
    p = _write_csv(tmp_path, "Date,port_ret,equity\n2007-01-03,0.0,1.0\n")
    with pytest.raises(ValueError):
        load_pnl(p)


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_pnl(tmp_path / "nope.csv")


def test_returns_series_and_as_of(tmp_path):
    p = _write_csv(tmp_path, "Date,port_ret,equity,drawdown\n"
                            "2007-01-03,0.0,1.0,0.0\n"
                            "2007-01-04,0.01,1.01,0.0\n"
                            "2007-01-05,-0.02,0.99,-0.02\n")
    df = load_pnl(p)
    s = returns_series(df)
    assert s.name == "port_ret"
    sliced = as_of(s, "2007-01-04")
    assert len(sliced) == 2
    assert sliced.index.max() == pd.Timestamp("2007-01-04")


def test_sorts_by_date(tmp_path):
    p = _write_csv(tmp_path, "Date,port_ret,equity,drawdown\n"
                            "2007-01-05,-0.02,0.99,-0.02\n"
                            "2007-01-03,0.0,1.0,0.0\n")
    df = load_pnl(p)
    assert df.index.is_monotonic_increasing
