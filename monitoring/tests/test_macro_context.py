import pandas as pd
import pytest

from macro.context import macro_context
from agentic.guardrails import assert_no_lookahead


@pytest.fixture
def macro_dir(tmp_path):
    pd.DataFrame({
        "date": pd.to_datetime(["2007-08-03", "2007-08-06"]),
        "value": [25.16, 26.25],
    }).to_parquet(tmp_path / "VIXCLS.parquet")
    pd.DataFrame({
        "date": pd.to_datetime(["2007-08-03", "2007-08-06"]),
        "value": [4.70, 4.73],
    }).to_parquet(tmp_path / "DGS10.parquet")
    pd.DataFrame({
        "date": pd.to_datetime(["2007-08-03", "2007-08-06"]),
        "value": [4.56, 4.58],
    }).to_parquet(tmp_path / "DGS2.parquet")
    pd.DataFrame({
        "date": pd.to_datetime(["2007-07-01"]),
        "value": [4.6],
        "realtime_start": pd.to_datetime(["2007-08-03"]),
        "realtime_end": pd.to_datetime(["2262-04-01"]),
    }).to_parquet(tmp_path / "UNRATE.parquet")
    return tmp_path


def test_macro_context_assembles_asof_block(macro_dir):
    ctx = macro_context("2007-08-06", macro_dir)
    assert ctx["vixcls"]["value"] == 26.25
    assert ctx["unrate"]["published"] == "2007-08-03"
    assert ctx["yield_curve_10y_2y"] == pytest.approx(0.15, abs=1e-9)


def test_macro_context_never_leaks_future(macro_dir):
    ctx = macro_context("2007-08-03", macro_dir)
    assert ctx["vixcls"]["obs_date"] == "2007-08-03"
    assert_no_lookahead({"macro": ctx}, "2007-08-03")   # must not raise


def test_macro_context_missing_series_is_omitted(macro_dir):
    ctx = macro_context("2007-08-06", macro_dir)
    assert "tedrate" not in ctx    # no TEDRATE.parquet in fixture
