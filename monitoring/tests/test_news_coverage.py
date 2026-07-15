import pandas as pd

from news.coverage import coverage_report
from windows import ALL_WINDOWS


def _cache(tmp_path):
    """A tiny pipeline-style cache parquet (schema of build_news_cache.py)."""
    df = pd.DataFrame({
        "pub_ts": ["2007-08-06T10:00:00", "2007-08-07T10:00:00",
                   "2013-06-03T10:00:00"],
        "headline": ["Quant meltdown hits funds", "Markets calm", "Quiet day"],
        "source": ["nyt", "fnspid", "nyt"],
    })
    path = tmp_path / "cache.parquet"
    df.to_parquet(path)
    return path


def test_coverage_report_counts_per_window(tmp_path):
    rep = coverage_report([_cache(tmp_path)], ALL_WINDOWS).set_index("window")
    assert rep.loc["quant_meltdown_2007", "n_articles"] == 2
    assert rep.loc["quant_meltdown_2007", "n_risk_articles"] == 1
    assert rep.loc["quant_meltdown_2007", "by_source"] == {"fnspid": 1, "nyt": 1}
    assert rep.loc["calm_2013_2014", "n_articles"] == 1
    assert rep.loc["gfc_lehman_2008", "n_articles"] == 0
    assert set(rep.columns) >= {"kind", "start", "end", "n_articles",
                                "n_risk_articles", "articles_per_day", "adequate"}
    # gate: 1 risk article on an event window is inadequate; calm always passes
    assert not rep.loc["quant_meltdown_2007", "adequate"]
    assert rep.loc["calm_2004_2006", "adequate"]
