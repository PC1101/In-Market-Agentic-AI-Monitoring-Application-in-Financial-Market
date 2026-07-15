import pandas as pd

from news.coverage import coverage_report
from news.store import build_store, NewsStore
from windows import ALL_WINDOWS


def test_coverage_report_counts_per_window(tmp_path):
    df = pd.DataFrame({
        "Date": ["2007-08-06 10:00:00 UTC", "2007-08-07 10:00:00 UTC",
                 "2013-06-03 10:00:00 UTC"],
        "Article_title": ["Quant meltdown hits funds", "Markets calm", "Quiet day"],
        "Stock_symbol": ["GS", "GS", "SPY"],
        "Lsa_summary": ["", "", ""],
        "Publisher": ["P"] * 3,
        "Url": ["u"] * 3,
    })
    raw = tmp_path / "raw.csv"
    df.to_csv(raw, index=False)
    build_store(raw, tmp_path / "store")

    rep = coverage_report(NewsStore(tmp_path / "store"), ALL_WINDOWS)
    rep = rep.set_index("window")
    assert rep.loc["quant_meltdown_2007", "n_articles"] == 2
    assert rep.loc["quant_meltdown_2007", "n_risk_articles"] == 1
    assert rep.loc["calm_2013_2014", "n_articles"] == 1
    assert rep.loc["gfc_lehman_2008", "n_articles"] == 0
    assert set(rep.columns) >= {"kind", "start", "end", "n_articles", "n_risk_articles",
                                "articles_per_day", "adequate"}
