import pandas as pd

from news.filter import score_text, filter_news


def test_score_text_matches_risk_terms():
    terms = score_text("Quant funds face margin calls as selloff deepens")
    assert "margin call" in terms and "sell-off/selloff" in terms


def test_score_text_clean_headline():
    assert score_text("Apple announces new iPhone lineup") == []


def test_filter_news_adds_columns_and_keeps_only_hits():
    df = pd.DataFrame({
        "date": pd.to_datetime(["2007-08-06", "2007-08-07"]),
        "ticker": ["GS", "AAPL"],
        "title": ["Hedge fund liquidation triggers market turmoil", "New product launch"],
        "summary": ["", ""],
        "publisher": ["P", "P"],
        "url": ["u", "u"],
    })
    out = filter_news(df)
    assert len(out) == 1
    assert out.iloc[0]["ticker"] == "GS"
    assert out.iloc[0]["n_risk_terms"] >= 2  # liquidation + turmoil
    assert isinstance(out.iloc[0]["risk_terms"], list)


def test_filter_news_empty_input():
    df = pd.DataFrame(columns=["date", "ticker", "title", "summary", "publisher", "url"])
    out = filter_news(df)
    assert out.empty and "n_risk_terms" in out.columns
