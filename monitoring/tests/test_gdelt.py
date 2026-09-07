"""Tests for the GDELT news ingestion adapter (pure mapping/parse logic, no network)."""
import pandas as pd

from news import gdelt
from news.store import COLS


def test_parse_seendate():
    ts = gdelt.parse_seendate("20200305T010000Z")
    assert ts == pd.Timestamp("2020-03-05 01:00:00")


def test_parse_seendate_bad_returns_nat():
    assert pd.isna(gdelt.parse_seendate("not-a-date"))


def test_article_to_row_maps_fields_and_tags_ticker():
    art = {
        "url": "https://example.com/a",
        "title": "Oil slumps on demand fears",
        "seendate": "20200305T010000Z",
        "domain": "example.com",
        "language": "English",
        "sourcecountry": "United States",
    }
    row = gdelt.article_to_row(art, ticker="XOM")
    assert row["ticker"] == "XOM"
    assert row["title"] == "Oil slumps on demand fears"
    assert row["publisher"] == "example.com"
    assert row["url"] == "https://example.com/a"
    assert row["summary"] == ""  # DOC API returns no body/summary
    assert row["date"] == pd.Timestamp("2020-03-05 01:00:00")


def test_articles_to_frame_has_exact_newsstore_schema():
    arts = [
        {"url": "u1", "title": "t1", "seendate": "20200305T010000Z", "domain": "d1"},
        {"url": "u2", "title": "t2", "seendate": "20200306T010000Z", "domain": "d2"},
    ]
    df = gdelt.articles_to_frame(arts, ticker="CVX")
    assert list(df.columns) == list(COLS)  # exact column order NewsStore expects
    assert len(df) == 2
    assert (df["ticker"] == "CVX").all()
    assert str(df["date"].dtype).startswith("datetime64")


def test_articles_to_frame_drops_unparseable_dates():
    arts = [
        {"url": "u1", "title": "t1", "seendate": "bad", "domain": "d1"},
        {"url": "u2", "title": "t2", "seendate": "20200306T010000Z", "domain": "d2"},
    ]
    df = gdelt.articles_to_frame(arts, ticker="CVX")
    assert len(df) == 1 and df.iloc[0]["url"] == "u2"


def test_build_query_ors_terms_and_quotes_multiword():
    q = gdelt.build_query(["Kinder Morgan", "KMI"])
    assert '"Kinder Morgan"' in q  # multi-word term quoted
    assert "KMI" in q              # single token unquoted
    assert " OR " in q


def test_store_build_and_provider_roundtrip(tmp_path, monkeypatch):
    """build_gdelt_store -> per-year parquet -> NewsStore/EnergyNews read-back.

    Mocks the network fetch (proven live separately) so the store-write and the
    exact FNSPID-schema read path are validated deterministically.
    """
    from providers.energy.news import EnergyNews

    fake = {
        "XOM": [
            {"url": "u1", "title": "Exxon cuts capex", "seendate": "20200315T120000Z", "domain": "reuters.com", "language": "English"},
            {"url": "u2", "title": "Exxon rallies", "seendate": "20200402T120000Z", "domain": "bloomberg.com", "language": "English"},
        ],
        "CVX": [
            {"url": "u3", "title": "Chevron trims budget", "seendate": "20200320T120000Z", "domain": "wsj.com", "language": "English"},
        ],
    }
    monkeypatch.setattr(gdelt, "fetch", lambda q, s, e, **kw: fake["XOM"] if "XOM" in q else fake["CVX"])

    counts = gdelt.build_gdelt_store(
        {"XOM": ["ExxonMobil", "XOM"], "CVX": ["Chevron", "CVX"]},
        "2020-03-01", "2020-04-30", tmp_path, throttle_s=0)
    assert counts.get("2020") == 3

    df = EnergyNews(root=tmp_path).query("2020-03-01", "2020-04-30")
    assert list(df.columns) == list(COLS)
    assert len(df) == 3
    assert set(df["ticker"]) == {"XOM", "CVX"}
    # Ticker filter (used by the agentic layer) works through the whole chain.
    xom = EnergyNews(root=tmp_path).query("2020-03-01", "2020-04-30", tickers=["XOM"])
    assert len(xom) == 2 and (xom["ticker"] == "XOM").all()


def test_english_only_filter():
    arts = [
        {"url": "u1", "title": "t1", "seendate": "20200305T010000Z", "domain": "d1", "language": "Spanish"},
        {"url": "u2", "title": "t2", "seendate": "20200305T010000Z", "domain": "d2", "language": "English"},
    ]
    df = gdelt.articles_to_frame(arts, ticker="CVX", english_only=True)
    assert len(df) == 1 and df.iloc[0]["url"] == "u2"
