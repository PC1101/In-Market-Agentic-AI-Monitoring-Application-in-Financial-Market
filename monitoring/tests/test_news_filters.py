"""Stress-lexicon filter tests: categories, benign pass-through, case handling."""

import pytest

from news.filters import classify_headline, split_records, FINANCIAL_SECTIONS


# ---- classify_headline ---------------------------------------------------------

@pytest.mark.parametrize("text,expected_cat", [
    ("Hedge funds face margin calls", "liquidity_credit"),
    ("Bank nears bankruptcy after write-downs", "liquidity_credit"),
    ("Stocks plunge in panic sell-off", "market_stress"),
    ("Volatility spikes as bear market fears grow", "market_stress"),
    ("Quant funds liquidate amid redemptions", "fund_quant_stress"),
    ("Forced selling hits leveraged funds", "fund_quant_stress"),
    ("Recession fears deepen after downgrade", "macro_policy"),
    ("Subprime mortgage-backed securities under pressure", "macro_policy"),
])
def test_category_hits(text, expected_cat):
    assert expected_cat in classify_headline(text)


@pytest.mark.parametrize("text", [
    "Retailers report steady quarterly earnings",
    "Tech company announces dividend increase",
    "Small business optimism edges higher",
    "",
])
def test_benign_headlines_no_hit(text):
    assert classify_headline(text) == []


def test_case_insensitive():
    assert classify_headline("MARKETS CRASH IN PANIC") == classify_headline(
        "markets crash in panic")
    assert classify_headline("Sell-Off Deepens") != []


def test_multi_category():
    cats = classify_headline("Hedge funds hit by margin calls as stocks plunge")
    assert "liquidity_credit" in cats
    assert "fund_quant_stress" in cats
    assert "market_stress" in cats
    assert cats == sorted(cats)


def test_none_text_safe():
    assert classify_headline(None) == []


# ---- split_records ---------------------------------------------------------------

def _rec(headline, section=None, abstract=None):
    return {"timestamp": "2010-06-01T09:00:00", "headline": headline,
            "abstract": abstract, "source": "t", "tickers": [], "section": section}


def test_split_matched_subset_of_candidates():
    records = [
        _rec("Stocks plunge in panic", section="Business"),
        _rec("Steady quarterly earnings", section="Business"),
        _rec("Airline expands routes", section="Travel"),   # non-financial, no hit
        _rec("Wire headline with no section"),               # sectionless candidate
    ]
    candidates, matched = split_records(records)
    assert len(matched) == 1
    assert matched[0]["categories"] == ["market_stress"]
    assert matched[0]["n_hit_categories"] == 1
    # plunge + earnings + sectionless; travel story excluded
    assert len(candidates) == 3
    assert all(m in candidates for m in matched)


def test_keyword_hit_in_any_section_qualifies():
    records = [_rec("Markets crash worldwide", section="Front Page")]
    candidates, matched = split_records(records)
    assert len(matched) == 1 and len(candidates) == 1


def test_abstract_is_searched():
    records = [_rec("Calm headline", section="Business",
                    abstract="but the abstract mentions a credit crunch")]
    _, matched = split_records(records)
    assert len(matched) == 1
    assert "liquidity_credit" in matched[0]["categories"]


def test_financial_sections_lowercase():
    # guard against accidentally adding mixed-case entries to the allow-list
    assert all(s == s.lower() for s in FINANCIAL_SECTIONS)
