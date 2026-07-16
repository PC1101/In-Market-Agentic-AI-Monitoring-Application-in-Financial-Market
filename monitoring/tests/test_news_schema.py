import pytest

from agentic.schemas import validate_news_flags, NewsFlags, SchemaError, OverallRisk


def _valid():
    return {
        "overall_risk": "HIGH",
        "risk_flags": [{"flag": "forced_deleveraging", "evidence": "margin-call headlines"}],
        "narrative": "Multiple reports of quant funds unwinding positions.",
        "confidence": 0.7,
        "as_of": "2007-08-09",
        "n_articles": 12,
    }


def test_valid_passes():
    nf = validate_news_flags(_valid())
    assert isinstance(nf, NewsFlags)
    assert nf.overall_risk == OverallRisk.HIGH
    assert nf.risk_flags[0]["flag"] == "forced_deleveraging"


def test_bad_risk_enum_rejected():
    bad = _valid() | {"overall_risk": "APOCALYPTIC"}
    with pytest.raises(SchemaError):
        validate_news_flags(bad)


def test_missing_field_rejected():
    bad = _valid()
    del bad["narrative"]
    with pytest.raises(SchemaError):
        validate_news_flags(bad)


def test_risk_flags_must_be_flag_evidence_dicts():
    bad = _valid() | {"risk_flags": ["just a string"]}
    with pytest.raises(SchemaError):
        validate_news_flags(bad)


def test_empty_flags_ok_for_low_risk():
    ok = _valid() | {"overall_risk": "LOW", "risk_flags": []}
    assert validate_news_flags(ok).risk_flags == []


def test_confidence_range_enforced():
    with pytest.raises(SchemaError):
        validate_news_flags(_valid() | {"confidence": 1.4})
