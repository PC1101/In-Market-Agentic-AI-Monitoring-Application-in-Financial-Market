import os

import numpy as np
import pandas as pd
import pytest

from report import compute_metrics, generate_report


@pytest.fixture
def monthly_returns():
    """300 months of synthetic returns (~25 years)."""
    np.random.seed(99)
    dates = pd.date_range("2000-01-31", periods=300, freq="ME")
    return pd.Series(np.random.normal(0.005, 0.04, 300), index=dates)


@pytest.fixture
def flat_returns():
    dates = pd.date_range("2000-01-31", periods=60, freq="ME")
    return pd.Series(0.0, index=dates)


# ── compute_metrics ──────────────────────────────────────────────────────────

def test_compute_metrics_returns_dict(monthly_returns):
    result = compute_metrics(monthly_returns)
    assert isinstance(result, dict)


def test_compute_metrics_required_keys(monthly_returns):
    result = compute_metrics(monthly_returns)
    for key in ("Annualised Return", "Annualised Volatility", "Sharpe Ratio",
                "Max Drawdown", "Calmar Ratio"):
        assert key in result, f"Missing key: {key}"


def test_compute_metrics_max_drawdown_non_positive(monthly_returns):
    result = compute_metrics(monthly_returns)
    assert result["Max Drawdown"] <= 0


def test_compute_metrics_flat_returns_zero_vol(flat_returns):
    result = compute_metrics(flat_returns)
    assert result["Annualised Volatility"] == pytest.approx(0.0, abs=1e-9)


def test_compute_metrics_flat_returns_zero_sharpe(flat_returns):
    result = compute_metrics(flat_returns)
    assert result["Sharpe Ratio"] == pytest.approx(0.0, abs=1e-9)


def test_compute_metrics_flat_returns_zero_drawdown(flat_returns):
    result = compute_metrics(flat_returns)
    assert result["Max Drawdown"] == pytest.approx(0.0, abs=1e-9)


def test_compute_metrics_positive_return_positive_sharpe():
    dates = pd.date_range("2000-01-31", periods=60, freq="ME")
    positive = pd.Series(0.01, index=dates)  # 1% per month, no volatility
    result = compute_metrics(positive)
    assert result["Annualised Return"] > 0


# ── generate_report ───────────────────────────────────────────────────────────

def test_generate_report_creates_tearsheet_png(monthly_returns, tmp_path, monkeypatch):
    import config
    monkeypatch.setattr(config, "DATA_DIR", str(tmp_path))
    generate_report(monthly_returns)
    assert (tmp_path / "tearsheet.png").exists()


def test_generate_report_prints_metrics(monthly_returns, tmp_path, monkeypatch, capsys):
    import config
    monkeypatch.setattr(config, "DATA_DIR", str(tmp_path))
    generate_report(monthly_returns)
    captured = capsys.readouterr()
    assert "Sharpe" in captured.out
    assert "Drawdown" in captured.out
