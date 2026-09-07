"""Energy commodity macro provider: as-of-correct WTI/Brent/natgas block (no network)."""
import pandas as pd

from providers.energy.macro import EnergyMacro, ENERGY_SERIES


def _write(dir_, sid, rows):
    """rows: list[(date_str, value)] -> parquet the asof_daily reader expects."""
    df = pd.DataFrame(rows, columns=["date", "value"])
    df["date"] = pd.to_datetime(df["date"])
    df.to_parquet(dir_ / f"{sid}.parquet")


def test_context_returns_asof_last_print(tmp_path):
    _write(tmp_path, "DCOILWTICO", [("2020-03-06", 41.0), ("2020-03-09", 31.0), ("2020-04-20", -37.0)])
    _write(tmp_path, "DCOILBRENTEU", [("2020-03-06", 45.0), ("2020-03-09", 34.0)])
    _write(tmp_path, "DHHNGSP", [("2020-03-06", 1.8)])

    ctx = EnergyMacro(data_dir=tmp_path).context("2020-03-10")
    # WTI: last print on/before 2020-03-10 is the 03-09 value, NOT the future 04-20 crash.
    assert ctx["wti_crude"]["value"] == 31.0
    assert ctx["brent_crude"]["value"] == 34.0
    assert ctx["henry_hub_natgas"]["value"] == 1.8


def test_brent_wti_spread_derived(tmp_path):
    _write(tmp_path, "DCOILWTICO", [("2020-03-09", 31.0)])
    _write(tmp_path, "DCOILBRENTEU", [("2020-03-09", 34.0)])
    ctx = EnergyMacro(data_dir=tmp_path).context("2020-03-10")
    assert ctx["brent_wti_spread"] == 3.0


def test_no_lookahead(tmp_path):
    # Value only exists AFTER the decision date -> must not appear.
    _write(tmp_path, "DCOILWTICO", [("2020-04-20", -37.0)])
    ctx = EnergyMacro(data_dir=tmp_path).context("2020-03-10")
    assert "wti_crude" not in ctx


def test_missing_series_omitted_gracefully(tmp_path):
    _write(tmp_path, "DCOILWTICO", [("2020-03-09", 31.0)])
    ctx = EnergyMacro(data_dir=tmp_path).context("2020-03-10")
    assert ctx["wti_crude"]["value"] == 31.0
    assert "henry_hub_natgas" not in ctx  # not written -> omitted, no crash


def test_energy_series_covers_wti_brent_natgas():
    assert "DCOILWTICO" in ENERGY_SERIES
    assert "DCOILBRENTEU" in ENERGY_SERIES
    assert "DHHNGSP" in ENERGY_SERIES
