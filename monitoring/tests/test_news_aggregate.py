import numpy as np
import pandas as pd

from news.aggregate import daily_signals
from news.filter import filter_news


def _mk_news(dates, titles):
    return pd.DataFrame({
        "date": pd.to_datetime(dates), "ticker": "", "title": titles,
        "summary": "", "publisher": "", "url": "",
    })


def test_daily_signals_counts_and_index():
    all_news = _mk_news(
        ["2007-08-01", "2007-08-01", "2007-08-03"],
        ["Calm day", "Another story", "Market selloff and panic"],
    )
    sig = daily_signals(all_news, filter_news(all_news))
    assert sig.loc["2007-08-01", "n_articles"] == 2
    assert sig.loc["2007-08-02", "n_articles"] == 0     # calendar-continuous
    assert sig.loc["2007-08-03", "n_risk"] == 1
    assert sig.loc["2007-08-03", "intensity"] == 2       # selloff + panic


def test_intensity_z_is_causal_and_spikes():
    # 80 quiet days then one very risky day
    dates = pd.date_range("2007-05-01", periods=81, freq="D")
    titles = ["Ordinary business story"] * 80 + \
             ["Meltdown: margin calls, liquidation, panic selloff"]
    all_news = _mk_news(dates, titles)
    sig = daily_signals(all_news, filter_news(all_news), baseline_days=60)
    last, prev = dates[-1], dates[-2]
    assert sig.loc[last, "intensity_z"] > 3
    # baseline for day t must exclude day t (shifted) => quiet prev day has small |z|
    assert abs(sig.loc[prev, "intensity_z"]) < 1 or np.isnan(sig.loc[prev, "intensity_z"])


def test_z_nan_during_warmup():
    all_news = _mk_news(["2007-08-01"], ["Calm"])
    sig = daily_signals(all_news, filter_news(all_news), baseline_days=60)
    assert np.isnan(sig["intensity_z"]).all()
