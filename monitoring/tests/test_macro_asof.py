import pandas as pd

from macro.asof import asof_vintage, asof_daily
from macro.fetch_macro import parse_observations


def _vintage_df():
    # UNRATE-style: July-2007 obs first published Aug-3-2007 at 4.6, revised to 4.7 on
    # Sep-7-2007; August obs published Sep-7-2007.
    return pd.DataFrame({
        "date": pd.to_datetime(["2007-07-01", "2007-07-01", "2007-08-01"]),
        "value": [4.6, 4.7, 4.6],
        "realtime_start": pd.to_datetime(["2007-08-03", "2007-09-07", "2007-09-07"]),
        # open-ended vintages: ALFRED's "9999-12-31" overflows datetime64[ns];
        # parse_observations maps it to Timestamp.max - fixtures use a safe far date
        "realtime_end": pd.to_datetime(["2007-09-06", "2262-04-01", "2262-04-01"]),
    })


def test_asof_vintage_uses_first_print_not_revision():
    v = asof_vintage(_vintage_df(), "2007-08-15")
    assert v == {"value": 4.6, "obs_date": "2007-07-01", "published": "2007-08-03"}


def test_asof_vintage_sees_revision_and_new_obs_later():
    v = asof_vintage(_vintage_df(), "2007-09-10")
    assert v["obs_date"] == "2007-08-01" and v["value"] == 4.6


def test_asof_vintage_before_any_release():
    assert asof_vintage(_vintage_df(), "2007-08-01") is None


def test_asof_daily_last_value_on_or_before():
    df = pd.DataFrame({
        "date": pd.to_datetime(["2007-08-03", "2007-08-06", "2007-08-07"]),
        "value": [25.16, 26.25, None],   # None = market holiday / missing print
    })
    v = asof_daily(df, "2007-08-07")
    assert v == {"value": 26.25, "obs_date": "2007-08-06"}
    assert asof_daily(df, "2007-08-01") is None


def test_parse_observations_handles_dots():
    payload = {"observations": [
        {"date": "2007-08-03", "value": "25.16",
         "realtime_start": "2007-08-03", "realtime_end": "9999-12-31"},
        {"date": "2007-08-06", "value": ".",
         "realtime_start": "2007-08-06", "realtime_end": "9999-12-31"},
    ]}
    df = parse_observations(payload)
    assert len(df) == 2
    assert df["value"].isna().iloc[1]
    assert str(df["date"].dtype).startswith("datetime64")
