"""
FinGPT regime stress filter for Avellaneda-Lee stat-arb backtest.

Converts FinGPT sentiment scores (from scripts/compute_stress.py) into a
daily boolean mask that suppresses new trade entries on high-stress days.

Stress series CSV format:
  date        : trading date
  stress      : float [0, 1]; already shifted by 1 business day in build()
  n_headlines : int; informational (how many headlines were scored that day)

LOOK-AHEAD GUARDS:
  Data:  Stress on trade day T uses headlines from T-1 (previous business
         day). Enforced in build() via the prev_date lookup.
  GPT:   FinGPT carries training-data look-ahead risk. See fingpt_sentiment.py
         and the diagnostic chart in results/.

NaN handling:
  NaN stress -> True (entry allowed). Arises when FNSPID has no coverage for
  a date or ticker. The backtest runs unmodified for those dates.
"""
import os

import numpy as np
import pandas as pd


class RegimeFilter:

    def build(
        self,
        news_by_date: dict,
        model,
        bt_dates: pd.DatetimeIndex,
        output_path: str,
    ) -> pd.Series:
        """
        Pre-compute the stress series for every backtest trading date.

        Parameters
        ----------
        news_by_date : dict
            {datetime.date -> list[str]} from FNSPIDLoader.load_titles_by_date().
            Keys are PUBLICATION dates (not shifted).
        model : FinGPTSentimentEngine
            Loaded and ready-to-infer sentiment engine.
        bt_dates : pd.DatetimeIndex
            Every trading date in the backtest return series.
        output_path : str
            Where to save the stress CSV (e.g. 'data/news/stress_xlf.csv').

        Returns
        -------
        pd.Series
            Index = bt_dates, values = float stress [0,1] or NaN.
            Stress on date T reflects headlines from PREVIOUS business day (T-1).
        """
        print(f"[RegimeFilter] Scoring headlines for {len(bt_dates)} trading days...")

        raw_stress = {}
        n_headlines_log = {}
        logged_every = max(1, len(bt_dates) // 20)   # print ~20 progress lines

        for i, date in enumerate(bt_dates):
            # LOOK-AHEAD GUARD: use previous business day's headlines for day T
            prev_date = (date - pd.offsets.BDay(1)).date()
            headlines = news_by_date.get(prev_date, [])

            raw_stress[date] = model.daily_stress(headlines)
            n_headlines_log[date] = len(headlines)

            if (i + 1) % logged_every == 0 or (i + 1) == len(bt_dates):
                stress_val = raw_stress[date]
                stress_str = f"{stress_val:.2f}" if not np.isnan(stress_val) else "NaN"
                print(
                    f"  [{i+1}/{len(bt_dates)}] {date.date()}  "
                    f"headlines={len(headlines)}  stress={stress_str}"
                )

        stress = pd.Series(raw_stress, name="stress")
        n_headlines = pd.Series(n_headlines_log, name="n_headlines")
        stress.index.name = "date"
        n_headlines.index.name = "date"

        out_df = pd.concat([stress, n_headlines], axis=1)
        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        out_df.to_csv(output_path)

        nan_days  = stress.isna().sum()
        above_thr = (stress > 0.40).sum()
        print(f"[RegimeFilter] Stress series saved -> {output_path}")
        print(f"  Total days       : {len(stress)}")
        print(f"  No-headline days : {nan_days} (NaN stress -> entry allowed)")
        print(f"  Days stress>0.40 : {above_thr} "
              f"({above_thr / max(stress.notna().sum(), 1):.1%} of days with data)")

        return stress

    @staticmethod
    def load(path: str) -> pd.Series:
        """
        Load a pre-computed stress series from CSV.

        Returns
        -------
        pd.Series
            DatetimeIndex, values = float stress [0, 1].
            Already shifted (T-1 look-ahead guard applied by build()).
        """
        df = pd.read_csv(path, index_col="date", parse_dates=True)
        if "stress" not in df.columns:
            raise ValueError(
                f"Stress CSV at '{path}' must have a 'stress' column. "
                f"Found: {list(df.columns)}"
            )
        return df["stress"].rename("stress")

    @staticmethod
    def to_mask(stress: pd.Series, threshold: float = 0.40) -> pd.Series:
        """
        Convert a stress series to a boolean entry-allowed mask.

        True  -> stress below threshold -> allow new long/short entries.
        False -> stress at/above threshold -> suppress new entries.
        NaN   -> True (no FNSPID coverage -> no suppression).

        Parameters
        ----------
        stress : pd.Series
        threshold : float
            Default 0.40 (suppress when ≥40% of day's headlines net-negative).

        Returns
        -------
        pd.Series[bool]
        """
        mask = stress < threshold
        mask = mask.fillna(True)
        return mask.rename("entry_allowed")
