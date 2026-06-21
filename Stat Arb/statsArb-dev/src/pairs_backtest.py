"""
Pairs trading backtest class — interface-compatible with src.backtest.bt.

Instead of hedging each stock against its sector ETF, PairsBt:
  1. Finds cointegrated pairs within the sector (via PairsProcessor).
  2. Trades the log-price spread between each pair as a dollar-neutral position.
  3. Reuses bt_tools.get_position_df, calc_port_ret and port_performance
     directly (pair_ret_df has the same Date x column_name shape that bt_tools
     already expects).

Usage
-----
    from src.pairs_backtest import PairsBt

    model = PairsBt(
        prices_file_path="data/xlf/xlf_full_hist_2020_11_20.csv",
        etf_name="xlf",
        st_dt="2015-01-02",
        ed_dt="2020-11-20",
        n_window=60,
        progress=True,
    )
    sharpe, maxdd, endpnl = model.run()
"""

import numpy as np
import pandas as pd

from src.pairs import PairsProcessor
from src.bt_tools import get_position_df, calc_port_ret, port_performance


class PairsBt:
    """Pairs trading backtest, mirroring the bt class interface."""

    # Default thresholds — same as Avellaneda-Lee baseline
    DEFAULT_THRESHOLDS = {'s_bo': -1.25, 's_bc': -0.50,
                          's_so': 1.25,  's_sc': 0.50}

    def __init__(self,
                 prices_file_path: str,
                 etf_name: str,
                 st_dt: str,
                 ed_dt: str,
                 n_window: int = 60,
                 p_value_cutoff: float = 0.05,
                 max_pairs: int = 50,
                 reselect_freq: int = 20,
                 kappa_min: float = None,
                 performance_only: bool = False,
                 progress: bool = False):
        """
        Load prices, run pair selection + OU signal generation.

        Parameters
        ----------
        prices_file_path : Path to sector CSV (wide format, 'Adj. Close' cols)
        etf_name         : Sector ETF ticker (excluded from constituent universe)
        st_dt / ed_dt    : Backtest date range (strings, e.g. '2015-01-02')
        n_window         : Rolling estimation window in trading days (default 60)
        p_value_cutoff   : Engle-Granger p-value threshold (default 0.05)
        max_pairs        : Max cointegrated pairs to trade (default 50)
        reselect_freq    : Days between pair reselections (default 20 ~ monthly)
        kappa_min        : Min OU mean-reversion speed; None = no filter
        performance_only : If True, skip interactive plot in run()
        progress         : Show tqdm bar during signal generation
        """
        self.etf_name = etf_name.upper()
        self.st_dt = st_dt
        self.ed_dt = ed_dt
        self.performance_only = performance_only

        # ---- Load prices (same parsing as bt.__init__) ----
        df = pd.read_csv(prices_file_path, index_col=['Date'],
                         parse_dates=True).sort_index()
        adj_cols = [c for c in df.columns if 'Adj. Close' in c]
        prices_df = df[adj_cols].astype('float64').ffill()
        prices_df.columns = [c.replace(' Adj. Close', '') for c in prices_df.columns]

        self.prices_df = prices_df
        self.ret_df = prices_df.pct_change()

        # ---- Generate pair s-scores ----
        proc = PairsProcessor()
        self.s_scores_df, self.pairs_info, self.hedge_ratios_df = \
            proc.trading_signal_pairs(
                df_prices=prices_df,
                etf_name=self.etf_name,
                n_window=n_window,
                st_dt=st_dt,
                ed_dt=ed_dt,
                p_value_cutoff=p_value_cutoff,
                max_pairs=max_pairs,
                reselect_freq=reselect_freq,
                kappa_min=kappa_min,
                progress=progress,
            )

        # ---- Build pair return series ----
        # Must be done after signal generation (hedge_ratios_df is now available)
        self.pair_ret_df = self._compute_pair_returns()

        # Align pair returns to the backtest window (drop the last row as in bt)
        self.bt_pair_ret_df = self.pair_ret_df.loc[st_dt: ed_dt].iloc[:-1]

        # Expose for callers (mirrors bt attributes)
        self.port_ret = None
        self.pos_long_df = None
        self.pos_short_df = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_pair_returns(self) -> pd.DataFrame:
        """
        Build a (Date x pair_id) DataFrame of dollar-neutral spread returns.

        For a long-spread position on pair (A, B) with hedge ratio beta:
            pair_ret = ret_A  -  beta * ret_B

        A short-spread position is the negative of this; bt_tools handles the
        sign flip internally (it negates ret_df for the short-side position).
        """
        pair_ids = self.s_scores_df.columns.tolist()
        pair_rets = {}

        for pid in pair_ids:
            if pid not in self.pairs_info:
                continue
            ta, tb = self.pairs_info[pid]

            if ta not in self.ret_df.columns or tb not in self.ret_df.columns:
                continue

            ret_a = self.ret_df[ta]
            ret_b = self.ret_df[tb]

            # Use the rolling hedge ratio from signal generation
            beta_series = self.hedge_ratios_df[pid].reindex(self.ret_df.index).ffill()

            pair_rets[pid] = ret_a - beta_series * ret_b

        return pd.DataFrame(pair_rets)

    # ------------------------------------------------------------------
    # Public API (mirrors bt.run)
    # ------------------------------------------------------------------

    def run(self,
            weighting_scheme: str = "equal_weighted",
            sl: float = -0.10,
            long_only: bool = False,
            short_only: bool = False,
            transaction_cost: tuple = (0.0010, 0.0010),
            s_thresholds: dict = None) -> tuple:
        """
        Apply s-score thresholds, build positions, compute portfolio returns.

        Transaction costs are set to 2x the baseline default (10 bps vs 5 bps)
        because each pair trade moves two stocks per leg (4 legs per round-trip).

        Returns
        -------
        (sharpe, maxdd, endpnl)
        Exposes self.port_ret, self.pos_long_df, self.pos_short_df after call.
        """
        s_threshold = s_thresholds if s_thresholds else self.DEFAULT_THRESHOLDS

        # Align s-scores to the backtest window
        s_scores_bt = self.s_scores_df.loc[self.st_dt: self.ed_dt].iloc[:-1]

        # Align pair returns to the same window and column set
        common_pairs = s_scores_bt.columns.intersection(self.bt_pair_ret_df.columns)
        s_scores_bt = s_scores_bt[common_pairs]
        bt_ret = self.bt_pair_ret_df[common_pairs].reindex(s_scores_bt.index)

        # Entry / exit boolean DataFrames (same logic as bt.run lines 51-54)
        long_entry  = s_scores_bt < s_threshold['s_bo']
        long_exit   = s_scores_bt > s_threshold['s_bc']
        short_entry = s_scores_bt > s_threshold['s_so']
        short_exit  = s_scores_bt < s_threshold['s_sc']

        # Position DataFrames — reuse bt_tools directly
        pos_long_df  = get_position_df(long_entry,  long_exit,  bt_ret,  sl_limit=sl)
        pos_short_df = get_position_df(short_entry, short_exit, -bt_ret, sl_limit=sl)

        # Portfolio returns
        epsilon = transaction_cost[0]
        long_port_ret     = calc_port_ret(bt_ret, pos_long_df,  epsilon,
                                          weighting_scheme=weighting_scheme,
                                          direction="long")
        short_port_ret    = calc_port_ret(bt_ret, pos_short_df, epsilon,
                                          weighting_scheme=weighting_scheme,
                                          direction="short")
        cum_short_port_ret = calc_port_ret(bt_ret, pos_short_df, epsilon,
                                           weighting_scheme=weighting_scheme,
                                           direction="cum_short")

        port_ret = pd.concat([
            long_port_ret.rename('long_ret'),
            short_port_ret.rename('short_ret'),
            cum_short_port_ret.rename('cum_short_ret'),
        ], axis=1).fillna(0)

        port_ret, sharpe, maxdd, endpnl = port_performance(port_ret,
                                                            long_only=long_only,
                                                            short_only=short_only)

        # Expose intermediates (mirrors bt attributes)
        self.port_ret    = port_ret
        self.pos_long_df  = pos_long_df
        self.pos_short_df = pos_short_df

        return sharpe, maxdd, endpnl
