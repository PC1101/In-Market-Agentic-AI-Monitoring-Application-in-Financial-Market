"""
Pairs trading signal generation for the Avellaneda-Lee stat-arb framework.

This module handles:
  1. Pair selection — rolling Engle-Granger cointegration test to find valid
     pairs within a sector universe.
  2. Spread OU fitting — AR(1) regression on the log-price spread to extract
     kappa, m, sig_eq (same maths as OUProcess._fit_OLS_group, but without the
     defactoring step because the spread itself is the mean-reverting process).
  3. S-score generation — rolling daily s-scores for every selected pair,
     assembled into a (Date x pair_id) DataFrame that PairsBt can consume
     directly via bt_tools.

Pair ids are formatted as "TICKA_TICKB" (alphabetically ordered so the same
pair is always identified the same way regardless of which leg triggered it).
"""

import itertools

import numpy as np
import pandas as pd
import statsmodels.tools
from statsmodels.api import OLS
from statsmodels.tsa.stattools import coint

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable=None, **kwargs):
        return iterable


class PairsProcessor:
    """Rolling pair selection and spread s-score generation."""

    def __init__(self):
        self.pair_params = {}  # (tickerA, tickerB) -> param_dict (last fitted)

    # ------------------------------------------------------------------
    # 1. Pair selection
    # ------------------------------------------------------------------

    @staticmethod
    def _engle_granger_test(log_prices_a: np.ndarray, log_prices_b: np.ndarray):
        """
        Run Engle-Granger cointegration test on two log-price series.

        Step 1  OLS:  log(P_a) = alpha + beta * log(P_b) + epsilon
        Step 2  ADF on epsilon  (via statsmodels.tsa.stattools.coint)

        Returns
        -------
        (beta_hedge, p_value, residuals)
        beta_hedge : float  — cointegrating coefficient
        p_value    : float  — MacKinnon p-value from ADF on residual
        residuals  : np.ndarray

        Returns (np.nan, 1.0, None) when inputs are non-finite so the pair is
        never selected (same guard pattern as ou_process.py:94-97).
        """
        if not (np.isfinite(log_prices_a).all() and np.isfinite(log_prices_b).all()):
            return np.nan, 1.0, None

        try:
            _, p_value, _ = coint(log_prices_a, log_prices_b, trend='c')
        except Exception:
            return np.nan, 1.0, None

        # Estimate hedge ratio via OLS: log_P_a = alpha + beta * log_P_b
        X = statsmodels.tools.add_constant(log_prices_b)
        reg = OLS(log_prices_a, X).fit()
        beta_hedge = reg.params[1]
        residuals = reg.resid

        return beta_hedge, p_value, residuals

    def select_pairs(self, log_prices_df: pd.DataFrame,
                     p_value_cutoff: float = 0.05,
                     max_pairs: int = 50):
        """
        Test all C(N, 2) combinations in log_prices_df for cointegration.

        Parameters
        ----------
        log_prices_df  : DataFrame (n_window x N_tickers) of log-prices
        p_value_cutoff : Engle-Granger MacKinnon p-value threshold
        max_pairs      : Maximum pairs to return (sorted by p-value ascending)

        Returns
        -------
        list of (tickerA, tickerB, beta_hedge, p_value)
        tickerA < tickerB alphabetically to ensure consistent pair ids.
        """
        tickers = log_prices_df.columns.tolist()
        candidates = []

        for ta, tb in itertools.combinations(tickers, 2):
            a = log_prices_df[ta].values
            b = log_prices_df[tb].values
            beta, pval, _ = self._engle_granger_test(a, b)
            if pval < p_value_cutoff:
                # Alphabetical order so pair_id is deterministic
                if ta > tb:
                    ta, tb = tb, ta
                    beta = 1.0 / beta if beta != 0 else np.nan
                candidates.append((ta, tb, beta, pval))

        # Sort by p-value, take top max_pairs
        candidates.sort(key=lambda x: x[3])
        return candidates[:max_pairs]

    # ------------------------------------------------------------------
    # 2. Spread OU fitting
    # ------------------------------------------------------------------

    def _fit_spread_ou(self, spread: np.ndarray, dt: float = 1.0 / 252):
        """
        Fit an OU process on a spread time series via AR(1) regression.

        Adapts the AR(1) block from OUProcess._fit_OLS_group (lines 123-133):

            X_{t+1} = a + b * X_t + v_{t+1}
            kappa   = (1/dt) * ln(1/b)
            m       = a / (1 - b)
            sig_eq  = std(resid) / sqrt(1 - b^2)

        Key difference from the baseline: NO defactoring step — the spread IS
        the mean-reverting process and we regress directly on it.

        Returns
        -------
        dict with keys {m, kappa, sig_eq, a_OU, b_OU}
        Returns NaN params when the fit fails or inputs are non-finite.
        """
        nan_params = {'m': np.nan, 'kappa': np.nan, 'sig_eq': np.nan,
                      'a_OU': np.nan, 'b_OU': np.nan}

        if not np.isfinite(spread).all() or len(spread) < 3:
            return nan_params

        try:
            X_const = statsmodels.tools.add_constant(spread[:-1])
            reg = OLS(spread[1:], X_const).fit()
            a, b = reg.params

            if b <= 0 or b >= 1:          # explosive or non-stationary AR(1)
                return nan_params

            kappa = (1.0 / dt) * np.log(1.0 / b)
            m = a / (1.0 - b)
            sig_eq = np.std(reg.resid) / np.sqrt(1.0 - b ** 2)

            return {'m': m, 'kappa': kappa, 'sig_eq': sig_eq, 'a_OU': a, 'b_OU': b}

        except Exception:
            return nan_params

    # ------------------------------------------------------------------
    # 3. Main rolling signal loop
    # ------------------------------------------------------------------

    def trading_signal_pairs(self,
                             df_prices: pd.DataFrame,
                             etf_name: str,
                             n_window: int = 60,
                             st_dt: str = '2007-01-03',
                             ed_dt: str = '2015-01-02',
                             p_value_cutoff: float = 0.05,
                             max_pairs: int = 50,
                             reselect_freq: int = 20,
                             kappa_min: float = None,
                             progress: bool = False):
        """
        Generate daily pair s-scores over the backtest period.

        For each day:
          - Every `reselect_freq` days, re-run Engle-Granger on the trailing
            n_window of log-prices to identify the top cointegrated pairs.
          - Every day, for each active pair, construct the log-price spread,
            fit AR(1) / OU on the trailing n_window, and compute s-score.

        Parameters
        ----------
        df_prices      : DataFrame (full history, Date x Ticker) Adj. Close prices
        etf_name       : ETF column to exclude from constituent universe
        n_window       : Rolling estimation window in trading days (default 60)
        st_dt / ed_dt  : Backtest date range
        p_value_cutoff : Engle-Granger p-value cutoff for pair selection
        max_pairs      : Max pairs to hold per reselection
        reselect_freq  : Days between pair reselections (default 20 ~ monthly)
        kappa_min      : If set, NaN out s-scores for pairs reverting slower
                         than this (annual rate). Same filter as baseline.
        progress       : Show tqdm progress bar.

        Returns
        -------
        s_scores_df    : DataFrame (Date x pair_id) — daily s-scores
        pairs_info     : dict  pair_id -> (tickerA, tickerB)
        hedge_ratios_df: DataFrame (Date x pair_id) — hedge ratio beta on each day
        """
        # Drop ETF column to get constituent tickers only
        constituent_prices = df_prices.drop(columns=[etf_name], errors='ignore')
        log_prices = np.log(constituent_prices.replace(0, np.nan))

        st_idx = df_prices.index.get_loc(st_dt)
        ed_idx = df_prices.index.get_loc(ed_dt)

        s_score_rows = []
        hedge_ratio_rows = []
        date_index = []

        current_pairs = []  # list of (tickerA, tickerB, beta_hedge, pval)
        pairs_info = {}     # pair_id -> (tickerA, tickerB)

        day_iter = range(st_idx, ed_idx)
        if progress:
            day_iter = tqdm(day_iter, desc=f"Pairs signal {etf_name.upper()}",
                            unit="day", total=ed_idx - st_idx)

        for rel_i, i in enumerate(range(st_idx, ed_idx)):
            window_log = log_prices.iloc[i - n_window: i]

            # ---- Pair reselection every reselect_freq days ----
            if rel_i % reselect_freq == 0:
                current_pairs = self.select_pairs(
                    window_log,
                    p_value_cutoff=p_value_cutoff,
                    max_pairs=max_pairs,
                )
                # Update pairs_info registry
                for ta, tb, _, _ in current_pairs:
                    pid = f"{ta}_{tb}"
                    pairs_info[pid] = (ta, tb)

            # ---- Daily OU fit on each active pair ----
            day_s = {}
            day_beta = {}
            date_val = df_prices.index[i]

            for ta, tb, beta_hedge, _ in current_pairs:
                pid = f"{ta}_{tb}"

                if ta not in window_log.columns or tb not in window_log.columns:
                    continue

                spread_window = (window_log[ta] - beta_hedge * window_log[tb]).values

                params = self._fit_spread_ou(spread_window, dt=1.0 / 252)

                m = params['m']
                kappa = params['kappa']
                sig_eq = params['sig_eq']

                if not np.isfinite(m) or not np.isfinite(sig_eq) or sig_eq == 0:
                    day_s[pid] = np.nan
                    day_beta[pid] = beta_hedge
                    continue

                # kappa filter — same logic as OUProcess._adj_m_bar
                if kappa_min is not None and kappa < kappa_min:
                    day_s[pid] = np.nan
                    day_beta[pid] = beta_hedge
                    continue

                # Current spread value (last observation in window)
                spread_current = spread_window[-1]
                s = (spread_current - m) / sig_eq

                day_s[pid] = s
                day_beta[pid] = beta_hedge

            s_score_rows.append(day_s)
            hedge_ratio_rows.append(day_beta)
            date_index.append(date_val)

        s_scores_df = pd.DataFrame(s_score_rows, index=date_index)
        s_scores_df.index.name = 'Date'

        hedge_ratios_df = pd.DataFrame(hedge_ratio_rows, index=date_index)
        hedge_ratios_df.index.name = 'Date'

        return s_scores_df, pairs_info, hedge_ratios_df
