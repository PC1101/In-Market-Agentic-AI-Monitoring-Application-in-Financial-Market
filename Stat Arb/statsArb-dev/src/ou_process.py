import numpy as np
from statsmodels.api import OLS
from statsmodels.tsa.tsatools import add_trend
import statsmodels
import matplotlib.pyplot as plt
import pandas as pd
try:
    from tqdm import tqdm
except ImportError:  ## progress bar is optional
    def tqdm(iterable=None, **kwargs):
        return iterable
class OUProcess(object):
    '''
    This module tries to model OU-process/ AR(1) process lay out
    in Lecture 6 Quant by Marco Avellaneda, and designed to be
    reusable even when the X_t is not specified,

    OU-process is defined as

    dX_t = kappa * (m - X_t) dt  + sigma dW_t




    '''
    def __init__(self):
        '''


        '''

        ## refitted every day for trading_signal_group
        self.single_stock_fitted_s = {}
        self.single_stock_params ={}
        self.group_params = {}
        pass

    def _fit_MLE(self, x, dt=1.0):
        '''
        MLE of OU-process
        :param x: mean-reverted X_t, a numpy array
        :return: empircal params of fitted x
        '''
        n = len(x)

        sum_x = np.sum(x[:-1])
        sum_y = np.sum(x[1:])
        sum_xx = np.sum(np.square(x[:-1]))
        sum_xy = np.sum(x[:-1]*x[1:])
        sum_yy = np.sum(np.square(x[1:]))
        m = (sum_y * sum_xx - sum_x * sum_xy) / (n*(sum_xx - sum_xy) - (sum_x**2 - sum_x*sum_y))
        kappa = (-1/dt) * np.log((sum_xy - m*sum_x - m*sum_y + n * (m**2)) / (sum_xx - 2*m*sum_x + n*(m**2)))
        alpha = np.e**(-kappa*dt)

        # sigma_hat_squared = (1/n)*(sum_yy - 2*alpha*sum_xy +
        #                            (alpha**2)*sum_xx -
        #                            2*m*(1-alpha)*(sum_y - alpha*sum_x) +
        #                            n*(m**2)*((1-alpha)**2)
        #                            )
        # sigma_squared = sigma_hat_squared * (2*kappa) / (1 - alpha**2)
        # print(f"sigma_hat_squared: {sigma_hat_squared}")


        ## theta is m, mu is kappa

        sig_term_a = sum_yy - 2*np.e**(-kappa*dt) * sum_xy
        sig_term_b = np.e**(-2*kappa*dt) * sum_xx
        sig_term_c = -2*m*(1 - np.e**(-kappa*dt)) * (sum_y - np.e**(-kappa *dt)*sum_x)
        sig_term_d = n*(m**2)*((1 - np.e**(-kappa*dt))**2)

        sigma_squared = (2*kappa/ (n*(1 - np.e**(-2*kappa*dt)))) * (sig_term_a + sig_term_b + sig_term_c + sig_term_d)

        self.params = {'m':m, 'kappa': kappa, 'sig': np.sqrt(sigma_squared)}


    def _fit_OLS_group(self, single_stock_daily_returns, etf_daily_returns, pca_ret, n_window=60, defactoring='etf'):
        '''
        fitting the model in the paper
        :param single_stock_daily_price:
        :param etf_daily_price:
        :param defactoring: method to defactoring single stock returns
        :return:
        '''
        # dt = 60.0/252
        dt =1.0/ 252

        assert single_stock_daily_returns.shape[0] == etf_daily_returns.shape[0] == n_window

        ## Guard: stocks that had not started trading yet at the start of the
        ## backtest produce NaN returns over the window. Feeding those into the OU
        ## regression makes statsmodels' SVD fail ("SVD did not converge"), so
        ## return NaN params (-> NaN s-score -> no signal) for non-finite windows.
        factor_ret = etf_daily_returns if defactoring == 'etf' else pca_ret
        if not (np.isfinite(np.asarray(single_stock_daily_returns, dtype='float64')).all()
                and np.isfinite(np.asarray(factor_ret, dtype='float64')).all()):
            return {'m': np.nan, 'kappa': np.nan, 'sig': np.nan, 'sig_eq': np.nan,
                    'beta': np.nan, 'alpha_trend': np.nan, 'a_OU': np.nan, 'b_OU': np.nan}

        ## Defactoring with ETF

        defact_X = add_trend(etf_daily_returns, trend='t') if defactoring=='etf' else add_trend(pca_ret, trend='t')
        # reg_ret = OLS(single_stock_daily_returns, add_trend(etf_daily_returns, trend='t'), hasconst=False).fit()
        reg_ret = OLS(single_stock_daily_returns, defact_X, hasconst=False).fit()



        beta = reg_ret.params #.values[0]
        alpha = beta.pop('trend')


        epsilon = reg_ret.resid


        ## paper Appendix p. 45
        x_60 = 0.00
        x_t = np.array([*np.cumsum(epsilon)[:-1], x_60])

        ## Estimation of OU models
        ## Avellaneda paper notation, not the same in lecture notes
        ## X_{n+1} = a + b X_n  + v_{n+1}

        ## where v_n i.i.d. N(0, sig**2 ((1-np.e**(-2kdt))/2k)
        x_const = statsmodels.tools.add_constant(x_t[:-1])

        reg_OU = OLS(x_t[1:], x_const, hasconst=False).fit()

        a, b = reg_OU.params

        kappa = (1/dt) * (np.log(1/b))
        m = a / (1 -b)

        sig = np.sqrt(np.var(reg_OU.resid) * 2 * kappa / (1 - b**2))
        sig_eq = np.std(reg_OU.resid)/(np.sqrt(1 - b**2))

        ## the long term equilibrium mean: m
        ## the long term equilibrium sig: sig_eq = sigma * np.sqrt(1/(2*kappa))  p.16 in the paper


        ##
        params = {'m': m,
                  'kappa': kappa,
                  'sig': sig,
                  'sig_eq': sig_eq,
                  'beta':beta,
                  'alpha_trend': alpha,
                  'a_OU': a,
                  'b_OU': b
                  }


        ## in the paper s is redefined as -m/sig_eq rather than (x_t - m)/ sog_eq
        ## this is because x_t = x_60 = 0
        ## OU process is re-estimated/ refit everyday to give a daily s score

        ## theoretical s
        # self.fitted_s = pd.Series((x_t - m) / sig_eq, single_stock_daily_price.index)

        ## s redefined in the paper, before readjusted for m_bar
        # cur_fitted_s = (-a * np.sqrt(1-b**2))/ ((1-b)*np.std(reg_OU.resid))


        return params
    def _adj_m_bar(self, kappa_min=None):
        '''
        :param kappa_min: if set, only keep s-scores for stocks whose OU mean-
            reversion speed kappa exceeds this. Avellaneda-Lee (p.27) only trade
            names that revert fast enough relative to the estimation window
            (1/kappa well below the window); slow / poorly-fitted names are
            dropped (s-score -> NaN -> no trade) to avoid trading noise.
        :return: adjusted s score for all single stock
        '''
        ## p.46 in the paper
        cur_params = pd.DataFrame.from_dict(self.group_params)
        ## Center the equilibrium across the tradable universe only: skip NaN/inf
        ## tickers (skipped fits, late IPOs) so they don't contaminate the mean
        ## and wipe out every stock's s-score for the day.
        equilibrium = (cur_params.loc['a_OU'] / (1 - cur_params.loc['b_OU'])).replace([np.inf, -np.inf], np.nan)
        m_bar = equilibrium - equilibrium.mean()

        ## modified s to adjust for drift/ alpha trend outlined in p.22
        s_score = - m_bar/ cur_params.loc['sig_eq'] - (cur_params.loc['alpha_trend']/(cur_params.loc['kappa']*cur_params.loc['sig_eq']))

        ## Trade only fast-reverting, well-fitted names
        if kappa_min is not None:
            s_score = s_score.where(cur_params.loc['kappa'] > kappa_min)

        return s_score

    def trading_signal_group(self,df_prices, etf_name, pca_ret, n_window=60, st_dt='2006-01-03', ed_dt='2007-12-03', defactoring='etf', kappa_min=None, progress=False):
        '''
        following Appendix of Avellaneda's paper, need of centering with bracketed m
        :param df: etf and single stock prices
        :param etf_name: name of the etf that includes single stocks
        :param n_window: size of estimation/ rolling window
        :param defactoring: method of defactoring single stock returns, either 'etf', 'full_mkt_pca' or 'industry_pca'
        :return:
        '''
        df_ret = df_prices.pct_change()
        etf_ret = df_ret.pop(etf_name)

        st_idx, ed_idx = df_ret.index.get_loc(st_dt), df_ret.index.get_loc(ed_dt)

        s_score_df = pd.DataFrame()

        pca_ret = df_ret.merge(pca_ret, how='left', left_index=True, right_index=True)[pca_ret.columns.tolist()].fillna(0)


        day_iter = range(st_idx, ed_idx)
        if progress:
            day_iter = tqdm(day_iter, desc=f"OU fit {etf_name}", unit="day", total=ed_idx - st_idx)

        for i in day_iter:
            for ticker in df_ret.columns:
                self.group_params[ticker] = self._fit_OLS_group(df_ret[ticker].iloc[i - n_window: i],
                                                                etf_ret.iloc[i - n_window: i],
                                                                pca_ret.iloc[i - n_window: i],
                                                                n_window=n_window,
                                                                defactoring=defactoring)


            ## DataFrame.append was removed in pandas 2.0; concat the daily s-score row instead
            s_score_row = self._adj_m_bar(kappa_min=kappa_min).rename(df_prices.iloc[i].name).to_frame().T
            s_score_df = pd.concat([s_score_df, s_score_row])

        return s_score_df.rename_axis(df_prices.index.name)

    def trading_signal_s(self, lookback_ret, etf_name, n_window=60):
        lookback_etf_ret = lookback_ret.pop(etf_name)
        for ticker in lookback_ret.columns:
            self.group_params[ticker] = self._fit_OLS_group(lookback_ret[ticker],
                                                            lookback_etf_ret,
                                                            n_window=n_window)


        return self._adj_m_bar().rename(lookback_ret.iloc[-1].name)





