"""
run_al_statarb.py — Avellaneda-Lee PCA stat arb for ASX 200 C&P (Norgate).

Replicates the AL_PCA strategy from Stat Arb/statsArb-dev/ on the Australian
market using Norgate's point-in-time S&P ASX 200 C&P universe.

Strategy:
  Defactoring : 15 PCA eigen-portfolios (rolling 60-day correlation matrix).
  Signal      : OU residual s-score, re-estimated daily per stock.
  Entry       : |s| > 1.25  (long if s < -1.25, short if s > +1.25).
  Exit        : |s| < 0.50  (or stop-loss at -10%).
  Filter      : Only trade stocks with OU mean-reversion speed kappa > 8.4/yr
                (i.e. 1/kappa < ~30 trading days).
  Portfolio   : Equal-weighted across open positions.
  Cost        : 10 bps per side.

The OU process follows Avellaneda & Lee (2010), Appendix.  For each stock i
on day t:
  1. Regress stock return on PCA factors + linear trend (OLS, 60-day window).
  2. Cumulate OLS residuals to get the co-integrated X_t process (pinned at 0).
  3. Fit AR(1) to X_t to get (a, b), convert to OU params (kappa, m, sig_eq).
  4. Cross-sectionally centre equilibrium levels: s_i = -(m_bar_i)/sig_eq_i
     minus a drift adjustment.

Outputs (AusEquities/results/al_statarb/):
  equity_curve.csv   — daily [long_ret, short_ret, cum_ret, equity, drawdown]
  trades.csv         — trade blotter [ticker, side, entry, exit, days, return]
  performance.png    — 2×2 dashboard
"""

import os
import sys
import logging

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.tsatools import add_trend
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kw):   # type: ignore[misc]
        return it

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import data_norgate as dn   # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

# ── parameters ────────────────────────────────────────────────────────────────
N_WINDOW    = 60             # rolling OU estimation window (trading days)
N_PCA       = 15             # PCA eigen-portfolios for defactoring
DT          = 1.0 / 252      # time step (daily)

S_BO        = -1.25          # buy-open  (go long when s < S_BO)
S_BC        = -0.50          # buy-close (exit long when s > S_BC)
S_SO        =  1.25          # sell-open (go short when s > S_SO)
S_SC        =  0.50          # sell-close (exit short when s < S_SC)
SL_LIMIT    = -0.10          # per-position stop-loss (cumulative return)
KAPPA_MIN   = 252.0 / 30.0  # minimum OU reversion speed (~8.4/yr)

# ASX tiered transaction costs (per trade, both entry and exit)
MIN_BROKERAGE  = 10.00       # $10 minimum brokerage (competitive online broker)
BROKERAGE_RATE = 0.0011      # 0.11% of trade value
MARKET_IMPACT  = 0.0007      # ~7 bps market impact for ASX 200 liquid stocks
INITIAL_VALUE  = 1_000_000   # $1M starting capital

# Short borrow cost
BORROW_RATE_ANNUAL = 0.01    # 1% p.a. for ASX 200 liquid stocks
BORROW_RATE_DAILY  = BORROW_RATE_ANNUAL / 252

OUT_DIR = os.path.join(HERE, "results", "al_statarb")


# ─────────────────────────────────────────────────────────────────────────────
# OU fitting (inlined from statsArb-dev/src/ou_process.py)
# ─────────────────────────────────────────────────────────────────────────────

def _fit_ou_group(
    stock_ret: np.ndarray,
    pca_ret: pd.DataFrame,
    dt: float = DT,
) -> dict:
    """
    Fit Avellaneda-Lee OU model for one (stock, day) pair.

    Parameters
    ----------
    stock_ret : 1-D array of length n_window — stock daily returns
    pca_ret   : DataFrame (n_window, K) — PCA factor returns (named columns)
    dt        : time step (1/252 for daily data)

    Returns a dict with keys: m, kappa, sig, sig_eq, alpha_trend, a_OU, b_OU.
    All NaN if inputs are non-finite or the AR fit is degenerate.
    """
    NAN_PARAMS = dict(m=np.nan, kappa=np.nan, sig=np.nan, sig_eq=np.nan,
                      alpha_trend=np.nan, a_OU=np.nan, b_OU=np.nan)

    pca_arr = pca_ret.values
    if not (np.isfinite(stock_ret).all() and np.isfinite(pca_arr).all()):
        return NAN_PARAMS

    # Step 1: regress stock returns on PCA factors + linear trend
    # add_trend on a DataFrame preserves column names so reg.params is a named Series
    X = add_trend(pca_ret, trend="t")
    try:
        reg = sm.OLS(stock_ret, X).fit()
    except Exception:
        return NAN_PARAMS

    # alpha_trend is the coefficient on the 'trend' column
    trend_col = "trend"
    if trend_col not in reg.params.index:
        return NAN_PARAMS
    alpha_trend = float(reg.params[trend_col])

    epsilon = reg.resid.values

    # Step 2: cumulate residuals, pin X_{60} = 0 (Appendix of AL 2010, p.45)
    x_t = np.concatenate([np.cumsum(epsilon[:-1]), [0.0]])

    # Step 3: AR(1) regression  X_{n+1} = a + b * X_n + v
    try:
        X_ar = sm.add_constant(x_t[:-1])
        reg_ou = sm.OLS(x_t[1:], X_ar).fit()
    except Exception:
        return NAN_PARAMS

    a, b = float(reg_ou.params[0]), float(reg_ou.params[1])

    if b <= 0 or b >= 1:   # degenerate / non-stationary AR(1)
        return NAN_PARAMS

    kappa   = np.log(1.0 / b) / dt
    m       = a / (1.0 - b)
    sig_eq  = float(np.std(reg_ou.resid)) / np.sqrt(1.0 - b ** 2)
    sig_var = np.var(reg_ou.resid) * 2 * kappa / (1.0 - b ** 2)
    sig     = float(np.sqrt(max(sig_var, 0.0)))

    return dict(m=m, kappa=kappa, sig=sig, sig_eq=sig_eq,
                alpha_trend=alpha_trend, a_OU=a, b_OU=b)


def compute_s_scores(
    prices: pd.DataFrame,
    membership: pd.DataFrame,
    pca_factors: pd.DataFrame,
    n_window: int = N_WINDOW,
    kappa_min: float | None = KAPPA_MIN,
) -> pd.DataFrame:
    """
    Generate daily s-scores for all ASX 200 members.

    For each trading day t and each member stock i, fit the OU model on the
    prior n_window days of returns defactored by the PCA eigen-portfolios,
    then cross-sectionally centre equilibrium levels to get the s-score.

    Returns
    -------
    DataFrame: trading dates × tickers, dtype float64.
               NaN where the stock is not a member or fails kappa filter.
    """
    ret = prices.astype(float).ffill().pct_change()
    # Align PCA factors onto the same date index (fill any gaps with 0)
    pca = pca_factors.reindex(ret.index).fillna(0.0)

    all_tickers = list(ret.columns)
    dates = ret.index

    s_rows: list[pd.Series] = []
    day_iter = range(n_window, len(dates))
    day_iter = tqdm(day_iter, desc="OU s-scores", unit="day",
                    total=len(dates) - n_window)

    for i in day_iter:
        dt_label = dates[i]

        # Active members on this day
        if dt_label in membership.index:
            active_mask = membership.loc[dt_label]
            active = active_mask[active_mask].index.tolist()
        else:
            active = all_tickers

        params: dict[str, dict] = {}
        for ticker in active:
            if ticker not in ret.columns:
                continue
            stock_window = ret[ticker].iloc[i - n_window: i].values
            pca_window   = pca.iloc[i - n_window: i]   # DataFrame (n_window, K)
            params[ticker] = _fit_ou_group(stock_window, pca_window)

        if not params:
            s_rows.append(pd.Series(np.nan, index=all_tickers, name=dt_label))
            continue

        pdf = pd.DataFrame(params).T   # (n_tickers, param_cols)

        # Cross-sectional centering of equilibrium (p.46 in AL 2010)
        equilibrium = (pdf["a_OU"] / (1.0 - pdf["b_OU"])).replace(
            [np.inf, -np.inf], np.nan
        )
        m_bar = equilibrium - equilibrium.mean()

        sig_eq = pdf["sig_eq"]
        alpha_t = pdf["alpha_trend"]
        kappa_v = pdf["kappa"]

        # s_i = -m_bar_i / sig_eq_i  −  alpha_trend_i / (kappa_i * sig_eq_i)
        with np.errstate(divide="ignore", invalid="ignore"):
            s = -m_bar / sig_eq - alpha_t / (kappa_v * sig_eq)

        # kappa filter: only trade fast-reverting names
        if kappa_min is not None:
            s = s.where(kappa_v > kappa_min)

        s_rows.append(s.rename(dt_label).reindex(all_tickers))

    s_df = pd.DataFrame(s_rows)
    s_df.index.name = "Date"
    return s_df


# ─────────────────────────────────────────────────────────────────────────────
# Position management (inlined from statsArb-dev/src/bt_tools.py)
# ─────────────────────────────────────────────────────────────────────────────

def _signal_search(df, entry_idx, exit_idx, pos_vec, sl_limit):
    if not entry_idx:
        return pos_vec
    pos_vec.append((entry_idx[0], "entry"))
    exit_idx = [x for x in exit_idx if x > pos_vec[-1][0]]
    if not exit_idx:
        entry_idx = []
    else:
        if sl_limit is not None:
            sl = df.loc[pos_vec[-1][0] + 1: exit_idx[0]].cumsum() < sl_limit
            pos_vec.append(
                (sl.idxmax(), "sl_exit") if sl.any() else (sl.index[-1], "exit")
            )
        else:
            pos_vec.append((exit_idx[0], "exit"))
    entry_idx = [x for x in entry_idx if x > pos_vec[-1][0]]
    return _signal_search(df, entry_idx, exit_idx, pos_vec, sl_limit)


def _build_pos_series(entry_s: pd.Series, exit_s: pd.Series,
                       ret_s: pd.Series, sl_limit: float | None) -> pd.Series:
    """Build a 0/1 position series for one ticker."""
    n = len(entry_s)
    int_idx = np.arange(n)
    entry_idx = int_idx[entry_s.values.astype(bool)].tolist()
    exit_idx  = int_idx[exit_s.values.astype(bool)].tolist()

    if not entry_idx or not exit_idx:
        return pd.Series(0.0, index=entry_s.index)

    ret_int = pd.Series(ret_s.values, index=int_idx, name="r")
    pos_vec = _signal_search(ret_int, entry_idx, exit_idx, [], sl_limit)
    pos_int = pd.Series(0.0, index=int_idx)
    for idx, sig in pos_vec:
        if sig == "entry":
            pos_int.loc[idx] = 1.0

    pos_int = pos_int.ffill()
    # Zero out after each exit signal
    for idx, sig in pos_vec:
        if sig in ("exit", "sl_exit"):
            pos_int.loc[idx:] = 0.0
            # re-set subsequent entries
            later_entries = [j for j, s2 in pos_vec
                             if s2 == "entry" and j > idx]
            if later_entries:
                pos_int.loc[later_entries[0]:] = 1.0  # will be fixed in next iter

    # Cleaner approach: replay the pos_vec sequentially
    pos_arr = np.zeros(n)
    in_pos = False
    entry_set = {idx for idx, s2 in pos_vec if s2 == "entry"}
    exit_set  = {idx for idx, s2 in pos_vec if s2 in ("exit", "sl_exit")}
    for k in range(n):
        if k in entry_set:
            in_pos = True
        if k in exit_set:
            in_pos = False
        pos_arr[k] = 1.0 if in_pos else 0.0

    pos = pd.Series(pos_arr, index=entry_s.index)
    # shift 1: position formed at close applies from next day open
    return pos.shift(1).fillna(0.0)


def build_position_df(
    entry_df: pd.DataFrame,
    exit_df: pd.DataFrame,
    ret_df: pd.DataFrame,
    sl_limit: float | None = SL_LIMIT,
) -> pd.DataFrame:
    """Build (date × ticker) 0/1 position DataFrame."""
    pos_dict: dict[str, pd.Series] = {}
    for ticker in entry_df.columns:
        pos_dict[ticker] = _build_pos_series(
            entry_df[ticker], exit_df[ticker], ret_df[ticker], sl_limit
        )
    return pd.DataFrame(pos_dict, index=entry_df.index)


def portfolio_returns(
    ret_df: pd.DataFrame,
    pos_df: pd.DataFrame,
    direction: str = "long",
    portfolio_value: float = INITIAL_VALUE,
) -> tuple[pd.Series, pd.Series]:
    """
    Equal-weighted portfolio return series with tiered ASX transaction costs.

    Tracks running portfolio value day-by-day so the $10 minimum brokerage
    correctly scales with the evolving portfolio size.

    Returns
    -------
    (gross_returns, net_returns) as a tuple of pd.Series.
    Caller should sum long+short gross for combined V tracking; here V
    is approximated as the leg's own running value starting at portfolio_value.
    """
    sign = 1.0 if direction == "long" else -1.0
    n_held = pos_df.sum(axis=1).replace(0, np.nan)
    w = pos_df.div(n_held, axis=0).fillna(0.0)
    ret_aligned = ret_df.reindex(columns=pos_df.columns).fillna(0.0)

    delta_pos = pos_df.diff().abs()

    V = portfolio_value
    gross_list, net_list = [], []
    for i in range(len(pos_df)):
        gross_r = float((sign * w.iloc[i] * ret_aligned.iloc[i]).sum())

        # Tiered cost per traded position
        dpos = delta_pos.iloc[i]
        active = dpos[dpos > 0]
        tc_frac = 0.0
        if not active.empty:
            n_active = int(n_held.iloc[i]) if not np.isnan(n_held.iloc[i]) else 1
            # weight of each changing position = 1/n_held
            dw_i = 1.0 / max(n_active, 1) * active.values  # Δweight per position
            min_fee_frac = MIN_BROKERAGE / V
            tc_frac = float(
                (np.maximum(min_fee_frac, BROKERAGE_RATE * dw_i)
                 + MARKET_IMPACT * dw_i).sum()
            )

        # Short borrow fee: 1% p.a. on total short weight held each day
        borrow_cost = 0.0
        if direction == "short":
            total_short_weight = float(w.iloc[i].sum())
            borrow_cost = total_short_weight * BORROW_RATE_DAILY

        net_r = gross_r - tc_frac - borrow_cost
        gross_list.append(gross_r)
        net_list.append(net_r)
        V = max(V * (1 + net_r), 1.0)

    idx = pos_df.index
    return pd.Series(gross_list, index=idx), pd.Series(net_list, index=idx)


# ─────────────────────────────────────────────────────────────────────────────
# Trade blotter
# ─────────────────────────────────────────────────────────────────────────────

def extract_trades(
    pos_df: pd.DataFrame,
    ret_df: pd.DataFrame,
    direction: str,
) -> pd.DataFrame:
    sign = 1.0 if direction == "long" else -1.0
    dates = pos_df.index
    rows = []
    for ticker in pos_df.columns:
        pos = pos_df[ticker].to_numpy()
        r   = ret_df[ticker].reindex(dates).fillna(0).to_numpy()
        in_pos, entry_i = False, 0
        for i, p in enumerate(pos):
            if p == 1 and not in_pos:
                in_pos, entry_i = True, i
            elif p != 1 and in_pos:
                in_pos = False
                trade_ret = float(np.prod(1 + sign * r[entry_i:i]) - 1)
                rows.append((ticker, direction, dates[entry_i], dates[i - 1],
                             i - entry_i, trade_ret))
        if in_pos:
            trade_ret = float(np.prod(1 + sign * r[entry_i:]) - 1)
            rows.append((ticker, direction, dates[entry_i], dates[-1],
                         len(pos) - entry_i, trade_ret))
    return pd.DataFrame(rows,
                        columns=["ticker", "side", "entry", "exit",
                                 "days", "trade_ret"])


# ─────────────────────────────────────────────────────────────────────────────
# Performance
# ─────────────────────────────────────────────────────────────────────────────

def build_equity_curve(
    long_ret: pd.Series,
    short_ret: pd.Series,
) -> pd.DataFrame:
    """Combine long and short legs into a combined equity curve."""
    cum_ret = (long_ret + short_ret).fillna(0)  # both legs are sign-adjusted profits

    # Trim to first day either leg is active (matches JT convention)
    active = (long_ret.abs() + short_ret.abs()).fillna(0) > 0
    if active.any():
        cum_ret = cum_ret.loc[active.idxmax():]
        long_ret  = long_ret.loc[active.idxmax():]
        short_ret = short_ret.loc[active.idxmax():]

    equity  = (1 + cum_ret).cumprod()
    dd_roll = equity.rolling(126, min_periods=1).apply(
        lambda x: x[-1] / x.max() - 1, raw=True
    )
    long_pnl  = (1 + long_ret.fillna(0)).cumprod()
    short_pnl = (1 + short_ret.fillna(0)).cumprod()
    return pd.DataFrame({
        "long_ret":  long_ret,
        "short_ret": short_ret,
        "cum_ret":   cum_ret,
        "equity":    equity,
        "drawdown":  dd_roll,
        "long_pnl":  long_pnl,
        "short_pnl": short_pnl,
    })


def summarise(curve: pd.DataFrame, trades: pd.DataFrame) -> dict:
    r = curve["cum_ret"]
    sharpe   = float(np.sqrt(252) * r.mean() / r.std()) if r.std() > 0 else np.nan
    end_pnl  = float(curve["equity"].iloc[-1])
    max_dd   = float(curve["drawdown"].min())
    ann_ret  = float(end_pnl ** (252 / len(r)) - 1)
    win_rate = float((trades["trade_ret"] > 0).mean()) if not trades.empty else np.nan
    return {
        "sharpe":     round(sharpe, 3),
        "ann_return": round(ann_ret, 4),
        "end_pnl":    round(end_pnl, 4),
        "max_dd":     round(max_dd, 4),
        "n_trades":   len(trades),
        "win_rate":   round(win_rate, 4) if not np.isnan(win_rate) else np.nan,
    }


def plot_performance(
    curve: pd.DataFrame,
    trades: pd.DataFrame,
    m: dict,
    out_dir: str,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    axes = axes.flatten()

    # Equity curve
    curve["equity"].plot(ax=axes[0], color="steelblue", lw=1.5, label="Strategy")
    axes[0].axhline(1.0, lw=0.6, ls="--", color="k", alpha=0.6)
    axes[0].set_title(
        f"Combined PnL  |  Sharpe {m['sharpe']:.2f}  "
        f"Ann {m['ann_return']:.1%}  MaxDD {m['max_dd']:.1%}"
    )
    axes[0].legend()

    # Rolling drawdown
    curve["drawdown"].plot(ax=axes[1], color="firebrick", lw=0.9)
    axes[1].axhline(0, lw=0.6, ls="--", color="k", alpha=0.6)
    axes[1].set_title("6-month rolling drawdown")

    # Long vs short PnL
    curve[["long_pnl", "short_pnl"]].plot(ax=axes[2])
    axes[2].axhline(1.0, lw=0.6, ls="--", color="k", alpha=0.6)
    axes[2].set_title("Long vs Short leg PnL")

    # Trade return histogram
    if not trades.empty:
        axes[3].hist(trades["trade_ret"] * 100, bins=50, color="steelblue",
                     edgecolor="white", lw=0.3)
        axes[3].axvline(0, lw=0.9, ls="--", color="k")
    axes[3].set_title(
        f"Trade return distribution  "
        f"({m['n_trades']} trades, win {m['win_rate']:.0%})"
    )
    axes[3].set_xlabel("Return (%)")

    fig.suptitle(
        f"ASX 200  |  AL PCA Stat Arb  |  "
        f"kappa_min={KAPPA_MIN:.1f}/yr  cost=max($10, 0.11%) + 7bps impact",
        fontsize=11,
    )
    fig.tight_layout()
    path = os.path.join(out_dir, "performance.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    log.info("Chart saved: %s", path)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    log.info("=== AL PCA Stat Arb — S&P ASX 200 C&P ===")

    log.info("Step 1/5  Loading prices...")
    prices = dn.load_prices()
    log.info("  %d trading days × %d tickers", *prices.shape)

    log.info("Step 2/5  Loading PIT membership mask...")
    membership = dn.load_membership(prices)

    log.info("Step 3/5  Computing rolling PCA factors (%d components, %d-day window)...",
             N_PCA, N_WINDOW)
    pca_factors = dn.compute_pca_factors(
        prices, membership, n_window=N_WINDOW, n_components=N_PCA
    )
    log.info("  PCA factors: %d rows × %d components", *pca_factors.shape)

    s_path = os.path.join(OUT_DIR, "s_scores.csv")
    if os.path.exists(s_path):
        log.info("Step 4/5  Loading cached s-scores from %s", s_path)
        s_scores = pd.read_csv(s_path, index_col=0, parse_dates=True)
    else:
        log.info("Step 4/5  Fitting OU process and generating s-scores...")
        log.info("  (This takes ~3 minutes for 446 days × 245 stocks)")
        s_scores = compute_s_scores(prices, membership, pca_factors)
        s_scores.to_csv(s_path)
        log.info("  s-scores saved: %s", s_path)

    log.info("Step 5/5  Running backtest...")
    ret = prices.astype(float).ffill().pct_change()

    # Align s-scores to the same date index as ret, forward-fill (hold last signal)
    s_aligned = s_scores.reindex(ret.index).ffill()

    long_entry  = s_aligned < S_BO
    long_exit   = s_aligned > S_BC
    short_entry = s_aligned > S_SO
    short_exit  = s_aligned < S_SC

    # Only run signals for tickers that appear in ret and s_scores
    common_tickers = [t for t in s_scores.columns if t in ret.columns]
    ret_common = ret[common_tickers]

    long_entry  = long_entry[common_tickers]
    long_exit   = long_exit[common_tickers]
    short_entry = short_entry[common_tickers]
    short_exit  = short_exit[common_tickers]

    log.info("  Building long positions...")
    pos_long  = build_position_df(long_entry,  long_exit,  ret_common)
    log.info("  Building short positions...")
    pos_short = build_position_df(short_entry, short_exit, ret_common)

    _, long_ret  = portfolio_returns(ret_common, pos_long,  direction="long")
    _, short_ret = portfolio_returns(ret_common, pos_short, direction="short")

    curve = build_equity_curve(long_ret, short_ret)
    curve.to_csv(os.path.join(OUT_DIR, "equity_curve.csv"))

    log.info("  Extracting trade blotter...")
    trades = pd.concat([
        extract_trades(pos_long,  ret_common, "long"),
        extract_trades(pos_short, ret_common, "short"),
    ]).sort_values("entry").reset_index(drop=True)
    trades.to_csv(os.path.join(OUT_DIR, "trades.csv"), index=False)

    m = summarise(curve, trades)

    print("\n=== Results ===")
    print(f"  Sharpe ratio   : {m['sharpe']:.3f}")
    print(f"  Ann. return    : {m['ann_return']:.2%}")
    print(f"  End PnL (×)    : {m['end_pnl']:.3f}")
    print(f"  Max drawdown   : {m['max_dd']:.2%}")
    print(f"  Total trades   : {m['n_trades']}")
    print(f"  Win rate       : {m['win_rate']:.1%}")

    if not trades.empty:
        print(f"\n  Avg trade ret  : {trades['trade_ret'].mean():+.2%}")
        print(f"  Median trade   : {trades['trade_ret'].median():+.2%}")
        print(f"  Avg hold (days): {trades['days'].mean():.1f}")

    plot_performance(curve, trades, m, OUT_DIR)
    print(f"\nOutputs in: {OUT_DIR}")


if __name__ == "__main__":
    main()
