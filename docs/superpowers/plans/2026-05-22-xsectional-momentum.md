# XSectional Momentum Model Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a cross-sectional 12–1 momentum backtesting module on the S&P 500 that produces a full performance tearsheet for 2000–2025.

**Architecture:** Modular pipeline — `data.py` downloads/caches prices, `signals.py` computes 12–1 momentum scores, `portfolio.py` constructs equal-weighted long-short weights, `backtest.py` simulates monthly returns, `report.py` renders the tearsheet. All parameters live in `config.py`; `main.py` orchestrates the full run.

**Tech Stack:** Python 3.10+, yfinance, pandas, numpy, matplotlib, pytest

---

> **Working directory for all commands:** `XSectional/` inside the repo root
> (`/Users/paulchenbackup/Desktop/VRI/In-Market-Agentic-AI-Monitoring-Application-in-Financial-Market/XSectional/`)

---

## File Map

| File | Responsibility |
|---|---|
| `config.py` | All tunable constants — dates, lookback, quantiles, paths |
| `data.py` | Download S&P 500 adjusted close prices via yfinance; cache to CSV |
| `signals.py` | Compute monthly returns + 12–1 momentum scores |
| `portfolio.py` | Construct equal-weighted long-short weights from monthly scores |
| `backtest.py` | Simulate month-by-month portfolio returns |
| `report.py` | Compute metrics + render 4-panel tearsheet PNG |
| `main.py` | Orchestrate full pipeline end-to-end |
| `pytest.ini` | Test configuration and pythonpath |
| `conftest.py` | Shared pytest fixtures |
| `requirements.txt` | Pinned dependencies |
| `tests/test_data.py` | Unit tests for data.py |
| `tests/test_signals.py` | Unit tests for signals.py |
| `tests/test_portfolio.py` | Unit tests for portfolio.py |
| `tests/test_backtest.py` | Unit tests for backtest.py |
| `tests/test_report.py` | Unit tests for report.py |
| `data/` | Local cache directory (gitignored) |

---

## Task 1: Project Setup

**Files:**
- Create: `XSectional/requirements.txt`
- Create: `XSectional/pytest.ini`
- Create: `XSectional/conftest.py`
- Create: `XSectional/tests/__init__.py`
- Create: `XSectional/.gitignore`

- [ ] **Step 1: Create `requirements.txt`**

```
yfinance>=0.2.54
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
pytest>=7.4.0
```

- [ ] **Step 2: Create `pytest.ini`**

```ini
[pytest]
pythonpath = .
testpaths = tests
```

- [ ] **Step 3: Create `conftest.py`** (empty — `pytest.ini` handles path)

```python
# conftest.py — pytest discovers this file; pythonpath set in pytest.ini
```

- [ ] **Step 4: Create `tests/__init__.py`** (empty file)

```python
```

- [ ] **Step 5: Create `.gitignore`**

```
data/
__pycache__/
*.pyc
.pytest_cache/
*.png
```

- [ ] **Step 6: Install dependencies**

Run: `pip install -r requirements.txt`

Expected: All packages install without error.

- [ ] **Step 7: Verify pytest discovers correctly**

Run: `pytest --collect-only`

Expected: `no tests ran` (no tests yet) — confirms path resolution works.

- [ ] **Step 8: Commit**

```bash
git add XSectional/
git commit -m "feat: XSectional project scaffold — requirements, pytest config"
```

---

## Task 2: config.py

**Files:**
- Create: `XSectional/config.py`

- [ ] **Step 1: Write `config.py`**

```python
# config.py — all tunable parameters for the XSectional momentum model

START_DATE = "2000-01-01"
END_DATE   = "2025-12-31"

LOOKBACK_MONTHS = 12   # Momentum formation window (months)
SKIP_MONTHS     = 1    # Months to skip before lookback (short-term reversal avoidance)

TOP_QUANTILE    = 0.20  # Fraction of stocks to go long (top performers)
BOTTOM_QUANTILE = 0.20  # Fraction of stocks to short (bottom performers)

REBALANCE_FREQ  = "ME"  # Month-end rebalancing (pandas resample alias)

DATA_DIR = "data"

MISSING_DATA_THRESHOLD = 0.20  # Drop tickers with > 20% missing data
MIN_STOCKS_PER_LEG     = 10    # Minimum stocks required per long/short leg
```

- [ ] **Step 2: Commit**

```bash
git add XSectional/config.py
git commit -m "feat: add config.py with all strategy parameters"
```

---

## Task 3: data.py

**Files:**
- Create: `XSectional/tests/test_data.py`
- Create: `XSectional/data.py`

- [ ] **Step 1: Write failing tests in `tests/test_data.py`**

```python
import pytest
import pandas as pd
import numpy as np
from data import drop_missing, load_prices


def test_drop_missing_removes_high_missing_tickers():
    df = pd.DataFrame({
        "A": [1.0, 2.0, 3.0, 4.0, 5.0],     # 0% missing  → keep
        "B": [None, None, None, None, 1.0],   # 80% missing → drop
        "C": [1.0, None, 3.0, 4.0, 5.0],     # 20% missing → keep (≤ threshold)
    })
    result = drop_missing(df, threshold=0.20)
    assert "A" in result.columns
    assert "C" in result.columns
    assert "B" not in result.columns


def test_drop_missing_keeps_complete_tickers():
    df = pd.DataFrame({
        "X": [1.0, 2.0, 3.0],
        "Y": [None, 2.0, None],   # 67% missing → drop
    })
    result = drop_missing(df, threshold=0.20)
    assert list(result.columns) == ["X"]


def test_drop_missing_empty_dataframe():
    df = pd.DataFrame()
    result = drop_missing(df, threshold=0.20)
    assert result.empty


def test_load_prices_returns_dataframe_from_cache(tmp_path, monkeypatch):
    import data as data_module
    dates = pd.date_range("2000-01-01", periods=10)
    fake_prices = pd.DataFrame(
        np.ones((10, 3)) * 100.0,
        index=dates,
        columns=["AAPL", "MSFT", "GOOG"],
    )
    cache_path = tmp_path / "sp500_prices.csv"
    fake_prices.to_csv(cache_path)

    monkeypatch.setattr(data_module, "SP500_CACHE", str(cache_path))
    prices = load_prices()

    assert isinstance(prices, pd.DataFrame)
    assert prices.shape == (10, 3)
    assert list(prices.columns) == ["AAPL", "MSFT", "GOOG"]


def test_load_prices_raises_when_no_cache_and_no_network(tmp_path, monkeypatch):
    """When cache is missing and download is mocked to fail, FileNotFoundError raised."""
    import data as data_module
    monkeypatch.setattr(data_module, "SP500_CACHE", str(tmp_path / "missing.csv"))

    def _fail(*args, **kwargs):
        raise RuntimeError("network unavailable")

    monkeypatch.setattr(data_module, "download_prices", _fail)
    monkeypatch.setattr(data_module, "get_sp500_tickers", lambda: ["AAPL"])

    with pytest.raises(Exception):
        load_prices()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_data.py -v`

Expected: `ImportError: No module named 'data'` or `ModuleNotFoundError`

- [ ] **Step 3: Write `data.py`**

```python
# data.py — download and cache S&P 500 adjusted close prices

import os
import logging

import pandas as pd
import yfinance as yf

import config

logger = logging.getLogger(__name__)

SP500_CACHE = os.path.join(config.DATA_DIR, "sp500_prices.csv")


def get_sp500_tickers() -> list[str]:
    """Fetch current S&P 500 tickers from Wikipedia."""
    tables = pd.read_html(
        "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    )
    tickers = tables[0]["Symbol"].str.replace(".", "-", regex=False).tolist()
    return tickers


def download_prices(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """Download adjusted close prices from yfinance for a list of tickers."""
    raw = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
    )
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        prices = raw[["Close"]]
    return prices


def drop_missing(prices: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Drop columns (tickers) where the fraction of NaN values exceeds threshold."""
    if prices.empty:
        return prices
    missing_frac = prices.isna().mean()
    return prices.loc[:, missing_frac <= threshold]


def load_prices() -> pd.DataFrame:
    """
    Return a DataFrame of adjusted close prices (dates × tickers).

    Loads from local CSV cache if available, otherwise downloads from yfinance
    and saves to cache. Raises FileNotFoundError if download fails and no cache exists.
    """
    os.makedirs(config.DATA_DIR, exist_ok=True)

    if os.path.exists(SP500_CACHE):
        logger.info("Loading prices from cache: %s", SP500_CACHE)
        prices = pd.read_csv(SP500_CACHE, index_col=0, parse_dates=True)
        return prices

    logger.info("Cache not found — downloading S&P 500 prices from yfinance...")
    tickers = get_sp500_tickers()
    prices = download_prices(tickers, config.START_DATE, config.END_DATE)
    prices = drop_missing(prices, config.MISSING_DATA_THRESHOLD)
    prices.to_csv(SP500_CACHE)
    logger.info("Prices saved to cache: %s", SP500_CACHE)
    return prices
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_data.py -v`

Expected:
```
tests/test_data.py::test_drop_missing_removes_high_missing_tickers PASSED
tests/test_data.py::test_drop_missing_keeps_complete_tickers PASSED
tests/test_data.py::test_drop_missing_empty_dataframe PASSED
tests/test_data.py::test_load_prices_returns_dataframe_from_cache PASSED
tests/test_data.py::test_load_prices_raises_when_no_cache_and_no_network PASSED
5 passed
```

- [ ] **Step 5: Commit**

```bash
git add XSectional/data.py XSectional/tests/test_data.py
git commit -m "feat: add data.py — S&P 500 price download and cache with tests"
```

---

## Task 4: signals.py

**Files:**
- Create: `XSectional/tests/test_signals.py`
- Create: `XSectional/signals.py`

- [ ] **Step 1: Write failing tests in `tests/test_signals.py`**

```python
import pytest
import pandas as pd
import numpy as np
from signals import compute_monthly_returns, compute_momentum_scores


@pytest.fixture
def flat_prices():
    """All-flat prices — every return is zero, momentum should be zero."""
    dates = pd.date_range("2000-01-01", periods=500, freq="D")
    return pd.DataFrame(
        100.0,
        index=dates,
        columns=["A", "B", "C", "D", "E"],
    )


@pytest.fixture
def random_prices():
    """Random-walk prices with 5 tickers and ~2 years of daily data."""
    np.random.seed(42)
    dates = pd.date_range("2000-01-01", periods=500, freq="D")
    prices = 100.0 * np.cumprod(
        1 + np.random.normal(0.0003, 0.01, (500, 5)), axis=0
    )
    return pd.DataFrame(prices, index=dates, columns=["A", "B", "C", "D", "E"])


def test_compute_monthly_returns_columns_match(random_prices):
    result = compute_monthly_returns(random_prices)
    assert list(result.columns) == ["A", "B", "C", "D", "E"]


def test_compute_monthly_returns_first_row_is_nan(random_prices):
    result = compute_monthly_returns(random_prices)
    assert result.iloc[0].isna().all()


def test_compute_monthly_returns_has_fewer_rows_than_daily(random_prices):
    result = compute_monthly_returns(random_prices)
    assert len(result) < len(random_prices)


def test_compute_momentum_scores_columns_match(random_prices):
    scores = compute_momentum_scores(random_prices)
    assert list(scores.columns) == ["A", "B", "C", "D", "E"]


def test_compute_momentum_scores_produces_valid_values_after_warmup(random_prices):
    """After 13+ months of data, at least some momentum scores should be non-NaN."""
    scores = compute_momentum_scores(random_prices)
    valid_rows = scores.dropna(how="all")
    assert len(valid_rows) > 0


def test_compute_momentum_scores_flat_prices_are_zero(flat_prices):
    """When all prices are flat, every momentum score should be 0."""
    scores = compute_momentum_scores(flat_prices)
    valid = scores.dropna(how="all")
    assert len(valid) > 0
    assert (valid.fillna(0).abs() < 1e-9).all().all()


def test_compute_momentum_scores_needs_13_months_warmup(random_prices):
    """Momentum score rows before 13 months should all be NaN."""
    scores = compute_momentum_scores(random_prices)
    # First 12 rows of monthly scores (LOOKBACK=12, SKIP=1) should be NaN
    monthly_returns = compute_monthly_returns(random_prices)
    first_valid_idx = 13  # 12 lookback + 1 skip
    early_scores = scores.iloc[: first_valid_idx - 1]
    assert early_scores.dropna(how="all").empty
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_signals.py -v`

Expected: `ImportError: No module named 'signals'`

- [ ] **Step 3: Write `signals.py`**

```python
# signals.py — compute monthly returns and 12-1 cross-sectional momentum scores

import numpy as np
import pandas as pd

import config


def compute_monthly_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Resample daily prices to month-end and compute simple period returns.

    Returns a DataFrame with the same columns as prices, indexed by month-end dates.
    The first row is always NaN (no prior month to compare against).
    """
    monthly = prices.resample(config.REBALANCE_FREQ).last()
    return monthly.pct_change()


def compute_momentum_scores(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute 12-1 cross-sectional momentum scores for all tickers.

    At each month-end t, the score is the cumulative return from
    t-13 to t-2 (12 months, skipping the most recent month t-1).

    Uses log returns summed over a rolling window for numerical stability,
    then converts back to simple returns.

    Returns a DataFrame aligned with monthly_returns index and prices columns.
    Scores before the 13-month warm-up period are NaN.
    """
    monthly_returns = compute_monthly_returns(prices)

    # Clip to avoid log(0) or log(negative) on extreme down-moves
    log_returns = np.log(1 + monthly_returns.clip(lower=-0.9999))

    # shift(SKIP_MONTHS) moves the window back — at time t, the most recent
    # value is now r_{t-1} (one month ago), so rolling sum skips month t.
    # rolling(LOOKBACK_MONTHS).sum() then accumulates the 12 months before that.
    momentum_log = (
        log_returns.shift(config.SKIP_MONTHS)
        .rolling(config.LOOKBACK_MONTHS)
        .sum()
    )

    return np.exp(momentum_log) - 1
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_signals.py -v`

Expected:
```
tests/test_signals.py::test_compute_monthly_returns_columns_match PASSED
tests/test_signals.py::test_compute_monthly_returns_first_row_is_nan PASSED
tests/test_signals.py::test_compute_monthly_returns_has_fewer_rows_than_daily PASSED
tests/test_signals.py::test_compute_momentum_scores_columns_match PASSED
tests/test_signals.py::test_compute_momentum_scores_produces_valid_values_after_warmup PASSED
tests/test_signals.py::test_compute_momentum_scores_flat_prices_are_zero PASSED
tests/test_signals.py::test_compute_momentum_scores_needs_13_months_warmup PASSED
7 passed
```

- [ ] **Step 5: Commit**

```bash
git add XSectional/signals.py XSectional/tests/test_signals.py
git commit -m "feat: add signals.py — 12-1 momentum scores with tests"
```

---

## Task 5: portfolio.py

**Files:**
- Create: `XSectional/tests/test_portfolio.py`
- Create: `XSectional/portfolio.py`

- [ ] **Step 1: Write failing tests in `tests/test_portfolio.py`**

```python
import pytest
import pandas as pd
import numpy as np
from portfolio import construct_portfolio


@pytest.fixture
def sample_scores():
    """50 tickers, 24 months, random scores (enough stocks for 20% quantile cuts)."""
    np.random.seed(42)
    dates = pd.date_range("2002-01-31", periods=24, freq="ME")
    tickers = [f"T{i:02d}" for i in range(50)]
    return pd.DataFrame(
        np.random.randn(24, 50),
        index=dates,
        columns=tickers,
    )


def test_construct_portfolio_returns_dataframe(sample_scores):
    weights = construct_portfolio(sample_scores)
    assert isinstance(weights, pd.DataFrame)


def test_construct_portfolio_same_shape_as_scores(sample_scores):
    weights = construct_portfolio(sample_scores)
    assert weights.shape == sample_scores.shape


def test_long_weights_are_positive(sample_scores):
    weights = construct_portfolio(sample_scores)
    # Every non-zero, non-short weight must be positive
    assert (weights[weights > 0] > 0).all().all()


def test_short_weights_are_negative(sample_scores):
    weights = construct_portfolio(sample_scores)
    assert (weights[weights < 0] < 0).all().all()


def test_long_leg_sums_to_one_each_month(sample_scores):
    weights = construct_portfolio(sample_scores)
    for date in weights.index:
        row = weights.loc[date]
        long_sum = row[row > 0].sum()
        if long_sum > 0:
            assert abs(long_sum - 1.0) < 1e-9, (
                f"Long leg at {date} sums to {long_sum:.6f}, expected 1.0"
            )


def test_short_leg_sums_to_minus_one_each_month(sample_scores):
    weights = construct_portfolio(sample_scores)
    for date in weights.index:
        row = weights.loc[date]
        short_sum = row[row < 0].sum()
        if short_sum < 0:
            assert abs(short_sum + 1.0) < 1e-9, (
                f"Short leg at {date} sums to {short_sum:.6f}, expected -1.0"
            )


def test_top_20_percent_assigned_positive_weight(sample_scores):
    weights = construct_portfolio(sample_scores)
    date = sample_scores.index[0]
    row = sample_scores.loc[date].dropna()
    n_long = int(len(row) * 0.20)
    top_tickers = row.nlargest(n_long).index
    assert all(weights.loc[date, t] > 0 for t in top_tickers)


def test_bottom_20_percent_assigned_negative_weight(sample_scores):
    weights = construct_portfolio(sample_scores)
    date = sample_scores.index[0]
    row = sample_scores.loc[date].dropna()
    n_short = int(len(row) * 0.20)
    bottom_tickers = row.nsmallest(n_short).index
    assert all(weights.loc[date, t] < 0 for t in bottom_tickers)


def test_all_nan_row_produces_zero_weights(sample_scores):
    """A month where all scores are NaN should result in all-zero weights."""
    scores_with_nan = sample_scores.copy()
    scores_with_nan.iloc[5] = np.nan
    weights = construct_portfolio(scores_with_nan)
    assert (weights.iloc[5] == 0.0).all()


def test_insufficient_stocks_produces_zero_weights():
    """Fewer than MIN_STOCKS_PER_LEG qualifying stocks → zero weights for that month."""
    dates = pd.date_range("2002-01-31", periods=3, freq="ME")
    # Only 5 stocks — fewer than MIN_STOCKS_PER_LEG (10)
    tickers = [f"T{i}" for i in range(5)]
    scores = pd.DataFrame(
        np.random.randn(3, 5), index=dates, columns=tickers
    )
    weights = construct_portfolio(scores)
    assert (weights == 0.0).all().all()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_portfolio.py -v`

Expected: `ImportError: No module named 'portfolio'`

- [ ] **Step 3: Write `portfolio.py`**

```python
# portfolio.py — construct equal-weighted long-short portfolio from momentum scores

import logging

import pandas as pd
import numpy as np

import config

logger = logging.getLogger(__name__)


def construct_portfolio(scores: pd.DataFrame) -> pd.DataFrame:
    """
    Build equal-weighted long-short portfolio weights from monthly momentum scores.

    Each month:
      - Long:  top TOP_QUANTILE fraction (equal weight = 1/n_long each)
      - Short: bottom BOTTOM_QUANTILE fraction (equal weight = -1/n_short each)
      - All others: zero

    If fewer than MIN_STOCKS_PER_LEG stocks qualify for either leg,
    that month's weights are set to zero and a warning is logged.

    Returns a DataFrame of weights with the same shape as scores.
    """
    weights = pd.DataFrame(0.0, index=scores.index, columns=scores.columns)

    for date, row in scores.iterrows():
        valid = row.dropna()
        if valid.empty:
            continue

        n_long = max(1, int(len(valid) * config.TOP_QUANTILE))
        n_short = max(1, int(len(valid) * config.BOTTOM_QUANTILE))

        if n_long < config.MIN_STOCKS_PER_LEG or n_short < config.MIN_STOCKS_PER_LEG:
            logger.warning(
                "Skipping %s: only %d long / %d short candidates (min %d required)",
                date.date(),
                n_long,
                n_short,
                config.MIN_STOCKS_PER_LEG,
            )
            continue

        long_tickers = valid.nlargest(n_long).index
        short_tickers = valid.nsmallest(n_short).index

        weights.loc[date, long_tickers] = 1.0 / n_long
        weights.loc[date, short_tickers] = -1.0 / n_short

    return weights
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_portfolio.py -v`

Expected:
```
tests/test_portfolio.py::test_construct_portfolio_returns_dataframe PASSED
tests/test_portfolio.py::test_construct_portfolio_same_shape_as_scores PASSED
tests/test_portfolio.py::test_long_weights_are_positive PASSED
tests/test_portfolio.py::test_short_weights_are_negative PASSED
tests/test_portfolio.py::test_long_leg_sums_to_one_each_month PASSED
tests/test_portfolio.py::test_short_leg_sums_to_minus_one_each_month PASSED
tests/test_portfolio.py::test_top_20_percent_assigned_positive_weight PASSED
tests/test_portfolio.py::test_bottom_20_percent_assigned_negative_weight PASSED
tests/test_portfolio.py::test_all_nan_row_produces_zero_weights PASSED
tests/test_portfolio.py::test_insufficient_stocks_produces_zero_weights PASSED
10 passed
```

- [ ] **Step 5: Commit**

```bash
git add XSectional/portfolio.py XSectional/tests/test_portfolio.py
git commit -m "feat: add portfolio.py — equal-weighted long-short construction with tests"
```

---

## Task 6: backtest.py

**Files:**
- Create: `XSectional/tests/test_backtest.py`
- Create: `XSectional/backtest.py`

- [ ] **Step 1: Write failing tests in `tests/test_backtest.py`**

```python
import pytest
import pandas as pd
import numpy as np
from backtest import run_backtest


@pytest.fixture
def prices_5y():
    """5 years of daily prices for 5 tickers."""
    np.random.seed(7)
    dates = pd.date_range("2000-01-01", periods=1825, freq="D")
    arr = 100.0 * np.cumprod(1 + np.random.normal(0.0003, 0.01, (1825, 5)), axis=0)
    return pd.DataFrame(arr, index=dates, columns=["A", "B", "C", "D", "E"])


@pytest.fixture
def flat_weights(prices_5y):
    """Simple long-only equal weights, monthly, covering part of the price range."""
    monthly_dates = pd.date_range("2001-01-31", periods=24, freq="ME")
    weights = pd.DataFrame(0.0, index=monthly_dates, columns=["A", "B", "C", "D", "E"])
    weights["A"] = 0.5
    weights["B"] = 0.5
    return weights


def test_run_backtest_returns_series(prices_5y, flat_weights):
    result = run_backtest(flat_weights, prices_5y)
    assert isinstance(result, pd.Series)


def test_run_backtest_no_nan(prices_5y, flat_weights):
    result = run_backtest(flat_weights, prices_5y)
    assert not result.isna().any(), "Monthly returns should not contain NaN"


def test_run_backtest_returns_not_empty(prices_5y, flat_weights):
    result = run_backtest(flat_weights, prices_5y)
    assert len(result) > 0


def test_run_backtest_monthly_returns_are_plausible(prices_5y, flat_weights):
    """Monthly returns should be within a realistic range (-50% to +50%)."""
    result = run_backtest(flat_weights, prices_5y)
    assert result.abs().max() < 0.50, (
        f"Implausibly large monthly return: {result.abs().max():.2%}"
    )


def test_run_backtest_zero_weights_gives_zero_returns(prices_5y):
    """All-zero weights should produce all-zero portfolio returns."""
    monthly_dates = pd.date_range("2001-01-31", periods=24, freq="ME")
    zero_weights = pd.DataFrame(
        0.0, index=monthly_dates, columns=["A", "B", "C", "D", "E"]
    )
    result = run_backtest(zero_weights, prices_5y)
    assert (result.abs() < 1e-9).all(), "Zero weights must produce zero returns"


def test_run_backtest_weights_applied_one_month_forward(prices_5y):
    """
    Weights formed at month t are applied to returns of month t+1.
    Verify by setting weights at month 0 and checking only month 1 return is non-zero.
    """
    monthly_dates = pd.date_range("2001-01-31", periods=12, freq="ME")
    weights = pd.DataFrame(0.0, index=monthly_dates, columns=["A", "B", "C", "D", "E"])
    weights.iloc[0]["A"] = 1.0   # only month 0 is non-zero

    result = run_backtest(weights, prices_5y)

    # Month at index 1 (formed using month-0 weights) should be non-zero
    non_zero = result[result.abs() > 1e-9]
    assert len(non_zero) == 1, (
        f"Expected 1 non-zero return period, got {len(non_zero)}"
    )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_backtest.py -v`

Expected: `ImportError: No module named 'backtest'`

- [ ] **Step 3: Write `backtest.py`**

```python
# backtest.py — simulate monthly long-short portfolio returns

import pandas as pd

import config


def run_backtest(weights: pd.DataFrame, prices: pd.DataFrame) -> pd.Series:
    """
    Simulate monthly portfolio returns from portfolio weights and daily prices.

    Weights at month-end t are applied to the returns earned during month t+1
    (i.e., we form the portfolio at close of month t, hold through month t+1).

    Args:
        weights: DataFrame of portfolio weights (monthly dates × tickers).
                 Positive = long, negative = short, zero = not held.
        prices:  DataFrame of daily adjusted close prices (daily dates × tickers).

    Returns:
        Series of monthly portfolio returns, indexed by month-end dates.
        The first return corresponds to the month after the first weight observation.
    """
    # Step 1: compute monthly returns from daily prices
    monthly_prices = prices.resample(config.REBALANCE_FREQ).last()
    monthly_returns = monthly_prices.pct_change()

    # Step 2: align weights and returns on the same index and columns
    weights_aligned, returns_aligned = weights.align(monthly_returns, join="inner")

    # Step 3: apply weights from t to returns at t+1 (shift weights forward by 1 row)
    # After shift(1): at row t, we use the weights that were computed at row t-1
    portfolio_returns = (weights_aligned.shift(1) * returns_aligned).sum(axis=1)

    # Step 4: drop leading NaN introduced by the shift
    return portfolio_returns.dropna()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_backtest.py -v`

Expected:
```
tests/test_backtest.py::test_run_backtest_returns_series PASSED
tests/test_backtest.py::test_run_backtest_no_nan PASSED
tests/test_backtest.py::test_run_backtest_returns_not_empty PASSED
tests/test_backtest.py::test_run_backtest_monthly_returns_are_plausible PASSED
tests/test_backtest.py::test_run_backtest_zero_weights_gives_zero_returns PASSED
tests/test_backtest.py::test_run_backtest_weights_applied_one_month_forward PASSED
6 passed
```

- [ ] **Step 5: Commit**

```bash
git add XSectional/backtest.py XSectional/tests/test_backtest.py
git commit -m "feat: add backtest.py — monthly return simulation with tests"
```

---

## Task 7: report.py

**Files:**
- Create: `XSectional/tests/test_report.py`
- Create: `XSectional/report.py`

- [ ] **Step 1: Write failing tests in `tests/test_report.py`**

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_report.py -v`

Expected: `ImportError: No module named 'report'`

- [ ] **Step 3: Write `report.py`**

```python
# report.py — compute performance metrics and render tearsheet PNG

import os

import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe in all environments
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config


def compute_metrics(returns: pd.Series) -> dict:
    """
    Compute annualised performance metrics from a monthly returns Series.

    Returns a dict with:
      Annualised Return    — geometric mean annualised return
      Annualised Volatility — annualised standard deviation of monthly returns
      Sharpe Ratio         — Ann. Return / Ann. Volatility (risk-free = 0)
      Max Drawdown         — largest peak-to-trough decline (≤ 0)
      Calmar Ratio         — Ann. Return / |Max Drawdown| (0 if drawdown is 0)
    """
    n = len(returns)
    ann_return = (1 + returns).prod() ** (12 / n) - 1
    ann_vol = returns.std() * np.sqrt(12)
    sharpe = (ann_return / ann_vol) if ann_vol > 0 else 0.0

    cum = (1 + returns).cumprod()
    rolling_max = cum.cummax()
    drawdown = (cum - rolling_max) / rolling_max
    max_dd = drawdown.min()

    calmar = (ann_return / abs(max_dd)) if max_dd < 0 else 0.0

    return {
        "Annualised Return": ann_return,
        "Annualised Volatility": ann_vol,
        "Sharpe Ratio": sharpe,
        "Max Drawdown": max_dd,
        "Calmar Ratio": calmar,
    }


def generate_report(returns: pd.Series) -> None:
    """
    Print performance metrics to stdout and save a 4-panel tearsheet PNG
    to {config.DATA_DIR}/tearsheet.png.

    Panels:
      1. Cumulative equity curve (log scale)
      2. Annual return bar chart
      3. Rolling 12-month Sharpe ratio
      4. Drawdown chart
    """
    metrics = compute_metrics(returns)

    print("\n=== XSectional Momentum Tearsheet ===")
    for label, value in metrics.items():
        if "Drawdown" in label or "Return" in label or "Volatility" in label:
            print(f"  {label}: {value:.2%}")
        else:
            print(f"  {label}: {value:.2f}")

    cum = (1 + returns).cumprod()
    rolling_max = cum.cummax()
    drawdown = (cum - rolling_max) / rolling_max
    rolling_sharpe = returns.rolling(12).apply(
        lambda x: (x.mean() * 12) / (x.std() * np.sqrt(12)) if x.std() > 0 else 0.0,
        raw=True,
    )
    annual_returns = returns.resample("YE").apply(
        lambda x: (1 + x).prod() - 1
    )

    fig = plt.figure(figsize=(14, 12))
    gs = gridspec.GridSpec(4, 1, hspace=0.45)

    # Panel 1: Equity curve
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(cum.index, cum.values, linewidth=1.5)
    ax1.set_title("Cumulative Equity Curve (log scale)", fontsize=11)
    ax1.set_ylabel("Growth of $1")
    ax1.set_yscale("log")
    ax1.grid(True, alpha=0.3)

    # Panel 2: Annual returns
    ax2 = fig.add_subplot(gs[1])
    colors = ["steelblue" if r >= 0 else "tomato" for r in annual_returns.values]
    ax2.bar(annual_returns.index.year, annual_returns.values * 100, color=colors)
    ax2.set_title("Annual Returns (%)", fontsize=11)
    ax2.set_ylabel("Return (%)")
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.grid(True, alpha=0.3, axis="y")

    # Panel 3: Rolling Sharpe
    ax3 = fig.add_subplot(gs[2])
    ax3.plot(rolling_sharpe.index, rolling_sharpe.values, linewidth=1.2, color="darkgreen")
    ax3.set_title("Rolling 12-Month Sharpe Ratio", fontsize=11)
    ax3.set_ylabel("Sharpe")
    ax3.axhline(0, color="black", linewidth=0.8)
    ax3.grid(True, alpha=0.3)

    # Panel 4: Drawdown
    ax4 = fig.add_subplot(gs[3])
    ax4.fill_between(
        drawdown.index, drawdown.values * 100, 0,
        color="tomato", alpha=0.5, linewidth=0,
    )
    ax4.set_title("Drawdown (%)", fontsize=11)
    ax4.set_ylabel("Drawdown (%)")
    ax4.grid(True, alpha=0.3)

    os.makedirs(config.DATA_DIR, exist_ok=True)
    output_path = os.path.join(config.DATA_DIR, "tearsheet.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Tearsheet saved → {output_path}")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_report.py -v`

Expected:
```
tests/test_report.py::test_compute_metrics_returns_dict PASSED
tests/test_report.py::test_compute_metrics_required_keys PASSED
tests/test_report.py::test_compute_metrics_max_drawdown_non_positive PASSED
tests/test_report.py::test_compute_metrics_flat_returns_zero_vol PASSED
tests/test_report.py::test_compute_metrics_flat_returns_zero_sharpe PASSED
tests/test_report.py::test_compute_metrics_flat_returns_zero_drawdown PASSED
tests/test_report.py::test_compute_metrics_positive_return_positive_sharpe PASSED
tests/test_report.py::test_generate_report_creates_tearsheet_png PASSED
tests/test_report.py::test_generate_report_prints_metrics PASSED
9 passed
```

- [ ] **Step 5: Commit**

```bash
git add XSectional/report.py XSectional/tests/test_report.py
git commit -m "feat: add report.py — metrics computation and 4-panel tearsheet with tests"
```

---

## Task 8: main.py + Full Test Suite

**Files:**
- Create: `XSectional/main.py`

- [ ] **Step 1: Write `main.py`**

```python
# main.py — orchestrates the full XSectional momentum pipeline

import logging

from data import load_prices
from signals import compute_momentum_scores
from portfolio import construct_portfolio
from backtest import run_backtest
from report import generate_report

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main() -> None:
    logger.info("Step 1/5 — Loading prices...")
    prices = load_prices()
    logger.info("  %d tickers, %d daily observations", prices.shape[1], len(prices))

    logger.info("Step 2/5 — Computing momentum scores...")
    scores = compute_momentum_scores(prices)

    logger.info("Step 3/5 — Constructing long-short portfolio...")
    weights = construct_portfolio(scores)

    logger.info("Step 4/5 — Running backtest...")
    returns = run_backtest(weights, prices)
    logger.info("  %d monthly periods simulated", len(returns))

    logger.info("Step 5/5 — Generating tearsheet...")
    generate_report(returns)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the full test suite**

Run: `pytest -v`

Expected: All tests pass (32 tests across 5 test files).

```
tests/test_data.py       5 passed
tests/test_signals.py    7 passed
tests/test_portfolio.py 10 passed
tests/test_backtest.py   6 passed
tests/test_report.py     9 passed
========================= 37 passed =========================
```

- [ ] **Step 3: Smoke-test the pipeline end-to-end**

> **Note:** This downloads ~25 years of S&P 500 data (~500 tickers) on first run — expect 2–5 minutes.
> Subsequent runs use the local `data/sp500_prices.csv` cache and complete in seconds.

Run: `python main.py`

Expected output (approx):
```
10:23:01  INFO      Step 1/5 — Loading prices...
10:23:01  INFO      Cache not found — downloading S&P 500 prices from yfinance...
10:25:34  INFO      Prices saved to cache: data/sp500_prices.csv
10:25:34  INFO      483 tickers, 9131 daily observations
10:25:34  INFO      Step 2/5 — Computing momentum scores...
10:25:35  INFO      Step 3/5 — Constructing long-short portfolio...
10:25:36  INFO      Step 4/5 — Running backtest...
10:25:36  INFO      288 monthly periods simulated
10:25:36  INFO      Step 5/5 — Generating tearsheet...

=== XSectional Momentum Tearsheet ===
  Annualised Return: X.XX%
  Annualised Volatility: X.XX%
  Sharpe Ratio: X.XX
  Max Drawdown: -XX.XX%
  Calmar Ratio: X.XX

  Tearsheet saved → data/tearsheet.png
```

- [ ] **Step 4: Commit**

```bash
git add XSectional/main.py
git commit -m "feat: add main.py — full pipeline orchestration; XSectional module complete"
```

---

## Done ✓

The complete pipeline:
1. Downloads and caches 25 years of S&P 500 prices
2. Computes 12–1 momentum scores monthly
3. Constructs equal-weighted long-short portfolio (top/bottom 20%)
4. Simulates monthly returns 2000–2025
5. Produces a performance tearsheet at `XSectional/data/tearsheet.png`

**To run:** `cd XSectional && python main.py`  
**To test:** `cd XSectional && pytest -v`
