# XSectional 2009–2016 Stress Test Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a 2009–2016 sub-period stress tearsheet to the XSectional pipeline that runs automatically after the full 2000–2025 backtest.

**Architecture:** Slice the existing monthly returns Series to the stress window and pass it to a parameterised version of `generate_report`. No data re-download, no re-run of signal/portfolio/backtest logic.

**Tech Stack:** Python 3.9+, pandas, matplotlib, pytest

---

## File Map

| File | Action | What changes |
|---|---|---|
| `XSectional/config.py` | Modify | Add `STRESS_START` and `STRESS_END` constants |
| `XSectional/report.py` | Modify | Add `label` and `filename` params to `generate_report` |
| `XSectional/main.py` | Modify | Slice returns and call `generate_report` a second time |
| `XSectional/tests/test_report.py` | Modify | Add tests for new `label` / `filename` parameters |

---

### Task 1: Add stress-period constants to `config.py`

**Files:**
- Modify: `XSectional/config.py`

- [ ] **Step 1: Open `config.py` and append the two constants**

Replace the end of `XSectional/config.py` so the full file reads:

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

# Stress-test sub-period
STRESS_START = "2009-01-01"
STRESS_END   = "2016-12-31"
```

- [ ] **Step 2: Verify constants are importable**

```bash
cd XSectional && python3 -c "import config; print(config.STRESS_START, config.STRESS_END)"
```

Expected output:
```
2009-01-01 2016-12-31
```

- [ ] **Step 3: Commit**

```bash
git add XSectional/config.py
git commit -m "feat: add STRESS_START / STRESS_END constants to config"
```

---

### Task 2: Parameterise `generate_report` in `report.py`

**Files:**
- Modify: `XSectional/report.py`
- Test: `XSectional/tests/test_report.py`

- [ ] **Step 1: Write the failing tests**

Add the following tests to `XSectional/tests/test_report.py` (append after the existing tests):

```python
def test_generate_report_custom_filename(monthly_returns, tmp_path, monkeypatch):
    """Custom filename parameter saves PNG under the given name."""
    import config
    monkeypatch.setattr(config, "DATA_DIR", str(tmp_path))
    generate_report(monthly_returns, filename="tearsheet_stress.png")
    assert (tmp_path / "tearsheet_stress.png").exists()


def test_generate_report_custom_label_in_stdout(monthly_returns, tmp_path, monkeypatch, capsys):
    """Custom label appears in the printed header."""
    import config
    monkeypatch.setattr(config, "DATA_DIR", str(tmp_path))
    generate_report(monthly_returns, label="Stress (2009–2016)", filename="tearsheet_stress.png")
    captured = capsys.readouterr()
    assert "Stress (2009–2016)" in captured.out


def test_generate_report_default_filename_unchanged(monthly_returns, tmp_path, monkeypatch):
    """Calling without filename still produces tearsheet.png (backward compat)."""
    import config
    monkeypatch.setattr(config, "DATA_DIR", str(tmp_path))
    generate_report(monthly_returns)
    assert (tmp_path / "tearsheet.png").exists()
```

- [ ] **Step 2: Run to confirm they fail**

```bash
cd XSectional && python3 -m pytest tests/test_report.py -v -k "custom_filename or custom_label or default_filename_unchanged"
```

Expected: **3 FAILED** — `generate_report` does not yet accept `label` or `filename` params.

- [ ] **Step 3: Update `generate_report` signature and body**

Replace the `generate_report` function in `XSectional/report.py` with:

```python
def generate_report(
    returns: pd.Series,
    label: str = "Full (2000–2025)",
    filename: str = "tearsheet.png",
) -> None:
    """
    Print performance metrics to stdout and save a 4-panel tearsheet PNG
    to {config.DATA_DIR}/{filename}.

    Args:
        returns:  Monthly returns Series.
        label:    Human-readable period label shown in the header and panel titles.
        filename: Output PNG filename (saved inside config.DATA_DIR).

    Panels:
      1. Cumulative equity curve (log scale)
      2. Annual return bar chart
      3. Rolling 12-month Sharpe ratio
      4. Drawdown chart
    """
    metrics = compute_metrics(returns)

    print(f"\n=== XSectional Momentum Tearsheet — {label} ===")
    for lbl, value in metrics.items():
        if "Drawdown" in lbl or "Return" in lbl or "Volatility" in lbl:
            print(f"  {lbl}: {value:.2%}")
        else:
            print(f"  {lbl}: {value:.2f}")

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
    ax1.set_title(f"Cumulative Equity Curve (log scale) — {label}", fontsize=11)
    ax1.set_ylabel("Growth of $1")
    ax1.set_yscale("log")
    ax1.grid(True, alpha=0.3)

    # Panel 2: Annual returns
    ax2 = fig.add_subplot(gs[1])
    colors = ["steelblue" if r >= 0 else "tomato" for r in annual_returns.values]
    ax2.bar(annual_returns.index.year, annual_returns.values * 100, color=colors)
    ax2.set_title(f"Annual Returns (%) — {label}", fontsize=11)
    ax2.set_ylabel("Return (%)")
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.grid(True, alpha=0.3, axis="y")

    # Panel 3: Rolling Sharpe
    ax3 = fig.add_subplot(gs[2])
    ax3.plot(rolling_sharpe.index, rolling_sharpe.values, linewidth=1.2, color="darkgreen")
    ax3.set_title(f"Rolling 12-Month Sharpe Ratio — {label}", fontsize=11)
    ax3.set_ylabel("Sharpe")
    ax3.axhline(0, color="black", linewidth=0.8)
    ax3.grid(True, alpha=0.3)

    # Panel 4: Drawdown
    ax4 = fig.add_subplot(gs[3])
    ax4.fill_between(
        drawdown.index, drawdown.values * 100, 0,
        color="tomato", alpha=0.5, linewidth=0,
    )
    ax4.set_title(f"Drawdown (%) — {label}", fontsize=11)
    ax4.set_ylabel("Drawdown (%)")
    ax4.grid(True, alpha=0.3)

    os.makedirs(config.DATA_DIR, exist_ok=True)
    output_path = os.path.join(config.DATA_DIR, filename)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Tearsheet saved → {output_path}")
```

- [ ] **Step 4: Run all report tests**

```bash
cd XSectional && python3 -m pytest tests/test_report.py -v
```

Expected: **all PASSED** (both old and new tests).

- [ ] **Step 5: Commit**

```bash
git add XSectional/report.py XSectional/tests/test_report.py
git commit -m "feat: add label and filename params to generate_report"
```

---

### Task 3: Wire stress test into `main.py`

**Files:**
- Modify: `XSectional/main.py`

- [ ] **Step 1: Update `main.py` to run the stress-test pass**

Replace the full content of `XSectional/main.py` with:

```python
# main.py — orchestrates the full XSectional momentum pipeline

import logging

from data import load_prices
from signals import compute_momentum_scores
from portfolio import construct_portfolio
from backtest import run_backtest
from report import generate_report
import config

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

    logger.info("Step 5/5 — Generating tearsheet (full period)...")
    generate_report(
        returns,
        label=f"Full ({config.START_DATE[:4]}–{config.END_DATE[:4]})",
        filename="tearsheet.png",
    )

    logger.info("Stress test — %s to %s sub-period...", config.STRESS_START[:4], config.STRESS_END[:4])
    stress_returns = returns.loc[config.STRESS_START : config.STRESS_END]
    logger.info("  %d monthly periods in stress window", len(stress_returns))
    generate_report(
        stress_returns,
        label=f"Stress ({config.STRESS_START[:4]}–{config.STRESS_END[:4]})",
        filename=f"tearsheet_stress_{config.STRESS_START[:4]}_{config.STRESS_END[:4]}.png",
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the full test suite to confirm nothing is broken**

```bash
cd XSectional && python3 -m pytest -v
```

Expected: **all tests PASSED**.

- [ ] **Step 3: Smoke-test the pipeline end-to-end (uses cached prices)**

```bash
cd XSectional && python3 main.py
```

Expected output (order may vary slightly):
```
...  INFO     Step 1/5 — Loading prices...
...  INFO       <N> tickers, <M> daily observations
...  INFO     Step 2/5 — Computing momentum scores...
...  INFO     Step 3/5 — Constructing long-short portfolio...
...  INFO     Step 4/5 — Running backtest...
...  INFO       <P> monthly periods simulated
...  INFO     Step 5/5 — Generating tearsheet (full period)...

=== XSectional Momentum Tearsheet — Full (2000–2025) ===
  Annualised Return: ...
  ...
  Tearsheet saved → data/tearsheet.png

...  INFO     Stress test — 2009 to 2016 sub-period...
...  INFO       <Q> monthly periods in stress window

=== XSectional Momentum Tearsheet — Stress (2009–2016) ===
  Annualised Return: ...
  ...
  Tearsheet saved → data/tearsheet_stress_2009_2016.png
```

Confirm both files exist:
```bash
ls -lh XSectional/data/tearsheet*.png
```

Expected:
```
... tearsheet.png
... tearsheet_stress_2009_2016.png
```

- [ ] **Step 4: Commit**

```bash
git add XSectional/main.py
git commit -m "feat: add 2009-2016 stress test tearsheet to pipeline"
```
