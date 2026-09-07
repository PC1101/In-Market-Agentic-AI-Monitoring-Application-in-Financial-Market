# Phase 4: Energy Strategy PnL Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce the two strategy PnL curves (JT momentum + AL PCA stat-arb) for the curated global-energy universe, expose them behind a `StrategyPnLProvider`, register the energy `MarketProfile`, and run the energy market end-to-end.

**Architecture:** Reuse the existing backtest engines rather than rewrite them. JT momentum's core (`XSectional/{signals,portfolio,backtest}.py`) is pure functions on a price DataFrame, so the energy JT curve is a thin runner feeding it the energy universe prices (no S&P 500 membership mask — the energy universe is a fixed curated set). AL stat-arb's `bt` class supports `defactoring='pca'`: for a single-sector universe, run it once with PCA defactoring (PCA strips the common energy/oil factor; the strategy trades the mean-reverting residuals) instead of the 11 sector-ETF sleeves. Both engines emit the canonical `Date,port_ret,equity,drawdown` curve that `pnl_loader.load_pnl` reads. `EnergyPnL` then serves those curves through the provider interface, and the entry points route energy via `profile.pnl` (lifting the Phase-1 fail-closed guard for energy only).

**Tech Stack:** Python (repo `.venv`), pandas, numpy, yfinance (energy prices), the existing `XSectional` + `Stat Arb/statsArb-dev` engines. Heavy AL PCA run (~30–45 min) targets vast.ai.

**Design reference:** `docs/superpowers/specs/2026-08-19-multi-market-abstraction-vastai-design.md` §6, §11.2/.6. Universe/news/macro/windows providers already built (`providers/energy/{universe,news,macro,windows}.py`); this plan adds the last provider (PnL) + profile + wiring.

**Prerequisites:** run from inside each engine dir with the venv active (`source ../.venv/bin/activate` from `monitoring/`, or the repo `.venv`). yfinance already installed. AL PCA run is slow — validate with a short window locally, full window on vast.ai.

---

## File Structure

| File | Responsibility |
|---|---|
| `XSectional/run_energy_pnl.py` | Energy JT momentum runner → `results/equity_curve_energy.csv` |
| `Stat Arb/statsArb-dev/scripts/build_energy_prices.py` | Build energy prices CSV (yfinance → bt input format) |
| `Stat Arb/statsArb-dev/run_energy_al.py` | Energy AL PCA runner (`defactoring='pca'`, single universe) → energy AL curve |
| `monitoring/providers/energy/pnl.py` | `EnergyPnL` StrategyPnLProvider (loads both curves) |
| `monitoring/providers/energy/profile.py` | Build + register the `energy` `MarketProfile` |
| `monitoring/run_classical.py`, `run_agentic.py` | Route `--market energy` via `profile.pnl` (lift fail-closed for energy) |
| `scripts/vast/job_energy_al.yaml` | vast.ai job spec for the heavy AL PCA run |
| `monitoring/tests/test_energy_pnl.py` | Provider + profile contract tests |

**Curve output convention:** both runners write `Date,port_ret,equity,drawdown` (the schema `pnl_loader.load_pnl` requires; verified against `XSectional/results/equity_curve_daily.csv`). Energy curves live under each engine's `results/` and are referenced by `providers/energy/pnl.py` via absolute paths from repo root.

---

## Task 1: Energy JT momentum curve

**Files:**
- Create: `XSectional/run_energy_pnl.py`
- Test: `monitoring/tests/test_energy_pnl.py` (curve-schema assertion)

- [ ] **Step 1: Write the failing test**

```python
# monitoring/tests/test_energy_pnl.py
from pathlib import Path
import pandas as pd
from pnl_loader import load_pnl, REQUIRED_COLUMNS

REPO = Path(__file__).resolve().parents[2]
JT_ENERGY = REPO / "XSectional" / "results" / "equity_curve_energy.csv"


def test_energy_jt_curve_has_valid_schema():
    assert JT_ENERGY.exists(), "run: python XSectional/run_energy_pnl.py"
    df = load_pnl(JT_ENERGY)               # raises if schema wrong
    assert list(df.columns) == list(REQUIRED_COLUMNS)
    assert len(df) > 250                    # multi-year daily curve
    assert df["equity"].iloc[-1] > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd monitoring && python -m pytest tests/test_energy_pnl.py::test_energy_jt_curve_has_valid_schema -q`
Expected: FAIL — file does not exist.

- [ ] **Step 3: Write the energy JT runner**

Reuses the market-agnostic momentum core; swaps the price source for the energy
universe and drops the S&P 500 PIT membership mask (energy is a fixed curated set).

```python
# XSectional/run_energy_pnl.py
"""JT momentum daily PnL for the curated global-energy universe.

Reuses the XSectional momentum core (signals/portfolio/backtest); the only change
from run_daily_pnl.py is the price source (energy universe via yfinance) and the
absence of an S&P 500 membership mask — the energy universe is a fixed curated set,
so every name is a candidate for its whole listed life.
"""
import logging
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "monitoring"))

from signals import compute_momentum_scores
from portfolio import construct_portfolio
from backtest import run_backtest_daily, write_daily_equity_curve
from providers.energy.universe import load_prices  # curated energy prices (yfinance)

logging.basicConfig(level=logging.INFO)
RESULTS = Path(__file__).resolve().parent / "results"


def main() -> None:
    prices = load_prices(start="2016-06-01", end="2022-12-31")  # covers energy windows + lookback
    prices = prices.dropna(axis=1, how="all")
    logging.info("energy universe: %d tickers, %d days", prices.shape[1], len(prices))
    scores = compute_momentum_scores(prices)
    weights = construct_portfolio(scores)
    daily = run_backtest_daily(weights, prices)
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / "equity_curve_energy.csv"
    write_daily_equity_curve(daily, str(out))
    logging.info("wrote %s (%d days, final equity %.3f)", out, len(daily), daily["equity"].iloc[-1])


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the runner, then the test**

Run: `cd XSectional && python run_energy_pnl.py && cd ../monitoring && python -m pytest tests/test_energy_pnl.py::test_energy_jt_curve_has_valid_schema -q`
Expected: runner logs ~34 tickers and writes the curve; test PASSES.

> If `compute_momentum_scores` needs a minimum lookback (it uses a 12-1 momentum window), the 2016-06 start gives ~6 months of warm-up before the first energy window (2017). If it errors on too few names per month, lower the portfolio's decile cutoff in `construct_portfolio` is NOT allowed here — instead widen the date range; do not edit the shared core.

- [ ] **Step 5: Commit**

```bash
git add XSectional/run_energy_pnl.py monitoring/tests/test_energy_pnl.py
git commit -m "feat(energy): JT momentum PnL runner (reuses XSectional core)"
```

---

## Task 2: Energy prices CSV for the AL engine

**Files:**
- Create: `Stat Arb/statsArb-dev/scripts/build_energy_prices.py`

The `bt` class reads a `prices_file_path` CSV indexed by `Date` with one column per
ticker. Build that for the energy universe from yfinance.

- [ ] **Step 1: Write the builder**

```python
# Stat Arb/statsArb-dev/scripts/build_energy_prices.py
"""Write the curated energy universe's adjusted-close prices as a bt-format CSV."""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "monitoring"))
from providers.energy.universe import load_prices

OUT = Path(__file__).resolve().parents[1] / "data" / "energy_universe" / "prices.csv"


def main() -> None:
    px = load_prices(start="2016-01-01", end="2022-12-31").dropna(axis=1, how="all")
    px.index.name = "Date"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    px.to_csv(OUT)
    print(f"wrote {OUT}: {px.shape[1]} tickers x {len(px)} days")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it and verify the CSV shape**

Run: `cd "Stat Arb/statsArb-dev" && python scripts/build_energy_prices.py`
Expected: `wrote .../data/energy_universe/prices.csv: ~34 tickers x ~1750 days`.

- [ ] **Step 3: Commit**

```bash
git add "Stat Arb/statsArb-dev/scripts/build_energy_prices.py"
git commit -m "feat(energy): build energy universe prices CSV for the AL engine"
```

---

## Task 3: Energy PCA factor returns

**Files:**
- Read: `Stat Arb/statsArb-dev/src/pca_factoring.py` (understand the factor-build API)
- Create: `Stat Arb/statsArb-dev/scripts/build_energy_pca.py`

`bt` reads PCA factor returns from `results/pca_factoring/ret_pca_port.csv`. For the
energy universe, generate the analogous file from the energy prices.

- [ ] **Step 1: Read `src/pca_factoring.py`** and identify the function that turns a
  price/returns frame into the PCA factor-portfolio returns (the producer of
  `ret_pca_port.csv`). Note its exact signature and output columns.

- [ ] **Step 2: Write the energy PCA builder** calling that function on the energy
  prices, writing `results/pca_factoring/ret_pca_port_energy.csv`.

```python
# Stat Arb/statsArb-dev/scripts/build_energy_pca.py
"""Build PCA factor-portfolio returns for the energy universe (input to bt pca mode)."""
import os, sys
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
os.chdir(HERE); sys.path.insert(0, str(HERE))
from src import pca_factoring  # noqa: E402  (API confirmed in Step 1)

PRICES = HERE / "data" / "energy_universe" / "prices.csv"
OUT = HERE / "results" / "pca_factoring" / "ret_pca_port_energy.csv"


def main() -> None:
    prices = pd.read_csv(PRICES, index_col="Date", parse_dates=True).sort_index()
    returns = prices.pct_change().dropna(how="all")
    # Replace build_factor_returns(...) with the actual function from Step 1.
    factor_ret = pca_factoring.build_factor_returns(returns)  # TODO(step1): confirm name
    OUT.parent.mkdir(parents=True, exist_ok=True)
    factor_ret.to_csv(OUT)
    print(f"wrote {OUT}: {factor_ret.shape}")


if __name__ == "__main__":
    main()
```

> The `build_factor_returns` name is a placeholder to be replaced with the real
> function identified in Step 1 — do not leave it unconfirmed. If `pca_factoring`
> writes `ret_pca_port.csv` directly with a hardcoded path, add a `--universe energy`
> option or parameterise the output path instead of duplicating logic.

- [ ] **Step 3: Run and verify** the energy factor-returns CSV exists with a
  DatetimeIndex and at least one factor column.

Run: `cd "Stat Arb/statsArb-dev" && python scripts/build_energy_pca.py`

- [ ] **Step 4: Commit**

```bash
git add "Stat Arb/statsArb-dev/scripts/build_energy_pca.py"
git commit -m "feat(energy): PCA factor returns for the energy AL stat-arb"
```

---

## Task 4: Energy AL PCA curve (single-sleeve, defactoring='pca')

**Files:**
- Create: `Stat Arb/statsArb-dev/run_energy_al.py`
- Modify (if needed): make `bt`'s PCA-returns path configurable (energy vs S&P 500)

- [ ] **Step 1: Handle the PCA-returns path.** `src/backtest.py` line ~20 hardcodes
  `results/pca_factoring/ret_pca_port.csv`. Add an optional `pca_ret_path` kwarg to
  `bt.__init__` defaulting to that path, so the energy runner can pass
  `ret_pca_port_energy.csv`. Change is additive (default preserves S&P 500 behaviour).

- [ ] **Step 2: Write the energy AL runner** — one `bt` run with `defactoring='pca'`
  over the whole energy universe, converting the daily combined return to the canonical
  curve.

```python
# Stat Arb/statsArb-dev/run_energy_al.py
"""AL (Avellaneda-Lee) stat-arb on the curated energy universe, PCA-defactored.

Energy is a single sector, so the S&P 500 sector-ETF sleeve scheme does not apply;
instead run the bt engine once with defactoring='pca' over the whole energy universe
(PCA strips the common energy/oil factor; the strategy trades the residuals).
"""
import argparse, os, sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
os.chdir(HERE); sys.path.insert(0, str(HERE))
from src import backtest  # noqa: E402

PRICES = HERE / "data" / "energy_universe" / "prices.csv"
PCA_RET = HERE / "results" / "pca_factoring" / "ret_pca_port_energy.csv"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2016-06-01")
    ap.add_argument("--end", default="2022-12-31")
    args = ap.parse_args()

    model = backtest.bt(
        prices_file_path=str(PRICES), etf_name="XLE",
        st_dt=args.start, ed_dt=args.end,
        defactoring="pca", performance_only=True, progress=True,
        pca_ret_path=str(PCA_RET),          # from Task 4 Step 1
    )
    model.run()
    daily_ret = model.port_ret["cum_ret"].diff().fillna(model.port_ret["cum_ret"].iloc[0])
    equity = (1 + daily_ret).cumprod()
    drawdown = equity / equity.cummax() - 1
    out = pd.DataFrame({"port_ret": daily_ret, "equity": equity, "drawdown": drawdown})
    out.index.name = "Date"
    dst = HERE / "results" / "equity_curve_energy_al.csv"
    out.to_csv(dst)
    print(f"wrote {dst}: {len(out)} days, final equity {equity.iloc[-1]:.3f}")


if __name__ == "__main__":
    main()
```

> Confirm the exact attribute holding the sleeve's daily return by reading `bt.run`
> (line ~62–108 of `src/backtest.py`): it builds `port_ret` with a `cum_ret` column in
> `run_full_universe.run_sector` (`model.port_ret["cum_ret"]`). If the daily return is
> exposed directly, use it instead of differencing the cumulative.

- [ ] **Step 3: Validate on a SHORT window locally** (fast) before the full run:

Run: `cd "Stat Arb/statsArb-dev" && python run_energy_al.py --start 2020-01-01 --end 2020-06-30`
Expected: writes `equity_curve_energy_al.csv`; sanity-check the 2020 oil crash shows a drawdown.

- [ ] **Step 4: Add the AL curve to the test**

```python
# append to monitoring/tests/test_energy_pnl.py
AL_ENERGY = REPO / "Stat Arb" / "statsArb-dev" / "results" / "equity_curve_energy_al.csv"

def test_energy_al_curve_has_valid_schema():
    assert AL_ENERGY.exists(), "run: python 'Stat Arb/statsArb-dev/run_energy_al.py'"
    df = load_pnl(AL_ENERGY)
    assert list(df.columns) == list(REQUIRED_COLUMNS)
    assert len(df) > 100
```

Run: `cd monitoring && python -m pytest tests/test_energy_pnl.py -q` → PASS.

- [ ] **Step 5: Commit**

```bash
git add "Stat Arb/statsArb-dev/run_energy_al.py" "Stat Arb/statsArb-dev/src/backtest.py" monitoring/tests/test_energy_pnl.py
git commit -m "feat(energy): AL PCA stat-arb runner (single-universe pca defactoring)"
```

---

## Task 5: EnergyPnL provider

**Files:**
- Create: `monitoring/providers/energy/pnl.py`
- Test: `monitoring/tests/test_energy_pnl.py` (contract)

- [ ] **Step 1: Write the failing contract test**

```python
# append to monitoring/tests/test_energy_pnl.py
from providers.base import StrategyPnLProvider

def test_energy_pnl_conforms_and_serves_both_strategies():
    from providers.energy.pnl import EnergyPnL
    p = EnergyPnL()
    assert isinstance(p, StrategyPnLProvider)
    assert set(p.strategies()) == {"AL_PCA", "JT_MOM"}
    s = p.returns("JT_MOM")
    assert s.name == "port_ret" and len(s) > 250
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd monitoring && python -m pytest tests/test_energy_pnl.py::test_energy_pnl_conforms_and_serves_both_strategies -q`
Expected: FAIL — module missing.

- [ ] **Step 3: Write EnergyPnL** (mirrors `providers/sp500/pnl.py`; maps each strategy
  to its curve file).

```python
# monitoring/providers/energy/pnl.py
"""Energy StrategyPnLProvider — serves the JT + AL PCA energy curves."""
from __future__ import annotations
from pathlib import Path
import pandas as pd
from pnl_loader import load_pnl, returns_series, as_of as _as_of

ROOT = Path(__file__).resolve().parents[3]
CURVES = {
    "JT_MOM": ROOT / "XSectional" / "results" / "equity_curve_energy.csv",
    "AL_PCA": ROOT / "Stat Arb" / "statsArb-dev" / "results" / "equity_curve_energy_al.csv",
}


class EnergyPnL:
    def strategies(self) -> list[str]:
        return list(CURVES)

    def returns(self, strategy: str, as_of=None) -> pd.Series:
        s = returns_series(load_pnl(CURVES[strategy]))
        s.name = "port_ret"
        return _as_of(s, as_of) if as_of is not None else s
```

- [ ] **Step 4: Run to verify it passes**

Run: `cd monitoring && python -m pytest tests/test_energy_pnl.py -q` → PASS.

- [ ] **Step 5: Commit**

```bash
git add monitoring/providers/energy/pnl.py monitoring/tests/test_energy_pnl.py
git commit -m "feat(energy): EnergyPnL StrategyPnLProvider over JT + AL curves"
```

---

## Task 6: Register the energy MarketProfile

**Files:**
- Create: `monitoring/providers/energy/profile.py`
- Modify: `monitoring/config/profiles.py` (lazy-import branch for `energy`)
- Test: `monitoring/tests/test_energy_pnl.py`

- [ ] **Step 1: Write the failing test**

```python
# append to monitoring/tests/test_energy_pnl.py
from config.profiles import get_profile
from providers.base import NewsProvider, MacroProvider

def test_energy_profile_registers_all_providers():
    prof = get_profile("energy")
    assert prof.key == "energy" and prof.base_ccy == "USD"
    assert isinstance(prof.pnl, StrategyPnLProvider)
    assert isinstance(prof.news, NewsProvider)
    assert isinstance(prof.macro, MacroProvider)
    assert len(prof.windows) >= 4
```

- [ ] **Step 2: Run to verify it fails** (unknown market `energy`).

Run: `cd monitoring && python -m pytest tests/test_energy_pnl.py::test_energy_profile_registers_all_providers -q`

- [ ] **Step 3: Write the profile + add the lazy-import branch**

```python
# monitoring/providers/energy/profile.py
"""Build + register the energy MarketProfile (import side-effect)."""
from __future__ import annotations
from config.profiles import MarketProfile, register
from providers.energy.pnl import EnergyPnL
from providers.energy.news import EnergyNews
from providers.energy.macro import EnergyMacro
from providers.energy.windows import ALL_WINDOWS

register(MarketProfile(
    key="energy", timezone="UTC", base_ccy="USD",
    pnl=EnergyPnL(), news=EnergyNews(), macro=EnergyMacro(),
    windows=list(ALL_WINDOWS),
))
```

In `monitoring/config/profiles.py`, extend the lazy-import in `get_profile`:

```python
        if key == "sp500":
            import providers.sp500.profile  # noqa: F401
        elif key == "energy":
            import providers.energy.profile  # noqa: F401
```

- [ ] **Step 4: Run to verify it passes**

Run: `cd monitoring && python -m pytest tests/test_energy_pnl.py -q` → PASS.

- [ ] **Step 5: Commit**

```bash
git add monitoring/providers/energy/profile.py monitoring/config/profiles.py monitoring/tests/test_energy_pnl.py
git commit -m "feat(energy): register the energy MarketProfile"
```

---

## Task 7: Route `--market energy` through the entry points

**Files:**
- Modify: `monitoring/run_classical.py`, `monitoring/run_agentic.py`
- Test: `monitoring/tests/test_energy_pnl.py`

- [ ] **Step 1: Write the failing test** (energy classical run produces output).

```python
# append to monitoring/tests/test_energy_pnl.py
import subprocess, sys
MON = REPO / "monitoring"

def test_classical_energy_runs():
    out = subprocess.run([sys.executable, "run_classical.py", "--market", "energy", "--model", "stub"],
                         cwd=MON, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
```

- [ ] **Step 2: Run to verify it fails** — the Phase-1 guard raises `SystemExit` for
  non-sp500 markets.

- [ ] **Step 3: Generalise the curve routing.** In `run_classical.py`, replace the
  `if profile.key != "sp500": raise SystemExit(...)` guard with generic routing:

```python
    profile = get_profile(args.market)
    if args.market == "sp500":
        from providers.sp500.pnl import STRATEGY_CURVES  # bit-identical US path
    else:
        # Generic path: build {strategy: [curve paths]} from the profile provider.
        STRATEGY_CURVES = {s: None for s in profile.pnl.strategies()}
        # run loop uses profile.pnl.returns(strat) directly when paths are None.
```

Then in the run loop, when `STRATEGY_CURVES[strat]` is `None`, obtain the series via
`profile.pnl.returns(strat)` instead of `load_pnl(path)`. Keep the sp500 path exactly
as-is so the golden-path test stays bit-identical.

> Do the minimal branch: sp500 keeps the path-based loop (proven bit-identical);
> energy uses `profile.pnl.returns(strat)`. Verify `test_golden_path_sp500.py` still
> passes after this change — that is the guardrail.

- [ ] **Step 4: Mirror the change in `run_agentic.py`** (same guard → generic routing).

- [ ] **Step 5: Run the tests**

Run: `cd monitoring && python -m pytest tests/test_energy_pnl.py tests/test_golden_path_sp500.py -q`
Expected: energy runs; **sp500 still bit-identical**.

- [ ] **Step 6: Commit**

```bash
git add monitoring/run_classical.py monitoring/run_agentic.py monitoring/tests/test_energy_pnl.py
git commit -m "feat(energy): route --market energy via profile.pnl (sp500 still bit-identical)"
```

---

## Task 8: vast.ai job for the heavy AL PCA run

**Files:**
- Create: `scripts/vast/job_energy_al.yaml`

- [ ] **Step 1: Write the job spec** so the ~30–45 min AL PCA backtest runs on a rented
  GPU/CPU box rather than the laptop.

```yaml
# scripts/vast/job_energy_al.yaml
market: energy
strategy: AL_PCA
window: oil_crash_2020
model: stub            # the backtest is CPU-bound; agentic layer uses stub here
image: python:3.12-slim
disk_gb: 20
# run: build_energy_prices.py -> build_energy_pca.py -> run_energy_al.py
```

- [ ] **Step 2: Extend `launch.py`** (or document a job-command hook) so a job can run a
  shell sequence, not just `run_agentic.py`. Minimal: add an optional `run:` list to the
  job YAML that `run_job` executes over ssh in order. Keep `--dry-run` behaviour.

- [ ] **Step 3: Dry-run it** (no spend):

Run: `python scripts/vast/launch.py --job scripts/vast/job_energy_al.yaml --dry-run`

- [ ] **Step 4: Commit**

```bash
git add scripts/vast/job_energy_al.yaml scripts/vast/launch.py
git commit -m "feat(vast): energy AL PCA job spec + multi-step run hook"
```

---

## Self-Review notes

- **Spec coverage:** implements §11.6 (reuse AL PCA + JT) and §11.2 (curated energy
  universe) end-to-end; completes the 5th energy provider (PnL) and registers the
  `energy` MarketProfile, satisfying §9 Phase 4.
- **Methodological wrinkle resolved:** single-sector energy uses `bt` `defactoring='pca'`
  once over the whole universe (Task 4), not the 11 S&P 500 sector-ETF sleeves — the
  design decision from the checkpoint is settled and documented in the runner docstring.
- **Bit-identical guardrail:** Task 7 keeps the sp500 path untouched; `test_golden_path_sp500.py`
  must stay green after routing energy generically — called out as the guard.
- **Named unknowns flagged, not hidden:** Task 3 Step 1 requires reading `src/pca_factoring.py`
  to confirm the real factor-build function (the `build_factor_returns` name is marked TODO
  to replace, not shipped as-is); Task 4 Step 1 requires confirming the `bt` daily-return
  attribute. These are read-and-confirm steps, not guesses baked into shipped code.
- **Heavy run offloaded:** Task 8 puts the slow AL PCA backtest on the now-validated
  vast.ai harness (the reason Phase 2 exists).
- **Prereq:** the energy commodity macro (`EnergyMacro`) needs `FRED_API_KEY` to fetch;
  the profile registers regardless (macro returns an empty block if unfetched), so the
  classical energy run in Task 7 does not block on the key.
```
