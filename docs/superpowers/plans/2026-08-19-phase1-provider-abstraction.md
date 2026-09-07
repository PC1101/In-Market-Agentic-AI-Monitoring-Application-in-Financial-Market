# Phase 1: Provider Abstraction (US S&P 500 refactor) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract today's US S&P 500 data wiring behind stable provider interfaces + a `MarketProfile`, selectable by `--market sp500`, with the existing US results proven **bit-identical** after the refactor.

**Architecture:** Introduce `runtime_checkable` Protocol interfaces in `monitoring/providers/base.py`, a `MarketProfile` dataclass + `REGISTRY` in `monitoring/config/profiles.py`, and a `sp500` provider package that thinly wraps the current `load_pnl` / `NewsStore` / `macro_context` / `windows` code. Entry points (`run_classical.py`, `run_agentic.py`) resolve a profile by `--market` (default `sp500`) instead of module-level constants. No behaviour change for the existing market — a golden-path regression test locks that in.

**Tech Stack:** Python 3 (repo `.venv`), pandas, pytest. No new dependencies.

**Design reference:** `docs/superpowers/specs/2026-08-19-multi-market-abstraction-vastai-design.md` (§2, §3). This plan is Phase 1 of §9 only; it introduces **no new market** and depends on none of the §11 open decisions.

---

## File Structure

| File | Responsibility |
|---|---|
| `monitoring/providers/__init__.py` | Package marker |
| `monitoring/providers/base.py` | `runtime_checkable` Protocols: `StrategyPnLProvider`, `NewsProvider`, `MacroProvider` |
| `monitoring/config/__init__.py` | Package marker |
| `monitoring/config/profiles.py` | `MarketProfile` dataclass, `REGISTRY`, `get_profile()` |
| `monitoring/providers/sp500/__init__.py` | Package marker |
| `monitoring/providers/sp500/pnl.py` | `SP500PnL` — wraps `pnl_loader.load_pnl` + the `STRATEGY_CURVES` path resolution |
| `monitoring/providers/sp500/news.py` | `SP500News` — wraps `news.store.NewsStore` |
| `monitoring/providers/sp500/macro.py` | `SP500Macro` — wraps `macro.context.macro_context` |
| `monitoring/providers/sp500/profile.py` | Builds and registers the `sp500` `MarketProfile` |
| `monitoring/run_classical.py` | Add `--market` (default `sp500`); resolve curves via profile |
| `monitoring/run_agentic.py` | Add `--market` (default `sp500`); resolve curves/news/macro via profile |
| `monitoring/tests/test_providers_contract.py` | Protocol conformance + sp500 provider equivalence |
| `monitoring/tests/test_golden_path_sp500.py` | Bit-identical classical output vs captured baseline |

**Convention note:** existing modules import as top-level (`from pnl_loader import ...`, `from macro.asof import ...`) because `monitoring/` is the working root for pytest/entry points (see `monitoring/pytest.ini`, `CLAUDE.md`). New code follows the same style: `from providers.base import ...`, `from config.profiles import ...`. Run all commands from inside `monitoring/` with the venv active: `cd monitoring && source ../.venv/bin/activate`.

---

## Task 0: Capture the golden baseline (do this FIRST, on clean `main` state)

**Files:**
- Create: `monitoring/tests/golden/sp500_classical_stub.json` (captured artifact)

- [ ] **Step 1: Confirm working tree is on the design branch, clean**

Run: `git status --short`
Expected: only untracked `docs/superpowers/plans/...` — no modified source.

- [ ] **Step 2: Run the current classical pipeline and capture its output verbatim**

```bash
cd monitoring && source ../.venv/bin/activate
mkdir -p tests/golden
python run_classical.py --model stub > tests/golden/sp500_classical_stub.stdout.txt 2>&1
# Copy the machine-readable results the script writes (JSON) into the golden dir:
cp results/classical_summary.json tests/golden/sp500_classical_stub.json
```
Expected: a `results/classical_summary.json` exists after the run. If the entry point writes a different filename, capture that exact file instead (grep `run_classical.py` for `json.dump`/`to_json` to find it).

- [ ] **Step 3: Record the exact captured file for later assertion**

Run: `python -c "import json,hashlib;print(hashlib.sha256(open('tests/golden/sp500_classical_stub.json','rb').read()).hexdigest())"`
Expected: prints a sha256 — note it in the commit message.

- [ ] **Step 4: Commit the baseline**

```bash
git add monitoring/tests/golden/sp500_classical_stub.json
git commit -m "test: capture sp500 classical golden baseline (pre-refactor)"
```

> If the classical run needs data not present locally (curves under `Stat Arb/.../results`), stop and tell the user — Phase 1 cannot prove bit-identical without the baseline data. This is the one hard prerequisite.

---

## Task 1: Provider Protocols

**Files:**
- Create: `monitoring/providers/__init__.py`
- Create: `monitoring/providers/base.py`
- Test: `monitoring/tests/test_providers_contract.py`

- [ ] **Step 1: Write the failing test**

```python
# monitoring/tests/test_providers_contract.py
import pandas as pd
from providers.base import StrategyPnLProvider, NewsProvider, MacroProvider


def test_a_minimal_impl_satisfies_strategy_pnl_protocol():
    class Fake:
        def strategies(self):
            return ["AL_PCA"]
        def returns(self, strategy, as_of=None):
            return pd.Series([0.0], index=pd.to_datetime(["2007-08-01"]), name="port_ret")
    assert isinstance(Fake(), StrategyPnLProvider)


def test_missing_method_fails_protocol():
    class Broken:
        def strategies(self):
            return []
    assert not isinstance(Broken(), StrategyPnLProvider)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_providers_contract.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'providers.base'`.

- [ ] **Step 3: Write minimal implementation**

```python
# monitoring/providers/__init__.py
"""Market-agnostic data provider interfaces and per-market implementations."""
```

```python
# monitoring/providers/base.py
"""Provider interfaces every market implements.

Each method is *as-of dated*: an ``as_of`` date means "return only what was
knowable on that date". Concrete providers wrap a market's real sources; the
pipeline depends only on these Protocols, never on a specific market.
"""
from __future__ import annotations

from datetime import date
from typing import Protocol, runtime_checkable

import pandas as pd


@runtime_checkable
class StrategyPnLProvider(Protocol):
    def strategies(self) -> list[str]:
        """Names of strategies this market exposes (e.g. ['AL_PCA', 'JT_MOM'])."""
        ...

    def returns(self, strategy: str, as_of: date | None = None) -> pd.Series:
        """Daily port_ret series for ``strategy``, sliced to <= as_of if given."""
        ...


@runtime_checkable
class NewsProvider(Protocol):
    def query(self, start, end, tickers: list[str] | None = None) -> pd.DataFrame:
        """News rows with publication date in [start, end], optionally by ticker."""
        ...


@runtime_checkable
class MacroProvider(Protocol):
    def context(self, as_of: date) -> dict:
        """As-of-correct JSON-serialisable macro block for the agent."""
        ...
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_providers_contract.py -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add monitoring/providers/__init__.py monitoring/providers/base.py monitoring/tests/test_providers_contract.py
git commit -m "feat(providers): runtime-checkable provider Protocols"
```

---

## Task 2: MarketProfile + REGISTRY

**Files:**
- Create: `monitoring/config/__init__.py`
- Create: `monitoring/config/profiles.py`
- Test: add to `monitoring/tests/test_providers_contract.py`

- [ ] **Step 1: Write the failing test**

```python
# append to monitoring/tests/test_providers_contract.py
import pytest
from config.profiles import MarketProfile, get_profile, REGISTRY


def test_unknown_market_raises():
    with pytest.raises(KeyError):
        get_profile("does_not_exist")


def test_profile_dataclass_holds_providers():
    prof = MarketProfile(key="x", timezone="UTC", base_ccy="USD",
                         pnl=None, news=None, macro=None, windows=[])
    assert prof.key == "x" and prof.timezone == "UTC"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_providers_contract.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'config.profiles'`.

- [ ] **Step 3: Write minimal implementation**

```python
# monitoring/config/__init__.py
"""Per-market profile definitions."""
```

```python
# monitoring/config/profiles.py
"""MarketProfile: names the concrete providers + calendar for one market."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class MarketProfile:
    key: str
    timezone: str
    base_ccy: str
    pnl: Any            # StrategyPnLProvider
    news: Any           # NewsProvider
    macro: Any          # MacroProvider
    windows: list = field(default_factory=list)


#: Populated by each market package's register() call (import side-effect).
REGISTRY: dict[str, MarketProfile] = {}


def register(profile: MarketProfile) -> None:
    REGISTRY[profile.key] = profile


def get_profile(key: str) -> MarketProfile:
    if key not in REGISTRY:
        # Import known markets lazily so entry points need not import each package.
        if key == "sp500":
            import providers.sp500.profile  # noqa: F401  (registers on import)
    if key not in REGISTRY:
        raise KeyError(f"unknown market '{key}'; registered: {sorted(REGISTRY)}")
    return REGISTRY[key]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_providers_contract.py -q`
Expected: PASS (4 passed). The `get_profile("does_not_exist")` path tries the sp500 import branch only for `"sp500"`, so the unknown key still raises `KeyError`.

- [ ] **Step 5: Commit**

```bash
git add monitoring/config/ monitoring/tests/test_providers_contract.py
git commit -m "feat(config): MarketProfile dataclass + REGISTRY"
```

---

## Task 3: SP500 providers (wrap existing code)

**Files:**
- Create: `monitoring/providers/sp500/__init__.py`, `pnl.py`, `news.py`, `macro.py`, `profile.py`
- Test: add to `monitoring/tests/test_providers_contract.py`

- [ ] **Step 1: Write the failing test (equivalence to current code path)**

```python
# append to monitoring/tests/test_providers_contract.py
def test_sp500_pnl_matches_direct_load():
    from config.profiles import get_profile
    from pnl_loader import load_pnl, returns_series
    from providers.sp500.pnl import AL_CURVE_PATHS  # list[Path] used by the wrapper

    prof = get_profile("sp500")
    direct = returns_series(load_pnl(AL_CURVE_PATHS[0]))
    via = prof.pnl.returns("AL_PCA")
    # Same first curve, same values head-to-head (wrapper concatenates in order).
    pd.testing.assert_series_equal(via.loc[direct.index], direct)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_providers_contract.py::test_sp500_pnl_matches_direct_load -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'providers.sp500'`.

- [ ] **Step 3: Write minimal implementation**

Reuse the exact path constants currently in `run_classical.py` (`STAT_ARB`, `JT_DAILY`, `STRATEGY_CURVES`, `_al_curve`) so the wrapper is provably the same source of truth.

```python
# monitoring/providers/sp500/__init__.py
"""US S&P 500 provider implementations (reference market)."""
```

```python
# monitoring/providers/sp500/pnl.py
"""S&P 500 strategy PnL: same curve paths run_classical.py uses today."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from pnl_loader import load_pnl, returns_series, as_of as _as_of

ROOT = Path(__file__).resolve().parents[3]  # repo root
STAT_ARB = ROOT / "Stat Arb" / "statsArb-dev" / "results" / "full_universe"
JT_DAILY = ROOT / "XSectional" / "results" / "equity_curve_daily.csv"


def _al_curve(tag: str) -> Path:
    pit = STAT_ARB / f"{tag}_pit" / "equity_curve.csv"
    return pit if pit.exists() else STAT_ARB / tag / "equity_curve.csv"


# Keep these lists identical to run_classical.STRATEGY_CURVES.
AL_CURVE_PATHS = [_al_curve("baseline_2007_2015"), _al_curve("calm_2004_2006")]
STRATEGY_CURVES = {"AL_PCA": AL_CURVE_PATHS, "JT_MOM": [JT_DAILY]}


class SP500PnL:
    def strategies(self) -> list[str]:
        return list(STRATEGY_CURVES)

    def returns(self, strategy: str, as_of=None) -> pd.Series:
        paths = [p for p in STRATEGY_CURVES[strategy] if p.exists()]
        parts = [returns_series(load_pnl(p)) for p in paths]
        s = pd.concat(parts).sort_index()
        s = s[~s.index.duplicated(keep="first")]
        s.name = "port_ret"
        return _as_of(s, as_of) if as_of is not None else s
```

```python
# monitoring/providers/sp500/news.py
"""S&P 500 news: wraps the FNSPID NewsStore at its default root."""
from __future__ import annotations

from pathlib import Path

from news.store import NewsStore

DEFAULT_ROOT = Path(__file__).resolve().parents[2] / "news" / "store"


class SP500News:
    def __init__(self, root: str | Path = DEFAULT_ROOT):
        self._store = NewsStore(root)

    def query(self, start, end, tickers=None):
        return self._store.query(start, end, tickers=tickers)
```

```python
# monitoring/providers/sp500/macro.py
"""S&P 500 macro: wraps FRED/ALFRED as-of context."""
from __future__ import annotations

from macro.context import macro_context


class SP500Macro:
    def context(self, as_of) -> dict:
        return macro_context(as_of)
```

```python
# monitoring/providers/sp500/profile.py
"""Build + register the sp500 MarketProfile (import side-effect)."""
from __future__ import annotations

from config.profiles import MarketProfile, register
from providers.sp500.pnl import SP500PnL
from providers.sp500.news import SP500News
from providers.sp500.macro import SP500Macro

try:
    from windows import ALL_WINDOWS  # per-market windows; name-check below
except Exception:
    ALL_WINDOWS = []

register(MarketProfile(
    key="sp500",
    timezone="America/New_York",
    base_ccy="USD",
    pnl=SP500PnL(),
    news=SP500News(),
    macro=SP500Macro(),
    windows=ALL_WINDOWS,
))
```

> **Name-check before running:** confirm the windows export in `monitoring/windows.py`. Run `grep -n "^ALL_WINDOWS\|^WINDOWS\|def .*window" monitoring/windows.py`. If the list is named differently (e.g. `WINDOWS`), fix the import in `profile.py` to match. Do not invent a name.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_providers_contract.py -q`
Expected: PASS (all). If the AL curve files are absent locally, this test errors on missing data — that is the same data prerequisite flagged in Task 0; resolve it there.

- [ ] **Step 5: Commit**

```bash
git add monitoring/providers/sp500/ monitoring/tests/test_providers_contract.py
git commit -m "feat(providers): sp500 provider package wrapping existing sources"
```

---

## Task 4: Wire `--market` into run_classical.py (bit-identical)

**Files:**
- Modify: `monitoring/run_classical.py` (argparse + curve resolution)
- Test: `monitoring/tests/test_golden_path_sp500.py`

- [ ] **Step 1: Write the failing golden-path test**

```python
# monitoring/tests/test_golden_path_sp500.py
import json
import subprocess
import sys
from pathlib import Path

MON = Path(__file__).resolve().parents[1]
GOLDEN = MON / "tests" / "golden" / "sp500_classical_stub.json"


def test_classical_sp500_bit_identical(tmp_path):
    # Re-run through the new --market path.
    out = subprocess.run(
        [sys.executable, "run_classical.py", "--market", "sp500", "--model", "stub"],
        cwd=MON, capture_output=True, text=True,
    )
    assert out.returncode == 0, out.stderr
    produced = json.loads((MON / "results" / "classical_summary.json").read_text())
    expected = json.loads(GOLDEN.read_text())
    assert produced == expected
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_golden_path_sp500.py -q`
Expected: FAIL — `run_classical.py: error: unrecognized arguments: --market`.

- [ ] **Step 3: Add the `--market` argument and resolve curves via the profile**

In `monitoring/run_classical.py`, inside `main()` where the parser is built (near line 155), add:

```python
    ap.add_argument("--market", default="sp500",
                    help="market profile key (default: sp500)")
```

Then replace the module-level `STRATEGY_CURVES` lookup used in the run loop with a profile lookup. At the top of `main()` after parsing args:

```python
    from config.profiles import get_profile
    profile = get_profile(args.market)
    strategy_curves = {
        s: [p for p in profile.pnl.STRATEGY_CURVES[s]] if hasattr(profile.pnl, "STRATEGY_CURVES")
        else None
        for s in profile.pnl.strategies()
    }
```

Simpler and exact: import the sp500 constant directly so paths are unchanged, and keep the existing loop body:

```python
    from providers.sp500.pnl import STRATEGY_CURVES  # identical paths as before
```

> Choose the direct-import form for Phase 1 (guarantees identical paths → identical output). The generic `profile.pnl` form lands in Phase 3 when a second market exists. Leave a `# TODO(phase3): resolve via profile.pnl` comment.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_golden_path_sp500.py -q`
Expected: PASS. `produced == expected` confirms bit-identical output.

- [ ] **Step 5: Commit**

```bash
git add monitoring/run_classical.py monitoring/tests/test_golden_path_sp500.py
git commit -m "feat(run_classical): --market flag (sp500 default), bit-identical"
```

---

## Task 5: Wire `--market` into run_agentic.py

**Files:**
- Modify: `monitoring/run_agentic.py` (argparse near line 286; curve/news/macro resolution)
- Test: extend `monitoring/tests/test_golden_path_sp500.py`

- [ ] **Step 1: Write the failing test (one-window stub run stable)**

```python
# append to monitoring/tests/test_golden_path_sp500.py
def test_agentic_sp500_accepts_market_flag():
    out = subprocess.run(
        [sys.executable, "run_agentic.py", "--market", "sp500",
         "--window", "quant_meltdown_2007", "--strategy", "AL_PCA", "--model", "stub"],
        cwd=MON, capture_output=True, text=True,
    )
    assert out.returncode == 0, out.stderr
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_golden_path_sp500.py::test_agentic_sp500_accepts_market_flag -q`
Expected: FAIL — `unrecognized arguments: --market`.

- [ ] **Step 3: Add `--market` and resolve sources via the sp500 profile**

In `run_agentic.py` near line 286, add:

```python
    ap.add_argument("--market", default="sp500", help="market profile key")
```

Where the news store and macro context are constructed today, resolve them from the profile (keeps sp500 identical, opens the seam):

```python
    from config.profiles import get_profile
    profile = get_profile(args.market)
    news = profile.news        # replaces the direct NewsStore(...) construction
    macro = profile.macro      # replaces the direct macro_context import call
    # STRATEGY_CURVES stays the direct sp500 import for bit-identical Phase-1 behaviour.
```

> Grep first: `grep -n "NewsStore\|macro_context\|STRATEGY_CURVES" monitoring/run_agentic.py` and swap only those construction sites. Do not change prompt/guardrail logic.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_golden_path_sp500.py -q`
Expected: PASS (all in file).

- [ ] **Step 5: Commit**

```bash
git add monitoring/run_agentic.py monitoring/tests/test_golden_path_sp500.py
git commit -m "feat(run_agentic): --market flag (sp500 default)"
```

---

## Task 6: Full suite green + design-doc status bump

**Files:**
- Modify: `docs/superpowers/specs/2026-08-19-multi-market-abstraction-vastai-design.md` (Phase 1 → done)

- [ ] **Step 1: Run both test suites**

Run: `cd monitoring && python -m pytest -q` then `cd ../XSectional && python -m pytest -q`
Expected: all pass; no regressions in the existing suites.

- [ ] **Step 2: Tick Phase 1 in the design doc rollout (§9) and note "US golden-path bit-identical: verified"**

- [ ] **Step 3: Commit**

```bash
git add docs/superpowers/specs/2026-08-19-multi-market-abstraction-vastai-design.md
git commit -m "docs: mark Phase 1 (provider abstraction) complete"
```

---

## Self-Review notes

- **Spec coverage:** implements §3 (provider interfaces + MarketProfile) and §9 Phase 1 (US refactor, bit-identical). Introduces no new market → correctly depends on **none** of §11.
- **Hard prerequisite:** the golden baseline (Task 0) needs the AL/JT curve data present locally. If absent, Phase 1 cannot prove bit-identical — flagged in Task 0 to surface to the user before proceeding.
- **Name-checks left as explicit grep steps** (windows export name; exact results JSON filename) rather than guessed — an executor must verify, not invent.
- **Bit-identical guarantee:** Tasks 4–5 keep the *direct sp500 path constants* rather than routing through the generic provider in Phase 1; the generic routing is deferred to Phase 3 when a second market first exercises it. This is deliberate and commented as `TODO(phase3)`.
