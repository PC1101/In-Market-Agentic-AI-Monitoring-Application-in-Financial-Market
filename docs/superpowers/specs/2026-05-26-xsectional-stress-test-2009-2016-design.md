# XSectional — 2009–2016 Stress Test Layer

**Date:** 2026-05-26  
**Status:** Approved  
**Scope:** XSectional module only

---

## Overview

Add a second stress-test tearsheet to the XSectional momentum pipeline, covering the 2009–2016 sub-period. The full 2000–2025 backtest continues to run unchanged. After it completes, the resulting returns Series is sliced to the stress window and a second report is generated.

---

## Strategy

- **Computation method:** Slice the existing returns Series (no re-download, no re-run of signal/portfolio/backtest logic)
- **Output:** Separate PNG tearsheet saved alongside the existing one

---

## Files Changed

### `config.py`
Add two constants at the bottom:

```python
STRESS_START = "2009-01-01"
STRESS_END   = "2016-12-31"
```

### `report.py`
Add two optional parameters to `generate_report`:

| Parameter | Type | Default | Purpose |
|---|---|---|---|
| `label` | `str` | `"Full (2000–2025)"` | Appears in stdout header and panel titles |
| `filename` | `str` | `"tearsheet.png"` | Output PNG filename within `config.DATA_DIR` |

- Backward compatible: existing callers with no arguments continue to work
- The printed header changes from `"=== XSectional Momentum Tearsheet ==="` to `"=== XSectional Momentum Tearsheet — {label} ==="`
- Panel titles gain the label as a suffix where space permits

### `main.py`
After the existing `generate_report(returns)` call, add:

```python
logger.info("Stress test — 2009–2016 sub-period...")
stress_returns = returns.loc[config.STRESS_START : config.STRESS_END]
generate_report(
    stress_returns,
    label=f"Stress ({config.STRESS_START[:4]}–{config.STRESS_END[:4]})",
    filename="tearsheet_stress_2009_2016.png",
)
```

---

## Data Flow

```
load_prices()
  → compute_momentum_scores()
    → construct_portfolio()
      → run_backtest()                    # full 2000–2025 returns Series
          ├─ generate_report(returns)     # → data/tearsheet.png
          └─ returns.loc[2009:2016]
               → generate_report(stress) # → data/tearsheet_stress_2009_2016.png
```

---

## Output Files

```
XSectional/data/
  tearsheet.png                    ← unchanged, full 2000–2025
  tearsheet_stress_2009_2016.png   ← new, 2009–2016 sub-period
```

---

## Testing

- All existing tests in `tests/test_report.py` pass unchanged — `generate_report` defaults keep backward compatibility.
- No new test files needed for this change; the new parameters are thin wrappers around the existing tested logic.
- Manual verification: run `python3 main.py` and confirm both PNGs are produced.

---

## Known Limitations (inherited from main model)

- Survivorship bias and static universe limitations apply equally to the stress sub-period.
- The stress period uses momentum scores trained on pre-2009 data — this is intentional (slice method) and appropriate for an out-of-sample stress view.
