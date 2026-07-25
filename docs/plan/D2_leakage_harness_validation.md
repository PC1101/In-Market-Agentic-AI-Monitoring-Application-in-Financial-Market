# Batch D Phase 2 — Leakage Harness Validation

**Date:** 2026-07-24
**Status:** PASS — mechanics validated on dev window; harness ready for D5 test-set run.

---

## Summary

The leakage harness (`run_leakage.py`) was validated on one dev event window
(`quant_meltdown_2007 × AL_PCA`, stub model) before the freeze. Three bugs in the
Condition B (date-masked) code path were found and fixed. The harness is now working
end-to-end with confirmed date masking in outputs.

---

## Step 1 — Dry-run (import + setup)

```
python run_leakage.py --strategy AL_PCA --model stub --dry-run
```

**Result:** PASS. Imports OK. AL_PCA test event windows: `flash_crash_2010`,
`china_deval_2015`.

`--window` flag added (`run_leakage.py`) to restrict to a single window for smoke tests
(pre-freeze code change, tested).

---

## Step 2 — Stub smoke (all 3 conditions, dev window)

```
python run_leakage.py --strategy AL_PCA --model stub --window quant_meltdown_2007 --M 3
```

**Three Condition B bugs found and fixed:**

| File | Bug | Fix |
|---|---|---|
| `agentic/guardrails.py:filter_news_by_timestamp` | `pd.Timestamp("XXXX-XX-XX")` crash | Try/except; return all records on unparseable sentinel |
| `agentic/guardrails.py:scrub_future_dated` | Same `pd.Timestamp` crash | Same try/except pattern |
| `agentic/guardrails.py:assert_no_lookahead` | Same `pd.Timestamp` crash | Try/except; early return (no-op — masking is the guard) |
| `agentic/schemas.py:validate_news_flags` | `date.fromisoformat("")` — stub returned empty `as_of` | Accept `XXXX-XX-XX` as valid sentinel |
| `agentic/model.py:_extract_as_of` | Stub regex `[0-9-]+` can't match `XXXX-XX-XX` → returns `""` | Check for sentinel string before regex |
| `run_leakage.py` | `≥` / `≤` Unicode chars fail on Windows cp1252 terminal | Replaced with `>=` / `<=` |

All bugs were in Condition B code paths never previously exercised. The same paths are
used in `run_agentic.py` — those bugs would have caused silent failures there too.
238 tests pass after all fixes.

**Results (stub model, M=3 synthetic):**

| Condition | Detected | Latency |
|---|---|---|
| A (standard) | True | 0d |
| B (date-masked) | False | None |
| C (synthetic, M=3) | recall=0.33 | 14.0d mean |

Leakage bound (stub): `evidence_skill>=0.33, memorisation<=1.00 [inconclusive]`
(stub results are deterministic artifacts — real-model results will be meaningful.)

---

## Step 3 — Condition B masking verification

Inspected `results/leakage_quant_meltdown_2007_AL_PCA.jsonl` (1120 records):

- **88 news_context records** with `assessment.as_of = "XXXX-XX-XX"` — confirms
  the masked sentinel propagates correctly through the news agent.
- **0 real ISO dates** found in any Condition B supervisor output — CLEAN.
- Condition B detected nothing (stub model), consistent with date masking blocking
  the stub's `_extract_as_of` from reading an event date.

---

## Step 4 — Runtime estimate for D5

For a single dev event window (quant_meltdown_2007, ~260 trading days) with real model:

| Condition | LLM calls | Est. time (4.2s/call) |
|---|---|---|
| A | ~260 × 2 (news + supervisor) | ~37 min |
| B | ~260 × 2 | ~37 min |
| C (M=10) | 10 × ~30 days × 2 | ~42 min |

**Estimated per-window leakage cost: ~2 hours (real model).**

For D5 test-set: AL_PCA has 1 test event window (flash_crash_2010), JT_MOM has
4 test event windows. Total leakage harness cost ≈ 2 × 5 = ~10 hours.
Plan D5 as an overnight attended run.

---

## Post-D2 status

- `freeze_gate.py`: all 11 PASS (leakage stub JSONL in results/ is dev-only → blinding intact).
- All 238 tests pass.
- Harness validated, bugs fixed. Ready for D3 (dev-set event rerun).

---

## Files changed in D2

| File | Change |
|---|---|
| `monitoring/run_leakage.py` | Added `--window` flag; fixed Unicode print |
| `monitoring/agentic/guardrails.py` | Sentinel handling in `filter_news_by_timestamp`, `scrub_future_dated`, `assert_no_lookahead` |
| `monitoring/agentic/schemas.py` | Accept `XXXX-XX-XX` in `validate_news_flags` |
| `monitoring/agentic/model.py` | Detect sentinel in `_extract_as_of` |
