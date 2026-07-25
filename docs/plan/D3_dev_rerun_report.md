# Batch D Phase 3 — Dev-Set Rerun Report

**Date:** 2026-07-25
**Status:** MINOR CHANGE

---

## Verdict

MINOR CHANGE: 4 pair(s) changed but Holm conclusions unchanged (latency reject=True, FPR reject=False). Document in deviations log.

---

## Per-Pair Comparison (Agentic Detection & Latency)

| Window | Strategy | Pre-Det | Post-Det | Pre-Lat | Post-Lat | Changed? |
|---|---|---|---|---|---|---|
| downgrade_2011 | AL_PCA | Y | Y | 3 | 5 | **YES** |
| downgrade_2011 | JT_MOM | Y | Y | 5 | 3 | **YES** |
| gfc_lehman_2008 | AL_PCA | Y | Y | 0 | 0 | no |
| gfc_lehman_2008 | JT_MOM | Y | Y | 1 | 2 | **YES** |
| momentum_crash_2009 | AL_PCA | Y | Y | 0 | 0 | no |
| momentum_crash_2009 | JT_MOM | Y | Y | 1 | 1 | no |
| quant_meltdown_2007 | AL_PCA | Y | Y | 0 | 0 | no |
| quant_meltdown_2007 | JT_MOM | Y | Y | 4 | 0 | **YES** |

## Test Statistics Comparison

| Metric | Pre-D3 | Post-D3 |
|---|---|---|
| Perm-test obs_diff (latency, days) | -9.375 | -9.75 |
| Perm-test p-value (latency, 1-sided) | 0.00390625 | 0.00390625 |
| Holm: latency reject | True | True |
| Holm: FPR reject | False | False |

---

## Acceptance Criteria

- [x] All 12 dev cells (8 event + 4 calm) carry `run_meta` with real model + PIT curve
- [x] `dev_analysis.json` regenerated
- [ ] Conclusions vs archived pre-D3 version documented (this report)

## Next Step

Phase D4: run `python scripts/analyze_viability.py` (D3 JSONLs are the final dev evidence).
Then review D5 freeze preparation.
