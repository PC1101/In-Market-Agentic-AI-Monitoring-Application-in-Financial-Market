# CLAUDE.md — In-Market Agentic AI Monitoring (Financial Markets)

## Dev environment
- Use the project venv: `source .venv/bin/activate` (created at repo root). The bare
  system `python3` (3.14) has NO packages installed; there is no `python` outside a venv.
- Tests: `python -m pytest -q` inside `monitoring/` and inside `XSectional/` (separate suites).
- End-to-end classical monitoring: `python monitoring/run_classical.py` (auto-prefers
  point-in-time `*_pit` AL curves when built).
- JT daily PnL (point-in-time universe): `python XSectional/run_daily_pnl.py`
  (`--survivorship` reproduces the biased curve for comparison).
- AL PCA backtests: `python "Stat Arb/statsArb-dev/run_full_universe.py" --pit --tag <name>`
  (PIT sleeves built by `scripts/build_pit_sleeves.py`). Runs take ~30-45 min.
- Current status doc: `docs/WEEK2_STATUS.md`.

## Project deliverables & schedule
This project is governed by a weekly deliverables plan (the "VRI" document).
**Always read `docs/VRI.md` at the start of any work on this project** — it defines
the Week 1–6 to-dos, deliverables, and the 2×2 experimental design.

**Current status: Week 2 — Classical monitoring + agentic structure.**

Week 2 open work:
- Implement + unit-test 4 detectors: Page-Hinkley, BOCPD, HMM, distributional threshold
- Aggregation rule: >=2 detectors firing within 5 days
- Run classical monitoring end-to-end: both strategies x all 6 windows (4 event, 2 calm)
- Metrics per-detector + aggregated: detection latency, false-positive rate, precision, recall
- Scaffold agentic framework: prompt templates, structured-JSON schema
  (state / action / root-cause / confidence), local-model integration, logging
- Information-parity guardrails: as-of dating + timestamp filtering (no lookahead)

## Methodological spine (non-negotiable)
Avoid lookahead / hindsight bias everywhere: point-in-time universe, as-of dating,
timestamp filtering, and (Week 4) two-pass with/without-events training control.

## Hypotheses under test
- H1: agentic monitoring is faster AND lower false-positive rate than classical
- H2: the advantage generalises across both strategies (AL PCA, JT)
- H3: the advantage differs by strategy
