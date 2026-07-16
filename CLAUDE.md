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
- Agentic run (one window): `python monitoring/run_agentic.py --window quant_meltdown_2007
  --strategy AL_PCA --model ollama:qwen2.5:3b` (use `--model stub` for offline).
- FNSPID store build: `python monitoring/news/download_fnspid.py` then
  `python -c "from news.store import build_store; ..."` (see docs/WEEK3_STATUS.md).
- Macro fetch: `FRED_API_KEY=<key> python monitoring/macro/fetch_macro.py`.
- Current status doc: `docs/WEEK3_STATUS.md`.

## Project deliverables & schedule
This project is governed by a weekly deliverables plan (the "VRI" document).
**Always read `docs/VRI.md` at the start of any work on this project** — it defines
the Week 1–6 to-dos, deliverables, and the 2×2 experimental design.

**Current status: Week 3 — News pipeline + agent build.** See `docs/WEEK3_STATUS.md`.

Week 3 open work (carried into Week 4):
- Prompt iteration under version tags; two-pass with/without-events training control;
  all-6-windows × both-strategies agentic run.

## Methodological spine (non-negotiable)
Avoid lookahead / hindsight bias everywhere: point-in-time universe, as-of dating,
timestamp filtering, and (Week 4) two-pass with/without-events training control.

## Hypotheses under test
- H1: agentic monitoring is faster AND lower false-positive rate than classical
- H2: the advantage generalises across both strategies (AL PCA, JT)
- H3: the advantage differs by strategy
