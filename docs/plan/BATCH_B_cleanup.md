# BATCH B — Codebase Cleanup (delete/archive from explicit lists ONLY)

> For the executing agent: this batch is MECHANICAL file operations from the exact
> lists below. You must NOT delete, move, or "tidy" ANYTHING not explicitly listed.
> When in doubt, skip the item and report it. No code edits. No test edits.
> Run BATCH A before this batch if possible; REQUIRED: do not run this batch while
> any Ollama/backtest run (BATCH C) is in progress (lock-file deletion would break it).

## Shared context (read once)

Repo root: `C:\Users\dungh\Downloads\In-Market-Agentic-AI-Monitoring-Application-in-Financial-Market`
Windows 10, bash shell, system Python 3.11. Git repo — use `git rm -r` / `git mv` for
tracked paths (history preserves everything), plain `rm -rf` only for untracked caches.
Check tracked status with `git ls-files <path> | head -1` before choosing rm vs git rm.

Why: research codebase heading to an eval freeze; stub-model artifacts are mixed with
real-model evidence, a 52 MB unused third-party repo sits in the tree, and caches/
notebooks clutter the structure. Exploration already confirmed: **no tests read
`monitoring/results/` contents; nothing imports arbitragelab; tests use tmp_path
fixtures only.**

## Step 0 — Safety preconditions
1. Verify no run is active: no `python.exe` process running `run_agentic.py` /
   `run_full_universe.py` (`tasklist | grep -i python` and ask the user if unsure), and
   `monitoring/results/.run_lock` either absent or stale (check its PID is not alive).
   If a run appears active → STOP and report; do not proceed.
2. `git status` — record the starting state in your report.

## Step 1 — DELETE (exact list)
| Path | Method |
|---|---|
| every `__pycache__/` dir under `monitoring/`, `XSectional/`, `Stat Arb/statsArb-dev/` | `rm -rf` (untracked) |
| every `.pytest_cache/` dir under the same roots | `rm -rf` |
| `Stat Arb/arbitragelab-master/` (entire dir, 52 MB) | `git rm -r` if tracked else `rm -rf` |
| `monitoring/results/.run_lock` and `monitoring/results/.lock_*` (only after Step 0 check) | `rm -f` |
| `monitoring/results/agent_log.jsonl` | `git rm` if tracked else `rm` |
| `docs/superpowers/` | `git rm -r` if tracked else `rm -rf` |

## Step 2 — ARCHIVE (move, never delete)
Create dirs as needed: `monitoring/results/archive/`, `monitoring/results/archive/stub/`,
`sp500-master/notebooks/`.

| From | To |
|---|---|
| `monitoring/results/agentic_calm_2004_2006_AL_PCA.jsonl` | `monitoring/results/archive/stub/` |
| `monitoring/results/agentic_calm_2004_2006_JT_MOM.jsonl` | `monitoring/results/archive/stub/` |
| `monitoring/results/agentic_calm_2013_2014_AL_PCA.jsonl` | `monitoring/results/archive/stub/` |
| `monitoring/results/agentic_calm_2013_2014_JT_MOM.jsonl` | `monitoring/results/archive/stub/` |
| `monitoring/results/model_benchmark.json` | `monitoring/results/archive/` |
| `monitoring/results/model_benchmark_log.jsonl` | `monitoring/results/archive/` |
| all 4 `sp500-master/*.ipynb` | `sp500-master/notebooks/` |

Reason the stub calm files are archived, not deleted: they are stub-model artifacts
(FPR from them is invalid) but are cited in earlier analyses; also BATCH C's resume
logic would wrongly SKIP the real-model calm reruns if these stayed in place. If
BATCH C already archived them, just verify and note it.

## Step 3 — KEEP (do NOT touch, even though they look like cruft)
- `monitoring/benchmark_model.py` — evidence for the documented llama3.2→qwen2.5 model deviation
- all 5 files in `monitoring/scripts/`
- all `sp500-master/*.csv` (PIT provenance chain, incl. the two large historical CSVs)
- `data/fnspid/store/` (857 MB news cache — expensive to rebuild)
- all 8 `monitoring/results/agentic_<event>_*.jsonl` event files (REAL-model evidence)
- `monitoring/results/{dev_analysis.json, classical_summary.json, calibration_grid.json, news_coverage.csv}`
- everything under `Stat Arb/statsArb-dev/` (incl. non-PIT result dirs — they are the
  snapshot-vs-PIT comparison evidence and the `_prefer_pit` fallback)
- `PREREGISTRATION_DRAFT.pdf`, all of `docs/`

## Step 4 — Verification (all must pass)
1. `cd monitoring && python -m pytest -q` → all green.
2. `cd ../XSectional && python -m pytest -q` → all green.
3. `cd ../monitoring && python run_classical.py` → runs end-to-end, prints PIT curve
   paths. (This regenerates `classical_summary.json` — that is acceptable here.)
4. `git status` — every change is explainable by Steps 1–2; include the full list in
   your final report.

## Forbidden actions
- No deletions/moves outside the exact lists above.
- No edits to any `.py`, `.md`, config, or data file.
- No `git commit` / `git push` — leave staged/working changes for user review.
- Never touch `data/`, `.git/`, or anything under `Stat Arb/statsArb-dev/`.
