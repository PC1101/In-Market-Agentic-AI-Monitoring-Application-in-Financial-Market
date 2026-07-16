# Week 3 Status — News pipeline + agent build

Against the VRI Week-3 checklist. Code lives in two new subpackages under
`monitoring/` (`news/`, `macro/`) plus extensions to `monitoring/agentic/` and the
new end-to-end driver `monitoring/run_agentic.py`.

## Deliverables

| VRI Week-3 item | Status | Where |
|---|---|---|
| FNSPID ingested + indexed by date (2003–2016) | ✅ | `monitoring/news/store.py` — 11,803,623 articles in `data/fnspid/store/year=*.parquet` |
| Macro integration (FRED/ALFRED, point-in-time vintages) | ✅ | `monitoring/macro/` — 3 vintage series (UNRATE 1,400 / CPIAUCSL 2,890 / INDPRO 33,666 vintage rows) + 5 daily series (VIXCLS, DGS10, DGS2, DFF, TEDRATE) in `data/macro/` |
| Filtering pipeline: risk filter → daily signals → triage | ✅ | `news/filter.py` → `news/aggregate.py` → `news/triage.py` |
| News Context Agent v1 (schema, prompt, guardrails, runner) | ✅ | `agentic/news_agent.py`, `agentic/schemas.py` (`NEWS_FLAGS_JSON_SCHEMA`), `agentic/prompts.py` (`news-context-v1`) |
| Performance Supervisor v3 (telemetry + classical + news + macro) | ✅ | `agentic/prompts.py` (`supervisor-v3`; v2 is the teammate's example-based telemetry-only prompt); `runner.run_supervisor` takes a prompt builder |
| Local model integration (real LLM, not stub) | ✅ | Ollama + qwen2.5:3b via `agentic/model.py` (`default_model`) |
| End-to-end agentic run on Aug-2007 with valid JSON | ✅ | `run_agentic.py` — see Run results below |

Tests: **103** in `monitoring/` + **53** in `XSectional/` — all passing.
The offline stub (`OfflineStubModel`) remains the CI model; no test needs Ollama.

## News coverage (the Week-3 data gate)

FNSPID density was measured per evaluation window before building on it
(`monitoring/news/coverage.py` → `monitoring/results/news_coverage.csv`):

| window | kind | n_articles | n_risk_articles | articles/day | adequate |
|---|---|---|---|---|---|
| quant_meltdown_2007 | event | 67,490 | 1,658 | 1,124.8 | ✅ |
| gfc_lehman_2008 | event | 281,797 | 8,662 | 2,683.8 | ✅ |
| momentum_crash_2009 | event | 283,769 | 5,264 | 2,782.1 | ✅ |
| downgrade_2011 | event | 410,909 | 8,785 | 4,669.4 | ✅ |
| calm_2004_2006 | calm | 91,065 | 1 | 83.4 | (calm windows do not gate) |
| calm_2013_2014 | calm | 2,514,284 | 47,319 | 3,453.7 | ✅ |

**Gate result: all four event windows are adequate** — including the headline
Aug-2007 window (1,658 risk-relevant articles), so no supervisor escalation was
needed. Caveat for the write-up: `calm_2004_2006` is thin (83 articles/day, and
only 1 risk-lexicon hit in three years), so news-side false-positive rates on that
calm window are measured against very sparse text; the 2013–14 calm window is the
denser calm control.

## Pipeline design (three stages before any LLM call)

1. **Risk filter** (`news/filter.py`) — a transparent, hand-built regex lexicon of
   market-stress language (23 patterns: sell-off, margin call, liquidation, quant
   fund stress, …). Recall-oriented by design; precision comes later.
2. **Daily signals** (`news/aggregate.py`) — per-day `n_articles`, `n_risk`,
   `intensity` (total risk-term hits) and `intensity_z`, a **causal** z-score
   against a trailing 60-day baseline shifted one day (day *t* never baselines on
   itself; warm-up days are NaN).
3. **Triage** (`news/triage.py`) — fixed a-priori thresholds decide daily spend:
   `skip` (no signal) / `cheap` (z ≥ 1 or one detector recent) / `thinking`
   (z ≥ 2.5) / `classical_escalation` (the ≥2-detectors-in-5-days aggregate fired —
   always assessed). Same no-in-sample-tuning discipline as the classical
   detectors: thresholds were not optimised against the evaluation windows.

## Model

qwen2.5:3b served by Ollama on CPU (16 GB Intel MBP), JSON mode, temperature 0.
`default_model("stub" | "ollama" | "ollama:<name>")` picks the backend;
`MONITOR_MODEL` env var overrides. One deliberate deviation from the VRI sketch:
triage modes route to one model with different escalation paths rather than two
differently-sized models — a CPU-latency tradeoff; `TriageDecision.mode` keeps the
design point so a second model can slot in later.

## Run results — Aug-2007 window × AL PCA (PIT curve)

Stub (offline sanity pass): 44 trading days; triage skip 23 / cheap 15 /
thinking 3 / classical_escalation 3; 21 schema-valid supervisor assessments.

qwen2.5:3b (real run): **in progress** — measured single-call latency ~100 s on
CPU (news agent, 40-article prompt, schema-valid output; e.g. 2007-08-09 →
ELEVATED, subprime stress, confidence 0.85). The full-window run (~42 LLM calls)
was interrupted twice by machine sleep/battery; final counts and mean latencies
will be added here when it completes. Triage is deterministic, so the day split
matches the stub run above.

Log: `monitoring/results/agentic_quant_meltdown_2007_AL_PCA.jsonl` — one triage
record per trading day plus one `news_context` and one `performance_supervisor`
invocation per escalated day, each with prompt version, raw output, validated
assessment, and latency.

## Point-in-time guarantees (Week-3 extensions)

- News publication timestamps are filtered to `<= as_of`
  (`guardrails.filter_news_by_timestamp`), and article *text* is scanned for
  mentions of dates after `as_of` — any hit drops the whole record, fail-closed
  (`guardrails.scrub_future_dated`). Both are re-applied inside `run_news_agent`
  regardless of what the caller did.
- Macro releases (UNRATE, CPIAUCSL, INDPRO) use **ALFRED vintages**: the value at
  `as_of` is the value *published* by `as_of`, not today's revision. This is also
  the point-in-time-correct route to BLS data (BLS's own API serves only revised
  figures — lookahead). Daily market series (VIXCLS, DGS10/DGS2, DFF, TEDRATE) are
  unrevised; last print `<= as_of`.
- `run_agentic.py` re-checks `assert_no_lookahead` on the assembled context every
  day; if the news agent's own narrative hallucinates a future date, the news block
  is dropped from the supervisor context (fail-closed) and flagged in the record.

## Open items for Week 4

1. Prompt iteration under version tags (`news-context-v2`, `supervisor-v4`, …)
   with the JSONL logs as the comparison substrate.
2. Two-pass with/without-events training control (the Week-4 2×2 cell).
3. All-6-windows × both-strategies agentic run + H1/H2/H3 metrics vs the classical
   baseline in `docs/WEEK2_STATUS.md`.
4. Carryover from Week 2: per-strategy detector calibration protocol (no in-sample
   tuning — decide with supervisor); confirm the 2011 downgrade window + onsets.

## Run it

```bash
cd monitoring && pip install -r requirements.txt
python -m pytest -q                     # 103 tests, offline (stub model)
python news/coverage.py                 # per-window FNSPID density gate
python run_agentic.py --window quant_meltdown_2007 --strategy AL_PCA --model stub
python run_agentic.py --window quant_meltdown_2007 --strategy AL_PCA --model ollama:qwen2.5:3b
```
