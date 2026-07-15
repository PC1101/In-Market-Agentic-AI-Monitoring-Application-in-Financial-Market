# Week 3 Status — News pipeline + News Context Agent + Supervisor with news

Against the VRI Week-3 checklist. Code lives in the new `monitoring/news/` subpackage,
extensions to `monitoring/agentic/`, and the `--news` path of `monitoring/run_classical.py`.

## Deliverables

| VRI Week-3 item | Status | Where |
|---|---|---|
| FNSPID ingested and indexed by date | ✅ (+ ingest bug fixed & re-ingested) | `Stat Arb/statsArb-dev/scripts/reingest_fnspid.py`, `monitoring/scripts/build_news_cache.py` |
| Macro data (ALFRED/FRED/BLS/Treasury) | ⏸ **Descoped for Week 3** (agreed) | — |
| News filtering pipeline (keyword filter → signal aggregator → triage) | ✅ | `monitoring/news/` (`filters.py`, `aggregate.py`, `triage.py`, `pipeline.py`) |
| News Context Agent v1 (structured risk flags + narrative) | ✅ | `agentic/schemas.py`, `prompts.py`, `runner.py` (`run_news_agent`) |
| Performance Supervisor v1 extended with news summary | ✅ (`supervisor-v3`) | `agentic/guardrails.py` (`as_of_context(..., news=)`), `prompts.py` |
| End-to-end agentic run on Aug-2007 window with valid JSON | ✅ | `python run_classical.py --news --model ollama:llama3.2:3b --windows quant_meltdown_2007` |

Tests: **135** in `monitoring/` (62 existing + 73 new news-path tests) — all passing,
all offline (JSONL fixture + `FakeScorer` + `OfflineStubModel`; no network, no ML deps).

## Data findings (must-know)

### FNSPID: pre-2009 is Russian, and the loader read the wrong column
- Pre-2009 FNSPID rows are a **Lenta.ru Russian general-news scrape with empty ticker
  tags** — the original `fnspid_xlf.parquet` had ~23 usable English headlines in the
  whole Aug-2007 window.
- Root cause of the *additional* sparsity: `news_loader.py`'s `_TEXT_COLS` matched the
  `Article` **body** column (mostly empty) and never `Article_title` (the actual
  headline). Fixed (`Article_title` now first), re-ingested from the still-cached
  HF CSVs (no re-download): **15,549,299 rows** in `fnspid_raw.parquet`.
- Post-fix usable **English** headlines per window (after Cyrillic/short-title drop and
  per-(day, headline) dedup in `build_news_cache.py`):

| window | usable headlines |
|---|---|
| quant_meltdown_2007 | 82,729 |
| gfc_lehman_2008 | 322,535 |
| momentum_crash_2009 | 317,340 |
| downgrade_2011 | 471,116 |
| calm_2004_2006 | **93** |
| calm_2013_2014 | 2,088,962 |

- Aug 1–15 2007 alone recovered 16,623 English headlines (was ~23), including
  stress-relevant wire copy ("Dow bows to credit pressures").

### NYT Archive supplement (fetched ✅)
`calm_2004_2006` is effectively uncovered by FNSPID (93 headlines / 3 years), and a
uniform market-wide source across all windows avoids a **source-mix confound**
(FNSPID's density varies 4 orders of magnitude across windows). The NYT Archive API
(exact `pub_date` timestamps → point-in-time safe) has been fetched and cached:
**25 months, 240,118 articles** (all 4 event windows + lead-in months, plus 3-month
calm samples 2005-03..05 and 2013-03..05) → `nyt_archive.parquet` via
`monitoring/scripts/fetch_nyt_archive.py` + `build_news_cache.py`. The pipeline
consumes whichever caches exist; source mix is a config knob (`NewsConfig.sources`),
default NYT + FNSPID. Note: NYT calm coverage is the 3-month samples, not the full
calm spans — extend with more archive months if Week-5 analysis needs it.

## Pipeline design (causal by construction)

- **T-1 calendar-day cutoff** — a decision on the morning of day *T* sees only records
  published through end-of-day *T-1* (`filter_news_by_timestamp` at `T 00:00 − 1s`).
  Matches the statsArb `RegimeFilter` convention; conservative for NYT exact
  timestamps; correct for FNSPID's midnight day-granularity stamps; calendar days so
  Monday decisions see weekend news.
- **Fail-closed guards** — undated/unparseable records are dropped; any record whose
  *text* contains an ISO date after the cutoff is dropped and counted (never silently
  forwarded); the assembled block is finally re-checked with `assert_no_lookahead`.
- **Keyword filter** (`filters.py`) — case-insensitive regex lexicon in 4 categories
  (liquidity_credit, market_stress, fund_quant_stress, macro_policy) that map to the
  agent's risk-flag enum. **Hindsight-bias control:** generic distress terms only —
  no institutions, funds, or people (a 2026-written list naming the 2007–2011 actors
  would leak hindsight); tone cross-checked against Loughran-McDonald, not against
  the six evaluation windows.
- **Signal aggregator** (`aggregate.py`) — per-day `n_articles`, `n_hits`, `hit_rate`,
  `stress` (FinBERT net-negative fraction over *matched* headlines), and `hit_z`
  (z-score of `n_hits` vs the **previous** 30 days — day *t* excluded from its own
  baseline).
- **Sentiment** (`sentiment.py`) — `FinBERTScorer` (ProsusAI/finbert), **CPU by
  default** to avoid VRAM contention with llama3.2:3b on the 4 GB GPU; `FakeScorer`
  for tests. LOOK-AHEAD WARNING: FinBERT's training corpus may include post-crisis
  text; mitigated project-level by the Week-4 with/without-events training control.

## Triage (cheap-model / thinking-model / classical-escalation)

Thresholds are module constants fixed **a priori** — not tuned on the six evaluation
windows (same no-in-sample-tuning protocol as detector calibration, WEEK2 open item 1).

| mode | rule | action |
|---|---|---|
| `SKIP` | no hits in 3d, stress < 0.30, no detector alarms | no LLM call; supervisor gets `{"status": "quiet"}` |
| `CHEAP_MODEL` | any 3d hits OR stress ≥ 0.30 OR ≥1 recent alarm | News Context Agent, standard prompt (`news-context-v1`) |
| `THINKING_MODEL` | (hit_z ≥ 2 OR stress ≥ 0.60) AND aggregate alarm | extended step-by-step prompt (`news-context-v1-thinking`); a genuinely larger thinking model is documented future work — the 4 GB VRAM cap binds |
| `CLASSICAL_ESCALATION` | zero news coverage 7d while detectors alarm, OR the LLM fails/invalid after one repair retry | supervisor runs classical-only; escalation logged in `agent_log.jsonl` |

## Agents

- **News Context Agent v1** — output contract `NEWS_CONTEXT_JSON_SCHEMA` (flat,
  enum-heavy for the 3B model): `as_of`, `risk_flags` (LIQUIDITY_STRESS, CREDIT_EVENT,
  MARKET_SELLOFF, FUND_STRESS, POLICY_INTERVENTION, RATING_ACTION,
  MACRO_DETERIORATION, NONE — NONE is exclusive), `narrative` (≤600 chars),
  `news_intensity` (LOW/ELEVATED/HIGH), `confidence`, `headlines_cited`. Prompt shows
  a **concrete example object, never a schema dump** (the supervisor-v2 lesson);
  Ollama decoding is grammar-constrained to the schema via `format`.
- **Supervisor v3** — `as_of_context(..., news=)` attaches the validated news summary
  (or `{"status": "quiet"|"unavailable"}` so the model knows *why* news is absent);
  prompt appends a `News context summary:` block and allows citing `news_context`.
  **The assessment output schema is unchanged** — Week-2 consumers unaffected.

## End-to-end run — Aug-2007, llama3.2:3b + FinBERT

`python run_classical.py --news --model ollama:llama3.2:3b --windows quant_meltdown_2007`
completed with **4/4 schema-valid agent outputs on the first try** (no repair retries,
no escalations) at the onset date 2007-08-06, logged in
`monitoring/results/agent_log.jsonl`:

| agent | prompt | strategy | output |
|---|---|---|---|
| news_context | news-context-v1 | AL_PCA | flags **LIQUIDITY_STRESS, CREDIT_EVENT**, intensity **ELEVATED**, conf 0.9, cites "Market rout shifts timing for next Fed move" |
| performance_supervisor | supervisor-v3 | AL_PCA | **ALERT / INVESTIGATE**, conf 0.8, cites page_hinkley + hmm |
| news_context | news-context-v1 | JT_MOM | flags LIQUIDITY_STRESS, CREDIT_EVENT, intensity ELEVATED, conf 0.9 |
| performance_supervisor | supervisor-v3 | JT_MOM | WATCH / INVESTIGATE, conf 0.8, cites page_hinkley + bocpd |

The T-1 news the agent saw is the genuine pre-meltdown tape — subprime-lender layoffs,
the Fed weighing the credit crunch, a Senate report on the SEC's hedge-fund probe —
and it correctly reads it as liquidity/credit stress. (Known 3B quirk, kept for the
write-up: one narrative drags in the Minneapolis bridge collapse from the same tape —
salience filtering, not causality, is the model's weak spot.)

**Daily triage tallies** (FinBERT signal, `results/news_triage_report.json`) —
Aug-2007 window, 44 trading days:

| strategy | SKIP | CHEAP_MODEL | THINKING_MODEL | CLASSICAL_ESCALATION |
|---|---|---|---|---|
| AL_PCA | 0 | 39 | **5** | 0 |
| JT_MOM | 0 | 44 | 0 | 0 |

Reading: the news channel is *never* quiet in this window (FNSPID wire coverage is
dense), THINKING escalations occur only for AL_PCA — exactly where the news spike
coincides with the classical aggregate alarm — and JT (whose detectors saw no aggregate
alarm here) stays at CHEAP. Calm-window quiet-day (SKIP) evidence needs the NYT fetch
(open item 1). Runtime note: FinBERT stress scoring runs on CPU (memoised per day);
the Aug-2007 window takes ~10 min end-to-end on this laptop.

## Interpretation caveat (state up front)
The Week-3 deliverable is a **valid end-to-end run**, not detection lift: the agent is
invoked at the event **onset date only** (2007-08-06), and with the T-1 cutoff it sees
*pre*-meltdown news — coverage of the quant unwind thickened Aug 9–10, 2007. Daily
agent cadence (where triage and the news channel can actually help latency/FPR) is the
Week-4 work.

## Open items / next
1. ~~NYT Archive fetch~~ ✅ done — 25 months / 240,118 articles cached
   (`nyt_archive.parquet`). Calm-window SKIP evidence now obtainable on the
   2005-03..05 / 2013-03..05 NYT samples.
2. **Daily agent cadence + evaluation** (Week 4): run triage → agents every trading
   day, score agentic latency/FPR against classical (H1), two-pass with/without-events
   control for FinBERT/LLM training leakage.
3. **Thinking model** — currently the same 3B model with an extended-reasoning prompt
   variant; a larger model is future work under the 4 GB VRAM cap.
4. **Source-mix confound** — decide (with supervisor) NYT-only vs NYT+FNSPID for the
   dense 2011/2013 windows; both cached, mix is a config flag.
5. Macro data (ALFRED/FRED) remains descoped; revisit if Week-4/5 analysis needs it.

## Merge note — `week3-news-agents` branch integrated (2026-07-15)

A teammate's parallel Week-3 implementation (`origin/week3-news-agents`, documented in
`docs/ARCHITECTURE.md`/`.pdf`) was merged into this branch. Resolution policy:

- **Taken from their branch**: `monitoring/macro/` (FRED/ALFRED point-in-time vintages +
  as-of guards + tests) — cached and tested but **not yet wired into the supervisor
  prompt** (Week-4 decision; keeps tonight's E2E results valid); `news/store.py`
  (per-year FNSPID parquet store, 11.8M articles), `news/filter.py`, `news/coverage.py`,
  `news/download_fnspid.py` + their standalone tests; `guardrails.scrub_future_dated`;
  `run_supervisor(prompt_builder=, prompt_version=)` extension points;
  `docs/ARCHITECTURE.*`.
- **Ported**: their `run_agentic.py` daily loop was re-based onto this branch's stack —
  it now reuses `run_classical`'s `_news_for_date`/`_frame_slice` plumbing (NYT+FNSPID
  caches, FinBERT signal, `TriageMode`, supervisor-v3, llama3.2:3b) so both entry points
  share one triage/guardrail implementation. `tests/test_run_agentic.py` rewritten
  against the ported loop (offline fixture + FakeScorer + stub).
- **Superseded/dropped**: their `agentic/news_agent.py`, `test_news_schema.py`,
  `test_supervisor_v2.py` (this branch's news agent, schema, and supervisor-v3 subsume
  them); colliding `agentic/` + `news/` files resolved to this branch's versions.
- **Finding kept for the write-up**: their original coverage gate ran on the raw FNSPID
  store and passed `calm_2004_2006` on volume (91,065 articles) yet found
  `n_risk_articles=1` — pre-2009 FNSPID is the Russian Lenta.ru scrape, so a store-level
  volume gate is blind to unusable content. `news/coverage.py` has been repointed at the
  pipeline's actual caches (NYT Archive + Cyrillic-cleaned FNSPID, i.e. what the agents
  see); the regenerated `news_coverage.csv` shows `calm_2004_2006` = 31,051 articles
  (30,958 NYT / 93 FNSPID) with **854** risk hits, and every event window passes the
  gate with a per-source breakdown.

## Run it
```bash
cd monitoring
python -m pytest -q                          # 135 tests, offline
python run_classical.py                      # baseline unchanged (no news path)
python scripts/build_news_cache.py           # FNSPID cache (NYT too once fetched)
python run_classical.py --news --news-scorer fake --windows quant_meltdown_2007   # offline news path
python run_classical.py --news --model ollama:llama3.2:3b --windows quant_meltdown_2007   # the Week-3 E2E deliverable
```
