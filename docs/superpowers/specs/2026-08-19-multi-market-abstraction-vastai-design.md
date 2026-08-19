# Design: Multi-Market Abstraction (ASX 200 + Global Energy) & vast.ai Orchestration

**Date:** 2026-08-19
**Status:** Draft for review
**Author:** reconstructed from approved architecture + section walkthroughs (tasks #2, #3)
**Scope:** Generalise the in-market monitoring system beyond US S&P 500 to two additional
markets — **ASX 200** (Australian large-cap equities) and the **global energy market** —
and move heavy compute (backtests + agentic LLM runs) onto rented **vast.ai** GPU instances.

> **Reviewer note.** The specific approach-selections approved in the earlier design
> session were not recoverable from disk (no draft was persisted; only the task skeleton
> survived). This doc reconstructs the most natural choices consistent with the existing
> codebase and flags every point that needs your confirmation in
> [§11 Open decisions](#11-open-decisions-to-confirm). Please verify those before we move
> to `writing-plans`.

---

## 1. Context & goals

The system today ([`docs/ARCHITECTURE.md`](../../ARCHITECTURE.md)) compares a **classical
layer** (Page-Hinkley, BOCPD, HMM, distributional threshold) against an **agentic layer**
(news + macro-fed LLM agents) at detecting strategy regime breaks, under strict
point-in-time information parity. It is hard-wired to **one market**: the US S&P 500, with
two strategies (AL PCA stat-arb, JT momentum), FNSPID US news, and FRED/ALFRED US macro.

We want to run the *same* experimental design (H1: faster + lower FPR; H2: generalises
across strategies; H3: differs by strategy) on two more markets:

1. **ASX 200** — Australian large-cap equities. Tests whether the agentic advantage
   generalises to a different, thinner, differently-newsed equity market.
2. **Global energy market** — energy-sector securities/commodities worldwide (not
   Australia-specific). Tests generalisation to a sector-and-commodity-driven regime
   rather than a broad-index regime.

**Goals**

- G1. A **market-abstraction layer** so the pipeline runs against any of the three markets
  by selecting a profile — no forked copies of `monitoring/`.
- G2. **Pluggable data providers** (universe, strategy PnL, news, macro/commodity) so each
  market supplies its own sources behind stable interfaces.
- G3. **vast.ai orchestration** to run the expensive backtests (AL PCA ~30–45 min/run) and
  the agentic LLM sweeps on rented GPUs, reproducibly and cheaply.
- G4. Preserve the **methodological spine**: point-in-time universe, as-of dating,
  timestamp filtering, two-pass leakage controls — enforced identically for every market.

**Non-goals**

- Not changing the classical detectors, the hypotheses, or the metrics.
- Not re-running or altering the frozen US results (`eval-freeze-v1`).
- Not a live/production trading system — this stays a research/backtest harness.

---

## 2. Current architecture (the seams we build on)

`monitoring/` is already close to market-agnostic in shape, and three seam directories
exist but are **empty** — a strong signal this abstraction was anticipated:

| Path | Today | Role in this design |
|---|---|---|
| `monitoring/providers/` | *(empty)* | Provider interfaces + per-market implementations |
| `monitoring/data/` | *(empty)* | Cached, point-in-time datasets per market |
| `monitoring/config/` | *(empty)* | `MarketProfile` definitions (one per market) |
| `monitoring/news/` | FNSPID US store, filter, triage | Becomes the `NewsProvider` reference impl |
| `monitoring/macro/` | FRED/ALFRED as-of context | Becomes the `MacroProvider` reference impl |
| `monitoring/pnl_loader.py` | Loads AL/JT S&P 500 curves | Becomes a `StrategyPnLProvider` impl |
| `monitoring/windows.py` | Named event/calm windows w/ onsets | Windows become **per-market** |
| `monitoring/run_*.py` | Entry points, market implicit | Take `--market {sp500,asx200,energy}` |

Nothing about the detectors, aggregation rule, guardrails, or metrics is US-specific — the
market-specific knowledge lives entirely in **data sourcing** and **calendar/universe**.
That is the surface this design isolates.

---

## 3. Market abstraction — approaches & choice

### Approaches considered

- **A. Fork per market.** Copy `monitoring/` into `monitoring_asx/`, `monitoring_energy/`.
  *Simple day 1, diverges fast; four leakage controls and metrics drift out of sync across
  three copies. Rejected — kills G4.*
- **B. Config-only branching.** One codebase, `if market == …` branches at each data call.
  *No new files, but branch-rot spreads market logic through every module. Rejected.*
- **C. Provider interfaces + `MarketProfile` config (chosen).** Define a small set of
  `Protocol`/ABC interfaces in `monitoring/providers/`; each market implements them; a
  `MarketProfile` in `monitoring/config/` names the concrete providers, calendar,
  timezone, universe source, and window set. Entry points resolve the profile by
  `--market` and inject providers. *Isolates all market-specific code behind stable
  seams, keeps one shared pipeline, matches the empty seam dirs.*

### Chosen shape

```
monitoring/
  config/
    profiles.py         # MarketProfile dataclass + REGISTRY {sp500, asx200, energy}
  providers/
    base.py             # Protocols: UniverseProvider, StrategyPnLProvider,
                        #            NewsProvider, MacroProvider, MarketCalendar
    sp500/              # reference impls wrapping today's FNSPID/FRED/AL/JT code
    asx200/
    energy/
```

```python
# monitoring/config/profiles.py  (illustrative)
@dataclass(frozen=True)
class MarketProfile:
    key: str                       # "sp500" | "asx200" | "energy"
    timezone: str                  # "America/New_York" | "Australia/Sydney" | "UTC"
    calendar: MarketCalendar       # trading days / holidays
    universe: UniverseProvider     # point-in-time membership incl. delisted
    strategies: dict[str, StrategyPnLProvider]
    news: NewsProvider
    macro: MacroProvider
    windows: list[Window]          # per-market event/calm windows w/ onsets
    base_ccy: str                  # "USD" | "AUD" | "USD"
```

Every provider method is **as-of dated**: signatures take an `as_of: date` and must return
only information knowable by that date. The lookahead assertion in
`monitoring/agentic/guardrails.py` runs unchanged against provider output, so information
parity is enforced identically for all three markets (**G4**).

---

## 4. Data providers per market

| Provider | US S&P 500 (exists) | ASX 200 (new) | Global energy (new) |
|---|---|---|---|
| **Universe** | S&P 500 PIT membership incl. delisted | ASX 200 historical constituents incl. delisted (**confirm source** — see §11) | Global energy universe: energy equities across exchanges + optionally energy futures (**confirm scope** — §11) |
| **Strategy PnL** | AL PCA stat-arb, JT momentum | Same two strategy *templates* refit on ASX universe | Same templates on energy universe; sector/commodity sleeves |
| **News** | FNSPID (11.8M US articles 2003–2016) | AU-market news source (**confirm** — §11); reuse `news/filter,triage,finbert` | Global energy news (commodity + company); reuse filter/triage |
| **Macro / commodity** | FRED/ALFRED vintage | RBA + ABS AU macro, ASX calendar; AUD (**confirm** — §11) | Energy-specific: crude/natgas curves, EIA/IEA, inventories, USD |

**Reuse:** the `news/` filtering → quantitative-signal → triage stack and the `finbert`
scorer are market-neutral once fed provider output; only the *source ingestion* changes.
Each new market needs its own point-in-time news store built with the same
publication-timestamp filtering FNSPID uses.

**Point-in-time obligation (non-negotiable, G4):** every new provider must be able to
answer "what was knowable on date D" — survivorship-free universe, vintage macro, and
timestamp-filtered news — or the market cannot be admitted to the experiment. This is the
same gate the US universe passes (2007 meltdown invisible on survivor-only data).

---

## 5. ASX 200 specifics

- **Calendar/timezone:** `Australia/Sydney`; ASX trading calendar + AU public holidays.
  As-of boundaries computed in exchange-local time before UTC normalisation.
- **Currency:** AUD base; if any USD inputs are mixed in, convert at PIT FX.
- **Universe thinness:** ~200 names vs ~500; momentum/stat-arb sleeves are smaller, so
  detector calibration (`calibrate_classical.py`) reruns per market — thresholds are **not**
  transferable from the US.
- **Candidate event/calm windows:** GFC 2008, 2015 China-driven commodity selloff, COVID
  2020, plus AU-idiosyncratic episodes; **confirm the window list in §11**. Windows must be
  labelled with onsets exactly as `windows.py` does today.

## 6. Global energy specifics

- **Timezone:** no single exchange — normalise to `UTC`; per-instrument local calendars.
- **Regime drivers are commodity-led:** oil/gas price shocks, OPEC decisions, supply
  disruptions — so the `MacroProvider` here carries **commodity curves and inventory
  prints**, not just rates. This is the biggest new-data lift.
- **Universe question:** equities-only (global energy sector stocks) vs equities + energy
  futures. Recommend **equities-first** to reuse the existing PnL machinery, add futures as
  a later extension. **Confirm in §11.**
- **Windows:** 2014–2016 oil crash, 2020 negative WTI, 2022 energy spike; **confirm §11.**

---

## 7. vast.ai orchestration — approaches & choice

**Why:** AL PCA backtests run ~30–45 min each; three markets × strategies × windows ×
two-pass leakage conditions × model sweeps is far past a laptop. The agentic layer runs
`ollama:qwen2.5:3b` and benefits from a GPU.

### Approaches considered

- **A. Manual SSH + rsync.** Rent an instance in the web UI, copy code, run by hand.
  *Zero build, but not reproducible, easy to leave a GPU billing overnight. Rejected as the
  primary path (fine as a fallback).*
- **B. Ephemeral, Dockerised, CLI-driven (chosen).** A pinned Docker image (Python venv +
  ollama + models baked or pulled once) pushed to a registry; `scripts/vast/launch.py`
  uses the `vastai` CLI to provision → run one job spec → pull `results/` → **destroy the
  instance**. Pay-per-run, reproducible, teardown guaranteed.
- **C. Persistent instance + job queue.** Long-lived box pulling from a queue. *Best
  throughput, worst idle cost and most ops. Deferred until run volume justifies it.*

### Chosen shape

```
scripts/vast/
  Dockerfile          # venv + requirements + ollama + qwen2.5:3b pull
  launch.py           # vastai search offers -> create -> run job -> rsync results -> destroy
  job.yaml            # market, strategy, windows, model, conditions, seed
  teardown.py         # safety: destroy any instance tagged by this project (cost guard)
```

- **Reproducibility:** image digest + `job.yaml` + git SHA are stamped into each result
  dir, so any run is re-creatable.
- **Cost guard:** every instance is tagged; `teardown.py` (and a launch-time max-price /
  max-duration cap) prevents runaway billing. **Confirm cost ceiling in §11.**
- **Data:** point-in-time stores are large — pull once to a vast.ai volume / object store
  rather than per-run. Secrets (`FRED_API_KEY`, news API keys) injected via env, never
  baked into the image.
- **Determinism:** LLM runs pin model digest + seed; `--model stub` stays the offline CI
  path so nothing in CI depends on renting a GPU.

---

## 8. Testing strategy

- **Provider contract tests:** one shared `pytest` suite parametrised over all three
  markets asserting each provider honours the `as_of` contract (no future data leaks). This
  is the new safety net that keeps three markets from diverging.
- **Lookahead gate per market:** the existing lookahead assertion + a market-specific
  "headline event invisible on survivor-only universe" gating test (the US analogue of the
  2007-meltdown check) must pass before a market is admitted.
- **Golden-path regression:** US S&P 500 results must be **bit-identical** after the
  refactor — the abstraction is behaviour-preserving for the existing market (protects
  `eval-freeze-v1`).
- **vast.ai smoke test:** `launch.py` against `--model stub` and a 1-window job, asserting
  provision → run → results-pulled → instance-destroyed, with a hard cost cap.
- CI stays laptop-only (`stub` model, cached mini-fixtures); GPU jobs are opt-in.

---

## 9. Rollout phases

1. **Refactor US to providers** ✅ *(done 2026-08-19)* — extracted the FNSPID/FRED/AL/JT
   wiring behind the provider interfaces; `--market` flag on both entry points; **US
   classical output verified byte-identical** to the pre-refactor golden baseline; full
   monitoring suite green. See `docs/superpowers/plans/2026-08-19-phase1-provider-abstraction.md`.
2. **vast.ai harness** 🟡 *(built 2026-08-19; live validation pending)* — `scripts/vast/`
   with `vastlib.py` (cost-guard core, 9 tests), `launch.py` (search→budget-gate→provision→
   run→pull→always-destroy), `teardown.py` (stray sweep), `job.yaml`, `Dockerfile`, `README`.
   $0.50/hr cap enforced 4 ways; `--dry-run` verified end-to-end; nothing spends without
   `--yes`. **Remaining:** live smoke test on a real GPU (needs vast.ai API key + ssh key on
   the host) and the one-US-window parity re-run.
3. **ASX 200 market** — providers, PIT stores, windows, recalibrate detectors, full run.
4. **Global energy market** — providers incl. commodity macro, windows, full run.
5. **Cross-market analysis** — extend `significance.py`/reporting to H1/H2/H3 across markets.

Each phase is independently reviewable and leaves the repo green.

---

## 10. Risks

| Risk | Mitigation |
|---|---|
| New markets lack survivorship-free / vintage data | Admission gate (§4, §8); a market that can't prove PIT is not run |
| Refactor perturbs frozen US results | Golden-path bit-identical regression test (§8) |
| vast.ai cost runaway | Ephemeral + tag + teardown + price/duration caps (§7) |
| Energy commodity macro is a large new ingest | Equities-first scope; futures deferred (§6) |
| Detector thresholds mis-transferred across markets | Recalibrate per market; thresholds never shared (§5) |

---

## 11. Open decisions — RESOLVED 2026-08-19

*(Resolved with the user after data-source research. These lock the Phase 3/4 providers.)*

1. **ASX 200 constituent source** ✅ **Free-first, Norgate fallback.** Reconstruct PIT
   membership from free S&P/ASX quarterly rebalance announcements (aggregated at
   marketindex.com.au) + free delisted-price archives (Kaggle "Arandkei", GitHub), then run
   a **survivorship gating test** — a known ASX drawdown must be *invisible* on a
   survivor-only universe (the ASX analogue of the US 2007-meltdown check). **If it fails the
   gate, buy Norgate Data Platinum (~USD 53/mo: PIT constituents incl. delisted + delisted
   prices + Python API).** The gating test is the go/no-go for admitting free ASX data.
2. **Global energy universe** ✅ **Equities-only — curated stable megacap universe** (refined
   2026-08-19 after spike). The *full* S&P Global 1200 Energy with survivorship-free PIT
   membership hits the same paywall as ASX, but energy is dominated by a small, very stable
   megacap set that rarely delists — so a **curated fixed universe of ~30-40 major global
   energy names** (US-listed + ADRs) is free (yfinance, verified: full 2007-2020 daily
   coverage, no gaps, no key) and methodologically defensible with a documented survivorship
   caveat (the few exits — BG→Shell, Anadarko→Oxy, Pioneer/Marathon→2024 M&A — are nameable).
   Regime signal is commodity-driven (free FRED curves), so the universe need only be
   representative, not index-exact. Futures deferred.
3. **News providers** ✅ **GDELT** for *both* new markets — free, global, point-in-time
   timestamps, themes/sentiment, multi-year history. Consumed by the existing
   filter→triage→FinBERT stack. Needs an ingestion adapter that writes the same per-year
   parquet schema as the FNSPID store (`date, ticker, title, summary, publisher, url`).
4. **Macro / commodity** ✅ **Reuse + extend FRED/ALFRED.** Energy commodity *prices* already
   available via the existing FRED/ALFRED integration — add series IDs `DCOILWTICO` (WTI),
   `DCOILBRENTEU` (Brent), `DHHNGSP` (Henry Hub natgas) to the macro fetch list. Add **EIA
   API** (free, key required) for inventories/supply. AU macro via **RBA + ABS.Stat** (free,
   community Python access). *(Default accepted; revisit if energy needs deeper commodity data.)*
5. **Window lists + onsets** 🟡 **Tentative — confirm during Phase 3/4 build.** ASX: GFC 2008,
   2015 commodity/China selloff, COVID 2020 (+ 2 calm). Energy: 2014–16 oil crash, 2020
   negative WTI, 2022 energy spike (+ 2 calm). Each needs a ground-truth onset date like
   `windows.py`.
6. **Strategies** ✅ **Reuse the AL PCA + JT templates** refit per market — keeps the H2/H3
   cross-market comparison clean and reuses existing code.
7. **vast.ai budget** ✅ **$0.50/hr** hourly cap (enforced in `scripts/vast/`). Optional
   per-run ceiling not set — hourly cap governs.
8. **Doc location/name** ✅ `docs/superpowers/specs/` confirmed as home.
