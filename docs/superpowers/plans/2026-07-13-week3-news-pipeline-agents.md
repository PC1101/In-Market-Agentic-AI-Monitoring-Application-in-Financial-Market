# Week 3: News Pipeline + Agent Build — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ingest FNSPID news + vintage-correct macro data, build the filter→aggregate→triage news pipeline, ship News Context Agent v1 and Performance Supervisor Agent v2 on a real local model (Ollama qwen2.5:3b), and complete an end-to-end agentic run on the Aug-2007 event window with valid JSON output.

**Architecture:** Two new subpackages under `monitoring/` — `news/` (FNSPID store, risk filter, signal aggregator, triage) and `macro/` (FRED/ALFRED fetch + as-of queries) — feeding the existing `agentic/` package, which gains a News Context Agent (new schema + prompt + runner) and a supervisor-v2 prompt that consumes news + macro blocks. A new `monitoring/run_agentic.py` drives a daily loop over one event window: triage decides skip / cheap-model / thinking-model / classical-escalation each day; escalated days call the news agent then the supervisor. All context passes through the existing `guardrails.as_of_context` / `assert_no_lookahead` spine; news additionally passes `filter_news_by_timestamp` + a new fail-closed `scrub_future_dated`.

**Tech Stack:** Python (project venv at repo root), pandas, pyarrow (parquet store), huggingface_hub (FNSPID download), Ollama + qwen2.5:3b (local LLM, CPU), FRED/ALFRED REST API (stdlib urllib), pytest.

**Conventions (match existing code):**
- All commands run from `monitoring/` with the repo venv active: `source ../.venv/bin/activate` (or `source .venv/bin/activate` from repo root).
- `monitoring/` modules import flat (`from windows import ...`); subpackages import as `from news.store import ...`.
- Tests live in `monitoring/tests/`, run with `python -m pytest -q` from `monitoring/`.
- Work on a new branch: `git checkout -b week3-news-agents` (from `week2-monitoring-pit`).

**Point-in-time non-negotiables (from CLAUDE.md):**
- Every piece of news context is timestamp-filtered to `<= as_of` AND scrubbed of articles whose *text* mentions dates after `as_of` (fail-closed).
- Macro releases (UNRATE, CPI, INDPRO) come from **ALFRED vintages** — the value used at `as_of` is the value *published* by `as_of`, not today's revised figure. This is also how "BLS" data is integrated point-in-time-correctly (BLS's own API only serves revised data → lookahead). Daily market series (VIX, Treasury DGS yields, fed funds, TED spread) are unrevised; use last value dated `<= as_of`.
- Known risk, gated early (Task 6): FNSPID coverage is thin pre-2009 — exactly where the Aug-2007 gating event lives. Task 6 measures per-window article counts and stops for a supervisor decision if the 2007 window is inadequate.

---

## File Structure

```
monitoring/
  news/
    __init__.py          # package exports
    download_fnspid.py   # CLI: list HF repo files, download raw news CSV, peek schema
    store.py             # build_store (chunked CSV → parquet by year), NewsStore.query
    coverage.py          # per-window article counts → results/news_coverage.csv (GATE)
    filter.py            # risk lexicon regex filter
    aggregate.py         # daily quantitative news signals (counts, intensity, causal z)
    triage.py            # skip / cheap / thinking / classical_escalation decision
  macro/
    __init__.py          # package exports
    fetch_macro.py       # CLI: FRED + ALFRED-vintage fetch → data/macro/*.parquet
    asof.py              # asof_vintage / asof_daily point-in-time queries
    context.py           # macro_context(as_of) → JSON-serialisable dict block
  agentic/
    model.py             # MODIFY: default_model() factory; stub handles news prompts;
                         #         OllamaModel default → qwen2.5:3b
    schemas.py           # MODIFY: add OverallRisk, NEWS_FLAGS_JSON_SCHEMA,
                         #         NewsFlags, validate_news_flags
    prompts.py           # MODIFY: add news-context-v1 prompt + supervisor-v2 prompt
    news_agent.py        # CREATE: run_news_agent (mirror of runner.run_supervisor)
    runner.py            # MODIFY: run_supervisor gains prompt_builder/prompt_version params
    guardrails.py        # MODIFY: add scrub_future_dated
    __init__.py          # MODIFY: export new symbols
  run_agentic.py         # CREATE: end-to-end daily loop on one window (Aug 2007)
  requirements.txt       # MODIFY: + pyarrow, huggingface_hub
  tests/
    test_news_store.py
    test_news_coverage.py
    test_news_filter.py
    test_news_aggregate.py
    test_news_triage.py
    test_macro_asof.py
    test_macro_context.py
    test_news_schema.py
    test_news_agent.py
    test_supervisor_v2.py
    test_run_agentic.py
data/                    # gitignored payloads
  fnspid/raw/            # raw FNSPID CSV (~22 GB; delete after store build if space needed)
  fnspid/store/          # year=YYYY.parquet files (the queryable index)
  macro/                 # one parquet per FRED/ALFRED series
docs/
  WEEK3_STATUS.md        # CREATE (final task)
```

---

### Task 0: Branch + dependencies

**Files:**
- Modify: `monitoring/requirements.txt`
- Modify: `.gitignore` (repo root)

- [ ] **Step 1: Create the branch**

```bash
cd /Users/paulchenbackup/Desktop/In-Market-Agentic-AI-Monitoring-Application-in-Financial-Market
git checkout -b week3-news-agents
```

- [ ] **Step 2: Add dependencies**

Append to `monitoring/requirements.txt` (keep existing lines):

```
pyarrow>=15
huggingface_hub>=0.23
```

Install: `source .venv/bin/activate && pip install -r monitoring/requirements.txt`
Expected: both packages install (or already satisfied).

- [ ] **Step 3: Gitignore the data payloads**

Append to `.gitignore` if not already covered:

```
data/fnspid/
data/macro/
```

- [ ] **Step 4: Commit**

```bash
git add monitoring/requirements.txt .gitignore
git commit -m "chore(week3): branch, deps (pyarrow, huggingface_hub), gitignore data payloads"
```

---

### Task 1: Ollama + qwen2.5:3b installed and benchmarked

Environment setup — no TDD; verified by a smoke script. This machine is a 16 GB Intel MBP (CPU inference): qwen2.5:3b is the chosen size/latency tradeoff.

**Files:** none (system install); benchmark output recorded in `docs/WEEK3_STATUS.md` later.

- [ ] **Step 1: Install Ollama**

```bash
brew install ollama
```

Expected: `ollama --version` prints a version. (If brew is unavailable, download the macOS app from https://ollama.com/download and skip to Step 2.)

- [ ] **Step 2: Start the server and pull the model**

```bash
brew services start ollama   # or: ollama serve &  (leave running)
ollama pull qwen2.5:3b       # ~1.9 GB download
```

Expected: `ollama list` shows `qwen2.5:3b`.

- [ ] **Step 3: Smoke-test JSON mode through the existing client**

```bash
cd monitoring && python -c "
import time
from agentic.model import OllamaModel
m = OllamaModel(model='qwen2.5:3b')
t0 = time.time()
out = m.complete(
    'Return ONLY a JSON object: {\"state\": one of [\"NORMAL\",\"WATCH\"], \"confidence\": number 0..1}',
    'The strategy lost 4% today. Return your JSON now.')
print(out, f'({time.time()-t0:.1f}s)')
"
```

Expected: a parsed dict printed with a latency figure. Record the latency (typical: 10–60 s/call on this CPU). If >120 s, raise `timeout` in `OllamaModel` when constructing it in later tasks.

- [ ] **Step 4: No commit** (nothing in-repo changed).

---

### Task 2: `default_model()` factory + stub news-mode + qwen default

**Files:**
- Modify: `monitoring/agentic/model.py`
- Modify: `monitoring/agentic/__init__.py`
- Test: `monitoring/tests/test_news_agent.py` (factory tests only; news-stub behaviour is tested in Task 10)

- [ ] **Step 1: Write the failing test**

Create `monitoring/tests/test_news_agent.py`:

```python
"""Tests for the model factory (Task 2) and News Context Agent (Task 10)."""
import pytest

from agentic.model import default_model, OfflineStubModel, OllamaModel


def test_default_model_stub():
    assert isinstance(default_model("stub"), OfflineStubModel)


def test_default_model_ollama_with_name():
    m = default_model("ollama:qwen2.5:3b")
    assert isinstance(m, OllamaModel)
    assert m.model == "qwen2.5:3b"


def test_default_model_ollama_default_is_qwen():
    m = default_model("ollama")
    assert isinstance(m, OllamaModel)
    assert m.model == "qwen2.5:3b"


def test_default_model_env_fallback(monkeypatch):
    monkeypatch.delenv("MONITOR_MODEL", raising=False)
    assert isinstance(default_model(None), OfflineStubModel)
    monkeypatch.setenv("MONITOR_MODEL", "ollama:qwen2.5:3b")
    m = default_model(None)
    assert isinstance(m, OllamaModel)


def test_default_model_rejects_unknown():
    with pytest.raises(ValueError):
        default_model("gpt-4")
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd monitoring && python -m pytest tests/test_news_agent.py -v`
Expected: FAIL — `ImportError: cannot import name 'default_model'`.

- [ ] **Step 3: Implement**

In `monitoring/agentic/model.py`:

(a) add `import os` to the imports block.

(b) change the `OllamaModel.__init__` default from `model: str = "llama3.1"` to `model: str = "qwen2.5:3b"`.

(c) append at the end of the file:

```python
def default_model(spec: str | None = None) -> LocalModel:
    """Build a LocalModel from a spec string.

    Specs: "stub" | "ollama" (qwen2.5:3b) | "ollama:<model-name>".
    Falls back to the MONITOR_MODEL env var, then to the offline stub, so tests
    and CI never require a running Ollama server.
    """
    spec = spec or os.environ.get("MONITOR_MODEL", "stub")
    if spec in ("stub", "offline"):
        return OfflineStubModel()
    if spec == "ollama":
        return OllamaModel()
    if spec.startswith("ollama:"):
        return OllamaModel(model=spec.split(":", 1)[1])
    raise ValueError(f"unknown model spec {spec!r}; use 'stub', 'ollama', or 'ollama:<name>'")
```

(d) In `OfflineStubModel.complete`, add news-prompt handling as the FIRST statement of the method (before `ctx = self._parse_user(user)`), so the stub can stand in for the News Context Agent in offline tests:

```python
        if "News Context Agent" in system:
            n_articles = user.count("\n- [")
            risky = any(t in user.lower() for t in ("meltdown", "crash", "liquidat", "crisis"))
            risk = "HIGH" if (risky and n_articles >= 3) else ("ELEVATED" if risky else "LOW")
            m = re.search(r"as_of\):\s*([0-9-]+)", user)
            return {
                "overall_risk": risk,
                "risk_flags": (
                    [{"flag": "stress_language_in_news", "evidence": "risk terms present in headlines"}]
                    if risky else []
                ),
                "narrative": f"Reviewed {n_articles} filtered articles; "
                             + ("stress language present." if risky else "no stress language."),
                "confidence": 0.5,
                "as_of": m.group(1) if m else "",
                "n_articles": n_articles,
            }
```

(e) In `monitoring/agentic/__init__.py`, add `default_model` to the imports from `.model` and to `__all__` (match the file's existing export style).

- [ ] **Step 4: Run to verify it passes**

Run: `cd monitoring && python -m pytest tests/test_news_agent.py -v`
Expected: 5 PASS. Also run the full suite (`python -m pytest -q`) — the stub change must not break Week-2 tests (the new branch only triggers on news prompts).

- [ ] **Step 5: Commit**

```bash
git add monitoring/agentic/model.py monitoring/agentic/__init__.py monitoring/tests/test_news_agent.py
git commit -m "feat(agentic): default_model factory, qwen2.5:3b default, stub news-mode"
```

---

### Task 3: FNSPID download script

Network/disk task — the *download* itself is manual-ish (22 GB, ~hours depending on bandwidth); the script is thin and is verified by running it. Free disk is ~81 GB; the raw CSV needs ~22 GB and the filtered store ~1–3 GB.

**Files:**
- Create: `monitoring/news/__init__.py`
- Create: `monitoring/news/download_fnspid.py`

- [ ] **Step 1: Create the package init**

`monitoring/news/__init__.py`:

```python
"""Week-3 news pipeline: FNSPID store, risk filter, signal aggregator, triage."""
```

- [ ] **Step 2: Write the download script**

`monitoring/news/download_fnspid.py`:

```python
"""Download the FNSPID news CSV from HuggingFace and peek at its schema.

FNSPID (Dong et al. 2024): 15.7M time-aligned financial news records, 1999-2023.
We download the news file once to data/fnspid/raw/, then Task 4 stream-filters it
to the backtest window (2003-2016) as a parquet store; the raw CSV can be deleted
afterwards if disk is tight.

Usage (from monitoring/):
    python news/download_fnspid.py --list          # show repo files + sizes, pick one
    python news/download_fnspid.py                 # download default news CSV
    python news/download_fnspid.py --peek          # print first rows of downloaded CSV
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

REPO_ID = "Zdong104/FNSPID_Financial_News_Dataset"
DEFAULT_FILE = "Stock_news/nasdaq_exteral_data.csv"  # (sic — typo is upstream's)
RAW_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "fnspid" / "raw"


def list_files() -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    info = api.repo_info(REPO_ID, repo_type="dataset", files_metadata=True)
    for f in sorted(info.siblings, key=lambda s: s.rfilename):
        size_gb = (f.size or 0) / 1e9
        print(f"{size_gb:8.2f} GB  {f.rfilename}")


def download(filename: str) -> Path:
    from huggingface_hub import hf_hub_download

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = hf_hub_download(
        repo_id=REPO_ID, filename=filename, repo_type="dataset",
        local_dir=RAW_DIR,
    )
    print(f"Downloaded to {path}")
    return Path(path)


def peek(filename: str, n: int = 3) -> None:
    path = RAW_DIR / filename
    head = pd.read_csv(path, nrows=n)
    print("Columns:", list(head.columns))
    print(head.to_string(max_colwidth=60))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true", help="list repo files and sizes")
    ap.add_argument("--peek", action="store_true", help="print head of downloaded CSV")
    ap.add_argument("--file", default=DEFAULT_FILE)
    args = ap.parse_args()
    if args.list:
        list_files()
    elif args.peek:
        peek(args.file)
    else:
        download(args.file)


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run `--list` and confirm the target file**

Run: `cd monitoring && python news/download_fnspid.py --list`
Expected: a file listing including a large news CSV (≈20+ GB). If `Stock_news/nasdaq_exteral_data.csv` is not present under that exact name, note the actual news-CSV path and pass it via `--file` in the next step (and use it in Task 4's build command).

- [ ] **Step 4: Download (long-running — run in background / overnight if needed)**

Run: `cd monitoring && python news/download_fnspid.py`
Expected: file lands under `data/fnspid/raw/`. Then `python news/download_fnspid.py --peek` prints the columns — expect names like `Date, Article_title, Stock_symbol, Url, Publisher, Author, Article, Lsa_summary, ...`. Task 4's column auto-detection covers these; if the peek shows different names, extend `CANDIDATES` in `news/store.py` accordingly.

- [ ] **Step 5: Commit (script only — data is gitignored)**

```bash
git add monitoring/news/__init__.py monitoring/news/download_fnspid.py
git commit -m "feat(news): FNSPID download/inspect script"
```

---

### Task 4: FNSPID parquet store + `NewsStore.query`

**Files:**
- Create: `monitoring/news/store.py`
- Test: `monitoring/tests/test_news_store.py`

- [ ] **Step 1: Write the failing tests**

`monitoring/tests/test_news_store.py`:

```python
"""build_store: chunked CSV -> per-year parquet; NewsStore: date/ticker queries."""
import pandas as pd
import pytest

from news.store import build_store, NewsStore, detect_columns


@pytest.fixture
def raw_csv(tmp_path):
    df = pd.DataFrame({
        "Date": [
            "2007-08-06 09:00:00 UTC", "2007-08-09 12:00:00 UTC",
            "2010-05-06 14:45:00 UTC", "1999-01-04 10:00:00 UTC",   # out of range
            "not-a-date",                                            # unparseable
        ],
        "Article_title": ["Quant funds hit", "Selloff deepens", "Flash crash", "Old", "Bad"],
        "Stock_symbol": ["GS", "", "SPY", "IBM", "XX"],
        "Lsa_summary": ["s1", "s2", "s3", "s4", "s5"],
        "Publisher": ["P"] * 5,
        "Url": ["u"] * 5,
    })
    p = tmp_path / "raw.csv"
    df.to_csv(p, index=False)
    return p


def test_detect_columns(raw_csv):
    m = detect_columns(raw_csv)
    assert m["Date"] == "date" and m["Article_title"] == "title"
    assert m["Stock_symbol"] == "ticker" and m["Lsa_summary"] == "summary"


def test_build_store_filters_dates_and_partitions_by_year(tmp_path, raw_csv):
    out = tmp_path / "store"
    n = build_store(raw_csv, out, start="2003-01-01", end="2016-12-31", chunksize=2)
    assert n == 3  # 1999 row and unparseable row dropped
    assert (out / "year=2007.parquet").exists()
    assert (out / "year=2010.parquet").exists()
    assert not (out / "year=1999.parquet").exists()


def test_query_by_date_range(tmp_path, raw_csv):
    out = tmp_path / "store"
    build_store(raw_csv, out, start="2003-01-01", end="2016-12-31")
    store = NewsStore(out)
    df = store.query("2007-08-01", "2007-08-31")
    assert len(df) == 2
    assert list(df.columns) == ["date", "ticker", "title", "summary", "publisher", "url"]
    assert df["date"].is_monotonic_increasing
    assert df["date"].dt.tz is None  # naive timestamps, comparable to curve index


def test_query_by_ticker(tmp_path, raw_csv):
    out = tmp_path / "store"
    build_store(raw_csv, out, start="2003-01-01", end="2016-12-31")
    df = NewsStore(out).query("2007-01-01", "2010-12-31", tickers=["SPY"])
    assert len(df) == 1 and df.iloc[0]["title"] == "Flash crash"


def test_query_missing_years_returns_empty(tmp_path, raw_csv):
    out = tmp_path / "store"
    build_store(raw_csv, out, start="2003-01-01", end="2016-12-31")
    df = NewsStore(out).query("2013-01-01", "2013-12-31")
    assert df.empty and list(df.columns) == ["date", "ticker", "title", "summary", "publisher", "url"]
```

- [ ] **Step 2: Run to verify they fail**

Run: `cd monitoring && python -m pytest tests/test_news_store.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'news.store'`.

- [ ] **Step 3: Implement**

`monitoring/news/store.py`:

```python
"""FNSPID parquet store: chunked build from the raw CSV, date-indexed queries.

The raw FNSPID news CSV is ~22 GB; we never load it whole. ``build_store`` streams
it in chunks, keeps only the backtest window and the columns the pipeline needs,
and writes one parquet file per calendar year. ``NewsStore.query`` then reads only
the year files a date range touches — "indexed by date, ready to query for any
date in the backtest window" (VRI Week 3).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

COLS = ["date", "ticker", "title", "summary", "publisher", "url"]

SCHEMA = pa.schema([
    ("date", pa.timestamp("ns")),
    ("ticker", pa.string()),
    ("title", pa.string()),
    ("summary", pa.string()),
    ("publisher", pa.string()),
    ("url", pa.string()),
])

#: raw-CSV column name -> canonical name (first match per canonical wins)
CANDIDATES = {
    "date": ["Date", "date", "Publish_date", "publish_date"],
    "title": ["Article_title", "Title", "title", "Headline"],
    "ticker": ["Stock_symbol", "Symbol", "ticker", "symbol"],
    "summary": ["Lsa_summary", "Textrank_summary", "Luhn_summary", "Summary", "summary"],
    "publisher": ["Publisher", "publisher"],
    "url": ["Url", "URL", "url"],
}


def detect_columns(csv_path) -> dict[str, str]:
    """Map raw header names to canonical names. Requires date + title."""
    header = list(pd.read_csv(csv_path, nrows=0).columns)
    mapping: dict[str, str] = {}
    for canon, options in CANDIDATES.items():
        for opt in options:
            if opt in header:
                mapping[opt] = canon
                break
    found = set(mapping.values())
    if "date" not in found or "title" not in found:
        raise ValueError(f"could not find date/title columns in {header}")
    return mapping


def build_store(csv_path, out_dir, start="2003-01-01", end="2016-12-31",
                chunksize: int = 200_000) -> int:
    """Stream-filter the raw CSV into per-year parquet files. Returns rows kept."""
    csv_path, out_dir = Path(csv_path), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)

    mapping = detect_columns(csv_path)
    writers: dict[int, pq.ParquetWriter] = {}
    kept = 0
    try:
        for chunk in pd.read_csv(csv_path, usecols=list(mapping), chunksize=chunksize,
                                 dtype=str, on_bad_lines="skip"):
            chunk = chunk.rename(columns=mapping)
            for c in COLS:
                if c not in chunk.columns:
                    chunk[c] = ""
            dates = pd.to_datetime(chunk["date"], errors="coerce", utc=True)
            chunk["date"] = dates.dt.tz_localize(None)
            chunk = chunk.dropna(subset=["date"])
            chunk = chunk[(chunk["date"] >= start_ts) & (chunk["date"] <= end_ts)]
            if chunk.empty:
                continue
            chunk = chunk[COLS]
            for c in COLS[1:]:
                chunk[c] = chunk[c].fillna("").astype(str)
            kept += len(chunk)
            for year, group in chunk.groupby(chunk["date"].dt.year):
                if year not in writers:
                    writers[year] = pq.ParquetWriter(out_dir / f"year={year}.parquet", SCHEMA)
                writers[year].write_table(
                    pa.Table.from_pandas(group, schema=SCHEMA, preserve_index=False))
    finally:
        for w in writers.values():
            w.close()
    return kept


class NewsStore:
    """Read-side of the store: date-range (and optional ticker) queries."""

    def __init__(self, root):
        self.root = Path(root)

    def query(self, start, end, tickers: list[str] | None = None) -> pd.DataFrame:
        start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
        frames = []
        for year in range(start_ts.year, end_ts.year + 1):
            path = self.root / f"year={year}.parquet"
            if path.exists():
                frames.append(pd.read_parquet(path))
        if not frames:
            return pd.DataFrame(columns=COLS)
        df = pd.concat(frames, ignore_index=True)
        df = df[(df["date"] >= start_ts) & (df["date"] <= end_ts)]
        if tickers is not None:
            df = df[df["ticker"].isin(tickers)]
        return df.sort_values("date").reset_index(drop=True)[COLS]
```

- [ ] **Step 4: Run to verify they pass**

Run: `cd monitoring && python -m pytest tests/test_news_store.py -v`
Expected: 5 PASS.

- [ ] **Step 5: Build the real store (long-running, ~30–90 min of streaming I/O)**

```bash
cd monitoring && python -c "
from news.store import build_store
n = build_store('../data/fnspid/raw/Stock_news/nasdaq_exteral_data.csv',
                '../data/fnspid/store')
print(f'{n:,} rows kept (2003-2016)')
"
```

Expected: prints a row count and `data/fnspid/store/year=*.parquet` files exist. (Adjust the raw path if Task 3 found a different filename.)

- [ ] **Step 6: Commit**

```bash
git add monitoring/news/store.py monitoring/tests/test_news_store.py
git commit -m "feat(news): FNSPID parquet store (chunked build + date-indexed query)"
```

---

### Task 5: Risk filter (regex/keyword)

**Files:**
- Create: `monitoring/news/filter.py`
- Test: `monitoring/tests/test_news_filter.py`

- [ ] **Step 1: Write the failing tests**

`monitoring/tests/test_news_filter.py`:

```python
import pandas as pd

from news.filter import score_text, filter_news


def test_score_text_matches_risk_terms():
    terms = score_text("Quant funds face margin calls as selloff deepens")
    assert "margin call" in terms and "sell-off/selloff" in terms


def test_score_text_clean_headline():
    assert score_text("Apple announces new iPhone lineup") == []


def test_filter_news_adds_columns_and_keeps_only_hits():
    df = pd.DataFrame({
        "date": pd.to_datetime(["2007-08-06", "2007-08-07"]),
        "ticker": ["GS", "AAPL"],
        "title": ["Hedge fund liquidation triggers market turmoil", "New product launch"],
        "summary": ["", ""],
        "publisher": ["P", "P"],
        "url": ["u", "u"],
    })
    out = filter_news(df)
    assert len(out) == 1
    assert out.iloc[0]["ticker"] == "GS"
    assert out.iloc[0]["n_risk_terms"] >= 2  # liquidation + turmoil
    assert isinstance(out.iloc[0]["risk_terms"], list)


def test_filter_news_empty_input():
    df = pd.DataFrame(columns=["date", "ticker", "title", "summary", "publisher", "url"])
    out = filter_news(df)
    assert out.empty and "n_risk_terms" in out.columns
```

- [ ] **Step 2: Run to verify they fail**

Run: `cd monitoring && python -m pytest tests/test_news_filter.py -v`
Expected: FAIL — no module `news.filter`.

- [ ] **Step 3: Implement**

`monitoring/news/filter.py`:

```python
"""Stage 1 of the news pipeline: cheap regex/keyword risk filter.

A hand-built lexicon of market-stress language. Deliberately transparent (it goes
in the methodology write-up) and deliberately recall-oriented — stage 2 (the
quantitative aggregator) and stage 3 (triage + LLM) handle precision.
"""

from __future__ import annotations

import re

import pandas as pd

#: (label, pattern) — label is what appears in risk_terms lists / agent context.
RISK_PATTERNS: list[tuple[str, str]] = [
    ("sell-off/selloff", r"sell[- ]?off"),
    ("meltdown", r"meltdown"),
    ("liquidation", r"liquidat\w*"),
    ("margin call", r"margin call"),
    ("deleveraging", r"deleverag\w*"),
    ("unwind", r"unwind\w*"),
    ("crash", r"crash\w*"),
    ("plunge", r"plung\w*"),
    ("turmoil", r"turmoil"),
    ("contagion", r"contagion"),
    ("credit crunch", r"credit crunch"),
    ("subprime", r"subprime"),
    ("default", r"default\w*"),
    ("bailout", r"bail[- ]?out\w*"),
    ("bankruptcy", r"bankrupt\w*"),
    ("downgrade", r"downgrad\w*"),
    ("recession", r"recession\w*"),
    ("volatility spike", r"volatil\w*"),
    ("panic", r"panic\w*"),
    ("crisis", r"cris[ie]s"),
    ("quant fund stress", r"quant\w*\s+(?:fund|strateg|model|trading)"),
    ("hedge fund stress", r"hedge[- ]fund\w*\s+(?:loss|losses|collapse|redemption|blow)"),
]

_COMPILED = [(label, re.compile(pat, re.IGNORECASE)) for label, pat in RISK_PATTERNS]


def score_text(text: str) -> list[str]:
    """Return the distinct risk-term labels matched in ``text``."""
    if not isinstance(text, str) or not text:
        return []
    return [label for label, rx in _COMPILED if rx.search(text)]


def filter_news(df: pd.DataFrame, text_cols=("title", "summary")) -> pd.DataFrame:
    """Keep only risk-relevant articles; add risk_terms / n_risk_terms columns."""
    out = df.copy()
    if out.empty:
        out["risk_terms"] = pd.Series(dtype=object)
        out["n_risk_terms"] = pd.Series(dtype=int)
        return out
    joined = out[list(text_cols)].fillna("").agg(" ".join, axis=1)
    out["risk_terms"] = joined.map(score_text)
    out["n_risk_terms"] = out["risk_terms"].map(len)
    return out[out["n_risk_terms"] > 0].reset_index(drop=True)
```

- [ ] **Step 4: Run to verify they pass**

Run: `cd monitoring && python -m pytest tests/test_news_filter.py -v`
Expected: 4 PASS.

- [ ] **Step 5: Commit**

```bash
git add monitoring/news/filter.py monitoring/tests/test_news_filter.py
git commit -m "feat(news): regex risk-lexicon filter (pipeline stage 1)"
```

---

### Task 6: Coverage report — GATE on Aug-2007

**Files:**
- Create: `monitoring/news/coverage.py`
- Test: `monitoring/tests/test_news_coverage.py`

- [ ] **Step 1: Write the failing test**

`monitoring/tests/test_news_coverage.py`:

```python
import pandas as pd

from news.coverage import coverage_report
from news.store import build_store, NewsStore
from windows import ALL_WINDOWS


def test_coverage_report_counts_per_window(tmp_path):
    df = pd.DataFrame({
        "Date": ["2007-08-06 10:00:00 UTC", "2007-08-07 10:00:00 UTC",
                 "2013-06-03 10:00:00 UTC"],
        "Article_title": ["Quant meltdown hits funds", "Markets calm", "Quiet day"],
        "Stock_symbol": ["GS", "GS", "SPY"],
        "Lsa_summary": ["", "", ""],
        "Publisher": ["P"] * 3,
        "Url": ["u"] * 3,
    })
    raw = tmp_path / "raw.csv"
    df.to_csv(raw, index=False)
    build_store(raw, tmp_path / "store")

    rep = coverage_report(NewsStore(tmp_path / "store"), ALL_WINDOWS)
    rep = rep.set_index("window")
    assert rep.loc["quant_meltdown_2007", "n_articles"] == 2
    assert rep.loc["quant_meltdown_2007", "n_risk_articles"] == 1
    assert rep.loc["calm_2013_2014", "n_articles"] == 1
    assert rep.loc["gfc_lehman_2008", "n_articles"] == 0
    assert set(rep.columns) >= {"kind", "start", "end", "n_articles", "n_risk_articles",
                                "articles_per_day", "adequate"}
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd monitoring && python -m pytest tests/test_news_coverage.py -v`
Expected: FAIL — no module `news.coverage`.

- [ ] **Step 3: Implement**

`monitoring/news/coverage.py`:

```python
"""Per-window FNSPID coverage report — the Week-3 data gate.

FNSPID's news density is known to be much higher post-2009. The Aug-2007 quant
meltdown is this project's headline gating event, so before building agents on
top of the news store we MEASURE how much news each evaluation window actually
has, write it to results/news_coverage.csv, and flag inadequate event windows.
An inadequate 2007 window is a supervisor-decision point, not something to
paper over (see WEEK3_STATUS).

Usage (from monitoring/): python news/coverage.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from news.filter import filter_news
from news.store import NewsStore
from windows import ALL_WINDOWS

#: an event window with fewer risk-relevant articles than this is flagged.
MIN_RISK_ARTICLES = 25

RESULTS = Path(__file__).resolve().parent.parent / "results"
STORE = Path(__file__).resolve().parent.parent.parent / "data" / "fnspid" / "store"


def coverage_report(store: NewsStore, windows=ALL_WINDOWS) -> pd.DataFrame:
    rows = []
    for w in windows:
        df = store.query(w.start, w.end)
        risky = filter_news(df)
        n_days = max((pd.Timestamp(w.end) - pd.Timestamp(w.start)).days, 1)
        rows.append({
            "window": w.name,
            "kind": w.kind,
            "start": w.start,
            "end": w.end,
            "n_articles": len(df),
            "n_risk_articles": len(risky),
            "articles_per_day": round(len(df) / n_days, 2),
            "adequate": bool(len(risky) >= MIN_RISK_ARTICLES) if w.kind == "event" else True,
        })
    return pd.DataFrame(rows)


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    rep = coverage_report(NewsStore(STORE))
    rep.to_csv(RESULTS / "news_coverage.csv", index=False)
    print(rep.to_string(index=False))
    thin = rep[(rep["kind"] == "event") & (~rep["adequate"])]
    if not thin.empty:
        print("\n*** GATE: inadequate news coverage on event window(s): "
              f"{', '.join(thin['window'])} ***")
        print("Escalate to supervisor before Week-5 evaluation; document in WEEK3_STATUS.md.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run to verify the test passes**

Run: `cd monitoring && python -m pytest tests/test_news_coverage.py -v`
Expected: PASS.

- [ ] **Step 5: Run the real gate**

Run: `cd monitoring && python news/coverage.py`
Expected: table printed + `monitoring/results/news_coverage.csv` written.

**GATE DECISION:** if `quant_meltdown_2007` is flagged inadequate, STOP and record the numbers in `docs/WEEK3_STATUS.md` (Task 13) with options for the supervisor: (a) run the Aug-2007 agentic pass on market-level macro + whatever news exists and report coverage as a limitation; (b) supplement 2007 news from another dated source. Continue with the remaining tasks either way — the pipeline is window-agnostic — but the end-to-end run in Task 12 must state the coverage caveat.

- [ ] **Step 6: Commit**

```bash
git add monitoring/news/coverage.py monitoring/tests/test_news_coverage.py monitoring/results/news_coverage.csv
git commit -m "feat(news): per-window coverage gate (FNSPID density by evaluation window)"
```

---

### Task 7: Quantitative signal aggregator

**Files:**
- Create: `monitoring/news/aggregate.py`
- Test: `monitoring/tests/test_news_aggregate.py`

- [ ] **Step 1: Write the failing tests**

`monitoring/tests/test_news_aggregate.py`:

```python
import numpy as np
import pandas as pd

from news.aggregate import daily_signals
from news.filter import filter_news


def _mk_news(dates, titles):
    return pd.DataFrame({
        "date": pd.to_datetime(dates), "ticker": "", "title": titles,
        "summary": "", "publisher": "", "url": "",
    })


def test_daily_signals_counts_and_index():
    all_news = _mk_news(
        ["2007-08-01", "2007-08-01", "2007-08-03"],
        ["Calm day", "Another story", "Market selloff and panic"],
    )
    sig = daily_signals(all_news, filter_news(all_news))
    assert sig.loc["2007-08-01", "n_articles"] == 2
    assert sig.loc["2007-08-02", "n_articles"] == 0     # calendar-continuous
    assert sig.loc["2007-08-03", "n_risk"] == 1
    assert sig.loc["2007-08-03", "intensity"] == 2       # selloff + panic


def test_intensity_z_is_causal_and_spikes():
    # 80 quiet days then one very risky day
    dates = pd.date_range("2007-05-01", periods=81, freq="D")
    titles = ["Ordinary business story"] * 80 + \
             ["Meltdown: margin calls, liquidation, panic selloff"]
    all_news = _mk_news(dates, titles)
    sig = daily_signals(all_news, filter_news(all_news), baseline_days=60)
    last, prev = dates[-1], dates[-2]
    assert sig.loc[last, "intensity_z"] > 3
    # baseline for day t must exclude day t (shifted) => quiet prev day has small |z|
    assert abs(sig.loc[prev, "intensity_z"]) < 1 or np.isnan(sig.loc[prev, "intensity_z"])


def test_z_nan_during_warmup():
    all_news = _mk_news(["2007-08-01"], ["Calm"])
    sig = daily_signals(all_news, filter_news(all_news), baseline_days=60)
    assert np.isnan(sig["intensity_z"]).all()
```

- [ ] **Step 2: Run to verify they fail**

Run: `cd monitoring && python -m pytest tests/test_news_aggregate.py -v`
Expected: FAIL — no module `news.aggregate`.

- [ ] **Step 3: Implement**

`monitoring/news/aggregate.py`:

```python
"""Stage 2 of the news pipeline: daily quantitative news signals.

Turns article-level risk matches into a causal daily intensity series:
  n_articles   — all articles that calendar day
  n_risk       — articles with >=1 risk-term hit
  intensity    — total risk-term hits (a risk-language volume proxy)
  intensity_z  — intensity vs a TRAILING baseline (mean/std over the prior
                 ``baseline_days`` days, shifted one day so day t never
                 baselines on itself — causal by construction)
"""

from __future__ import annotations

import pandas as pd


def daily_signals(all_news: pd.DataFrame, risk_news: pd.DataFrame,
                  baseline_days: int = 60, min_baseline: int = 20) -> pd.DataFrame:
    """Aggregate article- to day-level signals on a continuous calendar index."""
    if all_news.empty:
        return pd.DataFrame(columns=["n_articles", "n_risk", "intensity", "intensity_z"])

    day_all = all_news["date"].dt.normalize()
    idx = pd.date_range(day_all.min(), day_all.max(), freq="D")

    out = pd.DataFrame(index=idx)
    out["n_articles"] = day_all.value_counts().reindex(idx, fill_value=0)
    if risk_news.empty:
        out["n_risk"] = 0
        out["intensity"] = 0
    else:
        day_risk = risk_news["date"].dt.normalize()
        out["n_risk"] = day_risk.value_counts().reindex(idx, fill_value=0)
        out["intensity"] = (risk_news.groupby(day_risk)["n_risk_terms"].sum()
                            .reindex(idx, fill_value=0))

    base = out["intensity"].rolling(baseline_days, min_periods=min_baseline)
    mean, std = base.mean().shift(1), base.std(ddof=1).shift(1)
    # A zero-variance quiet baseline must not mute a spike: floor the std at one
    # risk-term hit. Warm-up days stay NaN via the NaN mean.
    std_eff = std.where(std > 0, 1.0)
    out["intensity_z"] = (out["intensity"] - mean) / std_eff
    return out
```

- [ ] **Step 4: Run to verify they pass**

Run: `cd monitoring && python -m pytest tests/test_news_aggregate.py -v`
Expected: 3 PASS.

- [ ] **Step 5: Commit**

```bash
git add monitoring/news/aggregate.py monitoring/tests/test_news_aggregate.py
git commit -m "feat(news): daily signal aggregator with causal trailing-z (stage 2)"
```

---

### Task 8: Triage logic

**Files:**
- Create: `monitoring/news/triage.py`
- Test: `monitoring/tests/test_news_triage.py`

- [ ] **Step 1: Write the failing tests**

`monitoring/tests/test_news_triage.py`:

```python
from news.triage import decide, TriageDecision


def test_classical_escalation_beats_everything():
    d = decide(intensity_z=0.0, n_detectors_recent=3, aggregate_recent=True)
    assert d.mode == "classical_escalation"


def test_high_news_z_routes_to_thinking():
    d = decide(intensity_z=3.0, n_detectors_recent=0, aggregate_recent=False)
    assert d.mode == "thinking"


def test_mild_news_or_single_detector_routes_to_cheap():
    assert decide(1.5, 0, False).mode == "cheap"
    assert decide(0.0, 1, False).mode == "cheap"


def test_quiet_day_skips():
    d = decide(intensity_z=0.1, n_detectors_recent=0, aggregate_recent=False)
    assert d.mode == "skip"


def test_nan_z_treated_as_no_news_signal():
    assert decide(float("nan"), 0, False).mode == "skip"
    assert decide(float("nan"), 1, False).mode == "cheap"


def test_decision_carries_reason():
    d = decide(3.0, 0, False)
    assert isinstance(d, TriageDecision) and d.reason
```

- [ ] **Step 2: Run to verify they fail**

Run: `cd monitoring && python -m pytest tests/test_news_triage.py -v`
Expected: FAIL — no module `news.triage`.

- [ ] **Step 3: Implement**

`monitoring/news/triage.py`:

```python
"""Stage 3 of the news pipeline: triage — how much model do we spend today?

Modes (VRI Week 3):
  skip                 — no signal anywhere; log a heuristic no-op, no LLM call
  cheap                — mild signal: one LLM pass on a compact context
  thinking             — strong NEWS signal: extended-reasoning prompt
  classical_escalation — the classical aggregate fired: always run the full
                         agentic assessment regardless of news

Thresholds are fixed a priori (same no-in-sample-tuning discipline as the
classical detectors — see WEEK2_STATUS open item 1).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

Z_CHEAP = 1.0
Z_THINKING = 2.5

MODES = ("skip", "cheap", "thinking", "classical_escalation")


@dataclass(frozen=True)
class TriageDecision:
    mode: str
    reason: str


def decide(intensity_z: float, n_detectors_recent: int, aggregate_recent: bool,
           z_cheap: float = Z_CHEAP, z_thinking: float = Z_THINKING) -> TriageDecision:
    """Pick a triage mode for one decision day.

    Args:
        intensity_z: causal news-intensity z-score for the day (NaN = warm-up/no news).
        n_detectors_recent: individual detectors that alarmed within the last 5 days.
        aggregate_recent: whether the >=2-within-5-days aggregate fired in that span.
    """
    z = 0.0 if (intensity_z is None or math.isnan(intensity_z)) else intensity_z
    if aggregate_recent:
        return TriageDecision("classical_escalation",
                              "classical aggregate alarm within the last 5 days")
    if z >= z_thinking:
        return TriageDecision("thinking", f"news intensity z={z:.1f} >= {z_thinking}")
    if z >= z_cheap or n_detectors_recent >= 1:
        return TriageDecision("cheap",
                              f"news z={z:.1f} or {n_detectors_recent} detector(s) recent")
    return TriageDecision("skip", "no news or detector signal")
```

- [ ] **Step 4: Run to verify they pass**

Run: `cd monitoring && python -m pytest tests/test_news_triage.py -v`
Expected: 6 PASS.

- [ ] **Step 5: Commit**

```bash
git add monitoring/news/triage.py monitoring/tests/test_news_triage.py
git commit -m "feat(news): triage logic — skip/cheap/thinking/classical-escalation (stage 3)"
```

---

### Task 9: Macro data — FRED/ALFRED fetch + as-of queries + context block

**Files:**
- Create: `monitoring/macro/__init__.py`
- Create: `monitoring/macro/fetch_macro.py`
- Create: `monitoring/macro/asof.py`
- Create: `monitoring/macro/context.py`
- Test: `monitoring/tests/test_macro_asof.py`, `monitoring/tests/test_macro_context.py`

Prereq: a free FRED API key from https://fred.stlouisfed.org/docs/api/api_key.html, exported as `FRED_API_KEY`. (The user must obtain this — it takes ~2 minutes. Unit tests do NOT need it; only the real fetch does.)

- [ ] **Step 1: Write the failing as-of tests**

`monitoring/tests/test_macro_asof.py`:

```python
import pandas as pd

from macro.asof import asof_vintage, asof_daily
from macro.fetch_macro import parse_observations


def _vintage_df():
    # UNRATE-style: July-2007 obs first published Aug-3-2007 at 4.6, revised to 4.7 on
    # Sep-7-2007; August obs published Sep-7-2007.
    return pd.DataFrame({
        "date": pd.to_datetime(["2007-07-01", "2007-07-01", "2007-08-01"]),
        "value": [4.6, 4.7, 4.6],
        "realtime_start": pd.to_datetime(["2007-08-03", "2007-09-07", "2007-09-07"]),
        # open-ended vintages: ALFRED's "9999-12-31" overflows datetime64[ns];
        # parse_observations maps it to Timestamp.max — fixtures use a safe far date
        "realtime_end": pd.to_datetime(["2007-09-06", "2262-04-01", "2262-04-01"]),
    })


def test_asof_vintage_uses_first_print_not_revision():
    v = asof_vintage(_vintage_df(), "2007-08-15")
    assert v == {"value": 4.6, "obs_date": "2007-07-01", "published": "2007-08-03"}


def test_asof_vintage_sees_revision_and_new_obs_later():
    v = asof_vintage(_vintage_df(), "2007-09-10")
    assert v["obs_date"] == "2007-08-01" and v["value"] == 4.6


def test_asof_vintage_before_any_release():
    assert asof_vintage(_vintage_df(), "2007-08-01") is None


def test_asof_daily_last_value_on_or_before():
    df = pd.DataFrame({
        "date": pd.to_datetime(["2007-08-03", "2007-08-06", "2007-08-07"]),
        "value": [25.16, 26.25, None],   # None = market holiday / missing print
    })
    v = asof_daily(df, "2007-08-07")
    assert v == {"value": 26.25, "obs_date": "2007-08-06"}
    assert asof_daily(df, "2007-08-01") is None


def test_parse_observations_handles_dots():
    payload = {"observations": [
        {"date": "2007-08-03", "value": "25.16",
         "realtime_start": "2007-08-03", "realtime_end": "9999-12-31"},
        {"date": "2007-08-06", "value": ".",
         "realtime_start": "2007-08-06", "realtime_end": "9999-12-31"},
    ]}
    df = parse_observations(payload)
    assert len(df) == 2
    assert df["value"].isna().iloc[1]
    assert str(df["date"].dtype).startswith("datetime64")
```

- [ ] **Step 2: Write the failing context test**

`monitoring/tests/test_macro_context.py`:

```python
import pandas as pd
import pytest

from macro.context import macro_context
from agentic.guardrails import assert_no_lookahead


@pytest.fixture
def macro_dir(tmp_path):
    pd.DataFrame({
        "date": pd.to_datetime(["2007-08-03", "2007-08-06"]),
        "value": [25.16, 26.25],
    }).to_parquet(tmp_path / "VIXCLS.parquet")
    pd.DataFrame({
        "date": pd.to_datetime(["2007-08-03", "2007-08-06"]),
        "value": [4.70, 4.73],
    }).to_parquet(tmp_path / "DGS10.parquet")
    pd.DataFrame({
        "date": pd.to_datetime(["2007-08-03", "2007-08-06"]),
        "value": [4.56, 4.58],
    }).to_parquet(tmp_path / "DGS2.parquet")
    pd.DataFrame({
        "date": pd.to_datetime(["2007-07-01"]),
        "value": [4.6],
        "realtime_start": pd.to_datetime(["2007-08-03"]),
        "realtime_end": pd.to_datetime(["2262-04-01"]),
    }).to_parquet(tmp_path / "UNRATE.parquet")
    return tmp_path


def test_macro_context_assembles_asof_block(macro_dir):
    ctx = macro_context("2007-08-06", macro_dir)
    assert ctx["vixcls"]["value"] == 26.25
    assert ctx["unrate"]["published"] == "2007-08-03"
    assert ctx["yield_curve_10y_2y"] == pytest.approx(0.15, abs=1e-9)


def test_macro_context_never_leaks_future(macro_dir):
    ctx = macro_context("2007-08-03", macro_dir)
    assert ctx["vixcls"]["obs_date"] == "2007-08-03"
    assert_no_lookahead({"macro": ctx}, "2007-08-03")   # must not raise


def test_macro_context_missing_series_is_omitted(macro_dir):
    ctx = macro_context("2007-08-06", macro_dir)
    assert "tedrate" not in ctx    # no TEDRATE.parquet in fixture
```

- [ ] **Step 3: Run to verify they fail**

Run: `cd monitoring && python -m pytest tests/test_macro_asof.py tests/test_macro_context.py -v`
Expected: FAIL — no module `macro`.

- [ ] **Step 4: Implement**

`monitoring/macro/__init__.py`:

```python
"""Week-3 macro data: FRED/ALFRED point-in-time series for agent context."""
```

`monitoring/macro/fetch_macro.py`:

```python
"""Fetch macro series from FRED/ALFRED into data/macro/*.parquet.

Point-in-time discipline:
  * VINTAGE_SERIES (monthly releases that get revised — UNRATE, CPIAUCSL, INDPRO)
    are fetched from ALFRED with the full realtime history, so ``asof_vintage``
    can reconstruct exactly what was published by any decision date. This is the
    point-in-time-correct route to BLS data (BLS's own API serves only the
    latest revisions — using it would be lookahead).
  * DAILY_SERIES (market prints, unrevised — VIX; Treasury constant-maturity
    yields DGS10/DGS2, i.e. the Treasury integration; fed funds; TED spread)
    are fetched as plain FRED series.

Usage (from monitoring/, with FRED_API_KEY exported):
    python macro/fetch_macro.py
"""

from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

FRED_OBS_URL = "https://api.stlouisfed.org/fred/series/observations"

VINTAGE_SERIES = ["UNRATE", "CPIAUCSL", "INDPRO"]
DAILY_SERIES = ["VIXCLS", "DGS10", "DGS2", "DFF", "TEDRATE"]

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "macro"
_LIMIT = 100_000


def parse_observations(payload: dict) -> pd.DataFrame:
    """FRED JSON payload -> DataFrame(date, value[, realtime_start, realtime_end])."""
    obs = payload["observations"]
    df = pd.DataFrame(obs)
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"].replace(".", np.nan), errors="coerce")
    for col in ("realtime_start", "realtime_end"):
        if col in df.columns:
            # ALFRED marks the current vintage with "9999-12-31", which overflows
            # datetime64[ns]; coerce it to NaT and pin to Timestamp.max.
            df[col] = pd.to_datetime(df[col], errors="coerce")
            if col == "realtime_end":
                df[col] = df[col].fillna(pd.Timestamp.max)
    return df


def fetch_series(series_id: str, api_key: str, vintages: bool = False) -> pd.DataFrame:
    params = {
        "series_id": series_id, "api_key": api_key, "file_type": "json",
        "limit": _LIMIT,
    }
    if vintages:
        # Full realtime history => one row per (obs date, vintage span).
        params["realtime_start"] = "1990-01-01"
        params["realtime_end"] = "9999-12-31"
    url = f"{FRED_OBS_URL}?{urllib.parse.urlencode(params)}"
    with urllib.request.urlopen(url, timeout=60) as resp:
        payload = json.loads(resp.read().decode())
    if payload.get("count", 0) >= _LIMIT:
        raise RuntimeError(f"{series_id}: hit the {_LIMIT}-row limit; add offset paging")
    df = parse_observations(payload)
    keep = ["date", "value"] + (["realtime_start", "realtime_end"] if vintages else [])
    return df[keep]


def main() -> None:
    api_key = os.environ.get("FRED_API_KEY")
    if not api_key:
        raise SystemExit("Set FRED_API_KEY (free key: https://fred.stlouisfed.org/docs/api/api_key.html)")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for sid in VINTAGE_SERIES:
        df = fetch_series(sid, api_key, vintages=True)
        df.to_parquet(DATA_DIR / f"{sid}.parquet")
        print(f"{sid:10} {len(df):6,} vintage rows -> {DATA_DIR / (sid + '.parquet')}")
    for sid in DAILY_SERIES:
        df = fetch_series(sid, api_key, vintages=False)
        df.to_parquet(DATA_DIR / f"{sid}.parquet")
        print(f"{sid:10} {len(df):6,} daily rows   -> {DATA_DIR / (sid + '.parquet')}")


if __name__ == "__main__":
    main()
```

`monitoring/macro/asof.py`:

```python
"""Point-in-time queries over the fetched macro parquets."""

from __future__ import annotations

import pandas as pd


def asof_vintage(df: pd.DataFrame, as_of) -> dict | None:
    """Latest observation *as it was known* at ``as_of`` (ALFRED vintage frame).

    Returns {"value", "obs_date", "published"} or None if nothing was
    published yet. Uses the newest vintage whose realtime_start <= as_of for
    the newest observation date published by then — i.e. first prints before
    their revisions arrive, revisions once they do.
    """
    as_of = pd.Timestamp(as_of)
    known = df[df["realtime_start"] <= as_of].dropna(subset=["value"])
    if known.empty:
        return None
    latest_obs = known["date"].max()
    row = (known[known["date"] == latest_obs]
           .sort_values("realtime_start").iloc[-1])
    return {"value": float(row["value"]),
            "obs_date": str(row["date"].date()),
            "published": str(row["realtime_start"].date())}


def asof_daily(df: pd.DataFrame, as_of) -> dict | None:
    """Last non-null daily print dated on/before ``as_of`` (unrevised series)."""
    as_of = pd.Timestamp(as_of)
    known = df[(df["date"] <= as_of)].dropna(subset=["value"])
    if known.empty:
        return None
    row = known.sort_values("date").iloc[-1]
    return {"value": float(row["value"]), "obs_date": str(row["date"].date())}
```

`monitoring/macro/context.py`:

```python
"""Assemble the JSON-serialisable macro block for agent context at a decision date.

Every value is as-of correct by construction (vintage or last-print-<=-as_of),
so the block passes ``guardrails.assert_no_lookahead`` — which run_agentic.py
still re-checks on the full context, fail-closed.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from macro.asof import asof_daily, asof_vintage
from macro.fetch_macro import DAILY_SERIES, VINTAGE_SERIES, DATA_DIR


def macro_context(as_of, data_dir: str | Path = DATA_DIR) -> dict:
    data_dir = Path(data_dir)
    out: dict = {}
    for sid in DAILY_SERIES:
        path = data_dir / f"{sid}.parquet"
        if path.exists():
            v = asof_daily(pd.read_parquet(path), as_of)
            if v is not None:
                out[sid.lower()] = v
    for sid in VINTAGE_SERIES:
        path = data_dir / f"{sid}.parquet"
        if path.exists():
            v = asof_vintage(pd.read_parquet(path), as_of)
            if v is not None:
                out[sid.lower()] = v
    if "dgs10" in out and "dgs2" in out:
        out["yield_curve_10y_2y"] = round(out["dgs10"]["value"] - out["dgs2"]["value"], 4)
    return out
```

- [ ] **Step 5: Run to verify tests pass**

Run: `cd monitoring && python -m pytest tests/test_macro_asof.py tests/test_macro_context.py -v`
Expected: 8 PASS.

- [ ] **Step 6: Real fetch (needs FRED_API_KEY; ~1 minute)**

```bash
export FRED_API_KEY=<key>
cd monitoring && python macro/fetch_macro.py
```

Expected: 8 parquet files under `data/macro/` with row counts printed. Sanity: `python -c "from macro.context import macro_context; print(macro_context('2007-08-09'))"` shows VIX in the mid-20s and a July-2007 UNRATE print published 2007-08-03.

- [ ] **Step 7: Commit**

```bash
git add monitoring/macro/ monitoring/tests/test_macro_asof.py monitoring/tests/test_macro_context.py
git commit -m "feat(macro): FRED/ALFRED vintage fetch + point-in-time as-of context block"
```

---

### Task 10: News Context Agent v1 — schema, prompt, guardrail scrub, runner

**Files:**
- Modify: `monitoring/agentic/schemas.py`
- Modify: `monitoring/agentic/prompts.py`
- Modify: `monitoring/agentic/guardrails.py`
- Create: `monitoring/agentic/news_agent.py`
- Modify: `monitoring/agentic/__init__.py`
- Test: `monitoring/tests/test_news_schema.py`, append to `monitoring/tests/test_news_agent.py`

- [ ] **Step 1: Write the failing schema tests**

`monitoring/tests/test_news_schema.py`:

```python
import pytest

from agentic.schemas import validate_news_flags, NewsFlags, SchemaError, OverallRisk


def _valid():
    return {
        "overall_risk": "HIGH",
        "risk_flags": [{"flag": "forced_deleveraging", "evidence": "margin-call headlines"}],
        "narrative": "Multiple reports of quant funds unwinding positions.",
        "confidence": 0.7,
        "as_of": "2007-08-09",
        "n_articles": 12,
    }


def test_valid_passes():
    nf = validate_news_flags(_valid())
    assert isinstance(nf, NewsFlags)
    assert nf.overall_risk == OverallRisk.HIGH
    assert nf.risk_flags[0]["flag"] == "forced_deleveraging"


def test_bad_risk_enum_rejected():
    bad = _valid() | {"overall_risk": "APOCALYPTIC"}
    with pytest.raises(SchemaError):
        validate_news_flags(bad)


def test_missing_field_rejected():
    bad = _valid()
    del bad["narrative"]
    with pytest.raises(SchemaError):
        validate_news_flags(bad)


def test_risk_flags_must_be_flag_evidence_dicts():
    bad = _valid() | {"risk_flags": ["just a string"]}
    with pytest.raises(SchemaError):
        validate_news_flags(bad)


def test_empty_flags_ok_for_low_risk():
    ok = _valid() | {"overall_risk": "LOW", "risk_flags": []}
    assert validate_news_flags(ok).risk_flags == []


def test_confidence_range_enforced():
    with pytest.raises(SchemaError):
        validate_news_flags(_valid() | {"confidence": 1.4})
```

- [ ] **Step 2: Write the failing agent/scrub tests**

Append to `monitoring/tests/test_news_agent.py`:

```python
import pytest

from agentic.guardrails import scrub_future_dated, LookaheadError, assert_no_lookahead
from agentic.news_agent import run_news_agent
from agentic.schemas import NewsFlags


def _articles():
    return [
        {"date": "2007-08-08", "ticker": "GS",
         "title": "Quant meltdown: funds face margin calls",
         "summary": "Widespread liquidation reported."},
        {"date": "2007-08-09", "ticker": "",
         "title": "Crisis talk grows as selloff continues", "summary": ""},
        {"date": "2007-08-09", "ticker": "MS",
         "title": "Fed meets on 2007-09-18 to discuss rates",   # future date in TEXT
         "summary": ""},
    ]


def test_scrub_future_dated_drops_future_text_mentions():
    kept = scrub_future_dated(_articles(), "2007-08-09")
    assert len(kept) == 2
    assert all("2007-09-18" not in a["title"] for a in kept)


def test_run_news_agent_returns_validated_flags():
    flags = run_news_agent("2007-08-09", _articles(), OfflineStubModel())
    assert isinstance(flags, NewsFlags)
    assert flags.as_of == "2007-08-09"
    assert flags.overall_risk in ("LOW", "ELEVATED", "HIGH", "SEVERE")


def test_run_news_agent_refuses_future_articles():
    future = [{"date": "2007-08-20", "ticker": "", "title": "tomorrow", "summary": ""}]
    flags = run_news_agent("2007-08-09", future, OfflineStubModel())
    assert flags.n_articles == 0   # timestamp filter removed everything, fail-closed
```

- [ ] **Step 3: Run to verify they fail**

Run: `cd monitoring && python -m pytest tests/test_news_schema.py tests/test_news_agent.py -v`
Expected: FAIL — `ImportError` on `validate_news_flags` / `scrub_future_dated` / `news_agent`.

- [ ] **Step 4: Implement — schemas**

In `monitoring/agentic/schemas.py`, add below the `Action` class:

```python
class OverallRisk:
    LOW = "LOW"
    ELEVATED = "ELEVATED"
    HIGH = "HIGH"
    SEVERE = "SEVERE"
    ALL = (LOW, ELEVATED, HIGH, SEVERE)
```

and append at the end of the file:

```python
#: Output contract for the News Context Agent (Week 3).
NEWS_FLAGS_JSON_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["overall_risk", "risk_flags", "narrative", "confidence", "as_of"],
    "properties": {
        "overall_risk": {"type": "string", "enum": list(OverallRisk.ALL)},
        "risk_flags": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["flag", "evidence"],
                "properties": {
                    "flag": {"type": "string", "minLength": 1, "maxLength": 80},
                    "evidence": {"type": "string", "minLength": 1, "maxLength": 500},
                },
            },
        },
        "narrative": {"type": "string", "minLength": 1, "maxLength": 1500},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "as_of": {"type": "string", "description": "ISO 8601 date (YYYY-MM-DD)"},
        "n_articles": {"type": "integer", "minimum": 0},
    },
}


@dataclass
class NewsFlags:
    """A validated News Context Agent output."""

    overall_risk: str
    risk_flags: list[dict]
    narrative: str
    confidence: float
    as_of: str
    n_articles: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


def validate_news_flags(obj: dict) -> NewsFlags:
    """Validate a raw dict against NEWS_FLAGS_JSON_SCHEMA (same style as
    ``validate_assessment``)."""
    if not isinstance(obj, dict):
        raise SchemaError(f"news flags must be a JSON object, got {type(obj).__name__}")

    missing = [k for k in NEWS_FLAGS_JSON_SCHEMA["required"] if k not in obj]
    if missing:
        raise SchemaError(f"missing required field(s): {missing}")

    if obj["overall_risk"] not in OverallRisk.ALL:
        raise SchemaError(f"invalid overall_risk {obj['overall_risk']!r}; allowed {OverallRisk.ALL}")

    flags = obj["risk_flags"]
    if not isinstance(flags, list):
        raise SchemaError("risk_flags must be a list")
    for f in flags:
        if (not isinstance(f, dict) or not isinstance(f.get("flag"), str)
                or not f["flag"].strip() or not isinstance(f.get("evidence"), str)
                or not f["evidence"].strip()):
            raise SchemaError(f"each risk flag needs non-empty 'flag' and 'evidence': {f!r}")

    narrative = obj["narrative"]
    if not isinstance(narrative, str) or not narrative.strip():
        raise SchemaError("narrative must be a non-empty string")

    conf = obj["confidence"]
    if not isinstance(conf, (int, float)) or isinstance(conf, bool):
        raise SchemaError("confidence must be a number")
    if not (0.0 <= float(conf) <= 1.0):
        raise SchemaError(f"confidence {conf} out of range [0, 1]")

    try:
        date.fromisoformat(obj["as_of"])
    except (ValueError, TypeError):
        raise SchemaError(f"as_of must be an ISO date, got {obj['as_of']!r}")

    n_articles = obj.get("n_articles", 0)
    if not isinstance(n_articles, int) or isinstance(n_articles, bool) or n_articles < 0:
        raise SchemaError("n_articles must be a non-negative integer")

    return NewsFlags(
        overall_risk=obj["overall_risk"],
        risk_flags=[{"flag": f["flag"].strip(), "evidence": f["evidence"].strip()} for f in flags],
        narrative=narrative.strip(),
        confidence=float(conf),
        as_of=obj["as_of"],
        n_articles=n_articles,
    )
```

- [ ] **Step 5: Implement — guardrail scrub**

Append to `monitoring/agentic/guardrails.py`:

```python
def scrub_future_dated(records, as_of, text_fields=("title", "summary")) -> list:
    """Drop records whose TEXT mentions a date after ``as_of`` (Week-3 news pipeline).

    ``filter_news_by_timestamp`` handles the publication timestamp; this handles the
    subtler leak of an on-time article whose body references a future scheduled date
    (fail-closed: the whole record is dropped, mirroring assert_no_lookahead's scan).
    """
    as_of = pd.Timestamp(as_of)
    kept = []
    for r in records:
        text = " ".join(str(r.get(f, "")) for f in text_fields)
        future = any(
            (ts := _try_date(m)) is not None and ts > as_of
            for m in _ISO_DATE_RE.findall(text)
        )
        if not future:
            kept.append(r)
    return kept
```

- [ ] **Step 6: Implement — prompt**

In `monitoring/agentic/prompts.py`, extend the `schemas` import to include `NEWS_FLAGS_JSON_SCHEMA, OverallRisk`, and append:

```python
NEWS_PROMPT_VERSION = "news-context-v1"

NEWS_SYSTEM = f"""\
You are a News Context Agent supporting the monitoring of a systematic trading strategy.
You receive financial news headlines/summaries published on or before a decision date.
Your job is to extract MARKET-RISK-relevant context: what stress, if any, is the news
describing, and how severe is it for systematic strategies?

You may ONLY use the information provided in the user message. It reflects what was
published as of the stated decision date. Do not use any knowledge of what happened
after that date.

Respond with a SINGLE JSON object and nothing else. It must conform to this schema:

{json.dumps(NEWS_FLAGS_JSON_SCHEMA, indent=2)}

Field guidance:
- overall_risk: {OverallRisk.ALL} — the aggregate stress level the news describes.
- risk_flags: zero or more {{flag, evidence}} items; 'flag' is a short snake_case label
  (e.g. "forced_deleveraging", "credit_stress"), 'evidence' quotes or paraphrases the
  specific headlines supporting it.
- narrative: 2-4 sentences summarising the news-implied market state.
- confidence: your calibrated confidence, 0..1.
- as_of: echo the decision date exactly.
- n_articles: how many articles you were shown.
"""


def build_news_prompt(as_of: str, articles: list[dict], max_articles: int = 40) -> tuple[str, str]:
    """Return (system, user) prompts for the News Context Agent.

    Args:
        as_of: decision date (ISO).
        articles: dicts with date/ticker/title/summary — already timestamp-filtered
                  and future-date-scrubbed by the caller.
    """
    lines = []
    for a in articles[:max_articles]:
        tick = a.get("ticker") or "MARKET"
        summ = (a.get("summary") or "")[:280]
        lines.append(f"- [{a['date']}] ({tick}) {a['title']}" + (f" — {summ}" if summ else ""))
    body = "\n".join(lines) if lines else "(no risk-relevant articles passed the filter)"
    user = (
        f"Decision date (as_of): {as_of}\n\n"
        f"Filtered news published on or before the decision date "
        f"({min(len(articles), max_articles)} of {len(articles)} shown):\n{body}\n\n"
        f"Return your JSON assessment now."
    )
    return NEWS_SYSTEM, user
```

- [ ] **Step 7: Implement — runner**

`monitoring/agentic/news_agent.py`:

```python
"""News Context Agent v1: filtered news in, validated risk flags out.

Mirror of ``runner.run_supervisor``: guardrails → prompt → local model →
schema validation (one repair retry) → JSONL log.
"""

from __future__ import annotations

from .guardrails import assert_no_lookahead, filter_news_by_timestamp, scrub_future_dated
from .model import LocalModel
from .prompts import build_news_prompt, NEWS_PROMPT_VERSION
from .schemas import validate_news_flags, NewsFlags, SchemaError


def run_news_agent(as_of: str, articles: list[dict], model: LocalModel,
                   logger=None, latency_s: float | None = None,
                   extra_label: str | None = None, max_articles: int = 40) -> NewsFlags:
    """Run the News Context Agent for one decision date.

    Args:
        as_of: decision date (ISO string).
        articles: dicts with keys date/ticker/title/summary; publication-date
                  filtering and future-text scrubbing are (re-)applied here,
                  fail-closed, regardless of what the caller did.
    """
    articles = filter_news_by_timestamp(articles, as_of, ts_field="date")
    articles = scrub_future_dated(articles, as_of)
    assert_no_lookahead({"as_of": as_of, "articles": articles}, as_of)

    system, user = build_news_prompt(as_of, articles, max_articles=max_articles)
    raw = model.complete(system, user)
    error = None
    try:
        flags = validate_news_flags(raw)
    except SchemaError as e:
        error = str(e)
        repair = user + f"\n\nYour previous output was invalid: {e}\nReturn corrected JSON."
        raw = model.complete(system, repair)
        flags = validate_news_flags(raw)

    # The model may not have been shown every article (max_articles cap) and may
    # miscount; pin the audited number.
    flags.n_articles = len(articles[:max_articles])

    if logger is not None:
        extra = {"model": model.name}
        if extra_label is not None:
            extra["label"] = extra_label
        logger.log_invocation(
            agent="news_context",
            prompt_version=NEWS_PROMPT_VERSION,
            as_of=as_of,
            raw_output=raw,
            assessment=flags.to_dict(),
            error=error,
            latency_s=latency_s,
            extra=extra,
        )
    return flags
```

- [ ] **Step 8: Export from the package**

In `monitoring/agentic/__init__.py`, additionally export: `run_news_agent`, `validate_news_flags`, `NewsFlags`, `OverallRisk`, `scrub_future_dated`, `filter_news_by_timestamp` (match the file's existing style).

- [ ] **Step 9: Run to verify everything passes**

Run: `cd monitoring && python -m pytest tests/test_news_schema.py tests/test_news_agent.py -v`
Expected: all PASS (6 schema + 5 factory + 3 agent). Then full suite: `python -m pytest -q` — no Week-2 regressions.

- [ ] **Step 10: Commit**

```bash
git add monitoring/agentic/ monitoring/tests/test_news_schema.py monitoring/tests/test_news_agent.py
git commit -m "feat(agentic): News Context Agent v1 — schema, prompt, fail-closed scrub, runner"
```

---

### Task 11: Performance Supervisor v2 — news + macro aware

**Files:**
- Modify: `monitoring/agentic/prompts.py`
- Modify: `monitoring/agentic/runner.py`
- Modify: `monitoring/agentic/__init__.py`
- Test: `monitoring/tests/test_supervisor_v2.py`

- [ ] **Step 1: Write the failing tests**

`monitoring/tests/test_supervisor_v2.py`:

```python
from agentic.prompts import (build_supervisor_prompt_v2, SUPERVISOR_PROMPT_VERSION_V2,
                             build_supervisor_prompt)
from agentic.runner import run_supervisor
from agentic.model import OfflineStubModel
from agentic.schemas import AgentAssessment


def _ctx(with_news=True, with_macro=True):
    ctx = {
        "as_of": "2007-08-09",
        "telemetry": {"last_date": "2007-08-09", "n_obs": 500,
                      "recent_mean_daily_return": -0.004, "recent_daily_vol": 0.012,
                      "recent_cum_return": -0.05, "current_drawdown": -0.06,
                      "recent_worst_day": -0.047},
        "detector_alarms": {"page_hinkley": ["2007-08-08"], "bocpd": []},
    }
    if with_macro:
        ctx["macro"] = {"vixcls": {"value": 26.25, "obs_date": "2007-08-08"},
                        "yield_curve_10y_2y": 0.15}
    if with_news:
        ctx["news"] = {"overall_risk": "HIGH",
                       "risk_flags": [{"flag": "forced_deleveraging",
                                       "evidence": "margin call headlines"}],
                       "narrative": "Quant funds unwinding.", "confidence": 0.7,
                       "as_of": "2007-08-09", "n_articles": 12}
    return ctx


def test_v2_prompt_includes_news_and_macro():
    system, user = build_supervisor_prompt_v2(_ctx())
    assert "forced_deleveraging" in user
    assert "vixcls" in user
    assert "2007-08-09" in user


def test_v2_prompt_omits_absent_blocks():
    _, user = build_supervisor_prompt_v2(_ctx(with_news=False, with_macro=False))
    assert "News Context Agent" not in user and "Macro" not in user


def test_run_supervisor_accepts_custom_builder_and_version():
    a = run_supervisor(_ctx(), OfflineStubModel(),
                       prompt_builder=build_supervisor_prompt_v2,
                       prompt_version=SUPERVISOR_PROMPT_VERSION_V2)
    assert isinstance(a, AgentAssessment)
    assert a.as_of == "2007-08-09"


def test_run_supervisor_default_builder_unchanged():
    ctx = _ctx(with_news=False, with_macro=False)
    a = run_supervisor(ctx, OfflineStubModel())
    assert isinstance(a, AgentAssessment)   # Week-2 path still works
```

- [ ] **Step 2: Run to verify they fail**

Run: `cd monitoring && python -m pytest tests/test_supervisor_v2.py -v`
Expected: FAIL — `ImportError: cannot import name 'build_supervisor_prompt_v2'`.

- [ ] **Step 3: Implement — prompt v2**

Append to `monitoring/agentic/prompts.py`:

```python
SUPERVISOR_PROMPT_VERSION_V2 = "supervisor-v2"

SUPERVISOR_SYSTEM_V2 = SUPERVISOR_SYSTEM + """
You may additionally receive:
- A macro-market block (VIX, Treasury yields, fed funds, TED spread, and the latest
  macro releases AS THEY WERE KNOWN on the decision date).
- A structured summary from a News Context Agent that has read the filtered financial
  news published up to the decision date.
Weigh telemetry and classical detectors first; use news/macro context to confirm,
explain (root_cause), or discount them. Cite detectors in detectors_cited as before.
"""


def build_supervisor_prompt_v2(context: dict) -> tuple[str, str]:
    """v2: telemetry + detector alarms + optional macro block + optional news summary."""
    parts = [
        f"Decision date (as_of): {context['as_of']}",
        f"Strategy telemetry:\n{json.dumps(context['telemetry'], indent=2)}",
        f"Classical detector alarms visible so far:\n"
        f"{json.dumps(context['detector_alarms'], indent=2)}",
    ]
    if context.get("macro"):
        parts.append(f"Macro/market context (as known on the decision date):\n"
                     f"{json.dumps(context['macro'], indent=2)}")
    if context.get("news"):
        parts.append(f"News Context Agent summary:\n{json.dumps(context['news'], indent=2)}")
    parts.append("Return your JSON assessment now.")
    return SUPERVISOR_SYSTEM_V2, "\n\n".join(parts)
```

- [ ] **Step 4: Implement — runner parameterisation**

In `monitoring/agentic/runner.py`:

(a) extend the prompts import: `from .prompts import build_supervisor_prompt, SUPERVISOR_PROMPT_VERSION`  (unchanged) — and change the `run_supervisor` signature and the two lines that use the builder/version:

```python
def run_supervisor(context: dict, model: LocalModel, logger=None,
                   latency_s: float | None = None,
                   extra_label: str | None = None,
                   prompt_builder=build_supervisor_prompt,
                   prompt_version: str = SUPERVISOR_PROMPT_VERSION) -> AgentAssessment:
```

inside the body replace `system, user = build_supervisor_prompt(context)` with `system, user = prompt_builder(context)` and, in the `logger.log_invocation(...)` call, replace `prompt_version=SUPERVISOR_PROMPT_VERSION` with `prompt_version=prompt_version`. Everything else is untouched (Week-2 callers keep the v1 defaults).

(b) In `monitoring/agentic/__init__.py`, additionally export `build_supervisor_prompt_v2` and `SUPERVISOR_PROMPT_VERSION_V2`.

- [ ] **Step 5: Run to verify they pass**

Run: `cd monitoring && python -m pytest tests/test_supervisor_v2.py -v` then `python -m pytest -q`
Expected: 4 PASS; full suite green (v1 defaults preserved).

- [ ] **Step 6: Commit**

```bash
git add monitoring/agentic/prompts.py monitoring/agentic/runner.py monitoring/agentic/__init__.py monitoring/tests/test_supervisor_v2.py
git commit -m "feat(agentic): supervisor-v2 prompt with macro + news blocks; runner accepts builder"
```

---

### Task 12: `run_agentic.py` — end-to-end on the Aug-2007 window

**Files:**
- Create: `monitoring/run_agentic.py`
- Test: `monitoring/tests/test_run_agentic.py`

- [ ] **Step 1: Write the failing test**

`monitoring/tests/test_run_agentic.py`:

```python
"""End-to-end agentic loop on synthetic data with the offline stub."""
import json

import numpy as np
import pandas as pd

from run_agentic import run_window
from agentic.model import OfflineStubModel
from agentic.logging_utils import RunLogger
from agentic.schemas import validate_assessment
from news.store import build_store, NewsStore
from windows import Window


def _synthetic(tmp_path):
    # 150 calm days then a 10-day crash inside the window
    rng = np.random.default_rng(7)
    dates = pd.bdate_range("2007-01-02", periods=160)
    rets = rng.normal(0.0003, 0.004, 160)
    rets[150:] = rng.normal(-0.03, 0.02, 10)
    series = pd.Series(rets, index=dates)

    raw = pd.DataFrame({
        "Date": [f"{d.date()} 10:00:00 UTC" for d in dates[145:155]],
        "Article_title": (["Ordinary market story"] * 5 +
                          ["Meltdown: quant funds hit by margin calls and liquidation"] * 5),
        "Stock_symbol": [""] * 10, "Lsa_summary": [""] * 10,
        "Publisher": ["P"] * 10, "Url": ["u"] * 10,
    })
    raw_path = tmp_path / "raw.csv"
    raw.to_csv(raw_path, index=False)
    build_store(raw_path, tmp_path / "store", start="2003-01-01", end="2016-12-31")

    window = Window(name="synthetic_event", kind="event",
                    start=str(dates[140].date()), end=str(dates[-1].date()),
                    onset=str(dates[150].date()), description="synthetic crash")
    return series, NewsStore(tmp_path / "store"), window


def test_run_window_end_to_end_stub(tmp_path):
    series, store, window = _synthetic(tmp_path)
    logger = RunLogger(tmp_path / "log.jsonl")
    records = run_window(series, window, store, OfflineStubModel(),
                         logger=logger, macro_dir=tmp_path / "no_macro")
    assert records, "expected at least one triage record"
    modes = {r["triage_mode"] for r in records}
    assert "skip" in modes                       # calm days skipped
    assert modes & {"cheap", "thinking", "classical_escalation"}  # crash escalates

    # every non-skip day produced a schema-valid supervisor assessment
    assessed = [r for r in records if r["triage_mode"] != "skip"]
    assert assessed
    for r in assessed:
        validate_assessment(r["assessment"])     # raises on invalid
        assert r["assessment"]["as_of"] <= window.end

    # the JSONL log is replayable
    lines = [json.loads(l) for l in (tmp_path / "log.jsonl").read_text().splitlines()]
    assert any(rec["agent"] == "news_context" for rec in lines)
    assert any(rec["agent"] == "performance_supervisor" for rec in lines)
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd monitoring && python -m pytest tests/test_run_agentic.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'run_agentic'`.

- [ ] **Step 3: Implement**

`monitoring/run_agentic.py`:

```python
"""End-to-end agentic monitoring on one evaluation window (VRI Week 3).

Daily loop over the window: build the day's causal context (telemetry, classical
alarms, as-of macro, filtered news), triage how much model to spend (skip /
cheap / thinking / classical-escalation), and on escalated days run the News
Context Agent then the Performance Supervisor (v2 prompt). Every step is
lookahead-guarded; every model call is logged to JSONL.

Usage (from monitoring/):
    python run_agentic.py --window quant_meltdown_2007 --strategy AL_PCA --model stub
    python run_agentic.py --window quant_meltdown_2007 --strategy AL_PCA \
        --model ollama:qwen2.5:3b
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from pnl_loader import load_pnl, returns_series
from windows import get_window
from detectors import PageHinkley, BOCPD, HMMDetector, DistributionalThreshold, aggregate_alarms
from run_classical import STRATEGY_CURVES
from agentic import as_of_context, RunLogger, run_supervisor
from agentic.guardrails import assert_no_lookahead, LookaheadError
from agentic.model import default_model
from agentic.news_agent import run_news_agent
from agentic.prompts import build_supervisor_prompt_v2, SUPERVISOR_PROMPT_VERSION_V2
from news.store import NewsStore
from news.filter import filter_news
from news.aggregate import daily_signals
from news.triage import decide

ROOT = Path(__file__).resolve().parent.parent
RESULTS = Path(__file__).resolve().parent / "results"
STORE_DIR = ROOT / "data" / "fnspid" / "store"
MACRO_DIR = ROOT / "data" / "macro"

RECENT_DAYS = 5      # detector/aggregate lookback for triage (matches the aggregation rule)
NEWS_DAYS = 5        # news shown to the agent: published in the last N days
BASELINE_PAD = 120   # calendar days of news history before the window for the z baseline


def _recent(alarms: list, day: pd.Timestamp, days: int = RECENT_DAYS) -> bool:
    lo = day - pd.Timedelta(days=days)
    return any(lo < pd.Timestamp(a) <= day for a in alarms)


def run_window(series: pd.Series, window, store: NewsStore, model,
               logger: RunLogger | None = None, macro_dir=MACRO_DIR,
               max_articles: int = 40) -> list[dict]:
    """Run the daily agentic loop over one window. Returns one record per trading day."""
    # Classical detectors run continuously over the full curve (warm-up history included).
    results = [d.detect(series) for d in
               (PageHinkley(), BOCPD(), HMMDetector(), DistributionalThreshold())]
    alarms_by_det = {r.detector: r.alarms for r in results}
    agg_alarms = aggregate_alarms(results, window_days=5, min_detectors=2)

    # News signals need trailing history for the causal z baseline.
    news_hist = store.query(window.start_ts - pd.Timedelta(days=BASELINE_PAD), window.end_ts)
    signals = daily_signals(news_hist, filter_news(news_hist))

    try:
        from macro.context import macro_context
    except ImportError:  # macro package optional at run time
        macro_context = None

    days = series.index[(series.index >= window.start_ts) & (series.index <= window.end_ts)]
    records: list[dict] = []
    for day in days:
        z = float(signals["intensity_z"].get(day.normalize(), float("nan"))) \
            if not signals.empty else float("nan")
        n_recent = sum(_recent(a, day) for a in alarms_by_det.values())
        dec = decide(z, n_recent, _recent(agg_alarms, day))

        rec = {"as_of": str(day.date()), "window": window.name,
               "triage_mode": dec.mode, "triage_reason": dec.reason,
               "intensity_z": None if pd.isna(z) else round(z, 2)}
        if logger is not None:
            logger.log({"agent": "triage", "prompt_version": "triage-v1", **rec})

        if dec.mode == "skip":
            records.append(rec)
            continue

        # --- News Context Agent ---
        day_news = store.query(day - pd.Timedelta(days=NEWS_DAYS), day)
        risky = filter_news(day_news).sort_values("n_risk_terms", ascending=False)
        articles = [{"date": str(r["date"].date()), "ticker": r["ticker"],
                     "title": r["title"], "summary": r["summary"]}
                    for _, r in risky.iterrows()]
        t0 = time.time()
        news = run_news_agent(str(day.date()), articles, model, logger=logger,
                              extra_label=f"{window.name}:{dec.mode}",
                              max_articles=max_articles)
        rec["news_latency_s"] = round(time.time() - t0, 1)

        # --- Performance Supervisor v2 ---
        ctx = as_of_context(series, day, alarms_by_det)
        if macro_context is not None and Path(macro_dir).exists():
            ctx["macro"] = macro_context(day, macro_dir)
        try:
            assert_no_lookahead({"news": news.to_dict()}, day)
            ctx["news"] = news.to_dict()
        except LookaheadError:
            # model narrative hallucinated a future date — drop the news block, fail closed
            rec["news_dropped_lookahead"] = True
        ctx["triage_mode"] = dec.mode

        t0 = time.time()
        assessment = run_supervisor(ctx, model, logger=logger,
                                    extra_label=f"{window.name}:{dec.mode}",
                                    prompt_builder=build_supervisor_prompt_v2,
                                    prompt_version=SUPERVISOR_PROMPT_VERSION_V2)
        rec["supervisor_latency_s"] = round(time.time() - t0, 1)
        rec["assessment"] = assessment.to_dict()
        records.append(rec)
    return records


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", default="quant_meltdown_2007")
    ap.add_argument("--strategy", default="AL_PCA", choices=list(STRATEGY_CURVES))
    ap.add_argument("--model", default=None, help="stub | ollama | ollama:<name> (default: MONITOR_MODEL env or stub)")
    args = ap.parse_args()

    window = get_window(args.window)
    curve = next((p for p in STRATEGY_CURVES[args.strategy]
                  if Path(p).exists()
                  and (s := returns_series(load_pnl(p))).index.min() <= window.start_ts
                  and s.index.max() >= window.end_ts - pd.Timedelta(days=7)), None)
    if curve is None:
        raise SystemExit(f"no {args.strategy} curve covers {window.name}")
    series = returns_series(load_pnl(curve))

    model = default_model(args.model)
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / f"agentic_{args.window}_{args.strategy}.jsonl"
    if out.exists():
        out.unlink()
    logger = RunLogger(out)

    records = run_window(series, window, NewsStore(STORE_DIR), model, logger=logger)

    modes = pd.Series([r["triage_mode"] for r in records]).value_counts().to_dict()
    n_assessed = sum("assessment" in r for r in records)
    lat = [r["supervisor_latency_s"] for r in records if "supervisor_latency_s" in r]
    print(f"{window.name} × {args.strategy} × {model.name}")
    print(f"  trading days: {len(records)}  triage: {modes}")
    print(f"  supervisor assessments (schema-valid): {n_assessed}")
    if lat:
        print(f"  mean supervisor latency: {sum(lat)/len(lat):.1f}s")
    print(f"  log: {out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run to verify the test passes**

Run: `cd monitoring && python -m pytest tests/test_run_agentic.py -v` then the full suite `python -m pytest -q`.
Expected: PASS; no regressions.

- [ ] **Step 5: Real run — stub first, then qwen (VRI Week-3 exit criterion)**

```bash
cd monitoring
python run_agentic.py --window quant_meltdown_2007 --strategy AL_PCA --model stub
python run_agentic.py --window quant_meltdown_2007 --strategy AL_PCA --model ollama:qwen2.5:3b
```

Expected: stub run completes in seconds; qwen run makes 2 LLM calls per escalated day (news + supervisor) at ~10–60 s each — budget up to ~1–2 h; run it in the background or overnight if needed. Success = script finishes, prints `supervisor assessments (schema-valid): N` with N ≥ 1 covering the onset region, and `results/agentic_quant_meltdown_2007_AL_PCA.jsonl` replays as valid JSON. Spot-read 2–3 assessments around 2007-08-06: does the root_cause reference the news/detectors sensibly? Note any coverage caveat from Task 6.

- [ ] **Step 6: Commit**

```bash
git add monitoring/run_agentic.py monitoring/tests/test_run_agentic.py
git commit -m "feat: end-to-end agentic run on one event window (triage + news agent + supervisor v2)"
```

---

### Task 13: Documentation + wrap-up

**Files:**
- Create: `docs/WEEK3_STATUS.md`
- Modify: `CLAUDE.md` (repo root)

- [ ] **Step 1: Write `docs/WEEK3_STATUS.md`**

Follow the structure of `docs/WEEK2_STATUS.md`: a deliverables table mapping each VRI Week-3 item to status + location, then sections filled with THIS week's actual numbers (do not leave placeholders — pull them from the runs):

- Deliverables table: FNSPID store (`monitoring/news/store.py`, row count from Task 4 Step 5), macro integration (`monitoring/macro/`, series list + vintage note), filtering pipeline (`news/filter.py` → `news/aggregate.py` → `news/triage.py`), News Context Agent v1 (`agentic/news_agent.py`), Supervisor v2 (`agentic/prompts.py`), end-to-end Aug-2007 run (`run_agentic.py`, assessment count + mean latency from Task 12 Step 5).
- **News coverage section (the gate):** paste the `results/news_coverage.csv` table; state plainly whether `quant_meltdown_2007` is adequate and, if not, the supervisor options recorded in Task 6.
- **Model section:** qwen2.5:3b via Ollama on CPU; measured latency from Task 1 Step 3 and Task 12 Step 5; note the stub remains the CI model.
- **Point-in-time guarantees:** timestamp filter + future-text scrub (fail-closed), ALFRED vintages (BLS via ALFRED rationale), `assert_no_lookahead` re-checked on the full context every day.
- **Open items for Week 4:** prompt iteration under version tags, two-pass with/without-events control, all-6-windows × both-strategies agentic run, triage thresholds documented as fixed-a-priori (no in-sample tuning), plus carryover items from WEEK2_STATUS (detector calibration protocol, 2011 window confirmation).
- Update test count (run `python -m pytest -q` and record the total).

- [ ] **Step 2: Update `CLAUDE.md`**

In the "Dev environment" section add:

```
- Agentic run (one window): `python monitoring/run_agentic.py --window quant_meltdown_2007
  --strategy AL_PCA --model ollama:qwen2.5:3b` (use `--model stub` for offline).
- FNSPID store build: `python monitoring/news/download_fnspid.py` then
  `python -c "from news.store import build_store; ..."` (see docs/WEEK3_STATUS.md).
- Macro fetch: `FRED_API_KEY=<key> python monitoring/macro/fetch_macro.py`.
```

and change the current-status line to `**Current status: Week 3 — News pipeline + agent build.**` with a pointer to `docs/WEEK3_STATUS.md`.

- [ ] **Step 3: Full verification**

```bash
cd monitoring && python -m pytest -q     # entire suite green
cd ../XSectional && python -m pytest -q  # untouched, still green
```

- [ ] **Step 4: Commit**

```bash
git add docs/WEEK3_STATUS.md CLAUDE.md
git commit -m "docs: Week 3 status — news pipeline, macro vintages, agentic e2e on Aug-2007"
```

---

## Execution notes

- **Long-running steps** (do not let them block coding): Task 3 Step 4 (22 GB download), Task 4 Step 5 (store build), Task 12 Step 5 (qwen run). Kick each off, continue with the next independent task, and return. Task order respects this: Tasks 5, 7, 8, 9, 10, 11 need no FNSPID data.
- **Dependencies:** Task 6 needs Tasks 4+5 and the real store; Task 12 Step 5 (real run) needs Tasks 1, 4, 6, 9 (fetch), 10, 11; everything else is test-driven on synthetic data and independent of downloads.
- **User-supplied inputs:** FRED API key (Task 9 Step 6); confirmation of the Task 6 gate decision if Aug-2007 coverage is thin.

## Self-review (done at plan time)

- VRI Week-3 coverage: FNSPID ingested/indexed → Tasks 3–4; macro sources → Task 9; filter→aggregator→triage → Tasks 5, 7, 8; News Context Agent v1 → Task 10; Performance Supervisor v1 (news+telemetry+classical) → Task 11; end-to-end Aug-2007 run with valid JSON → Task 12. Carryover local-LLM swap → Tasks 1–2. Guardrails (as-of + timestamp filtering) → reused + extended (scrub) in Task 10, asserted daily in Task 12.
- Known deviation to flag with the supervisor: triage modes use one model (qwen2.5:3b) with cheap-vs-thinking differing by escalation path rather than two differently-sized models — a deliberate CPU-latency tradeoff; the `TriageDecision.mode` field keeps the design point so a second model can slot in later.
