"""
Run the 8 confirmatory TEST-set agentic cells with the real model (D5 step 3).

Test roster (TEST_STRATEGY_WINDOWS ∩ curve coverage):
  AL_PCA: flash_crash_2010, calm_2012          (china_deval_2015 excluded — Deviation 3)
  JT_MOM: flash_crash_2010, china_deval_2015, volmageddon_2018, covid_2020,
          calm_2012, calm_2017

Hard rules (preregistration §11 / BATCH_D Phase D5):
  - No prompt edits, no threshold changes, no second runs.
  - Runtime failures: resume via this script (cell-level), never restart manually.
  - One Ollama consumer at a time.

Usage:
    cd monitoring && python scripts/run_test_windows.py
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pnl_loader import load_pnl, returns_series
from run_classical import STRATEGY_CURVES
from windows import get_window

RESULTS = Path(__file__).resolve().parent.parent / "results"
MONITORING = Path(__file__).resolve().parent.parent
PYTHON = sys.executable

MODEL = "ollama:qwen2.5:1.5b"
OLLAMA_MODEL = MODEL.split(":", 1)[1]  # "qwen2.5:1.5b"


def runner_healthy() -> bool:
    """Trivial generate call. The Ollama runner can die mid-session (observed
    2026-07-25: HTTP 200 zero-value structs, done=false, empty response — every
    subsequent call fails silently). Detect that state before/after each cell."""
    import urllib.request
    payload = {"model": OLLAMA_MODEL, "prompt": 'Return a JSON object {"ok": true}',
               "format": "json", "stream": False, "options": {"temperature": 0.0}}
    req = urllib.request.Request(
        "http://localhost:11434/api/generate",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            body = json.loads(resp.read().decode())
        return bool(body.get("done")) and bool(body.get("response"))
    except Exception:
        return False


def reset_runner() -> None:
    """Unload the model to force a fresh runner subprocess (recovers the
    zero-struct dead-runner state without restarting the Ollama server)."""
    import time
    subprocess.run(["ollama", "stop", OLLAMA_MODEL], capture_output=True)
    time.sleep(3)


def ensure_runner_healthy() -> bool:
    """Health-check with one reset attempt. Returns False if unrecoverable."""
    if runner_healthy():
        return True
    print("  Runner unhealthy — resetting (ollama stop)...")
    reset_runner()
    if runner_healthy():
        print("  Runner recovered.")
        return True
    print("  Runner STILL unhealthy after reset — check GPU / Eco Mode / Ollama server.")
    return False


def count_supervisor_days(fname: str) -> int:
    p = RESULTS / fname
    if not p.exists():
        return 0
    try:
        lines = [ln for ln in p.read_text(errors="replace").splitlines() if ln.strip()]
        return len(set(
            json.loads(ln).get("as_of") for ln in lines
            if json.loads(ln).get("agent") == "performance_supervisor"
        ) - {None})
    except Exception:
        return 0


def count_nonskip_days(fname: str) -> int:
    p = RESULTS / fname
    if not p.exists():
        return 0
    try:
        lines = [ln for ln in p.read_text(errors="replace").splitlines() if ln.strip()]
        return len(set(
            json.loads(ln).get("as_of") for ln in lines
            if json.loads(ln).get("agent") == "triage"
            and json.loads(ln).get("triage_mode") != "skip"
        ) - {None})
    except Exception:
        return 0

# Events first (co-primary latency evidence), then calm windows (FPR evidence).
TEST_CELLS = [
    ("flash_crash_2010", "AL_PCA"),
    ("flash_crash_2010", "JT_MOM"),
    ("china_deval_2015", "JT_MOM"),
    ("volmageddon_2018", "JT_MOM"),
    ("covid_2020", "JT_MOM"),
    ("calm_2012", "AL_PCA"),
    ("calm_2012", "JT_MOM"),
    ("calm_2017", "JT_MOM"),
]

LOCK = RESULTS / ".run_lock"


def compute_expected_days(window_name: str, strategy: str) -> int:
    """Count trading days in the window where the strategy curve has data."""
    w = get_window(window_name)
    for curve_path in STRATEGY_CURVES.get(strategy, []):
        p = Path(curve_path)
        if not p.exists():
            continue
        try:
            series = returns_series(load_pnl(p))
            mask = (series.index >= pd.Timestamp(w.start)) & (series.index <= pd.Timestamp(w.end))
            n = int(mask.sum())
            if n > 10:
                return n
        except Exception:
            continue
    return int(np.busday_count(str(w.start), str(w.end))) + 1


def count_processed_days(fname: str) -> int:
    p = RESULTS / fname
    if not p.exists():
        return 0
    try:
        lines = [ln for ln in p.read_text(errors="replace").splitlines() if ln.strip()]
        return len(set(json.loads(ln).get("as_of") for ln in lines) - {None})
    except Exception:
        return 0


def print_cell_summary(fname: str) -> None:
    p = RESULTS / fname
    if not p.exists():
        return
    try:
        lines = [ln for ln in p.read_text(errors="replace").splitlines() if ln.strip()]
        records = [json.loads(ln) for ln in lines]
    except Exception:
        return

    triage_modes: dict[str, int] = {}
    for r in records:
        if r.get("agent") == "triage":
            mode = r.get("triage_mode", "unknown")
            triage_modes[mode] = triage_modes.get(mode, 0) + 1
    total_triage = sum(triage_modes.values())
    skip_rate = triage_modes.get("skip", 0) / total_triage if total_triage else 0.0

    latencies = [
        r["latency_s"] for r in records
        if r.get("agent") == "performance_supervisor" and r.get("latency_s") is not None
    ]

    print(f"  Triage breakdown: {dict(sorted(triage_modes.items()))}")
    print(f"  Skip rate: {skip_rate:.1%}  (n_triage={total_triage})")
    if latencies:
        print(f"  Mean supervisor latency: {sum(latencies) / len(latencies):.1f}s  "
              f"(n_supervisor={len(latencies)})")
    else:
        print("  No supervisor calls recorded (all days skipped or run_meta only)")


def run_test_cell(window_name: str, strategy: str) -> None:
    fname = f"agentic_{window_name}_{strategy}.jsonl"
    expected = compute_expected_days(window_name, strategy)
    win_lock = RESULTS / f".lock_{window_name}_{strategy}"

    print(f"\n[{window_name} × {strategy}]  expected trading days: {expected}")

    # Guard: never overwrite a file lacking a run_meta header (unknown provenance).
    existing = RESULTS / fname
    if existing.exists():
        try:
            first = existing.read_text(errors="replace").splitlines()[0]
            rec = json.loads(first)
            if rec.get("agent") != "run_meta":
                print(f"  WARNING: {fname} has no run_meta header — unknown provenance.")
                print(f"  Move it to results/archive/ before running this cell. SKIPPING.")
                return
        except Exception:
            pass

    if win_lock.exists():
        print("  SKIP — lock exists (another process is running this cell)")
        return

    processed = count_processed_days(fname)
    # A cell only counts as complete if it also has supervisor evidence when
    # non-skip days exist (guards against the dead-runner state where triage
    # coverage is 100% but every LLM call silently failed).
    if processed >= int(expected * 0.97):
        if count_nonskip_days(fname) == 0 or count_supervisor_days(fname) > 0:
            print(f"  SKIP — already complete ({processed}/{expected} days)")
            print_cell_summary(fname)
            return
        print(f"  {fname} has {processed}/{expected} days but 0 supervisor records "
              f"on non-skip days — dead-runner artifact, rerunning.")

    win_lock.write_text(str(os.getpid()))
    try:
        for attempt in (1, 2):
            if not ensure_runner_healthy():
                print("  ABORT cell — runner unrecoverable.")
                return
            print(f"  Launching run_agentic.py (model={MODEL}, attempt {attempt})...")
            cmd = [
                PYTHON, "run_agentic.py",
                "--window", window_name,
                "--strategy", strategy,
                "--model", MODEL,
            ]
            env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
            result = subprocess.run(
                cmd, cwd=str(MONITORING),
                capture_output=True, text=True, encoding="utf-8", env=env,
            )
            print(result.stdout[-3000:] if result.stdout else "(no stdout)")
            if result.returncode != 0:
                print(f"  ERROR (exit {result.returncode}): {result.stderr[-1000:]}")

            # If the runner is dead at cell end, the cell's tail is suspect
            # (observed failure mode: mid-cell runner death → contiguous
            # trailing failure block). Reset and rerun the cell once.
            if not runner_healthy():
                print("  Runner DEAD at cell end — cell tail is suspect.")
                reset_runner()
                if attempt == 1:
                    print("  Rerunning cell once with a fresh runner...")
                    continue
                print("  Second attempt also ended with a dead runner — leaving "
                      "partial cell for manual review.")
            break

        n = count_processed_days(fname)
        print(f"  Done: {n}/{expected} trading days processed "
              f"({count_supervisor_days(fname)} supervisor days)")
        print_cell_summary(fname)
    finally:
        win_lock.unlink(missing_ok=True)


def main() -> None:
    if LOCK.exists():
        print(f"Lock file exists ({LOCK}) — another instance is running. Exiting.")
        return
    LOCK.write_text(str(os.getpid()))
    try:
        _main()
    finally:
        LOCK.unlink(missing_ok=True)


def _main() -> None:
    print("=" * 72)
    print("TEST-SET REAL-MODEL RUNS (D5 — confirmatory, one shot)")
    print("=" * 72)
    print("Hard rules: no prompt/threshold edits, no second runs, resume-only.\n")

    for window_name, strategy in TEST_CELLS:
        run_test_cell(window_name, strategy)

    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    all_complete = True
    for window_name, strategy in TEST_CELLS:
        fname = f"agentic_{window_name}_{strategy}.jsonl"
        expected = compute_expected_days(window_name, strategy)
        n = count_processed_days(fname)
        n_sup = count_supervisor_days(fname)
        n_nonskip = count_nonskip_days(fname)
        complete = n >= int(expected * 0.97) and (n_nonskip == 0 or n_sup > 0)
        status = "OK" if complete else "INCOMPLETE"
        print(f"  [{status}] {window_name} × {strategy}: {n}/{expected} days, "
              f"{n_sup}/{n_nonskip} supervisor/non-skip")
        if not complete:
            all_complete = False

    print()
    if all_complete:
        print("All 8 test cells complete.")
        print("Next: leakage harness on test events, then the single scoring pass.")
    else:
        print("Some cells incomplete. Re-run this script to resume (safe — per-window locks).")
        print("Do NOT delete partial JSONLs.")


if __name__ == "__main__":
    main()
