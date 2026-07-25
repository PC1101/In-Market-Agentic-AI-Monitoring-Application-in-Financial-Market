"""
Run the 4 calm-window agentic cells with the real model (H1 FPR endpoint).

Prerequisites:
  - Stub calm JSONLs must already be in results/archive/stub/ (done by Batch B).
    If any agentic_calm_*.jsonl still exists in results/, move them manually.
  - Ollama serving qwen2.5:1.5b. Check ~4s/call (GPU), not ~40s (CPU/Eco Mode).
  - One Ollama consumer at a time — do not run alongside other agentic scripts.

Usage:
    cd monitoring && python scripts/run_calm_windows.py
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Add monitoring/ to path so we can import pnl_loader, run_classical, windows
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pnl_loader import load_pnl, returns_series
from run_classical import STRATEGY_CURVES
from windows import get_window

RESULTS = Path(__file__).resolve().parent.parent / "results"
MONITORING = Path(__file__).resolve().parent.parent
PYTHON = sys.executable

CALM_CELLS = [
    ("calm_2004_2006", "AL_PCA"),
    ("calm_2004_2006", "JT_MOM"),
    ("calm_2013_2014", "AL_PCA"),
    ("calm_2013_2014", "JT_MOM"),
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
    # Fallback: raw business-day count
    return int(np.busday_count(str(w.start), str(w.end))) + 1


def count_processed_days(fname: str) -> int:
    """Count unique trading days with any JSONL record (triage-skip days included)."""
    p = RESULTS / fname
    if not p.exists():
        return 0
    try:
        lines = [ln for ln in p.read_text(errors="replace").splitlines() if ln.strip()]
        return len(set(
            json.loads(ln).get("as_of")
            for ln in lines
        ) - {None})
    except Exception:
        return 0


def print_cell_summary(fname: str) -> None:
    """Print triage-mode breakdown and mean supervisor latency for a completed cell."""
    p = RESULTS / fname
    if not p.exists():
        return
    try:
        lines = [ln for ln in p.read_text(errors="replace").splitlines() if ln.strip()]
        records = [json.loads(ln) for ln in lines]
    except Exception:
        return

    # Triage mode distribution
    triage_modes: dict[str, int] = {}
    for r in records:
        if r.get("agent") == "triage":
            mode = r.get("triage_mode", "unknown")
            triage_modes[mode] = triage_modes.get(mode, 0) + 1
    total_triage = sum(triage_modes.values())
    skip_rate = triage_modes.get("skip", 0) / total_triage if total_triage else 0.0

    # Supervisor latency
    latencies = [
        r["latency_s"] for r in records
        if r.get("agent") == "performance_supervisor" and r.get("latency_s") is not None
    ]

    print(f"  Triage breakdown: {dict(sorted(triage_modes.items()))}")
    print(f"  Skip rate: {skip_rate:.1%}  (n_triage={total_triage})")
    if latencies:
        mean_lat = sum(latencies) / len(latencies)
        print(f"  Mean supervisor latency: {mean_lat:.1f}s  (n_supervisor={len(latencies)})")
    else:
        print("  No supervisor calls recorded (all days skipped or run_meta only)")


def run_calm_cell(window_name: str, strategy: str) -> None:
    fname = f"agentic_{window_name}_{strategy}.jsonl"
    expected = compute_expected_days(window_name, strategy)
    win_lock = RESULTS / f".lock_{window_name}_{strategy}"

    print(f"\n[{window_name} × {strategy}]  expected trading days: {expected}")

    # Guard: block if file in results/ has no run_meta (stub/pre-Batch-A leftover).
    # A partial real-model file is fine — run_agentic.py deletes it on start.
    existing = RESULTS / fname
    if existing.exists():
        try:
            first = existing.read_text(errors="replace").splitlines()[0]
            rec = json.loads(first)
            if rec.get("agent") != "run_meta":
                print(f"  WARNING: {fname} has no run_meta header — looks like a stub/pre-Batch-A file.")
                print(f"  Move it to results/archive/stub/ before running this cell.")
                print(f"  SKIPPING to avoid overwriting.")
                return
        except Exception:
            pass  # unreadable first line — let run_agentic.py handle it

    # Per-window lock
    if win_lock.exists():
        print(f"  SKIP — lock exists (another process is running this cell)")
        return

    processed = count_processed_days(fname)
    if processed >= int(expected * 0.97):
        print(f"  SKIP — already complete ({processed}/{expected} days)")
        print_cell_summary(fname)
        return

    win_lock.write_text(str(os.getpid()))
    try:
        print(f"  Launching run_agentic.py (model=ollama:qwen2.5:1.5b)...")
        cmd = [
            PYTHON, "run_agentic.py",
            "--window", window_name,
            "--strategy", strategy,
            "--model", "ollama:qwen2.5:1.5b",
        ]
        env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
        result = subprocess.run(
            cmd, cwd=str(MONITORING),
            capture_output=True, text=True, encoding="utf-8", env=env,
        )
        print(result.stdout[-3000:] if result.stdout else "(no stdout)")
        if result.returncode != 0:
            print(f"  ERROR (exit {result.returncode}): {result.stderr[-1000:]}")
        else:
            n = count_processed_days(fname)
            print(f"  Done: {n}/{expected} trading days processed")
            print_cell_summary(fname)
    finally:
        win_lock.unlink(missing_ok=True)


def main() -> None:
    # Singleton lock
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
    print("CALM-WINDOW REAL-MODEL RUNS (H1 FPR endpoint)")
    print("=" * 72)
    print("Hazard check: each supervisor call should take ~4s (GPU) not ~40s (CPU).")
    print("If you observe ~40s calls, abort and disable laptop Eco Mode.\n")

    for window_name, strategy in CALM_CELLS:
        run_calm_cell(window_name, strategy)

    # Final summary
    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    all_complete = True
    for window_name, strategy in CALM_CELLS:
        fname = f"agentic_{window_name}_{strategy}.jsonl"
        expected = compute_expected_days(window_name, strategy)
        n = count_processed_days(fname)
        complete = n >= int(expected * 0.97)
        status = "OK" if complete else "INCOMPLETE"
        print(f"  [{status}] {window_name} × {strategy}: {n}/{expected} days")
        if not complete:
            all_complete = False

    print()
    if all_complete:
        print("All 4 calm cells complete.")
        print("Next: verify run_meta.model in each JSONL, then run:")
        print("  python scripts/analyze_dev_results.py")
    else:
        print("Some cells incomplete. Re-run this script to resume (safe — uses per-window locks).")
        print("Do NOT delete partial JSONLs.")


if __name__ == "__main__":
    main()
