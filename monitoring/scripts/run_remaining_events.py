"""
Run all incomplete event windows sequentially (no polling needed — run from scratch).
Usage: python scripts/run_remaining_events.py
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

RESULTS = Path(__file__).resolve().parent.parent / "results"
MONITORING = Path(__file__).resolve().parent.parent
PYTHON = sys.executable

EXPECTED_DAYS = {
    "quant_meltdown_2007": 44,
    "gfc_lehman_2008": 74,
    "momentum_crash_2009": 72,
    "downgrade_2011": 64,
}


def count_supervisor_days(fname: str) -> int:
    p = RESULTS / fname
    if not p.exists():
        return 0
    try:
        lines = [l for l in p.read_text(errors="replace").splitlines() if l.strip()]
        return len(set(
            json.loads(l).get("as_of")
            for l in lines
            if '"performance_supervisor"' in l
        ) - {None})
    except Exception:
        return 0


def wait_for(fname: str, expected: int, poll_s: int = 30) -> None:
    print(f"Waiting for {fname} to reach {expected} supervisor days...")
    while True:
        n = count_supervisor_days(fname)
        print(f"  {fname}: {n}/{expected} supervisor days", flush=True)
        if n >= int(expected * 0.97):  # allow 1-2 missing (failures)
            print(f"  -> Complete ({n}/{expected})")
            break
        time.sleep(poll_s)


def run_window(window: str, strategy: str) -> None:
    fname = f"agentic_{window}_{strategy}.jsonl"
    expected = EXPECTED_DAYS[window]
    win_lock = RESULTS / f".lock_{window}_{strategy}"

    # Per-window lock: skip if another process is already running this window
    if win_lock.exists():
        print(f"SKIP {window} x {strategy} — lock exists (another process running it)")
        return
    existing = count_supervisor_days(fname)
    if existing >= int(expected * 0.97):
        print(f"SKIP {window} x {strategy} (already {existing}/{expected} days)")
        return

    win_lock.write_text(str(os.getpid()))
    try:
        print(f"\nRunning {window} x {strategy}...")
        cmd = [PYTHON, "run_agentic.py",
               "--window", window,
               "--strategy", strategy,
               "--model", "ollama:qwen2.5:1.5b"]
        env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
        result = subprocess.run(cmd, cwd=str(MONITORING), capture_output=True,
                                text=True, encoding="utf-8", env=env)
        print(result.stdout[-2000:] if result.stdout else "(no stdout)")
        if result.returncode != 0:
            print(f"ERROR: {result.stderr[-500:]}")
        else:
            n = count_supervisor_days(fname)
            print(f"Done: {window} x {strategy} ({n}/{expected} days)")
    finally:
        win_lock.unlink(missing_ok=True)


LOCK = MONITORING / "results" / ".run_lock"


def main():
    # Singleton lock — exit immediately if another instance is running
    if LOCK.exists():
        print(f"Lock file exists ({LOCK}) — another instance is running. Exiting.")
        return
    LOCK.write_text(str(os.getpid()))
    try:
        _main()
    finally:
        LOCK.unlink(missing_ok=True)


def _main():
    # Run all 8 event windows sequentially (skips any already complete)
    windows = [
        ("quant_meltdown_2007", "AL_PCA"),
        ("quant_meltdown_2007", "JT_MOM"),
        ("gfc_lehman_2008", "AL_PCA"),
        ("gfc_lehman_2008", "JT_MOM"),
        ("momentum_crash_2009", "AL_PCA"),
        ("momentum_crash_2009", "JT_MOM"),
        ("downgrade_2011", "AL_PCA"),
        ("downgrade_2011", "JT_MOM"),
    ]
    for window, strategy in windows:
        run_window(window, strategy)

    print("\n=== All event windows complete. Running dev analysis... ===")
    env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
    result = subprocess.run(
        [PYTHON, "scripts/analyze_dev_results.py"],
        cwd=str(MONITORING), capture_output=True, text=True, encoding="utf-8", env=env
    )
    print(result.stdout)
    if result.returncode != 0:
        print(f"Analysis error: {result.stderr[-500:]}")


if __name__ == "__main__":
    main()
