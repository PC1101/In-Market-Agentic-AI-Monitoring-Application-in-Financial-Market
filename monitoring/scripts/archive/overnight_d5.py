"""
Overnight D5 driver (2026-07-26).

Sequence:
  1. Wait for the currently-running run_test_windows.py to finish (polls .run_lock;
     watchdog: if the lock exists but no agentic JSONL has been touched for 30 min,
     assume the orchestrator crashed, clear the lock and resume).
  2. Re-invoke run_test_windows.py once (idempotent) to verify/resume all 8 cells.
  3. Archive the DEV leakage result (leakage_analysis_AL_PCA.json contains
     quant_meltdown_2007) so the test-set leakage run does not overwrite it.
  4. Run the SS8 leakage harness on the test event windows:
       run_leakage.py --strategy AL_PCA  (flash_crash_2010; china_deval skipped, no curve)
       run_leakage.py --strategy JT_MOM  (flash_crash, china_deval, volmageddon, covid)
     with an Ollama runner health-check + reset between phases.

Usage:
    cd monitoring && python scripts/overnight_d5.py
"""
import json
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

MONITORING = Path(__file__).resolve().parent.parent
RESULTS = MONITORING / "results"
PYTHON = sys.executable
LOCK = RESULTS / ".run_lock"
MODEL = "ollama:qwen2.5:1.5b"
OLLAMA_MODEL = "qwen2.5:1.5b"

WAIT_TIMEOUT_S = 7 * 3600      # give phase 1 up to 7 hours
STAGNATION_S = 30 * 60         # watchdog: no JSONL writes for 30 min => crashed


def log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def runner_healthy() -> bool:
    payload = {"model": OLLAMA_MODEL, "prompt": 'Return a JSON object {"ok": true}',
               "format": "json", "stream": False, "options": {"temperature": 0.0}}
    req = urllib.request.Request(
        "http://localhost:11434/api/generate",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            body = json.loads(resp.read().decode())
        return bool(body.get("done")) and bool(body.get("response"))
    except Exception:
        return False


def reset_runner() -> None:
    subprocess.run(["ollama", "stop", OLLAMA_MODEL], capture_output=True)
    time.sleep(5)


def ensure_healthy(label: str) -> bool:
    if runner_healthy():
        log(f"{label}: runner healthy")
        return True
    log(f"{label}: runner unhealthy - resetting")
    reset_runner()
    ok = runner_healthy()
    log(f"{label}: after reset -> {'healthy' if ok else 'STILL DEAD'}")
    return ok


def newest_jsonl_age_s() -> float:
    files = list(RESULTS.glob("agentic_*.jsonl"))
    if not files:
        return float("inf")
    newest = max(f.stat().st_mtime for f in files)
    return time.time() - newest


def run_step(cmd: list[str], label: str) -> int:
    log(f"START {label}: {' '.join(cmd)}")
    t0 = time.time()
    result = subprocess.run(cmd, cwd=str(MONITORING))
    dt = (time.time() - t0) / 60
    log(f"END {label}: exit={result.returncode} ({dt:.1f} min)")
    return result.returncode


def main() -> None:
    log("Overnight D5 driver started")

    # ---- Phase 1: wait for the in-flight agentic run ----
    t0 = time.time()
    while LOCK.exists():
        if time.time() - t0 > WAIT_TIMEOUT_S:
            log("TIMEOUT waiting for run_test_windows.py - stopping driver")
            return
        if newest_jsonl_age_s() > STAGNATION_S:
            log("WATCHDOG: lock present but no JSONL writes for 30 min - "
                "assuming crashed orchestrator, clearing locks")
            LOCK.unlink(missing_ok=True)
            for lk in RESULTS.glob(".lock_*"):
                lk.unlink(missing_ok=True)
            break
        time.sleep(120)
    log("Phase 1 wait complete (no .run_lock)")

    # ---- Phase 2: verify/resume all 8 cells ----
    ensure_healthy("pre-verify")
    run_step([PYTHON, "scripts/run_test_windows.py"], "verify/resume agentic cells")

    # ---- Phase 3: archive the dev leakage artifact ----
    dev_leak = RESULTS / "leakage_analysis_AL_PCA.json"
    if dev_leak.exists():
        try:
            content = json.loads(dev_leak.read_text())
            if "quant_meltdown_2007" in content.get("windows", {}):
                dest_dir = RESULTS / "archive" / "leakage_dev"
                dest_dir.mkdir(parents=True, exist_ok=True)
                shutil.move(str(dev_leak), str(dest_dir / "leakage_analysis_AL_PCA.json"))
                jsonl = RESULTS / "leakage_quant_meltdown_2007_AL_PCA.jsonl"
                if jsonl.exists():
                    shutil.move(str(jsonl), str(dest_dir / jsonl.name))
                log("Archived dev leakage artifacts to results/archive/leakage_dev/")
        except Exception as e:
            log(f"WARNING: could not inspect/archive dev leakage file: {e}")

    # ---- Phase 4: leakage harness on test events ----
    for strat in ("AL_PCA", "JT_MOM"):
        if not ensure_healthy(f"pre-leakage {strat}"):
            log(f"ABORT leakage {strat}: runner unrecoverable")
            continue
        run_step(
            [PYTHON, "run_leakage.py", "--strategy", strat, "--model", MODEL],
            f"leakage {strat}",
        )
        if not runner_healthy():
            log(f"WARNING: runner dead after leakage {strat} - results may have "
                f"a trailing failure block; resetting for next phase")
            reset_runner()

    log("Overnight D5 driver finished")
    log("Morning next steps: scoring pass (analyze_test_results) -> test_analysis.json")


if __name__ == "__main__":
    main()
