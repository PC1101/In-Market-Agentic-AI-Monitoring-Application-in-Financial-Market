"""Resilient resume loop: keeps restarting run_agentic.py until all days are done.

Handles the recurring Ollama runner death pattern on GPU-constrained hardware.
Each restart picks up from the last completed day (day-level resume).

Usage:
    cd monitoring && python scripts/resilient_resume.py --window covid_2020 --strategy JT_MOM
"""
import json
import subprocess
import sys
import time
from pathlib import Path

MONITORING = Path(__file__).resolve().parent.parent
RESULTS = MONITORING / "results"
PYTHON = sys.executable
MODEL = "ollama:qwen2.5:1.5b"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def count_done(jsonl: Path) -> tuple[int, int]:
    """Return (supervisor_days, skip_days) from the JSONL."""
    sup, skip = set(), set()
    if not jsonl.exists():
        return 0, 0
    with open(jsonl) as f:
        for line in f:
            try:
                rec = json.loads(line)
                d = rec.get("as_of", "")
                a = rec.get("agent", "")
                if a == "performance_supervisor":
                    sup.add(d)
                elif a == "triage" and rec.get("triage_mode") == "skip":
                    skip.add(d)
            except json.JSONDecodeError:
                pass
    return len(sup), len(skip)


def reset_runner() -> None:
    subprocess.run(["ollama", "stop", "qwen2.5:1.5b"], capture_output=True)
    time.sleep(5)


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", required=True)
    ap.add_argument("--strategy", required=True)
    ap.add_argument("--expected-days", type=int, required=True)
    ap.add_argument("--max-attempts", type=int, default=50)
    ap.add_argument("--max-articles", type=int, default=40,
                    help="max news articles per day (lower = faster inference)")
    args = ap.parse_args()

    jsonl = RESULTS / f"agentic_{args.window}_{args.strategy}.jsonl"

    for attempt in range(1, args.max_attempts + 1):
        sup, skip = count_done(jsonl)
        total_done = sup + skip
        log(f"Attempt {attempt}: {total_done}/{args.expected_days} done "
            f"({sup} assessed, {skip} skipped)")
        if total_done >= args.expected_days:
            log("All days complete!")
            return

        reset_runner()
        result = subprocess.run(
            [PYTHON, "run_agentic.py", "--window", args.window,
             "--strategy", args.strategy, "--model", MODEL,
             "--max-articles", str(args.max_articles)],
            cwd=str(MONITORING),
        )
        log(f"run_agentic.py exited with code {result.returncode}")

    sup, skip = count_done(jsonl)
    log(f"Max attempts reached. Final: {sup + skip}/{args.expected_days} done")


if __name__ == "__main__":
    main()
