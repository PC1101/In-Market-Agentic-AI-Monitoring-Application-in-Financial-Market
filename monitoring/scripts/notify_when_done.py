"""Poll result files; beep + popup when all 8 event windows and dev analysis are done.

Read-only: never touches Ollama or the result-writing processes.
Usage: python scripts/notify_when_done.py
"""
import json
import subprocess
import time
from pathlib import Path

RESULTS = Path(__file__).resolve().parent.parent / "results"

EXPECTED = {
    "quant_meltdown_2007": 44,
    "gfc_lehman_2008": 74,
    "momentum_crash_2009": 72,
    "downgrade_2011": 64,
}
STRATEGIES = ["AL_PCA", "JT_MOM"]


def supervisor_days(fname: str) -> int:
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


def status() -> tuple[bool, str]:
    parts, all_done = [], True
    for window, exp in EXPECTED.items():
        for strat in STRATEGIES:
            n = supervisor_days(f"agentic_{window}_{strat}.jsonl")
            done = n >= int(exp * 0.97)
            all_done &= done
            parts.append(f"{window[:12]}_{strat[:2]}: {n}/{exp}{'' if done else ' *'}")
    return all_done, " | ".join(parts)


def notify(msg: str) -> None:
    for _ in range(3):
        subprocess.run(["powershell", "-c", "[console]::beep(1000,400)"])
        time.sleep(0.2)
    subprocess.run([
        "powershell", "-c",
        f"(New-Object -ComObject Wscript.Shell).Popup('{msg}', 0, 'Agentic Monitoring', 64)"
    ])


def main():
    analysis = RESULTS / "dev_analysis.json"
    start_mtime = analysis.stat().st_mtime if analysis.exists() else 0

    while True:
        all_done, line = status()
        print(time.strftime("%H:%M:%S"), line, flush=True)
        if all_done:
            break
        time.sleep(120)

    # Windows done — give analyze_dev_results.py up to 5 min to write fresh output.
    analysis_fresh = False
    for _ in range(10):
        if analysis.exists() and analysis.stat().st_mtime > start_mtime:
            analysis_fresh = True
            break
        time.sleep(30)

    msg = "All 8 event windows complete."
    msg += " dev_analysis.json regenerated." if analysis_fresh else " (analysis output not yet refreshed — check manually)"
    print(msg, flush=True)
    notify(msg)


if __name__ == "__main__":
    main()
