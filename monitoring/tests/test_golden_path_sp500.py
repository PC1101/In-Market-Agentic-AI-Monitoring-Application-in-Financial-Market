"""Golden-path: the provider refactor must not change US S&P 500 output."""
import json
import subprocess
import sys
from pathlib import Path

MON = Path(__file__).resolve().parents[1]
GOLDEN = MON / "tests" / "golden" / "sp500_classical_stub.json"


def test_classical_sp500_bit_identical():
    out = subprocess.run(
        [sys.executable, "run_classical.py", "--market", "sp500", "--model", "stub"],
        cwd=MON, capture_output=True, text=True,
    )
    assert out.returncode == 0, out.stderr
    produced = json.loads((MON / "results" / "classical_summary.json").read_text())
    expected = json.loads(GOLDEN.read_text())
    assert produced == expected


def test_unknown_market_fails_closed():
    out = subprocess.run(
        [sys.executable, "run_classical.py", "--market", "asx200", "--model", "stub"],
        cwd=MON, capture_output=True, text=True,
    )
    assert out.returncode != 0
    assert "asx200" in (out.stderr + out.stdout)
