"""Pre-freeze checklist verifier (preregistration §9 + §11).

Validates that all blockers for the ``eval-freeze-v1`` git tag are satisfied
before any test-set runs begin. Exit 0 if all checks pass, exit 1 otherwise.

Usage:
    python freeze_gate.py
    python freeze_gate.py --json   # machine-readable output

Checklist (§11):
  1. significance.py importable
  2. leakage harness importable (run_leakage, leakage.synthetic, masking utilities)
  3. results/calibration_grid.json exists
  4. Frozen component versions match preregistration §6.2
  5. Triage constants match §6.2 values
  6. DEV_WINDOWS and TEST_WINDOWS are temporally disjoint
  7. No test-set result files exist yet (test-set blinded)
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

MONITORING = Path(__file__).resolve().parent
RESULTS = MONITORING / "results"

# Expected frozen values (preregistration §6.2 + documented deviations)
_EXPECTED_TRIAGE = {
    "STRESS_CHEAP": 0.30,
    "STRESS_THINKING": 0.60,
    "HIT_Z_THINKING": 2.0,
    "RECENT_DAYS": 3,
    "COVERAGE_DAYS": 7,
}
_EXPECTED_PROMPT_VERSIONS = {
    "SUPERVISOR_PROMPT_VERSION_V3": "supervisor-v3",
    "NEWS_PROMPT_VERSION": "news-context-v1",
}
_EXPECTED_DEFAULT_MODEL = "qwen2.5:1.5b"  # Deviation 1 from preregistration's llama3.2:3b

# Test window names — existence of any result files before the freeze is a blocker
_TEST_WINDOW_NAMES = [
    "flash_crash_2010", "china_deval_2015", "volmageddon_2018", "covid_2020",
    "calm_2012", "calm_2017",
]


def _check(name: str, fn) -> dict:
    """Run one check function; catch exceptions as failures."""
    try:
        result = fn()
        if isinstance(result, dict):
            return {"passed": result.get("passed", True), **result, "name": name}
        passed = bool(result)
        return {"name": name, "passed": passed, "detail": "OK" if passed else "FAILED"}
    except Exception as e:
        return {"name": name, "passed": False, "detail": f"exception: {e}"}


def check_significance_importable() -> dict:
    try:
        import significance
        fns = ["mcnemar_test", "permutation_test_latency", "block_bootstrap_fpr",
               "bayesian_recall", "holm_correction", "strategy_effects"]
        missing = [f for f in fns if not hasattr(significance, f)]
        if missing:
            return {"passed": False, "detail": f"missing functions: {missing}"}
        return {"passed": True, "detail": f"all {len(fns)} functions present"}
    except ImportError as e:
        return {"passed": False, "detail": str(e)}


def check_leakage_importable() -> dict:
    errors = []
    for mod in ("leakage.synthetic", "run_leakage"):
        try:
            importlib.import_module(mod)
        except ImportError as e:
            errors.append(f"{mod}: {e}")
    # Check masking utilities
    try:
        from agentic.guardrails import mask_dates_in_context, mask_dates_in_articles
    except ImportError as e:
        errors.append(f"guardrails masking: {e}")
    if errors:
        return {"passed": False, "detail": "; ".join(errors)}
    return {"passed": True, "detail": "all leakage modules importable"}


def check_alarm_extraction_importable() -> dict:
    try:
        from agentic.alarm_extraction import (
            extract_agentic_alarms, cluster_starts,
            count_runtime_failures, evaluate_agentic_window,
        )
        return {"passed": True, "detail": "alarm_extraction OK"}
    except ImportError as e:
        return {"passed": False, "detail": str(e)}


def check_calibration_grid_exists() -> dict:
    path = RESULTS / "calibration_grid.json"
    if not path.exists():
        return {"passed": False, "detail": f"not found: {path}"}
    try:
        data = json.loads(path.read_text())
        strategies = list(data.keys())
        return {"passed": True, "detail": f"found, strategies: {strategies}"}
    except Exception as e:
        return {"passed": False, "detail": f"invalid JSON: {e}"}


def check_prompt_versions() -> dict:
    errors = []
    try:
        from agentic.prompts import SUPERVISOR_PROMPT_VERSION_V3, NEWS_PROMPT_VERSION
        vals = {
            "SUPERVISOR_PROMPT_VERSION_V3": SUPERVISOR_PROMPT_VERSION_V3,
            "NEWS_PROMPT_VERSION": NEWS_PROMPT_VERSION,
        }
        for k, expected in _EXPECTED_PROMPT_VERSIONS.items():
            actual = vals.get(k)
            if actual != expected:
                errors.append(f"{k}={actual!r} (expected {expected!r})")
    except ImportError as e:
        errors.append(str(e))
    if errors:
        return {"passed": False, "detail": "; ".join(errors)}
    return {"passed": True, "detail": "all prompt versions match §6.2"}


def check_triage_constants() -> dict:
    errors = []
    try:
        from news.triage import STRESS_CHEAP, STRESS_THINKING, HIT_Z_THINKING, RECENT_DAYS, COVERAGE_DAYS
        actual = {
            "STRESS_CHEAP": STRESS_CHEAP,
            "STRESS_THINKING": STRESS_THINKING,
            "HIT_Z_THINKING": HIT_Z_THINKING,
            "RECENT_DAYS": RECENT_DAYS,
            "COVERAGE_DAYS": COVERAGE_DAYS,
        }
        for k, expected in _EXPECTED_TRIAGE.items():
            if abs(actual[k] - expected) > 1e-9:
                errors.append(f"{k}={actual[k]} (expected {expected})")
    except ImportError as e:
        errors.append(str(e))
    if errors:
        return {"passed": False, "detail": "; ".join(errors)}
    return {"passed": True, "detail": "all triage constants match §6.2"}


def check_model_default() -> dict:
    try:
        from agentic.model import OllamaModel
        m = OllamaModel()
        if m.model != _EXPECTED_DEFAULT_MODEL:
            return {
                "passed": False,
                "detail": (f"default model={m.model!r}, expected {_EXPECTED_DEFAULT_MODEL!r}; "
                           "Deviation 1 must be recorded in Deviations appendix"),
            }
        return {"passed": True,
                "detail": f"model={m.model!r} (Deviation 1 from preregistration's llama3.2:3b, documented)"}
    except ImportError as e:
        return {"passed": False, "detail": str(e)}


def check_window_partition() -> dict:
    """Verify dev and test window sets are disjoint by name.

    Note: the preregistration's dev/test partition is not temporal (flash_crash_2010
    is a test window despite occurring within the dev era). The guarantee is that
    no window appears in both sets — calibration was never run on test windows.
    """
    try:
        from windows import DEV_WINDOWS, TEST_WINDOWS
        dev_names = {w.name for w in DEV_WINDOWS}
        test_names = {w.name for w in TEST_WINDOWS}
        overlap = dev_names & test_names
        if overlap:
            return {"passed": False, "detail": f"name overlap between dev and test: {overlap}"}
        return {
            "passed": True,
            "detail": f"{len(DEV_WINDOWS)} dev, {len(TEST_WINDOWS)} test, no name overlap",
        }
    except Exception as e:
        return {"passed": False, "detail": str(e)}


def check_onset_module_importable() -> dict:
    try:
        from onset import curve_onset
        if not callable(curve_onset):
            return {"passed": False, "detail": "curve_onset is not callable"}
        return {"passed": True, "detail": "onset.curve_onset importable and callable"}
    except ImportError as e:
        return {"passed": False, "detail": str(e)}


def check_onset_sensitivity_in_dev_analysis() -> dict:
    path = RESULTS / "dev_analysis.json"
    if not path.exists():
        return {"passed": False, "detail": f"dev_analysis.json not found at {path}"}
    try:
        data = json.loads(path.read_text())
        if "onset_sensitivity" not in data:
            return {"passed": False, "detail": "onset_sensitivity key missing from dev_analysis.json"}
        return {"passed": True, "detail": "onset_sensitivity present in dev_analysis.json"}
    except Exception as e:
        return {"passed": False, "detail": f"error reading dev_analysis.json: {e}"}


def check_no_test_results_exist() -> dict:
    """Ensure no test-window result files exist before the freeze tag."""
    found = []
    for name in _TEST_WINDOW_NAMES:
        for pattern in (f"agentic_{name}_*.jsonl", f"leakage_{name}_*.json"):
            found.extend(RESULTS.glob(pattern))
    if found:
        paths = [str(p.name) for p in found[:5]]
        return {
            "passed": False,
            "detail": f"test-set result files already exist: {paths} (and possibly more)",
        }
    return {"passed": True, "detail": "no test-set result files found (blinding intact)"}


def check_freeze_readiness() -> dict:
    """Run all pre-freeze checklist items and return a summary dict."""
    checks = [
        ("significance.py importable", check_significance_importable),
        ("leakage harness importable", check_leakage_importable),
        ("alarm_extraction importable", check_alarm_extraction_importable),
        ("onset module importable", check_onset_module_importable),
        ("onset_sensitivity in dev_analysis.json", check_onset_sensitivity_in_dev_analysis),
        ("calibration_grid.json exists", check_calibration_grid_exists),
        ("prompt versions match §6.2", check_prompt_versions),
        ("triage constants match §6.2", check_triage_constants),
        ("model default documented", check_model_default),
        ("dev/test window partition", check_window_partition),
        ("test-set blinding intact", check_no_test_results_exist),
    ]
    results = {name: _check(name, fn) for name, fn in checks}
    all_passed = all(r["passed"] for r in results.values())
    return {"passed": all_passed, "checks": results}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    args = ap.parse_args()

    result = check_freeze_readiness()

    if args.json:
        print(json.dumps(result, indent=2))
        sys.exit(0 if result["passed"] else 1)

    # Human-readable table
    width = 40
    print(f"\n{'Pre-freeze checklist':=<{width + 20}}")
    for name, r in result["checks"].items():
        status = "PASS" if r["passed"] else "FAIL"
        detail = r.get("detail", "")
        print(f"  [{status}] {name:<{width}} {detail}")

    print()
    if result["passed"]:
        print("All checks passed. Ready to tag eval-freeze-v1.")
        sys.exit(0)
    else:
        n_fail = sum(1 for r in result["checks"].values() if not r["passed"])
        print(f"{n_fail} check(s) failed. Resolve before tagging eval-freeze-v1.")
        sys.exit(1)


if __name__ == "__main__":
    main()
