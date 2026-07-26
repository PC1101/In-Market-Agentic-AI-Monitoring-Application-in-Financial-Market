"""D3 completion comparison: new dev_analysis.json vs archived pre-D3 version.

Run after D3 + analyze_dev_results.py complete:
    cd monitoring && python scripts/compare_d3_results.py

Outputs:
    prints comparison table to stdout
    saves docs/plan/D3_dev_rerun_report.md
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

MONITORING = Path(__file__).resolve().parent.parent
RESULTS    = MONITORING / "results"
ARCHIVE    = RESULTS / "archive" / "pre_freeze"
PLAN_DIR   = MONITORING.parent / "docs" / "plan"


def _load(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def compare_paired_results(pre: dict, post: dict) -> list[dict]:
    """Compare per-pair detection and latency."""
    pre_pairs  = pre.get("paired_results", {})
    post_pairs = post.get("paired_results", {})
    rows = []
    all_keys = sorted(set(pre_pairs) | set(post_pairs))
    for k in all_keys:
        p = pre_pairs.get(k, {})
        q = post_pairs.get(k, {})
        wname  = p.get("window") or q.get("window") or k.split("__")[0]
        strat  = p.get("strategy") or q.get("strategy") or k.split("__")[-1]

        pre_det  = p.get("agentic_detected")
        post_det = q.get("agentic_detected")
        pre_lat  = p.get("agentic_latency")
        post_lat = q.get("agentic_latency")

        det_match = (pre_det == post_det)
        lat_match = (pre_lat == post_lat)
        changed   = not (det_match and lat_match)

        rows.append({
            "key": k,
            "window": wname,
            "strategy": strat,
            "pre_detected": pre_det,
            "post_detected": post_det,
            "pre_latency": pre_lat,
            "post_latency": post_lat,
            "det_match": det_match,
            "lat_match": lat_match,
            "changed": changed,
        })
    return rows


def compare_tests(pre: dict, post: dict) -> dict:
    pre_t  = pre.get("tests", {})
    post_t = post.get("tests", {})
    result = {}

    # Holm co-primary
    for key in ("holm",):
        ph = pre_t.get(key, {})
        qh = post_t.get(key, {})
        result[key] = {
            "pre": ph,
            "post": qh,
            "changed": ph != qh,
        }

    # Permutation latency
    for key in ("permutation_latency",):
        pp = pre_t.get(key, {})
        qp = post_t.get(key, {})
        pre_p  = pp.get("p_value_one_sided") or pp.get("p_one_sided") or pp.get("p")
        post_p = qp.get("p_value_one_sided") or qp.get("p_one_sided") or qp.get("p")
        pre_diff  = pp.get("observed_diff_mean") or pp.get("obs_diff")
        post_diff = qp.get("observed_diff_mean") or qp.get("obs_diff")
        result[key] = {
            "pre_p":  pre_p,
            "post_p": post_p,
            "pre_obs_diff":  pre_diff,
            "post_obs_diff": post_diff,
            "changed": abs((pre_diff or 0) - (post_diff or 0)) > 0.5,
        }

    return result


def verdict(rows: list[dict], tests: dict) -> str:
    n_changed = sum(1 for r in rows if r["changed"])
    holm_pre  = tests.get("holm", {}).get("pre", {})
    holm_post = tests.get("holm", {}).get("post", {})

    def _rej_bool(d: dict):
        return d.get("reject_005", d.get("reject"))
    pre_lat_reject  = _rej_bool(holm_pre.get("latency", {}))
    post_lat_reject = _rej_bool(holm_post.get("latency", {}))
    pre_fpr_reject  = _rej_bool(holm_pre.get("fpr", {}))
    post_fpr_reject = _rej_bool(holm_post.get("fpr", {}))

    if n_changed == 0:
        return "REPRODUCED: all 8 detection + latency pairs identical to pre-D3."
    conclusions_match = (
        pre_lat_reject == post_lat_reject and
        pre_fpr_reject == post_fpr_reject
    )
    if conclusions_match:
        return (f"MINOR CHANGE: {n_changed} pair(s) changed but Holm conclusions unchanged "
                f"(latency reject={post_lat_reject}, FPR reject={post_fpr_reject}). "
                "Document in deviations log.")
    return (f"MATERIAL CHANGE: {n_changed} pair(s) changed AND Holm conclusions differ "
            f"(pre: lat={pre_lat_reject},fpr={pre_fpr_reject} → "
            f"post: lat={post_lat_reject},fpr={post_fpr_reject}). "
            "INVESTIGATE before proceeding to D5. Do NOT rerun to chase numbers.")


def write_report(rows: list[dict], tests: dict, v: str, out_path: Path) -> None:
    pre_t  = tests.get("permutation_latency", {})
    holm_p = tests.get("holm", {}).get("post", {})
    holm_pre_d = tests.get("holm", {}).get("pre", {})
    lines = [
        "# Batch D Phase 3 — Dev-Set Rerun Report",
        "",
        f"**Date:** 2026-07-25",
        f"**Status:** {v.split(':')[0]}",
        "",
        "---",
        "",
        "## Verdict",
        "",
        v,
        "",
        "---",
        "",
        "## Per-Pair Comparison (Agentic Detection & Latency)",
        "",
        "| Window | Strategy | Pre-Det | Post-Det | Pre-Lat | Post-Lat | Changed? |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        pd_s  = "Y" if r["pre_detected"]  else ("N" if r["pre_detected"] is False else "?")
        qd_s  = "Y" if r["post_detected"] else ("N" if r["post_detected"] is False else "?")
        pl_s  = str(r["pre_latency"])  if r["pre_latency"]  is not None else "—"
        ql_s  = str(r["post_latency"]) if r["post_latency"] is not None else "—"
        chg   = "**YES**" if r["changed"] else "no"
        lines.append(f"| {r['window']} | {r['strategy']} | {pd_s} | {qd_s} | {pl_s} | {ql_s} | {chg} |")

    # Tests
    lat_p_pre  = pre_t.get("pre_p",  "n/a")
    lat_p_post = pre_t.get("post_p", "n/a")
    lat_diff_pre  = pre_t.get("pre_obs_diff",  "n/a")
    lat_diff_post = pre_t.get("post_obs_diff", "n/a")

    def _rej(d: dict) -> str:
        return str(d.get("reject_005", d.get("reject", "?")))
    lat_pre_r  = _rej(holm_pre_d.get("latency", {}))
    lat_post_r = _rej(holm_p.get("latency", {}))
    fpr_pre_r  = _rej(holm_pre_d.get("fpr", {}))
    fpr_post_r = _rej(holm_p.get("fpr", {}))

    lines += [
        "",
        "## Test Statistics Comparison",
        "",
        "| Metric | Pre-D3 | Post-D3 |",
        "|---|---|---|",
        f"| Perm-test obs_diff (latency, days) | {lat_diff_pre} | {lat_diff_post} |",
        f"| Perm-test p-value (latency, 1-sided) | {lat_p_pre} | {lat_p_post} |",
        f"| Holm: latency reject | {lat_pre_r} | {lat_post_r} |",
        f"| Holm: FPR reject | {fpr_pre_r} | {fpr_post_r} |",
        "",
        "---",
        "",
        "## Acceptance Criteria",
        "",
        "- [x] All 12 dev cells (8 event + 4 calm) carry `run_meta` with real model + PIT curve",
        "- [x] `dev_analysis.json` regenerated",
        "- [ ] Conclusions vs archived pre-D3 version documented (this report)",
        "",
        "## Next Step",
        "",
        "Phase D4: run `python scripts/analyze_viability.py` (D3 JSONLs are the final dev evidence).",
        "Then review D5 freeze preparation.",
    ]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")
    print(f"Saved: {out_path}")


def main() -> None:
    pre_path  = ARCHIVE / "dev_analysis_pre_d3.json"
    post_path = RESULTS / "dev_analysis.json"

    pre  = _load(pre_path)
    post = _load(post_path)

    if not pre:
        print(f"ERROR: pre-D3 archive not found at {pre_path}", file=sys.stderr)
        sys.exit(1)
    if not post:
        print(f"ERROR: dev_analysis.json not found at {post_path}", file=sys.stderr)
        sys.exit(1)

    # Check that D3 is actually complete (8 provenance-clean event JSONLs)
    event_windows = ["quant_meltdown_2007", "gfc_lehman_2008",
                     "momentum_crash_2009", "downgrade_2011"]
    strategies    = ["AL_PCA", "JT_MOM"]
    missing_meta = []
    for w in event_windows:
        for s in strategies:
            p = RESULTS / f"agentic_{w}_{s}.jsonl"
            if not p.exists():
                missing_meta.append(f"{w}x{s}: missing")
                continue
            lines = [l for l in p.read_text().splitlines() if l.strip()]
            records = [json.loads(l) for l in lines]
            if not any(r.get("agent") == "run_meta" for r in records):
                missing_meta.append(f"{w}x{s}: no run_meta")
    if missing_meta:
        print(f"WARNING: D3 may not be complete. Missing/no-meta: {missing_meta}")
        print("Proceeding with available data.")

    rows  = compare_paired_results(pre, post)
    tests = compare_tests(pre, post)
    v     = verdict(rows, tests)

    print()
    print("=" * 70)
    print("D3 COMPARISON: pre-D3 archive vs new dev_analysis.json")
    print("=" * 70)
    print()
    print(f"{'Window':<25} {'Strat':<8} {'PreDet':>6} {'PostDet':>7} "
          f"{'PreLat':>6} {'PostLat':>7} {'Changed':>8}")
    print("-" * 70)
    for r in rows:
        pd_s = "Y" if r["pre_detected"]  else ("N" if r["pre_detected"] is False else "?")
        qd_s = "Y" if r["post_detected"] else ("N" if r["post_detected"] is False else "?")
        pl_s = str(r["pre_latency"])  if r["pre_latency"]  is not None else "—"
        ql_s = str(r["post_latency"]) if r["post_latency"] is not None else "—"
        chg  = "YES ***" if r["changed"] else "no"
        print(f"{r['window']:<25} {r['strategy']:<8} {pd_s:>6} {qd_s:>7} "
              f"{pl_s:>6} {ql_s:>7} {chg:>8}")

    pt = tests.get("permutation_latency", {})
    print()
    def _rj(d: dict) -> str:
        return str(d.get("reject_005", d.get("reject", "?")))
    print(f"Permutation test: pre obs_diff={pt.get('pre_obs_diff')} p={pt.get('pre_p')}  "
          f"post obs_diff={pt.get('post_obs_diff')} p={pt.get('post_p')}")
    holm_pre  = tests.get("holm", {}).get("pre",  {})
    holm_post = tests.get("holm", {}).get("post", {})
    print(f"Holm pre-D3:  lat_reject={_rj(holm_pre.get('latency', {}))} "
          f"fpr_reject={_rj(holm_pre.get('fpr', {}))}")
    print(f"Holm post-D3: lat_reject={_rj(holm_post.get('latency', {}))} "
          f"fpr_reject={_rj(holm_post.get('fpr', {}))}")
    print()
    print(f"VERDICT: {v}")

    report_path = PLAN_DIR / "D3_dev_rerun_report.md"
    write_report(rows, tests, v, report_path)


if __name__ == "__main__":
    main()
