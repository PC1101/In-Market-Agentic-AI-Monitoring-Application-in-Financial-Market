# scripts/archive/

Eval-phase orchestration scripts — used to run the 12 dev + 8 test agentic cells
across multiple overnight sessions (Week 3–4). Archived for provenance; not needed
for reproduction (use `run_agentic.py` directly with the two-pass cache pattern).

| Script | Purpose |
|--------|---------|
| `run_remaining_events.py` | Batched remaining dev-set event windows |
| `notify_when_done.py` | Desktop toast when a long run finishes |
| `compare_d3_results.py` | Quick dev-set latency comparison (superseded by `analyze_dev_results.py`) |
| `resilient_resume.py` | Wrapper with auto-restart on DeadRunnerError (before day-level retry was built in) |
| `overnight_d5.py` | Overnight batch for day-5 test-set windows |
| `analyze_viability.py` | Produced `viability_dev.json` (operational metrics per window) |

Note: `run_remaining_events.py`, `notify_when_done.py`, and `compare_d3_results.py`
were removed from tracking in an earlier commit; only `resilient_resume.py`,
`overnight_d5.py`, and `analyze_viability.py` physically remain here.
