# monitoring/ — Classical + Agentic strategy monitoring (Week 2)

Strategy-agnostic monitoring layer for the In-Market Agentic AI Monitoring project.
It consumes any strategy's **daily PnL** (`Date,port_ret,equity,drawdown`) and runs:

1. **Classical change-point detectors** (from scratch, NumPy/SciPy only):
   - `detectors/page_hinkley.py` — Page-Hinkley sequential mean-shift test
   - `detectors/bocpd.py` — Bayesian Online Change-Point Detection (Adams & MacKay 2007)
   - `detectors/hmm.py` — 2-state Gaussian HMM regime detector (Baum-Welch + Viterbi)
   - `detectors/distributional.py` — rolling distributional-threshold detector
2. **Aggregation** (`detectors/aggregate.py`): alarm when ≥2 distinct detectors fire within 5 days.
3. **Metrics** (`metrics.py`): detection latency, false-positive rate, precision, recall vs labelled events.
4. **Agentic scaffold** (`agentic/`): prompt templates, structured-JSON output schema
   (state / action / root_cause / confidence), local-model client, information-parity guardrails, logging.

## Data contract
All detectors operate on a **daily return series** (`port_ret`) indexed by trading date.
Both strategies must emit the same daily schema:

| column   | meaning                                  |
|----------|------------------------------------------|
| Date     | trading date (YYYY-MM-DD)                |
| port_ret | daily portfolio return (decimal)         |
| equity   | cumulative equity (starts at 1.0)        |
| drawdown | running drawdown from peak (<= 0 or >=0) |

- **AL PCA**: `Stat Arb/statsArb-dev/results/full_universe/*/equity_curve.csv`
- **JT momentum**: emitted daily by `XSectional/backtest.py:run_backtest_daily` (added Week 2).

## Six windows (4 event + 2 calm)
Defined in `windows.py`. Event windows carry a ground-truth onset date used for latency/precision/recall.

## Information parity (non-negotiable)
Detectors are strictly **online / causal**: the statistic at day *t* uses only data up to *t*.
The agentic layer enforces the same via `agentic/guardrails.py` (as-of dating + timestamp filtering).

## Run
```bash
cd monitoring
pip install -r requirements.txt
python -m pytest tests/ -q          # unit tests
python run_classical.py             # end-to-end classical monitoring on both strategies x 6 windows
```
