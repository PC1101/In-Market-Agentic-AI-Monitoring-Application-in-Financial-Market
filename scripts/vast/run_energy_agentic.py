#!/usr/bin/env python3
"""Run the energy-market AGENTIC layer on a vast.ai GPU (qwen2.5:3b via ollama).

Why vast.ai here (unlike the CPU backtests): the agentic loop is GPU-bound LLM
inference, and the box's clean IP can query GDELT (the local dev IP is rate-limit
banned from iterating). This script does the whole thing on one ephemeral box and
always destroys it:

  provision (budget-capped) -> ship monitoring/ + the two energy curves ->
  install ollama + qwen2.5:3b + light deps -> ingest GDELT energy news (clean IP) ->
  run_agentic.py --market energy --model ollama:qwen2.5:3b -> pull results -> destroy

FinBERT news-stress is skipped for this run (zero stress-cache) to keep the box
light (no torch/model download); the News Context Agent still reads the real GDELT
articles via the LLM, which is the core of the agentic layer. Add FinBERT later by
installing torch/transformers on the box and dropping the --finbert-cache flag.

Cost guard: --max-price defaults to $0.50/hr, enforced by vastlib (search query +
assert_within_budget). Nothing spends without --yes.

Usage:
  python scripts/vast/run_energy_agentic.py --dry-run          # preview, no spend
  python scripts/vast/run_energy_agentic.py --yes              # real run (oil_crash_2020)
  python scripts/vast/run_energy_agentic.py --yes --window energy_spike_2022
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
import vastlib  # noqa: E402
from launch import _vastai, _ssh_exec_retry, _wait_running, search_offers  # noqa: E402

# News baseline: the agentic loop uses ~120 calendar days before the window for the
# news z-score, so ingest from a bit before each window's start.
WINDOW_INGEST = {
    "oil_crash_2020": ("2019-10-15", "2020-05-29"),
    "energy_spike_2022": ("2021-10-01", "2022-06-30"),
    "calm_energy_2017": ("2016-12-01", "2017-09-29"),
    "calm_energy_2019": ("2018-12-01", "2019-09-30"),
}

IMAGE = "python:3.12-slim"
DISK_GB = 24
RESULTS_LOCAL = REPO / "monitoring" / "results" / "vast_energy"


def _remote_setup_and_run(window: str, strategy: str, model: str) -> str:
    """The single bash script run on the box (setup -> ingest -> agentic run)."""
    ing_start, ing_end = WINDOW_INGEST[window]
    return f"""set -e
echo '=== apt + ollama ==='
apt-get update -qq && apt-get install -y -qq curl >/dev/null
curl -fsSL https://ollama.com/install.sh | sh
echo '=== python deps ==='
pip install -q pandas numpy pyarrow
echo '=== start ollama + pull model ==='
(ollama serve >/root/ollama.log 2>&1 &) ; sleep 8
ollama pull {model.split(':',1)[-1]}
echo '{{}}' > /root/empty_finbert.json
cd /root/monitoring
echo '=== ingest GDELT energy news ({ing_start}..{ing_end}) on the box clean IP ==='
python -c "from news import gdelt; from providers.energy.news import energy_symbol_queries, DEFAULT_ROOT; \
print(gdelt.build_gdelt_store(energy_symbol_queries(), '{ing_start}', '{ing_end}', DEFAULT_ROOT, throttle_s=5, maxrecords=250))"
echo '=== run energy agentic loop ({window} x {strategy} x {model}) ==='
python run_agentic.py --market energy --window {window} --strategy {strategy} \
  --model {model} --finbert-cache /root/empty_finbert.json
echo '=== DONE_AGENTIC ==='
"""


def run(window: str, strategy: str, model: str, max_price: float, dry_run: bool) -> None:
    tag = vastlib.PROJECT_TAG
    offers = search_offers(max_price, dry_run)
    offer = vastlib.pick_cheapest_offer(offers, max_price)
    if offer is None:
        raise SystemExit(f"no offer at or under ${max_price}/hr")
    price = vastlib.assert_within_budget(offer, max_price)
    print(f"-> selected offer {offer['id']} ({offer.get('gpu_name','?')}) at ${price}/hr")

    instance_id = None
    try:
        out = _vastai("create", "instance", str(offer["id"]),
                      "--image", IMAGE, "--disk", str(DISK_GB),
                      "--label", tag, "--onstart-cmd", "sleep infinity",
                      "--ssh", "--direct", "--raw", dry_run=dry_run, capture=True)
        instance_id = 999999 if dry_run else (json.loads(out).get("new_contract")
                                              or json.loads(out).get("id"))
        print(f"-> instance {instance_id}; waiting to run...")
        if not dry_run:
            _wait_running(instance_id)

        # Ship code + the two energy curves, recreating the repo-relative layout the
        # energy PnL provider expects (/root/XSectional/results, /root/Stat Arb/...).
        print("-> ship monitoring/ + energy curves")
        _ssh_exec_retry(instance_id,
                        'mkdir -p "/root/XSectional/results" "/root/Stat Arb/statsArb-dev/results"',
                        dry_run=dry_run)
        _vastai("copy", str(REPO / "monitoring"), f"{instance_id}:/root/monitoring", dry_run=dry_run)
        _vastai("copy", str(REPO / "XSectional" / "results" / "equity_curve_energy.csv"),
                f"{instance_id}:/root/XSectional/results/equity_curve_energy.csv", dry_run=dry_run)
        _vastai("copy", str(REPO / "Stat Arb" / "statsArb-dev" / "results" / "equity_curve_energy_al.csv"),
                f"{instance_id}:/root/Stat Arb/statsArb-dev/results/equity_curve_energy_al.csv", dry_run=dry_run)

        print(f"-> setup + ingest + agentic run on box ({window} x {strategy} x {model})")
        result = _ssh_exec_retry(instance_id, _remote_setup_and_run(window, strategy, model),
                                 dry_run=dry_run, attempts=1)
        if not dry_run:
            tail = "\n".join(result.strip().splitlines()[-25:])
            print("-- remote tail --\n" + tail)
            if "DONE_AGENTIC" not in result:
                raise RuntimeError("agentic run did not reach DONE_AGENTIC")

        print("-> pull results")
        RESULTS_LOCAL.mkdir(parents=True, exist_ok=True)
        _vastai("copy", f"{instance_id}:/root/monitoring/results/agentic_{window}_{strategy}.jsonl",
                str(RESULTS_LOCAL / f"agentic_{window}_{strategy}.jsonl"), dry_run=dry_run)
        print(f"OK -> results in {RESULTS_LOCAL}")
    finally:
        if instance_id is not None:
            print("-> destroy instance (cost guard)")
            _vastai("destroy", "instance", str(instance_id), "-y", dry_run=dry_run)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--window", default="oil_crash_2020", choices=list(WINDOW_INGEST))
    ap.add_argument("--strategy", default="AL_PCA", choices=["AL_PCA", "JT_MOM"])
    ap.add_argument("--model", default="ollama:qwen2.5:3b")
    ap.add_argument("--max-price", type=float, default=vastlib.MAX_PRICE_PER_HOUR)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--yes", action="store_true", help="authorise real spend")
    args = ap.parse_args()
    if not args.dry_run and not args.yes:
        raise SystemExit("Re-run with --dry-run to preview, or --yes to authorise spend "
                         f"(capped ${args.max_price}/hr).")
    print(f"energy agentic on vast.ai - cap ${args.max_price}/hr - {args.window} x {args.strategy}"
          + (" [DRY RUN]" if args.dry_run else ""))
    run(args.window, args.strategy, args.model, args.max_price, args.dry_run)


if __name__ == "__main__":
    main()
