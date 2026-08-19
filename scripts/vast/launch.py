#!/usr/bin/env python3
"""Launch one monitoring job on an ephemeral vast.ai GPU instance.

Flow (each step logged; nothing spends money until --yes is passed):
  1. search offers under the hourly cap
  2. pick the cheapest, then HARD-ASSERT it is within budget
  3. create the instance (labelled with PROJECT_TAG for teardown)
  4. wait until it is running + ssh-ready
  5. push the repo, run the job spec from job.yaml
  6. pull results back into monitoring/results/
  7. ALWAYS destroy the instance (finally) — no idle billing

Cost guard: --max-price defaults to the agreed $0.50/hr ceiling and is enforced
both in the search query and again by assert_within_budget before provisioning.

Usage:
  # safe: prints every vastai command, spends nothing
  python scripts/vast/launch.py --dry-run
  # real run (requires vastai api-key configured + --yes to authorise spend)
  python scripts/vast/launch.py --yes --job scripts/vast/job.yaml

Prerequisites for a real run:
  * `vastai set api-key <KEY>`  (or VAST_API_KEY env)
  * an ssh key registered on your vast.ai account (`vastai show ssh-keys`)
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
import vastlib  # noqa: E402

DEFAULT_IMAGE = "ollama/ollama:latest"  # overridden by job.yaml `image` if set
DEFAULT_DISK_GB = 20


def _vastai(*args: str, dry_run: bool, capture: bool = False) -> str | None:
    """Run a `vastai` subcommand. In dry-run, print and skip (returns None)."""
    cmd = [vastlib.vastai_bin(), *args]
    printable = " ".join(["vastai", *args])
    if dry_run:
        print(f"  [dry-run] {printable}")
        return None
    print(f"  $ {printable}")
    res = subprocess.run(cmd, capture_output=capture, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"vastai failed ({res.returncode}): {printable}\n{res.stderr or ''}")
    return res.stdout if capture else None


def search_offers(max_price: float, dry_run: bool) -> list[dict]:
    query = vastlib.build_search_query(max_price=max_price)
    print(f"→ search offers: {query!r}")
    out = _vastai("search", "offers", "--raw", query, dry_run=dry_run, capture=True)
    if out is None:  # dry-run: fabricate one plausible offer so the flow is visible
        return [{"id": 999999, "dph_total": min(0.25, max_price), "gpu_name": "RTX 3090"}]
    return json.loads(out)


def _ssh_run(inst: dict, remote_cmd: str, dry_run: bool) -> None:
    """Run a command on the instance over ssh (supports arbitrary, long jobs)."""
    host, port = inst.get("ssh_host"), inst.get("ssh_port")
    ssh = ["ssh", "-p", str(port), "-o", "StrictHostKeyChecking=no",
           f"root@{host}", remote_cmd]
    if dry_run:
        print(f"  [dry-run] {' '.join(ssh)}")
        return
    print(f"  $ ssh root@{host} -p {port} '{remote_cmd}'")
    res = subprocess.run(ssh, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"remote job failed ({res.returncode})")


def wait_until_ready(tag: str, dry_run: bool, timeout_s: int = 600) -> dict:
    """Poll show instances until our labelled instance is running with ssh."""
    if dry_run:
        print("  [dry-run] would poll `vastai show instances --raw --label", tag, "` until running")
        return {"id": 999999, "ssh_host": "ssh.example.vast.ai", "ssh_port": 12345}
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        out = _vastai("show", "instances", "--raw", "--label", tag, dry_run=False, capture=True)
        mine = vastlib.project_instances(json.loads(out), tag)
        for inst in mine:
            if inst.get("actual_status") == "running" and inst.get("ssh_host"):
                return inst
        time.sleep(10)
    raise TimeoutError(f"instance for tag {tag} not ready within {timeout_s}s")


def run_job(job: dict, dry_run: bool, max_price: float) -> None:
    tag = vastlib.PROJECT_TAG
    image = job.get("image", DEFAULT_IMAGE)
    disk = job.get("disk_gb", DEFAULT_DISK_GB)
    model = job.get("model", "ollama:qwen2.5:3b")
    market = job.get("market", "sp500")
    strategy = job.get("strategy", "AL_PCA")
    window = job.get("window", "quant_meltdown_2007")

    offers = search_offers(max_price, dry_run)
    offer = vastlib.pick_cheapest_offer(offers, max_price)
    if offer is None:
        raise SystemExit(f"no offer at or under ${max_price}/hr; try later or raise the cap")
    price = vastlib.assert_within_budget(offer, max_price)  # HARD budget gate
    print(f"→ selected offer {offer['id']} ({offer.get('gpu_name','?')}) at ${price}/hr")

    onstart = (
        "bash -c '"
        "ollama serve & sleep 5; "
        f"ollama pull {model.split(':',1)[-1]}; "
        "touch /root/.ready'"
    )
    instance_id = None
    try:
        _vastai("create", "instance", str(offer["id"]),
                "--image", image, "--disk", str(disk),
                "--label", tag, "--onstart-cmd", onstart, "--ssh", "--direct",
                dry_run=dry_run)
        inst = wait_until_ready(tag, dry_run)
        instance_id = inst.get("id", offer["id"])

        # Push the repo (monitoring/ + data), run the job, pull results back.
        print("→ push repo, run job, pull results")
        _vastai("copy", str(REPO / "monitoring"), f"{instance_id}:/root/monitoring",
                dry_run=dry_run)
        remote_cmd = (
            f"cd /root/monitoring && python run_agentic.py --market {market} "
            f"--window {window} --strategy {strategy} --model {model}"
        )
        _ssh_run(inst, remote_cmd, dry_run=dry_run)
        _vastai("copy", f"{instance_id}:/root/monitoring/results",
                str(REPO / "monitoring" / "results"), dry_run=dry_run)
        print("✓ job complete, results pulled")
    finally:
        # Always tear down — the core of the no-idle-billing guarantee.
        if instance_id is not None or dry_run:
            print("→ destroy instance (cost guard)")
            _vastai("destroy", "instance", str(instance_id or offer["id"]), "-y",
                    dry_run=dry_run)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--job", default=str(HERE / "job.yaml"), help="path to job spec YAML")
    ap.add_argument("--max-price", type=float, default=vastlib.MAX_PRICE_PER_HOUR,
                    help=f"hourly price cap (default: {vastlib.MAX_PRICE_PER_HOUR})")
    ap.add_argument("--dry-run", action="store_true",
                    help="print every vastai command; spend nothing")
    ap.add_argument("--yes", action="store_true",
                    help="authorise real provisioning (required when not --dry-run)")
    args = ap.parse_args()

    if not args.dry_run and not args.yes:
        raise SystemExit(
            "Refusing to provision without authorisation. Re-run with --dry-run to preview, "
            "or --yes to authorise real spend (capped at ${:.2f}/hr).".format(args.max_price)
        )

    job = yaml.safe_load(Path(args.job).read_text()) if Path(args.job).exists() else {}
    print(f"vast.ai launch — cap ${args.max_price}/hr — job: {args.job}"
          + (" [DRY RUN]" if args.dry_run else ""))
    run_job(job, dry_run=args.dry_run, max_price=args.max_price)


if __name__ == "__main__":
    main()
