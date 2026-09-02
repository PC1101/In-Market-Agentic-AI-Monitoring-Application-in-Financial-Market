#!/usr/bin/env python3
"""Destroy every vast.ai instance we created — the standalone cost guard.

``launch.py`` already destroys its instance in a finally block, but if it was
killed (Ctrl-C, crash, network drop) an instance could still be billing. Run
this to sweep up anything labelled with our PROJECT_TAG.

Usage:
  python scripts/vast/teardown.py --dry-run   # list what would be destroyed
  python scripts/vast/teardown.py --yes        # actually destroy them
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import vastlib  # noqa: E402


def _vastai(*args: str, capture: bool = False) -> str | None:
    res = subprocess.run([vastlib.vastai_bin(), *args], capture_output=capture, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"vastai failed: {' '.join(args)}\n{res.stderr or ''}")
    return res.stdout if capture else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="list only, destroy nothing")
    ap.add_argument("--yes", action="store_true", help="authorise destruction")
    args = ap.parse_args()

    out = _vastai("show", "instances", "--raw", capture=True)
    mine = vastlib.project_instances(json.loads(out), vastlib.PROJECT_TAG)

    if not mine:
        print(f"No instances labelled '{vastlib.PROJECT_TAG}'. Nothing to tear down.")
        return

    print(f"Found {len(mine)} instance(s) labelled '{vastlib.PROJECT_TAG}':")
    for inst in mine:
        print(f"  id={inst['id']} status={inst.get('actual_status')} "
              f"${inst.get('dph_total','?')}/hr {inst.get('gpu_name','?')}")

    if args.dry_run or not args.yes:
        print("\n(dry-run / not authorised) re-run with --yes to destroy these.")
        return

    for inst in mine:
        print(f"→ destroy instance {inst['id']}")
        _vastai("destroy", "instance", str(inst["id"]), "-y")
    print("✓ all project instances destroyed")


if __name__ == "__main__":
    main()
