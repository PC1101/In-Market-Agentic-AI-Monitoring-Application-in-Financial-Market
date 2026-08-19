# vast.ai orchestration harness

Ephemeral, Dockerised GPU runs for the monitoring pipeline (design §7). Rent the
cheapest GPU under an hourly cap, run one job, pull results back, **always**
destroy the instance. Pay per run; no idle billing.

## Cost guard (why you won't get a surprise bill)

The agreed ceiling is **$0.50/hr** (`vastlib.MAX_PRICE_PER_HOUR`). It is enforced
four ways:

1. The search query only asks for offers `dph_total <= 0.50`.
2. `pick_cheapest_offer` re-filters under the cap client-side.
3. `assert_within_budget` is a hard gate that **raises** immediately before any
   `create instance` — provisioning over budget is impossible without editing code.
4. Every instance is labelled `inmarket-monitor`; `teardown.py` finds and destroys
   any stray, and `launch.py` destroys its own instance in a `finally` block.

Nothing spends money without `--yes`. `--dry-run` previews every `vastai` command.

## One-time setup (on this machine)

```bash
# 1. API key (from https://cloud.vast.ai/account/)
vastai set api-key <YOUR_KEY>          # stored in ~/.config/vastai/vast_api_key
# 2. register an ssh key so `copy`/`execute` can reach the instance
vastai show ssh-keys                    # confirm one exists, else `vastai create ssh-key`
```

## Use

```bash
# Preview — safe, spends nothing:
python scripts/vast/launch.py --dry-run

# Real run (authorise spend; capped at $0.50/hr):
python scripts/vast/launch.py --yes --job scripts/vast/job.yaml

# Safety sweep — destroy any instance we left running:
python scripts/vast/teardown.py --dry-run     # list
python scripts/vast/teardown.py --yes          # destroy
```

Edit `job.yaml` to choose market / strategy / window / model. The `image:` there
defaults to `ollama/ollama:latest` (model pulled on start); for warm starts build
and push the `Dockerfile` and point `image:` at your tag.

## Tests

Pure cost-guard logic (no network) is covered by
`monitoring/tests/test_vast_harness.py`. The live path is exercised with
`launch.py --dry-run`.
