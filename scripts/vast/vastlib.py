"""Pure logic for the vast.ai orchestration harness — no network here.

Everything that decides *whether to spend money* lives in this module so it can
be unit-tested without hitting the vast.ai API. ``launch.py`` and ``teardown.py``
are thin CLIs over these functions.

Cost guard (defence in depth):
  1. ``build_search_query`` bakes the price cap into the server-side search.
  2. ``pick_cheapest_offer`` re-filters offers under the cap client-side.
  3. ``assert_within_budget`` is a hard gate called immediately before any
     ``create instance`` — it raises rather than provisioning over budget.
  4. Every instance is created with ``PROJECT_TAG`` as its label so
     ``project_instances`` can find and destroy strays.
"""
from __future__ import annotations

#: Agreed budget ceiling (USD per hour). See design §7 / §11.7.
MAX_PRICE_PER_HOUR = 0.50

#: Label stamped on every instance we create, so teardown can find our own.
PROJECT_TAG = "inmarket-monitor"

#: qwen2.5:3b is small; 8 GiB of GPU RAM is comfortable headroom.
MIN_GPU_RAM_GB = 8


class BudgetExceededError(RuntimeError):
    """Raised when an offer's price is missing or above the hourly cap."""


def build_search_query(max_price: float = MAX_PRICE_PER_HOUR,
                       min_gpu_ram_gb: int = MIN_GPU_RAM_GB,
                       num_gpus: int = 1) -> str:
    """vast.ai search query string enforcing the price + GPU constraints."""
    return (
        f"dph_total <= {max_price} "
        f"gpu_ram >= {min_gpu_ram_gb} "
        f"num_gpus == {num_gpus} "
        f"rentable = true verified = true"
    )


def pick_cheapest_offer(offers: list[dict], max_price: float = MAX_PRICE_PER_HOUR) -> dict | None:
    """Cheapest offer whose hourly price is known and <= ``max_price`` (or None)."""
    affordable = [
        o for o in offers
        if o.get("dph_total") is not None and o["dph_total"] <= max_price
    ]
    if not affordable:
        return None
    return min(affordable, key=lambda o: o["dph_total"])


def assert_within_budget(offer: dict, max_price: float = MAX_PRICE_PER_HOUR) -> float:
    """Hard gate before provisioning. Returns the price, or raises.

    Called immediately before ``create instance`` — the last line of defence
    against renting something above the agreed ceiling.
    """
    price = offer.get("dph_total")
    if price is None:
        raise BudgetExceededError(f"offer {offer.get('id')} has no price; refusing to rent")
    if price > max_price:
        raise BudgetExceededError(
            f"offer {offer.get('id')} at ${price}/hr exceeds cap ${max_price}/hr"
        )
    return price


def project_instances(instances: list[dict], tag: str = PROJECT_TAG) -> list[dict]:
    """Instances labelled with our project tag (the ones teardown may destroy)."""
    return [i for i in instances if i.get("label") == tag]
