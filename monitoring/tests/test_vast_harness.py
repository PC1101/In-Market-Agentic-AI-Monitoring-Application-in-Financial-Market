"""Tests for the vast.ai orchestration harness cost-guard + selection logic.

These cover the *pure* logic (no network): search-query construction, offer
selection under the price cap, the hard budget assertion, and tag-based
instance filtering for teardown. The live provisioning path is exercised
separately via ``launch.py --dry-run`` (no spend).
"""
import sys
from pathlib import Path

import pytest

# scripts/vast is not a package on sys.path; add it so we can import vastlib.
sys.path.insert(0, str(Path(__file__).resolve().parents[1].parent / "scripts" / "vast"))

import vastlib  # noqa: E402


def test_build_search_query_includes_price_cap():
    q = vastlib.build_search_query(max_price=0.50, min_gpu_ram_gb=8, num_gpus=1)
    assert "dph_total <= 0.5" in q
    assert "gpu_ram >= 8" in q
    assert "num_gpus == 1" in q


def test_pick_cheapest_offer_respects_cap():
    offers = [
        {"id": 1, "dph_total": 0.80, "gpu_name": "RTX 4090"},
        {"id": 2, "dph_total": 0.20, "gpu_name": "RTX 3060"},
        {"id": 3, "dph_total": 0.45, "gpu_name": "RTX 3090"},
    ]
    pick = vastlib.pick_cheapest_offer(offers, max_price=0.50)
    assert pick["id"] == 2  # cheapest under cap


def test_pick_cheapest_offer_none_when_all_too_expensive():
    offers = [{"id": 1, "dph_total": 0.60}, {"id": 2, "dph_total": 0.99}]
    assert vastlib.pick_cheapest_offer(offers, max_price=0.50) is None


def test_pick_ignores_offers_missing_price():
    offers = [{"id": 1, "dph_total": None}, {"id": 2, "dph_total": 0.30}]
    assert vastlib.pick_cheapest_offer(offers, max_price=0.50)["id"] == 2


def test_assert_within_budget_passes_under_cap():
    offer = {"id": 7, "dph_total": 0.49}
    assert vastlib.assert_within_budget(offer, max_price=0.50) == 0.49


def test_assert_within_budget_raises_over_cap():
    offer = {"id": 7, "dph_total": 0.51}
    with pytest.raises(vastlib.BudgetExceededError):
        vastlib.assert_within_budget(offer, max_price=0.50)


def test_assert_within_budget_raises_on_missing_price():
    with pytest.raises(vastlib.BudgetExceededError):
        vastlib.assert_within_budget({"id": 7, "dph_total": None}, max_price=0.50)


def test_project_instances_filters_by_tag():
    instances = [
        {"id": 10, "label": "inmarket-monitor"},
        {"id": 11, "label": "someone-else"},
        {"id": 12, "label": "inmarket-monitor"},
        {"id": 13, "label": None},
    ]
    ids = [i["id"] for i in vastlib.project_instances(instances, "inmarket-monitor")]
    assert ids == [10, 12]


def test_default_cap_is_50_cents():
    # The agreed budget ceiling; guards against accidental edits.
    assert vastlib.MAX_PRICE_PER_HOUR == 0.50
