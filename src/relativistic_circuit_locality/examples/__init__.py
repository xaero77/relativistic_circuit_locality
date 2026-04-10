from __future__ import annotations

"""Runnable example scenarios for the package."""

from . import core_phases, field_sampling, research
from ._shared import print_kv_items

SCENARIOS = {
    "core": core_phases.collect_results,
    "field": field_sampling.collect_results,
    "research": research.collect_results,
}


def run_scenario(name: str) -> None:
    items = SCENARIOS[name]()
    print_kv_items(items)


def run_all() -> None:
    first = True
    for name in ("core", "field", "research"):
        if not first:
            print()
        first = False
        print(f"[{name}]")
        run_scenario(name)


__all__ = ["SCENARIOS", "run_all", "run_scenario"]
