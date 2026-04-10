"""Lazy access to vendored exact Lebedev spherical quadrature tables."""

from __future__ import annotations

import json
from functools import lru_cache
from importlib.resources import files


@lru_cache(maxsize=1)
def _load_tabulated_lebedev_tables() -> dict[int, str]:
    data_path = files("relativistic_circuit_locality").joinpath("data/lebedev_tables.json")
    raw_tables = json.loads(data_path.read_text(encoding="utf-8"))
    return {int(order): table for order, table in raw_tables.items()}


def get_tabulated_lebedev_orders() -> tuple[int, ...]:
    return tuple(sorted(_load_tabulated_lebedev_tables()))


def get_tabulated_lebedev_table(order: int) -> str | None:
    return _load_tabulated_lebedev_tables().get(order)
