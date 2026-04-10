from __future__ import annotations

"""Shared helpers for runnable example scenarios."""

from collections.abc import Iterable

from ..core import BranchPath, TrajectoryPoint


def branch(label: str, charge: float, x0: float) -> BranchPath:
    return BranchPath(
        label=label,
        charge=charge,
        points=(
            TrajectoryPoint(0.0, (x0, 0.0, 0.0)),
            TrajectoryPoint(2.0, (x0, 0.0, 0.0)),
            TrajectoryPoint(4.0, (x0, 0.0, 0.0)),
        ),
    )


def base_branches() -> tuple[tuple[BranchPath, ...], tuple[BranchPath, ...]]:
    branches_a = (branch("A0", 1.0, -2.0), branch("A1", 1.0, -1.0))
    branches_b = (branch("B0", 1.0, 1.0), branch("B1", 1.0, 2.0))
    return branches_a, branches_b


def print_kv_items(items: Iterable[tuple[str, object]]) -> None:
    for key, value in items:
        print(f"{key} = {value}")
