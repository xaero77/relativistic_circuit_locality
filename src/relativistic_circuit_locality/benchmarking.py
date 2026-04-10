"""Lightweight benchmark and profiling entrypoints for representative workloads."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from time import perf_counter
from typing import Callable

from .core import (
    BranchPath,
    compute_branch_phase_matrix,
    compute_wavepacket_phase_matrix,
)
from .experimental import solve_finite_difference_kg
from .geometry import TrajectoryPoint


@dataclass(frozen=True)
class BenchmarkResult:
    name: str
    iterations: int
    average_seconds: float
    best_seconds: float
    worst_seconds: float


def profile_call(callback: Callable[[], object]) -> float:
    started_at = perf_counter()
    callback()
    return perf_counter() - started_at


def run_benchmark(name: str, callback: Callable[[], object], *, iterations: int = 5) -> BenchmarkResult:
    if iterations <= 0:
        raise ValueError("iterations must be positive.")
    durations = tuple(profile_call(callback) for _ in range(iterations))
    return BenchmarkResult(
        name=name,
        iterations=iterations,
        average_seconds=mean(durations),
        best_seconds=min(durations),
        worst_seconds=max(durations),
    )


def _benchmark_branches() -> tuple[BranchPath, BranchPath]:
    source = BranchPath(
        label="source",
        charge=1.0,
        points=(
            TrajectoryPoint(0.0, (0.0, 0.0, 0.0)),
            TrajectoryPoint(2.0, (0.0, 0.2, 0.0)),
            TrajectoryPoint(4.0, (0.0, 0.4, 0.0)),
        ),
    )
    target = BranchPath(
        label="target",
        charge=-1.0,
        points=(
            TrajectoryPoint(0.0, (2.0, 0.0, 0.0)),
            TrajectoryPoint(2.0, (2.0, 0.2, 0.0)),
            TrajectoryPoint(4.0, (2.0, 0.4, 0.0)),
        ),
    )
    return source, target


def benchmark_representative_workloads(*, iterations: int = 5) -> tuple[BenchmarkResult, ...]:
    source, target = _benchmark_branches()
    return (
        run_benchmark(
            "branch_phase_matrix",
            lambda: compute_branch_phase_matrix(
                (source, target),
                (source, target),
                mass=0.3,
                propagation="kg_retarded",
                quadrature_order=4,
            ),
            iterations=iterations,
        ),
        run_benchmark(
            "wavepacket_phase_matrix",
            lambda: compute_wavepacket_phase_matrix(
                (source, target),
                (source, target),
                widths_a=(0.0, 0.2),
                widths_b=(0.1, 0.3),
                mass=0.3,
                propagation="kg_retarded",
                quadrature_order=4,
                radial_quadrature_order=5,
            ),
            iterations=iterations,
        ),
        run_benchmark(
            "finite_difference_kg",
            lambda: solve_finite_difference_kg(
                source,
                time_slices=(0.0, 0.2, 0.4, 0.6),
                spatial_points=((-1.0, 0.0, 0.0), (0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
                mass=0.3,
                time_error_tolerance=1e-5,
            ),
            iterations=iterations,
        ),
    )


def format_benchmark_report(results: tuple[BenchmarkResult, ...]) -> str:
    lines = ["name iterations average_s best_s worst_s"]
    for result in results:
        lines.append(
            f"{result.name} {result.iterations} "
            f"{result.average_seconds:.6f} {result.best_seconds:.6f} {result.worst_seconds:.6f}"
        )
    return "\n".join(lines)


def main() -> None:
    print(format_benchmark_report(benchmark_representative_workloads()))


if __name__ == "__main__":
    main()
