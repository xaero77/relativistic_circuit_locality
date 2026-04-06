from __future__ import annotations

from .scalar_field import (
    BranchPath,
    TrajectoryPoint,
    compute_entanglement_phase,
    simulate,
)


def _branch(label: str, charge: float, x0: float) -> BranchPath:
    return BranchPath(
        label=label,
        charge=charge,
        points=(
            TrajectoryPoint(0.0, (x0, 0.0, 0.0)),
            TrajectoryPoint(1.0, (x0, 0.0, 0.0)),
            TrajectoryPoint(2.0, (x0, 0.0, 0.0)),
        ),
    )


def main() -> None:
    branches_a = (_branch("A0", 1.0, -2.0), _branch("A1", 1.0, -1.0))
    branches_b = (_branch("B0", 1.0, 1.0), _branch("B1", 1.0, 2.0))
    result = simulate(branches_a, branches_b, mass=0.5)
    print("closest_approach =", round(result.closest_approach, 6))
    print("mediation_intervals =", result.mediation_intervals)
    print("phase_matrix =", result.phase_matrix)
    print(
        "relative_entangling_phase =",
        round(compute_entanglement_phase(result.phase_matrix, 0, 1, 0, 1), 6),
    )


if __name__ == "__main__":
    main()
