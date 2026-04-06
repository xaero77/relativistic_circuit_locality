from __future__ import annotations

"""스칼라장 시뮬레이션 API를 실행해 보는 작은 예제."""

from .scalar_field import (
    BranchPath,
    TrajectoryPoint,
    compute_entanglement_phase,
    simulate,
)


def _branch(label: str, charge: float, x0: float) -> BranchPath:
    # 데모에서는 거리 의존 위상 효과만 보이도록 각 분기를 정지 상태로 둔다.
    return BranchPath(
        label=label,
        charge=charge,
        points=(
            TrajectoryPoint(0.0, (x0, 0.0, 0.0)),
            TrajectoryPoint(2.0, (x0, 0.0, 0.0)),
            TrajectoryPoint(4.0, (x0, 0.0, 0.0)),
        ),
    )


def main() -> None:
    # 각 계에 두 개의 분기를 두어 2x2 상호작용 위상 행렬을 만든다.
    branches_a = (_branch("A0", 1.0, -2.0), _branch("A1", 1.0, -1.0))
    branches_b = (_branch("B0", 1.0, 1.0), _branch("B1", 1.0, 2.0))
    instantaneous = simulate(branches_a, branches_b, mass=0.5)
    retarded = simulate(branches_a, branches_b, mass=0.5, propagation="retarded")
    print("closest_approach =", round(instantaneous.closest_approach, 6))
    print("mediation_intervals =", instantaneous.mediation_intervals)
    print("instantaneous_phase_matrix =", instantaneous.phase_matrix)
    print("retarded_phase_matrix =", retarded.phase_matrix)
    print(
        "instantaneous_relative_entangling_phase =",
        round(compute_entanglement_phase(instantaneous.phase_matrix, 0, 1, 0, 1), 6),
    )
    print(
        "retarded_relative_entangling_phase =",
        round(compute_entanglement_phase(retarded.phase_matrix, 0, 1, 0, 1), 6),
    )


if __name__ == "__main__":
    main()
