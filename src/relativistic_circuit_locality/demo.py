from __future__ import annotations

"""스칼라장 시뮬레이션 API를 실행해 보는 작은 예제."""

from .scalar_field import (
    BranchPath,
    TrajectoryPoint,
    analyze_branch_pair_coherent_state,
    analyze_branch_pair_phase,
    compute_branch_displacement_amplitudes,
    compute_branch_pair_displacements,
    compute_displacement_operator_phase,
    compute_wavepacket_phase_matrix,
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
    time_symmetric = simulate(branches_a, branches_b, mass=0.5, propagation="time_symmetric")
    causal_history = simulate(branches_a, branches_b, mass=0.5, propagation="causal_history")
    kg_retarded = simulate(branches_a, branches_b, mass=0.5, propagation="kg_retarded")
    print("closest_approach =", round(instantaneous.closest_approach, 6))
    print("mediation_intervals =", instantaneous.mediation_intervals)
    print("instantaneous_phase_matrix =", instantaneous.phase_matrix)
    print("retarded_phase_matrix =", retarded.phase_matrix)
    print("time_symmetric_phase_matrix =", time_symmetric.phase_matrix)
    print("causal_history_phase_matrix =", causal_history.phase_matrix)
    print("kg_retarded_phase_matrix =", kg_retarded.phase_matrix)
    print(
        "instantaneous_relative_entangling_phase =",
        round(compute_entanglement_phase(instantaneous.phase_matrix, 0, 1, 0, 1), 6),
    )
    print(
        "retarded_relative_entangling_phase =",
        round(compute_entanglement_phase(retarded.phase_matrix, 0, 1, 0, 1), 6),
    )
    print(
        "time_symmetric_relative_entangling_phase =",
        round(compute_entanglement_phase(time_symmetric.phase_matrix, 0, 1, 0, 1), 6),
    )
    print(
        "causal_history_relative_entangling_phase =",
        round(compute_entanglement_phase(causal_history.phase_matrix, 0, 1, 0, 1), 6),
    )
    print(
        "kg_retarded_relative_entangling_phase =",
        round(compute_entanglement_phase(kg_retarded.phase_matrix, 0, 1, 0, 1), 6),
    )
    momenta = ((0.0, 0.0, 0.5), (0.5, 0.0, 0.0), (0.5, 0.5, 0.0))
    displacement_a = compute_branch_displacement_amplitudes(
        branches_a,
        momenta,
        field_mass=0.5,
        source_width=0.2,
    )
    pair_displacements = compute_branch_pair_displacements(
        branches_a,
        branches_b,
        momenta,
        field_mass=0.5,
        source_width_a=0.2,
        source_width_b=0.2,
    )
    print("single_branch_displacements_A =", displacement_a)
    print("pair_displacements =", pair_displacements)
    print(
        "pair_displacement_phase_A0B0_vs_A1B1 =",
        round(compute_displacement_operator_phase(pair_displacements[0][0], pair_displacements[1][1]), 6),
    )
    coherent_state = analyze_branch_pair_coherent_state(
        branches_a[0],
        branches_b[0],
        momenta,
        field_mass=0.5,
        source_width_a=0.2,
        source_width_b=0.2,
        elapsed_time=1.5,
    )
    print("coherent_state_A0_B0 =", coherent_state)
    wavepacket_matrix = compute_wavepacket_phase_matrix(
        branches_a,
        branches_b,
        widths_a=(0.3, 0.5),
        widths_b=(0.3, 0.5),
        mass=0.5,
    )
    print("wavepacket_phase_matrix =", wavepacket_matrix)
    pair_breakdown = analyze_branch_pair_phase(
        branches_a[0],
        branches_b[0],
        mass=0.5,
        propagation="kg_retarded",
        cutoff=0.1,
    )
    print("pair_breakdown_A0_B0 =", pair_breakdown)


if __name__ == "__main__":
    main()
