"""스칼라장 궤적 및 위상 도우미에 대한 회귀 테스트."""

import unittest
from math import asinh, pi

from relativistic_circuit_locality.scalar_field import (
    BranchPath,
    TrajectoryPoint,
    analyze_branch_pair_coherent_state,
    analyze_branch_pair_phase,
    analyze_phase_decomposition,
    analyze_wavepacket_phase_decomposition,
    compute_branch_displacement_amplitudes,
    compute_branch_phase_matrix,
    compute_branch_pair_displacements,
    compute_closest_approach,
    compute_displacement_operator_phase,
    compute_entanglement_phase,
    compute_wavepacket_phase_matrix,
    evolve_coherent_state,
    field_mediation_intervals,
    is_field_mediated,
)


def branch(label: str, charge: float, samples: list[tuple[float, tuple[float, float, float]]]) -> BranchPath:
    # 테스트에서는 간단한 헬퍼를 써서 핵심 기하 구조가 잘 보이게 한다.
    return BranchPath(
        label=label,
        charge=charge,
        points=tuple(TrajectoryPoint(t, position) for t, position in samples),
    )


class ScalarFieldTests(unittest.TestCase):
    def test_closest_approach_tracks_minimum_distance(self) -> None:
        left = branch("A0", 1.0, [(0.0, (-2.0, 0.0, 0.0)), (1.0, (-1.0, 0.0, 0.0))])
        right = branch("B0", 1.0, [(0.0, (3.0, 0.0, 0.0)), (1.0, (1.5, 0.0, 0.0))])
        self.assertAlmostEqual(compute_closest_approach(left, right), 2.5)

    def test_closest_approach_uses_continuous_segment_minimum_with_mismatched_sampling(self) -> None:
        left = branch("A0", 1.0, [(0.0, (-1.0, 0.0, 0.0)), (2.0, (1.0, 0.0, 0.0))])
        right = branch("B0", 1.0, [(0.0, (0.0, 1.0, 0.0)), (1.0, (0.0, 1.0, 0.0)), (2.0, (0.0, 1.0, 0.0))])
        self.assertAlmostEqual(compute_closest_approach(left, right), 1.0)

    def test_full_interval_is_field_mediated_when_branches_stay_spacelike(self) -> None:
        branches_a = (
            branch("A0", 1.0, [(0.0, (-3.0, 0.0, 0.0)), (1.0, (-3.0, 0.0, 0.0)), (2.0, (-3.0, 0.0, 0.0))]),
        )
        branches_b = (
            branch("B0", 1.0, [(0.0, (3.0, 0.0, 0.0)), (1.0, (3.0, 0.0, 0.0)), (2.0, (3.0, 0.0, 0.0))]),
        )
        self.assertEqual(field_mediation_intervals(branches_a, branches_b), ((0.0, 2.0),))
        self.assertTrue(is_field_mediated(branches_a, branches_b))

    def test_field_mediation_intervals_support_mismatched_sampling(self) -> None:
        branches_a = (
            branch("A0", 1.0, [(0.0, (-2.0, 0.0, 0.0)), (2.0, (-2.0, 0.0, 0.0))]),
        )
        branches_b = (
            branch("B0", 1.0, [(0.0, (2.0, 0.0, 0.0)), (1.0, (2.0, 0.0, 0.0)), (2.0, (2.0, 0.0, 0.0))]),
        )
        self.assertEqual(field_mediation_intervals(branches_a, branches_b), ((0.0, 2.0),))
        self.assertTrue(is_field_mediated(branches_a, branches_b))

    def test_contact_breaks_field_mediation(self) -> None:
        branches_a = (
            branch("A0", 1.0, [(0.0, (-1.0, 0.0, 0.0)), (1.0, (0.0, 0.0, 0.0)), (2.0, (1.0, 0.0, 0.0))]),
        )
        branches_b = (
            branch("B0", 1.0, [(0.0, (1.0, 0.0, 0.0)), (1.0, (0.0, 0.0, 0.0)), (2.0, (-1.0, 0.0, 0.0))]),
        )
        self.assertEqual(field_mediation_intervals(branches_a, branches_b), ())
        self.assertFalse(is_field_mediated(branches_a, branches_b))

    def test_phase_matrix_is_symmetric_for_equal_charges(self) -> None:
        branches_a = (
            branch("A0", 1.0, [(0.0, (-2.0, 0.0, 0.0)), (1.0, (-2.0, 0.0, 0.0))]),
            branch("A1", 1.0, [(0.0, (-1.0, 0.0, 0.0)), (1.0, (-1.0, 0.0, 0.0))]),
        )
        branches_b = (
            branch("B0", 1.0, [(0.0, (1.0, 0.0, 0.0)), (1.0, (1.0, 0.0, 0.0))]),
            branch("B1", 1.0, [(0.0, (2.0, 0.0, 0.0)), (1.0, (2.0, 0.0, 0.0))]),
        )
        matrix = compute_branch_phase_matrix(branches_a, branches_b, mass=0.5)
        self.assertAlmostEqual(matrix[0][0], matrix[1][1])
        self.assertLess(matrix[1][0], matrix[0][0])

    def test_relative_entangling_phase_detects_branch_dependence(self) -> None:
        branches_a = (
            branch("A0", 1.0, [(0.0, (-2.0, 0.0, 0.0)), (1.0, (-2.0, 0.0, 0.0))]),
            branch("A1", 1.0, [(0.0, (-1.0, 0.0, 0.0)), (1.0, (-1.0, 0.0, 0.0))]),
        )
        branches_b = (
            branch("B0", 1.0, [(0.0, (1.0, 0.0, 0.0)), (1.0, (1.0, 0.0, 0.0))]),
            branch("B1", 1.0, [(0.0, (3.0, 0.0, 0.0)), (1.0, (3.0, 0.0, 0.0))]),
        )
        matrix = compute_branch_phase_matrix(branches_a, branches_b, mass=0.25)
        self.assertNotEqual(compute_entanglement_phase(matrix, 0, 1, 0, 1), 0.0)

    def test_phase_matrix_is_stable_under_time_resampling(self) -> None:
        # 같은 세계선이라면 샘플 밀도가 달라도 적분 위상은 같아야 한다.
        coarse = branch("A0", 1.0, [(0.0, (-2.0, 0.0, 0.0)), (2.0, (-2.0, 0.0, 0.0))])
        fine = branch(
            "A0",
            1.0,
            [(0.0, (-2.0, 0.0, 0.0)), (1.0, (-2.0, 0.0, 0.0)), (2.0, (-2.0, 0.0, 0.0))],
        )
        target = branch("B0", 1.0, [(0.0, (2.0, 0.0, 0.0)), (1.0, (2.0, 0.0, 0.0)), (2.0, (2.0, 0.0, 0.0))])
        coarse_matrix = compute_branch_phase_matrix((coarse,), (target,), mass=0.5)
        fine_matrix = compute_branch_phase_matrix((fine,), (target,), mass=0.5)
        self.assertAlmostEqual(coarse_matrix[0][0], fine_matrix[0][0])

    def test_retarded_phase_is_smaller_when_signal_has_not_fully_arrived(self) -> None:
        source = branch("A0", 1.0, [(0.0, (0.0, 0.0, 0.0)), (1.0, (0.0, 0.0, 0.0)), (2.0, (0.0, 0.0, 0.0))])
        target = branch("B0", 1.0, [(0.0, (1.0, 0.0, 0.0)), (1.0, (1.0, 0.0, 0.0)), (2.0, (1.0, 0.0, 0.0))])
        instantaneous = compute_branch_phase_matrix((source,), (target,), mass=0.0)
        retarded = compute_branch_phase_matrix((source,), (target,), mass=0.0, propagation="retarded")
        self.assertLess(abs(retarded[0][0]), abs(instantaneous[0][0]))
        self.assertAlmostEqual(retarded[0][0], -1.0 / (4.0 * pi), places=12)

    def test_retarded_phase_respects_light_speed_parameter(self) -> None:
        source = branch("A0", 1.0, [(0.0, (0.0, 0.0, 0.0)), (1.0, (0.0, 0.0, 0.0)), (2.0, (0.0, 0.0, 0.0))])
        target = branch("B0", 1.0, [(0.0, (1.0, 0.0, 0.0)), (1.0, (1.0, 0.0, 0.0)), (2.0, (1.0, 0.0, 0.0))])
        faster = compute_branch_phase_matrix(
            (source,),
            (target,),
            mass=0.0,
            propagation="retarded",
            light_speed=2.0,
        )
        slower = compute_branch_phase_matrix(
            (source,),
            (target,),
            mass=0.0,
            propagation="retarded",
            light_speed=0.5,
        )
        self.assertLess(abs(slower[0][0]), abs(faster[0][0]))

    def test_time_symmetric_phase_sits_between_directional_retarded_orders(self) -> None:
        moving = branch("A0", 1.0, [(0.0, (-4.0, 0.0, 0.0)), (2.0, (0.0, 0.0, 0.0)), (4.0, (4.0, 0.0, 0.0))])
        static = branch("B0", 1.0, [(0.0, (0.0, 1.0, 0.0)), (2.0, (1.0, 1.0, 0.0)), (4.0, (2.0, 1.0, 0.0))])
        forward = compute_branch_phase_matrix((moving,), (static,), mass=0.0, propagation="retarded")
        backward = compute_branch_phase_matrix((static,), (moving,), mass=0.0, propagation="retarded")
        symmetric = compute_branch_phase_matrix((moving,), (static,), mass=0.0, propagation="time_symmetric")
        low = min(forward[0][0], backward[0][0])
        high = max(forward[0][0], backward[0][0])
        self.assertGreaterEqual(symmetric[0][0], low)
        self.assertLessEqual(symmetric[0][0], high)

    def test_causal_history_is_sensitive_to_past_source_support(self) -> None:
        short_source = branch("A0", 1.0, [(0.0, (0.0, 0.0, 0.0)), (1.5, (0.0, 0.0, 0.0)), (3.0, (0.0, 0.0, 0.0))])
        long_source = branch("A0", 1.0, [(-2.0, (0.0, 0.0, 0.0)), (0.0, (0.0, 0.0, 0.0)), (1.5, (0.0, 0.0, 0.0)), (3.0, (0.0, 0.0, 0.0))])
        target = branch("B0", 1.0, [(0.0, (1.0, 0.0, 0.0)), (1.5, (1.0, 0.0, 0.0)), (3.0, (1.0, 0.0, 0.0))])
        short_history = compute_branch_phase_matrix((short_source,), (target,), mass=0.5, propagation="causal_history")
        long_history = compute_branch_phase_matrix((long_source,), (target,), mass=0.5, propagation="causal_history")
        self.assertGreater(abs(long_history[0][0]), abs(short_history[0][0]))

    def test_causal_history_vanishes_without_past_light_cone_overlap(self) -> None:
        source = branch("A0", 1.0, [(0.0, (0.0, 0.0, 0.0)), (0.4, (0.0, 0.0, 0.0))])
        target = branch("B0", 1.0, [(0.0, (2.0, 0.0, 0.0)), (0.4, (2.0, 0.0, 0.0))])
        causal_history = compute_branch_phase_matrix((source,), (target,), mass=0.0, propagation="causal_history")
        self.assertAlmostEqual(causal_history[0][0], 0.0)

    def test_higher_order_quadrature_better_matches_analytic_moving_worldline_integral(self) -> None:
        moving = branch("A0", 1.0, [(0.0, (-1.0, 0.0, 0.0)), (2.0, (1.0, 0.0, 0.0))])
        static = branch("B0", 1.0, [(0.0, (0.0, 1.0, 0.0)), (2.0, (0.0, 1.0, 0.0))])
        expected = -asinh(1.0) / (2.0 * pi)
        midpoint = compute_branch_phase_matrix((moving,), (static,), mass=0.0, quadrature_order=1)
        higher_order = compute_branch_phase_matrix((moving,), (static,), mass=0.0, quadrature_order=5)
        self.assertLess(abs(higher_order[0][0] - expected), abs(midpoint[0][0] - expected))
        self.assertAlmostEqual(higher_order[0][0], expected, places=4)

    def test_quadrature_order_must_be_positive(self) -> None:
        source = branch("A0", 1.0, [(0.0, (0.0, 0.0, 0.0)), (1.0, (0.0, 0.0, 0.0))])
        target = branch("B0", 1.0, [(0.0, (1.0, 0.0, 0.0)), (1.0, (1.0, 0.0, 0.0))])
        with self.assertRaises(ValueError):
            compute_branch_phase_matrix((source,), (target,), mass=0.0, quadrature_order=0)

    def test_phase_breakdown_matches_instantaneous_cross_matrix(self) -> None:
        branch_a = branch("A0", 1.0, [(0.0, (-2.0, 0.0, 0.0)), (2.0, (-2.0, 0.0, 0.0))])
        branch_b = branch("B0", 2.0, [(0.0, (1.0, 0.0, 0.0)), (2.0, (1.0, 0.0, 0.0))])
        breakdown = analyze_branch_pair_phase(branch_a, branch_b, mass=0.5)
        matrix = compute_branch_phase_matrix((branch_a,), (branch_b,), mass=0.5)
        self.assertAlmostEqual(breakdown.directed_cross_phase_ab, matrix[0][0])
        self.assertAlmostEqual(breakdown.directed_cross_phase_ab, breakdown.directed_cross_phase_ba)
        self.assertAlmostEqual(breakdown.interaction_phase, matrix[0][0])
        self.assertAlmostEqual(
            breakdown.total_phase,
            breakdown.self_phase_a + breakdown.interaction_phase + breakdown.self_phase_b,
        )

    def test_retarded_phase_breakdown_exposes_directional_asymmetry(self) -> None:
        moving = branch("A0", 1.0, [(0.0, (-4.0, 0.0, 0.0)), (2.0, (0.0, 0.0, 0.0)), (4.0, (4.0, 0.0, 0.0))])
        static = branch("B0", 1.0, [(0.0, (0.0, 1.0, 0.0)), (2.0, (1.0, 1.0, 0.0)), (4.0, (2.0, 1.0, 0.0))])
        breakdown = analyze_branch_pair_phase(moving, static, mass=0.0, propagation="retarded")
        self.assertNotAlmostEqual(breakdown.directed_cross_phase_ab, breakdown.directed_cross_phase_ba)
        self.assertAlmostEqual(
            breakdown.interaction_phase,
            0.5 * (breakdown.directed_cross_phase_ab + breakdown.directed_cross_phase_ba),
        )

    def test_time_symmetric_breakdown_removes_directional_asymmetry(self) -> None:
        moving = branch("A0", 1.0, [(0.0, (-4.0, 0.0, 0.0)), (2.0, (0.0, 0.0, 0.0)), (4.0, (4.0, 0.0, 0.0))])
        static = branch("B0", 1.0, [(0.0, (0.0, 1.0, 0.0)), (2.0, (1.0, 1.0, 0.0)), (4.0, (2.0, 1.0, 0.0))])
        breakdown = analyze_branch_pair_phase(moving, static, mass=0.0, propagation="time_symmetric")
        self.assertAlmostEqual(breakdown.directed_cross_phase_ab, breakdown.directed_cross_phase_ba)
        self.assertAlmostEqual(breakdown.interaction_phase, breakdown.directed_cross_phase_ab)

    def test_phase_decomposition_total_matrix_combines_self_and_cross_terms(self) -> None:
        branches_a = (
            branch("A0", 1.0, [(0.0, (-2.0, 0.0, 0.0)), (2.0, (-2.0, 0.0, 0.0))]),
            branch("A1", 1.0, [(0.0, (-1.0, 0.0, 0.0)), (2.0, (-1.0, 0.0, 0.0))]),
        )
        branches_b = (
            branch("B0", 1.0, [(0.0, (1.0, 0.0, 0.0)), (2.0, (1.0, 0.0, 0.0))]),
            branch("B1", 1.0, [(0.0, (2.0, 0.0, 0.0)), (2.0, (2.0, 0.0, 0.0))]),
        )
        decomposition = analyze_phase_decomposition(branches_a, branches_b, mass=0.5)
        for r in range(len(branches_a)):
            for s in range(len(branches_b)):
                self.assertAlmostEqual(
                    decomposition.total_matrix[r][s],
                    decomposition.self_phases_a[r]
                    + decomposition.interaction_matrix[r][s]
                    + decomposition.self_phases_b[s],
                )

    def test_wavepacket_phase_recovers_point_particle_limit_at_zero_width(self) -> None:
        left = branch("A0", 1.0, [(0.0, (-2.0, 0.0, 0.0)), (2.0, (-2.0, 0.0, 0.0))])
        right = branch("B0", 1.0, [(0.0, (2.0, 0.0, 0.0)), (2.0, (2.0, 0.0, 0.0))])
        pointlike = compute_branch_phase_matrix((left,), (right,), mass=0.5)
        wavepacket = compute_wavepacket_phase_matrix((left,), (right,), widths_a=0.0, widths_b=0.0, mass=0.5)
        self.assertAlmostEqual(pointlike[0][0], wavepacket[0][0])

    def test_wavepacket_width_softens_cross_phase_magnitude(self) -> None:
        left = branch("A0", 1.0, [(0.0, (-1.0, 0.0, 0.0)), (2.0, (-1.0, 0.0, 0.0))])
        right = branch("B0", 1.0, [(0.0, (1.0, 0.0, 0.0)), (2.0, (1.0, 0.0, 0.0))])
        narrow = compute_wavepacket_phase_matrix((left,), (right,), widths_a=0.1, widths_b=0.1, mass=0.5)
        wide = compute_wavepacket_phase_matrix((left,), (right,), widths_a=0.8, widths_b=0.8, mass=0.5)
        self.assertLess(abs(wide[0][0]), abs(narrow[0][0]))

    def test_wavepacket_phase_decomposition_total_matrix_combines_terms(self) -> None:
        branches_a = (
            branch("A0", 1.0, [(0.0, (-2.0, 0.0, 0.0)), (2.0, (-2.0, 0.0, 0.0))]),
            branch("A1", 1.0, [(0.0, (-1.0, 0.0, 0.0)), (2.0, (-1.0, 0.0, 0.0))]),
        )
        branches_b = (
            branch("B0", 1.0, [(0.0, (1.0, 0.0, 0.0)), (2.0, (1.0, 0.0, 0.0))]),
            branch("B1", 1.0, [(0.0, (2.0, 0.0, 0.0)), (2.0, (2.0, 0.0, 0.0))]),
        )
        decomposition = analyze_wavepacket_phase_decomposition(
            branches_a,
            branches_b,
            widths_a=(0.3, 0.5),
            widths_b=(0.4, 0.6),
            mass=0.5,
        )
        for r in range(len(branches_a)):
            for s in range(len(branches_b)):
                self.assertAlmostEqual(
                    decomposition.total_matrix[r][s],
                    decomposition.self_phases_a[r]
                    + decomposition.interaction_matrix[r][s]
                    + decomposition.self_phases_b[s],
                )

    def test_pair_displacement_is_sum_of_single_branch_displacements(self) -> None:
        left = branch("A0", 1.0, [(0.0, (-1.0, 0.0, 0.0)), (2.0, (-1.0, 0.0, 0.0))])
        right = branch("B0", 2.0, [(0.0, (1.0, 0.0, 0.0)), (2.0, (1.0, 0.0, 0.0))])
        momenta = ((0.0, 0.0, 0.5), (0.5, 0.0, 0.0))
        displacement_left = compute_branch_displacement_amplitudes((left,), momenta, field_mass=0.5, source_width=0.2)
        displacement_right = compute_branch_displacement_amplitudes((right,), momenta, field_mass=0.5, source_width=0.3)
        pair = compute_branch_pair_displacements(
            (left,),
            (right,),
            momenta,
            field_mass=0.5,
            source_width_a=0.2,
            source_width_b=0.3,
        )
        for index in range(len(momenta)):
            self.assertAlmostEqual(pair[0][0][index].real, displacement_left[0][index].real + displacement_right[0][index].real)
            self.assertAlmostEqual(pair[0][0][index].imag, displacement_left[0][index].imag + displacement_right[0][index].imag)

    def test_displacement_phase_vanishes_for_identical_profiles(self) -> None:
        profile = (1.0 + 2.0j, -0.5j)
        self.assertAlmostEqual(compute_displacement_operator_phase(profile, profile), 0.0)

    def test_coherent_state_free_evolution_preserves_occupation_number(self) -> None:
        momenta = ((0.0, 0.0, 0.5), (0.5, 0.0, 0.0))
        initial = (1.0 + 2.0j, -0.25 + 0.75j)
        evolved = evolve_coherent_state(initial, momenta, field_mass=0.5, elapsed_time=3.0)
        self.assertAlmostEqual(evolved.occupation_number, sum(abs(value) ** 2 for value in initial))

    def test_branch_pair_coherent_state_tracks_pair_displacement_norm(self) -> None:
        left = branch("A0", 1.0, [(0.0, (-1.0, 0.0, 0.0)), (2.0, (-1.0, 0.0, 0.0))])
        right = branch("B0", 1.0, [(0.0, (1.0, 0.0, 0.0)), (2.0, (1.0, 0.0, 0.0))])
        momenta = ((0.0, 0.0, 0.5), (0.5, 0.0, 0.0), (0.5, 0.5, 0.0))
        pair = compute_branch_pair_displacements((left,), (right,), momenta, field_mass=0.5, source_width_a=0.2, source_width_b=0.2)
        coherent = analyze_branch_pair_coherent_state(
            left,
            right,
            momenta,
            field_mass=0.5,
            source_width_a=0.2,
            source_width_b=0.2,
            elapsed_time=1.0,
        )
        self.assertAlmostEqual(coherent.occupation_number, sum(abs(value) ** 2 for value in pair[0][0]))


if __name__ == "__main__":
    unittest.main()
