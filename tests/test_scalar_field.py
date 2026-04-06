"""스칼라장 궤적 및 위상 도우미에 대한 회귀 테스트."""

import unittest
from math import asinh, pi

from relativistic_circuit_locality.scalar_field import (
    BranchPath,
    TrajectoryPoint,
    analyze_branch_pair_phase,
    analyze_phase_decomposition,
    analyze_wavepacket_phase_decomposition,
    compute_branch_phase_matrix,
    compute_closest_approach,
    compute_entanglement_phase,
    compute_wavepacket_phase_matrix,
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


if __name__ == "__main__":
    unittest.main()
