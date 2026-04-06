"""스칼라장 궤적 및 위상 도우미에 대한 회귀 테스트."""

import unittest

from relativistic_circuit_locality.scalar_field import (
    BranchPath,
    TrajectoryPoint,
    compute_branch_phase_matrix,
    compute_closest_approach,
    compute_entanglement_phase,
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


if __name__ == "__main__":
    unittest.main()
