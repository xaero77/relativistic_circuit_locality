from _shared import *


class FundamentalLimitationTests(unittest.TestCase):

    def test_spline_branch_path_static_matches_linear(self):
        """정지 궤적에서 spline 보간은 linear 보간과 동일해야 한다."""
        sp = SplineBranchPath(
            label="S0", charge=1.0,
            points=(
                TrajectoryPoint(0.0, (1.0, 0.0, 0.0)),
                TrajectoryPoint(2.0, (1.0, 0.0, 0.0)),
                TrajectoryPoint(4.0, (1.0, 0.0, 0.0)),
            ),
        )
        pos = sp.position_at(1.0)
        self.assertAlmostEqual(pos[0], 1.0)
        self.assertAlmostEqual(pos[1], 0.0)

    def test_spline_branch_path_curved_differs_from_linear(self):
        """비선형 궤적에서 spline 보간은 piecewise linear 과 달라야 한다."""
        points = (
            TrajectoryPoint(0.0, (0.0, 0.0, 0.0)),
            TrajectoryPoint(1.0, (1.0, 2.0, 0.0)),
            TrajectoryPoint(2.0, (3.0, 0.0, 0.0)),
            TrajectoryPoint(3.0, (4.0, 2.0, 0.0)),
            TrajectoryPoint(4.0, (6.0, 0.0, 0.0)),
        )
        sp = SplineBranchPath(label="C0", charge=1.0, points=points)
        bp = BranchPath(label="C0", charge=1.0, points=points)
        sp_mid = sp.position_at(0.5)
        bp_mid = bp.position_at(0.5)
        # spline 의 y값은 linear 보간의 1.0 과 달라야 한다
        self.assertNotAlmostEqual(sp_mid[1], bp_mid[1], places=2)

    def test_spline_refined_branch_path_has_more_points(self):
        """refined_branch_path 는 원래보다 더 많은 점을 가져야 한다."""
        sp = SplineBranchPath(
            label="R0", charge=1.0,
            points=(
                TrajectoryPoint(0.0, (0.0, 0.0, 0.0)),
                TrajectoryPoint(2.0, (1.0, 0.0, 0.0)),
                TrajectoryPoint(4.0, (2.0, 0.0, 0.0)),
            ),
        )
        refined = sp.refined_branch_path(subdivisions=4)
        self.assertGreater(len(refined.points), len(sp.points))
        # 2 segments * 4 subdivisions + 1 = 9 points
        self.assertEqual(len(refined.points), 9)

    def test_spline_phase_matrix_produces_result(self):
        """compute_spline_branch_phase_matrix 가 유효한 위상 행렬을 반환한다."""
        sp_a = (SplineBranchPath("A0", 1.0, (
            TrajectoryPoint(0.0, (-2.0, 0.0, 0.0)),
            TrajectoryPoint(2.0, (-2.0, 0.0, 0.0)),
            TrajectoryPoint(4.0, (-2.0, 0.0, 0.0)),
        )),)
        sp_b = (SplineBranchPath("B0", 1.0, (
            TrajectoryPoint(0.0, (2.0, 0.0, 0.0)),
            TrajectoryPoint(2.0, (2.0, 0.0, 0.0)),
            TrajectoryPoint(4.0, (2.0, 0.0, 0.0)),
        )),)
        matrix = compute_spline_branch_phase_matrix(sp_a, sp_b, mass=0.5)
        self.assertEqual(len(matrix), 1)
        self.assertEqual(len(matrix[0]), 1)
        self.assertNotEqual(matrix[0][0], 0.0)

    def test_fock_space_evolution_returns_parametric_and_correction(self):
        """Fock-space 진화가 parametric phase 와 time-ordering correction 을 반환한다."""
        branches_a = (branch("A0", 1.0, [(0.0, (-2.0, 0.0, 0.0)), (2.0, (-2.0, 0.0, 0.0)), (4.0, (-2.0, 0.0, 0.0))]),)
        branches_b = (branch("B0", 1.0, [(0.0, (2.0, 0.0, 0.0)), (2.0, (2.0, 0.0, 0.0)), (4.0, (2.0, 0.0, 0.0))]),)
        momenta = MOMENTA_AXIAL_2
        result = compute_fock_space_evolution(
            branches_a[0], branches_b[0], momenta,
            field_mass=0.5, source_width_a=0.2, source_width_b=0.2,
        )
        self.assertEqual(len(result.mode_evolutions), 2)
        self.assertIsInstance(result.parametric_phase, float)
        self.assertIsInstance(result.time_ordering_correction, float)
        self.assertAlmostEqual(
            result.total_phase,
            result.parametric_phase + result.time_ordering_correction + result.third_order_correction,
        )

    def test_fock_space_evolution_moving_source_has_correction(self):
        """움직이는 source 에 대해 time-ordering correction 이 0 이 아니어야 한다."""
        branch_a = BranchPath("A0", 1.0, (
            TrajectoryPoint(0.0, (-2.0, 0.0, 0.0)),
            TrajectoryPoint(2.0, (0.0, 1.0, 0.0)),
            TrajectoryPoint(4.0, (2.0, 0.0, 0.0)),
        ))
        branch_b = BranchPath("B0", 1.0, (
            TrajectoryPoint(0.0, (2.0, 0.0, 0.0)),
            TrajectoryPoint(2.0, (0.0, -1.0, 0.0)),
            TrajectoryPoint(4.0, (-2.0, 0.0, 0.0)),
        ))
        momenta = MOMENTA_AXIAL_3
        result = compute_fock_space_evolution(
            branch_a, branch_b, momenta,
            field_mass=0.5, source_width_a=0.2, source_width_b=0.2,
            quadrature_order=5,
        )
        # moving sources → non-trivial time-ordering correction
        self.assertNotAlmostEqual(result.time_ordering_correction, 0.0, places=10)

    def test_adaptive_phase_integral_converges(self):
        """adaptive quadrature 가 일반 quadrature 와 가까운 값을 준다."""
        branches_a = (
            branch("A0", 1.0, [(0.0, (-2.0, 0.0, 0.0)), (2.0, (-2.0, 0.0, 0.0)), (4.0, (-2.0, 0.0, 0.0))]),
            branch("A1", 1.0, [(0.0, (-1.0, 0.0, 0.0)), (2.0, (-1.0, 0.0, 0.0)), (4.0, (-1.0, 0.0, 0.0))]),
        )
        branches_b = (
            branch("B0", 1.0, [(0.0, (1.0, 0.0, 0.0)), (2.0, (1.0, 0.0, 0.0)), (4.0, (1.0, 0.0, 0.0))]),
            branch("B1", 1.0, [(0.0, (2.0, 0.0, 0.0)), (2.0, (2.0, 0.0, 0.0)), (4.0, (2.0, 0.0, 0.0))]),
        )
        adaptive = compute_adaptive_phase_integral(
            branches_a, branches_b, mass=0.5, tolerance=1e-8,
        )
        from relativistic_circuit_locality.scalar_field import compute_branch_phase_matrix
        fixed = compute_branch_phase_matrix(branches_a, branches_b, mass=0.5, quadrature_order=5)
        for i in range(2):
            for j in range(2):
                self.assertAlmostEqual(adaptive[i][j].phase, fixed[i][j], places=4)

    def test_adaptive_phase_integral_reports_error(self):
        """adaptive quadrature 가 오차 추정치를 반환한다."""
        branches_a = (branch("A0", 1.0, [(0.0, (-2.0, 0.0, 0.0)), (2.0, (-2.0, 0.0, 0.0)), (4.0, (-2.0, 0.0, 0.0))]),)
        branches_b = (branch("B0", 1.0, [(0.0, (2.0, 0.0, 0.0)), (2.0, (2.0, 0.0, 0.0)), (4.0, (2.0, 0.0, 0.0))]),)
        result = compute_adaptive_phase_integral(
            branches_a[:1], branches_b[:1], mass=0.5,
        )
        self.assertGreaterEqual(result[0][0].estimated_error, 0.0)
        self.assertGreater(result[0][0].segments_used, 0)

    def test_richardson_extrapolation_improves_estimate(self):
        """Richardson extrapolation 이 외삽 값을 반환한다."""
        branches_a = (branch("A0", 1.0, [(0.0, (-2.0, 0.0, 0.0)), (2.0, (-2.0, 0.0, 0.0)), (4.0, (-2.0, 0.0, 0.0))]),)
        branches_b = (branch("B0", 1.0, [(0.0, (2.0, 0.0, 0.0)), (2.0, (2.0, 0.0, 0.0)), (4.0, (2.0, 0.0, 0.0))]),)
        result = compute_richardson_extrapolated_phase(
            branches_a[:1], branches_b[:1], mass=0.5, orders=(2, 4, 6),
        )
        self.assertEqual(len(result[0][0].phases_by_order), 3)
        self.assertEqual(result[0][0].orders_used, (2, 4, 6))
        self.assertIsInstance(result[0][0].extrapolated_phase, float)
        self.assertGreaterEqual(result[0][0].estimated_error, 0.0)

    def test_richardson_extrapolation_converges_for_static(self):
        """정적 branch 에서 다른 order 의 결과가 일관성을 보여야 한다."""
        branches_a = (branch("A0", 1.0, [(0.0, (-2.0, 0.0, 0.0)), (2.0, (-2.0, 0.0, 0.0)), (4.0, (-2.0, 0.0, 0.0))]),)
        branches_b = (branch("B0", 1.0, [(0.0, (2.0, 0.0, 0.0)), (2.0, (2.0, 0.0, 0.0)), (4.0, (2.0, 0.0, 0.0))]),)
        result = compute_richardson_extrapolated_phase(
            branches_a[:1], branches_b[:1], mass=0.5, orders=(2, 4, 6, 8),
        )
        phases = result[0][0].phases_by_order
        # 정적이므로 order 가 커져도 값이 크게 변하지 않아야 한다
        for p in phases:
            self.assertAlmostEqual(p, phases[-1], places=4)

