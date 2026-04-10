from _shared import *


class ExtendedFundamentalLimitationTests(unittest.TestCase):

    def test_fock_space_third_order_magnus_adds_correction(self):
        """3차 Magnus 항이 moving source 에서 0 이 아닌 보정을 준다."""
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
        momenta = MOMENTA_AXIAL_2
        result = compute_fock_space_evolution(
            branch_a, branch_b, momenta,
            field_mass=0.5, source_width_a=0.2, source_width_b=0.2,
            magnus_order=3,
        )
        self.assertNotAlmostEqual(result.third_order_correction, 0.0, places=10)
        self.assertAlmostEqual(
            result.total_phase,
            result.parametric_phase + result.time_ordering_correction + result.third_order_correction,
        )

    def test_fock_space_magnus_order_2_omits_third(self):
        """magnus_order=2 일 때 3차 보정이 0 이다."""
        branch_a = BranchPath("A0", 1.0, (
            TrajectoryPoint(0.0, (-2.0, 0.0, 0.0)),
            TrajectoryPoint(2.0, (0.0, 1.0, 0.0)),
            TrajectoryPoint(4.0, (2.0, 0.0, 0.0)),
        ))
        branch_b = BranchPath("B0", 1.0, (
            TrajectoryPoint(0.0, (2.0, 0.0, 0.0)),
            TrajectoryPoint(2.0, (2.0, 0.0, 0.0)),
            TrajectoryPoint(4.0, (2.0, 0.0, 0.0)),
        ))
        momenta = MOMENTA_AXIAL_1
        result = compute_fock_space_evolution(
            branch_a, branch_b, momenta,
            field_mass=0.5, source_width_a=0.2, source_width_b=0.2,
            magnus_order=2,
        )
        self.assertEqual(result.third_order_correction, 0.0)

    def test_interpolate_field_lattice_at_grid_point(self):
        """격자점에서 보간 값이 원래 값과 일치한다."""
        source = branch("S0", 1.0, [(0.0, (0.0, 0.0, 0.0)), (2.0, (0.0, 0.0, 0.0))])
        lattice = solve_field_lattice(
            source, time_slices=(0.0, 1.0, 2.0),
            spatial_points=((1.0, 0.0, 0.0), (2.0, 0.0, 0.0)),
            mass=0.5,
        )
        # 격자점 중 하나를 정확히 선택
        result = interpolate_field_lattice(lattice, 1.0, (1.0, 0.0, 0.0))
        # 해당 격자점의 값과 일치
        expected = [s.value for s in lattice.samples if s.t == 1.0 and s.position == (1.0, 0.0, 0.0)][0]
        self.assertAlmostEqual(result.value, expected)

    def test_interpolate_field_lattice_between_points(self):
        """격자점 사이의 보간 값이 nearest 값과 다를 수 있다."""
        source = branch("S0", 1.0, [(0.0, (0.0, 0.0, 0.0)), (2.0, (0.0, 0.0, 0.0))])
        lattice = solve_field_lattice(
            source, time_slices=(0.0, 2.0),
            spatial_points=((1.0, 0.0, 0.0), (3.0, 0.0, 0.0)),
            mass=0.5,
        )
        result = interpolate_field_lattice(lattice, 1.0, (2.0, 0.0, 0.0))
        self.assertIsInstance(result.value, float)
        self.assertGreater(len(result.interpolation_weights), 0)

    def test_running_coupling_differs_from_bare(self):
        """running coupling 이 bare coupling 과 다른 위상을 준다."""
        branches_a = (branch("A0", 1.0, [(0.0, (-2.0, 0.0, 0.0)), (2.0, (-2.0, 0.0, 0.0)), (4.0, (-2.0, 0.0, 0.0))]),)
        branches_b = (branch("B0", 1.0, [(0.0, (2.0, 0.0, 0.0)), (2.0, (2.0, 0.0, 0.0)), (4.0, (2.0, 0.0, 0.0))]),)
        result = compute_running_coupling_phase_matrix(
            branches_a, branches_b, mass=0.5,
            energy_scale=10.0, beta_coefficient=0.1,
        )
        # running coupling 이 적용되면 bare 와 달라야 함
        self.assertNotAlmostEqual(result.bare_matrix[0][0], result.running_matrix[0][0])
        self.assertEqual(result.energy_scale, 10.0)

    def test_running_coupling_at_reference_scale_matches_bare(self):
        """reference scale 에서 running coupling 은 bare 와 일치한다."""
        branches_a = (branch("A0", 1.0, [(0.0, (-2.0, 0.0, 0.0)), (2.0, (-2.0, 0.0, 0.0)), (4.0, (-2.0, 0.0, 0.0))]),)
        branches_b = (branch("B0", 1.0, [(0.0, (2.0, 0.0, 0.0)), (2.0, (2.0, 0.0, 0.0)), (4.0, (2.0, 0.0, 0.0))]),)
        result = compute_running_coupling_phase_matrix(
            branches_a, branches_b, mass=0.5,
            energy_scale=1.0, reference_scale=1.0, beta_coefficient=0.1,
        )
        # E = μ 이면 α(E) = α₀
        self.assertAlmostEqual(result.bare_matrix[0][0], result.running_matrix[0][0])

    def test_validate_symbolic_bookkeeping_consistent(self):
        """유효한 mode sample 에서 bookkeeping 검증이 consistent 를 반환한다."""
        labeled = (
            ("A0", ((1.0 + 0.0j, 0.2 + 0.0j),)),
            ("B0", ((0.8 + 0.0j, 0.1 + 0.0j),)),
        )
        result = validate_symbolic_bookkeeping(labeled)
        self.assertTrue(result.is_consistent)
        self.assertEqual(len(result.symbolic_norms), 2)
        self.assertEqual(len(result.numerical_norms), 2)
        for residual in result.norm_residuals:
            self.assertLess(residual, 1e-6)

    def test_validate_symbolic_bookkeeping_phase_antisymmetry(self):
        """검증에서 위상 행렬 antisymmetry residual 이 작아야 한다."""
        labeled = (
            ("X", ((0.5 + 0.3j,),)),
            ("Y", ((0.7 - 0.1j,),)),
            ("Z", ((0.2 + 0.4j,),)),
        )
        result = validate_symbolic_bookkeeping(labeled)
        self.assertLess(result.phase_matrix_residual, 1e-6)


if __name__ == "__main__":
    unittest.main()
