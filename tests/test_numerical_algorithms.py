from _shared import *


class NumericalAlgorithmTests(unittest.TestCase):

    def test_gauss_legendre_order_6_produces_result(self) -> None:
        source = branch("A0", 1.0, [(0.0, (0.0, 0.0, 0.0)), (2.0, (0.0, 0.0, 0.0))])
        target = branch("B0", 1.0, [(0.0, (1.0, 0.0, 0.0)), (2.0, (1.0, 0.0, 0.0))])
        matrix = compute_branch_phase_matrix((source,), (target,), mass=0.5, quadrature_order=6)
        self.assertNotEqual(matrix[0][0], 0.0)

    def test_gauss_legendre_order_10_produces_result(self) -> None:
        source = branch("A0", 1.0, [(0.0, (0.0, 0.0, 0.0)), (2.0, (0.0, 0.0, 0.0))])
        target = branch("B0", 1.0, [(0.0, (1.0, 0.0, 0.0)), (2.0, (1.0, 0.0, 0.0))])
        matrix = compute_branch_phase_matrix((source,), (target,), mass=0.5, quadrature_order=10)
        self.assertNotEqual(matrix[0][0], 0.0)

    def test_3d_backreaction_shifts_off_axis_target(self) -> None:
        source = branch("A0", 1.0, [(0.0, (0.0, 0.0, 0.0)), (2.0, (0.0, 0.0, 0.0))])
        target = branch("B0", 1.0, [(0.0, (1.0, 1.0, 0.0)), (2.0, (1.0, 1.0, 0.0))])
        updated = evolve_backreacted_branch(source, target, mass=0.5, propagation="instantaneous", response_strength=0.1)
        self.assertNotEqual(updated.points[0].position[0], target.points[0].position[0])
        self.assertNotEqual(updated.points[0].position[1], target.points[0].position[1])

    def test_lebedev_quadrature_returns_amplitudes(self) -> None:
        source = branch("A0", 1.0, [(0.0, (0.0, 0.0, 0.0)), (2.0, (0.0, 0.0, 0.0))])
        result = compute_lebedev_displacement_amplitudes(
            (source,), field_mass=0.5, momentum_cutoff=1.0, lebedev_order=14,
        )
        self.assertEqual(result.direction_count, 14)
        self.assertEqual(result.quadrature_order, 14)
        self.assertEqual(len(result.amplitudes), 1)
        self.assertEqual(len(result.angular_error_estimate), 1)

    def test_lebedev_26_more_directions_than_14(self) -> None:
        source = branch("A0", 1.0, [(0.0, (0.0, 0.0, 0.0)), (2.0, (0.0, 0.0, 0.0))])
        r14 = compute_lebedev_displacement_amplitudes(
            (source,), field_mass=0.5, momentum_cutoff=1.0, lebedev_order=14,
        )
        r26 = compute_lebedev_displacement_amplitudes(
            (source,), field_mass=0.5, momentum_cutoff=1.0, lebedev_order=26,
        )
        self.assertGreater(r26.direction_count, r14.direction_count)

    def test_lebedev_supports_higher_order_direction_rules(self) -> None:
        source = branch("A0", 1.0, [(0.0, (0.0, 0.0, 0.0)), (2.0, (0.0, 0.0, 0.0))])
        r50 = compute_lebedev_displacement_amplitudes(
            (source,), field_mass=0.5, momentum_cutoff=1.0, lebedev_order=50,
        )
        r110 = compute_lebedev_displacement_amplitudes(
            (source,), field_mass=0.5, momentum_cutoff=1.0, lebedev_order=110,
        )
        r194 = compute_lebedev_displacement_amplitudes(
            (source,), field_mass=0.5, momentum_cutoff=1.0, lebedev_order=194,
        )
        self.assertEqual(r50.direction_count, 50)
        self.assertEqual(r110.direction_count, 110)
        self.assertEqual(r194.direction_count, 194)
        self.assertGreater(r194.direction_count, r110.direction_count)
        self.assertGreater(r110.direction_count, r50.direction_count)
        self.assertGreaterEqual(r194.angular_error_estimate[0], 0.0)

    def test_tabulated_high_order_lebedev_rule_integrates_degree_10_polynomial(self) -> None:
        directions, weights = scalar_field._resolve_lebedev_rule(50)
        self.assertAlmostEqual(sum(weights), 1.0, places=12)
        x10_average = sum(weight * (direction[0] ** 10) for direction, weight in zip(directions, weights))
        self.assertAlmostEqual(x10_average, 1.0 / 11.0, places=12)

    def test_lebedev_rejects_noncanonical_direction_count(self) -> None:
        source = branch("A0", 1.0, [(0.0, (0.0, 0.0, 0.0)), (2.0, (0.0, 0.0, 0.0))])
        with self.assertRaisesRegex(ValueError, "exact supported Lebedev orders"):
            compute_lebedev_displacement_amplitudes(
                (source,), field_mass=0.5, momentum_cutoff=1.0, lebedev_order=42,
            )

    def test_extrapolated_lebedev_returns_multi_order_summary(self) -> None:
        source = branch("A0", 1.0, [(0.0, (1.7, 0.0, 0.0)), (2.0, (1.7, 0.0, 0.0))])
        result = compute_extrapolated_lebedev_displacement_amplitudes(
            (source,), field_mass=0.5, momentum_cutoff=2.5, lebedev_orders=(26, 50, 110, 194),
        )
        self.assertEqual(result.orders_used, (26, 50, 110, 194))
        self.assertEqual(len(result.raw_results), 4)
        self.assertEqual(result.raw_results[-1].direction_count, 194)
        self.assertEqual(result.amplitudes, result.raw_results[-1].amplitudes)
        self.assertEqual(len(result.extrapolated_amplitudes), 1)
        self.assertGreaterEqual(result.estimated_error[0], 0.0)

    def test_extrapolated_lebedev_rejects_noncanonical_orders(self) -> None:
        source = branch("A0", 1.0, [(0.0, (1.7, 0.0, 0.0)), (2.0, (1.7, 0.0, 0.0))])
        with self.assertRaisesRegex(ValueError, "exact supported Lebedev orders"):
            compute_extrapolated_lebedev_displacement_amplitudes(
                (source,), field_mass=0.5, momentum_cutoff=2.5, lebedev_orders=(26, 42, 194),
            )

