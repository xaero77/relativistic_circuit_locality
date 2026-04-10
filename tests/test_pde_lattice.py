from _shared import *


class PdeAndLatticeTests(unittest.TestCase):

    def test_finite_difference_kg_produces_field(self) -> None:
        source = branch("A0", 1.0, [(0.0, (0.0, 0.0, 0.0)), (3.0, (0.0, 0.0, 0.0))])
        result = solve_finite_difference_kg(
            source,
            time_slices=(0.0, 0.5, 1.0, 1.5),
            spatial_points=SPATIAL_POINTS_LINE_5,
            mass=0.5,
            light_speed=1.0,
            boundary="absorbing",
        )
        self.assertEqual(len(result.field_values), 4)
        self.assertEqual(len(result.field_values[0]), 5)
        self.assertGreater(result.courant_number, 0.0)
        self.assertEqual(result.grid_shape, (5, 1, 1))
        self.assertEqual(result.spatial_dimension, 1)
        self.assertEqual(result.substeps_per_interval, (1, 1, 1))
        self.assertAlmostEqual(result.effective_time_step, 0.5)
        self.assertEqual(result.local_error_estimates, (0.0, 0.0, 0.0))
        self.assertEqual(result.time_error_tolerance, 0.0)

    def test_finite_difference_kg_periodic_boundary(self) -> None:
        source = branch("A0", 1.0, [(0.0, (0.0, 0.0, 0.0)), (2.0, (0.0, 0.0, 0.0))])
        result = solve_finite_difference_kg(
            source,
            time_slices=(0.0, 0.5, 1.0),
            spatial_points=SPATIAL_POINTS_LINE_3,
            mass=0.5,
            boundary="periodic",
        )
        self.assertEqual(len(result.field_values), 3)

    def test_finite_difference_kg_supports_cartesian_3d_grid(self) -> None:
        source = branch("A0", 1.0, [(0.0, (0.0, 0.0, 0.0)), (2.0, (0.0, 0.0, 0.0))])
        points = tuple(
            (x, y, z)
            for z in (-1.0, 1.0)
            for y in (-1.0, 1.0)
            for x in (-1.0, 1.0)
        )
        result = solve_finite_difference_kg(
            source,
            time_slices=(0.0, 0.25, 0.5),
            spatial_points=points,
            mass=0.5,
            boundary="periodic",
        )
        self.assertEqual(len(result.field_values), 3)
        self.assertEqual(len(result.field_values[0]), 8)
        self.assertEqual(result.grid_shape, (2, 2, 2))
        self.assertEqual(result.spatial_dimension, 3)

    def test_finite_difference_kg_adaptive_substeps_reduce_effective_dt(self) -> None:
        source = branch("A0", 1.0, [(0.0, (0.0, 0.0, 0.0)), (2.0, (0.0, 0.0, 0.0))])
        result = solve_finite_difference_kg(
            source,
            time_slices=(0.0, 2.0),
            spatial_points=SPATIAL_POINTS_LINE_3,
            mass=0.5,
            boundary="reflecting",
            max_courant=0.9,
        )
        self.assertGreater(result.courant_number, 0.9)
        self.assertGreater(result.substeps_per_interval[0], 1)
        self.assertLess(result.effective_time_step, 2.0)

    def test_finite_difference_kg_local_error_control_refines_substeps(self) -> None:
        source = branch("A0", 1.0, [(0.0, (0.0, 0.0, 0.0)), (2.0, (0.0, 0.0, 0.0))])
        uncontrolled = solve_finite_difference_kg(
            source,
            time_slices=(0.0, 1.5),
            spatial_points=SPATIAL_POINTS_LINE_3,
            mass=0.5,
            boundary="reflecting",
            max_courant=2.0,
        )
        controlled = solve_finite_difference_kg(
            source,
            time_slices=(0.0, 1.5),
            spatial_points=SPATIAL_POINTS_LINE_3,
            mass=0.5,
            boundary="reflecting",
            max_courant=2.0,
            time_error_tolerance=1e-3,
        )
        self.assertGreaterEqual(controlled.substeps_per_interval[0], uncontrolled.substeps_per_interval[0])
        self.assertGreater(controlled.local_error_estimates[0], 0.0)
        self.assertEqual(controlled.time_error_tolerance, 1e-3)

    def test_finite_difference_kg_reflecting_boundary(self) -> None:
        source = branch("A0", 1.0, [(0.0, (0.0, 0.0, 0.0)), (2.0, (0.0, 0.0, 0.0))])
        result = solve_finite_difference_kg(
            source,
            time_slices=(0.0, 0.5, 1.0),
            spatial_points=SPATIAL_POINTS_LINE_3,
            mass=0.5,
            boundary="reflecting",
        )
        self.assertEqual(len(result.field_values), 3)

    def test_finite_difference_kg_adaptive_mesh_refinement_increases_grid_points(self) -> None:
        source = branch("A0", 1.0, [(0.0, (0.0, 0.0, 0.0)), (2.0, (0.0, 0.0, 0.0))])
        result = solve_finite_difference_kg(
            source,
            time_slices=(0.0, 0.5, 1.0),
            spatial_points=SPATIAL_POINTS_WIDE,
            mass=0.5,
            adaptive_mesh_refinement_rounds=1,
        )
        self.assertGreater(len(result.spatial_points), 3)
        self.assertEqual(result.refinement_rounds, 1)

    def test_finite_difference_kg_metric_adaptive_remeshing_prefers_weighted_axis(self) -> None:
        source = branch("A0", 1.0, [(0.0, (0.0, 0.0, 0.0)), (2.0, (0.0, 0.0, 0.0))])
        points = tuple(
            (x, y, 0.0)
            for y in (-2.0, 0.0, 2.0)
            for x in (-2.0, 0.0, 2.0)
        )
        result = solve_finite_difference_kg(
            source,
            time_slices=(0.0, 0.5, 1.0),
            spatial_points=points,
            mass=0.5,
            adaptive_mesh_refinement_rounds=1,
            adaptive_mesh_radius_factor=0.75,
            remeshing_metric=((4.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        )
        xs = sorted({point[0] for point in result.spatial_points})
        ys = sorted({point[1] for point in result.spatial_points})
        self.assertGreater(len(xs), len(ys))
        self.assertEqual(result.remeshing_metric[0][0], 4.0)

    def test_finite_difference_kg_fourth_order_stencil_reports_metadata(self) -> None:
        source = branch("A0", 1.0, [(0.0, (0.0, 0.0, 0.0)), (2.0, (0.0, 0.0, 0.0))])
        result = solve_finite_difference_kg(
            source,
            time_slices=(0.0, 0.5, 1.0),
            spatial_points=SPATIAL_POINTS_LINE_5,
            mass=0.5,
            stencil_order=4,
        )
        self.assertEqual(result.stencil_order, 4)
        self.assertEqual(len(result.field_values[0]), 5)

    def test_finite_difference_kg_level_set_boundary_masks_outside_points(self) -> None:
        source = branch("A0", 1.0, [(0.0, (0.0, 0.0, 0.0)), (2.0, (0.0, 0.0, 0.0))])
        points = tuple(
            (x, y, z)
            for z in (0.0,)
            for y in (-1.0, 0.0, 1.0)
            for x in (-1.0, 0.0, 1.0)
        )
        result = solve_finite_difference_kg(
            source,
            time_slices=(0.0, 0.25, 0.5),
            spatial_points=points,
            mass=0.5,
            boundary_level_set=lambda p: p[0] * p[0] + p[1] * p[1] - 0.6 * 0.6,
        )
        self.assertEqual(result.boundary_geometry, "level_set")
        self.assertLess(sum(result.active_point_mask), len(points))
        for active, value in zip(result.active_point_mask, result.field_values[-1]):
            if not active:
                self.assertAlmostEqual(value, 0.0)

    def test_finite_difference_kg_cut_cell_boundary_reports_flux_geometry(self) -> None:
        source = branch("A0", 1.0, [(0.0, (0.0, 0.0, 0.0)), (2.0, (0.0, 0.0, 0.0))])
        points = tuple(
            (x, y, 0.0)
            for y in (-1.0, 0.0, 1.0)
            for x in (-1.0, 0.0, 1.0)
        )
        result = solve_finite_difference_kg(
            source,
            time_slices=(0.0, 0.25, 0.5),
            spatial_points=points,
            mass=0.5,
            boundary_level_set=lambda p: p[0] * p[0] + p[1] * p[1] - 0.85 * 0.85,
        )
        self.assertTrue(any(0.0 < fraction < 1.0 for fraction in result.cell_volume_fractions))
        self.assertTrue(
            any(
                0.0 < aperture < 1.0
                for apertures in result.face_apertures
                for aperture in apertures
            )
        )

    def test_physical_lattice_dynamics_leapfrog(self) -> None:
        source = branch("A0", 1.0, [(0.0, (0.0, 0.0, 0.0)), (3.0, (0.0, 0.0, 0.0))])
        result = solve_physical_lattice_dynamics(
            source,
            time_slices=(0.0, 1.0, 2.0, 3.0),
            spatial_points=((1.0, 0.0, 0.0), (2.0, 0.0, 0.0)),
            mass=0.5,
            propagation="instantaneous",
            method="leapfrog",
        )
        self.assertEqual(len(result.lattices), 4)
        self.assertEqual(result.method, "leapfrog")
        self.assertGreater(result.time_step, 0.0)

    def test_physical_lattice_dynamics_with_damping(self) -> None:
        # 움직이는 source 를 써서 field 에 시간 변화를 만든다.
        source = branch("A0", 1.0, [(0.0, (0.0, 0.0, 0.0)), (2.5, (1.0, 0.0, 0.0)), (5.0, (0.0, 0.0, 0.0))])
        slices = tuple(i * 0.5 for i in range(11))
        undamped = solve_physical_lattice_dynamics(
            source,
            time_slices=slices,
            spatial_points=((2.0, 0.0, 0.0), (3.0, 0.0, 0.0)),
            mass=0.5,
            propagation="instantaneous",
            damping_rate=0.0,
        )
        damped = solve_physical_lattice_dynamics(
            source,
            time_slices=slices,
            spatial_points=((2.0, 0.0, 0.0), (3.0, 0.0, 0.0)),
            mass=0.5,
            propagation="instantaneous",
            damping_rate=2.0,
        )
        last_undamped = undamped.lattices[-1].samples[0].value
        last_damped = damped.lattices[-1].samples[0].value
        self.assertNotAlmostEqual(last_undamped, last_damped)

