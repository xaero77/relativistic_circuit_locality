"""스칼라장 궤적 및 위상 도우미에 대한 회귀 테스트."""

import unittest
from math import asinh, exp, pi

from relativistic_circuit_locality.scalar_field import (
    BranchPath,
    CatModeState,
    CompositeBranch,
    GaussianModeState,
    GeneralGaussianState,
    ModeSuperpositionState,
    TrajectoryPoint,
    compute_proper_time_worldline,
    compute_tensor_mediated_phase_matrix,
    compute_renormalized_phase_matrix,
    compute_decoherence_model,
    compute_multi_body_correlation,
    evolve_relativistic_backreaction,
    compute_entanglement_measures,
    compute_mode_occupation_distribution,
    solve_finite_difference_kg,
    solve_physical_lattice_dynamics,
    compute_lebedev_displacement_amplitudes,
    SplineBranchPath,
    compute_spline_branch_phase_matrix,
    compute_fock_space_evolution,
    compute_adaptive_phase_integral,
    compute_richardson_extrapolated_phase,
    interpolate_field_lattice,
    compute_running_coupling_phase_matrix,
    validate_symbolic_bookkeeping,
    compute_adaptive_continuum_displacement_amplitudes,
    compute_anisotropic_sampled_spacetime_phase,
    analyze_branch_pair_coherent_overlap,
    analyze_branch_pair_coherent_state,
    analyze_branch_pair_phase,
    analyze_phase_decomposition,
    analyze_wavepacket_phase_decomposition,
    compute_branch_displacement_amplitudes,
    compute_branch_phase_matrix,
    compute_branch_pair_displacements,
    compute_composite_phase_matrix,
    compute_continuum_displacement_amplitudes,
    compute_closest_approach,
    compute_displacement_operator_phase,
    compute_entanglement_phase,
    compute_extrapolated_continuum_displacement_amplitudes,
    compute_certified_spectral_displacement_amplitudes,
    compute_high_order_spectral_displacement_amplitudes,
    compute_provable_spectral_control,
    evaluate_retarded_green_function,
    estimate_spectral_continuum_error_bound,
    estimate_spectral_convergence,
    estimate_continuum_displacement_amplitudes,
    compute_mediated_phase_matrix,
    compute_phi_rs_samples,
    compute_sampled_spacetime_phase,
    compute_split_continuum_displacement_amplitudes,
    compute_wavepacket_phase_matrix,
    compare_cat_mode_states,
    compare_general_gaussian_states,
    compare_gaussian_mode_states,
    compare_coherent_states,
    compare_superposition_states,
    evolve_backreacted_branch,
    evolve_coherent_state,
    field_mediation_intervals,
    is_field_mediated,
    sample_branch_field,
    sample_mediator_field,
    solve_fft_lattice_evolution,
    solve_dynamic_boundary_lattice,
    solve_surrogate_4d_field_equation,
    solve_mediator_self_consistent_backreaction,
    solve_effective_field_equation_backreaction,
    solve_large_scale_pde_surrogate,
    solve_gauge_gravity_field_system,
    solve_full_qft_surrogate,
    solve_nonlinear_backreaction,
    solve_spectral_lattice,
    solve_self_consistent_backreaction,
    solve_coupled_backreaction,
    solve_field_lattice_dynamics,
    solve_field_lattice,
    iterate_backreaction,
    solve_multiscale_field_lattice,
    tomograph_cat_mode_state,
    tomograph_general_gaussian_state,
    tomograph_multimode_family,
    summarize_symbolic_multimode_bookkeeping,
    verify_multimode_analytic_identities,
    compile_multimode_state_transform,
    compile_comprehensive_multimode_bookkeeping,
    compile_appendix_d_bookkeeping,
    analyze_sampled_phase_decomposition,
    compute_generalized_wavepacket_phase_matrix,
    evaluate_microcausality_commutator,
    solve_reference_pde_control,
    solve_high_fidelity_pde_bundle,
    compile_universal_state_family,
    compile_complete_state_family_bundle,
    solve_exact_mediator_surrogate,
    solve_exact_dynamics_surrogate,
    close_current_limitations,
    close_research_grade_limitations,
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

    def test_kg_retarded_reduces_to_retarded_when_massless(self) -> None:
        source = branch("A0", 1.0, [(0.0, (0.0, 0.0, 0.0)), (1.0, (0.0, 0.0, 0.0)), (2.0, (0.0, 0.0, 0.0))])
        target = branch("B0", 1.0, [(0.0, (1.0, 0.0, 0.0)), (1.0, (1.0, 0.0, 0.0)), (2.0, (1.0, 0.0, 0.0))])
        retarded = compute_branch_phase_matrix((source,), (target,), mass=0.0, propagation="retarded")
        kg_retarded = compute_branch_phase_matrix((source,), (target,), mass=0.0, propagation="kg_retarded")
        self.assertAlmostEqual(retarded[0][0], kg_retarded[0][0], places=12)

    def test_kg_retarded_is_sensitive_to_tail_history_for_massive_field(self) -> None:
        short_source = branch("A0", 1.0, [(0.0, (0.0, 0.0, 0.0)), (1.5, (0.0, 0.0, 0.0)), (3.0, (0.0, 0.0, 0.0))])
        long_source = branch("A0", 1.0, [(-2.0, (0.0, 0.0, 0.0)), (0.0, (0.0, 0.0, 0.0)), (1.5, (0.0, 0.0, 0.0)), (3.0, (0.0, 0.0, 0.0))])
        target = branch("B0", 1.0, [(0.0, (1.0, 0.0, 0.0)), (1.5, (1.0, 0.0, 0.0)), (3.0, (1.0, 0.0, 0.0))])
        short_history = compute_branch_phase_matrix((short_source,), (target,), mass=0.5, propagation="kg_retarded")
        long_history = compute_branch_phase_matrix((long_source,), (target,), mass=0.5, propagation="kg_retarded")
        self.assertGreater(abs(long_history[0][0]), abs(short_history[0][0]))

    def test_sample_branch_field_returns_requested_samples(self) -> None:
        source = branch("A0", 1.0, [(0.0, (0.0, 0.0, 0.0)), (2.0, (0.0, 0.0, 0.0))])
        samples = sample_branch_field(
            source,
            ((0.5, (1.0, 0.0, 0.0)), (1.5, (1.0, 0.0, 0.0))),
            mass=0.0,
            propagation="retarded",
        )
        self.assertEqual(len(samples), 2)
        self.assertGreater(samples[1].value, 0.0)

    def test_phi_rs_samples_follow_target_worldline(self) -> None:
        source = branch("A0", 1.0, [(0.0, (0.0, 0.0, 0.0)), (2.0, (0.0, 0.0, 0.0))])
        target = branch("B0", 1.0, [(0.0, (1.0, 0.0, 0.0)), (2.0, (1.0, 0.0, 0.0))])
        samples = compute_phi_rs_samples(source, target, sample_count=3, mass=0.0, propagation="retarded")
        self.assertEqual(samples[0].position, (1.0, 0.0, 0.0))
        self.assertEqual(len(samples), 3)

    def test_sampled_spacetime_phase_matches_pointlike_limit_at_zero_width(self) -> None:
        left = branch("A0", 1.0, [(0.0, (-2.0, 0.0, 0.0)), (2.0, (-2.0, 0.0, 0.0))])
        right = branch("B0", 1.0, [(0.0, (2.0, 0.0, 0.0)), (2.0, (2.0, 0.0, 0.0))])
        pointlike = compute_branch_phase_matrix((left,), (right,), mass=0.5, propagation="kg_retarded")
        sampled = compute_sampled_spacetime_phase(left, right, mass=0.5, target_width=0.0, propagation="kg_retarded")
        self.assertAlmostEqual(sampled, pointlike[0][0])

    def test_sampled_spacetime_phase_changes_with_target_width(self) -> None:
        left = branch("A0", 1.0, [(0.0, (-1.0, 0.0, 0.0)), (2.0, (-1.0, 0.0, 0.0))])
        right = branch("B0", 1.0, [(0.0, (1.0, 0.0, 0.0)), (2.0, (1.0, 0.0, 0.0))])
        pointlike = compute_sampled_spacetime_phase(left, right, mass=0.5, target_width=0.0, propagation="kg_retarded")
        finite_width = compute_sampled_spacetime_phase(left, right, mass=0.5, target_width=0.4, propagation="kg_retarded")
        self.assertNotAlmostEqual(pointlike, finite_width)

    def test_anisotropic_sampled_phase_changes_with_width_tensor(self) -> None:
        left = branch("A0", 1.0, [(0.0, (-1.0, 0.0, 0.0)), (2.0, (-1.0, 0.0, 0.0))])
        right = branch("B0", 1.0, [(0.0, (1.0, 0.0, 0.0)), (2.0, (1.0, 0.0, 0.0))])
        isotropic_like = compute_anisotropic_sampled_spacetime_phase(
            left,
            right,
            mass=0.5,
            target_widths=(0.2, 0.2, 0.2),
            propagation="instantaneous",
        )
        anisotropic = compute_anisotropic_sampled_spacetime_phase(
            left,
            right,
            mass=0.5,
            target_widths=(0.2, 0.6, 0.1),
            propagation="instantaneous",
        )
        self.assertNotAlmostEqual(isotropic_like, anisotropic)

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

    def test_continuum_displacement_amplitudes_are_nonzero_for_static_source(self) -> None:
        source = branch("A0", 1.0, [(0.0, (0.0, 0.0, 0.0)), (2.0, (0.0, 0.0, 0.0))])
        amplitudes = compute_continuum_displacement_amplitudes((source,), field_mass=0.5, momentum_cutoff=1.0)
        self.assertEqual(len(amplitudes), 1)
        self.assertNotEqual(amplitudes[0], 0.0j)

    def test_adaptive_continuum_displacement_returns_value(self) -> None:
        source = branch("A0", 1.0, [(0.0, (0.0, 0.0, 0.0)), (2.0, (0.0, 0.0, 0.0))])
        amplitudes = compute_adaptive_continuum_displacement_amplitudes((source,), field_mass=0.5, momentum_cutoff=1.0)
        self.assertEqual(len(amplitudes), 1)
        self.assertNotEqual(amplitudes[0], 0.0j)

    def test_split_continuum_displacement_returns_value(self) -> None:
        source = branch("A0", 1.0, [(0.0, (0.0, 0.0, 0.0)), (2.0, (0.0, 0.0, 0.0))])
        amplitudes = compute_split_continuum_displacement_amplitudes((source,), field_mass=0.5, momentum_cutoff=1.0)
        self.assertEqual(len(amplitudes), 1)
        self.assertNotEqual(amplitudes[0], 0.0j)

    def test_continuum_displacement_error_estimate_is_finite(self) -> None:
        source = branch("A0", 1.0, [(0.0, (0.0, 0.0, 0.0)), (2.0, (0.0, 0.0, 0.0))])
        result = estimate_continuum_displacement_amplitudes((source,), field_mass=0.5, momentum_cutoff=1.0)
        self.assertEqual(len(result.amplitudes), 1)
        self.assertGreaterEqual(result.error_estimate, 0.0)

    def test_extrapolated_continuum_displacement_tracks_mode_counts(self) -> None:
        source = branch("A0", 1.0, [(0.0, (0.0, 0.0, 0.0)), (2.0, (0.0, 0.0, 0.0))])
        result = compute_extrapolated_continuum_displacement_amplitudes((source,), field_mass=0.5, momentum_cutoff=1.0)
        self.assertEqual(len(result.amplitudes), 1)
        self.assertGreaterEqual(len(result.mode_counts), 1)
        self.assertGreaterEqual(result.mode_counts[-1], result.mode_counts[0])

    def test_spectral_continuum_error_bound_is_finite(self) -> None:
        source = branch("A0", 1.0, [(0.0, (0.0, 0.0, 0.0)), (2.0, (0.0, 0.0, 0.0))])
        result = estimate_spectral_continuum_error_bound((source,), field_mass=0.5, momentum_cutoff=1.0)
        self.assertEqual(len(result.amplitudes), 1)
        self.assertGreaterEqual(result.absolute_bound, 0.0)
        self.assertGreaterEqual(result.relative_bound, 0.0)

    def test_spectral_convergence_tracks_mode_counts(self) -> None:
        source = branch("A0", 1.0, [(0.0, (0.0, 0.0, 0.0)), (2.0, (0.0, 0.0, 0.0))])
        result = estimate_spectral_convergence((source,), field_mass=0.5, momentum_cutoff=1.0)
        self.assertEqual(len(result.amplitudes), 1)
        self.assertGreaterEqual(len(result.mode_counts), 1)

    def test_certified_spectral_displacement_returns_certificate(self) -> None:
        source = branch("A0", 1.0, [(0.0, (0.0, 0.0, 0.0)), (2.0, (0.0, 0.0, 0.0))])
        result = compute_certified_spectral_displacement_amplitudes((source,), field_mass=0.5, momentum_cutoff=1.0)
        self.assertEqual(len(result.amplitudes), 1)
        self.assertGreaterEqual(result.certificate_error, 0.0)

    def test_high_order_spectral_displacement_reports_effective_order(self) -> None:
        source = branch("A0", 1.0, [(0.0, (0.0, 0.0, 0.0)), (2.0, (0.0, 0.0, 0.0))])
        result = compute_high_order_spectral_displacement_amplitudes((source,), field_mass=0.5, momentum_cutoff=1.0)
        self.assertEqual(len(result.certified.amplitudes), 1)
        self.assertGreaterEqual(result.effective_order, 1)

    def test_provable_spectral_control_reports_certificate(self) -> None:
        source = branch("A0", 1.0, [(0.0, (0.0, 0.0, 0.0)), (2.0, (0.0, 0.0, 0.0))])
        result = compute_provable_spectral_control((source,), field_mass=0.5, momentum_cutoff=1.0)
        self.assertEqual(len(result.high_order.certified.amplitudes), 1)
        self.assertGreaterEqual(result.strict_certificate_error, 0.0)

    def test_evaluate_retarded_green_function_returns_samples(self) -> None:
        source = branch("A0", 1.0, [(0.0, (0.0, 0.0, 0.0)), (2.0, (0.0, 0.0, 0.0))])
        result = evaluate_retarded_green_function(
            source,
            samples=((1.0, (1.0, 0.0, 0.0)),),
            mass=0.5,
        )
        self.assertEqual(len(result.samples), 1)

    def test_displacement_phase_vanishes_for_identical_profiles(self) -> None:
        profile = (1.0 + 2.0j, -0.5j)
        self.assertAlmostEqual(compute_displacement_operator_phase(profile, profile), 0.0)

    def test_compare_coherent_states_returns_unit_overlap_for_identical_profiles(self) -> None:
        profile = (1.0 + 0.5j, -0.25j)
        comparison = compare_coherent_states(profile, profile)
        self.assertAlmostEqual(comparison.vacuum_suppression, 1.0)
        self.assertAlmostEqual(abs(comparison.overlap), 1.0)

    def test_compare_gaussian_mode_states_reduces_overlap_with_covariance_mismatch(self) -> None:
        left = GaussianModeState(displacement=(1.0 + 0.0j,), covariance_diag=(0.5,))
        right = GaussianModeState(displacement=(1.0 + 0.0j,), covariance_diag=(2.0,))
        comparison = compare_gaussian_mode_states(left, right)
        self.assertLess(comparison.vacuum_suppression, 1.0)

    def test_compare_general_gaussian_states_supports_off_diagonal_covariance(self) -> None:
        left = GeneralGaussianState(displacement=(1.0 + 0.0j, 0.0j), covariance=((1.0, 0.2), (0.2, 1.5)))
        right = GeneralGaussianState(displacement=(1.0 + 0.0j, 0.1j), covariance=((1.2, 0.1), (0.1, 1.4)))
        comparison = compare_general_gaussian_states(left, right)
        self.assertLessEqual(abs(comparison.overlap), 1.0)

    def test_tomograph_general_gaussian_state_returns_mean_and_covariance(self) -> None:
        state = tomograph_general_gaussian_state(((1.0 + 0.0j, 0.0j), (0.0j, 1.0 + 0.0j)))
        self.assertEqual(len(state.displacement), 2)
        self.assertEqual(len(state.covariance), 2)

    def test_compare_superposition_states_returns_finite_overlap(self) -> None:
        left = ModeSuperpositionState(weights=(1.0 + 0.0j, 0.5 + 0.0j), components=((1.0 + 0.0j,), (0.0 + 0.0j,)))
        right = ModeSuperpositionState(weights=(1.0 + 0.0j,), components=((1.0 + 0.0j,),))
        overlap = compare_superposition_states(left, right)
        self.assertNotEqual(overlap, 0.0j)

    def test_compare_cat_mode_states_returns_finite_overlap(self) -> None:
        left = CatModeState(
            weights=(1.0 + 0.0j, 0.5 + 0.0j),
            components=(
                GaussianModeState(displacement=(1.0 + 0.0j,), covariance_diag=(1.0,)),
                GaussianModeState(displacement=(-1.0 + 0.0j,), covariance_diag=(1.0,)),
            ),
        )
        right = CatModeState(
            weights=(1.0 + 0.0j,),
            components=(GaussianModeState(displacement=(1.0 + 0.0j,), covariance_diag=(1.0,)),),
        )
        overlap = compare_cat_mode_states(left, right)
        self.assertNotEqual(overlap, 0.0j)

    def test_tomograph_cat_mode_state_builds_components(self) -> None:
        state = tomograph_cat_mode_state(((1.0 + 0.0j,), (-1.0 + 0.0j,)))
        self.assertEqual(len(state.components), 2)

    def test_tomograph_multimode_family_returns_phase_matrix(self) -> None:
        result = tomograph_multimode_family(
            (
                ((1.0 + 0.0j, 0.0j), (0.5 + 0.0j, 0.5 + 0.0j)),
                ((-1.0 + 0.0j, 0.0j), (-0.5 + 0.0j, -0.5 + 0.0j)),
            )
        )
        self.assertEqual(len(result.component_states), 2)
        self.assertEqual(len(result.relative_phase_matrix), 2)

    def test_symbolic_bookkeeping_returns_labels(self) -> None:
        result = summarize_symbolic_multimode_bookkeeping(
            (
                ("r0", ((1.0 + 0.0j,), (0.5 + 0.0j,))),
                ("r1", ((-1.0 + 0.0j,), (-0.5 + 0.0j,))),
            )
        )
        self.assertEqual(result.labels, ("r0", "r1"))
        self.assertEqual(len(result.relative_phase_matrix), 2)

    def test_verify_multimode_analytic_identities_returns_small_errors(self) -> None:
        result = verify_multimode_analytic_identities(
            (
                ("r0", ((1.0 + 0.0j,), (0.5 + 0.0j,))),
                ("r1", ((-1.0 + 0.0j,), (-0.5 + 0.0j,))),
            )
        )
        self.assertEqual(result.labels, ("r0", "r1"))
        self.assertGreaterEqual(result.phase_antisymmetry_error, 0.0)
        self.assertGreaterEqual(result.norm_consistency_error, 0.0)

    def test_compile_multimode_state_transform_returns_overlap_matrix(self) -> None:
        result = compile_multimode_state_transform(
            (
                ("r0", ((1.0 + 0.0j,), (0.5 + 0.0j,))),
                ("r1", ((-1.0 + 0.0j,), (-0.5 + 0.0j,))),
            )
        )
        self.assertEqual(result.labels, ("r0", "r1"))
        self.assertEqual(len(result.overlap_matrix), 2)

    def test_compile_comprehensive_multimode_bookkeeping_returns_all_views(self) -> None:
        result = compile_comprehensive_multimode_bookkeeping(
            (
                ("r0", ((1.0 + 0.0j,), (0.5 + 0.0j,))),
                ("r1", ((-1.0 + 0.0j,), (-0.5 + 0.0j,))),
            )
        )
        self.assertEqual(result.transform.labels, ("r0", "r1"))
        self.assertEqual(result.bookkeeping.labels, ("r0", "r1"))

    def test_compile_appendix_d_bookkeeping_returns_transform_and_multimode(self) -> None:
        result = compile_appendix_d_bookkeeping(
            (
                ("r0", ((1.0 + 0.0j,), (0.5 + 0.0j,))),
                ("r1", ((-1.0 + 0.0j,), (-0.5 + 0.0j,))),
            )
        )
        self.assertEqual(result.comprehensive.bookkeeping.labels, ("r0", "r1"))
        self.assertEqual(result.state_transform.labels, ("r0", "r1"))

    def test_analyze_sampled_phase_decomposition_returns_matrix(self) -> None:
        source = branch("A0", 1.0, [(0.0, (0.0, 0.0, 0.0)), (2.0, (0.0, 0.0, 0.0))])
        target = branch("B0", 1.0, [(0.0, (1.0, 0.0, 0.0)), (2.0, (1.0, 0.0, 0.0))])
        result = analyze_sampled_phase_decomposition((source,), (target,), mass=0.5, propagation="instantaneous")
        self.assertEqual(len(result.interaction_matrix), 1)

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

    def test_branch_pair_coherent_overlap_detects_branch_difference(self) -> None:
        left0 = branch("A0", 1.0, [(0.0, (-1.0, 0.0, 0.0)), (2.0, (-1.0, 0.0, 0.0))])
        right0 = branch("B0", 1.0, [(0.0, (1.0, 0.0, 0.0)), (2.0, (1.0, 0.0, 0.0))])
        left1 = branch("A1", 1.0, [(0.0, (-2.0, 0.0, 0.0)), (2.0, (-2.0, 0.0, 0.0))])
        right1 = branch("B1", 1.0, [(0.0, (2.0, 0.0, 0.0)), (2.0, (2.0, 0.0, 0.0))])
        momenta = ((0.0, 0.0, 0.5), (0.5, 0.0, 0.0))
        comparison = analyze_branch_pair_coherent_overlap(
            (left0, right0),
            (left1, right1),
            momenta,
            field_mass=0.5,
            source_width_a=0.2,
            source_width_b=0.2,
        )
        self.assertLess(abs(comparison.overlap), 1.0)

    def test_mediated_phase_matrix_supports_gravity_sign_convention(self) -> None:
        left = branch("A0", -2.0, [(0.0, (-1.0, 0.0, 0.0)), (2.0, (-1.0, 0.0, 0.0))])
        right = branch("B0", 3.0, [(0.0, (1.0, 0.0, 0.0)), (2.0, (1.0, 0.0, 0.0))])
        vector = compute_mediated_phase_matrix((left,), (right,), mass=0.5, mediator="vector", propagation="instantaneous")
        gravity = compute_mediated_phase_matrix((left,), (right,), mass=0.5, mediator="gravity", propagation="instantaneous")
        self.assertGreater(abs(gravity[0][0]), 0.0)
        self.assertAlmostEqual(abs(gravity[0][0]), abs(vector[0][0]))
        self.assertNotAlmostEqual(gravity[0][0], vector[0][0])

    def test_sample_mediator_field_supports_gravity_mode(self) -> None:
        source = branch("A0", -2.0, [(0.0, (0.0, 0.0, 0.0)), (2.0, (0.0, 0.0, 0.0))])
        samples = sample_mediator_field(source, ((1.5, (1.0, 0.0, 0.0)),), mass=0.5, mediator="gravity", propagation="instantaneous")
        self.assertGreater(samples[0].value, 0.0)

    def test_solve_field_lattice_returns_full_grid(self) -> None:
        source = branch("A0", 1.0, [(0.0, (0.0, 0.0, 0.0)), (2.0, (0.0, 0.0, 0.0))])
        lattice = solve_field_lattice(
            source,
            time_slices=(0.0, 1.0),
            spatial_points=((1.0, 0.0, 0.0), (2.0, 0.0, 0.0)),
            mass=0.5,
            propagation="instantaneous",
        )
        self.assertEqual(len(lattice.samples), 4)

    def test_solve_field_lattice_dynamics_returns_time_slices(self) -> None:
        source = branch("A0", 1.0, [(0.0, (0.0, 0.0, 0.0)), (2.0, (0.0, 0.0, 0.0))])
        evolution = solve_field_lattice_dynamics(
            source,
            time_slices=(0.0, 1.0, 2.0),
            spatial_points=((1.0, 0.0, 0.0), (2.0, 0.0, 0.0), (3.0, 0.0, 0.0)),
            mass=0.5,
            propagation="instantaneous",
        )
        self.assertEqual(len(evolution.lattices), 3)

    def test_solve_multiscale_field_lattice_refines_spatial_grid(self) -> None:
        source = branch("A0", 1.0, [(0.0, (0.0, 0.0, 0.0)), (2.0, (0.0, 0.0, 0.0))])
        evolution = solve_multiscale_field_lattice(
            source,
            time_slices=(0.0, 1.0),
            spatial_points=((1.0, 0.0, 0.0), (3.0, 0.0, 0.0)),
            mass=0.5,
            propagation="instantaneous",
            refinement_levels=3,
        )
        self.assertEqual(len(evolution.levels), 3)
        self.assertGreater(evolution.spatial_point_counts[-1], evolution.spatial_point_counts[0])

    def test_solve_spectral_lattice_honors_dirichlet_boundary(self) -> None:
        source = branch("A0", 1.0, [(0.0, (0.0, 0.0, 0.0)), (2.0, (0.0, 0.0, 0.0))])
        result = solve_spectral_lattice(
            source,
            time_slices=(0.0,),
            spatial_points=((1.0, 0.0, 0.0), (2.0, 0.0, 0.0), (3.0, 0.0, 0.0)),
            mass=0.5,
            propagation="instantaneous",
            boundary_condition="dirichlet",
        )
        self.assertEqual(result.boundary_condition, "dirichlet")
        self.assertEqual(result.lattice.samples[0].value, 0.0)
        self.assertEqual(result.lattice.samples[-1].value, 0.0)

    def test_solve_dynamic_boundary_lattice_tracks_schedule(self) -> None:
        source = branch("A0", 1.0, [(0.0, (0.0, 0.0, 0.0)), (2.0, (0.0, 0.0, 0.0))])
        result = solve_dynamic_boundary_lattice(
            source,
            time_slices=(0.0, 1.0),
            spatial_points=((1.0, 0.0, 0.0), (2.0, 0.0, 0.0), (3.0, 0.0, 0.0)),
            mass=0.5,
            boundary_schedule=("dirichlet", "periodic"),
            propagation="instantaneous",
        )
        self.assertEqual(result.boundary_schedule, ("dirichlet", "periodic"))
        self.assertEqual(len(result.slices), 2)

    def test_solve_fft_lattice_evolution_returns_damping_profile(self) -> None:
        source = branch("A0", 1.0, [(0.0, (0.0, 0.0, 0.0)), (2.0, (0.0, 0.0, 0.0))])
        result = solve_fft_lattice_evolution(
            source,
            time_slices=(0.0, 1.0),
            spatial_points=((1.0, 0.0, 0.0), (2.0, 0.0, 0.0), (3.0, 0.0, 0.0)),
            mass=0.5,
            boundary_schedule=("open", "periodic"),
            propagation="instantaneous",
        )
        self.assertEqual(len(result.slices), 2)
        self.assertEqual(len(result.damping_profile), 2)

    def test_solve_surrogate_4d_field_equation_reports_sample_count(self) -> None:
        source = branch("A0", 1.0, [(0.0, (0.0, 0.0, 0.0)), (2.0, (0.0, 0.0, 0.0))])
        result = solve_surrogate_4d_field_equation(
            source,
            time_slices=(0.0, 1.0),
            spatial_points=((1.0, 0.0, 0.0), (2.0, 0.0, 0.0)),
            mass=0.5,
            boundary_schedule=("open", "periodic"),
            propagation="instantaneous",
        )
        self.assertGreaterEqual(result.total_sample_count, 1)

    def test_solve_large_scale_pde_surrogate_reports_grid_points(self) -> None:
        source = branch("A0", 1.0, [(0.0, (0.0, 0.0, 0.0)), (2.0, (0.0, 0.0, 0.0))])
        result = solve_large_scale_pde_surrogate(
            source,
            time_slices=(0.0, 1.0),
            spatial_points=((1.0, 0.0, 0.0), (2.0, 0.0, 0.0)),
            mass=0.5,
            boundary_schedule=("open", "periodic"),
            propagation="instantaneous",
        )
        self.assertGreaterEqual(result.total_grid_points, 1)

    def test_compute_generalized_wavepacket_phase_matrix_accepts_tensor_widths(self) -> None:
        source = branch("A0", 1.0, [(0.0, (0.0, 0.0, 0.0)), (2.0, (0.0, 0.0, 0.0))])
        target = branch("B0", 1.0, [(0.0, (1.0, 0.0, 0.0)), (2.0, (1.0, 0.0, 0.0))])
        result = compute_generalized_wavepacket_phase_matrix(
            (source,),
            (target,),
            widths_a=((0.2, 0.3, 0.4),),
            widths_b=((0.3, 0.4, 0.5),),
            mass=0.5,
            propagation="instantaneous",
        )
        self.assertEqual(len(result.matrix), 1)

    def test_backreacted_branch_shifts_target_positions(self) -> None:
        source = branch("A0", 1.0, [(0.0, (0.0, 0.0, 0.0)), (2.0, (0.0, 0.0, 0.0))])
        target = branch("B0", 1.0, [(0.0, (1.0, 0.0, 0.0)), (2.0, (1.0, 0.0, 0.0))])
        updated = evolve_backreacted_branch(source, target, mass=0.5, propagation="instantaneous", response_strength=0.1)
        self.assertNotEqual(updated.points[0].position, target.points[0].position)

    def test_iterate_backreaction_accumulates_updates(self) -> None:
        source = branch("A0", 1.0, [(0.0, (0.0, 0.0, 0.0)), (2.0, (0.0, 0.0, 0.0))])
        target = branch("B0", 1.0, [(0.0, (1.0, 0.0, 0.0)), (2.0, (1.0, 0.0, 0.0))])
        once = evolve_backreacted_branch(source, target, mass=0.5, propagation="instantaneous", response_strength=0.05)
        repeated = iterate_backreaction(source, target, iterations=3, mass=0.5, propagation="instantaneous", response_strength=0.05)
        self.assertNotEqual(once.points[0].position, repeated.points[0].position)

    def test_solve_coupled_backreaction_updates_both_branches(self) -> None:
        source = branch("A0", 1.0, [(0.0, (-1.0, 0.0, 0.0)), (2.0, (-1.0, 0.0, 0.0))])
        target = branch("B0", 1.0, [(0.0, (1.0, 0.0, 0.0)), (2.0, (1.0, 0.0, 0.0))])
        result = solve_coupled_backreaction(
            source,
            target,
            iterations=2,
            mass=0.5,
            propagation="instantaneous",
            response_strength=0.05,
        )
        self.assertEqual(result.iterations, 2)
        self.assertNotEqual(result.source.points[0].position, source.points[0].position)
        self.assertNotEqual(result.target.points[0].position, target.points[0].position)

    def test_solve_nonlinear_backreaction_reports_update_norm(self) -> None:
        source = branch("A0", 1.0, [(0.0, (-1.0, 0.0, 0.0)), (2.0, (-1.0, 0.0, 0.0))])
        target = branch("B0", 1.0, [(0.0, (1.0, 0.0, 0.0)), (2.0, (1.0, 0.0, 0.0))])
        result = solve_nonlinear_backreaction(
            source,
            target,
            iterations=2,
            mass=0.5,
            propagation="instantaneous",
            response_strength=0.05,
        )
        self.assertEqual(result.iterations, 2)
        self.assertGreaterEqual(result.max_update_norm, 0.0)

    def test_solve_self_consistent_backreaction_reports_residual(self) -> None:
        source = branch("A0", 1.0, [(0.0, (-1.0, 0.0, 0.0)), (2.0, (-1.0, 0.0, 0.0))])
        target = branch("B0", 1.0, [(0.0, (1.0, 0.0, 0.0)), (2.0, (1.0, 0.0, 0.0))])
        result = solve_self_consistent_backreaction(
            source,
            target,
            max_iterations=3,
            mass=0.5,
            propagation="instantaneous",
            response_strength=0.05,
        )
        self.assertGreaterEqual(result.iterations, 1)
        self.assertGreaterEqual(result.residual, 0.0)

    def test_solve_mediator_self_consistent_backreaction_returns_phase_shift(self) -> None:
        source = branch("A0", 1.0, [(0.0, (-1.0, 0.0, 0.0)), (2.0, (-1.0, 0.0, 0.0))])
        target = branch("B0", 1.0, [(0.0, (1.0, 0.0, 0.0)), (2.0, (1.0, 0.0, 0.0))])
        result = solve_mediator_self_consistent_backreaction(
            source,
            target,
            mediator="gravity",
            max_iterations=2,
            mass=0.5,
            propagation="instantaneous",
        )
        self.assertEqual(result.mediator, "gravity")
        self.assertNotEqual(result.phase_shift, 0.0)

    def test_solve_effective_field_equation_backreaction_returns_mediator_result(self) -> None:
        source = branch("A0", 1.0, [(0.0, (-1.0, 0.0, 0.0)), (2.0, (-1.0, 0.0, 0.0))])
        target = branch("B0", 1.0, [(0.0, (1.0, 0.0, 0.0)), (2.0, (1.0, 0.0, 0.0))])
        result = solve_effective_field_equation_backreaction(
            source,
            target,
            mediator="scalar",
            time_slices=(0.0, 1.0),
            spatial_points=((1.0, 0.0, 0.0), (2.0, 0.0, 0.0)),
            boundary_schedule=("open", "periodic"),
            max_iterations=2,
            mass=0.5,
            propagation="instantaneous",
        )
        self.assertEqual(result.mediator, "scalar")
        self.assertEqual(result.backreaction.mediator, "scalar")

    def test_solve_gauge_gravity_field_system_returns_all_mediators(self) -> None:
        source = branch("A0", 1.0, [(0.0, (-1.0, 0.0, 0.0)), (2.0, (-1.0, 0.0, 0.0))])
        target = branch("B0", 1.0, [(0.0, (1.0, 0.0, 0.0)), (2.0, (1.0, 0.0, 0.0))])
        result = solve_gauge_gravity_field_system(
            source,
            target,
            time_slices=(0.0, 1.0),
            spatial_points=((1.0, 0.0, 0.0), (2.0, 0.0, 0.0)),
            boundary_schedule=("open", "periodic"),
            max_iterations=2,
            mass=0.5,
            propagation="instantaneous",
        )
        self.assertEqual(result.scalar_result.mediator, "scalar")
        self.assertEqual(result.vector_result.mediator, "vector")
        self.assertEqual(result.gravity_result.mediator, "gravity")

    def test_evaluate_microcausality_commutator_reports_bound(self) -> None:
        source = branch("A0", 1.0, [(0.0, (-2.0, 0.0, 0.0)), (2.0, (-2.0, 0.0, 0.0))])
        target = branch("B0", 1.0, [(0.0, (2.0, 0.0, 0.0)), (2.0, (2.0, 0.0, 0.0))])
        result = evaluate_microcausality_commutator((source,), (target,))
        self.assertTrue(result.bounded)
        self.assertTrue(all(abs(value) <= 1e-9 for value in result.interval_commutators))

    def test_evaluate_microcausality_commutator_detects_timelike_overlap(self) -> None:
        source = branch("A0", 1.0, [(0.0, (0.0, 0.0, 0.0)), (2.0, (0.0, 0.0, 0.0))])
        target = branch("B0", 1.0, [(0.0, (0.0, 0.0, 0.0)), (2.0, (0.0, 0.0, 0.0))])
        result = evaluate_microcausality_commutator((source,), (target,), mass=0.5)
        self.assertFalse(result.bounded)
        self.assertGreater(max(result.interval_commutators), 1e-6)

    def test_solve_full_qft_surrogate_returns_nested_results(self) -> None:
        source = branch("A0", 1.0, [(0.0, (-1.0, 0.0, 0.0)), (2.0, (-1.0, 0.0, 0.0))])
        target = branch("B0", 1.0, [(0.0, (1.0, 0.0, 0.0)), (2.0, (1.0, 0.0, 0.0))])
        result = solve_full_qft_surrogate(
            source,
            target,
            time_slices=(0.0, 1.0),
            spatial_points=((1.0, 0.0, 0.0), (2.0, 0.0, 0.0)),
            boundary_schedule=("open", "periodic"),
            max_iterations=2,
            mass=0.5,
            propagation="instantaneous",
        )
        self.assertGreaterEqual(result.pde.total_grid_points, 1)
        self.assertTrue(result.microcausality.bounded)

    def test_solve_reference_pde_control_returns_certificate(self) -> None:
        source = branch("A0", 1.0, [(0.0, (-1.0, 0.0, 0.0)), (2.0, (-1.0, 0.0, 0.0))])
        target = branch("B0", 1.0, [(0.0, (1.0, 0.0, 0.0)), (2.0, (1.0, 0.0, 0.0))])
        result = solve_reference_pde_control(
            source,
            target,
            time_slices=(0.0, 1.0),
            spatial_points=((1.0, 0.0, 0.0), (2.0, 0.0, 0.0)),
            boundary_schedule=("open", "periodic"),
            mass=0.5,
            propagation="instantaneous",
        )
        self.assertGreaterEqual(result.effective_grid_points, 1)
        self.assertGreaterEqual(result.spectral_control.strict_certificate_error, 0.0)

    def test_compile_universal_state_family_returns_phase_matrix(self) -> None:
        source = branch("A0", 1.0, [(0.0, (0.0, 0.0, 0.0)), (2.0, (0.0, 0.0, 0.0))])
        target = branch("B0", 1.0, [(0.0, (1.0, 0.0, 0.0)), (2.0, (1.0, 0.0, 0.0))])
        result = compile_universal_state_family(
            (source,),
            (target,),
            widths_a=((0.2, 0.3, 0.4),),
            widths_b=((0.3, 0.4, 0.5),),
            labeled_mode_samples=(
                ("A0", ((1.0 + 0.0j, 0.2 + 0.0j),)),
                ("B0", ((0.8 + 0.0j, 0.1 + 0.0j),)),
            ),
            mass=0.5,
            propagation="instantaneous",
        )
        self.assertEqual(len(result.generalized_wavepacket.matrix), 1)
        self.assertEqual(len(result.overlap_phase_matrix), 2)

    def test_solve_exact_mediator_surrogate_reports_consistency(self) -> None:
        source = branch("A0", 1.0, [(0.0, (-1.0, 0.0, 0.0)), (2.0, (-1.0, 0.0, 0.0))])
        target = branch("B0", 1.0, [(0.0, (1.0, 0.0, 0.0)), (2.0, (1.0, 0.0, 0.0))])
        result = solve_exact_mediator_surrogate(
            source,
            target,
            time_slices=(0.0, 1.0),
            spatial_points=((1.0, 0.0, 0.0), (2.0, 0.0, 0.0)),
            boundary_schedule=("open", "periodic"),
            mass=0.5,
            propagation="instantaneous",
        )
        self.assertTrue(result.consistent)
        self.assertTrue(result.microcausality.bounded)

    def test_close_current_limitations_returns_all_bundles(self) -> None:
        source = branch("A0", 1.0, [(0.0, (-1.0, 0.0, 0.0)), (2.0, (-1.0, 0.0, 0.0))])
        target = branch("B0", 1.0, [(0.0, (1.0, 0.0, 0.0)), (2.0, (1.0, 0.0, 0.0))])
        result = close_current_limitations(
            source,
            target,
            time_slices=(0.0, 1.0),
            spatial_points=((1.0, 0.0, 0.0), (2.0, 0.0, 0.0)),
            boundary_schedule=("open", "periodic"),
            widths_a=((0.2, 0.3, 0.4),),
            widths_b=((0.3, 0.4, 0.5),),
            labeled_mode_samples=(
                ("A0", ((1.0 + 0.0j, 0.2 + 0.0j),)),
                ("B0", ((0.8 + 0.0j, 0.1 + 0.0j),)),
            ),
            mass=0.5,
            propagation="instantaneous",
        )
        self.assertGreaterEqual(result.reference_pde.effective_grid_points, 1)
        self.assertEqual(len(result.universal_state.overlap_phase_matrix), 2)
        self.assertTrue(result.exact_mediator.consistent)

    def test_solve_high_fidelity_pde_bundle_returns_score(self) -> None:
        source = branch("A0", 1.0, [(0.0, (-1.0, 0.0, 0.0)), (2.0, (-1.0, 0.0, 0.0))])
        target = branch("B0", 1.0, [(0.0, (1.0, 0.0, 0.0)), (2.0, (1.0, 0.0, 0.0))])
        result = solve_high_fidelity_pde_bundle(
            source,
            target,
            time_slices=(0.0, 1.0),
            spatial_points=((1.0, 0.0, 0.0), (2.0, 0.0, 0.0)),
            boundary_schedule=("open", "periodic"),
            widths_a=((0.2, 0.3, 0.4),),
            widths_b=((0.3, 0.4, 0.5),),
            labeled_mode_samples=(("A0", ((1.0 + 0.0j, 0.2 + 0.0j),)), ("B0", ((0.8 + 0.0j, 0.1 + 0.0j),))),
            mass=0.5,
            propagation="retarded",
        )
        self.assertGreaterEqual(result.refined_pde.total_grid_points, 1)
        self.assertGreaterEqual(result.fidelity_score, 0.0)

    def test_compile_complete_state_family_bundle_returns_multimode(self) -> None:
        source = branch("A0", 1.0, [(0.0, (0.0, 0.0, 0.0)), (2.0, (0.0, 0.0, 0.0))])
        target = branch("B0", 1.0, [(0.0, (1.0, 0.0, 0.0)), (2.0, (1.0, 0.0, 0.0))])
        result = compile_complete_state_family_bundle(
            (source,),
            (target,),
            widths_a=((0.2, 0.3, 0.4),),
            widths_b=((0.3, 0.4, 0.5),),
            labeled_mode_samples=(("A0", ((1.0 + 0.0j, 0.2 + 0.0j),)), ("B0", ((0.8 + 0.0j, 0.1 + 0.0j),))),
            mass=0.5,
            propagation="retarded",
        )
        self.assertEqual(len(result.multimode.relative_phase_matrix), 2)

    def test_solve_exact_dynamics_surrogate_returns_retarded_samples(self) -> None:
        source = branch("A0", 1.0, [(0.0, (-1.0, 0.0, 0.0)), (2.0, (-1.0, 0.0, 0.0))])
        target = branch("B0", 1.0, [(0.0, (1.0, 0.0, 0.0)), (2.0, (1.0, 0.0, 0.0))])
        result = solve_exact_dynamics_surrogate(
            source,
            target,
            time_slices=(0.0, 1.0),
            spatial_points=((1.0, 0.0, 0.0), (2.0, 0.0, 0.0)),
            boundary_schedule=("open", "periodic"),
            mass=0.5,
            propagation="retarded",
        )
        self.assertGreaterEqual(len(result.retarded_green.samples), 1)
        self.assertTrue(result.exact_mediator.consistent)

    def test_close_research_grade_limitations_returns_all_layers(self) -> None:
        source = branch("A0", 1.0, [(0.0, (-1.0, 0.0, 0.0)), (2.0, (-1.0, 0.0, 0.0))])
        target = branch("B0", 1.0, [(0.0, (1.0, 0.0, 0.0)), (2.0, (1.0, 0.0, 0.0))])
        result = close_research_grade_limitations(
            source,
            target,
            time_slices=(0.0, 1.0),
            spatial_points=((1.0, 0.0, 0.0), (2.0, 0.0, 0.0)),
            boundary_schedule=("open", "periodic"),
            widths_a=((0.2, 0.3, 0.4),),
            widths_b=((0.3, 0.4, 0.5),),
            labeled_mode_samples=(("A0", ((1.0 + 0.0j, 0.2 + 0.0j),)), ("B0", ((0.8 + 0.0j, 0.1 + 0.0j),))),
            mass=0.5,
            propagation="retarded",
        )
        self.assertGreaterEqual(result.high_fidelity_pde.refined_pde.total_grid_points, 1)
        self.assertEqual(len(result.complete_state_family.multimode.relative_phase_matrix), 2)
        self.assertGreaterEqual(len(result.exact_dynamics.retarded_green.samples), 1)

    def test_composite_phase_matrix_sums_component_pairs(self) -> None:
        a0 = branch("A0", 1.0, [(0.0, (-2.0, 0.0, 0.0)), (2.0, (-2.0, 0.0, 0.0))])
        a1 = branch("A1", 1.0, [(0.0, (-1.0, 0.0, 0.0)), (2.0, (-1.0, 0.0, 0.0))])
        b0 = branch("B0", 1.0, [(0.0, (1.0, 0.0, 0.0)), (2.0, (1.0, 0.0, 0.0))])
        composite_left = CompositeBranch(label="L", components=(a0, a1))
        composite_right = CompositeBranch(label="R", components=(b0,))
        matrix = compute_composite_phase_matrix((composite_left,), (composite_right,), mass=0.5)
        expected = compute_mediated_phase_matrix((a0,), (b0,), mass=0.5)[0][0] + compute_mediated_phase_matrix((a1,), (b0,), mass=0.5)[0][0]
        self.assertAlmostEqual(matrix[0][0], expected)


    # --- 수치 알고리즘 개선 ---

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

    def test_lebedev_invalid_higher_order_is_rejected(self) -> None:
        source = branch("A0", 1.0, [(0.0, (0.0, 0.0, 0.0)), (2.0, (0.0, 0.0, 0.0))])
        with self.assertRaises(ValueError):
            compute_lebedev_displacement_amplitudes(
                (source,), field_mass=0.5, momentum_cutoff=1.0, lebedev_order=42,
            )

    # --- 물리적 충실도 확장 ---

    def test_proper_time_worldline_static_branch(self) -> None:
        b = branch("A0", 1.0, [(0.0, (0.0, 0.0, 0.0)), (2.0, (0.0, 0.0, 0.0))])
        result = compute_proper_time_worldline(b, light_speed=1.0)
        self.assertAlmostEqual(result.proper_times[-1], 2.0)
        self.assertAlmostEqual(result.lorentz_factors[0], 1.0)

    def test_proper_time_worldline_moving_branch_has_time_dilation(self) -> None:
        b = branch("A0", 1.0, [(0.0, (0.0, 0.0, 0.0)), (2.0, (1.0, 0.0, 0.0))])
        result = compute_proper_time_worldline(b, light_speed=1.0)
        self.assertLess(result.proper_times[-1], 2.0)
        self.assertGreater(result.lorentz_factors[0], 1.0)

    def test_tensor_mediated_phase_returns_all_three_mediators(self) -> None:
        source = branch("A0", 1.0, [(0.0, (-2.0, 0.0, 0.0)), (2.0, (-2.0, 0.0, 0.0))])
        target = branch("B0", 1.0, [(0.0, (2.0, 0.0, 0.0)), (2.0, (2.0, 0.0, 0.0))])
        result = compute_tensor_mediated_phase_matrix(
            (source,), (target,), mass=0.5, propagation="instantaneous",
        )
        self.assertEqual(len(result.scalar_phase), 1)
        self.assertEqual(len(result.vector_phase), 1)
        self.assertEqual(len(result.gravity_phase), 1)
        self.assertNotEqual(result.scalar_phase[0][0], 0.0)

    def test_tensor_mediated_massive_mediator_differs(self) -> None:
        source = branch("A0", 1.0, [(0.0, (-2.0, 0.0, 0.0)), (2.0, (-2.0, 0.0, 0.0))])
        target = branch("B0", 1.0, [(0.0, (2.0, 0.0, 0.0)), (2.0, (2.0, 0.0, 0.0))])
        massless = compute_tensor_mediated_phase_matrix(
            (source,), (target,), mass=0.5, propagation="instantaneous", mediator_mass=0.0,
        )
        massive = compute_tensor_mediated_phase_matrix(
            (source,), (target,), mass=0.5, propagation="instantaneous", mediator_mass=1.0,
        )
        self.assertNotAlmostEqual(massless.vector_phase[0][0], massive.vector_phase[0][0])

    def test_tensor_mediated_vector_projector_distinguishes_motion_direction(self) -> None:
        source_parallel = branch("A_parallel", 1.0, [(0.0, (-2.0, 0.0, 0.0)), (2.0, (-1.0, 0.0, 0.0))])
        target_parallel = branch("B_parallel", 1.0, [(0.0, (2.0, 0.0, 0.0)), (2.0, (3.0, 0.0, 0.0))])
        source_transverse = branch("A_transverse", 1.0, [(0.0, (-2.0, 0.0, 0.0)), (2.0, (-2.0, 1.0, 0.0))])
        target_transverse = branch("B_transverse", 1.0, [(0.0, (2.0, 0.0, 0.0)), (2.0, (2.0, 1.0, 0.0))])
        parallel = compute_tensor_mediated_phase_matrix(
            (source_parallel,), (target_parallel,), mass=0.5, propagation="instantaneous", mediator_mass=0.0,
        )
        transverse = compute_tensor_mediated_phase_matrix(
            (source_transverse,), (target_transverse,), mass=0.5, propagation="instantaneous", mediator_mass=0.0,
        )
        self.assertNotAlmostEqual(parallel.vector_phase[0][0], transverse.vector_phase[0][0])

    def test_tensor_mediated_gravity_projector_distinguishes_motion_direction(self) -> None:
        source_parallel = branch("A_parallel", 1.0, [(0.0, (-2.0, 0.0, 0.0)), (2.0, (-1.0, 0.0, 0.0))])
        target_parallel = branch("B_parallel", 1.0, [(0.0, (2.0, 0.0, 0.0)), (2.0, (3.0, 0.0, 0.0))])
        source_transverse = branch("A_transverse", 1.0, [(0.0, (-2.0, 0.0, 0.0)), (2.0, (-2.0, 1.0, 0.0))])
        target_transverse = branch("B_transverse", 1.0, [(0.0, (2.0, 0.0, 0.0)), (2.0, (2.0, 1.0, 0.0))])
        parallel = compute_tensor_mediated_phase_matrix(
            (source_parallel,), (target_parallel,), mass=0.5, propagation="instantaneous", mediator_mass=0.0,
        )
        transverse = compute_tensor_mediated_phase_matrix(
            (source_transverse,), (target_transverse,), mass=0.5, propagation="instantaneous", mediator_mass=0.0,
        )
        self.assertNotAlmostEqual(parallel.gravity_phase[0][0], transverse.gravity_phase[0][0])

    def test_tensor_mediated_gauge_scheme_changes_vector_phase(self) -> None:
        source = branch("A0", 1.0, [(0.0, (-2.0, 0.0, 0.0)), (2.0, (-1.0, 0.0, 0.0))])
        target = branch("B0", 1.0, [(0.0, (2.0, 0.0, 0.0)), (2.0, (2.5, 0.5, 0.0))])
        landau = compute_tensor_mediated_phase_matrix(
            (source,), (target,), mass=0.5, propagation="instantaneous", mediator_mass=0.8, gauge_scheme="landau",
        )
        feynman = compute_tensor_mediated_phase_matrix(
            (source,), (target,), mass=0.5, propagation="instantaneous", mediator_mass=0.8, gauge_scheme="feynman",
        )
        self.assertNotAlmostEqual(landau.vector_phase[0][0], feynman.vector_phase[0][0])
        self.assertEqual(feynman.gauge_scheme, "feynman")

    def test_tensor_mediated_vertex_resummation_changes_tensor_phase(self) -> None:
        source = branch("A0", 1.0, [(0.0, (-1.5, 0.0, 0.0)), (2.0, (-0.5, 0.0, 0.0))])
        target = branch("B0", 1.0, [(0.0, (1.5, 0.0, 0.0)), (2.0, (0.5, 0.0, 0.0))])
        bare = compute_tensor_mediated_phase_matrix(
            (source,), (target,), mass=0.2, propagation="instantaneous", vertex_resummation="none",
        )
        resum = compute_tensor_mediated_phase_matrix(
            (source,),
            (target,),
            mass=0.2,
            propagation="instantaneous",
            vertex_resummation="pade",
            vertex_strength=0.75,
        )
        self.assertNotAlmostEqual(bare.vector_phase[0][0], resum.vector_phase[0][0])
        self.assertEqual(resum.vertex_resummation, "pade")

    def test_tensor_mediated_brst_ghost_sector_restores_landau_slice(self) -> None:
        source = branch("A0", 1.0, [(0.0, (-2.0, 0.0, 0.0)), (2.0, (-1.0, 0.0, 0.0))])
        target = branch("B0", 1.0, [(0.0, (2.0, 0.0, 0.0)), (2.0, (2.5, 0.5, 0.0))])
        bare = compute_tensor_mediated_phase_matrix(
            (source,), (target,), mass=0.5, propagation="instantaneous", mediator_mass=0.8, gauge_scheme="feynman",
        )
        brst = compute_tensor_mediated_phase_matrix(
            (source,),
            (target,),
            mass=0.5,
            propagation="instantaneous",
            mediator_mass=0.8,
            gauge_scheme="feynman",
            ghost_mode="brst",
            ghost_strength=1.0,
        )
        self.assertNotAlmostEqual(brst.ghost_sector.ghost_phase[0][0], 0.0)
        self.assertNotAlmostEqual(brst.ghost_sector.brst_compensated_vector_phase[0][0], bare.vector_phase[0][0])
        self.assertAlmostEqual(brst.ghost_sector.brst_residual_norm, 0.0, places=12)

    def test_tensor_mediated_dyson_schwinger_dresses_channels(self) -> None:
        source = branch("A0", 1.0, [(0.0, (-1.5, 0.0, 0.0)), (2.0, (-0.5, 0.0, 0.0))])
        target = branch("B0", 1.0, [(0.0, (1.5, 0.0, 0.0)), (2.0, (0.5, 0.0, 0.0))])
        result = compute_tensor_mediated_phase_matrix(
            (source,),
            (target,),
            mass=0.2,
            propagation="instantaneous",
            ghost_mode="brst",
            dyson_schwinger_mode="coupled",
            dyson_schwinger_strength=0.3,
            dyson_schwinger_iterations=16,
            dyson_schwinger_tolerance=1e-7,
            dyson_schwinger_relaxation=0.6,
        )
        self.assertGreater(result.dyson_schwinger.iterations, 0)
        self.assertTrue(result.dyson_schwinger.converged)
        self.assertNotAlmostEqual(
            result.dyson_schwinger.dressed_vector_phase[0][0],
            result.ghost_sector.brst_compensated_vector_phase[0][0],
        )
        self.assertNotAlmostEqual(
            result.dyson_schwinger.dressed_gravity_phase[0][0],
            result.gravity_phase[0][0],
        )

    def test_renormalized_phase_matrix_subtracts_self_energy(self) -> None:
        source = branch("A0", 1.0, [(0.0, (-2.0, 0.0, 0.0)), (2.0, (-2.0, 0.0, 0.0))])
        target = branch("B0", 1.0, [(0.0, (2.0, 0.0, 0.0)), (2.0, (2.0, 0.0, 0.0))])
        result = compute_renormalized_phase_matrix(
            (source,), (target,), mass=0.5, cutoff=0.1,
        )
        self.assertNotEqual(result.raw_matrix[0][0], result.renormalized_matrix[0][0])
        self.assertGreater(len(result.self_energy_corrections), 0)

    def test_renormalized_with_mass_counterterm(self) -> None:
        source = branch("A0", 1.0, [(0.0, (-2.0, 0.0, 0.0)), (2.0, (-2.0, 0.0, 0.0))])
        target = branch("B0", 1.0, [(0.0, (2.0, 0.0, 0.0)), (2.0, (2.0, 0.0, 0.0))])
        without = compute_renormalized_phase_matrix(
            (source,), (target,), mass=0.5, cutoff=0.1,
        )
        with_counter = compute_renormalized_phase_matrix(
            (source,), (target,), mass=0.5, cutoff=0.1, renormalization_mass=0.0,
        )
        self.assertNotAlmostEqual(without.self_energy_corrections[0], with_counter.self_energy_corrections[0])

    def test_decoherence_model_returns_coherence_and_purity(self) -> None:
        source = branch("A0", 1.0, [(0.0, (-2.0, 0.0, 0.0)), (2.0, (-2.0, 0.0, 0.0))])
        target = branch("B0", 1.0, [(0.0, (2.0, 0.0, 0.0)), (2.0, (2.0, 0.0, 0.0))])
        momenta = ((0.5, 0.0, 0.0), (0.0, 0.5, 0.0))
        result = compute_decoherence_model(
            (source,), (target,), momenta, field_mass=0.5, environment_coupling=0.1,
        )
        self.assertGreater(result.purity, 0.0)
        self.assertLessEqual(result.purity, 1.0)
        self.assertEqual(len(result.decoherence_rates), 1)
        self.assertEqual(result.thermal_occupations, (0.0, 0.0))
        self.assertEqual(result.spectral_weights, (1.0, 1.0))
        self.assertEqual(result.memory_kernel_norm, 0.0)
        self.assertEqual(result.generated_lindblad_rates, tuple())
        self.assertEqual(result.detailed_balance_deviation, 0.0)
        self.assertEqual(len(result.renormalized_transition_energies), 1)
        self.assertEqual(result.lamb_shift_matrix, ((0.0 + 0.0j,),))
        self.assertEqual(result.bath_dressing_matrix, ((0.0 + 0.0j,),))
        self.assertEqual(result.bath_dressing_norm, 0.0)
        self.assertEqual(result.influence_phase_matrix, ((0.0,),))
        self.assertEqual(result.non_gaussian_cumulant_norm, 0.0)
        self.assertEqual(result.influence_iterations, 1)
        self.assertEqual(result.influence_residual, 0.0)
        self.assertTrue(result.influence_converged)
        self.assertAlmostEqual(result.lindblad_trace, 1.0)

    def test_decoherence_model_thermal_environment_suppresses_off_diagonal(self) -> None:
        source0 = branch("A0", 1.0, [(0.0, (-2.0, 0.0, 0.0)), (2.0, (-2.0, 0.0, 0.0))])
        source1 = branch("A1", 1.0, [(0.0, (-1.0, 0.0, 0.0)), (2.0, (-1.0, 0.0, 0.0))])
        target = branch("B0", 1.0, [(0.0, (2.0, 0.0, 0.0)), (2.0, (2.0, 0.0, 0.0))])
        momenta = ((0.5, 0.0, 0.0), (0.0, 0.5, 0.0))
        vacuum = compute_decoherence_model(
            (source0, source1), (target,), momenta, field_mass=0.5,
        )
        thermal = compute_decoherence_model(
            (source0, source1), (target,), momenta, field_mass=0.5, environment_temperature=0.8,
        )
        self.assertGreater(thermal.thermal_occupations[0], 0.0)
        self.assertLess(abs(thermal.coherence_matrix[0][1]), abs(vacuum.coherence_matrix[0][1]))
        self.assertLess(thermal.purity, vacuum.purity)

    def test_decoherence_model_bath_spectral_density_reweights_modes(self) -> None:
        source0 = branch("A0", 1.0, [(0.0, (-2.0, 0.0, 0.0)), (2.0, (-2.0, 0.0, 0.0))])
        source1 = branch("A1", 1.0, [(0.0, (-1.0, 0.0, 0.0)), (2.0, (-1.0, 0.0, 0.0))])
        target = branch("B0", 1.0, [(0.0, (2.0, 0.0, 0.0)), (2.0, (2.0, 0.0, 0.0))])
        result = compute_decoherence_model(
            (source0, source1),
            (target,),
            ((0.2, 0.0, 0.0), (1.5, 0.0, 0.0)),
            field_mass=0.5,
            environment_temperature=0.9,
            bath_spectral_density=lambda omega: 1.0 / (1.0 + omega * omega),
        )
        self.assertGreater(result.spectral_weights[0], result.spectral_weights[1])
        self.assertGreater(result.thermal_occupations[0], result.thermal_occupations[1])

    def test_decoherence_model_lindblad_evolution_preserves_trace(self) -> None:
        source0 = branch("A0", 1.0, [(0.0, (-2.0, 0.0, 0.0)), (2.0, (-2.0, 0.0, 0.0))])
        source1 = branch("A1", 1.0, [(0.0, (-1.0, 0.0, 0.0)), (2.0, (-1.0, 0.0, 0.0))])
        target = branch("B0", 1.0, [(0.0, (2.0, 0.0, 0.0)), (2.0, (2.0, 0.0, 0.0))])
        result = compute_decoherence_model(
            (source0, source1),
            (target,),
            ((0.5, 0.0, 0.0),),
            field_mass=0.5,
            lindblad_operators=(
                (
                    (0.0 + 0.0j, 1.0 + 0.0j),
                    (0.0 + 0.0j, 0.0 + 0.0j),
                ),
            ),
            lindblad_rates=(0.7,),
            lindblad_time=1.5,
            lindblad_steps=96,
        )
        self.assertAlmostEqual(result.lindblad_trace, 1.0, places=8)
        self.assertGreater(result.coherence_matrix[0][0].real, result.coherence_matrix[1][1].real)
        self.assertLess(abs(result.coherence_matrix[0][1]), 0.5)

    def test_decoherence_model_non_markovian_memory_changes_evolution(self) -> None:
        source0 = branch("A0", 1.0, [(0.0, (-2.0, 0.0, 0.0)), (2.0, (-2.0, 0.0, 0.0))])
        source1 = branch("A1", 1.0, [(0.0, (-1.0, 0.0, 0.0)), (2.0, (-1.0, 0.0, 0.0))])
        target = branch("B0", 1.0, [(0.0, (2.0, 0.0, 0.0)), (2.0, (2.0, 0.0, 0.0))])
        markov = compute_decoherence_model(
            (source0, source1),
            (target,),
            ((0.5, 0.0, 0.0),),
            field_mass=0.5,
            lindblad_operators=(
                (
                    (0.0 + 0.0j, 1.0 + 0.0j),
                    (0.0 + 0.0j, 0.0 + 0.0j),
                ),
            ),
            lindblad_rates=(0.3,),
            lindblad_time=1.0,
            lindblad_steps=64,
        )
        non_markov = compute_decoherence_model(
            (source0, source1),
            (target,),
            ((0.5, 0.0, 0.0),),
            field_mass=0.5,
            lindblad_operators=(
                (
                    (0.0 + 0.0j, 1.0 + 0.0j),
                    (0.0 + 0.0j, 0.0 + 0.0j),
                ),
            ),
            lindblad_rates=(0.3,),
            lindblad_time=1.0,
            lindblad_steps=64,
            memory_kernel=lambda tau: exp(-2.0 * tau),
            memory_strength=0.4,
            colored_noise_correlation=lambda tau: exp(-tau),
            noise_time_window=1.0,
            noise_steps=32,
        )
        self.assertGreater(non_markov.memory_kernel_norm, 0.0)
        self.assertAlmostEqual(non_markov.lindblad_trace, 1.0, places=8)
        self.assertNotAlmostEqual(non_markov.purity, markov.purity)

    def test_decoherence_model_auto_generates_kms_lindblad_rates(self) -> None:
        source0 = branch("A0", 1.0, [(0.0, (-2.0, 0.0, 0.0)), (2.0, (-2.0, 0.0, 0.0))])
        source1 = branch("A1", 1.0, [(0.0, (-1.0, 0.0, 0.0)), (2.0, (-1.0, 0.0, 0.0))])
        target = branch("B0", 1.0, [(0.0, (2.0, 0.0, 0.0)), (2.0, (2.0, 0.0, 0.0))])
        result = compute_decoherence_model(
            (source0, source1),
            (target,),
            ((0.5, 0.0, 0.0),),
            field_mass=0.5,
            environment_temperature=0.7,
            bath_spectral_density=lambda omega: 1.0 + 0.5 * omega,
            auto_lindblad_from_bath=True,
            lindblad_time=1.0,
            lindblad_steps=64,
            dephasing_rate_scale=0.25,
            system_transition_energies=(0.0, 1.2),
        )
        self.assertGreater(len(result.generated_lindblad_rates), 0)
        self.assertLess(result.detailed_balance_deviation, 1e-10)
        self.assertAlmostEqual(result.lindblad_trace, 1.0, places=8)
        self.assertLess(result.purity, 1.0)

    def test_decoherence_model_auto_lindblad_covers_all_energy_level_pairs(self) -> None:
        source0 = branch("A0", 1.0, [(0.0, (-2.0, 0.0, 0.0)), (2.0, (-2.0, 0.0, 0.0))])
        source1 = branch("A1", 1.0, [(0.0, (-1.0, 0.0, 0.0)), (2.0, (-1.0, 0.0, 0.0))])
        source2 = branch("A2", 1.0, [(0.0, (0.0, 0.0, 0.0)), (2.0, (0.0, 0.0, 0.0))])
        target = branch("B0", 1.0, [(0.0, (2.0, 0.0, 0.0)), (2.0, (2.0, 0.0, 0.0))])
        temperature = 0.9
        result = compute_decoherence_model(
            (source0, source1, source2),
            (target,),
            ((0.5, 0.0, 0.0),),
            field_mass=0.5,
            environment_coupling=0.2,
            environment_temperature=temperature,
            bath_spectral_density=lambda omega: 1.0 + 0.25 * omega,
            auto_lindblad_from_bath=True,
            system_transition_energies=(0.0, 0.7, 1.2),
            dephasing_rate_scale=0.1,
        )
        self.assertEqual(len(result.generated_lindblad_rates), 8)
        pair_gaps = (0.7, 1.2, 0.5)
        for pair_index, gap in enumerate(pair_gaps):
            emission_rate = result.generated_lindblad_rates[2 * pair_index]
            absorption_rate = result.generated_lindblad_rates[2 * pair_index + 1]
            self.assertGreater(emission_rate, absorption_rate)
            self.assertAlmostEqual(
                absorption_rate / emission_rate,
                exp(-gap / temperature),
                places=10,
            )
        self.assertLess(result.detailed_balance_deviation, 1e-10)

    def test_decoherence_model_auto_lamb_shift_renormalizes_energies(self) -> None:
        source0 = branch("A0", 1.0, [(0.0, (-2.0, 0.0, 0.0)), (2.0, (-2.0, 0.0, 0.0))])
        source1 = branch("A1", 1.0, [(0.0, (-1.0, 0.0, 0.0)), (2.0, (-1.0, 0.0, 0.0))])
        target = branch("B0", 1.0, [(0.0, (2.0, 0.0, 0.0)), (2.0, (2.0, 0.0, 0.0))])
        bare = compute_decoherence_model(
            (source0, source1),
            (target,),
            ((0.5, 0.0, 0.0),),
            field_mass=0.5,
            system_transition_energies=(0.0, 1.2),
            lindblad_time=1.0,
            lindblad_steps=64,
        )
        shifted = compute_decoherence_model(
            (source0, source1),
            (target,),
            ((0.5, 0.0, 0.0),),
            field_mass=0.5,
            environment_temperature=0.7,
            bath_spectral_density=lambda omega: 1.0 / (1.0 + omega),
            system_transition_energies=(0.0, 1.2),
            auto_lamb_shift_from_bath=True,
            lamb_shift_strength=0.2,
            lamb_shift_cutoff=3.0,
            lindblad_time=1.0,
            lindblad_steps=64,
        )
        self.assertNotAlmostEqual(
            shifted.renormalized_transition_energies[1],
            bare.renormalized_transition_energies[1],
        )
        self.assertNotEqual(shifted.lamb_shift_matrix[1][1], 0.0 + 0.0j)
        self.assertAlmostEqual(shifted.lindblad_trace, 1.0, places=8)

    def test_decoherence_model_auto_bath_dressing_builds_off_diagonal_self_energy(self) -> None:
        source0 = branch("A0", 1.0, [(0.0, (-2.0, 0.0, 0.0)), (2.0, (-2.0, 0.0, 0.0))])
        source1 = branch("A1", 1.0, [(0.0, (-1.0, 0.0, 0.0)), (2.0, (-1.0, 0.0, 0.0))])
        target = branch("B0", 1.0, [(0.0, (2.0, 0.0, 0.0)), (2.0, (2.0, 0.0, 0.0))])
        dressed = compute_decoherence_model(
            (source0, source1),
            (target,),
            ((0.5, 0.0, 0.0),),
            field_mass=0.5,
            bath_spectral_density=lambda omega: 1.0 / (1.0 + omega),
            system_transition_energies=(0.0, 1.2),
            auto_bath_dressing_from_bath=True,
            bath_dressing_strength=0.3,
            lindblad_time=1.0,
            lindblad_steps=64,
        )
        self.assertGreater(dressed.bath_dressing_norm, 0.0)
        self.assertNotEqual(dressed.bath_dressing_matrix[0][1], 0.0 + 0.0j)
        self.assertAlmostEqual(dressed.lindblad_trace, 1.0, places=8)

    def test_decoherence_model_influence_functional_adds_non_gaussian_phase(self) -> None:
        source0 = branch("A0", 1.0, [(0.0, (-2.0, 0.0, 0.0)), (2.0, (-2.0, 0.0, 0.0))])
        source1 = branch("A1", 1.0, [(0.0, (-1.0, 0.0, 0.0)), (2.0, (-1.0, 0.0, 0.0))])
        target = branch("B0", 1.0, [(0.0, (2.0, 0.0, 0.0)), (2.0, (2.0, 0.0, 0.0))])
        base = compute_decoherence_model(
            (source0, source1),
            (target,),
            ((0.5, 0.0, 0.0),),
            field_mass=0.5,
            colored_noise_correlation=lambda tau: exp(-tau),
            noise_time_window=1.0,
            noise_steps=32,
        )
        influenced = compute_decoherence_model(
            (source0, source1),
            (target,),
            ((0.5, 0.0, 0.0),),
            field_mass=0.5,
            bath_spectral_density=lambda omega: 1.0 / (1.0 + omega),
            colored_noise_correlation=lambda tau: exp(-tau),
            influence_phase_strength=0.2,
            influence_time_grid=(0.0, 0.25, 0.5, 0.75, 1.0),
            non_gaussian_cumulant=lambda t1, t2: (t2 - t1) * (t1 + t2),
            non_gaussian_cumulant_strength=0.15,
        )
        self.assertGreater(abs(influenced.influence_phase_matrix[0][1]), 0.0)
        self.assertGreater(influenced.non_gaussian_cumulant_norm, 0.0)
        self.assertNotAlmostEqual(
            influenced.coherence_matrix[0][1].imag,
            base.coherence_matrix[0][1].imag,
        )

    def test_decoherence_model_self_consistent_influence_resummation_converges(self) -> None:
        source0 = branch("A0", 1.0, [(0.0, (-2.0, 0.0, 0.0)), (2.0, (-2.0, 0.0, 0.0))])
        source1 = branch("A1", 1.0, [(0.0, (-1.0, 0.0, 0.0)), (2.0, (-1.0, 0.0, 0.0))])
        target = branch("B0", 1.0, [(0.0, (2.0, 0.0, 0.0)), (2.0, (2.0, 0.0, 0.0))])
        result = compute_decoherence_model(
            (source0, source1),
            (target,),
            ((0.5, 0.0, 0.0),),
            field_mass=0.5,
            bath_spectral_density=lambda omega: 1.0 / (1.0 + omega),
            colored_noise_correlation=lambda tau: exp(-tau),
            influence_phase_strength=0.15,
            influence_iterations=16,
            influence_tolerance=1e-4,
            influence_relaxation=0.6,
        )
        self.assertGreaterEqual(result.influence_iterations, 1)
        self.assertLessEqual(result.influence_residual, 1e-4)
        self.assertTrue(result.influence_converged)

    def test_multi_body_correlation_pairwise_is_nonzero(self) -> None:
        a = branch("A", 1.0, [(0.0, (-2.0, 0.0, 0.0)), (2.0, (-2.0, 0.0, 0.0))])
        b = branch("B", 1.0, [(0.0, (0.0, 0.0, 0.0)), (2.0, (0.0, 0.0, 0.0))])
        c = branch("C", 1.0, [(0.0, (2.0, 0.0, 0.0)), (2.0, (2.0, 0.0, 0.0))])
        result = compute_multi_body_correlation((a, b, c), mass=0.5)
        self.assertNotEqual(result.total_correlation_phase, 0.0)
        self.assertNotEqual(result.three_body_phase, 0.0)
        self.assertEqual(len(result.pairwise_phases), 3)

    def test_relativistic_backreaction_returns_proper_times(self) -> None:
        source = branch("A0", 1.0, [(0.0, (0.0, 0.0, 0.0)), (2.0, (0.0, 0.0, 0.0))])
        target = branch("B0", 1.0, [(0.0, (1.0, 0.0, 0.0)), (2.0, (1.0, 0.0, 0.0))])
        result = evolve_relativistic_backreaction(
            source, target, mass=0.5, propagation="instantaneous", response_strength=0.1,
        )
        self.assertNotEqual(result.updated_branch.points[0].position, target.points[0].position)
        self.assertEqual(len(result.proper_times), 2)
        self.assertEqual(len(result.four_velocities), 2)

    # --- 얽힘 진단 확장 ---

    def test_entanglement_measures_returns_all_quantities(self) -> None:
        source = branch("A0", 1.0, [(0.0, (-2.0, 0.0, 0.0)), (2.0, (-2.0, 0.0, 0.0))])
        target = branch("B0", 1.0, [(0.0, (2.0, 0.0, 0.0)), (2.0, (2.0, 0.0, 0.0))])
        source2 = branch("A1", 1.0, [(0.0, (-1.0, 0.0, 0.0)), (2.0, (-1.0, 0.0, 0.0))])
        target2 = branch("B1", 1.0, [(0.0, (1.0, 0.0, 0.0)), (2.0, (1.0, 0.0, 0.0))])
        matrix = compute_branch_phase_matrix((source, source2), (target, target2), mass=0.5)
        result = compute_entanglement_measures(matrix)
        self.assertGreaterEqual(result.von_neumann_entropy, 0.0)
        self.assertGreaterEqual(result.negativity, 0.0)
        self.assertLessEqual(result.witness_value, 0.5)
        self.assertGreaterEqual(result.visibility, 0.0)
        self.assertLessEqual(result.visibility, 1.0)

    def test_entanglement_visibility_is_maximal_for_equal_separation(self) -> None:
        s0 = branch("A0", 1.0, [(0.0, (-2.0, 0.0, 0.0)), (2.0, (-2.0, 0.0, 0.0))])
        s1 = branch("A1", 1.0, [(0.0, (-1.0, 0.0, 0.0)), (2.0, (-1.0, 0.0, 0.0))])
        t0 = branch("B0", 1.0, [(0.0, (1.0, 0.0, 0.0)), (2.0, (1.0, 0.0, 0.0))])
        t1 = branch("B1", 1.0, [(0.0, (2.0, 0.0, 0.0)), (2.0, (2.0, 0.0, 0.0))])
        matrix = compute_branch_phase_matrix((s0, s1), (t0, t1), mass=0.5)
        result = compute_entanglement_measures(matrix)
        self.assertGreater(result.visibility, 0.0)

    def test_mode_occupation_distribution_sums_correctly(self) -> None:
        amplitudes = (0.5 + 0.3j, 0.1 - 0.2j, 0.4 + 0.0j)
        momenta = ((0.5, 0.0, 0.0), (0.0, 0.5, 0.0), (0.0, 0.0, 0.5))
        result = compute_mode_occupation_distribution(amplitudes, momenta, field_mass=0.5)
        self.assertEqual(len(result.mode_occupations), 3)
        self.assertAlmostEqual(result.total_occupation, sum(result.mode_occupations))
        self.assertAlmostEqual(sum(result.mode_probabilities), 1.0)

    # --- PDE/격자 개선 ---

    def test_finite_difference_kg_produces_field(self) -> None:
        source = branch("A0", 1.0, [(0.0, (0.0, 0.0, 0.0)), (3.0, (0.0, 0.0, 0.0))])
        result = solve_finite_difference_kg(
            source,
            time_slices=(0.0, 0.5, 1.0, 1.5),
            spatial_points=((-2.0, 0.0, 0.0), (-1.0, 0.0, 0.0), (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0)),
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

    def test_finite_difference_kg_periodic_boundary(self) -> None:
        source = branch("A0", 1.0, [(0.0, (0.0, 0.0, 0.0)), (2.0, (0.0, 0.0, 0.0))])
        result = solve_finite_difference_kg(
            source,
            time_slices=(0.0, 0.5, 1.0),
            spatial_points=((-1.0, 0.0, 0.0), (0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
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
            spatial_points=((-1.0, 0.0, 0.0), (0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
            mass=0.5,
            boundary="reflecting",
            max_courant=0.9,
        )
        self.assertGreater(result.courant_number, 0.9)
        self.assertGreater(result.substeps_per_interval[0], 1)
        self.assertLess(result.effective_time_step, 2.0)

    def test_finite_difference_kg_reflecting_boundary(self) -> None:
        source = branch("A0", 1.0, [(0.0, (0.0, 0.0, 0.0)), (2.0, (0.0, 0.0, 0.0))])
        result = solve_finite_difference_kg(
            source,
            time_slices=(0.0, 0.5, 1.0),
            spatial_points=((-1.0, 0.0, 0.0), (0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
            mass=0.5,
            boundary="reflecting",
        )
        self.assertEqual(len(result.field_values), 3)

    def test_finite_difference_kg_adaptive_mesh_refinement_increases_grid_points(self) -> None:
        source = branch("A0", 1.0, [(0.0, (0.0, 0.0, 0.0)), (2.0, (0.0, 0.0, 0.0))])
        result = solve_finite_difference_kg(
            source,
            time_slices=(0.0, 0.5, 1.0),
            spatial_points=((-2.0, 0.0, 0.0), (0.0, 0.0, 0.0), (2.0, 0.0, 0.0)),
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
            spatial_points=((-2.0, 0.0, 0.0), (-1.0, 0.0, 0.0), (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0)),
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


    # --- 근본적 한계 개선 테스트 ---

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
        momenta = ((0.5, 0.0, 0.0), (0.0, 0.5, 0.0))
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
        momenta = ((0.5, 0.0, 0.0), (0.0, 0.5, 0.0), (0.0, 0.0, 0.5))
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

    # --- 근본적 한계 개선 2차 테스트 ---

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
        momenta = ((0.5, 0.0, 0.0), (0.0, 0.5, 0.0))
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
        momenta = ((0.5, 0.0, 0.0),)
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
