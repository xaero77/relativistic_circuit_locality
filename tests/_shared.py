"""Shared test helpers and imports for scalar-field regression tests."""

import unittest
from math import asinh, exp, pi
import relativistic_circuit_locality as package_api
import relativistic_circuit_locality.scalar_field as scalar_field

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
    compute_extrapolated_lebedev_displacement_amplitudes,
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


MOMENTA_PLANAR_2 = ((0.0, 0.0, 0.5), (0.5, 0.0, 0.0))
MOMENTA_PLANAR_3 = ((0.0, 0.0, 0.5), (0.5, 0.0, 0.0), (0.5, 0.5, 0.0))
MOMENTA_AXIAL_2 = ((0.5, 0.0, 0.0), (0.0, 0.5, 0.0))
MOMENTA_AXIAL_3 = ((0.5, 0.0, 0.0), (0.0, 0.5, 0.0), (0.0, 0.0, 0.5))
MOMENTA_AXIAL_1 = ((0.5, 0.0, 0.0),)

SPATIAL_POINTS_NEAR = ((1.0, 0.0, 0.0), (2.0, 0.0, 0.0))
SPATIAL_POINTS_NEAR_3 = ((1.0, 0.0, 0.0), (2.0, 0.0, 0.0), (3.0, 0.0, 0.0))
SPATIAL_POINTS_LINE_3 = ((-1.0, 0.0, 0.0), (0.0, 0.0, 0.0), (1.0, 0.0, 0.0))
SPATIAL_POINTS_LINE_5 = ((-2.0, 0.0, 0.0), (-1.0, 0.0, 0.0), (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0))
SPATIAL_POINTS_WIDE = ((-2.0, 0.0, 0.0), (0.0, 0.0, 0.0), (2.0, 0.0, 0.0))


