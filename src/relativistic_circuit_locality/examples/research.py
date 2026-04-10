from __future__ import annotations

"""Research-grade and surrogate examples beyond the stable core API."""

from ..core import BranchPath, SplineBranchPath, TrajectoryPoint, compute_spline_branch_phase_matrix
from ..experimental import (
    close_current_limitations,
    close_research_grade_limitations,
    compile_complete_state_family_bundle,
    compute_adaptive_phase_integral,
    compute_decoherence_model,
    compute_entanglement_measures,
    compute_fock_space_evolution,
    compute_lebedev_displacement_amplitudes,
    compute_mode_occupation_distribution,
    compute_multi_body_correlation,
    compute_proper_time_worldline,
    compute_renormalized_phase_matrix,
    compute_richardson_extrapolated_phase,
    compute_running_coupling_phase_matrix,
    compute_tensor_mediated_phase_matrix,
    evolve_relativistic_backreaction,
    interpolate_field_lattice,
    solve_exact_dynamics_surrogate,
    solve_field_lattice,
    solve_finite_difference_kg,
    solve_high_fidelity_pde_bundle,
    solve_physical_lattice_dynamics,
    validate_symbolic_bookkeeping,
)
from ._shared import base_branches


def collect_results() -> tuple[tuple[str, object], ...]:
    branches_a, branches_b = base_branches()
    momenta = ((0.0, 0.0, 0.5), (0.5, 0.0, 0.0), (0.5, 0.5, 0.0))
    labeled_mode_samples = (
        ("A0", ((1.0 + 0.0j, 0.2 + 0.0j),)),
        ("B0", ((0.8 + 0.0j, 0.1 + 0.0j),)),
    )
    current_limitations = close_current_limitations(
        branches_a[0],
        branches_b[0],
        time_slices=(0.0, 2.0, 4.0),
        spatial_points=((1.0, 0.0, 0.0), (2.0, 0.0, 0.0)),
        boundary_schedule=("open", "periodic", "open"),
        widths_a=((0.2, 0.3, 0.4),),
        widths_b=((0.3, 0.4, 0.5),),
        labeled_mode_samples=labeled_mode_samples,
        mass=0.5,
        propagation="kg_retarded",
    )
    high_fidelity = solve_high_fidelity_pde_bundle(
        branches_a[0],
        branches_b[0],
        time_slices=(0.0, 2.0, 4.0),
        spatial_points=((1.0, 0.0, 0.0), (2.0, 0.0, 0.0)),
        boundary_schedule=("open", "periodic", "open"),
        widths_a=((0.2, 0.3, 0.4),),
        widths_b=((0.3, 0.4, 0.5),),
        labeled_mode_samples=labeled_mode_samples,
        mass=0.5,
        propagation="kg_retarded",
    )
    complete_state = compile_complete_state_family_bundle(
        (branches_a[0],),
        (branches_b[0],),
        widths_a=((0.2, 0.3, 0.4),),
        widths_b=((0.3, 0.4, 0.5),),
        labeled_mode_samples=labeled_mode_samples,
        mass=0.5,
        propagation="kg_retarded",
    )
    exact_dynamics = solve_exact_dynamics_surrogate(
        branches_a[0],
        branches_b[0],
        time_slices=(0.0, 2.0, 4.0),
        spatial_points=((1.0, 0.0, 0.0), (2.0, 0.0, 0.0)),
        boundary_schedule=("open", "periodic", "open"),
        mass=0.5,
        propagation="kg_retarded",
    )
    research_grade = close_research_grade_limitations(
        branches_a[0],
        branches_b[0],
        time_slices=(0.0, 2.0, 4.0),
        spatial_points=((1.0, 0.0, 0.0), (2.0, 0.0, 0.0)),
        boundary_schedule=("open", "periodic", "open"),
        widths_a=((0.2, 0.3, 0.4),),
        widths_b=((0.3, 0.4, 0.5),),
        labeled_mode_samples=labeled_mode_samples,
        mass=0.5,
        propagation="kg_retarded",
    )
    proper_time = compute_proper_time_worldline(branches_a[0])
    tensor_source = BranchPath(
        label="A_tensor",
        charge=1.0,
        points=(
            TrajectoryPoint(0.0, (-2.0, 0.0, 0.0)),
            TrajectoryPoint(2.0, (-1.0, 0.5, 0.0)),
            TrajectoryPoint(4.0, (0.0, 1.0, 0.0)),
        ),
    )
    tensor_target = BranchPath(
        label="B_tensor",
        charge=1.0,
        points=(
            TrajectoryPoint(0.0, (2.0, 0.0, 0.0)),
            TrajectoryPoint(2.0, (2.5, 0.5, 0.0)),
            TrajectoryPoint(4.0, (3.0, 1.0, 0.0)),
        ),
    )
    tensor_phase = compute_tensor_mediated_phase_matrix(
        (tensor_source,),
        (tensor_target,),
        mass=0.5,
        propagation="instantaneous",
        gauge_scheme="feynman",
        vertex_resummation="pade",
        vertex_strength=0.5,
        ghost_mode="brst",
        dyson_schwinger_mode="coupled",
        dyson_schwinger_strength=0.3,
        dyson_schwinger_iterations=24,
        dyson_schwinger_tolerance=1e-7,
        dyson_schwinger_relaxation=0.6,
    )
    renormalized = compute_renormalized_phase_matrix(branches_a, branches_b, mass=0.5, cutoff=0.1)
    decoherence = compute_decoherence_model(branches_a[:1], branches_b[:1], momenta, field_mass=0.5)
    multi_body = compute_multi_body_correlation((branches_a[0], branches_a[1], branches_b[0]), mass=0.5)
    relativistic = evolve_relativistic_backreaction(
        branches_a[0], branches_b[0], mass=0.5, propagation="instantaneous", response_strength=0.1,
    )
    entanglement = compute_entanglement_measures(((0.1, 0.05), (0.07, 0.2)))
    mode_dist = compute_mode_occupation_distribution(
        ((0.1 + 0.0j), (0.2 + 0.0j), (0.05 + 0.0j)),
        momenta,
        field_mass=0.5,
    )
    fd_pde = solve_finite_difference_kg(
        branches_a[0],
        time_slices=(0.0, 0.5, 1.0, 1.5),
        spatial_points=((-2.0, 0.0, 0.0), (-1.0, 0.0, 0.0), (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0)),
        mass=0.5,
        boundary="absorbing",
    )
    phys_lattice = solve_physical_lattice_dynamics(
        branches_a[0],
        time_slices=(0.0, 1.0, 2.0, 3.0, 4.0),
        spatial_points=((1.0, 0.0, 0.0), (2.0, 0.0, 0.0)),
        mass=0.5,
        propagation="instantaneous",
        method="leapfrog",
        damping_rate=0.1,
    )
    lebedev = compute_lebedev_displacement_amplitudes(branches_a, field_mass=0.5, momentum_cutoff=1.0, lebedev_order=14)
    sp_a = (SplineBranchPath("A0", 1.0, (
        TrajectoryPoint(0.0, (-2.0, 0.0, 0.0)),
        TrajectoryPoint(2.0, (-1.5, 0.5, 0.0)),
        TrajectoryPoint(4.0, (-2.0, 0.0, 0.0)),
    )),)
    sp_b = (SplineBranchPath("B0", 1.0, (
        TrajectoryPoint(0.0, (2.0, 0.0, 0.0)),
        TrajectoryPoint(2.0, (1.5, -0.5, 0.0)),
        TrajectoryPoint(4.0, (2.0, 0.0, 0.0)),
    )),)
    spline_phase = compute_spline_branch_phase_matrix(sp_a, sp_b, mass=0.5)
    fock = compute_fock_space_evolution(
        branches_a[0], branches_b[0], momenta,
        field_mass=0.5, source_width_a=0.2, source_width_b=0.2,
    )
    adaptive = compute_adaptive_phase_integral(branches_a[:1], branches_b[:1], mass=0.5, tolerance=1e-8)
    richardson = compute_richardson_extrapolated_phase(branches_a[:1], branches_b[:1], mass=0.5, orders=(2, 4, 6, 8))
    fock3 = compute_fock_space_evolution(
        branches_a[0], branches_b[0], momenta,
        field_mass=0.5, source_width_a=0.2, source_width_b=0.2, magnus_order=3,
    )
    lattice = solve_field_lattice(
        branches_a[0],
        time_slices=(0.0, 2.0, 4.0),
        spatial_points=((1.0, 0.0, 0.0), (2.0, 0.0, 0.0)),
        mass=0.5,
    )
    interp = interpolate_field_lattice(lattice, 1.0, (1.5, 0.0, 0.0))
    running = compute_running_coupling_phase_matrix(
        branches_a[:1], branches_b[:1], mass=0.5, energy_scale=5.0, beta_coefficient=0.1,
    )
    validation = validate_symbolic_bookkeeping(labeled_mode_samples)
    return (
        ("current_limitations_grid_points", current_limitations.reference_pde.effective_grid_points),
        ("high_fidelity_pde_score", round(high_fidelity.fidelity_score, 6)),
        ("complete_state_relative_phase_matrix", complete_state.multimode.relative_phase_matrix),
        ("exact_dynamics_retarded_sample_count", len(exact_dynamics.retarded_green.samples)),
        ("research_grade_exact_dynamics_sample_count", len(research_grade.exact_dynamics.retarded_green.samples)),
        ("proper_time_A0", proper_time.proper_times),
        ("lorentz_factors_A0", proper_time.lorentz_factors),
        ("tensor_scalar_phase", tensor_phase.scalar_phase),
        ("tensor_vector_phase", tensor_phase.vector_phase),
        ("tensor_gravity_phase", tensor_phase.gravity_phase),
        ("tensor_gauge_scheme", tensor_phase.gauge_scheme),
        ("tensor_vertex_resummation", tensor_phase.vertex_resummation),
        ("tensor_ghost_phase", tensor_phase.ghost_sector.ghost_phase),
        ("tensor_brst_residual_norm", tensor_phase.ghost_sector.brst_residual_norm),
        ("tensor_ds_converged", tensor_phase.dyson_schwinger.converged),
        ("tensor_ds_dressed_vector_phase", tensor_phase.dyson_schwinger.dressed_vector_phase),
        ("renormalized_phase", renormalized.renormalized_matrix),
        ("self_energy_corrections", renormalized.self_energy_corrections),
        ("decoherence_purity", round(decoherence.purity, 6)),
        ("three_body_phase", round(multi_body.three_body_phase, 6)),
        ("relativistic_backreaction_proper_times", relativistic.proper_times),
        ("von_neumann_entropy", round(entanglement.von_neumann_entropy, 6)),
        ("negativity", round(entanglement.negativity, 6)),
        ("witness_value", round(entanglement.witness_value, 6)),
        ("visibility", round(entanglement.visibility, 6)),
        ("mode_occupations", mode_dist.mode_occupations),
        ("mode_probabilities", tuple(round(p, 4) for p in mode_dist.mode_probabilities)),
        ("fd_pde_courant", round(fd_pde.courant_number, 4)),
        ("fd_pde_slices", len(fd_pde.field_values)),
        ("fd_pde_stencil_order", fd_pde.stencil_order),
        ("fd_pde_refinement_rounds", fd_pde.refinement_rounds),
        ("fd_pde_boundary_geometry", fd_pde.boundary_geometry),
        ("fd_pde_active_points", sum(fd_pde.active_point_mask)),
        ("physical_lattice_method", phys_lattice.method),
        ("physical_lattice_step", phys_lattice.time_step),
        ("lebedev_direction_count", lebedev.direction_count),
        ("lebedev_amplitudes", lebedev.amplitudes),
        ("spline_phase_matrix", spline_phase),
        ("fock_parametric_phase", round(fock.parametric_phase, 6)),
        ("fock_time_ordering_correction", round(fock.time_ordering_correction, 8)),
        ("fock_total_phase", round(fock.total_phase, 6)),
        ("fock_mode_count", len(fock.mode_evolutions)),
        ("adaptive_phase", round(adaptive[0][0].phase, 6)),
        ("adaptive_error", adaptive[0][0].estimated_error),
        ("adaptive_segments", adaptive[0][0].segments_used),
        ("richardson_phases_by_order", tuple(round(p, 6) for p in richardson[0][0].phases_by_order)),
        ("richardson_extrapolated", round(richardson[0][0].extrapolated_phase, 6)),
        ("richardson_error", richardson[0][0].estimated_error),
        ("fock_third_order_correction", round(fock3.third_order_correction, 8)),
        ("fock_total_phase_3rd", round(fock3.total_phase, 6)),
        ("interpolated_field_value", round(interp.value, 6)),
        ("nearest_field_value", round(interp.nearest_value, 6)),
        ("bare_phase", round(running.bare_matrix[0][0], 6)),
        ("running_phase", round(running.running_matrix[0][0], 6)),
        ("bookkeeping_consistent", validation.is_consistent),
        ("bookkeeping_max_norm_residual", max(validation.norm_residuals)),
    )
