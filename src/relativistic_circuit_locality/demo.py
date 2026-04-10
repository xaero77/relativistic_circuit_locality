from __future__ import annotations

"""스칼라장 시뮬레이션 API를 실행해 보는 작은 예제."""

from .core import (
    BranchPath,
    SplineBranchPath,
    TrajectoryPoint,
    compute_entanglement_phase,
    compute_spline_branch_phase_matrix,
    simulate,
)
from .experimental import (
    analyze_branch_pair_coherent_state,
    analyze_branch_pair_coherent_overlap,
    analyze_branch_pair_phase,
    close_current_limitations,
    close_research_grade_limitations,
    compile_complete_state_family_bundle,
    CompositeBranch,
    compute_adaptive_phase_integral,
    compute_branch_displacement_amplitudes,
    compute_branch_pair_displacements,
    compute_composite_phase_matrix,
    compute_continuum_displacement_amplitudes,
    compute_decoherence_model,
    compute_displacement_operator_phase,
    compute_entanglement_measures,
    compute_fock_space_evolution,
    compute_lebedev_displacement_amplitudes,
    compute_mediated_phase_matrix,
    compute_mode_occupation_distribution,
    compute_multi_body_correlation,
    compute_phi_rs_samples,
    compute_proper_time_worldline,
    compute_renormalized_phase_matrix,
    compute_richardson_extrapolated_phase,
    compute_running_coupling_phase_matrix,
    compute_sampled_spacetime_phase,
    compute_tensor_mediated_phase_matrix,
    compute_wavepacket_phase_matrix,
    evolve_relativistic_backreaction,
    interpolate_field_lattice,
    sample_branch_field,
    solve_exact_dynamics_surrogate,
    solve_field_lattice,
    solve_finite_difference_kg,
    solve_high_fidelity_pde_bundle,
    solve_physical_lattice_dynamics,
    validate_symbolic_bookkeeping,
)


def _branch(label: str, charge: float, x0: float) -> BranchPath:
    # 데모에서는 거리 의존 위상 효과만 보이도록 각 분기를 정지 상태로 둔다.
    return BranchPath(
        label=label,
        charge=charge,
        points=(
            TrajectoryPoint(0.0, (x0, 0.0, 0.0)),
            TrajectoryPoint(2.0, (x0, 0.0, 0.0)),
            TrajectoryPoint(4.0, (x0, 0.0, 0.0)),
        ),
    )


def main() -> None:
    # 각 계에 두 개의 분기를 두어 2x2 상호작용 위상 행렬을 만든다.
    branches_a = (_branch("A0", 1.0, -2.0), _branch("A1", 1.0, -1.0))
    branches_b = (_branch("B0", 1.0, 1.0), _branch("B1", 1.0, 2.0))
    instantaneous = simulate(branches_a, branches_b, mass=0.5)
    retarded = simulate(branches_a, branches_b, mass=0.5, propagation="retarded")
    time_symmetric = simulate(branches_a, branches_b, mass=0.5, propagation="time_symmetric")
    causal_history = simulate(branches_a, branches_b, mass=0.5, propagation="causal_history")
    kg_retarded = simulate(branches_a, branches_b, mass=0.5, propagation="kg_retarded")
    print("closest_approach =", round(instantaneous.closest_approach, 6))
    print("mediation_intervals =", instantaneous.mediation_intervals)
    print("instantaneous_phase_matrix =", instantaneous.phase_matrix)
    print("retarded_phase_matrix =", retarded.phase_matrix)
    print("time_symmetric_phase_matrix =", time_symmetric.phase_matrix)
    print("causal_history_phase_matrix =", causal_history.phase_matrix)
    print("kg_retarded_phase_matrix =", kg_retarded.phase_matrix)
    print(
        "instantaneous_relative_entangling_phase =",
        round(compute_entanglement_phase(instantaneous.phase_matrix, 0, 1, 0, 1), 6),
    )
    print(
        "retarded_relative_entangling_phase =",
        round(compute_entanglement_phase(retarded.phase_matrix, 0, 1, 0, 1), 6),
    )
    print(
        "time_symmetric_relative_entangling_phase =",
        round(compute_entanglement_phase(time_symmetric.phase_matrix, 0, 1, 0, 1), 6),
    )
    print(
        "causal_history_relative_entangling_phase =",
        round(compute_entanglement_phase(causal_history.phase_matrix, 0, 1, 0, 1), 6),
    )
    print(
        "kg_retarded_relative_entangling_phase =",
        round(compute_entanglement_phase(kg_retarded.phase_matrix, 0, 1, 0, 1), 6),
    )
    sampled_field = sample_branch_field(
        branches_a[0],
        ((1.0, (0.0, 0.0, 0.0)), (2.0, (0.0, 0.0, 0.0)), (3.0, (0.0, 0.0, 0.0))),
        mass=0.5,
        propagation="kg_retarded",
    )
    phi_samples = compute_phi_rs_samples(
        branches_a[0],
        branches_b[0],
        sample_count=4,
        mass=0.5,
        propagation="kg_retarded",
    )
    print("sampled_field_A0 =", sampled_field)
    print("phi_rs_samples_A0_B0 =", phi_samples)
    sampled_phase = compute_sampled_spacetime_phase(
        branches_a[0],
        branches_b[0],
        mass=0.5,
        target_width=0.3,
        propagation="kg_retarded",
    )
    print("sampled_spacetime_phase_A0_B0 =", sampled_phase)
    momenta = ((0.0, 0.0, 0.5), (0.5, 0.0, 0.0), (0.5, 0.5, 0.0))
    displacement_a = compute_branch_displacement_amplitudes(
        branches_a,
        momenta,
        field_mass=0.5,
        source_width=0.2,
    )
    pair_displacements = compute_branch_pair_displacements(
        branches_a,
        branches_b,
        momenta,
        field_mass=0.5,
        source_width_a=0.2,
        source_width_b=0.2,
    )
    print("single_branch_displacements_A =", displacement_a)
    print("pair_displacements =", pair_displacements)
    print(
        "continuum_displacements_A =",
        compute_continuum_displacement_amplitudes(branches_a, field_mass=0.5, momentum_cutoff=1.0),
    )
    print(
        "pair_displacement_phase_A0B0_vs_A1B1 =",
        round(compute_displacement_operator_phase(pair_displacements[0][0], pair_displacements[1][1]), 6),
    )
    print(
        "pair_coherent_overlap_A0B0_vs_A1B1 =",
        analyze_branch_pair_coherent_overlap(
            (branches_a[0], branches_b[0]),
            (branches_a[1], branches_b[1]),
            momenta,
            field_mass=0.5,
            source_width_a=0.2,
            source_width_b=0.2,
        ),
    )
    coherent_state = analyze_branch_pair_coherent_state(
        branches_a[0],
        branches_b[0],
        momenta,
        field_mass=0.5,
        source_width_a=0.2,
        source_width_b=0.2,
        elapsed_time=1.5,
    )
    print("coherent_state_A0_B0 =", coherent_state)
    print(
        "gravity_phase_matrix =",
        compute_mediated_phase_matrix(branches_a, branches_b, mass=0.5, mediator="gravity"),
    )
    print(
        "composite_phase_matrix =",
        compute_composite_phase_matrix(
            (CompositeBranch("A", branches_a),),
            (CompositeBranch("B", branches_b),),
            mass=0.5,
        ),
    )
    wavepacket_matrix = compute_wavepacket_phase_matrix(
        branches_a,
        branches_b,
        widths_a=(0.3, 0.5),
        widths_b=(0.3, 0.5),
        mass=0.5,
    )
    print("wavepacket_phase_matrix =", wavepacket_matrix)
    pair_breakdown = analyze_branch_pair_phase(
        branches_a[0],
        branches_b[0],
        mass=0.5,
        propagation="kg_retarded",
        cutoff=0.1,
    )
    print("pair_breakdown_A0_B0 =", pair_breakdown)
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
    print("current_limitations_grid_points =", current_limitations.reference_pde.effective_grid_points)
    print("high_fidelity_pde_score =", round(high_fidelity.fidelity_score, 6))
    print("complete_state_relative_phase_matrix =", complete_state.multimode.relative_phase_matrix)
    print("exact_dynamics_retarded_sample_count =", len(exact_dynamics.retarded_green.samples))
    print("research_grade_exact_dynamics_sample_count =", len(research_grade.exact_dynamics.retarded_green.samples))

    # --- 새 기능 데모 ---
    proper_time = compute_proper_time_worldline(branches_a[0])
    print("proper_time_A0 =", proper_time.proper_times)
    print("lorentz_factors_A0 =", proper_time.lorentz_factors)
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
    print("tensor_scalar_phase =", tensor_phase.scalar_phase)
    print("tensor_vector_phase =", tensor_phase.vector_phase)
    print("tensor_gravity_phase =", tensor_phase.gravity_phase)
    print("tensor_gauge_scheme =", tensor_phase.gauge_scheme)
    print("tensor_vertex_resummation =", tensor_phase.vertex_resummation)
    print("tensor_ghost_phase =", tensor_phase.ghost_sector.ghost_phase)
    print("tensor_brst_residual_norm =", tensor_phase.ghost_sector.brst_residual_norm)
    print("tensor_ds_converged =", tensor_phase.dyson_schwinger.converged)
    print("tensor_ds_dressed_vector_phase =", tensor_phase.dyson_schwinger.dressed_vector_phase)
    renormalized = compute_renormalized_phase_matrix(
        branches_a, branches_b, mass=0.5, cutoff=0.1,
    )
    print("renormalized_phase =", renormalized.renormalized_matrix)
    print("self_energy_corrections =", renormalized.self_energy_corrections)
    decoherence = compute_decoherence_model(
        branches_a[:1], branches_b[:1], momenta, field_mass=0.5,
    )
    print("decoherence_purity =", round(decoherence.purity, 6))
    multi_body = compute_multi_body_correlation(
        (branches_a[0], branches_a[1], branches_b[0]), mass=0.5,
    )
    print("three_body_phase =", round(multi_body.three_body_phase, 6))
    relativistic = evolve_relativistic_backreaction(
        branches_a[0], branches_b[0], mass=0.5, propagation="instantaneous", response_strength=0.1,
    )
    print("relativistic_backreaction_proper_times =", relativistic.proper_times)
    entanglement = compute_entanglement_measures(instantaneous.phase_matrix)
    print("von_neumann_entropy =", round(entanglement.von_neumann_entropy, 6))
    print("negativity =", round(entanglement.negativity, 6))
    print("witness_value =", round(entanglement.witness_value, 6))
    print("visibility =", round(entanglement.visibility, 6))
    mode_dist = compute_mode_occupation_distribution(
        displacement_a[0], momenta, field_mass=0.5,
    )
    print("mode_occupations =", mode_dist.mode_occupations)
    print("mode_probabilities =", tuple(round(p, 4) for p in mode_dist.mode_probabilities))
    fd_pde = solve_finite_difference_kg(
        branches_a[0],
        time_slices=(0.0, 1.0, 2.0, 3.0, 4.0),
        spatial_points=((-2.0, 0.0, 0.0), (-1.0, 0.0, 0.0), (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0)),
        mass=0.5,
        boundary="absorbing",
        stencil_order=4,
        adaptive_mesh_refinement_rounds=1,
        boundary_level_set=lambda p: p[0] * p[0] - 2.25,
    )
    print("fd_pde_courant =", round(fd_pde.courant_number, 4))
    print("fd_pde_slices =", len(fd_pde.field_values))
    print("fd_pde_stencil_order =", fd_pde.stencil_order)
    print("fd_pde_refinement_rounds =", fd_pde.refinement_rounds)
    print("fd_pde_boundary_geometry =", fd_pde.boundary_geometry)
    print("fd_pde_active_points =", sum(fd_pde.active_point_mask))
    phys_lattice = solve_physical_lattice_dynamics(
        branches_a[0],
        time_slices=(0.0, 1.0, 2.0, 3.0, 4.0),
        spatial_points=((1.0, 0.0, 0.0), (2.0, 0.0, 0.0)),
        mass=0.5,
        propagation="instantaneous",
        method="leapfrog",
        damping_rate=0.1,
    )
    print("physical_lattice_method =", phys_lattice.method)
    print("physical_lattice_step =", phys_lattice.time_step)
    lebedev = compute_lebedev_displacement_amplitudes(
        branches_a, field_mass=0.5, momentum_cutoff=1.0, lebedev_order=14,
    )
    print("lebedev_direction_count =", lebedev.direction_count)
    print("lebedev_amplitudes =", lebedev.amplitudes)

    # --- 근본적 한계 개선 데모 ---
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
    print("spline_phase_matrix =", spline_phase)
    fock = compute_fock_space_evolution(
        branches_a[0], branches_b[0], momenta,
        field_mass=0.5, source_width_a=0.2, source_width_b=0.2,
    )
    print("fock_parametric_phase =", round(fock.parametric_phase, 6))
    print("fock_time_ordering_correction =", round(fock.time_ordering_correction, 8))
    print("fock_total_phase =", round(fock.total_phase, 6))
    print("fock_mode_count =", len(fock.mode_evolutions))
    adaptive = compute_adaptive_phase_integral(
        branches_a[:1], branches_b[:1], mass=0.5, tolerance=1e-8,
    )
    print("adaptive_phase =", round(adaptive[0][0].phase, 6))
    print("adaptive_error =", adaptive[0][0].estimated_error)
    print("adaptive_segments =", adaptive[0][0].segments_used)
    richardson = compute_richardson_extrapolated_phase(
        branches_a[:1], branches_b[:1], mass=0.5, orders=(2, 4, 6, 8),
    )
    print("richardson_phases_by_order =", tuple(round(p, 6) for p in richardson[0][0].phases_by_order))
    print("richardson_extrapolated =", round(richardson[0][0].extrapolated_phase, 6))
    print("richardson_error =", richardson[0][0].estimated_error)
    fock3 = compute_fock_space_evolution(
        branches_a[0], branches_b[0], momenta,
        field_mass=0.5, source_width_a=0.2, source_width_b=0.2,
        magnus_order=3,
    )
    print("fock_third_order_correction =", round(fock3.third_order_correction, 8))
    print("fock_total_phase_3rd =", round(fock3.total_phase, 6))
    lattice = solve_field_lattice(
        branches_a[0], time_slices=(0.0, 2.0, 4.0),
        spatial_points=((1.0, 0.0, 0.0), (2.0, 0.0, 0.0)),
        mass=0.5,
    )
    interp = interpolate_field_lattice(lattice, 1.0, (1.5, 0.0, 0.0))
    print("interpolated_field_value =", round(interp.value, 6))
    print("nearest_field_value =", round(interp.nearest_value, 6))
    running = compute_running_coupling_phase_matrix(
        branches_a[:1], branches_b[:1], mass=0.5,
        energy_scale=5.0, beta_coefficient=0.1,
    )
    print("bare_phase =", round(running.bare_matrix[0][0], 6))
    print("running_phase =", round(running.running_matrix[0][0], 6))
    validation = validate_symbolic_bookkeeping(labeled_mode_samples)
    print("bookkeeping_consistent =", validation.is_consistent)
    print("bookkeeping_max_norm_residual =", max(validation.norm_residuals))


if __name__ == "__main__":
    main()
