# API Surface

## 패키지 계층

- `relativistic_circuit_locality`
  안정 공개 진입점. 핵심 기하, 인과성, 위상 계산만 노출한다.
- `relativistic_circuit_locality.core`
  패키지 루트와 동일한 안정 API 모듈이다.
- `relativistic_circuit_locality.experimental`
  surrogate, lattice, open-system, tomography, 연구형 bundle 을 포함한 확장 namespace 다.

## 안정 API

### Geometry and Causality

- `BranchPath`
- `TrajectoryPoint`
- `SplineBranchPath`
- `compute_closest_approach`
- `field_mediation_intervals`
- `is_field_mediated`

### Core Phase Model

- `compute_branch_phase_matrix`
- `compute_wavepacket_phase_matrix`
- `compute_entanglement_phase`
- `simulate`

## Experimental API Families

### Phase Decomposition and Sampling

- `analyze_branch_pair_phase`
- `analyze_phase_decomposition`
- `sample_branch_field`
- `compute_sampled_spacetime_phase`
- `compute_anisotropic_sampled_spacetime_phase`
- `compute_phi_rs_samples`
- `evaluate_retarded_green_function`
- `analyze_sampled_phase_decomposition`
- `compute_generalized_wavepacket_phase_matrix`
- `analyze_wavepacket_phase_decomposition`

### Continuum and Spectral Displacement

- `compute_branch_displacement_amplitudes`
- `compute_continuum_displacement_amplitudes`
- `compute_adaptive_continuum_displacement_amplitudes`
- `compute_split_continuum_displacement_amplitudes`
- `estimate_continuum_displacement_amplitudes`
- `compute_extrapolated_continuum_displacement_amplitudes`
- `estimate_spectral_continuum_error_bound`
- `estimate_spectral_convergence`
- `compute_certified_spectral_displacement_amplitudes`
- `compute_high_order_spectral_displacement_amplitudes`
- `compute_provable_spectral_control`
- `compute_lebedev_displacement_amplitudes`
- `compute_extrapolated_lebedev_displacement_amplitudes`

### State Overlap and Tomography

- `evolve_coherent_state`
- `analyze_branch_pair_coherent_state`
- `analyze_branch_pair_coherent_overlap`
- `compare_coherent_states`
- `compare_gaussian_mode_states`
- `compare_general_gaussian_states`
- `compare_superposition_states`
- `compare_cat_mode_states`
- `tomograph_general_gaussian_state`
- `tomograph_cat_mode_state`
- `tomograph_multimode_family`
- `summarize_symbolic_multimode_bookkeeping`
- `verify_multimode_analytic_identities`
- `compile_multimode_state_transform`
- `compile_comprehensive_multimode_bookkeeping`
- `compile_appendix_d_bookkeeping`
- `compile_universal_state_family`
- `compile_complete_state_family_bundle`

### Lattice, PDE, and Backreaction

- `solve_field_lattice`
- `solve_field_lattice_dynamics`
- `solve_multiscale_field_lattice`
- `solve_spectral_lattice`
- `solve_dynamic_boundary_lattice`
- `solve_fft_lattice_evolution`
- `solve_surrogate_4d_field_equation`
- `solve_large_scale_pde_surrogate`
- `evolve_backreacted_branch`
- `iterate_backreaction`
- `solve_coupled_backreaction`
- `solve_nonlinear_backreaction`
- `solve_self_consistent_backreaction`
- `solve_mediator_self_consistent_backreaction`
- `solve_effective_field_equation_backreaction`
- `solve_gauge_gravity_field_system`
- `solve_full_qft_surrogate`
- `solve_reference_pde_control`
- `solve_exact_mediator_surrogate`
- `close_current_limitations`
- `solve_high_fidelity_pde_bundle`
- `solve_exact_dynamics_surrogate`
- `close_research_grade_limitations`
- `solve_finite_difference_kg`
- `solve_physical_lattice_dynamics`
- `interpolate_field_lattice`

### Physical Fidelity Extensions

- `compute_proper_time_worldline`
- `compute_tensor_mediated_phase_matrix`
- `compute_renormalized_phase_matrix`
- `compute_decoherence_model`
- `compute_multi_body_correlation`
- `evolve_relativistic_backreaction`
- `compute_entanglement_measures`
- `compute_mode_occupation_distribution`
- `compute_fock_space_evolution`
- `compute_adaptive_phase_integral`
- `compute_richardson_extrapolated_phase`
- `compute_running_coupling_phase_matrix`
- `validate_symbolic_bookkeeping`

### Tooling and Profiling

- `benchmark_representative_workloads`
- `format_benchmark_report`
- `profile_call`

## 예제 진입점

- `python -m relativistic_circuit_locality.examples core`
  안정 API 와 핵심 위상 계산 예제
- `python -m relativistic_circuit_locality.examples field`
  field sampling, wavepacket, lattice 관련 예제
- `python -m relativistic_circuit_locality.examples research`
  surrogate/PDE/closure 계열 예제
