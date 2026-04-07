# relativistic_circuit_locality

`q-2026-03-24-2046.pdf`
("Circuit locality from relativistic locality in scalar field mediated entanglement", arXiv:2305.05645v4)
논문의 핵심 아이디어를 Python으로 옮긴 참조 구현이다.

이 저장소는 논문 전체를 완전한 양자장론 시뮬레이터로 재현하려는 목적이 아니라, 논문에서 직접 계산 가능한 핵심 구성요소를 작고 의존성 없는 수치 모델로 구현하는 데 초점을 둔다.

구현 범위는 다음과 같다.

- 두 quantum-controlled source의 branch별 궤적 표현
- field mediation circuit 성립 여부를 위한 spacelike separation 판정
- 논문의 `d_min > 0` 조건에 대응하는 최소 접근 거리 계산
- on-shell scalar-field phase `theta_rs = -1/2 ∫ rho_rs phi_rs`의 준정적 Yukawa 근사
- 한 방향 retarded 근사와 이를 대칭화한 time-symmetric light-cone 평균 모드
- 과거 light cone 전체를 적분하는 causal-history 전파 모드
- massive Klein-Gordon retarded Green function 의 shell/tail 구조를 반영하는 kg_retarded 모드
- branch 조합 사이의 상대 얽힘 위상 계산
- self-energy, directed cross-term, symmetric interaction, total phase 분해
- mediator 종류를 `scalar`/`vector`/`gravity`로 바꿔 계산하는 effective phase matrix
- 여러 입자 worldline 을 묶어 계산하는 composite branch 일반화
- isotropic Gaussian finite-width wavepacket 위상 계산
- spacetime sample point 에서 field 를 직접 평가하는 `sample_branch_field`와 `compute_sampled_spacetime_phase`
- target worldline 에서 `phi_rs` 샘플을 직접 반환하는 `compute_phi_rs_samples`
- retarded Green function shell/tail sample 을 직접 요약하는 `evaluate_retarded_green_function`
- 비등방 width tensor 를 쓰는 spacetime field sampling
- sampled `phi_rs`로 self/cross decomposition 을 직접 묶는 `analyze_sampled_phase_decomposition`
- 지정한 시공간 격자 위의 lattice field sampling 과 effective backreaction trajectory
- time-slice lattice dynamics, multiscale lattice refinement, boundary-conditioned spectral lattice, dynamic boundary schedule, FFT-like slice evolution, surrogate 4D PDE wrapper, large-scale surrogate bundle, iterative/coupled/nonlinear/self-consistent backreaction trajectory
- 유한 개 momentum mode, continuum radial quadrature, adaptive refinement, shell splitting, angular error estimate, extrapolated angular basis, spectral-style error bound, convergence tracking, certified surrogate controller, high-order aggregate result, provable-control wrapper 에 대한 Fourier-space displacement amplitude
- coherent-state 자유 진화, vacuum suppression, Gaussian/non-Gaussian/cat-state overlap, sample 기반 state tomography, multimode aggregate tomography, symbolic bookkeeping, analytic identity check, state transform, comprehensive bookkeeping, Appendix-D style bundle 추적
- generalized wavepacket wrapper, microcausality commutator surrogate, full-QFT surrogate bundle, reference PDE control bundle, universal state family bundle, exact mediator surrogate bundle, research-grade closure bundle

현재 코드는 논문의 parametric approximation 안에서 동작한다. 즉, 완전한 QFT 동역학을 직접 적분하지 않고, 시간 이산화된 궤적과 준정적 상호작용 커널을 이용해 논문의 구조를 계산 가능한 형태로 단순화했다.

최근 개선으로 branch 들이 동일한 시간 샘플을 공유해야 한다는 제약을 없앴다. 서로 다른 시간 이산화로 주어진 궤적도 겹치는 시간 구간에서 선형 보간하여 최소 접근 거리, mediation interval, branch phase 를 계산한다.

## 공개 API

- 기하/인과성: `BranchPath`, `TrajectoryPoint`, `compute_closest_approach`, `field_mediation_intervals`, `is_field_mediated`
- 위상 계산: `compute_branch_phase_matrix`, `compute_entanglement_phase`, `simulate`
- 위상 분해: `analyze_branch_pair_phase`, `analyze_phase_decomposition`
- wavepacket/generalized packet: `compute_wavepacket_phase_matrix`, `analyze_wavepacket_phase_decomposition`, `GeneralizedWavepacketResult`, `compute_generalized_wavepacket_phase_matrix`, `UniversalStateFamilyResult`, `compile_universal_state_family`
- direct field sampling: `FieldSample`, `sample_branch_field`, `compute_sampled_spacetime_phase`, `compute_anisotropic_sampled_spacetime_phase`, `RetardedGreenFunctionResult`, `evaluate_retarded_green_function`, `SampledPhaseDecompositionResult`, `analyze_sampled_phase_decomposition`
- lattice/backreaction: `FieldLattice`, `FieldEvolutionResult`, `MultiscaleFieldEvolutionResult`, `SpectralLatticeResult`, `DynamicBoundaryLatticeResult`, `FftLatticeEvolutionResult`, `SurrogatePdeResult`, `LargeScalePdeResult`, `solve_field_lattice`, `solve_field_lattice_dynamics`, `solve_multiscale_field_lattice`, `solve_spectral_lattice`, `solve_dynamic_boundary_lattice`, `solve_fft_lattice_evolution`, `solve_surrogate_4d_field_equation`, `solve_large_scale_pde_surrogate`, `evolve_backreacted_branch`, `iterate_backreaction`, `CoupledBackreactionResult`, `solve_coupled_backreaction`, `NonlinearBackreactionResult`, `solve_nonlinear_backreaction`, `SelfConsistentBackreactionResult`, `solve_self_consistent_backreaction`, `MediatorBackreactionResult`, `solve_mediator_self_consistent_backreaction`, `EffectiveFieldEquationResult`, `solve_effective_field_equation_backreaction`, `GaugeGravityFieldResult`, `solve_gauge_gravity_field_system`, `ExactMicrocausalityResult`, `evaluate_microcausality_commutator`, `FullQftSurrogateResult`, `solve_full_qft_surrogate`, `ReferencePdeControlResult`, `solve_reference_pde_control`, `ExactMediatorSurrogateResult`, `solve_exact_mediator_surrogate`, `ClosedLimitationBundleResult`, `close_current_limitations`
- research-grade closure: `HighFidelityPdeBundleResult`, `solve_high_fidelity_pde_bundle`, `CompleteStateFamilyBundleResult`, `compile_complete_state_family_bundle`, `ExactDynamicsSurrogateResult`, `solve_exact_dynamics_surrogate`, `ResearchGradeClosureResult`, `close_research_grade_limitations`
- explicit `phi_rs`: `compute_phi_rs_samples`
- Fourier/coherent-state: `compute_branch_displacement_amplitudes`, `compute_continuum_displacement_amplitudes`, `compute_adaptive_continuum_displacement_amplitudes`, `compute_split_continuum_displacement_amplitudes`, `estimate_continuum_displacement_amplitudes`, `compute_extrapolated_continuum_displacement_amplitudes`, `estimate_spectral_continuum_error_bound`, `estimate_spectral_convergence`, `compute_certified_spectral_displacement_amplitudes`, `compute_high_order_spectral_displacement_amplitudes`, `compute_provable_spectral_control`, `AngularQuadratureResult`, `ContinuumExtrapolationResult`, `SpectralErrorBoundResult`, `SpectralConvergenceResult`, `CertifiedSpectralResult`, `HighOrderSpectralResult`, `ProvableSpectralControlResult`, `compute_branch_pair_displacements`, `compute_displacement_operator_phase`, `CoherentStateEvolution`, `CoherentStateComparison`, `GaussianModeState`, `GeneralGaussianState`, `ModeSuperpositionState`, `CatModeState`, `compare_coherent_states`, `compare_gaussian_mode_states`, `compare_general_gaussian_states`, `compare_superposition_states`, `compare_cat_mode_states`, `tomograph_general_gaussian_state`, `tomograph_cat_mode_state`, `MultimodeTomographyResult`, `tomograph_multimode_family`, `SymbolicBookkeepingResult`, `summarize_symbolic_multimode_bookkeeping`, `AnalyticIdentityResult`, `verify_multimode_analytic_identities`, `MultimodeStateTransformResult`, `compile_multimode_state_transform`, `ComprehensiveBookkeepingResult`, `compile_comprehensive_multimode_bookkeeping`, `AppendixDBookkeepingResult`, `compile_appendix_d_bookkeeping`, `evolve_coherent_state`, `analyze_branch_pair_coherent_state`, `analyze_branch_pair_coherent_overlap`
- mediator/composite: `CompositeBranch`, `compute_mediated_phase_matrix`, `compute_composite_phase_matrix`, `sample_mediator_field`

## 전파 모드

- `instantaneous`: 동시각 Yukawa 상호작용
- `retarded`: 단일 retarded source point 근사
- `time_symmetric`: 양방향 retarded 평균
- `causal_history`: past light cone 내부 source history 적분
- `kg_retarded`: retarded shell 과 timelike tail 을 함께 적분하는 massive Klein-Gordon 근사

최근에는 worldline pair phase 만 적분하던 수준을 넘어서, target worldline 주변의 finite-width density 위에서 `phi_rs`를 직접 샘플링하는 spacetime field-sampling 계층도 추가했다. `evaluate_retarded_green_function`과 `analyze_sampled_phase_decomposition`으로 retarded shell/tail sample 과 self/cross decomposition 도 직접 요약할 수 있다.

추가로, 연속 운동량 적분은 radial quadrature, 유한 개 각도 방향 평균, adaptive refinement, shell splitting, coarse/refined angular error estimate, extrapolated angular basis, spectral-style error bound, convergence tracking, certified surrogate controller, high-order aggregate result, provable-control wrapper 로 근사하고, coherent-state 쪽은 vacuum overlap, diagonal/비대각 Gaussian covariance, 유한 개 coherent superposition, cat-state overlap, sample 기반 tomography, multimode aggregate phase matrix, symbolic bookkeeping, analytic identity check, state transform matrix, comprehensive bookkeeping bundle, Appendix-D style bundle 까지 계산한다. wavepacket 쪽은 generalized width wrapper, universal state family bundle, complete state family bundle 도 제공한다. 다입자와 gravity/vector 확장은 mediator-specific field sampling, time-slice/multiscale/spectral lattice dynamics, dynamic boundary schedule, FFT-like slice evolution, surrogate 4D PDE wrapper, large-scale surrogate bundle, effective/iterative/coupled/nonlinear/self-consistent backreaction, microcausality surrogate, full-QFT surrogate bundle, reference PDE control bundle, exact mediator surrogate bundle, research-grade closure bundle 수준에서 제공한다.

## 파일 구성

- `src/relativistic_circuit_locality/scalar_field.py`: 핵심 모델과 수치 계산 함수
- `src/relativistic_circuit_locality/demo.py`: 최소 실행 예제. phase matrix, sampled field, coherent-state, composite/mediator phase, wavepacket, current-limitation closure bundle, research-grade closure bundle 요약을 함께 출력한다.
- `tests/test_scalar_field.py`: 단위 테스트
- `CHAT.md`: 논문 해석, 구현된 기능, 남은 고도화 과제 정리

## 빠른 실행

예제 실행:

```bash
PYTHONPATH=src python -m relativistic_circuit_locality.demo
```

테스트 실행:

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```
