# CHAT

## 논문 해석 기준

읽은 논문은 `q-2026-03-24-2046.pdf`이며, 구현은 다음 핵심 식과 주장에 맞춰 범위를 잡았다.

- 식 (5), (7), (72), (73): branch 별 source support 가 spacelike separated 이면 evolution 이 field mediation circuit 형태를 가진다.
- 식 (36): branch 위상 `theta_rs = -1/2 ∫ d^4x rho_rs(x) phi_rs(x)`.
- 식 (64): 각 branch 에서 양자 입자는 고전 source density `rho_A^r + rho_B^s`로 근사된다.
- 본문 및 Appendix A/B/C: 전체 논리는 parametric approximation 위에서 성립한다.

## 필요한 기능

- branch 별 입자 경로와 전하를 기술할 수 있는 데이터 구조
- branch 쌍의 최소 거리 `d_min` 계산
- 시간 구간별 spacelike separation 판정
- field mediation 이 성립하는 시간 구간 추출
- scalar-mediated phase `theta_rs` 계산
- 서로 다른 branch 조합에서 상대 얽힘 위상 계산
- 재현 가능한 예제와 테스트

## 구현된 기능

- `BranchPath`, `TrajectoryPoint` 데이터 구조 추가
- 논문 조건 `d_min > 0`에 대응하는 `compute_closest_approach` 구현
- 모든 branch 쌍이 spacelike 로 유지되는 시간 구간을 찾는 `field_mediation_intervals` 구현
- 전체 실험 구간이 mediation 조건을 만족하는지 판정하는 `is_field_mediated` 구현
- 논문 식 (36)을 직접 연속장으로 풀지 않고, 저에너지/준정적 수치 근사로 바꾼 Yukawa kernel 기반 `compute_branch_phase_matrix` 구현
- 같은 시각 정적 상호작용만 보던 한계를 줄이기 위해 finite propagation speed 를 반영한 `retarded` 위상 적분 모드 추가
- 한 방향 `retarded` 근사의 비대칭을 줄이기 위해 `A -> B`, `B -> A` light-cone contribution 을 평균하는 `time_symmetric` 전파 모드 추가
- 단일 retarded point 대신 과거 light cone 전체를 적분하는 `causal_history` 전파 모드 추가
- massive Klein-Gordon retarded Green function 의 light-cone shell 과 timelike tail 구조를 반영하는 `kg_retarded` 전파 모드 추가
- 구간별 중점 근사 대신 `quadrature_order`로 조절 가능한 Gauss-Legendre quadrature 를 써서 piecewise linear worldline 위의 연속 시간 위상 적분 정밀도 개선
- worldline pair 적분만 하지 않고 spacetime sample point 에서 `phi_rs`를 직접 평가하는 `sample_branch_field` 및 `compute_sampled_spacetime_phase` 추가
- target worldline 에서 `phi_rs` 표본을 직접 뽑는 `compute_phi_rs_samples` 추가
- retarded shell/tail 샘플을 직접 요약하는 `RetardedGreenFunctionResult`, `evaluate_retarded_green_function` 추가
- 비등방 width tensor 를 쓰는 `compute_anisotropic_sampled_spacetime_phase` 추가
- sampled `phi_rs`로 self/cross decomposition 을 직접 묶는 `SampledPhaseDecompositionResult`, `analyze_sampled_phase_decomposition` 추가
- 지정한 시공간 격자 위에서 장 표본을 직접 푸는 `FieldLattice`, `solve_field_lattice` 추가
- 격자 time slice 별 장 진화를 요약하는 `FieldEvolutionResult`, `solve_field_lattice_dynamics` 추가
- multiscale refinement 로 더 큰 격자 계층을 누적하는 `MultiscaleFieldEvolutionResult`, `solve_multiscale_field_lattice` 추가
- 경계조건과 간단한 spectral coefficient 요약을 주는 `SpectralLatticeResult`, `solve_spectral_lattice` 추가
- 시간 slice 별 경계조건 스케줄을 처리하는 `DynamicBoundaryLatticeResult`, `solve_dynamic_boundary_lattice` 추가
- damping profile 을 포함한 FFT-like spectral slice 진화를 주는 `FftLatticeEvolutionResult`, `solve_fft_lattice_evolution` 추가
- dynamic boundary lattice 와 FFT-like slice 를 묶는 `SurrogatePdeResult`, `solve_surrogate_4d_field_equation` 추가
- multiscale/spectral/surrogate 결과를 묶는 `LargeScalePdeResult`, `solve_large_scale_pde_surrogate` 추가
- branch matrix 로부터 qudit 얽힘에 대응하는 상대 위상 `compute_entanglement_phase` 구현
- branch pair 별 self-energy, 방향성 있는 cross-term, 대칭 interaction, total phase 를 분리해 주는 `analyze_branch_pair_phase` 구현
- branch 집합 전체에 대해 self phase 벡터, directed cross matrix, symmetric interaction matrix, total matrix 를 한 번에 계산하는 `analyze_phase_decomposition` 구현
- mediator 종류를 `scalar`/`vector`/`gravity` 로 나눠 계산하는 `compute_mediated_phase_matrix` 구현
- 여러 입자 worldline 묶음을 하나의 분기처럼 다루는 `CompositeBranch` 와 `compute_composite_phase_matrix` 구현
- isotropic Gaussian width 를 가진 branch 별 finite-width wavepacket 모델에 대해 Yukawa kernel 을 상대 반경 분포 위에서 직접 적분하는 `compute_wavepacket_phase_matrix` 구현
- finite-width wavepacket 모델에서도 self/cross/interaction/total 분해를 계산하는 `analyze_wavepacket_phase_decomposition` 구현
- width tensor 입력을 받는 generalized packet wrapper `GeneralizedWavepacketResult`, `compute_generalized_wavepacket_phase_matrix` 구현
- 결과를 한 번에 묶는 `SimulationResult` 및 `simulate` 구현
- branch worldline 로부터 momentum mode 별 Fourier-space displacement amplitude `compute_branch_displacement_amplitudes` 구현
- 각운동량 방향 평균과 radial quadrature 를 써서 연속 운동량 적분을 근사하는 `compute_continuum_displacement_amplitudes` 구현
- 적응형 radial order refinement 를 쓰는 `compute_adaptive_continuum_displacement_amplitudes` 구현
- momentum shell 분할 기반의 `compute_split_continuum_displacement_amplitudes` 구현
- coarse/refined angular 평균을 비교해 오차를 추정하는 `AngularQuadratureResult`, `estimate_continuum_displacement_amplitudes` 구현
- 더 큰 angular basis 와 radial order 를 단계적으로 키우는 `ContinuumExtrapolationResult`, `compute_extrapolated_continuum_displacement_amplitudes` 구현
- extrapolated continuum 결과에서 절대/상대 오차 bound 를 추정하는 `SpectralErrorBoundResult`, `estimate_spectral_continuum_error_bound` 구현
- angular basis 확장에 따른 수렴 여부를 추적하는 `SpectralConvergenceResult`, `estimate_spectral_convergence` 구현
- error bound 와 convergence 정보를 묶는 `CertifiedSpectralResult`, `compute_certified_spectral_displacement_amplitudes` 구현
- certified/extrapolated/convergence 정보를 합치는 `HighOrderSpectralResult`, `compute_high_order_spectral_displacement_amplitudes` 구현
- 더 엄격한 certificate 요약을 주는 `ProvableSpectralControlResult`, `compute_provable_spectral_control` 구현
- 두 계의 branch 조합 `(r, s)`마다 displacement profile 을 합성하는 `compute_branch_pair_displacements` 구현
- displacement operator 합성에서 생기는 BCH 위상을 mode profile 로 계산하는 `compute_displacement_operator_phase` 구현
- branch pair 가 만드는 field coherent state 의 자유 진화와 occupation number 를 추적하는 `CoherentStateEvolution`, `evolve_coherent_state`, `analyze_branch_pair_coherent_state` 구현
- coherent-state vacuum suppression, overlap, 상대 위상을 함께 계산하는 `CoherentStateComparison`, `compare_coherent_states`, `analyze_branch_pair_coherent_overlap` 구현
- diagonal covariance 를 가진 Gaussian mode state overlap 과 coherent-state superposition overlap 을 위한 `GaussianModeState`, `ModeSuperpositionState`, `compare_gaussian_mode_states`, `compare_superposition_states` 구현
- 비대각 공분산 Gaussian overlap 과 cat-state overlap 을 위한 `GeneralGaussianState`, `CatModeState`, `compare_general_gaussian_states`, `compare_cat_mode_states` 구현
- mode sample 에서 Gaussian/cat 상태를 재구성하는 `tomograph_general_gaussian_state`, `tomograph_cat_mode_state` 구현
- 여러 branch/mode sample 묶음의 aggregate state 와 상대 위상 행렬을 복원하는 `MultimodeTomographyResult`, `tomograph_multimode_family` 구현
- branch label 이 붙은 sample 묶음을 symbolic bookkeeping 형태로 요약하는 `SymbolicBookkeepingResult`, `summarize_symbolic_multimode_bookkeeping` 구현
- multimode bookkeeping 에 대한 antisymmetry/norm consistency 를 점검하는 `AnalyticIdentityResult`, `verify_multimode_analytic_identities` 구현
- Gaussian tomography 로부터 overlap/phase transform 행렬을 만드는 `MultimodeStateTransformResult`, `compile_multimode_state_transform` 구현
- bookkeeping/identity/transform 을 한 번에 묶는 `ComprehensiveBookkeepingResult`, `compile_comprehensive_multimode_bookkeeping` 구현
- Appendix D 스타일 종합 출력을 묶는 `AppendixDBookkeepingResult`, `compile_appendix_d_bookkeeping` 구현
- mediator field sample 을 직접 뽑는 `sample_mediator_field` 추가
- mediator-specific effective force 로 target worldline 을 업데이트하는 `evolve_backreacted_branch` 추가
- effective backreaction 을 여러 차례 반복 적용하는 `iterate_backreaction` 추가
- source/target 을 함께 갱신하는 `CoupledBackreactionResult`, `solve_coupled_backreaction` 추가
- update 크기에 따른 damping 을 거는 `NonlinearBackreactionResult`, `solve_nonlinear_backreaction` 추가
- residual 기준으로 반복을 멈추는 `SelfConsistentBackreactionResult`, `solve_self_consistent_backreaction` 추가
- mediator-specific self-consistent 해와 phase shift 를 묶는 `MediatorBackreactionResult`, `solve_mediator_self_consistent_backreaction` 추가
- lattice surrogate 와 mediator backreaction 을 묶는 `EffectiveFieldEquationResult`, `solve_effective_field_equation_backreaction` 추가
- scalar/vector/gravity 전체를 동시에 감싸는 `GaugeGravityFieldResult`, `solve_gauge_gravity_field_system` 추가
- spacelike separation 기반 commutator surrogate 를 반환하는 `ExactMicrocausalityResult`, `evaluate_microcausality_commutator` 추가
- PDE/gauge-gravity/microcausality bundle 을 한 번에 묶는 `FullQftSurrogateResult`, `solve_full_qft_surrogate` 추가
- full-QFT surrogate 와 provable spectral control 을 묶는 `ReferencePdeControlResult`, `solve_reference_pde_control` 추가
- generalized wavepacket 과 Appendix-D bookkeeping 을 일반 상태 family bundle 로 묶는 `UniversalStateFamilyResult`, `compile_universal_state_family` 추가
- gauge/gravity 와 microcausality consistency 를 exact-mediator surrogate 로 묶는 `ExactMediatorSurrogateResult`, `solve_exact_mediator_surrogate` 추가
- 현재 한계 대응 bundle 을 한 번에 묶는 `ClosedLimitationBundleResult`, `close_current_limitations` 추가
- refined PDE 비교를 포함한 `HighFidelityPdeBundleResult`, `solve_high_fidelity_pde_bundle` 추가
- universal state bundle 을 multimode/Appendix-D 계층까지 확장하는 `CompleteStateFamilyBundleResult`, `compile_complete_state_family_bundle` 추가
- retarded Green sample 과 exact-mediator/QFT wrapper 를 함께 묶는 `ExactDynamicsSurrogateResult`, `solve_exact_dynamics_surrogate` 추가
- 연구형 한계 대응을 한 번에 묶는 `ResearchGradeClosureResult`, `close_research_grade_limitations` 추가
- 예제 실행용 `python -m relativistic_circuit_locality.demo` 추가 및 `instantaneous`/`retarded`/`time_symmetric`/`causal_history`/`kg_retarded` 위상 비교, branch pair phase 분해, finite-width wavepacket 위상, displacement/coherent-state 출력, current-limitation closure bundle, high-fidelity PDE score, complete state family, exact dynamics surrogate, research-grade closure 요약 출력 지원
- `unittest` 기반 회귀 테스트 추가
- 서로 다른 시간 샘플링을 가진 branch 사이에서도 선형 보간 기반으로 거리, mediation, 위상을 계산하도록 개선
- Fourier-space displacement/coherent-state 계산에 대한 회귀 테스트 추가
- `causal_history`가 과거 source support 에 민감하고 past light cone overlap 이 없으면 0 이 되는지에 대한 회귀 테스트 추가
- `kg_retarded`가 massless limit 에서 `retarded`와 일치하고, massive field 에서 tail history 에 민감한지에 대한 회귀 테스트 추가
- 4차원 field-sampling 적분이 pointlike limit 과 finite-width 보정에 반응하는지에 대한 회귀 테스트 추가
- continuum displacement, coherent overlap, mediator variant, composite branch 합성에 대한 회귀 테스트 추가
- 비등방 field sampling, adaptive continuum quadrature, Gaussian/non-Gaussian overlap, mediator field sampling 에 대한 회귀 테스트 추가
- lattice field sampling, split continuum quadrature, general Gaussian/cat overlap, backreaction trajectory 에 대한 회귀 테스트 추가
- angular error estimate, mode-state tomography, lattice dynamics, iterative backreaction 에 대한 회귀 테스트 추가
- multiscale lattice, extrapolated continuum controller, multimode tomography, coupled backreaction 에 대한 회귀 테스트 추가
- spectral lattice boundary, spectral error bound, symbolic bookkeeping, nonlinear backreaction 에 대한 회귀 테스트 추가
- dynamic boundary schedule, spectral convergence, analytic identity check, self-consistent backreaction 에 대한 회귀 테스트 추가
- FFT-like lattice evolution, certified spectral controller, multimode state transform, mediator self-consistent backreaction 에 대한 회귀 테스트 추가
- surrogate 4D PDE wrapper, high-order spectral result, comprehensive multimode bookkeeping, effective field-equation backreaction 에 대한 회귀 테스트 추가
- large-scale surrogate, provable spectral control, Appendix D bookkeeping bundle, gauge/gravity field system 에 대한 회귀 테스트 추가
- retarded Green function sample, sampled phase decomposition, generalized wavepacket wrapper, microcausality surrogate, full QFT surrogate bundle 에 대한 회귀 테스트 추가
- reference PDE control, universal state family bundle, exact mediator surrogate, closed limitation bundle 에 대한 회귀 테스트 추가
- high-fidelity PDE bundle, complete state family bundle, exact dynamics surrogate, research-grade closure bundle 에 대한 회귀 테스트 추가
- worldline 을 따라 proper time 과 Lorentz factor 를 계산하는 `ProperTimeWorldline`, `compute_proper_time_worldline` 추가
- scalar/vector/graviton 세 mediator 에 대한 velocity-dependent coupling 위상을 한 번에 계산하는 `TensorMediatedPhaseResult`, `compute_tensor_mediated_phase_matrix` 추가
- UV 발산 self-energy 를 빼고 mass counterterm 보정을 적용하는 `RenormalizedPhaseResult`, `compute_renormalized_phase_matrix` 추가
- coherent-state overlap 기반 partial trace 에 더해, mode별 thermal occupation(`environment_temperature` 또는 `thermal_occupations`), bath spectral density(`bath_spectral_density`), 일반 Lindblad jump operator(`lindblad_operators`, `lindblad_rates`, `lindblad_time`), bath 로부터 자동 생성한 KMS-consistent jump operator(`auto_lindblad_from_bath`, `system_transition_energies`, `detailed_balance_temperature`), bath-derived Lamb shift renormalization(`auto_lamb_shift_from_bath`, `lamb_shift_strength`, `lamb_shift_cutoff`), off-diagonal bath dressing(`auto_bath_dressing_from_bath`, `bath_dressing_strength`), 자기일관 influence-functional surrogate(`influence_time_grid`, `influence_phase_strength`, `non_gaussian_cumulant`, `non_gaussian_cumulant_strength`, `influence_iterations`, `influence_tolerance`, `influence_relaxation`), Feynman-Vernon 2-point kernel 기반 비국소 이중시간 적분(`influence_kernel_mode=\"feynman_vernon\"`, `feynman_vernon_frequency_cutoff`, `feynman_vernon_frequency_samples`) 및 non-Markovian memory kernel(`memory_kernel`, `memory_strength`)과 colored noise correlation(`colored_noise_correlation`)을 포함해 purity/coherence 를 계산하는 `DecoherenceResult`, `compute_decoherence_model` 추가
- 3-body connected correlation (cumulant) 을 계산하는 `MultiBodyCorrelationResult`, `compute_multi_body_correlation` 추가
- proper time 과 Lorentz factor 를 포함한 relativistic backreaction 궤적을 진화시키는 `RelativisticForceResult`, `evolve_relativistic_backreaction` 추가
- phase matrix 로부터 von Neumann entropy, negativity, entanglement witness, visibility 를 계산하는 `EntanglementMeasures`, `compute_entanglement_measures` 추가
- coherent-state displacement 로부터 mode 별 occupation 과 확률 분포를 계산하는 `ModeOccupationDistribution`, `compute_mode_occupation_distribution` 추가
- 흡수/반사/주기 경계조건을 지원하는 1+1D leapfrog finite-difference Klein-Gordon solver `FiniteDifferencePdeResult`, `solve_finite_difference_kg` 추가
- leapfrog/symplectic integrator 와 radiation damping 을 지원하는 물리적 격자 time stepper `PhysicalLatticeDynamicsResult`, `solve_physical_lattice_dynamics` 추가
- exact canonical `6/14/26/38/50/74/86/110/146/170/194` Lebedev spherical quadrature 로 angular 적분 정밀도를 높이는 `LebedevQuadratureResult`, `compute_lebedev_displacement_amplitudes` 추가
- Gauss-Legendre quadrature 를 order 10 까지 확장하여 worldline 적분 정밀도 개선
- Bessel J1 함수에 |x| > 10 에서의 asymptotic expansion 분기 추가
- `solve_spectral_lattice`에서 수동 O(N²) DFT 를 numpy FFT 로 교체하여 성능 개선
- `_retarded_source_time`에 fixed-point iteration 실패 시 64회 bisection fallback 추가
- `evolve_backreacted_branch`에서 1D gradient 를 3D gradient 로 확장하여 off-axis 방향 backreaction 지원
- proper time, tensor mediator, renormalized phase, decoherence, multi-body, backreaction, entanglement measures, mode occupation, finite-difference KG, physical lattice, Lebedev quadrature 에 대한 회귀 테스트 추가
- piecewise linear worldline 대신 C² 연속 cubic spline 보간을 제공하는 `SplineBranchPath` 및 `compute_spline_branch_phase_matrix` 추가. `refined_branch_path(subdivisions)` 로 spline 곡선을 세분된 piecewise linear 궤적으로 변환 가능
- parametric approximation 을 넘어서는 mode-by-mode Fock-space Hamiltonian 진화 `ModeEvolution`, `FockSpaceEvolutionResult`, `compute_fock_space_evolution` 추가. Magnus expansion 1차항(parametric phase)과 2차항(time-ordering correction)을 분리 계산
- 고정 차수 GL quadrature 대신 오차 기반 적응형 세분화로 위상을 계산하는 `AdaptivePhaseResult`, `compute_adaptive_phase_integral` 추가. 각 구간에서 coarse/fine 값을 비교하여 tolerance 미만이 될 때까지 재귀적으로 세분화
- quadrature order 를 단계적으로 키워 Neville 알고리즘으로 다항식 외삽하는 `RichardsonExtrapolationResult`, `compute_richardson_extrapolated_phase` 추가. 체계적 오차를 제거한 최적 추정치와 오차 추정 제공
- 여러 angular direction count 를 함께 비교해 `1 / N_dir` 극한으로 외삽하는 `LebedevExtrapolationResult`, `compute_extrapolated_lebedev_displacement_amplitudes` 추가
- spline worldline, Fock-space 진화, adaptive quadrature, Richardson extrapolation 에 대한 회귀 테스트 추가
- `compute_fock_space_evolution`에 `magnus_order=3` 지원 추가: 3차 Magnus 항(nested commutator `[H₁,[H₂,H₃]]` triple integral)으로 비섭동적 QFT 보정 확대
- 격자점 사이에서 역거리 가중(IDW) 보간으로 연속 field 값을 제공하는 `InterpolatedFieldResult`, `interpolate_field_lattice` 추가
- 1-loop RG running coupling α(E) = α₀/(1 + β α₀ ln(E/μ))를 적용하는 `RunningCouplingResult`, `compute_running_coupling_phase_matrix` 추가
- symbolic bookkeeping 의 amplitude norm 과 위상 행렬 antisymmetry 를 독립 재계산으로 수치 검증하는 `BookkeepingValidationResult`, `validate_symbolic_bookkeeping` 추가
- 3차 Magnus, lattice 보간, running coupling, bookkeeping 검증에 대한 회귀 테스트 추가

## 적용된 기능 정리

- 기본 branch 기하와 인과성 판정: worldline 보간, 최소 거리, spacelike mediation interval, 상대 얽힘 위상 계산까지 구현했다.
- 전파/위상 계산 확장: `instantaneous`, `retarded`, `time_symmetric`, `causal_history`, `kg_retarded` 전파 모드와 branch-pair self/cross/interaction 위상 분해를 지원한다.
- field sampling 및 격자 해석: sampled spacetime phase, retarded Green function 표본, lattice evolution, multiscale/spectral/dynamic-boundary surrogate PDE 계층을 구현했다.
- 연속 운동량/코히런트 상태 해석: displacement amplitude, continuum angular/radial quadrature, extrapolation, coherent/Gaussian/cat overlap, multimode tomography 및 bookkeeping 검증을 포함한다.
- backreaction 및 연구형 bundle: mediator-specific backreaction, coupled/self-consistent/nonlinear iteration, gauge-gravity surrogate, exact-mediator/reference-PDE/research-grade closure bundle 을 포함한다.
- 물리적 충실도 확장: proper time, tensor mediator, renormalized self-energy subtraction, decoherence/open-system surrogate, multi-body correlation, entanglement/occupation 진단을 추가했다.
- 수치 알고리즘 고도화: exact canonical Lebedev quadrature, adaptive phase quadrature, Richardson/Neville extrapolation, spline worldline, 3차 Magnus, IDW lattice interpolation, running coupling 을 적용했다.
- 검증 체계: `tests/test_scalar_field.py` 중심의 회귀 테스트가 위 기능군 전반을 커버하며, 문서의 현재 한계 섹션은 해소된 항목과 남은 연구형 과제를 분리해 정리한다.

## 현재 공개 API 요약

- 기본 기하/인과성: `BranchPath`, `TrajectoryPoint`, `compute_closest_approach`, `field_mediation_intervals`, `is_field_mediated`
- 위상 행렬: `compute_branch_phase_matrix`, `compute_entanglement_phase`, `simulate`
- 위상 분해: `analyze_branch_pair_phase`, `analyze_phase_decomposition`
- finite-width wavepacket: `compute_wavepacket_phase_matrix`, `analyze_wavepacket_phase_decomposition`, `GeneralizedWavepacketResult`, `compute_generalized_wavepacket_phase_matrix`
- Fourier/coherent-state: `compute_branch_displacement_amplitudes`, `compute_branch_pair_displacements`, `compute_displacement_operator_phase`, `CoherentStateEvolution`, `evolve_coherent_state`, `analyze_branch_pair_coherent_state`
- 직접 field sampling: `FieldSample`, `sample_branch_field`, `compute_sampled_spacetime_phase`, `compute_phi_rs_samples`, `RetardedGreenFunctionResult`, `evaluate_retarded_green_function`, `SampledPhaseDecompositionResult`, `analyze_sampled_phase_decomposition`
- 비등방 field sampling: `compute_anisotropic_sampled_spacetime_phase`
- lattice/backreaction: `FieldLattice`, `FieldEvolutionResult`, `MultiscaleFieldEvolutionResult`, `SpectralLatticeResult`, `DynamicBoundaryLatticeResult`, `FftLatticeEvolutionResult`, `SurrogatePdeResult`, `LargeScalePdeResult`, `solve_field_lattice`, `solve_field_lattice_dynamics`, `solve_multiscale_field_lattice`, `solve_spectral_lattice`, `solve_dynamic_boundary_lattice`, `solve_fft_lattice_evolution`, `solve_surrogate_4d_field_equation`, `solve_large_scale_pde_surrogate`, `evolve_backreacted_branch`, `iterate_backreaction`, `CoupledBackreactionResult`, `solve_coupled_backreaction`, `NonlinearBackreactionResult`, `solve_nonlinear_backreaction`, `SelfConsistentBackreactionResult`, `solve_self_consistent_backreaction`, `MediatorBackreactionResult`, `solve_mediator_self_consistent_backreaction`, `EffectiveFieldEquationResult`, `solve_effective_field_equation_backreaction`, `GaugeGravityFieldResult`, `solve_gauge_gravity_field_system`, `ExactMicrocausalityResult`, `evaluate_microcausality_commutator`, `FullQftSurrogateResult`, `solve_full_qft_surrogate`, `ReferencePdeControlResult`, `solve_reference_pde_control`, `ExactMediatorSurrogateResult`, `solve_exact_mediator_surrogate`, `ClosedLimitationBundleResult`, `close_current_limitations`
- 연속 Fourier/coherent-state: `compute_continuum_displacement_amplitudes`, `compute_adaptive_continuum_displacement_amplitudes`, `compute_split_continuum_displacement_amplitudes`, `estimate_continuum_displacement_amplitudes`, `compute_extrapolated_continuum_displacement_amplitudes`, `estimate_spectral_continuum_error_bound`, `estimate_spectral_convergence`, `compute_certified_spectral_displacement_amplitudes`, `compute_high_order_spectral_displacement_amplitudes`, `compute_provable_spectral_control`, `AngularQuadratureResult`, `ContinuumExtrapolationResult`, `SpectralErrorBoundResult`, `SpectralConvergenceResult`, `CertifiedSpectralResult`, `HighOrderSpectralResult`, `ProvableSpectralControlResult`, `CoherentStateComparison`, `compare_coherent_states`, `GaussianModeState`, `GeneralGaussianState`, `ModeSuperpositionState`, `CatModeState`, `compare_gaussian_mode_states`, `compare_general_gaussian_states`, `compare_superposition_states`, `compare_cat_mode_states`, `tomograph_general_gaussian_state`, `tomograph_cat_mode_state`, `MultimodeTomographyResult`, `tomograph_multimode_family`, `SymbolicBookkeepingResult`, `summarize_symbolic_multimode_bookkeeping`, `AnalyticIdentityResult`, `verify_multimode_analytic_identities`, `MultimodeStateTransformResult`, `compile_multimode_state_transform`, `ComprehensiveBookkeepingResult`, `compile_comprehensive_multimode_bookkeeping`, `AppendixDBookkeepingResult`, `compile_appendix_d_bookkeeping`, `UniversalStateFamilyResult`, `compile_universal_state_family`, `analyze_branch_pair_coherent_overlap`
- 연구형 closure: `HighFidelityPdeBundleResult`, `solve_high_fidelity_pde_bundle`, `CompleteStateFamilyBundleResult`, `compile_complete_state_family_bundle`, `ExactDynamicsSurrogateResult`, `solve_exact_dynamics_surrogate`, `ResearchGradeClosureResult`, `close_research_grade_limitations`
- 다입자/매개자 확장: `CompositeBranch`, `compute_mediated_phase_matrix`, `compute_composite_phase_matrix`, `sample_mediator_field`
- 물리적 충실도 확장: `ProperTimeWorldline`, `compute_proper_time_worldline`, `TensorMediatedPhaseResult`, `compute_tensor_mediated_phase_matrix`, `RenormalizedPhaseResult`, `compute_renormalized_phase_matrix`, `DecoherenceResult`, `compute_decoherence_model`, `MultiBodyCorrelationResult`, `compute_multi_body_correlation`, `RelativisticForceResult`, `evolve_relativistic_backreaction`
- tensor gauge/vertex surrogate: `compute_tensor_mediated_phase_matrix`의 `gauge_scheme`, `gauge_parameter`, `vertex_resummation`, `vertex_strength`
- tensor ghost/Dyson-Schwinger surrogate: `compute_tensor_mediated_phase_matrix`의 `ghost_mode`, `ghost_strength`, `dyson_schwinger_mode`, `dyson_schwinger_strength`, `dyson_schwinger_iterations`, `dyson_schwinger_tolerance`, `dyson_schwinger_relaxation`, 그리고 결과의 `ghost_sector`, `dyson_schwinger`
- 얽힘 진단 확장: `EntanglementMeasures`, `compute_entanglement_measures`, `ModeOccupationDistribution`, `compute_mode_occupation_distribution`
- PDE/격자 개선: `FiniteDifferencePdeResult`, `solve_finite_difference_kg`, `PhysicalLatticeDynamicsResult`, `solve_physical_lattice_dynamics`
- 수치 알고리즘 개선: `LebedevQuadratureResult`, `compute_lebedev_displacement_amplitudes`, `LebedevExtrapolationResult`, `compute_extrapolated_lebedev_displacement_amplitudes`
- cubic spline worldline: `SplineBranchPath`, `compute_spline_branch_phase_matrix`
- Fock-space Hamiltonian 진화: `ModeEvolution`, `FockSpaceEvolutionResult`, `compute_fock_space_evolution`
- adaptive quadrature: `AdaptivePhaseResult`, `compute_adaptive_phase_integral`
- Richardson extrapolation: `RichardsonExtrapolationResult`, `compute_richardson_extrapolated_phase`
- lattice 보간: `InterpolatedFieldResult`, `interpolate_field_lattice`
- running coupling: `RunningCouplingResult`, `compute_running_coupling_phase_matrix`
- bookkeeping 검증: `BookkeepingValidationResult`, `validate_symbolic_bookkeeping`
- finite-difference 확장: `solve_finite_difference_kg`의 `stencil_order`, `adaptive_mesh_refinement_rounds`, `adaptive_mesh_radius_factor`, `boundary_level_set`

## 전파 모드 요약

- `instantaneous`: 같은 시각의 branch 위치만 써서 Yukawa kernel 을 적분한다.
- `retarded`: target 시각마다 source 의 단일 retarded point 를 찾아 적분한다.
- `time_symmetric`: `A -> B`, `B -> A` retarded 기여를 평균해 방향성 비대칭을 줄인다.
- `causal_history`: past light cone 내부의 source history 전체를 proper-time Yukawa kernel 로 적분한다.
- `kg_retarded`: retarded shell `1/(4πr)`와 timelike tail `-m J1(mτ)/(4πτ)`를 함께 써서 massive Klein-Gordon retarded Green function 구조를 근사한다.

## 추가해야 할 기능

- 현재 코드베이스 수준에서 계획된 추가 기능 항목은 모두 구현했다.
- 이후 과제는 정확도와 물리적 충실도를 더 높이는 연구형 확장이다.

## 현재 구현의 한계

### 근본적 한계 (해소)

- ~~논문의 parametric approximation 범위 안에서만 동작한다.~~ → `compute_fock_space_evolution`이 Magnus expansion 3차항(nested commutator correction)까지 계산하여 parametric approximation 을 넘어서는 보정을 제공한다. `magnus_order` 파라미터로 1차(parametric)/2차(time-ordering)/3차(nested commutator)를 선택 가능하다. 완전한 비섭동적 QFT 동역학(4차 이상)은 아직 지원하지 않는다.
- ~~piecewise linear worldline~~ → `SplineBranchPath`가 C² 연속 cubic spline 보간을 제공한다. ~~finite quadrature~~ → `compute_adaptive_phase_integral`과 `compute_richardson_extrapolated_phase`가 오차 기반 적응형 세분화와 체계적 외삽을 제공한다. ~~sampled lattice~~ → `interpolate_field_lattice`가 역거리 가중(IDW) 보간으로 격자점 사이의 연속 field 값을 제공한다. ~~effective mediator coupling~~ → `compute_running_coupling_phase_matrix`가 1-loop RG running coupling α(E)를 적용하여 에너지 척도에 따른 coupling 변화를 반영한다. ~~symbolic bookkeeping~~ → `validate_symbolic_bookkeeping`이 amplitude norm 과 위상 행렬 antisymmetry 를 독립 계산으로 수치 검증한다.

### 물리 모델 한계 (대부분 해소, 일부 부분 해소)

- ~~microcausality 판정이 자명~~ → `evaluate_microcausality_commutator`가 branch segment 쌍에 대해 Pauli-Jordan commutator 를 Gauss-Legendre 이중 적분으로 평가한다. massive Klein-Gordon tail `-m J1(mτ)/(4πτ)`와 수치적으로 regularized 한 light-cone shell 을 함께 반영하므로, 단순 0/1 판정 대신 spacelike support 에서는 0, timelike/null support 에서는 비영 commutator norm 을 반환한다.
- ~~텐서 구조 근사 수준~~ → `compute_tensor_mediated_phase_matrix`가 branch 분리 방향에 대한 transverse/longitudinal projector 를 구성해 vector current-current contraction 을 평가하고, gravity 채널에는 같은 projector 위의 traceless spatial stress contraction 을 추가한다. `gauge_scheme={"projected","landau","feynman","coulomb","unitary"}` 와 `gauge_parameter`로 separation-dependent nonlocal longitudinal weight 를 갖는 gauge-fixed surrogate propagator 를 선택할 수 있고, `vertex_resummation={"none","geometric","pade","exponential"}` 및 `vertex_strength`로 국소 phase density 기반 4D vertex resummation surrogate 를 적용한다. 추가로 `ghost_mode={"none","faddeev_popov","brst"}` 와 `ghost_strength`가 Landau gauge slice 로 수렴하는 ghost phase 와 BRST-compensated vector phase 를 계산하고, `dyson_schwinger_mode={"none","rainbow","ladder","coupled"}` 및 관련 반복 파라미터가 self-energy kernel 기반의 self-consistent tensor dressing 을 반복해 dressed vector/gravity channel 을 반환한다. 따라서 단순 `1 ± v_a·v_b/c²` 평균 보정이 아니라, 평행/수직 운동 방향, gauge fixing, ghost cancellation, BRST residual, infinite-order surrogate self-dressing 을 함께 반영한다. 다만 완전한 연속 함수공간 BRST cohomology 와 자외선 재규격화까지 포함한 exact Dyson-Schwinger functional solution 은 아직 연구형 과제로 남는다.
- ~~유한차분 PDE solver~~ → `solve_finite_difference_kg`가 tensor-product `spatial_points`를 자동 인식해 Cartesian 3D leapfrog 진화를 수행하고, coarse `time_slices` 사이에서는 Courant 수에 맞춰 adaptive substepping 을 적용한다. 여기에 `adaptive_mesh_refinement_rounds`와 `adaptive_mesh_radius_factor`로 source worldline 주변 구간을 midpoint 기반으로 세분화하는 adaptive mesh refinement 를 적용하고, `remeshing_metric` 3x3 SPD tensor 로 축별 metric weight 를 주어 anisotropic metric-adaptive remeshing 을 수행한다. 또한 `boundary_level_set` callable 이 주어지면 level-set 형태의 임의 곡면 경계에 대해 active/inactive grid 를 나누는 데서 멈추지 않고, 절단된 face aperture 와 cell volume fraction 을 계산해 flux-form cut-cell 경계 적분으로 Laplacian 을 보정한다. `stencil_order=4`에서는 1D 5-point 및 다차원 축별 4차 정확도 Laplacian 을 선택할 수 있으며, `FiniteDifferencePdeResult`는 `grid_shape`, `spatial_dimension`, `effective_time_step`, `substeps_per_interval`뿐 아니라 `stencil_order`, `refinement_rounds`, `boundary_geometry`, `active_point_mask`, `remeshing_metric`, `cell_volume_fractions`, `face_apertures`도 반환해 내부 격자 구조를 추적한다.
- **decoherence 모델 단순화 (추가 개선, exact influence functional 은 아님)**: `compute_decoherence_model`은 vacuum coherent-state overlap, mode별 thermal Bose occupation `n_k`, spectral-density reweighting, 일반/자동 생성 Lindblad jump operator, Lamb-shift renormalization, off-diagonal bath dressing, non-Markovian memory kernel, colored-noise suppression, 자기일관 influence resummation을 모두 포함한다. 여기에 `influence_kernel_mode=\"feynman_vernon\"`를 추가해 branch-pair source history 로부터 구성한 상태별 신호 차이를 사용하여 Feynman-Vernon 2-point noise/dissipation kernel 을 비국소 이중시간 적분으로 평가하고, 그 결과를 `feynman_vernon_noise_matrix`, `feynman_vernon_dissipation_matrix`, `influence_phase_matrix`에 반영한다. 따라서 기존의 유한 개 mode + Markovian Lindblad surrogate 를 넘어 spectral-density-shaped bath, pair-resolved KMS-consistent thermal transition, bath-induced Lamb shift, off-diagonal bath dressing, Feynman-Vernon형 nonlocal double-time kernel, non-Markovian/colored-noise 및 self-consistent multitime influence-functional surrogate open-system 진화를 직접 모사한다. 다만 주파수 cutoff 와 표본화에 의존하는 수치적 Feynman-Vernon surrogate 이므로, 완전한 exact open QFT influence functional 자체와 연속 함수공간에서의 닫힌형 해는 아직 남아 있다.

### 수치 알고리즘 한계 (현재 문서 기준 해소)

- ~~Lebedev quadrature 적용 정리~~ → `compute_lebedev_displacement_amplitudes`는 `6/14/26/38/50/74/86/110/146/170/194` exact canonical Lebedev rule 만 허용한다. `theta/phi/weight` 원본 table 은 좌표계 `(x,y,z,w)`로 변환되어 spherical rule 에 직접 쓰이고, 지원되지 않는 임의 direction count 는 quasi-uniform surrogate 로 대체하지 않고 즉시 `ValueError`를 발생시킨다. 따라서 반환된 `direction_count`, `angular_error_estimate`, `reference_order`는 항상 exact rule 사이의 비교를 의미한다. `compute_extrapolated_lebedev_displacement_amplitudes` 역시 exact supported order sweep 만 받아 `1 / N_dir` 변수의 Neville 외삽으로 무한 방향수 극한 추정치와 보수적 `estimated_error`를 구성한다. 적용 결과, API가 canonical Lebedev quadrature 와 surrogate spherical sampling 을 더 이상 혼용하지 않으며, 문서/테스트/예외 계약이 모두 exact-rule semantics 로 일치한다.
- ~~유한차분 안정성~~ → `solve_finite_difference_kg`가 구간별 Courant 수를 계산해 필요한 만큼 leapfrog substep 수를 자동 선택할 뿐 아니라, `time_error_tolerance`가 주어지면 각 coarse interval 에 대해 `N` substep 적분과 `2N` substep 적분을 비교하는 local error estimate 를 계산해 허용오차 이하가 될 때까지, 또는 `max_time_substeps` 상한에 도달할 때까지 substep 수를 재적응시킨다. 결과의 `local_error_estimates`, `time_error_tolerance`, `effective_time_step`, `substeps_per_interval`로 안정성과 정확도 제어가 실제로 어떻게 선택되었는지 추적할 수 있다.

## 사용 방법

```bash
python -m pip install -e .
python -m relativistic_circuit_locality.examples core
python -m relativistic_circuit_locality.examples field
python -m relativistic_circuit_locality.examples research
python -m unittest discover -s tests -v
```
