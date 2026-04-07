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

## 전파 모드 요약

- `instantaneous`: 같은 시각의 branch 위치만 써서 Yukawa kernel 을 적분한다.
- `retarded`: target 시각마다 source 의 단일 retarded point 를 찾아 적분한다.
- `time_symmetric`: `A -> B`, `B -> A` retarded 기여를 평균해 방향성 비대칭을 줄인다.
- `causal_history`: past light cone 내부의 source history 전체를 proper-time Yukawa kernel 로 적분한다.
- `kg_retarded`: retarded shell `1/(4πr)`와 timelike tail `-m J1(mτ)/(4πτ)`를 함께 써서 massive Klein-Gordon retarded Green function 구조를 근사한다.

## 추가해야 할 기능

### 물리적 충실도 확장

- 벡터장 매개자에 대한 spin-1 편광 텐서 구조 반영 (현재는 massless scalar 로 대체)
- 중력 매개자에 대한 spin-2 linearized gravity 텐서 구조 반영 (현재는 `|charge|` 결합의 massless scalar 로 대체)
- massive vector/graviton propagator 지원 (현재 vector/gravity 매개자는 `mass=0` 으로 고정)
- self-energy 위상에 대한 UV 정규화(renormalization) 절차 도입 (현재는 cutoff 만으로 정규화)
- 고유 시간(proper time) 기반 worldline 매개변수화 (현재는 좌표 시간만 사용)
- 비관성/가속 worldline 에 대한 상대론적 운동방정식(Lorentz force 등) 통합
- 환경 결합 및 decoherence 모델 추가
- genuinely multi-body quantum correlation 처리 (현재 `CompositeBranch`는 pairwise 합산만 수행)

### 수치 정밀도 및 알고리즘 개선

- Gauss-Legendre quadrature 를 6차 이상으로 확장 (현재 최대 5차, 하드코딩된 lookup table)
- `_bessel_j1` Taylor 급수를 큰 인수에서도 안정적인 구현으로 교체 (asymptotic expansion 또는 라이브러리 활용)
- 수동 DFT `O(N²)` 를 실제 FFT `O(N log N)` 으로 교체
- 구면 quadrature 를 Lebedev 격자 등 체계적 방법으로 교체 (현재 최대 26개 축/대각/모서리 방향)
- retarded time finder 의 수렴 보장 강화 (현재 단순 고정점 반복, 상대론적 source 에서 실패 가능)
- backreaction gradient 를 3차원 전체로 확장 (현재 x축 방향만 계산, y/z는 0으로 고정)

### PDE/격자 개선

- 실제 유한차분 또는 spectral PDE solver 로 Klein-Gordon 방정식 직접 적분 (현재 격자 위에서 커널 함수만 평가)
- 격자 smoothing 을 물리적 시간 stepper 로 교체 (현재 nearest-neighbor 평균)
- 경계조건을 물리적 조건에서 도출 (현재 ad-hoc edge averaging/zeroing/neighbor copy)
- damping profile 을 물리적 산일(dissipation)에서 도출 (현재 인위적 `1/(1 + strength*index)`)

### 얽힘 진단 확장

- von Neumann entropy, negativity 등 얽힘 측도 계산
- entanglement witness / visibility 계산
- field mode 별 occupation number 분포 (현재는 총 occupation number 만 계산)

## 현재 구현의 한계

### 근본적 한계

- 논문의 parametric approximation 범위 안에서만 동작한다. 완전한 QFT 동역학을 직접 적분하지 않는다.
- piecewise linear worldline, finite quadrature, sampled lattice, effective mediator coupling, symbolic bookkeeping 을 결합한 참조 코드 수준이다.

### 물리 모델 한계

- **매개자 단순화**: `vector` 매개자는 massless scalar (mass=0) 로, `gravity` 매개자는 `|charge|` 결합의 massless scalar 로 대체되어 있다. 실제 spin-1/spin-2 텐서 구조가 없다 (`_mediator_field_value`, line 856–871).
- **자기에너지 정규화 없음**: self-energy 위상이 Yukawa cutoff 로만 정규화된다. 체계적 UV 정규화 절차가 없다.
- **좌표 시간만 사용**: proper time 매개변수화가 없으므로 가속 worldline 에서의 상대론적 효과가 정확하지 않다.
- **microcausality 판정이 자명**: `evaluate_microcausality_commutator`는 spacelike 구간에서 항상 0 을 반환하고, 비분리 구간에서 1 을 반환한다. 실제 장 교환자(commutator)를 계산하지 않는다 (line 3289).
- **다체 상관 미반영**: `CompositeBranch`는 구성요소 간 pairwise 위상 합산만 수행하며, genuinely multi-body quantum correlation 을 포착하지 않는다.
- **decoherence/환경 결합 없음**: 폐쇄 계(closed system) 만 다루며 환경과의 상호작용에 의한 결어긋남을 모델링하지 않는다.

### 수치 알고리즘 한계

- **quadrature 차수 제한**: Gauss-Legendre 규칙이 1~5차까지만 하드코딩되어 있다 (line 661–662). 높은 정밀도가 필요한 경우 확장 불가.
- **Bessel 함수 정밀도**: `_bessel_j1` 이 Taylor 급수로 구현되어 있어 큰 인수(`m·τ ≫ 1`)에서 수렴이 느리거나 정밀도가 떨어질 수 있다 (line 630–639).
- **DFT 비효율**: `solve_spectral_lattice`의 DFT 가 `O(N²)` 수동 루프로 구현되어 있다 (line 2645–2650). 큰 격자에서 비효율적.
- **각도 방향 제한**: 연속 운동량 적분의 각도 평균이 최대 26개(축 6 + 대각 8 + 모서리 12) 이산 방향으로만 수행된다. 체계적 구면 quadrature(Lebedev 등)가 아니다.
- **retarded time 수렴**: `_retarded_source_time`이 단순 고정점 반복(12회)으로 구현되어 있어, 상대론적으로 빠른 source 에서 수렴이 보장되지 않는다 (line 666–691).

### 격자/PDE 한계

- **PDE 미적분**: 모든 격자/PDE 함수(`solve_field_lattice`, `solve_surrogate_4d_field_equation` 등)는 격자 점에서 커널 함수를 평가할 뿐, 실제 편미분 방정식을 유한차분이나 spectral 방법으로 풀지 않는다.
- **1차원 backreaction**: `evolve_backreacted_branch`가 x축 방향 gradient 만 계산하고 y, z 방향 shift 는 0 으로 고정한다 (line 2916–2918).
- **인위적 smoothing/damping**: 격자 dynamics 의 smoothing 은 nearest-neighbor 평균이고, FFT evolution 의 damping 은 `1/(1 + strength*index)` 형태로, 물리적 근거 없이 도입되었다.
- **ad-hoc 경계조건**: spectral lattice 의 경계조건(periodic, Dirichlet, Neumann)이 물리에서 도출된 것이 아니라 가장자리 값 조작으로 구현되어 있다.

### 진단/출력 한계

- **얽힘 측도 부재**: 상대 위상(phase)만 계산하며, von Neumann entropy, negativity, entanglement witness 등 표준 얽힘 측도를 제공하지 않는다.
- **mode 별 occupation 분포 없음**: 총 occupation number 만 계산하고, 개별 momentum mode 별 분포를 분석하지 않는다.

## 사용 방법

```bash
PYTHONPATH=src python -m relativistic_circuit_locality.demo
PYTHONPATH=src python -m unittest discover -s tests -v
```
